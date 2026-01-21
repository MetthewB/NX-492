import torch
import numpy as np

# Custom modules
from config import *
from environment import MyoElbowPose2D6MFixed
from neural_network import NeuralLimbController, compute_feedback_signals
from biomechanics import compute_muscle_lengths, compute_jacobian, update_muscle_kinematics, ForceLength, ForceVelocity, differentiable_physics_step
from utils import set_seed, reset_environment_and_controller, initialize_epoch_params, initialize_buffers, update_buffers, compute_desired_force, accumulate_costs, normalize_and_compute_total_loss, compute_pose_error, get_cost_weights
from visualization import plot_detailed_cost_history, plot_trajectory_evolution_grid_xy, plot_detailed_trajectory_comparison, plot_neural_activity_comparison

def main():
    # --- INITIALIZATION ---
    set_seed(42)
    # Switch between 100 or 3000 depending on what you want to run
    num_epochs = 100  
    
    env = MyoElbowPose2D6MFixed()
    controller = NeuralLimbController(n_inputs=n_inputs, n_outputs=n_muscles)

    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4, weight_decay=1e-5)
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Ensure Moment Arm is a Float Tensor
    if not isinstance(M_m, torch.Tensor):
        M_m_tensor = torch.tensor(M_m, dtype=torch.float32)
    else:
        M_m_tensor = M_m.clone().detach().float()

    # Storage
    epoch_metadata = []
    epoch_qpos_over_time = []
    epoch_neural_activity = [] 
    epoch_costs_over_time = []
    target_results = {i: {"pose_errors": [], "rewards": [], "costs": {key: [] for key in ["neural", "force", "kinematic", "total"]}} for i in range(len(perturbation_directions))}

    print(f"Starting training for {num_epochs} epochs...")

    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        # Detach hidden states from previous epoch
        controller.y_s = controller.y_s.detach()
        controller.y_m = controller.y_m.detach()
        controller.y_act = controller.y_act.detach()

        # 0. Update Costs for this Epoch
        alpha, beta, gamma = get_cost_weights(epoch, num_epochs)

        # 1. Setup Targets
        base_qs, base_qe = np.radians(35), np.radians(90)
        base_x = l1 * np.cos(base_qs) + l2 * np.cos(base_qs + base_qe)
        base_y = l1 * np.sin(base_qs) + l2 * np.sin(base_qs + base_qe)
        offset_x, offset_y = base_x, base_y

        target_qpos = (base_qs, base_qe)
        target_qpos_tensor = torch.tensor([base_qs, base_qe], dtype=torch.float32)

        # 2. Epoch Params
        background_load, load_condition, G_s, perturbation_idx, direction_name = initialize_epoch_params(
            perturbation_directions, background_load_low, background_load_high, G_s_low, G_s_high
        )

        # 3. Reset Env
        obs_numpy = reset_environment_and_controller(env, controller, target_qpos, offset_x, offset_y)
        
        # Force Perfect Start
        env.set_joint_angles([base_qs, 0.0, base_qe, 0.0])
        obs_numpy = env._get_obs()

        # Log metadata
        epoch_metadata.append({"load": load_condition, "direction": direction_name})
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")

        # 4. Buffers & Init
        activation_buffer = [torch.zeros(n_muscles) for _ in range(activation_delay_steps)]
        delayed_activations = activation_buffer[0]
        loss_neural_accum, loss_force_accum, loss_kinematic_accum = 0, 0, 0
        pose_errors, rewards, qpos_over_time = [], [], []
        epoch_costs = {key: [] for key in ["neural", "force", "kinematic", "total"]}
        trial_neural_history = []

        # Init Muscle Lengths
        theta_s_ref, theta_e_ref = obs_numpy[0], obs_numpy[2]
        initial_lengths = compute_muscle_lengths(theta_s_ref, theta_e_ref, theta_s_ref, theta_e_ref, M_m)
        L_m_initial = initial_lengths.clone().detach() 
        
        delta_L_m, velocity_L_m, acceleration_L_m, jerk_L_m = [torch.zeros(n_muscles) for _ in range(4)]
        
        # CRITICAL: Initialize Tensors that will carry the gradient
        current_qpos = torch.tensor([obs_numpy[0], obs_numpy[2]], dtype=torch.float32)
        current_qvel = torch.tensor([obs_numpy[1], obs_numpy[3]], dtype=torch.float32)
        
        initial_x_m = torch.zeros((n_muscles, 4), dtype=torch.float32)
        initial_aff = (G_aff @ initial_x_m.T).flatten()
        
        # Initialize buffers with Tensors
        cortical_buffer, spinal_buffer = initialize_buffers(
            feedback_delay_steps, reflex_delay, current_qpos, current_qvel, initial_aff
        )

        # --- SIMULATION STEPS ---
        for step in range(T_step):
            # A. Targets & Forces
            desired_force = compute_desired_force(step, Tb_step, Tp_step, Tsp_step, background_load, perturbation_load, perturbation_directions, direction_name)

            # B. Physics Setup
            theta_s, theta_e = current_qpos[0], current_qpos[1]
            
            # 1. Use the current angles for the Jacobian
            J = compute_jacobian(theta_s.detach().item(), theta_e.detach().item(), l1, l2)
            
            # 2. Apply Torque
            tau = -(J.T @ desired_force) 
            tau_tensor = torch.tensor(tau, dtype=torch.float32)
            env.set_external_torque(tau)
            
            # 3. Tare Sensors at Perturbation Onset
            if step == Tp_step:
                theta_s_ref, theta_e_ref = current_qpos[0].item(), current_qpos[1].item()
                ref_lengths = compute_muscle_lengths(
                    theta_s_ref, theta_e_ref, theta_s_ref, theta_e_ref, M_m_tensor
                )
                L_m_initial = ref_lengths.clone().detach()

            # C. Update Muscle Kinematics
            L_m, delta_L_m, velocity_L_m, acceleration_L_m, jerk_L_m, dLm_dt = update_muscle_kinematics(
                qpos=current_qpos, 
                qvel=current_qvel,
                theta_ref=torch.tensor([theta_s_ref, theta_e_ref]), 
                L_m_init=L_m_initial,                               
                M_m=M_m_tensor, 
                dt=dt,
                vel_Lm_prev=velocity_L_m,          
                acc_Lm_prev=acceleration_L_m       
            )
            
            x_m = torch.stack([delta_L_m, velocity_L_m, acceleration_L_m, jerk_L_m], dim=1)
            current_aff = (G_aff @ x_m.T).flatten()

            # D. Muscle Dynamics
            FL_active, FL_passive = ForceLength.compute(L_m)
            FV = ForceVelocity.compute(dLm_dt.squeeze()).detach() 
            F_m = (delayed_activations * FL_active * FV) + FL_passive

            # E. Control Step
            delayed_qpos, delayed_qvel = cortical_buffer[0]
            delayed_aff = spinal_buffer[0]
            
            u_fb_cortical, u_spinal = compute_feedback_signals(
                None, delayed_qpos, delayed_qvel, delayed_aff, 
                target_qpos_tensor, G_p, G_d, G_f, F_m
            )

            delayed_aff = spinal_buffer[0] 
            u_spinal_reflex = (G_s * delayed_aff).detach()

            muscle_activations = controller.forward_step(
                u_fb_cortical, u_spinal_reflex, controller.y_s, controller.y_m, controller.y_act
            )
            trial_neural_history.append(controller.y_m.abs().mean().item())

            update_buffers(cortical_buffer, spinal_buffer, current_qpos, current_qvel, current_aff)
            delayed_activations = activation_buffer.pop(0)
            activation_buffer.append(muscle_activations)
                
            # F. Differentiable Physics Step
            next_qpos, next_qvel = differentiable_physics_step(current_qpos, current_qvel, F_m, tau_tensor, M_m_tensor)
            
            # G. Loss Accumulation
            loss_neural_accum, loss_force_accum, loss_kinematic_accum = accumulate_costs(
                controller, F_m, next_qpos, next_qvel, target_qpos_tensor, 
                loss_neural_accum, loss_force_accum, loss_kinematic_accum, 
                Tsb_step, Tp_step, Tsp_step, step
            )
            
            # Logging (Visuals only)
            obs_display = np.array([next_qpos[0].item(), next_qvel[0].item(), next_qpos[1].item(), next_qvel[1].item()])
            pose_error, (x, y) = compute_pose_error(obs_display, (base_qs, base_qe), l1, l2, offset_x, offset_y)
            pose_errors.append(pose_error)
            qpos_over_time.append((x, y))

            current_qpos = next_qpos
            current_qvel = next_qvel

        # --- END EPOCH ---
        epoch_neural_activity.append(trial_neural_history)

        # Optimization
        total_trial_loss = normalize_and_compute_total_loss(
            loss_neural_accum, loss_force_accum, loss_kinematic_accum, alpha, beta, gamma, T_step
        )
        
        optimizer.zero_grad()
        total_trial_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        c_neural = alpha * (loss_neural_accum / T_step).item()
        c_force = beta * (loss_force_accum / T_step).item()
        c_kinematic = gamma * (loss_kinematic_accum / T_step).item()
        
        target_results[perturbation_idx]["costs"]["neural"].append(c_neural)
        target_results[perturbation_idx]["costs"]["force"].append(c_force)
        target_results[perturbation_idx]["costs"]["kinematic"].append(c_kinematic)
        target_results[perturbation_idx]["costs"]["total"].append(c_neural + c_force + c_kinematic)
        target_results[perturbation_idx]["pose_errors"].append(sum(pose_errors) / len(pose_errors))
        target_results[perturbation_idx]["rewards"].append(sum(rewards))
        
        epoch_qpos_over_time.append(qpos_over_time) 
        epoch_costs_over_time.append(epoch_costs) 
        scheduler.step()

    # --- ANALYSIS PLOTS ---
    print("Training complete. Generating plots...")
    plot_detailed_cost_history(target_results)
    plot_trajectory_evolution_grid_xy(epoch_qpos_over_time, epoch_metadata)
    plot_detailed_trajectory_comparison(epoch_metadata, epoch_qpos_over_time, perturbation_directions)
    plot_neural_activity_comparison(epoch_metadata, epoch_neural_activity, Tp_step)

if __name__ == "__main__":
    main()