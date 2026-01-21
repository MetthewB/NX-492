import os
import random
from collections import deque
import numpy as np
import torch
from config import *

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def reset_environment_and_controller(env, controller, target_qpos, offset_x, offset_y):
    """Resets environment/controller and returns initial observation."""
    controller.reset_states()
    obs, _ = env.reset(target_qpos=target_qpos, offset=(offset_x, offset_y))
    return obs

def initialize_epoch_params(perturbation_directions, bg_low, bg_high, G_s_low, G_s_high):
    """Selects random load conditions and perturbation direction for the epoch."""
    background_load = random.choice([bg_low, bg_high])
    load_condition = "High" if background_load == bg_high else "Low"
    G_s = G_s_high if background_load == bg_high else G_s_low
    
    idx = random.randint(0, len(perturbation_directions) - 1)
    dir_name = list(perturbation_directions.keys())[idx]
    
    return background_load, load_condition, G_s, idx, dir_name

def initialize_buffers(feedback_delay, reflex_delay, init_qpos, init_qvel, init_aff):
    """Creates history buffers for delayed feedback."""
    cortical = deque([ (init_qpos, init_qvel) for _ in range(feedback_delay) ], maxlen=feedback_delay)
    spinal = deque([ init_aff for _ in range(reflex_delay) ], maxlen=reflex_delay)
    return cortical, spinal

def update_buffers(cortical, spinal, qpos, qvel, aff):
    """Pushes current state into delay buffers."""
    cortical.append((qpos, qvel))
    spinal.append(aff)

def compute_pose_error(obs, target_qpos, l1, l2, off_x, off_y):
    """Calculates Euclidean distance from end-effector to target."""
    x = l1 * np.cos(obs[0]) + l2 * np.cos(obs[0] + obs[2]) - off_x
    y = l1 * np.sin(obs[0]) + l2 * np.sin(obs[0] + obs[2]) - off_y
    dist = np.sqrt((target_qpos[0] - x)**2 + (target_qpos[1] - y)**2)
    return dist, (x, y)

def compute_desired_force(step, Tb, Tp, Tsp, bg_load, p_load, p_dirs, dir_name):
    """Returns external force vector based on simulation phase."""
    force_mag = 0.0
    if Tb <= step < Tp: force_mag = bg_load
    elif Tp <= step < Tsp: force_mag = bg_load + p_load
    elif step >= Tsp: force_mag = bg_load
    return force_mag * p_dirs[dir_name]

def get_cost_weights(epoch, total_epochs):
    progress = epoch / total_epochs
    if progress < 0.2:
        return 0.0001, 0.1, 1.0  
    elif progress < 0.6:
        return 0.0001, 0.05, 2.0 
    else:
        return 0.0001, 0.01, 0.5
    
def compute_kinematic_penalty_radians(qpos, qvel, target_qpos):
    angle_error = qpos - target_qpos
    pos_cost = torch.sum(angle_error.pow(2))
    vel_cost = 0.5 * torch.sum(qvel.pow(2))
    return pos_cost + vel_cost

def accumulate_costs(controller, F_m, qpos, qvel, target_qpos, 
                     loss_n, loss_f, loss_k, Tsb_step, Tp_step, Tsp_step, step):
    """Accumulates neural, force, and kinematic costs."""
    # Neural Energy Cost
    loss_n += (controller.y_s.pow(2).sum() + controller.y_m.pow(2).sum())
    
    # Muscle Force Cost
    loss_f += F_m.pow(2).sum()

    # Kinematic Cost (with grace period logic)
    grace_period_steps = int(0.08 / dt) 
    is_background_steady = (Tsb_step <= step < Tp_step)
    is_perturbation_steady = (step >= Tp_step + grace_period_steps)
    
    if is_background_steady or is_perturbation_steady:
        loss_k += compute_kinematic_penalty_radians(qpos, qvel, target_qpos)

    return loss_n, loss_f, loss_k

def normalize_and_compute_total_loss(l_n, l_f, l_k, alpha, beta, gamma, steps):
    return alpha*(l_n/steps) + beta*(l_f/steps) + gamma*(l_k/steps)