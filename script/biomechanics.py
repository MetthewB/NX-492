import torch
import numpy as np

class MuscleConstants:
    """Physiological constants for muscle dynamics."""
    F_MAX = 120.0             # Maximum isometric force (N)
    L_OPT = 1.0               # Optimal fiber length (normalized)
    V_MAX = 10.0              # Max shortening velocity (L_opt/s)
    
    # Hill's Equation coefficients
    A_HILL = 0.25             
    B_HILL = A_HILL * V_MAX   

    # Lengthening coefficients
    K_ECC = 0.15              
    F_MAX_ECC = 1.4           

class ForceLength:
    """Computes the force-length relationship (Active + Passive)."""
    @staticmethod
    def compute(L_m, L_opt=MuscleConstants.L_OPT, width_factor=0.4, passive_scale=0.01):
        # Active Component: Gaussian bell curve around optimal length
        FL_active = torch.exp(-((L_m - L_opt) / width_factor).pow(2))
        
        # Passive Component: Exponential spring for stretching
        strain = (L_m - L_opt) / L_opt
        k_pe = 4.0 
        FL_passive = passive_scale * (torch.exp(k_pe * strain) - 1.0)
        
        return FL_active, torch.clamp(FL_passive, min=0.0)

class ForceVelocity:
    """Computes the force-velocity relationship (Hill's Equation)."""
    @staticmethod
    def compute(dLm_dt, V_max=MuscleConstants.V_MAX, A_hill=MuscleConstants.A_HILL, 
                B_hill=MuscleConstants.B_HILL, F_max_ecc=MuscleConstants.F_MAX_ECC, 
                K_ecc=MuscleConstants.K_ECC):
        epsilon = 1e-6
        # Clamp velocity to prevent div by zero
        dLm_dt_clamped = torch.clamp(dLm_dt, min=-0.95 * V_max, max=V_max)

        # Shortening (dLm_dt <= 0)
        FV_shortening = (B_hill - dLm_dt_clamped) / (B_hill + A_hill * dLm_dt_clamped + epsilon)
        
        # Lengthening (dLm_dt > 0)
        FV_lengthening = (F_max_ecc - 1.0) * (1.0 - torch.exp(-K_ecc * dLm_dt_clamped)) + 1.0
        
        return torch.where(dLm_dt_clamped <= 0.0, FV_shortening, FV_lengthening)

def compute_muscle_lengths(theta_s, theta_e, theta_s_ref, theta_e_ref, M_m):
    """Calculates muscle length changes via moment arm matrix."""
    delta_theta = torch.tensor([theta_s - theta_s_ref, theta_e - theta_e_ref], dtype=torch.float32)
    return 1.0 + (-M_m @ delta_theta)

def compute_jacobian(theta_s, theta_e, l1, l2):
    """Analytic Jacobian for 2-link planar arm."""
    s1, c1 = np.sin(theta_s), np.cos(theta_s)
    s12, c12 = np.sin(theta_s + theta_e), np.cos(theta_s + theta_e)
    return np.array([[-l1*s1 - l2*s12, -l2*s12], 
                     [ l1*c1 + l2*c12,  l2*c12]])

def forward_kinematics_torch(qpos, l1, l2):
    """Computes (x, y) from joint angles (Tensor safe)."""
    ts, te = qpos[..., 0], qpos[..., 1]
    x = l1 * torch.cos(ts) + l2 * torch.cos(ts + te)
    y = l1 * torch.sin(ts) + l2 * torch.sin(ts + te)
    return x, y

def differentiable_physics_step(qpos, qvel, normalized_muscle_force, external_torque, M_m, inertia=0.1, damping=0.4, dt=0.01):
    """Differentiable physics integration step."""
    muscle_forces = normalized_muscle_force * 60.0
    internal_torque = M_m.T @ muscle_forces
    total_torque = internal_torque + external_torque
    
    joint_accel = (total_torque - damping * qvel) / inertia
    
    new_qvel = torch.clamp(qvel + joint_accel * dt, -20, 20)
    new_qpos = qpos + new_qvel * dt
    
    return new_qpos, new_qvel

def update_muscle_kinematics(qpos, qvel, theta_ref, L_m_init, M_m, dt, vel_Lm_prev, acc_Lm_prev):
    """Updates muscle lengths using differentiable tensors."""
    if not isinstance(theta_ref, torch.Tensor): 
        theta_ref = torch.tensor(theta_ref, dtype=torch.float32)
    
    delta_theta = qpos - theta_ref
    length_change = (-M_m @ delta_theta)
    new_L_m = L_m_init + length_change
    dLm_dt = (-M_m @ qvel)
    
    new_acc = (dLm_dt - vel_Lm_prev) / dt
    new_jerk = (new_acc - acc_Lm_prev) / dt
    
    return new_L_m, length_change, dLm_dt, new_acc, new_jerk, dLm_dt