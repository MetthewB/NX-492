"""
MyoElbowPose2D6MFixed.py
Custom Gymnasium Environment for a 2-DOF Arm with 6 Muscles.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyoElbowPose2D6MFixed(gym.Env):
    def __init__(self):
        super().__init__()
        
        # --- 1. Action & Observation Spaces ---
        # Observation: [theta_s, dot_theta_s, theta_e, dot_theta_e, m_act_1...m_act_6]
        high_obs = np.array([np.pi, 20.0, np.pi, 20.0] + [1.0]*6, dtype=np.float32)
        low_obs = np.array([-np.pi, -20.0, -np.pi, -20.0] + [0.0]*6, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # --- 2. Physics Constants ---
        # Synchronized with Training Loop
        self.dt = 0.01
        self.inertia = 0.1
        self.damping_coefficient = 0.4
        
        # --- 3. Muscle Geometry (Moment Arms) ---
        # Rows: Muscles (0-5), Columns: Joints (Shoulder, Elbow)
        self.M_m = np.array([
            [ 0.10,  0.00], # Muscle 0: Mono Shoulder Flexor
            [-0.10,  0.00], # Muscle 1: Mono Shoulder Extensor
            [ 0.00,  0.10], # Muscle 2: Mono Elbow Flexor
            [ 0.00, -0.10], # Muscle 3: Mono Elbow Extensor
            [ 0.08,  0.08], # Muscle 4: Bi-articular Flexor
            [-0.08, -0.08]  # Muscle 5: Bi-articular Extensor
        ], dtype=np.float32)

        # --- 4. Internal State Init ---
        self.target_shoulder_pos = 0.0
        self.target_elbow_pos = 0.0
        self.external_torque = np.zeros(2) 
        self._reset_internal_state()

    # =========================================================================
    # Core Gym Interface
    # =========================================================================

    def reset(self, seed=None, options=None, initial_qpos=None, target_qpos=None, offset=None):
        """Resets the environment to a clean initial state."""
        super().reset(seed=seed)
        self._reset_internal_state()
        
        # Optional Overrides for Custom Start/Target
        if initial_qpos is not None:
            self.set_joint_angles(initial_qpos)
            
        if target_qpos is not None:
            self.target_shoulder_pos = target_qpos[0] 
            self.target_elbow_pos = target_qpos[1]
            
        return self._get_obs(), {}

    def step(self, action):
        """Executes one simulation step."""
        # 1. Apply Action (Clip inputs)
        self.muscle_activations = np.clip(action, 0.0, 1.0)
        
        # 2. Run Physics
        self._update_physics(self.muscle_activations)
        
        # 3. Update Metrics
        self.pose_error = self._compute_pose_error()
        self.current_step += 1
        
        # 4. Check Termination (Fixed horizon)
        terminated = self.current_step >= 300
        
        return self._get_obs(), 0.0, terminated, False, {}

    # =========================================================================
    # Physics Engine
    # =========================================================================

    def _update_physics(self, activations):
        """Integrates joint dynamics based on muscle and external torques."""
        # A. Force Generation (Scaled to match training loop max force)
        muscle_forces = activations * 60.0 

        # B. Internal Torque Calculation (M_m.T @ F)
        internal_torques = self.M_m.T @ muscle_forces
        
        # C. Total Torque Accumulation
        total_tau_s = internal_torques[0] + self.external_torque[0]
        total_tau_e = internal_torques[1] + self.external_torque[1]

        # D. Euler Integration with Damping
        # Acceleration = (Torque - Damping * Velocity) / Inertia
        acc_s = (total_tau_s - self.damping_coefficient * self.shoulder_vel) / self.inertia
        acc_e = (total_tau_e - self.damping_coefficient * self.elbow_vel) / self.inertia

        # Update Velocity
        self.shoulder_vel += acc_s * self.dt
        self.elbow_vel += acc_e * self.dt
        
        # Safety Clamping (Prevents numerical instability)
        self.shoulder_vel = np.clip(self.shoulder_vel, -20, 20)
        self.elbow_vel = np.clip(self.elbow_vel, -20, 20)

        # Update Position
        self.shoulder_pos += self.shoulder_vel * self.dt
        self.elbow_pos += self.elbow_vel * self.dt

    # =========================================================================
    # Helpers & State Management
    # =========================================================================

    def set_external_torque(self, external_torque):
        """Updates the external perturbation torque vector."""
        self.external_torque = np.array(external_torque, dtype=np.float32)

    def set_joint_angles(self, joint_angles):
        """Manually overrides joint state (useful for forcing initial conditions)."""
        self.shoulder_pos, self.shoulder_vel, self.elbow_pos, self.elbow_vel = joint_angles

    def _reset_internal_state(self):
        """Resets kinematic variables to zero."""
        self.shoulder_pos = 0.0
        self.shoulder_vel = 0.0
        self.elbow_pos = 0.0
        self.elbow_vel = 0.0
        self.muscle_activations = np.zeros(6)
        self.pose_error = 0.0
        self.current_step = 0

    def _compute_pose_error(self):
        """Computes Euclidean distance to target in joint space."""
        s_err = self.shoulder_pos - self.target_shoulder_pos
        e_err = self.elbow_pos - self.target_elbow_pos
        return np.sqrt(s_err**2 + e_err**2)

    def _get_obs(self):
        """Constructs the standard observation vector."""
        return np.array([
            self.shoulder_pos, self.shoulder_vel, 
            self.elbow_pos, self.elbow_vel,
            *self.muscle_activations
        ], dtype=np.float32)