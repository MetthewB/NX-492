from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import torch

class MyoElbowPose2D6MFixed(Env):
    def __init__(self):
        super().__init__()
        
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -20.0, -np.pi, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.pi, 20.0, np.pi, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # Environment constants (Synchronized with Training Loop)
        self.dt = 0.01
        self.inertia = 0.1
        self.damping_coefficient = 0.4
        
        # Moment Arm Matrix (Must match M_m in training script)
        # Columns: [Shoulder_MA, Elbow_MA]
        self.M_m = np.array([
            [ 0.10,  0.00], # Mono Shoulder Flexor
            [-0.10,  0.00], # Mono Shoulder Extensor
            [ 0.00,  0.10], # Mono Elbow Flexor
            [ 0.00, -0.10], # Mono Elbow Extensor
            [ 0.08,  0.08], # Bi-articular Flexor
            [-0.08, -0.08]  # Bi-articular Extensor
        ])

        self.target_shoulder_pos = 0.0
        self.target_elbow_pos = 0.0
        self.external_torque = np.zeros(2) 
        
        self.reset_state()

    def reset_state(self):
        self.shoulder_pos = 0.0
        self.shoulder_vel = 0.0
        self.elbow_pos = 0.0
        self.elbow_vel = 0.0
        self.muscle_activations = np.zeros(6)
        self.pose_error = 0.0
        self.current_step = 0
        self.update_state()
    
    def reset(self, seed=None, options=None, initial_qpos=None, target_qpos=None, offset=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.reset_state()
        
        if initial_qpos is not None:
            self.set_joint_angles(initial_qpos)
            
        if target_qpos is not None:
            self.target_shoulder_pos = target_qpos[0] 
            self.target_elbow_pos = target_qpos[1]
            
        self.update_state()
        return self.state, {}

    def set_external_torque(self, external_torque):
        # external_torque should be calculated as -(J.T @ force) in the loop
        self.external_torque = external_torque

    def step(self, action):
        self.muscle_activations = np.clip(action, 0.0, 1.0)
        
        # SYNC 1: Match Force Scaling (using 60.0 to match the loop)
        muscle_forces = self.muscle_activations * 60.0 

        # SYNC 2: Use Moment Arm Matrix for Torque
        # Torque = M_m.T @ muscle_forces
        joint_torques = self.M_m.T @ muscle_forces
        
        total_shoulder_torque = joint_torques[0] + self.external_torque[0]
        total_elbow_torque = joint_torques[1] + self.external_torque[1]

        # SYNC 3: Update Joint Dynamics with light inertia and explicit damping
        self._update_joint_dynamics(total_shoulder_torque, total_elbow_torque)

        self.pose_error = self._compute_pose_error()
        self.current_step += 1
        
        terminated = self.current_step >= 300
        return self._get_obs(), 0.0, terminated, False, {}

    def _update_joint_dynamics(self, shoulder_torque, elbow_torque):
        # Physics implementation matching differentiable_physics_step
        # 1. Accelerations (tau - damping*v) / I
        acc_s = (shoulder_torque - self.damping_coefficient * self.shoulder_vel) / self.inertia
        acc_e = (elbow_torque - self.damping_coefficient * self.elbow_vel) / self.inertia

        # 2. Integration
        self.shoulder_vel += acc_s * self.dt
        self.elbow_vel += acc_e * self.dt
        
        # 3. Safety Clamps (prevents NaN from high speed)
        self.shoulder_vel = np.clip(self.shoulder_vel, -20, 20)
        self.elbow_vel = np.clip(self.elbow_vel, -20, 20)

        self.shoulder_pos += self.shoulder_vel * self.dt
        self.elbow_pos += self.elbow_vel * self.dt

    def _compute_pose_error(self):
        # Error in meters (for the environment's internal check)
        shoulder_error = self.shoulder_pos - self.target_shoulder_pos
        elbow_error = self.elbow_pos - self.target_elbow_pos
        return np.sqrt(shoulder_error**2 + elbow_error**2)

    def set_joint_angles(self, joint_angles):
        self.shoulder_pos, self.shoulder_vel, self.elbow_pos, self.elbow_vel = joint_angles
        self.update_state()

    def update_state(self):
        self.state = np.array([
            self.shoulder_pos, self.shoulder_vel, self.elbow_pos, self.elbow_vel,
            *self.muscle_activations
        ], dtype=np.float32)

    def _get_obs(self):
        return self.update_state() or self.state