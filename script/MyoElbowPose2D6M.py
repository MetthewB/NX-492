from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import torch

class MyoElbowPose2D6MFixed(Env):
    def __init__(self):
        super().__init__()
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -10.0, -np.pi, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.pi, 10.0, np.pi, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # Target positions for the shoulder and elbow joints
        self.target_shoulder_pos = 0.0
        self.target_elbow_pos = 0.0

        # External torque applied by the robot
        self.external_torque = np.zeros(2)  # [shoulder_torque, elbow_torque]

        # Initialize integral error terms for integral control
        self.integral_error_shoulder = 0.0
        self.integral_error_elbow = 0.0
        
        # Initialize state variables
        self.reset_state()

    def reset_state(self):
        """Initialize or reset the state variables."""
        self.shoulder_pos = 0.0
        self.shoulder_vel = 0.0
        self.elbow_pos = 0.0
        self.elbow_vel = 0.0
        self.muscle_activations = np.zeros(6)
        self.pose_error = 0.0
        self.current_step = 0

        self.update_state()

    def reset(self, seed=None, options=None, initial_qpos=None, target_qpos=None, offset=(0.0, 0.0)):
        """Reset the environment to its initial state."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.reset_state()

        if initial_qpos is not None:
            self.set_joint_angles(initial_qpos)

        if target_qpos is not None:
            self.target_shoulder_pos = target_qpos[0] - offset[0]
            self.target_elbow_pos = target_qpos[1] - offset[1]

        self.update_state()

        return self.state, {}

    def set_external_torque(self, external_torque):
        self.external_torque = external_torque

    def update_state(self):
        """Update the state of the environment based on the current joint angles."""
        self.state = np.array([
            self.shoulder_pos,
            self.shoulder_vel,
            self.elbow_pos,
            self.elbow_vel,
            *self.muscle_activations
        ], dtype=np.float32)

    def set_joint_angles(self, joint_angles):
        """Set the joint angles directly in the environment."""
        if len(joint_angles) != 4:
            raise ValueError("joint_angles must be a list or array of length 4: [shoulder_pos, shoulder_vel, elbow_pos, elbow_vel]")

        self.shoulder_pos = joint_angles[0]
        self.shoulder_vel = joint_angles[1]
        self.elbow_pos = joint_angles[2]
        self.elbow_vel = joint_angles[3]

        self.update_state()

    def step(self, action):
        """Perform one step in the environment with external torque."""
        self.muscle_activations = np.clip(action, self.action_space.low, self.action_space.high)
        muscle_forces = self.muscle_activations * 100.0  # Scale muscle forces

        # Compute joint torques from muscle effort (Internal Torque)
        internal_shoulder_torque = self._compute_torque(muscle_forces, joint="shoulder")
        internal_elbow_torque = self._compute_torque(muscle_forces, joint="elbow")

        # Total torque = Internal Muscle Torque + External Robot Torque
        self.total_shoulder_torque = internal_shoulder_torque + self.external_torque[0]
        self.total_elbow_torque = internal_elbow_torque + self.external_torque[1]

        # Update joint velocities and positions
        self._update_joint_dynamics(self.total_shoulder_torque, self.total_elbow_torque)

        # Compute pose error and reward
        self.pose_error = self._compute_pose_error()
        reward = self._compute_reward()

        # Check termination conditions
        perturbation_occurred = self.current_step >= 250
        terminated = (self.current_step >= 300 or (self.pose_error < 0.001 and perturbation_occurred))
        self.current_step += 1

        return self._get_obs(), reward, terminated, False, {}

    def _compute_torque(self, muscle_forces, joint):
        """Compute the torque for a given joint from muscle forces + implicit spring term (for stability)."""
        if joint == "shoulder":
            torque = (
                -muscle_forces[0] * 0.01 - muscle_forces[1] * 0.02 - muscle_forces[2] * 0.03 +
                muscle_forces[3] * 0.03 + muscle_forces[4] * 0.02 + muscle_forces[5] * 0.01
            )
            torque += -0.5 * self.shoulder_vel

        elif joint == "elbow":
            torque = (
                -muscle_forces[0] * 0.005 - muscle_forces[1] * 0.01 - muscle_forces[2] * 0.015 +
                muscle_forces[3] * 0.015 + muscle_forces[4] * 0.01 + muscle_forces[5] * 0.005
            )
            torque += -0.5 * self.elbow_vel

        return torque

    def _update_joint_dynamics(self, shoulder_torque, elbow_torque):
        """Update joint velocities and positions based on total torques."""
        damping = 0.02
        dt = 0.01

        # Update velocities
        self.shoulder_vel += (shoulder_torque / 0.5) * dt  
        self.elbow_vel += (elbow_torque / 0.5) * dt       

        # Apply damping
        self.shoulder_vel *= (1 - damping)
        self.elbow_vel *= (1 - damping)

        # Update positions (semi-implicit Euler)
        self.shoulder_pos += self.shoulder_vel * dt
        self.elbow_pos += self.elbow_vel * dt

    def _compute_pose_error(self):
        """Compute the current pose error."""
        shoulder_error = self.shoulder_pos - self.target_shoulder_pos
        elbow_error = self.elbow_pos - self.target_elbow_pos
        pose_error = np.sqrt(shoulder_error**2 + elbow_error**2)
        return pose_error

    def _compute_reward(self):
        """Compute the reward for the current step."""
        # Convert muscle activations to a PyTorch tensor
        muscle_activations = torch.tensor(self.muscle_activations, dtype=torch.float32)

        # Compute penalties
        activation_penalty = torch.sum(muscle_activations**2).item()
        velocity_penalty = self.shoulder_vel**2 + self.elbow_vel**2
        time_penalty = 0.001 * self.current_step
        return -50 * self.pose_error - 0.00 * activation_penalty - 0.01 * velocity_penalty - 0.1 * time_penalty
        
    def _get_obs(self):
        """Get the current observation."""
        obs = np.array([
            self.shoulder_pos,
            self.shoulder_vel,
            self.elbow_pos,
            self.elbow_vel,
            *self.muscle_activations
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)