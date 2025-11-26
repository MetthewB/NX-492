from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np

class MyoElbowPose2D6MFixed(Env):
    def __init__(self):
        super().__init__()
        
        # Define observation space (shoulder + elbow positions and velocities, muscle activations, pose error)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(11,), dtype=np.float32)
        
        # Define action space (muscle activations for 6 muscles)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Initialize state variables
        self.shoulder_pos = 0.0
        self.shoulder_vel = 0.0
        self.elbow_pos = 0.0
        self.elbow_vel = 0.0
        self.muscle_activations = np.zeros(6)
        self.pose_error = 0.0

        # Target positions for the joints
        self.target_shoulder_pos = 1.0
        self.target_elbow_pos = 1.0

    def reset(self, seed=None, options=None, initial_qpos=None, target_qpos=None):
        # Set the seed for reproducibility
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset the state variables
        self.current_step = 0
        self.shoulder_vel = 0.0
        self.elbow_vel = 0.0
        self.muscle_activations = np.zeros(6)

        # Set initial joint positions
        if initial_qpos is not None:
            self.shoulder_pos, self.elbow_pos = initial_qpos
        else:
            self.shoulder_pos = 0.6898
            self.elbow_pos = 2.1120

        # Set target joint positions
        if target_qpos is not None:
            self.target_shoulder_pos, self.target_elbow_pos = target_qpos
        else:
            # Generate random target positions within a specified range
            self.target_shoulder_pos = np.random.uniform(0.0, 1.0)  # Shoulder angle range
            self.target_elbow_pos = np.random.uniform(1.5, 2.5)     # Elbow angle range

        # Recompute the initial pose error
        self.pose_error = np.sqrt(
            (self.shoulder_pos - self.target_shoulder_pos)**2 +
            (self.elbow_pos - self.target_elbow_pos)**2
        )

        # Return the initial observation
        return self._get_obs(), {}

    def step(self, action):
        # Update muscle activations
        self.muscle_activations = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute muscle forces (proportional to activations)
        muscle_forces = self.muscle_activations * 10.0  # Scale factor for muscle forces

        # Compute joint torques
        shoulder_torque = (
            -muscle_forces[0] * 0.02
            - muscle_forces[1] * 0.03
            - muscle_forces[2] * 0.04
            + muscle_forces[3] * 0.04
            + muscle_forces[4] * 0.03
            + muscle_forces[5] * 0.02
        )

        elbow_torque = (
            -muscle_forces[0] * 0.01
            - muscle_forces[1] * 0.02
            - muscle_forces[2] * 0.03
            + muscle_forces[3] * 0.03
            + muscle_forces[4] * 0.02
            + muscle_forces[5] * 0.01
        )

        # Add proportional control to drive the joints toward the target positions
        shoulder_torque += - 5.0 * (self.shoulder_pos - self.target_shoulder_pos)
        elbow_torque += - 5.0 * (self.elbow_pos - self.target_elbow_pos)

        # Update joint velocities (considering inertia and damping)
        shoulder_inertia = 1.0   # Inertia for the shoulder joint
        elbow_inertia = 1.0      # Inertia for the elbow joint
        damping = 0.05           # Reduced damping coefficient

        self.shoulder_vel += (shoulder_torque / shoulder_inertia) * 0.01
        self.elbow_vel += (elbow_torque / elbow_inertia) * 0.01

        # Apply damping to the velocities
        self.shoulder_vel *= (1 - damping)
        self.elbow_vel *= (1 - damping)

        # Update joint positions
        self.shoulder_pos += self.shoulder_vel * 0.02
        self.elbow_pos += self.elbow_vel * 0.02

        # Compute pose error
        self.pose_error = np.sqrt(
            (self.shoulder_pos - self.target_shoulder_pos)**2 +
            (self.elbow_pos - self.target_elbow_pos)**2
        )

        # Compute reward
        activation_penalty = np.sum(self.muscle_activations**2)
        velocity_penalty = self.shoulder_vel**2 + self.elbow_vel**2
        time_penalty = 0.001 * self.current_step

        reward = -self.pose_error - 0.01 * activation_penalty - 0.01 * velocity_penalty - time_penalty

        # Check termination condition
        max_steps = 1000           # Maximum number of steps per episode
        success_threshold = 0.001  # Pose error threshold for success
        self.current_step += 1

        terminated, truncated = False, False

        if self.current_step >= max_steps:
            terminated = True
        elif self.pose_error < success_threshold:
            terminated = True

        # Return the observation, reward, and termination flags
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.array([
            self.shoulder_pos,
            self.shoulder_vel,
            self.elbow_pos,
            self.elbow_vel,
            *self.muscle_activations,
            self.pose_error
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)