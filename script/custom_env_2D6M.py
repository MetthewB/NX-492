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
        self.smoothed_pose_error = 0.0

    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset the state variables
        self.current_step = 0
        self.shoulder_pos = np.random.uniform(-1.0, 1.0)
        self.shoulder_vel = 0.0
        self.elbow_pos = np.random.uniform(-1.0, 1.0)
        self.elbow_vel = 0.0
        self.muscle_activations = np.zeros(6)
        self.pose_error = 0.0
        self.smoothed_pose_error = 0.0

        # Return the initial observation
        return self._get_obs(), {}

    def step(self, action):
        # Update muscle activations
        self.muscle_activations = action

        # Simulate dynamics (placeholder for actual dynamics)
        self.shoulder_pos += self.shoulder_vel * 0.01
        self.elbow_pos += self.elbow_vel * 0.01
        self.shoulder_vel += 0.0
        self.elbow_vel += 0.0

        # Compute pose error (placeholder for actual computation)
        self.pose_error = np.abs(self.shoulder_pos - 1.0) + np.abs(self.elbow_pos - 1.0)

        self.smoothed_pose_error = 0.9 * self.smoothed_pose_error + 0.1 * self.pose_error
        reward = -self.smoothed_pose_error

        # Check termination condition
        max_steps = 200  # Maximum number of steps per episode
        success_threshold = 0.1  # Pose error threshold for success
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