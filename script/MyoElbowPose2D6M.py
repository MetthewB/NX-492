from gymnasium import Env, spaces
import numpy as np

class MyoElbowPose2D6MFixed(Env):
    def __init__(self):
        super().__init__()
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        
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

    def reset(self, seed=None, options=None, initial_qpos=None, target_qpos=None):
        """Reset the environment to its initial state."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.reset_state()

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
            self.target_shoulder_pos = np.random.uniform(0.0, 1.0)  # Shoulder angle range
            self.target_elbow_pos = np.random.uniform(1.5, 2.5)     # Elbow angle range

        # Compute initial pose error
        self.pose_error = self._compute_pose_error()

        return self._get_obs(), {}

    def step(self, action):
        """Perform one step in the environment."""
        self.muscle_activations = np.clip(action, self.action_space.low, self.action_space.high)
        muscle_forces = self.muscle_activations * 10.0  # Scale muscle forces

        # Compute joint torques
        shoulder_torque = self._compute_torque(muscle_forces, joint="shoulder")
        elbow_torque = self._compute_torque(muscle_forces, joint="elbow")

        # Update joint velocities and positions
        self._update_joint_dynamics(shoulder_torque, elbow_torque)

        # Compute pose error and reward
        self.pose_error = self._compute_pose_error()
        reward = self._compute_reward()

        # Check termination conditions
        terminated = self.current_step >= 150 or self.pose_error < 0.001
        self.current_step += 1

        return self._get_obs(), reward, terminated, False, {}

    def _compute_torque(self, muscle_forces, joint):
        """Compute the torque for a given joint."""
        if joint == "shoulder":
            torque = (
                -muscle_forces[0] * 0.02 - muscle_forces[1] * 0.03 - muscle_forces[2] * 0.04 +
                muscle_forces[3] * 0.04 + muscle_forces[4] * 0.03 + muscle_forces[5] * 0.02
            )
            torque += -5.0 * (self.shoulder_pos - self.target_shoulder_pos)
        elif joint == "elbow":
            torque = (
                -muscle_forces[0] * 0.01 - muscle_forces[1] * 0.02 - muscle_forces[2] * 0.03 +
                muscle_forces[3] * 0.03 + muscle_forces[4] * 0.02 + muscle_forces[5] * 0.01
            )
            torque += -5.0 * (self.elbow_pos - self.target_elbow_pos)
        return torque

    def _update_joint_dynamics(self, shoulder_torque, elbow_torque):
        """Update joint velocities and positions based on torques."""
        damping = 0.05
        dt = 0.01  # Time step

        # Update velocities
        self.shoulder_vel += (shoulder_torque / 1.0) * dt  # Shoulder inertia = 1.0
        self.elbow_vel += (elbow_torque / 1.0) * dt        # Elbow inertia = 1.0

        # Apply damping
        self.shoulder_vel *= (1 - damping)
        self.elbow_vel *= (1 - damping)

        # Update positions
        self.shoulder_pos += self.shoulder_vel * 0.02  # Position update with dt = 0.02
        self.elbow_pos += self.elbow_vel * 0.02

    def _compute_pose_error(self):
        """Compute the current pose error."""
        return np.sqrt(
            (self.shoulder_pos - self.target_shoulder_pos)**2 +
            (self.elbow_pos - self.target_elbow_pos)**2
        )

    def _compute_reward(self):
        """Compute the reward for the current step."""
        activation_penalty = np.sum(self.muscle_activations**2)
        velocity_penalty = self.shoulder_vel**2 + self.elbow_vel**2
        time_penalty = 0.001 * self.current_step
        return -self.pose_error - 0.01 * activation_penalty - 0.01 * velocity_penalty - time_penalty

    def _get_obs(self):
        """Get the current observation."""
        obs = np.array([
            self.shoulder_pos,
            self.shoulder_vel,
            self.elbow_pos,
            self.elbow_vel,
            *self.muscle_activations,
            self.pose_error
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)