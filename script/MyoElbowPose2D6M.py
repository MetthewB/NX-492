from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np

class MyoElbowPose2D6MFixed(Env):
    def __init__(self):
        super().__init__()
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # External torque applied by the robot
        self.external_torque = np.zeros(2)  # [shoulder_torque, elbow_torque]
        
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
            self.shoulder_pos = 0.61
            self.elbow_pos = 1.57

        # Set target joint positions
        if target_qpos is not None:
            self.target_shoulder_pos, self.target_elbow_pos = target_qpos
        else:
            self.target_shoulder_pos = 0.61
            self.target_elbow_pos = 1.57

        # Compute initial pose error
        self.pose_error = self._compute_pose_error()

        return self._get_obs(), {}

    def set_external_torque(self, external_torque):
        self.external_torque = external_torque

    def step(self, action):
        """Perform one step in the environment with external torque."""
        self.muscle_activations = np.clip(action, self.action_space.low, self.action_space.high)
        muscle_forces = self.muscle_activations * 10.0  # Scale muscle forces

        # Compute joint torques from muscle effort (Internal Torque)
        internal_shoulder_torque = self._compute_torque(muscle_forces, joint="shoulder")
        internal_elbow_torque = self._compute_torque(muscle_forces, joint="elbow")

        # Total torque = Internal Muscle Torque + External Robot Torque
        total_shoulder_torque = internal_shoulder_torque + self.external_torque[0]
        total_elbow_torque = internal_elbow_torque + self.external_torque[1]

        # Update joint velocities and positions
        self._update_joint_dynamics(total_shoulder_torque, total_elbow_torque)

        # Compute pose error and reward
        self.pose_error = self._compute_pose_error()
        reward = self._compute_reward()

        # Check termination conditions
        terminated = self.current_step >= 300 or self.pose_error < 0.01
        self.current_step += 1

        return self._get_obs(), reward, terminated, False, {}

    def _compute_torque(self, muscle_forces, joint):
        """Compute the torque for a given joint from muscle forces + implicit spring term (for stability)."""
        # NOTE: The movement arm matrix M_m from the notebook is more detailed. 
        # This implementation uses simplified hardcoded moment arms.
        
        if joint == "shoulder":
            # Simplified muscle moment arm contribution
            torque = (
                -muscle_forces[0] * 0.02 - muscle_forces[1] * 0.03 - muscle_forces[2] * 0.04 +
                muscle_forces[3] * 0.04 + muscle_forces[4] * 0.03 + muscle_forces[5] * 0.02
            )
            # Implicit spring term (for basic stability, often present in simplified models)
            torque += -5.0 * (self.shoulder_pos - self.target_shoulder_pos)
        elif joint == "elbow":
            # Simplified muscle moment arm contribution
            torque = (
                -muscle_forces[0] * 0.01 - muscle_forces[1] * 0.02 - muscle_forces[2] * 0.03 +
                muscle_forces[3] * 0.03 + muscle_forces[4] * 0.02 + muscle_forces[5] * 0.01
            )
            # Implicit spring term (for basic stability)
            torque += -5.0 * (self.elbow_pos - self.target_elbow_pos)
        return torque

    def _update_joint_dynamics(self, shoulder_torque, elbow_torque):
        """Update joint velocities and positions based on total torques."""
        damping = 0.05
        dt = 0.01  # Time step

        # Update velocities (simple Euler integration: a = tau/I)
        self.shoulder_vel += (shoulder_torque / 1.0) * dt  # Shoulder inertia = 1.0
        self.elbow_vel += (elbow_torque / 1.0) * dt        # Elbow inertia = 1.0

        # Apply damping
        self.shoulder_vel *= (1 - damping)
        self.elbow_vel *= (1 - damping)

        # Update positions
        self.shoulder_pos += self.shoulder_vel * dt
        self.elbow_pos += self.elbow_vel * dt

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