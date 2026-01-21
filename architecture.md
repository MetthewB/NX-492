# Model Architecture

---

#### **Key Components to Implement**
1. **Recurrent Neural Networks (RNNs)**:
   - Two layers: Sensory layer and M1 layer.
   - Sensory layer processes proprioceptive feedback.
   - M1 layer generates descending commands to muscles.

2. **Spinal Reflex**:
   - Afferent feedback from muscle kinematics (length, velocity, acceleration, jerk).
   - Reflex gain applied to the afferent feedback.

3. **Muscle Activation Dynamics**:
   - Combines descending commands from M1 and spinal reflex.
   - Includes time delays for neural and spinal reflex pathways.

4. **Musculoskeletal Model**:
   - Two-segment arm with six muscles.
   - Muscle activations produce forces, which generate joint torques and motion.

5. **Feedback Loop**:
   - Joint angles, velocities, and muscle states are fed back to the RNNs and spinal reflex.

---

#### **1. Sensory Layer (`y_s`)**
- **Description**: Represents the sensory cortex, which processes sensory feedback (`u_fb`).
- **Number of Units**: `n_s = 200` (as per the researchers' description).
- **Shape**: `[n_s]` or `[200]` (a 1D tensor with 200 elements).

#### **2. Motor Layer (`y_m`)**
- **Description**: Represents the motor cortex, which generates descending commands to muscles.
- **Number of Units**: `n_m1 = 200` (as per the researchers' description).
- **Shape**: `[n_m1]` or `[200]` (a 1D tensor with 200 elements).

#### **3. Muscle Activation Layer (`y_act`)**
- **Description**: Projects the motor cortex output to the muscles. The number of units matches the number of muscles.
- **Number of Units**: `n_muscles = 6` (6 muscles in the `myoElbow` model).
- **Shape**: `[n_muscles]` or `[6]` (a 1D tensor with 6 elements).

#### **4. Feedback Input (`u_fb`)**
- **Description**: Proprioceptive feedback, including joint angles, velocities, and muscle forces.
- **Components**:
  - `θ_d(t)`: Delayed joint angles (e.g., shoulder and elbow).
  - `θ_g`: Goal joint angles.
  - `F_m`: Muscle forces (6 muscles).
- **Shape**: `[n_inputs]` or `[10]` (2 joint angles + 2 joint velocities + 6 muscle forces).

#### **5. Afferent Feedback (`x_aff`)**
- **Description**: Encodes muscle kinematics (length, velocity, acceleration, jerk) for each muscle.
- **Components**:
  - `Δx_ml`: Muscle length changes.
  - `x_mv`: Muscle velocities.
  - `x_ma`: Muscle accelerations.
  - `x_mj`: Muscle jerks.
- **Shape**: `[n_muscles, 4]` or `[6, 4]` (6 muscles × 4 kinematic states).

#### **6. Weight Matrices**
- **Sensory Layer Weights (`w_sr`, `w_si`)**:
  - `w_sr`: Recurrent weights for the sensory layer.
    - Shape: `[n_s, n_s]` or `[200, 200]`.
  - `w_si`: Weights connecting sensory feedback to the sensory layer.
    - Shape: `[n_s, n_inputs]` or `[200, 10]`.

- **Motor Layer Weights (`w_mr`, `w_mi`)**:
  - `w_mr`: Recurrent weights for the motor layer.
    - Shape: `[n_m1, n_m1]` or `[200, 200]`.
  - `w_mi`: Weights connecting the sensory layer to the motor layer.
    - Shape: `[n_m1, n_s]` or `[200, 200]`.

- **Muscle Activation Weights (`w_act`)**:
  - Weights connecting the motor layer to the muscle activation layer.
    - Shape: `[n_muscles, n_m1]` or `[6, 200]`.

#### **7. Bias Terms**
- **Sensory Layer Bias (`b_s`)**:
  - Shape: `[n_s]` or `[200]`.
- **Motor Layer Bias (`b_m`)**:
  - Shape: `[n_m1]` or `[200]`.

#### **8. Reflex Parameters**
- **Stretch Reflex Gain (`G_s`)**:
  - Scalar value (e.g., `0.8` or `1.9`).
- **Afferent Encoding Gains (`G_aff`)**:
  - Shape: `[4]` (for length, velocity, acceleration, jerk).

#### **9. Neural Dynamics**
- **Sensory Layer Dynamics**:
  - `y_s(t+Δt) = (1-(Δt/τ_n))*y_s(t) + (Δt/τ_n) * tanh(u_s(t))`
  - Shape of `y_s`: `[n_s]` or `[200]`.
- **Motor Layer Dynamics**:
  - `y_m(t+Δt) = (1-(Δt/τ_n))*y_m(t) + (Δt/τ_n) * tanh(u_m(t))`
  - Shape of `y_m`: `[n_m1]` or `[200]`.
- **Muscle Activation Dynamics**:
  - `y_act(t+Δt) = (1-(Δt/τ_m))*y_act(t) + (Δt/τ_m) * ReLU(u_act(t))`
  - Shape of `y_act`: `[n_muscles]` or `[6]`.

#### **10. Observation (`obs`)**
- **Description**: The observation returned by the environment.
- **Components**:
  - `qpos`: Joint positions (e.g., shoulder and elbow).
  - `qvel`: Joint velocities.
  - `act`: Muscle activations.
  - `pose_err`: Pose error.
- **Shape**: `[11]` (as per the `myoElbow` environment).

#### **Summary of Shapes**

| Component                  | Shape         | Description                                      |
|----------------------------|---------------|--------------------------------------------------|
| `y_s` (Sensory Layer)      | `[200]`       | Sensory cortex output.                          |
| `y_m` (Motor Layer)        | `[200]`       | Motor cortex output.                            |
| `y_act` (Muscle Activation)| `[6]`         | Muscle activation output.                       |
| `u_fb` (Feedback Input)    | `[10]`         | Proprioceptive feedback.                        |
| `x_aff` (Afferent Feedback)| `[6, 4]`      | Muscle kinematics (length, velocity, etc.).     |
| `w_sr`                    | `[200, 200]`  | Sensory layer recurrent weights.               |
| `w_si`                    | `[200, 10]`    | Sensory feedback to sensory layer weights.      |
| `w_mr`                    | `[200, 200]`  | Motor layer recurrent weights.                 |
| `w_mi`                    | `[200, 200]`  | Sensory layer to motor layer weights.           |
| `w_act`                   | `[6, 200]`    | Motor layer to muscle activation weights.       |
| `b_s` (Sensory Bias)       | `[200]`       | Sensory layer bias.                             |
| `b_m` (Motor Bias)         | `[200]`       | Motor layer bias.                               |
| `G_s`                     | `scalar`      | Stretch reflex gain.                            |
| `G_aff`                   | `[4]`         | Afferent encoding gains.                        |
| `obs`                     | `[11]`         | Observation from the environment.               |