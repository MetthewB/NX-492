# -- Imports --
import torch
import numpy as np

# --- Simulation Time Parameters ---
dt = 0.01
tau_n = 0.02
tau_m = 0.05

Tb, Tsb, Tp, Tsp, T = 0.20, 1.00, 1.50, 2.50, 3.00
Tb_step = int(Tb / dt)
Tsb_step = int(Tsb / dt)
Tp_step = int(Tp / dt)
Tsp_step = int(Tsp / dt)
T_step = int(T / dt)

activation_delay_steps = int(0.03 / dt)
feedback_delay_steps = int(0.04 / dt)
reflex_delay = int(0.02 / dt)

# --- Neural Network & Gains ---
n_s, n_m1, n_muscles, n_inputs = 200, 200, 6, 10
G_p, G_d, G_f = 1.0, 1.0, 1.0
G_aff = torch.tensor([1.0, 0.1, 0.01, 0.0001])
G_s_low, G_s_high = 0.8, 1.9

# --- Biomechanics & Physics ---
l1, l2 = 0.12, 0.10  # Arm segment lengths

# Moment arm matrix (Muscle geometry)
M_m = torch.tensor([
    [ 0.10,  0.00],       # Muscle 0: Mono Shoulder Flexor
    [-0.10,  0.00],       # Muscle 1: Mono Shoulder Extensor
    [ 0.00,  0.10],       # Muscle 2: Mono Elbow Flexor
    [ 0.00, -0.10],       # Muscle 3: Mono Elbow Extensor
    [ 0.08,  0.08],       # Muscle 4: Bi-articular Flexor
    [-0.08, -0.08]        # Muscle 5: Bi-articular Extensor
], dtype=torch.float32)

# --- Training & Costs ---
alpha, beta, gamma = 0.0001, 0.001, 0.5  # Penalty coefficients
background_load_low, background_load_high, perturbation_load = 0.1, 0.55, 0.25

# --- Task Definitions ---
perturbation_directions = {
    "Right": np.array([1.0, 0.0]), "Left": np.array([-1.0, 0.0]),
    "Up": np.array([0.0, 1.0]), "Down": np.array([0.0, -1.0]),
    "Up-Right": np.array([0.707, 0.707]), "Up-Left": np.array([-0.707, 0.707]),
    "Down-Left": np.array([-0.707, -0.707]), "Down-Right": np.array([0.707, -0.707])
}