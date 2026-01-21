import torch
import torch.nn as nn
from config import *

class NeuralLimbController(nn.Module):
    def __init__(self, n_inputs, n_outputs=n_muscles):
        super().__init__()
        
        # Sensory layer
        self.w_sr = nn.Parameter(torch.empty(n_s, n_s))
        self.w_si = nn.Parameter(torch.empty(n_s, n_inputs))
        self.b_s = nn.Parameter(torch.zeros(n_s))

        # M1 layer
        self.w_mr = nn.Parameter(torch.empty(n_m1, n_m1))
        self.w_mi = nn.Parameter(torch.empty(n_m1, n_s))
        self.b_m = nn.Parameter(torch.zeros(n_m1))

        # Muscle activation layer
        self.w_act = nn.Parameter(torch.empty(n_muscles, n_m1))
        
        # Initialization
        self._initialize_stable()

        # State variables
        self.y_s = torch.zeros(n_s)
        self.y_m = torch.zeros(n_m1)
        self.y_act = torch.zeros(n_muscles)

    def _initialize_stable(self):
        nn.init.orthogonal_(self.w_sr)
        nn.init.orthogonal_(self.w_mr)
        nn.init.xavier_uniform_(self.w_si)
        nn.init.xavier_uniform_(self.w_mi)
        nn.init.xavier_uniform_(self.w_act)
    
    def reset_states(self):
        self.y_s.zero_()
        self.y_m.zero_()
        self.y_act.zero_()

    def reset_weights(self):
        self._initialize_stable() 

    def neural_activation_step(self, y_t, u_t):
        return (1 - (dt / tau_n)) * y_t + (dt / tau_n) * torch.tanh(u_t)
    
    def muscle_activation_step(self, y_act_t, u_act_t):
        activation = torch.nn.functional.softplus(u_act_t, beta=5)
        return (1 - (dt / tau_m)) * y_act_t + (dt / tau_m) * activation

    def forward_step(self, u_fb_cortical, u_spinal_reflex, y_s_prev, y_m_prev, y_act_prev):
        # Sensory layer
        u_s = self.w_sr @ y_s_prev + self.w_si @ u_fb_cortical + self.b_s
        self.y_s = self.neural_activation_step(y_s_prev, u_s) 

        # Motor layer
        u_m = self.w_mr @ y_m_prev + self.w_mi @ self.y_s + self.b_m
        self.y_m = self.neural_activation_step(y_m_prev, u_m)

        # Muscle activation
        u_act = self.w_act @ self.y_m + u_spinal_reflex
        self.y_act = self.muscle_activation_step(y_act_prev, u_act)

        return self.y_act

def compute_feedback_signals(obs, d_qpos, d_qvel, d_aff, target, G_p, G_d, G_f, F_m):
    """Assembles cortical (LLR) feedback vectors with Normalization."""
    pos_error = d_qpos - target.detach()
    norm_pos_error = pos_error / 0.5 
    norm_vel = d_qvel / 10.0
    norm_force = F_m / 60.0
    
    u_cortical = torch.cat([
        G_p * norm_pos_error, 
        G_d * norm_vel, 
        G_f * norm_force
    ], dim=0)
    
    return u_cortical, d_aff