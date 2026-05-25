import numpy as np
import torch
import torch.nn as nn
import json

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

from run_deep_experiments import MLP, generate_random_orthogonal_matrix, run_deep_experiment
from run_residual_experiments import ResidualMLP

def run_deep_fw_experiment(seed, d=64, num_layers=4, num_tasks=5, rotation_scale=0.08, drift_scale=0.02, is_residual=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Initialize Base MLP
    if is_residual:
        base_model = ResidualMLP(d, num_layers)
    else:
        base_model = MLP(d, num_layers)
        
    for layer in base_model.layers:
        with torch.no_grad():
            layer.weight.copy_(generate_random_orthogonal_matrix(d))
            
    # 2. Generate Target Expert MLPs
    experts = []
    for t_idx in range(num_tasks):
        if is_residual:
            expert = ResidualMLP(d, num_layers)
        else:
            expert = MLP(d, num_layers)
            
        for l_idx in range(num_layers):
            W0 = base_model.layers[l_idx].weight.data.clone()
            
            skew = torch.randn(d, d)
            skew = 0.5 * (skew - skew.T)
            P = torch.linalg.matrix_exp(rotation_scale * skew)
            
            U = torch.randn(d, 3) / np.sqrt(d)
            V = torch.randn(d, 3) / np.sqrt(d)
            task_vector = torch.matmul(U, V.T)
            
            W_expert = torch.matmul(P, W0 + task_vector)
            expert.layers[l_idx].weight.data.copy_(W_expert)
            
        experts.append(expert)
        
    # 3. Compute Empirical Fisher Information for each Expert
    N_calib = 150
    X_calib = torch.randn(N_calib, d)
    
    fisher_dicts = []
    for t_idx, expert in enumerate(experts):
        expert_fishers = []
        for l_idx in range(num_layers):
            F_layer = torch.zeros_like(expert.layers[l_idx].weight.data)
            expert_fishers.append(F_layer)
            
        for n in range(N_calib):
            x_sample = X_calib[n:n+1]
            out = expert(x_sample)
            
            v = torch.randn_like(out)
            loss = torch.sum(v * out)
            
            expert.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for l_idx in range(num_layers):
                    grad = expert.layers[l_idx].weight.grad
                    if grad is not None:
                        expert_fishers[l_idx] += grad ** 2
                        
        for l_idx in range(num_layers):
            expert_fishers[l_idx] /= N_calib
            expert_fishers[l_idx] = torch.sqrt(expert_fishers[l_idx] + 1e-6)
            expert_fishers[l_idx] /= expert_fishers[l_idx].mean()
            
        fisher_dicts.append(expert_fishers)
        
    # 4. Parameter Drift
    observed_experts = []
    for t_idx, expert in enumerate(experts):
        if is_residual:
            obs_expert = ResidualMLP(d, num_layers)
        else:
            obs_expert = MLP(d, num_layers)
            
        for l_idx in range(num_layers):
            W_target = expert.layers[l_idx].weight.data.clone()
            F = fisher_dicts[t_idx][l_idx]
            
            epsilon = 0.05
            drift = drift_scale * torch.randn_like(W_target) / torch.sqrt(F + epsilon)
            obs_expert.layers[l_idx].weight.data.copy_(W_target + drift)
        observed_experts.append(obs_expert)
        
    # 5. Merging
    if is_residual:
        merged_models = {
            "pretrained": base_model,
            "ta": ResidualMLP(d, num_layers),
            "fwm": ResidualMLP(d, num_layers) # Fisher-Weighted Merging
        }
    else:
        merged_models = {
            "pretrained": base_model,
            "ta": MLP(d, num_layers),
            "fwm": MLP(d, num_layers)
        }
        
    for l_idx in range(num_layers):
        W0 = base_model.layers[l_idx].weight.data.clone()
        W_obs = [exp.layers[l_idx].weight.data.clone() for exp in observed_experts]
        
        # A. TA
        W_ta = sum(W_obs) / num_tasks
        merged_models["ta"].layers[l_idx].weight.data.copy_(W_ta)
        
        # B. Fisher-Weighted Merging (FWM)
        F_sum = sum([fisher_dicts[t_idx][l_idx] for t_idx in range(num_tasks)]) + 1e-6
        W_fwm = torch.zeros_like(W0)
        for t_idx, W_exp in enumerate(W_obs):
            F = fisher_dicts[t_idx][l_idx]
            W_fwm += W_exp * F
        W_fwm = W_fwm / F_sum
        merged_models["fwm"].layers[l_idx].weight.data.copy_(W_fwm)
        
    # 6. Evaluate
    N_test = 500
    X_test = torch.randn(N_test, d)
    
    losses = {key: [] for key in merged_models.keys()}
    for t_idx, expert in enumerate(experts):
        with torch.no_grad():
            Y_expert = expert(X_test)
            for key, model in merged_models.items():
                Y_pred = model(X_test)
                loss = 0.5 * torch.mean(torch.sum((Y_pred - Y_expert)**2, dim=1)).item()
                losses[key].append(loss)
                
    return {key: float(np.mean(val)) for key, val in losses.items()}

print("--- Testing FWM on Standard MLP ---")
print(run_deep_fw_experiment(42, num_layers=4, rotation_scale=0.08, drift_scale=0.02, is_residual=False))

print("--- Testing FWM on Residual MLP ---")
print(run_deep_fw_experiment(42, num_layers=6, rotation_scale=0.15, drift_scale=0.04, is_residual=True))
