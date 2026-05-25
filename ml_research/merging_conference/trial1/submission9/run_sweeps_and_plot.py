import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os

# Set style for publication-quality plots
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "savefig.bbox": "tight",
    "savefig.dpi": 300
})

class MLP(nn.Module):
    def __init__(self, d, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

def generate_random_orthogonal_matrix(d):
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    dists = torch.diag(R)
    ph = dists / torch.abs(dists)
    Q = Q * ph
    return Q

def solve_standard_procrustes(W_target, W0):
    M = torch.matmul(W_target, W0.T)
    U, S, Vh = torch.linalg.svd(M)
    R = torch.matmul(U, Vh)
    return R

def solve_weighted_procrustes(W_target, W0, F, lr=0.03, num_iters=250):
    d = W0.shape[0]
    R = solve_standard_procrustes(W_target, W0).clone()
    for i in range(num_iters):
        diff = W_target - torch.matmul(R, W0)
        grad_E = - torch.matmul(diff * F, W0.T)
        grad_R = 0.5 * (grad_E - torch.matmul(torch.matmul(R, grad_E.T), R))
        
        # Check for NaN/Inf in gradient
        if torch.isnan(grad_R).any() or torch.isinf(grad_R).any():
            break
            
        grad_norm = torch.linalg.matrix_norm(grad_R, 'fro')
        if grad_norm > 1.0 or torch.isnan(grad_norm):
            grad_R = grad_R / (grad_norm + 1e-8)
            
        R_next = R - lr * grad_R
        
        try:
            U, S, Vh = torch.linalg.svd(R_next)
            R = torch.matmul(U, Vh)
        except RuntimeError:
            # Gracefully handle SVD convergence failure and return last valid R
            break
            
    return R

def cayley_transform(R):
    d = R.shape[0]
    I = torch.eye(d)
    inv_part = torch.linalg.inv(R + I + 1e-6 * I)
    Q = torch.matmul(R - I, inv_part)
    Q = 0.5 * (Q - Q.T)
    return Q

def inv_cayley_transform(Q):
    d = Q.shape[0]
    I = torch.eye(d)
    inv_part = torch.linalg.inv(I - Q)
    R = torch.matmul(I + Q, inv_part)
    return R

def run_deep_experiment(seed, d=64, num_layers=5, num_tasks=5, rotation_scale=0.12, drift_scale=0.05):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    base_model = MLP(d, num_layers)
    for layer in base_model.layers:
        with torch.no_grad():
            layer.weight.copy_(generate_random_orthogonal_matrix(d))
            
    experts = []
    for t_idx in range(num_tasks):
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
        
    observed_experts = []
    for t_idx, expert in enumerate(experts):
        obs_expert = MLP(d, num_layers)
        for l_idx in range(num_layers):
            W_target = expert.layers[l_idx].weight.data.clone()
            F = fisher_dicts[t_idx][l_idx]
            epsilon = 0.05
            drift = drift_scale * torch.randn_like(W_target) / torch.sqrt(F + epsilon)
            obs_expert.layers[l_idx].weight.data.copy_(W_target + drift)
        observed_experts.append(obs_expert)
        
    merged_models = {
        "pretrained": base_model,
        "ta": MLP(d, num_layers),
        "ties": MLP(d, num_layers),
        "orthomerge": MLP(d, num_layers),
        "fomm": MLP(d, num_layers),
        "fwm": MLP(d, num_layers)
    }
    
    for l_idx in range(num_layers):
        W0 = base_model.layers[l_idx].weight.data.clone()
        W_obs = [exp.layers[l_idx].weight.data.clone() for exp in observed_experts]
        
        # TA
        W_ta = sum(W_obs) / num_tasks
        merged_models["ta"].layers[l_idx].weight.data.copy_(W_ta)
        
        # Ties
        task_vectors = [W_exp - W0 for W_exp in W_obs]
        pruned_vectors = []
        for tv in task_vectors:
            flat_tv = tv.flatten()
            threshold = torch.quantile(torch.abs(flat_tv), 0.8)
            mask = torch.abs(tv) >= threshold
            pruned_vectors.append(tv * mask)
        W_ties = W0 + sum(pruned_vectors) / num_tasks
        merged_models["ties"].layers[l_idx].weight.data.copy_(W_ties)
        
        # OrthoMerge
        R_list_om = []
        rho_list_om = []
        for W_exp in W_obs:
            R = solve_standard_procrustes(W_exp, W0)
            R_list_om.append(R)
            rho_list_om.append(W_exp - torch.matmul(R, W0))
            
        Q_list_om = [cayley_transform(R) for R in R_list_om]
        Q_sum_om = sum(Q_list_om)
        norms_om = sum([torch.linalg.matrix_norm(Q, 'fro') for Q in Q_list_om])
        sum_norm_om = torch.linalg.matrix_norm(Q_sum_om, 'fro')
        scale_om = norms_om / (sum_norm_om + 1e-6)
        Q_merged_om = scale_om * (Q_sum_om / num_tasks)
        R_merged_om = inv_cayley_transform(Q_merged_om)
        rho_merged_om = sum(rho_list_om) / num_tasks
        W_om = torch.matmul(R_merged_om, W0) + rho_merged_om
        merged_models["orthomerge"].layers[l_idx].weight.data.copy_(W_om)
        
        # FOMM
        R_list_fomm = []
        rho_list_fomm = []
        for t_idx, W_exp in enumerate(W_obs):
            F = fisher_dicts[t_idx][l_idx]
            R = solve_weighted_procrustes(W_exp, W0, F, lr=0.03, num_iters=250)
            R_list_fomm.append(R)
            rho_list_fomm.append(W_exp - torch.matmul(R, W0))
            
        Q_list_fomm = [cayley_transform(R) for R in R_list_fomm]
        task_weights = [fisher_dicts[t_idx][l_idx].sum().item() for t_idx in range(num_tasks)]
        total_weight = sum(task_weights) + 1e-6
        task_weights = [w / total_weight for w in task_weights]
        
        Q_weighted_sum_fomm = torch.zeros_like(Q_list_fomm[0])
        for t_idx, Q in enumerate(Q_list_fomm):
            Q_weighted_sum_fomm += task_weights[t_idx] * Q
            
        norms_fomm = sum([task_weights[t_idx] * torch.linalg.matrix_norm(Q, 'fro') for t_idx, Q in enumerate(Q_list_fomm)])
        sum_norm_fomm = torch.linalg.matrix_norm(Q_weighted_sum_fomm, 'fro')
        scale_fomm = norms_fomm / (sum_norm_fomm + 1e-6)
        Q_merged_fomm = scale_fomm * Q_weighted_sum_fomm
        R_merged_fomm = inv_cayley_transform(Q_merged_fomm)
        
        F_sum = sum([fisher_dicts[t_idx][l_idx] for t_idx in range(num_tasks)]) + 1e-6
        rho_weighted = torch.zeros_like(rho_list_fomm[0])
        for t_idx, rho in enumerate(rho_list_fomm):
            F = fisher_dicts[t_idx][l_idx]
            rho_weighted += rho * F
        rho_merged_fomm = rho_weighted / F_sum
        
        W_fomm = torch.matmul(R_merged_fomm, W0) + rho_merged_fomm
        merged_models["fomm"].layers[l_idx].weight.data.copy_(W_fomm)
        
        # FWM
        F_sum_fwm = sum([fisher_dicts[t_idx][l_idx] for t_idx in range(num_tasks)]) + 1e-6
        W_fwm = torch.zeros_like(W0)
        for t_idx, W_exp in enumerate(W_obs):
            F = fisher_dicts[t_idx][l_idx]
            W_fwm += W_exp * F
        W_fwm = W_fwm / F_sum_fwm
        merged_models["fwm"].layers[l_idx].weight.data.copy_(W_fwm)

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

if __name__ == "__main__":
    seeds = [42, 100, 2026, 7, 999]
    
    # ------------------ SWEEP 1: DRIFT SCALE ------------------
    print("--- Running Sweep over Parameter Drift Scale ---")
    drifts = [0.0, 0.02, 0.05, 0.08, 0.12, 0.15]
    drift_results = {key: [] for key in ["pretrained", "ta", "ties", "orthomerge", "fomm", "fwm"]}
    
    for drift in drifts:
        print(f"Drift scale = {drift}")
        temp_losses = {key: [] for key in drift_results.keys()}
        for seed in seeds:
            losses = run_deep_experiment(seed, drift_scale=drift, rotation_scale=0.12)
            for key in temp_losses.keys():
                temp_losses[key].append(losses[key])
        for key in drift_results.keys():
            drift_results[key].append(np.mean(temp_losses[key]))
            
    # Plot Sweep 1
    plt.figure(figsize=(6.5, 4.5))
    colors = {
        "pretrained": "#7f7f7f",
        "ta": "#1f77b4",
        "ties": "#bcbd22",
        "orthomerge": "#d62728",
        "fomm": "#ff7f0e",
        "fwm": "#2ca02c"
    }
    labels = {
        "pretrained": "Pretrained Base",
        "ta": "Task Arithmetic",
        "ties": "Ties-Merging",
        "orthomerge": "OrthoMerge (standard)",
        "fomm": "FOMM (Ours)",
        "fwm": "FWM / FW-TA (Ours)"
    }
    markers = {
        "pretrained": "o",
        "ta": "s",
        "ties": "^",
        "orthomerge": "x",
        "fomm": "d",
        "fwm": "*"
    }
    
    for key in drift_results.keys():
        plt.plot(drifts, drift_results[key], label=labels[key], color=colors[key], 
                 marker=markers[key], linewidth=1.75, markersize=6)
        
    plt.xlabel("Parameter Drift Scale (Noise)")
    plt.ylabel("Functional Test MSE Loss (lower is better)")
    plt.title("Model Merging Sensitivity to Parameter Drift (5-Layer MLP)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("loss_vs_drift.pdf")
    plt.close()
    
    # ------------------ SWEEP 2: ROTATION SCALE ------------------
    print("--- Running Sweep over Rotation Scale ---")
    rotations = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    rot_results = {key: [] for key in ["pretrained", "ta", "ties", "orthomerge", "fomm", "fwm"]}
    
    for rot in rotations:
        print(f"Rotation scale = {rot}")
        temp_losses = {key: [] for key in rot_results.keys()}
        for seed in seeds:
            losses = run_deep_experiment(seed, drift_scale=0.05, rotation_scale=rot)
            for key in temp_losses.keys():
                temp_losses[key].append(losses[key])
        for key in rot_results.keys():
            rot_results[key].append(np.mean(temp_losses[key]))
            
    # Plot Sweep 2
    plt.figure(figsize=(6.5, 4.5))
    for key in rot_results.keys():
        plt.plot(rotations, rot_results[key], label=labels[key], color=colors[key], 
                 marker=markers[key], linewidth=1.75, markersize=6)
        
    plt.xlabel("Rotation Scale (Manifold Misalignment)")
    plt.ylabel("Functional Test MSE Loss (lower is better)")
    plt.title("Model Merging Sensitivity to Rotation Scale (5-Layer MLP)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("loss_vs_rotation.pdf")
    plt.close()
    
    print("Plots generated successfully!")
