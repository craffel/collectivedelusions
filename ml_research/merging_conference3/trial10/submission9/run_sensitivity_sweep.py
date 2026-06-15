import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from run_experiments import (
    AIR, get_task_signatures, extract_pca_bases, generate_stream, set_seed, evaluate_router
)

# We modify the training function to accept lambda_smooth
def train_router_with_lambda(model, cal_h3, cal_target_y, V_bases, lambda_smooth, epochs=200, lr=0.01):
    T_cal, B, D = cal_h3.shape
    K = len(V_bases)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    e = torch.zeros(T_cal, B, K)
    for t in range(T_cal):
        z_t = cal_h3[t]
        z_norm = z_t / (torch.norm(z_t, p=2, dim=1, keepdim=True) + 1e-8)
        for k in range(K):
            proj = z_norm @ V_bases[k]
            e[t, :, k] = torch.norm(proj, p=2, dim=1)
            
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0.0
        
        if isinstance(model, AIR):
            model.reset(e[0])
            ce_loss = 0.0
            smoothness_loss = 0.0
            spectral_penalty = torch.tensor(0.0, device=e[0].device)
            
            prev_alpha = None
            for t in range(1, T_cal):
                logits_t = model(e[t], return_logits=True)
                ce_loss += F.cross_entropy(logits_t, cal_target_y[t])
                
                # Sequential Jitter Regularization during calibration
                alpha_t = F.softmax(logits_t, dim=1)
                if prev_alpha is not None:
                    smoothness_loss += torch.sum((alpha_t - prev_alpha) ** 2, dim=1).mean()
                prev_alpha = alpha_t
                
            ce_loss = ce_loss / (T_cal - 1)
            smoothness_loss = smoothness_loss / (T_cal - 2)
            
            # Total multi-objective loss balancing accuracy, smoothness, and convergence stability
            loss = ce_loss + lambda_smooth * smoothness_loss + 0.1 * spectral_penalty
            
        loss.backward()
        optimizer.step()

def main():
    seeds = [42, 43, 44, 45, 46]
    sigmas = [0.05, 0.15, 0.40, 1.20]
    K = 4
    T_cal = 32
    T_test = 200
    B = 16
    cfg = "orthogonal"
    
    lambdas = [0.0, 0.01, 0.05, 0.20, 1.00]
    
    print("Starting Smoothness Regularization Lambda Sweep on Orthogonal Manifolds...")
    print(f"Lambdas to evaluate: {lambdas}")
    
    results = {}
    for l_val in lambdas:
        results[l_val] = {
            'hom_acc': [],
            'hom_jit': [],
            'het_acc': [],
            'het_jit': []
        }
        
    for l_val in lambdas:
        print(f"Evaluating lambda_smooth = {l_val}...")
        for seed in seeds:
            set_seed(seed)
            v_signatures = get_task_signatures(cfg)
            V_bases = extract_pca_bases(v_signatures, sigmas, config=cfg)
            
            # Generate calibration and test streams
            cal_h3, cal_target_y = generate_stream(v_signatures, sigmas, stream_type="homogeneous", T=T_cal, B=B, config=cfg)
            
            hom_test_h3, hom_test_target_y = generate_stream(v_signatures, sigmas, stream_type="homogeneous", T=T_test, B=B, config=cfg)
            het_test_h3, het_test_target_y = generate_stream(v_signatures, sigmas, stream_type="heterogeneous", T=T_test, B=B, config=cfg)
            
            # Instantiate and train AIR model
            model = AIR(K, N_steps=5, eta_test=0.1)
            train_router_with_lambda(model, cal_h3, cal_target_y, V_bases, lambda_smooth=l_val, epochs=200, lr=0.01)
            
            # Evaluate on Homogeneous
            _, hom_align_acc, hom_jitter, _ = evaluate_router(model, hom_test_h3, hom_test_target_y, v_signatures, V_bases)
            results[l_val]['hom_acc'].append(hom_align_acc * 100.0)
            results[l_val]['hom_jit'].append(hom_jitter)
            
            # Evaluate on Heterogeneous
            _, het_align_acc, het_jitter, _ = evaluate_router(model, het_test_h3, het_test_target_y, v_signatures, V_bases)
            results[l_val]['het_acc'].append(het_align_acc * 100.0)
            results[l_val]['het_jit'].append(het_jitter)
            
    print("\n--- Sweep Results Summary (Averaged over 5 seeds) ---")
    print("| Lambda_smooth | Homogeneous Acc (%) | Homogeneous Jitter | Heterogeneous Acc (%) | Heterogeneous Jitter |")
    print("|---------------|----------------------|--------------------|------------------------|----------------------|")
    for l_val in lambdas:
        hom_acc_mean, hom_acc_std = np.mean(results[l_val]['hom_acc']), np.std(results[l_val]['hom_acc'])
        hom_jit_mean, hom_jit_std = np.mean(results[l_val]['hom_jit']), np.std(results[l_val]['hom_jit'])
        het_acc_mean, het_acc_std = np.mean(results[l_val]['het_acc']), np.std(results[l_val]['het_acc'])
        het_jit_mean, het_jit_std = np.mean(results[l_val]['het_jit']), np.std(results[l_val]['het_jit'])
        print(f"| {l_val:<13} | {hom_acc_mean:.2f}% ± {hom_acc_std:.2f}% | {hom_jit_mean:.4f} ± {hom_jit_std:.4f} | {het_acc_mean:.2f}% ± {het_acc_std:.2f}% | {het_jit_mean:.4f} ± {het_jit_std:.4f} |")

if __name__ == "__main__":
    main()
