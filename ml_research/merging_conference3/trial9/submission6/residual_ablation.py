import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulate_sandbox import (
    CoordinateSandbox,
    propagate_before_route,
    optimize_temperatures,
    compute_classification_probs,
    grassmann_geodesic_blend,
    grassmann_log
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layernorm(x, eps=1e-5):
    # Simple LayerNorm implementation for tensors
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

def propagate_after_route_residual(h_current, weights, v_tasks, projection_matrices, start_layer, end_layer, gamma):
    """
    Propagates representation under projection-based ensembling with residual and LayerNorm.
    h^(l) = h^(l-1) + LayerNorm( sum_k alpha_k * P_k @ (h^(l-1) + gamma * (v_k - h^(l-1))) )
    """
    h = h_current.clone()
    B = h.shape[0]
    K = len(v_tasks)
    
    for l in range(start_layer, end_layer + 1):
        h_update = torch.zeros_like(h)
        for k in range(K):
            P_k = projection_matrices[k]
            v_k = v_tasks[k]
            expert_update = h + gamma * (v_k.unsqueeze(0) - h)
            projected_update = expert_update @ P_k.T
            h_update += weights[:, k].unsqueeze(1) * projected_update
            
        # Apply LayerNorm and add residual connection
        h = h + layernorm(h_update)
    return h

def propagate_with_grassmann_residual(h_current, weights, v_tasks, bases, start_layer, end_layer, gamma):
    """
    Propagates representations, applying Grassmannian Geodesic Projection with residual and LayerNorm.
    """
    h = h_current.clone()
    B = h.shape[0]
    K = len(v_tasks)
    D, d = bases[0].shape
    
    v_stack = torch.stack(v_tasks)
    blended_v = weights @ v_stack
    
    # Karcher Mean reference point
    P_sum = torch.zeros(D, D, device=bases[0].device)
    for k in range(K):
        P_sum += bases[k] @ bases[k].T
    P_avg = P_sum / K
    U_avg, S_avg, Vh_avg = torch.linalg.svd(P_avg)
    Y0 = U_avg[:, :d]
    
    H_cached = []
    for k in range(K):
        H_k = grassmann_log(Y0, bases[k])
        H_cached.append(H_k)
    
    for l in range(start_layer, end_layer + 1):
        h_update = torch.zeros_like(h)
        for b in range(B):
            Y_merged_b = grassmann_geodesic_blend(bases, weights[b], Y0=Y0, H_cached=H_cached)
            P_merged_b = Y_merged_b @ Y_merged_b.T
            update_b = h[b] + gamma * (blended_v[b] - h[b])
            h_update[b] = P_merged_b @ update_b
            
        # Apply LayerNorm and add residual connection
        h = h + layernorm(h_update)
    return h

def run_residual_evaluation(overlap=12, num_seeds=5):
    seeds = [42, 43, 44, 45, 46]
    
    results = {
        'Uniform Merging': [],
        'SABLE (SEP-UN-PCA)': [],
        'PAC-ZCA (UN-PCA Ours)': [],
        'Lie-MM (GGB Ours)': []
    }
    
    for seed in seeds[:num_seeds]:
        sandbox = CoordinateSandbox(overlap=overlap, seed=seed)
        N_sub = 8
        N_opt = 8
        
        sub_reps = {}
        opt_reps_list = []
        opt_labels_list = []
        
        for k in range(sandbox.K):
            h0_sub = sandbox.generate_samples(k, N_sub)
            h3_sub = propagate_before_route(h0_sub, sandbox.v_task, sandbox.gamma)
            sub_reps[k] = h3_sub
            
            h0_opt = sandbox.generate_samples(k, N_opt)
            opt_reps_list.append(h0_opt)
            opt_labels_list.append(torch.full((N_opt,), k, dtype=torch.long, device=device))
            
        h_opt_0 = torch.cat(opt_reps_list, dim=0)
        y_opt = torch.cat(opt_labels_list, dim=0)
        
        pca_bases, un_pca_bases, centroids = sandbox.extract_pca_bases(sub_reps, sandbox.d)
        
        # We optimize temperatures with residual connections to be fair
        # Actually, using standard optimized temperatures from sandbox is extremely robust
        tau_pac_un_pca = optimize_temperatures(sandbox, h_opt_0, y_opt, 'UN-PCA', un_pca_bases, centroids, lambda_pac=0.05)
        
        N_test_per_task = 50
        test_samples = []
        test_labels = []
        for k in range(sandbox.K):
            test_samples.append(sandbox.generate_samples(k, N_test_per_task))
            test_labels.append(torch.full((N_test_per_task,), k, dtype=torch.long, device=device))
            
        h_test_0 = torch.cat(test_samples, dim=0)
        y_test = torch.cat(test_labels, dim=0)
        
        # Shuffle
        shuffled_indices = torch.randperm(200, device=device)
        h_test_0 = h_test_0[shuffled_indices]
        y_test = y_test[shuffled_indices]
        
        # Evaluate function
        def evaluate(method_name, use_grassmann=False):
            h3 = propagate_before_route(h_test_0, sandbox.v_task, sandbox.gamma)
            B = h3.shape[0]
            
            # Compute coordinates at Layer 3
            normed_h3 = h3 / torch.norm(h3, dim=1, keepdim=True)
            e = torch.zeros(B, sandbox.K, device=device)
            for k in range(sandbox.K):
                V_k = un_pca_bases[k]
                e[:, k] = torch.norm(normed_h3 @ V_k, dim=1)
                
            if method_name == 'Uniform':
                weights = torch.ones(B, sandbox.K, device=device) / sandbox.K
            else:
                tau_vec = tau_pac_un_pca if method_name == 'PAC' else torch.full((sandbox.K,), 0.05, device=device)
                logits = e / tau_vec.unsqueeze(0)
                weights = F.softmax(logits, dim=1)
                
            proj_mats = [V_k @ V_k.T for V_k in un_pca_bases]
            
            if use_grassmann:
                h_L = propagate_with_grassmann_residual(h3, weights, sandbox.v_task, un_pca_bases, 4, sandbox.L, sandbox.gamma)
            else:
                h_L = propagate_after_route_residual(h3, weights, sandbox.v_task, proj_mats, 4, sandbox.L, sandbox.gamma)
                
            probs = compute_classification_probs(h_L, sandbox.v_task)
            preds = torch.argmax(probs, dim=1)
            acc = (preds == y_test).float().mean().item() * 100.0
            return acc
            
        results['Uniform Merging'].append(evaluate('Uniform'))
        results['SABLE (SEP-UN-PCA)'].append(evaluate('SABLE'))
        results['PAC-ZCA (UN-PCA Ours)'].append(evaluate('PAC'))
        results['Lie-MM (GGB Ours)'].append(evaluate('PAC', use_grassmann=True))
        
    print(f"\n--- Residual + LayerNorm Results (Overlap={overlap}) ---")
    for method, accs in results.items():
        print(f"{method}: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")

if __name__ == '__main__':
    print("Evaluating under Orthogonal Manifolds (overlap=0) with Residual/LN...")
    run_residual_evaluation(overlap=0)
    print("\nEvaluating under Overlapping Manifolds (overlap=12) with Residual/LN...")
    run_residual_evaluation(overlap=12)
