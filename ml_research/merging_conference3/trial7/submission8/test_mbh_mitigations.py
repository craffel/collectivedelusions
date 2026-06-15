import torch
import torch.nn as nn
import numpy as np
import os
from simulate import generate_expert_heads, generate_data, set_seed, train_router, compute_cghr_coefficients, compute_pfsr_coefficients

D = 192
K = 4
d = D // K
C = 10
optimal_metric = "max"
optimal_th = 0.85

set_seed(42)
W = generate_expert_heads()
train_z, train_tasks, train_classes = generate_data(W, 16, [42]) # N=64
test_z, test_tasks, test_classes = generate_data(W, 250, [42]) # 1000 test samples

router = train_router(train_z, train_tasks, wd=0.1, epochs=150, lr=0.01)

# Helper to compute confidence
def get_confidence(z_b, router):
    with torch.no_grad():
        logits = router(z_b)
        alpha = torch.softmax(logits, dim=1)
    return torch.max(alpha, dim=1).values

# 1. Soft-Confidence Fallback Homogenization
def evaluate_soft_confidence_fallback(test_z, test_tasks, test_classes, W, router, error_rate=0.0, beta=0.5, gamma_fallback=0.60):
    correct = 0
    total = len(test_z)
    batch_size = 256
    
    # Shuffle for heterogeneous stream
    indices = torch.randperm(total)
    shuffled_z = test_z[indices]
    shuffled_tasks = test_tasks[indices]
    shuffled_classes = test_classes[indices]
    
    batches = []
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batches.append((shuffled_z[i:end_idx], shuffled_tasks[i:end_idx], shuffled_classes[i:end_idx]))
        
    for batch_z, batch_tasks, batch_classes in batches:
        B_curr = batch_z.shape[0]
        
        # Compute hybrid routing coefficients
        alpha, u_prime = compute_cghr_coefficients(batch_z, W, router, optimal_metric, optimal_th)
        
        # Get parametric router confidence for fallback partitioning
        with torch.no_grad():
            logits_param = router(batch_z)
            alpha_param = torch.softmax(logits_param, dim=1)
        conf = torch.max(alpha_param, dim=1).values
        
        # Standard predicted task from PFSR projection
        k_star = torch.argmax(u_prime, dim=1)
        
        # Corrupt predicted task with error_rate
        corrupted_k_star = k_star.clone()
        for b in range(B_curr):
            if np.random.rand() < error_rate:
                other_tasks = [t for t in range(K) if t != k_star[b].item()]
                corrupted_k_star[b] = np.random.choice(other_tasks)
                
        # Group into normal micro-batches or dedicated fallback batch
        fallback_mask = (conf < optimal_th) # If confidence is below gating threshold, we fall back
        # Let's say intermediate ambiguous or low confidence samples go to fallback micro-batch
        
        # Process fallback micro-batch
        if fallback_mask.any():
            fallback_z = batch_z[fallback_mask]
            fallback_tasks = batch_tasks[fallback_mask]
            fallback_classes = batch_classes[fallback_mask]
            fallback_alpha = alpha[fallback_mask]
            
            # Apply Equation 13: soft uniform blend
            alpha_hybrid_avg = torch.mean(fallback_alpha, dim=0)
            alpha_fallback_avg = beta * alpha_hybrid_avg + (1.0 - beta) * (1.0 / K)
            
            for b in range(fallback_z.shape[0]):
                c_b = fallback_classes[b].item()
                logits_c = torch.zeros(C)
                for k in range(K):
                    z_kb = fallback_z[b, k*d : (k+1)*d]
                    z_kb_norm = z_kb / torch.norm(z_kb)
                    expert_logits = torch.matmul(W[k], z_kb_norm)
                    logits_c += alpha_fallback_avg[k] * expert_logits
                pred = torch.argmax(logits_c).item()
                if pred == c_b:
                    correct += 1
                    
        # Process regular homogeneous micro-batches for high confidence samples
        normal_mask = ~fallback_mask
        if normal_mask.any():
            normal_z = batch_z[normal_mask]
            normal_tasks = batch_tasks[normal_mask]
            normal_classes = batch_classes[normal_mask]
            normal_alpha = alpha[normal_mask]
            normal_corrupted_k = corrupted_k_star[normal_mask]
            
            for g in range(K):
                g_mask = (normal_corrupted_k == g)
                if not g_mask.any():
                    continue
                micro_z = normal_z[g_mask]
                micro_classes = normal_classes[g_mask]
                micro_alpha = normal_alpha[g_mask]
                
                alpha_avg_g = torch.mean(micro_alpha, dim=0)
                
                for b in range(micro_z.shape[0]):
                    c_b = micro_classes[b].item()
                    logits_c = torch.zeros(C)
                    for k in range(K):
                        z_kb = micro_z[b, k*d : (k+1)*d]
                        z_kb_norm = z_kb / torch.norm(z_kb)
                        expert_logits = torch.matmul(W[k], z_kb_norm)
                        logits_c += alpha_avg_g[k] * expert_logits
                    pred = torch.argmax(logits_c).item()
                    if pred == c_b:
                        correct += 1
                        
    return (correct / total) * 100.0


# 2. Hierarchical MBH (H-MBH)
def evaluate_hierarchical_mbh(test_z, test_tasks, test_classes, W, router, error_rate=0.0):
    correct = 0
    total = len(test_z)
    batch_size = 256
    
    # Define groups: Group 0 is grayscale {0, 1}, Group 1 is color {2, 3}
    groups = {0: [0, 1], 1: [2, 3]}
    task_to_group = {0: 0, 1: 0, 2: 1, 3: 1}
    
    # Shuffle for heterogeneous stream
    indices = torch.randperm(total)
    shuffled_z = test_z[indices]
    shuffled_tasks = test_tasks[indices]
    shuffled_classes = test_classes[indices]
    
    batches = []
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batches.append((shuffled_z[i:end_idx], shuffled_tasks[i:end_idx], shuffled_classes[i:end_idx]))
        
    for batch_z, batch_tasks, batch_classes in batches:
        B_curr = batch_z.shape[0]
        
        # Compute hybrid routing coefficients
        alpha, u_prime = compute_cghr_coefficients(batch_z, W, router, optimal_metric, optimal_th)
        
        # Standard predicted task from PFSR projection
        k_star = torch.argmax(u_prime, dim=1)
        
        # Corrupt predicted task with error_rate
        corrupted_k_star = k_star.clone()
        for b in range(B_curr):
            if np.random.rand() < error_rate:
                other_tasks = [t for t in range(K) if t != k_star[b].item()]
                corrupted_k_star[b] = np.random.choice(other_tasks)
                
        # Group into cluster-level micro-batches
        for g_idx in [0, 1]:
            g_tasks = groups[g_idx]
            # Select samples whose corrupted predicted task belongs to group g_idx
            mask = torch.tensor([task_to_group[corrupted_k_star[b].item()] == g_idx for b in range(B_curr)], dtype=torch.bool)
            if not mask.any():
                continue
                
            micro_z = batch_z[mask]
            micro_classes = batch_classes[mask]
            micro_alpha = alpha[mask]
            
            # Apply Equation 15: cluster-homogenized average
            alpha_avg_g = torch.zeros(K)
            alpha_mean = torch.mean(micro_alpha, dim=0)
            for k in range(K):
                if k in g_tasks:
                    alpha_avg_g[k] = alpha_mean[k]
                else:
                    alpha_avg_g[k] = 0.0
                    
            # Normalize to sum to 1 if not zero (standard projection/merging)
            denom = torch.sum(alpha_avg_g)
            if denom > 0:
                alpha_avg_g = alpha_avg_g / denom
                
            for b in range(micro_z.shape[0]):
                c_b = micro_classes[b].item()
                logits_c = torch.zeros(C)
                for k in range(K):
                    z_kb = micro_z[b, k*d : (k+1)*d]
                    z_kb_norm = z_kb / torch.norm(z_kb)
                    expert_logits = torch.matmul(W[k], z_kb_norm)
                    logits_c += alpha_avg_g[k] * expert_logits
                pred = torch.argmax(logits_c).item()
                if pred == c_b:
                    correct += 1
                    
    return (correct / total) * 100.0


error_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75]
print("Evaluating Mitigations under Cascaded Error Propagation:")
print("| Error Rate (%) | Standard MBH (%) | Fallback (beta=0.5) (%) | Fallback (beta=0.0) (%) | Hierarchical MBH (%) |")
print("| :---: | :---: | :---: | :---: | :---: |")

for err in error_rates:
    std_accs = []
    fb_accs_05 = []
    fb_accs_00 = []
    h_accs = []
    
    # Average across 5 random trials for stability
    for seed in [101, 102, 103, 104, 105]:
        set_seed(seed)
        
        # 1. Standard MBH
        # We can implement a simplified run or import from test_mbh_error_propagation but let's just run it
        from test_mbh_error_propagation import evaluate_mbh_with_routing_errors
        std_acc = evaluate_mbh_with_routing_errors(test_z, test_tasks, test_classes, W, router, error_rate=err)
        std_accs.append(std_acc)
        
        # 2. Fallback (beta=0.5)
        fb_acc_05 = evaluate_soft_confidence_fallback(test_z, test_tasks, test_classes, W, router, error_rate=err, beta=0.5)
        fb_accs_05.append(fb_acc_05)
        
        # 3. Fallback (beta=0.0) (Graceful decay to uniform)
        fb_acc_00 = evaluate_soft_confidence_fallback(test_z, test_tasks, test_classes, W, router, error_rate=err, beta=0.0)
        fb_accs_00.append(fb_acc_00)
        
        # 4. Hierarchical MBH
        h_acc = evaluate_hierarchical_mbh(test_z, test_tasks, test_classes, W, router, error_rate=err)
        h_accs.append(h_acc)
        
    print(f"| {err*100:.1f}% | {np.mean(std_accs):.2f}% | {np.mean(fb_accs_05):.2f}% | {np.mean(fb_accs_00):.2f}% | {np.mean(h_accs):.2f}% |")
