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

# Evaluate under CGHR with MBH, but with simulated routing errors in MBH partitioning
def evaluate_mbh_with_routing_errors(test_z, test_tasks, test_classes, W, router, error_rate=0.0):
    correct = 0
    total = len(test_z)
    
    # Process in batches of size 256, heterogeneous stream (fully shuffled!)
    batch_size = 256
    
    # Shuffle for a true heterogeneous stream
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
        
        # Task prediction from PFSR (standard MBH task prediction)
        k_star = torch.argmax(u_prime, dim=1) # [B_curr]
        
        # Corrupt task prediction with error_rate
        corrupted_k_star = k_star.clone()
        for b in range(B_curr):
            if np.random.rand() < error_rate:
                # Pick a random different task
                other_tasks = [t for t in range(K) if t != k_star[b].item()]
                corrupted_k_star[b] = np.random.choice(other_tasks)
                
        # Partition the batch into homogeneous micro-batches using corrupted task predictions
        for g in range(K):
            mask = (corrupted_k_star == g)
            if not mask.any():
                continue
                
            micro_z = batch_z[mask]
            micro_tasks = batch_tasks[mask]
            micro_classes = batch_classes[mask]
            micro_alpha = alpha[mask]
            
            # Average coefficients across this micro-batch
            alpha_avg_g = torch.mean(micro_alpha, dim=0)
            
            for b in range(micro_z.shape[0]):
                t_b = micro_tasks[b].item()
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
print("Measuring Cascaded Error Propagation in Micro-Batch Homogenization (MBH):")
print("| Task-Classification Error Rate (%) | Downstream Classification Accuracy (%) | Performance Drop (%) |")
print("| :---: | :---: | :---: |")

# Accuracy with 0% error is the base (averaged over seeds for stability)
base_accs = []
for seed in [101, 102, 103, 104, 105]:
    set_seed(seed)
    base_accs.append(evaluate_mbh_with_routing_errors(test_z, test_tasks, test_classes, W, router, error_rate=0.0))
base_acc = np.mean(base_accs)
print(f"| 0.0% (No Error) | {base_acc:.2f}% | 0.00% |")

for err in error_rates[1:]:
    # Average across 5 random trials to get stable results
    accs = []
    for seed in [101, 102, 103, 104, 105]:
        set_seed(seed)
        acc = evaluate_mbh_with_routing_errors(test_z, test_tasks, test_classes, W, router, error_rate=err)
        accs.append(acc)
    mean_acc = np.mean(accs)
    drop = base_acc - mean_acc
    print(f"| {err*100:.1f}% | {mean_acc:.2f}% | {drop:.2f}% |")
