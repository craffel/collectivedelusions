import torch
import torch.nn as nn
import numpy as np
import time
from simulate import generate_expert_heads, generate_data, set_seed, train_router, evaluate_model, compute_cghr_coefficients

D = 192
K = 4
d = D // K
C = 10
optimal_metric = "max"
optimal_th = 0.85

set_seed(42)
W = generate_expert_heads()
train_z, train_tasks, train_classes = generate_data(W, 16, [42]) # N=64 (16 per task)
test_z, test_tasks, test_classes = generate_data(W, 250, [42])

router = train_router(train_z, train_tasks, wd=0.1, epochs=150, lr=0.01)

# Continuous coefficients
alpha_continuous, _ = compute_cghr_coefficients(test_z, W, router, optimal_metric, optimal_th)

# Dynamic Weight Fusion latency (no cache)
print("Benchmarking Continuous Weight Fusion (no cache)...")
start = time.perf_counter()
for i in range(len(test_z)):
    # Continuous merge of weights
    W_fused = torch.zeros(C, d)
    for k in range(K):
        W_fused += alpha_continuous[i, k] * W[k]
end = time.perf_counter()
continuous_latency_ms = ((end - start) / len(test_z)) * 1000
print(f"Average Continuous Weight Fusion Latency: {continuous_latency_ms:.4f} ms")

# Discretized coefficients and caching simulation
def discretize_coefficients(alpha, step=0.1):
    # Round coefficients to nearest step, maintaining partition of unity
    alpha_discretized = torch.round(alpha / step) * step
    # Normalize to sum to 1.0 to ensure partition of unity
    sums = alpha_discretized.sum(dim=1, keepdim=True)
    sums[sums == 0] = 1.0
    alpha_discretized = alpha_discretized / sums
    return alpha_discretized

# Let's sweep discretization step sizes: 0.2, 0.1, 0.05, 0.01
steps = [0.2, 0.1, 0.05, 0.01]
print("\nEvaluating Accuracy and Latency with Fusion Weight Caching:")
print("| Discretization Step | Joint Mean Accuracy (%) | Fusion Latency (ms) | Speedup Factor |")
print("| :---: | :---: | :---: | :---: |")

# Base baseline (Continuous, no cache)
# Calculate baseline accuracy with continuous coefficients
correct_continuous = 0
for b in range(len(test_z)):
    alpha = alpha_continuous[b:b+1]
    logits_c = torch.zeros(C)
    for k in range(K):
        z_kb = test_z[b, k*d : (k+1)*d]
        z_kb_norm = z_kb / torch.norm(z_kb)
        expert_logits = torch.matmul(W[k], z_kb_norm)
        logits_c += alpha[0, k] * expert_logits
    if torch.argmax(logits_c).item() == test_classes[b].item():
        correct_continuous += 1
acc_continuous = (correct_continuous / len(test_z)) * 100.0
print(f"| Continuous (Baseline) | {acc_continuous:.2f}% | {continuous_latency_ms:.4f} ms | 1.00x |")

for step in steps:
    alpha_disc = discretize_coefficients(alpha_continuous, step=step)
    
    # 1. Measure Accuracy
    correct_disc = 0
    for b in range(len(test_z)):
        alpha = alpha_disc[b:b+1]
        logits_c = torch.zeros(C)
        for k in range(K):
            z_kb = test_z[b, k*d : (k+1)*d]
            z_kb_norm = z_kb / torch.norm(z_kb)
            expert_logits = torch.matmul(W[k], z_kb_norm)
            logits_c += alpha[0, k] * expert_logits
        if torch.argmax(logits_c).item() == test_classes[b].item():
            correct_disc += 1
    acc_disc = (correct_disc / len(test_z)) * 100.0
    
    # 2. Measure Latency with Caching
    # Pre-populate cache of fused weights
    weight_cache = {}
    hits = 0
    misses = 0
    start = time.perf_counter()
    for b in range(len(test_z)):
        # Key is tuple of rounded coefficients
        key = tuple(np.round(alpha_disc[b].numpy(), decimals=4))
        if key in weight_cache:
            W_fused = weight_cache[key]
            hits += 1
        else:
            W_fused = torch.zeros(C, d)
            for k in range(K):
                W_fused += alpha_disc[b, k] * W[k]
            weight_cache[key] = W_fused
            misses += 1
    end = time.perf_counter()
    cached_latency_ms = ((end - start) / len(test_z)) * 1000
    speedup = continuous_latency_ms / (cached_latency_ms + 1e-9)
    print(f"| {step:.2f} (Cache Hits: {hits}/{len(test_z)}) | {acc_disc:.2f}% | {cached_latency_ms:.4f} ms | {speedup:.2f}x |")
