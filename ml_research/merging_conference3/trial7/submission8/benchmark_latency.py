import torch
import torch.nn as nn
import time
import numpy as np

# Set dimensions
D = 192
K = 4
d = D // K
C = 10
B_sizes = [1, 8, 32, 128, 256, 512]

# Initialize dummy weights
W = [torch.randn(C, d) for _ in range(K)]
z_dummy = torch.randn(512, D)

# Define simple forward pass with weight fusion
def standard_forward(z, alpha_avg, W):
    B = z.shape[0]
    logits_c = torch.zeros(B, C)
    for b in range(B):
        for k in range(K):
            z_kb = z[b, k*d : (k+1)*d]
            z_kb_norm = z_kb / (torch.norm(z_kb) + 1e-8)
            expert_logits = torch.matmul(W[k], z_kb_norm)
            logits_c[b] += alpha_avg[k] * expert_logits
    return logits_c

# Measure dynamic weight fusion
def measure_weight_fusion():
    alpha = torch.rand(K)
    alpha = alpha / alpha.sum()
    
    # Simulate a single model merge (weight interpolation of classification heads)
    start_time = time.perf_counter()
    for _ in range(100):
        W_fused = torch.zeros(C, d)
        for k in range(K):
            W_fused += alpha[k] * W[k]
    end_time = time.perf_counter()
    avg_fusion_ms = ((end_time - start_time) / 100) * 1000
    return avg_fusion_ms

# Measure execution times across batch sizes
def profile_latency():
    print("Starting empirical systems latency profiling...")
    fusion_time_ms = measure_weight_fusion()
    print(f"Average dynamic weight fusion latency (W_fused): {fusion_time_ms:.4f} ms")
    
    results = []
    
    for B in B_sizes:
        z = torch.randn(B, D)
        alpha_avg = torch.full((K,), 1.0 / K)
        
        # 1. Profile Standard (No MBH)
        start = time.perf_counter()
        for _ in range(20):
            _ = standard_forward(z, alpha_avg, W)
        end = time.perf_counter()
        std_lat_ms = ((end - start) / 20) * 1000
        
        # 2. Profile MBH (sequential execution of G=4 micro-batches)
        # We partition the batch into G=4 equal groups of size B/4
        sub_B = max(1, B // K)
        start = time.perf_counter()
        for _ in range(20):
            # Run G=4 sequential inferences plus dynamic model fusion
            for g in range(K):
                z_sub = z[g * sub_B : (g+1) * sub_B]
                if len(z_sub) == 0:
                    continue
                _ = standard_forward(z_sub, alpha_avg, W)
        end = time.perf_counter()
        # Total latency is sequential passes plus fusion overheads for G=4 micro-batches
        mbh_lat_ms = (((end - start) / 20) * 1000) + (K * fusion_time_ms)
        
        results.append((B, std_lat_ms, mbh_lat_ms, mbh_lat_ms / (std_lat_ms + 1e-9)))
        
    print("\n--- Latency Profiling Results ---")
    print("| Batch Size B | Standard (No MBH) (ms) | MBH (G=4) (ms) | Overhead Multiplier |")
    print("| :---: | :---: | :---: | :---: |")
    for B, std, mbh, mult in results:
        print(f"| {B} | {std:.4f} | {mbh:.4f} | {mult:.2f}x |")
        
if __name__ == "__main__":
    profile_latency()
