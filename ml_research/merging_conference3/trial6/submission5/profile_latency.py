import torch
import time
import numpy as np

def run_profile():
    torch.manual_seed(42)
    L = 14
    D_in = 192
    D_out = 192
    K = 4
    
    # Pre-allocate inputs and parameters
    W_base = torch.randn(D_out, D_in) * 0.01
    b_base = torch.zeros(D_out)
    
    # Task vectors (expert differences)
    V_weights = [torch.randn(D_out, D_in) * 0.01 for _ in range(K)]
    V_biases = [torch.zeros(D_out) for _ in range(K)]
    V_weights_stacked = torch.stack(V_weights, dim=0) # [K, D_out, D_in]
    V_biases_stacked = torch.stack(V_biases, dim=0)   # [K, D_out]
    
    batch_sizes = [1, 8, 32, 128, 512]
    num_runs = 50
    
    print(f"{'Batch Size (B)':<15} | {'Static Uniform Latency (ms)':<30} | {'Dynamic Assembly Latency (ms)':<30} | {'Latency Slowdown':<18}")
    print("-" * 100)
    
    results = {}
    
    for B in batch_sizes:
        # Generate random inputs and routing coefficients
        X = torch.randn(B, D_in)
        alpha_b_k = torch.softmax(torch.randn(B, K), dim=1) # [B, K]
        alpha_avg = alpha_b_k.mean(dim=0) # [K]
        
        # 1. Profile Static Uniform Merging
        # Warmup
        for _ in range(5):
            # Merge once per layer
            for l in range(L):
                W_merged = W_base + sum(alpha_avg[k] * V_weights[k] for k in range(K))
                b_merged = b_base + sum(alpha_avg[k] * V_biases[k] for k in range(K))
                Y = X @ W_merged.t() + b_merged
                
        static_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            # For each layer l, merge once, then project
            for l in range(L):
                W_merged = W_base + sum(alpha_avg[k] * V_weights[k] for k in range(K))
                b_merged = b_base + sum(alpha_avg[k] * V_biases[k] for k in range(K))
                Y = X @ W_merged.t() + b_merged
            static_times.append((time.perf_counter() - t_start) * 1000) # ms
            
        static_mean = np.mean(static_times)
        static_std = np.std(static_times)
        
        # 2. Profile Dynamic Sample-wise Assembly
        # Warmup
        for _ in range(5):
            for l in range(L):
                W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
                b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
                Y = torch.einsum('bd,bod->bo', X, W_merged) + b_merged
                
        dynamic_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            for l in range(L):
                W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
                b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
                Y = torch.einsum('bd,bod->bo', X, W_merged) + b_merged
            dynamic_times.append((time.perf_counter() - t_start) * 1000) # ms
            
        dynamic_mean = np.mean(dynamic_times)
        dynamic_std = np.std(dynamic_times)
        
        slowdown = dynamic_mean / static_mean
        print(f"{B:<15} | {static_mean:>.4f} +- {static_std:>.4f} ms      | {dynamic_mean:>.4f} +- {dynamic_std:>.4f} ms       | {slowdown:>.2f}x")
        
        results[B] = {
            "static_mean": static_mean,
            "static_std": static_std,
            "dynamic_mean": dynamic_mean,
            "dynamic_std": dynamic_std,
            "slowdown": slowdown
        }
        
if __name__ == "__main__":
    run_profile()
