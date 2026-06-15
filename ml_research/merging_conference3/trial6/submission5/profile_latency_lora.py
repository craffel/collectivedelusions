import torch
import time
import numpy as np

def run_profile():
    torch.manual_seed(42)
    L = 14
    D_in = 192
    D_out = 192
    K = 4
    r = 8  # LoRA rank
    
    # Pre-allocate inputs and parameters
    W_base = torch.randn(D_out, D_in) * 0.01
    b_base = torch.zeros(D_out)
    
    # Task vectors (expert differences) for Full Parameter
    V_weights = [torch.randn(D_out, D_in) * 0.01 for _ in range(K)]
    V_biases = [torch.zeros(D_out) for _ in range(K)]
    V_weights_stacked = torch.stack(V_weights, dim=0) # [K, D_out, D_in]
    V_biases_stacked = torch.stack(V_biases, dim=0)   # [K, D_out]
    
    # LoRA adapters for each expert k
    A_adapters = [torch.randn(r, D_in) * 0.01 for _ in range(K)]
    B_adapters = [torch.randn(D_out, r) * 0.01 for _ in range(K)]
    A_stacked = torch.stack(A_adapters, dim=0) # [K, r, D_in]
    B_stacked = torch.stack(B_adapters, dim=0) # [K, D_out, r]
    
    batch_sizes = [1, 8, 32, 128, 512]
    num_runs = 50
    
    print(f"{'B':<5} | {'Static Uniform (ms)':<20} | {'Dynamic Full (ms)':<20} | {'Dynamic LoRA (ms)':<20} | {'Full/Static':<12} | {'LoRA/Static':<12}")
    print("-" * 105)
    
    for B in batch_sizes:
        X = torch.randn(B, D_in)
        alpha_b_k = torch.softmax(torch.randn(B, K), dim=1) # [B, K]
        alpha_avg = alpha_b_k.mean(dim=0) # [K]
        
        # 1. Profile Static Uniform Merging
        # Warmup
        for _ in range(5):
            for l in range(L):
                W_merged = W_base + sum(alpha_avg[k] * V_weights[k] for k in range(K))
                b_merged = b_base + sum(alpha_avg[k] * V_biases[k] for k in range(K))
                Y = X @ W_merged.t() + b_merged
                
        static_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            for l in range(L):
                W_merged = W_base + sum(alpha_avg[k] * V_weights[k] for k in range(K))
                b_merged = b_base + sum(alpha_avg[k] * V_biases[k] for k in range(K))
                Y = X @ W_merged.t() + b_merged
            static_times.append((time.perf_counter() - t_start) * 1000)
            
        static_mean = np.mean(static_times)
        static_std = np.std(static_times)
        
        # 2. Profile Dynamic Full-Parameter Assembly
        # Warmup
        for _ in range(5):
            for l in range(L):
                W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
                b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
                Y = torch.einsum('bd,bod->bo', X, W_merged) + b_merged
                
        dynamic_full_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            for l in range(L):
                W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
                b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
                Y = torch.einsum('bd,bod->bo', X, W_merged) + b_merged
            dynamic_full_times.append((time.perf_counter() - t_start) * 1000)
            
        full_mean = np.mean(dynamic_full_times)
        full_std = np.std(dynamic_full_times)
        
        # 3. Profile Dynamic LoRA Assembly
        # Warmup
        for _ in range(5):
            for l in range(L):
                Y_base = X @ W_base.t() + b_base
                H = torch.einsum('bd,krd->bkr', X, A_stacked)
                H_scaled = H * alpha_b_k.unsqueeze(-1)
                Y_lora = torch.einsum('bkr,kor->bo', H_scaled, B_stacked)
                Y = Y_base + Y_lora
                
        dynamic_lora_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            for l in range(L):
                Y_base = X @ W_base.t() + b_base
                H = torch.einsum('bd,krd->bkr', X, A_stacked)
                H_scaled = H * alpha_b_k.unsqueeze(-1)
                Y_lora = torch.einsum('bkr,kor->bo', H_scaled, B_stacked)
                Y = Y_base + Y_lora
            dynamic_lora_times.append((time.perf_counter() - t_start) * 1000)
            
        lora_mean = np.mean(dynamic_lora_times)
        lora_std = np.std(dynamic_lora_times)
        
        slowdown_full = full_mean / static_mean
        slowdown_lora = lora_mean / static_mean
        
        print(f"{B:<5} | {static_mean:>.3f} +- {static_std:>.3f} | {full_mean:>.3f} +- {full_std:>.3f} | {lora_mean:>.3f} +- {lora_std:>.3f} | {slowdown_full:>.2f}x | {slowdown_lora:>.2f}x")

if __name__ == "__main__":
    run_profile()
