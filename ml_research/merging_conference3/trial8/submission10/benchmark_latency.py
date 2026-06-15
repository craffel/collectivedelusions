import time
import numpy as np
import torch
import torch.nn as nn

def benchmark():
    # Fix seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    D = 192
    K_list = [2, 4, 8, 16]
    B_list = [1, 32, 256]
    
    # Warm up
    _ = torch.randn(100, 100) @ torch.randn(100, 100)
    
    results = {}
    
    for K in K_list:
        results[K] = {}
        for B in B_list:
            results[K][B] = {}
            
            # Create dummy tensors on CPU
            h_b = torch.randn(B, D)
            centroids = torch.randn(K, D)
            # Normalize centroids for cosine similarity
            centroids = centroids / torch.norm(centroids, dim=1, keepdim=True)
            
            # 1. Linear Router
            linear_layer = nn.Linear(D, K)
            # Warmup
            with torch.no_grad():
                _ = torch.softmax(linear_layer(h_b), dim=-1)
            # Measure
            start = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    logits = linear_layer(h_b)
                    _ = torch.softmax(logits, dim=-1)
            results[K][B]["Linear Router"] = ((time.perf_counter() - start) / 100) * 1000  # ms
            
            # 2. SABLE
            # Warmup
            h_b_norm = h_b / torch.norm(h_b, dim=1, keepdim=True)
            _ = torch.softmax(torch.matmul(h_b_norm, centroids.t()) / 0.05, dim=-1)
            # Measure
            start = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    h_b_norm = h_b / torch.norm(h_b, dim=1, keepdim=True)
                    u = torch.matmul(h_b_norm, centroids.t())
                    _ = torch.softmax(u / 0.05, dim=-1)
            results[K][B]["SABLE"] = ((time.perf_counter() - start) / 100) * 1000  # ms
            
            # 3. SPS-ZCA
            # Warmup
            h_b_norm = h_b / torch.norm(h_b, dim=1, keepdim=True)
            _ = torch.softmax(torch.matmul(h_b_norm, centroids.t()) / 0.001, dim=-1)
            # Measure
            start = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    h_b_norm = h_b / torch.norm(h_b, dim=1, keepdim=True)
                    u = torch.matmul(h_b_norm, centroids.t())
                    _ = torch.softmax(u / 0.001, dim=-1)
            results[K][B]["SPS-ZCA"] = ((time.perf_counter() - start) / 100) * 1000  # ms
            
            # 4. ESM-LVC (Ours)
            Gamma = torch.randn(K, K)
            # Warmup
            h_b_norm = h_b / torch.norm(h_b, dim=1, keepdim=True)
            u = torch.matmul(h_b_norm, centroids.t())
            alpha_t = torch.softmax(u / 0.03, dim=-1)
            
            # Compute Adaptive Step-Size for DESS to guarantee stability
            G = torch.sum(torch.clamp(Gamma, min=0.0) * (1.0 - torch.eye(K)), dim=1)
            max_G = torch.max(G)
            u_max = 1.0
            eta_stable = 0.9
            max_G_val = max_G.item()
            if max_G_val < 1.0:
                alpha_max = max(1.0, u_max / (1.0 - max_G_val))
                delta_tau_val = min(0.2, eta_stable / alpha_max)
            else:
                N_steps = 5
                alpha_max_t = 1.0
                for _ in range(N_steps):
                    alpha_max_t = (1.0 + max_G_val) * alpha_max_t + u_max
                delta_tau_val = min(0.2, eta_stable / alpha_max_t)
            delta_tau = torch.tensor(delta_tau_val)
            
            beta = 1.0
            for _ in range(5):
                # Batch matrix multiplication for interaction term: shape (B, K)
                # Gamma is (K, K), alpha_t is (B, K). We do alpha_t @ Gamma.t() -> shape (B, K)
                interaction = torch.matmul(alpha_t, Gamma.t())
                d_alpha = alpha_t * (u + interaction - beta * alpha_t)
                alpha_t = alpha_t + delta_tau * d_alpha
                alpha_t = torch.clamp(alpha_t, min=0.0)
            sum_alpha = torch.sum(alpha_t, dim=1, keepdim=True)
            _ = torch.where(sum_alpha > 0, alpha_t / sum_alpha, torch.full_like(alpha_t, 1.0 / K))
            
            # Measure
            start = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    # Cosine similarities
                    h_b_norm = h_b / torch.norm(h_b, dim=1, keepdim=True)
                    u = torch.matmul(h_b_norm, centroids.t())
                    
                    # Initial state
                    alpha_t = torch.softmax(u / 0.03, dim=-1)
                    
                    # Compute Adaptive Step-Size for DESS
                    G = torch.sum(torch.clamp(Gamma, min=0.0) * (1.0 - torch.eye(K)), dim=1)
                    max_G = torch.max(G)
                    u_max = 1.0
                    eta_stable = 0.9
                    max_G_val = max_G.item()
                    if max_G_val < 1.0:
                        alpha_max = max(1.0, u_max / (1.0 - max_G_val))
                        delta_tau_val = min(0.2, eta_stable / alpha_max)
                    else:
                        N_steps = 5
                        alpha_max_t = 1.0
                        for _ in range(N_steps):
                            alpha_max_t = (1.0 + max_G_val) * alpha_max_t + u_max
                        delta_tau_val = min(0.2, eta_stable / alpha_max_t)
                    delta_tau = torch.tensor(delta_tau_val)
                    
                    # Solve
                    for step in range(5):
                        interaction = torch.matmul(alpha_t, Gamma.t())
                        d_alpha = alpha_t * (u + interaction - beta * alpha_t)
                        alpha_t = alpha_t + delta_tau * d_alpha
                        alpha_t = torch.clamp(alpha_t, min=0.0)
                        
                    # Normalize
                    sum_alpha = torch.sum(alpha_t, dim=1, keepdim=True)
                    _ = torch.where(sum_alpha > 0, alpha_t / sum_alpha, torch.full_like(alpha_t, 1.0 / K))
            results[K][B]["ESM-LVC"] = ((time.perf_counter() - start) / 100) * 1000  # ms
            
    # Print LaTeX Table
    print("\\begin{table}[h]")
    print("\\caption{Routing execution latency benchmark (in milliseconds) on Intel Xeon CPU @ 2.30GHz.}")
    print("\\label{tab:latency_results}")
    print("\\vskip 0.15in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{sc}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("K & B & Linear Router & SABLE & SPS-ZCA & \\textbf{ESM-LVC (Ours)} \\\\")
    print("\\midrule")
    for K in K_list:
        for B in B_list:
            l_r = results[K][B]["Linear Router"]
            sable = results[K][B]["SABLE"]
            sps = results[K][B]["SPS-ZCA"]
            esm = results[K][B]["ESM-LVC"]
            print(f"{K} & {B} & {l_r:.4f} & {sable:.4f} & {sps:.4f} & \\textbf{{{esm:.4f}}} \\\\")
        if K != K_list[-1]:
            print("\\hline")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{sc}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\vskip -0.1in")
    print("\\end{table}")

if __name__ == "__main__":
    benchmark()
