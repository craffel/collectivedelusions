import torch
import time
import numpy as np

def profile_solver():
    print("======================================================================")
    print("PROFILING BATCHED CHOLESKY-FACTORIZED ACTIVE INFERENCE ROUTING SOLVER")
    print("======================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    K_values = [4, 16, 64]
    B_values = [1, 4, 16, 64, 256]
    
    # Warmup
    print("Warming up...")
    dummy_H = torch.eye(16, device=device)
    dummy_b = torch.randn(16, 16, device=device)
    for _ in range(100):
        dummy_L = torch.linalg.cholesky(dummy_H)
        torch.linalg.solve_triangular(dummy_L, dummy_b, upper=False)
    
    results = {}
    
    for K in K_values:
        results[K] = {}
        # Create a positive-definite Hessian matrix
        A_mat = torch.randn(K, K, device=device)
        H_mat = torch.mm(A_mat, A_mat.t()) + torch.eye(K, device=device) * 0.1
        # Pre-compute Cholesky factor L
        L_mat = torch.linalg.cholesky(H_mat)
        
        for B in B_values:
            # Batched target vector b_t of shape B x K
            # Transposed to K x B for solve_triangular
            b_batched = torch.randn(K, B, device=device)
            
            # Number of trials
            num_trials = 1000
            
            # Synchronize before starting
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            for _ in range(num_trials):
                # Solve L * Y = b_batched (K x B)
                Y = torch.linalg.solve_triangular(L_mat, b_batched, upper=False)
                # Solve L^T * mu = Y (K x B)
                mu = torch.linalg.solve_triangular(L_mat.t(), Y, upper=True)
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_duration = end_time - start_time
            avg_batch_latency_us = (total_duration / num_trials) * 1e6
            avg_query_latency_us = avg_batch_latency_us / B
            qps = B * num_trials / total_duration
            
            results[K][B] = {
                "batch_latency": avg_batch_latency_us,
                "query_latency": avg_query_latency_us,
                "qps": qps
            }
            
            print(f"K={K:2d}, Batch Size={B:3d} | Batch Latency: {avg_batch_latency_us:6.2f} us | Latency/Query: {avg_query_latency_us:6.2f} us | QPS: {qps:9.1f}")
            
    # Format and save results as Markdown
    md_content = "# Batched Solver Latency and QPS Throughput Profiling\n\n"
    md_content += f"**Device Profiled:** {device}\n"
    if device.type == "cuda":
        md_content += f"- **GPU Model:** {torch.cuda.get_device_name(0)}\n"
    md_content += "\n## Throughput and Latency Matrix\n\n"
    
    for K in K_values:
        md_content += f"### Expert Registry Size $K = {K}$\n\n"
        md_content += "| Batch Size $B$ | Batch Latency ($\\mu$s) | Latency per Query ($\\mu$s) | Throughput (QPS) |\n"
        md_content += "|---|---|---|---|\n"
        for B in B_values:
            res = results[K][B]
            md_content += f"| {B:3d} | {res['batch_latency']:8.2f} | {res['query_latency']:18.2f} | {res['qps']:14.1f} |\n"
        md_content += "\n"
        
    with open("profile_results.md", "w") as f:
        f.write(md_content)
        
    print("\nProfiling Complete! 'profile_results.md' has been generated.")

if __name__ == "__main__":
    profile_solver()
