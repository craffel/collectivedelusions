import torch
import time

def run_scaling_benchmark():
    dim_in, dim_out = 4096, 4096
    r = 8
    num_experts = 4
    
    # We will test batch sizes B and active tasks G
    batch_sizes = [16, 64, 256]
    active_tasks = [1, 2, 3, 4]
    
    # Set up random matrices
    W_base = torch.randn(dim_out, dim_in, dtype=torch.float32)
    lora_As = [torch.randn(r, dim_in, dtype=torch.float32) for _ in range(num_experts)]
    lora_Bs = [torch.randn(dim_out, r, dtype=torch.float32) for _ in range(num_experts)]
    
    iters = 10
    scale = 1000.0 / iters
    
    print("| Batch Size (B) | Active Tasks (G) | Latency (ms) | Throughput (samples/sec) |")
    print("|---|---|---|---|")
    
    for B in batch_sizes:
        for G in active_tasks:
            # Generate input features
            X = torch.randn(B, dim_in, dtype=torch.float32)
            
            # Run benchmark
            start_time = time.perf_counter()
            for _ in range(iters):
                # 1. Similarity scoring (proxy)
                sims = X @ W_base.t()
                
                # 2. Dynamic partitioning to G active tasks
                # Assign samples to exactly G tasks
                g_indices = torch.randint(0, G, (B,))
                
                # 3. Sequential dynamic merge and forward passes
                for g in range(G):
                    mask = (g_indices == g)
                    X_g = X[mask]
                    if len(X_g) > 0:
                        # Dynamic LoRA merge
                        W_merged_g = W_base + lora_Bs[g] @ lora_As[g]
                        # Forward pass
                        logits_g = X_g @ W_merged_g.t()
                        
            latency_ms = (time.perf_counter() - start_time) * scale
            throughput = (B / (latency_ms / 1000.0))
            print(f"| {B} | {G} | {latency_ms:.2f} | {throughput:.2f} |")

if __name__ == "__main__":
    run_scaling_benchmark()
