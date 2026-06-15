import time
import torch
import json

def benchmark_svd_for_dim(d, device='cpu', num_runs=20):
    # Warmup
    x = torch.randn(d, d, device=device)
    for _ in range(5):
        torch.linalg.svd(x, full_matrices=False)
    if device == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(num_runs):
        U, S, V = torch.linalg.svd(x, full_matrices=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000
    return avg_time_ms

def main():
    dims = [192, 512, 1024, 2048, 4096]
    results = {}
    
    print("=== Benchmarking SVD Execution Time ===")
    
    # CPU Benchmark
    print("\nRunning CPU Benchmark...")
    results["cpu"] = []
    for d in dims:
        print(f"Dim {d}x{d}...", end="", flush=True)
        try:
            t_ms = benchmark_svd_for_dim(d, device='cpu', num_runs=10 if d >= 2048 else 20)
            print(f" {t_ms:.2f} ms")
            results["cpu"].append({"dim": d, "time_ms": t_ms})
        except Exception as e:
            print(f" Error: {e}")
            
    # CUDA Benchmark
    if torch.cuda.is_available():
        print("\nRunning CUDA Benchmark...")
        results["cuda"] = []
        for d in dims:
            print(f"Dim {d}x{d}...", end="", flush=True)
            try:
                t_ms = benchmark_svd_for_dim(d, device='cuda', num_runs=10 if d >= 2048 else 20)
                print(f" {t_ms:.2f} ms")
                results["cuda"].append({"dim": d, "time_ms": t_ms})
            except Exception as e:
                print(f" Error: {e}")
    else:
        print("\nCUDA is not available for benchmarking.")
        
    # Save to JSON
    with open("results/svd_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark completed and saved to results/svd_benchmark.json")

if __name__ == "__main__":
    main()
