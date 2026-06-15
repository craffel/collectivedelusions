import time
import torch

def benchmark_large():
    print("=== Direct BF16 Benchmark on CPU (Large LLM-scale) ===")
    B, D, r = 256, 4096, 16
    
    # 1. FP32
    X_f32 = torch.randn(B, D, dtype=torch.float32)
    A_f32 = torch.randn(D, r, dtype=torch.float32)
    B_f32 = torch.randn(r, D, dtype=torch.float32)
    
    for _ in range(10):
        _ = torch.matmul(torch.matmul(X_f32, A_f32), B_f32)
        
    t_start = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        H = torch.matmul(X_f32, A_f32)
        Y = torch.matmul(H, B_f32)
    t_end = time.perf_counter()
    f32_time = (t_end - t_start) / iterations * 1000.0
    print(f"FP32: {f32_time:.6f} ms")
    
    # 2. BF16
    X_bf16 = torch.randn(B, D, dtype=torch.bfloat16)
    A_bf16 = torch.randn(D, r, dtype=torch.bfloat16)
    B_bf16 = torch.randn(r, D, dtype=torch.bfloat16)
    
    for _ in range(10):
        _ = torch.matmul(torch.matmul(X_bf16, A_bf16), B_bf16)
        
    t_start = time.perf_counter()
    for _ in range(iterations):
        H = torch.matmul(X_bf16, A_bf16)
        Y = torch.matmul(H, B_bf16)
    t_end = time.perf_counter()
    bf16_time = (t_end - t_start) / iterations * 1000.0
    print(f"BF16: {bf16_time:.6f} ms")
    speedup = f32_time / bf16_time
    print(f"BF16 Speedup: {speedup:.2f}x")
    
    with open("physical_results_large.txt", "w") as f:
        f.write(f"FP32_time_ms: {f32_time:.6f}\n")
        f.write(f"LowPrecision_time_ms: {bf16_time:.6f}\n")
        f.write(f"Physical_Speedup: {speedup:.2f}\n")

if __name__ == '__main__':
    benchmark_large()
