import time
import torch

def benchmark_bf16():
    print("=== Direct BF16 Benchmark on CPU ===")
    B, D, r = 256, 192, 8
    
    # 1. FP32
    X_f32 = torch.randn(B, D, dtype=torch.float32)
    A_f32 = torch.randn(D, r, dtype=torch.float32)
    B_f32 = torch.randn(r, D, dtype=torch.float32)
    
    for _ in range(100):
        _ = torch.matmul(torch.matmul(X_f32, A_f32), B_f32)
        
    t_start = time.perf_counter()
    iterations = 20000
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
    
    for _ in range(100):
        _ = torch.matmul(torch.matmul(X_bf16, A_bf16), B_bf16)
        
    t_start = time.perf_counter()
    for _ in range(iterations):
        H = torch.matmul(X_bf16, A_bf16)
        Y = torch.matmul(H, B_bf16)
    t_end = time.perf_counter()
    bf16_time = (t_end - t_start) / iterations * 1000.0
    print(f"BF16: {bf16_time:.6f} ms")
    print(f"BF16 Speedup: {f32_time / bf16_time:.2f}x")
    
    # Let's save BF16 details
    with open("physical_results.txt", "w") as f:
        f.write(f"FP32_time_ms: {f32_time:.6f}\n")
        f.write(f"LowPrecision_time_ms: {bf16_time:.6f}\n")
        f.write(f"Physical_Speedup: {f32_time / bf16_time:.2f}\n")

if __name__ == '__main__':
    benchmark_bf16()
