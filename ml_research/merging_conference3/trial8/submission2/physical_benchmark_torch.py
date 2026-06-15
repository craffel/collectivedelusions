import time
import torch

def benchmark_torch_cpu():
    print("=== Physical Hardware Micro-benchmarking with PyTorch on Intel(R) Xeon(R) CPU ===")
    
    # Dimensions corresponding to our ViT-Tiny Low-Rank experts
    B = 256      # Batch size
    D = 192      # Hidden dimension
    r = 8        # Low-rank adapter rank
    
    # 1. FP32 tensors
    X_fp32 = torch.randn(B, D, dtype=torch.float32)
    A_fp32 = torch.randn(D, r, dtype=torch.float32)
    B_fp32 = torch.randn(r, D, dtype=torch.float32)
    
    # Warm-up
    for _ in range(100):
        _ = torch.matmul(torch.matmul(X_fp32, A_fp32), B_fp32)
        
    t_start = time.perf_counter()
    iterations = 10000
    for _ in range(iterations):
        H_fp32 = torch.matmul(X_fp32, A_fp32)
        Y_fp32 = torch.matmul(H_fp32, B_fp32)
    t_end = time.perf_counter()
    fp32_time = (t_end - t_start) / iterations * 1000.0 # ms per batch
    print(f"FP32 Low-Rank Adapter Projection: {fp32_time:.6f} ms per batch (B={B}, D={D}, r={r})")
    
    # 2. FP16 tensors (if supported on CPU, otherwise BF16)
    try:
        X_fp16 = torch.randn(B, D, dtype=torch.float16)
        A_fp16 = torch.randn(D, r, dtype=torch.float16)
        B_fp16 = torch.randn(r, D, dtype=torch.float16)
        
        # Warm-up
        for _ in range(100):
            _ = torch.matmul(torch.matmul(X_fp16, A_fp16), B_fp16)
            
        t_start = time.perf_counter()
        for _ in range(iterations):
            H_fp16 = torch.matmul(X_fp16, A_fp16)
            Y_fp16 = torch.matmul(H_fp16, B_fp16)
        t_end = time.perf_counter()
        fp16_time = (t_end - t_start) / iterations * 1000.0
        print(f"FP16 Low-Rank Adapter Projection: {fp16_time:.6f} ms per batch")
        speedup = fp32_time / fp16_time
        print(f"Physical CPU Speedup (FP32 vs FP16): {speedup:.2f}x")
    except Exception as e:
        print("FP16 matmul failed or not supported on this CPU, trying BFloat16...")
        X_bf16 = torch.randn(B, D, dtype=torch.bfloat16)
        A_bf16 = torch.randn(D, r, dtype=torch.bfloat16)
        B_bf16 = torch.randn(r, D, dtype=torch.bfloat16)
        
        # Warm-up
        for _ in range(100):
            _ = torch.matmul(torch.matmul(X_bf16, A_bf16), B_bf16)
            
        t_start = time.perf_counter()
        for _ in range(iterations):
            H_bf16 = torch.matmul(X_bf16, A_bf16)
            Y_bf16 = torch.matmul(H_bf16, B_bf16)
        t_end = time.perf_counter()
        fp16_time = (t_end - t_start) / iterations * 1000.0
        print(f"BFloat16 Low-Rank Adapter Projection: {fp16_time:.6f} ms per batch")
        speedup = fp32_time / fp16_time
        print(f"Physical CPU Speedup (FP32 vs BFloat16): {speedup:.2f}x")
        
    # Write to physical_results.txt
    with open("physical_results.txt", "w") as f:
        f.write(f"FP32_time_ms: {fp32_time:.6f}\n")
        f.write(f"LowPrecision_time_ms: {fp16_time:.6f}\n")
        f.write(f"Physical_Speedup: {speedup:.2f}\n")
    print("Physical results saved to physical_results.txt")

if __name__ == '__main__':
    benchmark_torch_cpu()
