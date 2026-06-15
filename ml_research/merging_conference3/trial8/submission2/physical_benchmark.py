import time
import numpy as np
import torch

def benchmark_cpu():
    print("=== Physical Hardware Micro-benchmarking on Intel(R) Xeon(R) CPU ===")
    
    # Dimensions corresponding to our ViT-Tiny Low-Rank experts
    B = 256      # Batch size
    D = 192      # Hidden dimension
    r = 8        # Low-rank adapter rank
    K = 4        # Number of experts
    
    # 1. Prepare FP32 inputs and weights
    X_fp32 = np.random.randn(B, D).astype(np.float32)
    A_fp32 = np.random.randn(D, r).astype(np.float32)
    B_fp32 = np.random.randn(r, D).astype(np.float32)
    
    # Warm-up
    for _ in range(100):
        _ = np.dot(np.dot(X_fp32, A_fp32), B_fp32)
        
    # Benchmark FP32 low-rank projection
    t_start = time.perf_counter()
    iterations = 5000
    for _ in range(iterations):
        H_fp32 = np.dot(X_fp32, A_fp32)
        Y_fp32 = np.dot(H_fp32, B_fp32)
    t_end = time.perf_counter()
    fp32_time = (t_end - t_start) / iterations * 1000.0 # ms per batch
    print(f"FP32 Low-Rank Adapter Projection: {fp32_time:.4f} ms per batch (B={B}, D={D}, r={r})")
    
    # 2. Prepare INT8 inputs and weights (simulated using NumPy int8 / int32 dot products)
    X_int8 = np.clip(np.random.randn(B, D) * 32, -128, 127).astype(np.int8)
    A_int8 = np.clip(np.random.randn(D, r) * 32, -128, 127).astype(np.int8)
    B_int8 = np.clip(np.random.randn(r, D) * 32, -128, 127).astype(np.int8)
    
    # Warm-up
    for _ in range(100):
        # NumPy's np.dot on int8 promotes to int32/int64
        _ = np.dot(np.dot(X_int8.astype(np.int32), A_int8.astype(np.int32)), B_int8.astype(np.int32))
        
    t_start = time.perf_counter()
    for _ in range(iterations):
        # We model integer matrix multiplication and scaling/unpacking
        H_int32 = np.dot(X_int8.astype(np.int32), A_int8.astype(np.int32))
        # Simulated dynamic scaling/clipping to represent dynamic quantization back to int8
        H_int8 = np.clip(H_int32 >> 5, -128, 127).astype(np.int8)
        Y_int32 = np.dot(H_int8.astype(np.int32), B_int8.astype(np.int32))
    t_end = time.perf_counter()
    int8_time = (t_end - t_start) / iterations * 1000.0 # ms per batch
    print(f"INT8 Low-Rank Adapter Projection: {int8_time:.4f} ms per batch (B={B}, D={D}, r={r})")
    
    # Calculate physical speedup
    speedup = fp32_time / int8_time
    print(f"Physical Intel Xeon CPU Speedup (FP32 vs INT8): {speedup:.2f}x")
    
    # 3. Save physical results to a file for use in experimental updates
    with open("physical_results.txt", "w") as f:
        f.write(f"FP32_time_ms: {fp32_time:.6f}\n")
        f.write(f"INT8_time_ms: {int8_time:.6f}\n")
        f.write(f"Physical_Speedup: {speedup:.2f}\n")
    print("Physical results saved to physical_results.txt")

if __name__ == '__main__':
    benchmark_cpu()
