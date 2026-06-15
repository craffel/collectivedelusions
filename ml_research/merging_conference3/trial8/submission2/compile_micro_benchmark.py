import time
import torch

def low_rank_projection(X, A, B):
    # h = X A, Y = h B
    H = torch.matmul(X, A)
    Y = torch.matmul(H, B)
    return Y

def main():
    print("=== Physical Micro-benchmarking with torch.compile ===")
    
    # Dimensions corresponding to our ViT-Tiny Low-Rank experts
    B_dim = 256      # Batch size
    D_dim = 192      # Hidden dimension
    r = 8            # Low-rank adapter rank
    
    # 1. Prepare tensors
    X_f32 = torch.randn(B_dim, D_dim, dtype=torch.float32)
    A_f32 = torch.randn(D_dim, r, dtype=torch.float32)
    B_f32 = torch.randn(r, D_dim, dtype=torch.float32)
    
    # FP32 eager benchmark
    # Warm-up
    for _ in range(100):
        _ = low_rank_projection(X_f32, A_f32, B_f32)
        
    t_start = time.perf_counter()
    iterations = 5000
    for _ in range(iterations):
        _ = low_rank_projection(X_f32, A_f32, B_f32)
    t_end = time.perf_counter()
    f32_eager_time = (t_end - t_start) / iterations * 1000.0 # ms per batch
    print(f"FP32 Eager: {f32_eager_time:.6f} ms per batch")
    
    # 2. Compile with torch.compile
    print("Compiling low_rank_projection with torch.compile(mode='reduce-overhead')...")
    compiled_projection = torch.compile(low_rank_projection, mode="reduce-overhead")
    
    # Compilation warm-up
    print("Running compiled warm-up iterations (this triggers compilation)...")
    for _ in range(100):
        _ = compiled_projection(X_f32, A_f32, B_f32)
        
    t_start = time.perf_counter()
    for _ in range(iterations):
        _ = compiled_projection(X_f32, A_f32, B_f32)
    t_end = time.perf_counter()
    f32_compiled_time = (t_end - t_start) / iterations * 1000.0 # ms per batch
    print(f"FP32 Compiled: {f32_compiled_time:.6f} ms per batch")
    
    speedup = f32_eager_time / f32_compiled_time
    print(f"Compiled Speedup (Eager FP32 vs Compiled FP32): {speedup:.2f}x")
    
    # 3. Compile with BF16
    X_bf16 = torch.randn(B_dim, D_dim, dtype=torch.bfloat16)
    A_bf16 = torch.randn(D_dim, r, dtype=torch.bfloat16)
    B_bf16 = torch.randn(r, D_dim, dtype=torch.bfloat16)
    
    print("Compiling BF16 projection...")
    compiled_bf16 = torch.compile(low_rank_projection, mode="reduce-overhead")
    
    for _ in range(100):
        _ = compiled_bf16(X_bf16, A_bf16, B_bf16)
        
    t_start = time.perf_counter()
    for _ in range(iterations):
        _ = compiled_bf16(X_bf16, A_bf16, B_bf16)
    t_end = time.perf_counter()
    bf16_compiled_time = (t_end - t_start) / iterations * 1000.0
    print(f"BF16 Compiled: {bf16_compiled_time:.6f} ms per batch")
    
    bf16_speedup = f32_eager_time / bf16_compiled_time
    print(f"Compiled BF16 Speedup (Eager FP32 vs Compiled BF16): {bf16_speedup:.2f}x")
    
    # Save to file
    with open("compile_benchmark_results.txt", "w") as f:
        f.write("=== torch.compile physical micro-benchmark ===\n")
        f.write(f"FP32 Eager: {f32_eager_time:.6f} ms per batch\n")
        f.write(f"FP32 Compiled: {f32_compiled_time:.6f} ms per batch\n")
        f.write(f"BF16 Compiled: {bf16_compiled_time:.6f} ms per batch\n")
        f.write(f"FP32 Compiled Speedup: {speedup:.2f}x\n")
        f.write(f"BF16 Compiled Speedup: {bf16_speedup:.2f}x\n")
        
if __name__ == '__main__':
    main()
