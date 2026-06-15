import torch
import time

def benchmark(d, b):
    # d is total dimension, b is block size
    num_blocks = d // b
    # Generate mock skew-symmetric generators of shape (num_blocks, b, b)
    X = torch.randn(num_blocks, b, b, device='cuda')
    Q = 0.5 * (X - X.transpose(-1, -2)) # force skew-symmetry
    
    # Warmup
    for _ in range(5):
        # Batched SVD on GPU
        U, S, Vh = torch.linalg.svd(Q)
        # Batched Cayley transform: R = (I + Q) (I - Q)^-1
        I = torch.eye(b, device='cuda').unsqueeze(0).expand(num_blocks, -1, -1)
        R = torch.bmm(I + Q, torch.linalg.inv(I - Q))
    
    torch.cuda.synchronize()
    
    # Start timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        U, S, Vh = torch.linalg.svd(Q)
        I = torch.eye(b, device='cuda').unsqueeze(0).expand(num_blocks, -1, -1)
        R = torch.bmm(I + Q, torch.linalg.inv(I - Q))
    end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / 50.0
    return avg_time_ms

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(1)
        
    print("GPU Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Benchmark d = 256 (Split-MNIST scale)
    print("\n--- Split-MNIST Scale (d = 256) ---")
    for b in [32, 64, 128, 256]:
        t_ms = benchmark(256, b)
        print(f"Block size b = {b:3d}: {t_ms:.4f} ms")
        
    # Benchmark d = 4096 (LLM scale)
    print("\n--- LLM Scale (d = 4096) ---")
    for b in [32, 64, 128, 256, 512, 1024]:
        t_ms = benchmark(4096, b)
        print(f"Block size b = {b:4d}: {t_ms:.4f} ms")
