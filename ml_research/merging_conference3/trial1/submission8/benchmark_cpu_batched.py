import torch
import time

def benchmark_sequential(d, b):
    num_blocks = d // b
    Q_list = [torch.randn(b, b) for _ in range(num_blocks)]
    Q_list = [0.5 * (X - X.T) for X in Q_list]
    
    # Warmup
    for _ in range(5):
        for Q in Q_list:
            U, S, Vh = torch.linalg.svd(Q)
            I = torch.eye(b)
            R = torch.matmul(I + Q, torch.linalg.inv(I - Q))
            
    # Measure
    t0 = time.perf_counter()
    for _ in range(20):
        for Q in Q_list:
            U, S, Vh = torch.linalg.svd(Q)
            I = torch.eye(b)
            R = torch.matmul(I + Q, torch.linalg.inv(I - Q))
    t1 = time.perf_counter()
    return (t1 - t0) / 20.0 * 1000.0 # ms

def benchmark_batched(d, b):
    num_blocks = d // b
    X = torch.randn(num_blocks, b, b)
    Q = 0.5 * (X - X.transpose(-1, -2))
    
    # Warmup
    for _ in range(5):
        U, S, Vh = torch.linalg.svd(Q)
        I = torch.eye(b).unsqueeze(0).expand(num_blocks, -1, -1)
        R = torch.bmm(I + Q, torch.linalg.inv(I - Q))
        
    # Measure
    t0 = time.perf_counter()
    for _ in range(20):
        U, S, Vh = torch.linalg.svd(Q)
        I = torch.eye(b).unsqueeze(0).expand(num_blocks, -1, -1)
        R = torch.bmm(I + Q, torch.linalg.inv(I - Q))
    t1 = time.perf_counter()
    return (t1 - t0) / 20.0 * 1000.0 # ms

if __name__ == '__main__':
    print("--- Split-MNIST Scale (d = 256) ---")
    for b in [32, 64, 128, 256]:
        t_seq = benchmark_sequential(256, b)
        t_batch = benchmark_batched(256, b)
        print(f"Block size b = {b:3d} | Sequential: {t_seq:8.4f} ms | Batched CPU: {t_batch:8.4f} ms")
        
    print("\n--- LLM Scale (d = 4096) ---")
    for b in [32, 64, 128, 256, 512, 1024]:
        t_seq = benchmark_sequential(4096, b)
        t_batch = benchmark_batched(4096, b)
        print(f"Block size b = {b:4d} | Sequential: {t_seq:8.4f} ms | Batched CPU: {t_batch:8.4f} ms")
