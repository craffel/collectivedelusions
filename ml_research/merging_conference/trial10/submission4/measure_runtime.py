import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from main import (
    set_seed, SimpleCNN, get_datasets, compute_prototypes
)
from run_experiments import (
    make_custom_stream, evaluate_stream_custom
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for runtime measurement: {device}")
    
    # Prepare datasets
    print("Loading Datasets...")
    mnist_train, mnist_test, fashion_train, fashion_test, kmnist_test = get_datasets()
    loader_mnist_test = DataLoader(mnist_test, batch_size=64, shuffle=False)
    loader_fashion_test = DataLoader(fashion_test, batch_size=64, shuffle=False)
    loader_kmnist_test = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    # Initialize experts
    standard_expert0 = SimpleCNN(is_cosface=False)
    standard_expert1 = SimpleCNN(is_cosface=False)
    cosface_expert0 = SimpleCNN(is_cosface=True)
    cosface_expert1 = SimpleCNN(is_cosface=True)
    
    standard_expert0.load_state_dict(torch.load("./checkpoints/standard_expert_mnist.pt", map_location=device, weights_only=True))
    standard_expert1.load_state_dict(torch.load("./checkpoints/standard_expert_fashion.pt", map_location=device, weights_only=True))
    cosface_expert0.load_state_dict(torch.load("./checkpoints/cosface_expert_mnist.pt", map_location=device, weights_only=True))
    cosface_expert1.load_state_dict(torch.load("./checkpoints/cosface_expert_fashion.pt", map_location=device, weights_only=True))
    
    standard_expert0.to(device).eval()
    standard_expert1.to(device).eval()
    cosface_expert0.to(device).eval()
    cosface_expert1.to(device).eval()
    
    # Precompute class prototypes
    print("Precomputing prototypes...")
    P0_std, P1_std = compute_prototypes(standard_expert0, standard_expert1, loader_mnist_test, loader_fashion_test, device=device)
    P0_cos, P1_cos = compute_prototypes(cosface_expert0, cosface_expert1, loader_mnist_test, loader_fashion_test, device=device)
    
    # Create stream with noise sigma = 0.6
    custom_stream = make_custom_stream(loader_mnist_test, loader_fashion_test, loader_kmnist_test, sigma=0.6)
    
    # We want to measure the average execution time per batch (in milliseconds)
    methods = ["Always Sparse", "MoG-Angular", "CP-AM (Baseline)", "BK-AHR", "FL-AHR (Ours)"]
    runtimes = {}
    
    # Warmup
    print("Warming up...")
    evaluate_stream_custom(
        standard_expert0, standard_expert1, P0_std, P1_std,
        cosface_expert0, cosface_expert1, P0_cos, P1_cos,
        custom_stream[:2], "FL-AHR (Ours)", layer=2, alpha=1.5, gating_threshold=0.535, device=device
    )
    
    for m in methods:
        print(f"Measuring runtime for {m}...")
        start_time = time.perf_counter()
        # Run over the full 50 batches
        evaluate_stream_custom(
            standard_expert0, standard_expert1, P0_std, P1_std,
            cosface_expert0, cosface_expert1, P0_cos, P1_cos,
            custom_stream, m, layer=2, alpha=1.5, gating_threshold=0.535, device=device
        )
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000.0
        avg_time_ms = total_time_ms / len(custom_stream)
        runtimes[m] = avg_time_ms
        print(f"Method: {m} | Average time per batch: {avg_time_ms:.2f} ms")
        
    print("\nRuntime Summary Table:")
    print("Method | Avg Latency per Batch (ms)")
    print("-" * 40)
    for m, t in runtimes.items():
        print(f"{m:20} | {t:.2f} ms")

if __name__ == "__main__":
    main()
