import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from eval_tta import evaluate_method, compute_fim_priors, CNNEncoder, ClassificationHead, apply_corruption, kl_divergence_loss, project_gradients
import copy
from torch.nn.utils.stateless import functional_call
import torch.nn.functional as F

def benchmark_latency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Precompute FIM priors (needed for EWC-TTA)
    fim_priors = compute_fim_priors(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = datasets.MNIST(root="./data", train=False, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    expert_data = (mnist_loader, fashion_loader, kmnist_loader)
    
    methods = [
        "Static Merged",
        "Standard TTA",
        "EWC-TTA",
        "TTA with OPR (Ours)",
        "PC-Merge with OPR (Ours)"
    ]
    
    results_time = {}
    
    # We will run on Sequential Clean to measure latency
    for m in methods:
        print(f"\nBenchmarking latency of {m}...")
        
        # We will time the evaluate_method run
        start_time = time.time()
        # Run evaluation (150 steps)
        acc = evaluate_method(m, "Clean", "Sequential", expert_data, fim_priors, device)
        elapsed = time.time() - start_time
        
        avg_step_ms = (elapsed / 150) * 1000
        results_time[m] = avg_step_ms
        print(f"{m} - Accuracy: {acc*100:.2f}%, Avg time per step: {avg_step_ms:.2f} ms")
        
    print("\n### LATENCY BENCHMARK SUMMARY")
    print("| Method | Avg Time per Step (ms) | Overhead vs Standard TTA | Accuracy (Clean) |")
    print("|--------|------------------------|--------------------------|------------------|")
    std_time = results_time["Standard TTA"]
    for m in methods:
        overhead = f"{(results_time[m] - std_time) / std_time * 100:+.1f}%" if m != "Standard TTA" else "0.0%"
        if m == "Static Merged":
            overhead = "N/A"
        print(f"| {m} | {results_time[m]:.2f} ms | {overhead} | {evaluate_method(m, 'Clean', 'Sequential', expert_data, fim_priors, device)*100:.2f}% |")

if __name__ == "__main__":
    benchmark_latency()
