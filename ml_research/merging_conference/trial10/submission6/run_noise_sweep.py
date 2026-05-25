import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from run_stream import SimpleCNN, load_experts, evaluate_method

def prepare_test_stream_with_noise(sigma):
    torch.manual_seed(42) # Set seed for perfectly reproducible noise across levels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_val = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_val = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_val = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    B = 64
    stream_batches = []
    
    # Batches 0-9: Clean MNIST
    for i in range(10):
        subset = Subset(mnist_val, list(range(i * B, (i + 1) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        stream_batches.append((x, y, 'C-MN'))
        
    # Batches 10-19: Noisy MNIST (with parameter sigma)
    for i in range(10):
        subset = Subset(mnist_val, list(range((10 + i) * B, (11 + i) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        if sigma > 0:
            noisy_x = torch.clamp(x + torch.randn_like(x) * sigma, -1.0, 1.0)
        else:
            noisy_x = x
        stream_batches.append((noisy_x, y, 'N-MN'))
        
    # Batches 20-29: Clean FashionMNIST
    for i in range(10):
        subset = Subset(fashion_val, list(range(i * B, (i + 1) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        stream_batches.append((x, y, 'C-FN'))
        
    # Batches 30-39: Noisy FashionMNIST (with parameter sigma)
    for i in range(10):
        subset = Subset(fashion_val, list(range((10 + i) * B, (11 + i) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        if sigma > 0:
            noisy_x = torch.clamp(x + torch.randn_like(x) * sigma, -1.0, 1.0)
        else:
            noisy_x = x
        stream_batches.append((noisy_x, y, 'N-FN'))
        
    # Batches 40-49: Novel KMNIST
    for i in range(10):
        subset = Subset(kmnist_val, list(range(i * B, (i + 1) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        stream_batches.append((x, y, 'Nov-K'))
        
    return stream_batches

def main():
    torch.manual_seed(42)
    experts = load_experts()
    
    sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    
    results_method_e = []
    results_method_f = []
    
    print("Starting noise sweep...")
    for sigma in sigmas:
        print(f"\nEvaluating stream at noise standard deviation sigma = {sigma:.1f}")
        stream_batches = prepare_test_stream_with_noise(sigma)
        
        # Evaluate Method E (SOTA)
        res_e = evaluate_method('Method E (BK-AHR with TTBN, SOTA)', experts, stream_batches)
        # Evaluate Method F (Ours)
        res_f = evaluate_method('Method F (SMT-LDAC, Ours)', experts, stream_batches, s_momentum=0.85, depth_coherence=True, eta=0.07, beta=2.0, gamma_c=0.02)
        
        noisy_avg_e = 0.5 * (res_e['N-MN'] + res_e['N-FN'])
        noisy_avg_f = 0.5 * (res_f['N-MN'] + res_f['N-FN'])
        
        results_method_e.append({
            'sigma': sigma,
            'overall': res_e['Overall'],
            'noisy_avg': noisy_avg_e,
            'N-MN': res_e['N-MN'],
            'N-FN': res_e['N-FN']
        })
        
        results_method_f.append({
            'sigma': sigma,
            'overall': res_f['Overall'],
            'noisy_avg': noisy_avg_f,
            'N-MN': res_f['N-MN'],
            'N-FN': res_f['N-FN']
        })
        
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # 1. Plot Overall Accuracy
    plt.subplot(1, 2, 1)
    overall_e = [r['overall'] for r in results_method_e]
    overall_f = [r['overall'] for r in results_method_f]
    plt.plot(sigmas, overall_e, 'o--', label='Method E (BK-AHR, SOTA)', color='#0072B2', linewidth=2)
    plt.plot(sigmas, overall_f, 'o-', label='Method F (SMT-LDAC, Ours)', color='#009E73', linewidth=2)
    plt.xlabel('Noise Standard Deviation ($\\sigma$)', fontsize=11, fontweight='bold')
    plt.ylabel('Overall Stream Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('Overall Stream Performance', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 2. Plot Noisy Segment Average Accuracy
    plt.subplot(1, 2, 2)
    noisy_e = [r['noisy_avg'] for r in results_method_e]
    noisy_f = [r['noisy_avg'] for r in results_method_f]
    plt.plot(sigmas, noisy_e, 'o--', label='Method E (BK-AHR, SOTA)', color='#0072B2', linewidth=2)
    plt.plot(sigmas, noisy_f, 'o-', label='Method F (SMT-LDAC, Ours)', color='#009E73', linewidth=2)
    plt.xlabel('Noise Standard Deviation ($\\sigma$)', fontsize=11, fontweight='bold')
    plt.ylabel('Noisy Segments Avg Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('Performance on Noisy Segments', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/noise_robustness.png", dpi=300)
    print("Saved plots/noise_robustness.png successfully!")
    
    # Let's save the data points to print a nice latex table or use it in the response
    print("\n--- Noise Sweep Summary Table (With Controlled Seed) ---")
    print(f"{'Sigma':<6} | {'Method E Overall':<16} | {'Method F Overall':<16} | {'Method E Noisy':<14} | {'Method F Noisy':<14}")
    print("-" * 75)
    for idx, sigma in enumerate(sigmas):
        e_o = results_method_e[idx]['overall']
        f_o = results_method_f[idx]['overall']
        e_n = results_method_e[idx]['noisy_avg']
        f_n = results_method_f[idx]['noisy_avg']
        print(f"{sigma:<6.1f} | {e_o:>14.2f}% | {f_o:>14.2f}% | {e_n:>12.2f}% | {f_n:>12.2f}%")

if __name__ == "__main__":
    main()
