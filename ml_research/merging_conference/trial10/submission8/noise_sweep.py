import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd

from evaluate_ttmm import SimpleCNN, precompute_prototypes, run_test_stream

def construct_stream_with_noise(mnist_test, fmnist_test, kmnist_test, seed, batches_per_phase=50, noise_sigma=0.6):
    # Set random seeds right before constructing stream to ensure absolute reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # We use batch size 64. Total 250 batches.
    stream_batches = []
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Phase 0: Clean MNIST
    for _ in range(batches_per_phase):
        stream_batches.append(next(mnist_iter))
        
    # Phase 1: Noisy MNIST
    for _ in range(batches_per_phase):
        images, labels = next(mnist_iter)
        if noise_sigma > 0:
            noise = torch.randn_like(images) * noise_sigma
            images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        else:
            images_noisy = images
        stream_batches.append((images_noisy, labels))
        
    # Phase 2: Clean FashionMNIST
    for _ in range(batches_per_phase):
        stream_batches.append(next(fmnist_iter))
        
    # Phase 3: Noisy FashionMNIST
    for _ in range(batches_per_phase):
        images, labels = next(fmnist_iter)
        if noise_sigma > 0:
            noise = torch.randn_like(images) * noise_sigma
            images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        else:
            images_noisy = images
        stream_batches.append((images_noisy, labels))
        
    # Phase 4: Novel KMNIST (unseen OOD)
    for _ in range(batches_per_phase):
        stream_batches.append(next(kmnist_iter))
        
    return stream_batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Pre-trained Experts
    print("Loading expert models...")
    model0 = SimpleCNN()
    model1 = SimpleCNN()
    model0.load_state_dict(torch.load('checkpoints/expert0.pth', map_location=device))
    model1.load_state_dict(torch.load('checkpoints/expert1.pth', map_location=device))
    model0.to(device)
    model1.to(device)
    
    # 2. Precompute Class Prototypes on Clean Calibration Subsets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_cal = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_cal = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_cal_loader = DataLoader(Subset(mnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    fmnist_cal_loader = DataLoader(Subset(fmnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    
    print("Precomputing expert class prototypes...")
    proto0 = precompute_prototypes(model0, mnist_cal_loader, device)
    proto1 = precompute_prototypes(model1, fmnist_cal_loader, device)
    
    # 3. Load all datasets
    print("Loading test datasets...")
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    noise_sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = [42, 43, 44]
    batches_per_phase = 50
    
    methods = [
        ("static", {}),
        ("fixed_tta", {"lr": 0.05, "steps": 5}),
        ("ahr_sats_dun", {"lr": 0.05, "steps": 5}),
        ("sam_ttmm", {"lr": 0.5, "rho": 0.05}),
        ("sw_sam_ttmm", {"lr": 2.0, "rho": 0.05})
    ]
    
    results_names = [m[0] for m in methods]
    results = {name: [] for name in results_names}
    results_std = {name: [] for name in results_names}
    
    all_sweep_data = []
    
    for sigma in noise_sigmas:
        print(f"\n==========================================")
        print(f"RUNNING NOISE SWEEP FOR SIGMA = {sigma}")
        print(f"==========================================")
        
        # Temp results for this sigma across seeds
        temp_accs = {name: [] for name in results_names}
        
        for seed in seeds:
            print(f"--- Seed {seed} ---")
            stream_batches = construct_stream_with_noise(mnist_test, fmnist_test, kmnist_test, seed, batches_per_phase, noise_sigma=sigma)
            
            for name, kwargs in methods:
                avg_acc, _ = run_test_stream(model0, model1, proto0, proto1, stream_batches, device, method_name=name, **kwargs)
                temp_accs[name].append(avg_acc)
                all_sweep_data.append({
                    "sigma": sigma,
                    "seed": seed,
                    "method": name,
                    "accuracy": avg_acc
                })
                
        for name in results_names:
            mean_acc = np.mean(temp_accs[name])
            std_acc = np.std(temp_accs[name])
            results[name].append(mean_acc)
            results_std[name].append(std_acc)
            print(f"Method: {name.upper()} | Mean Accuracy: {mean_acc*100:.2f}±{std_acc*100:.2f}%")
            
    # Save sweep details to CSV
    df = pd.DataFrame(all_sweep_data)
    df.to_csv("noise_sweep_results.csv", index=False)
    print("\nSaved detailed sweep results to noise_sweep_results.csv")
    
    # Plotting Accuracy vs Noise Sigma
    plt.figure(figsize=(8, 5))
    colors = {
        "static": "#7f7f7f",
        "fixed_tta": "#d62728",
        "ahr_sats_dun": "#ff7f0e",
        "sam_ttmm": "#1f77b4",
        "sw_sam_ttmm": "#2ca02c"
    }
    markers = {
        "static": "o",
        "fixed_tta": "x",
        "ahr_sats_dun": "s",
        "sam_ttmm": "^",
        "sw_sam_ttmm": "D"
    }
    labels = {
        "static": "Static Merging",
        "fixed_tta": "Fixed TTA",
        "ahr_sats_dun": "AHR-SATS-DUN",
        "sam_ttmm": "SAM-TTMM",
        "sw_sam_ttmm": "SW-SAM-TTMM (Ours)"
    }
    
    for name in results_names:
        plt.errorbar(
            noise_sigmas, 
            [acc * 100 for acc in results[name]], 
            yerr=[std * 100 for std in results_std[name]], 
            label=labels[name], 
            color=colors[name],
            marker=markers[name],
            capsize=4,
            linewidth=2,
            markersize=6
        )
        
    plt.xlabel("Gaussian Noise Intensity ($\sigma$)", fontsize=12)
    plt.ylabel("Overall Stream Accuracy (%)", fontsize=12)
    plt.title("Robustness of Test-Time Merging to Noise Intensity", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig("accuracy_vs_noise.png", dpi=300)
    print("Saved sweep figure to accuracy_vs_noise.png")

if __name__ == "__main__":
    main()
