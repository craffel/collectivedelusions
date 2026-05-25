import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import json
import matplotlib.pyplot as plt
import os

from train_experts import SimpleCNN
from eval_stream import Evaluators, run_evaluation, set_seed

def generate_stream_with_noise(sigma, device="cpu"):
    set_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    stream_batches = []
    
    # Segment 1: Clean MNIST (batches 0-9)
    mnist_loader_clean = DataLoader(Subset(mnist_test, range(0, 640)), batch_size=64, shuffle=False)
    for imgs, labels in mnist_loader_clean:
        stream_batches.append((imgs.to(device), labels.to(device), "Clean MNIST"))
        
    # Segment 2: Noisy MNIST (batches 10-19)
    mnist_loader_noisy = DataLoader(Subset(mnist_test, range(640, 1280)), batch_size=64, shuffle=False)
    for imgs, labels in mnist_loader_noisy:
        noisy_imgs = imgs + torch.randn_like(imgs) * sigma
        stream_batches.append((noisy_imgs.to(device), labels.to(device), "Noisy MNIST"))
        
    # Segment 3: Clean FashionMNIST (batches 20-29)
    fashion_loader_clean = DataLoader(Subset(fashion_test, range(0, 640)), batch_size=64, shuffle=False)
    for imgs, labels in fashion_loader_clean:
        stream_batches.append((imgs.to(device), labels.to(device), "Clean Fashion"))
        
    # Segment 4: Noisy FashionMNIST (batches 30-39)
    fashion_loader_noisy = DataLoader(Subset(fashion_test, range(640, 1280)), batch_size=64, shuffle=False)
    for imgs, labels in fashion_loader_noisy:
        noisy_imgs = imgs + torch.randn_like(imgs) * sigma
        stream_batches.append((noisy_imgs.to(device), labels.to(device), "Noisy Fashion"))
        
    # Segment 5: Novel KMNIST (batches 40-49)
    kmnist_loader = DataLoader(Subset(kmnist_test, range(0, 640)), batch_size=64, shuffle=False)
    for imgs, labels in kmnist_loader:
        stream_batches.append((imgs.to(device), labels.to(device), "Novel KMNIST"))
        
    return stream_batches

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for noise sweep: {device}")
    
    expert0_std = SimpleCNN(use_cosface=False).to(device)
    expert1_std = SimpleCNN(use_cosface=False).to(device)
    expert0_std.load_state_dict(torch.load("models/mnist_standard.pt", map_location=device))
    expert1_std.load_state_dict(torch.load("models/fashionmnist_standard.pt", map_location=device))
    
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods_to_evaluate = {
        "Static Merging": Evaluators().static_merging,
        "BK-CoMerge": Evaluators().bk_co_merge,
        "AdaSim-CoMerge (Ours)": Evaluators().adasim_co_merge
    }
    
    sweep_results = {method: [] for method in methods_to_evaluate}
    
    for sigma in noise_levels:
        print(f"\n==================================================")
        print(f" EVALUATING WITH NOISE LEVEL SIGMA = {sigma:.1f} ")
        print(f"==================================================")
        
        stream_batches = generate_stream_with_noise(sigma, device=device)
        
        for name, fn in methods_to_evaluate.items():
            evals = Evaluators() # Create new instance to avoid state contamination
            method_fn = getattr(evals, fn.__name__)
            res = run_evaluation(name, method_fn, stream_batches, expert0_std, expert1_std, device=device)
            sweep_results[name].append(res["Overall"])
            
    # Save results to json
    out_data = {
        "noise_levels": noise_levels,
        "results": sweep_results
    }
    with open("noise_sweep_results.json", "w") as f:
        json.dump(out_data, f, indent=4)
    print("\nNoise sweep results saved to noise_sweep_results.json")
    
    # Generate the plot
    plt.figure(figsize=(7, 5))
    styles = {
        "Static Merging": ("o--", "gray"),
        "BK-CoMerge": ("s-.", "blue"),
        "AdaSim-CoMerge (Ours)": ("^-", "red")
    }
    
    for name, accs in sweep_results.items():
        marker, color = styles[name]
        plt.plot(noise_levels, accs, marker, label=name, color=color, linewidth=2, markersize=8)
        
    plt.title("Overall Stream Accuracy vs. Environmental Noise Intensity", fontsize=12, fontweight='bold')
    plt.xlabel("Gaussian Noise Standard Deviation ($\\sigma$)", fontsize=11)
    plt.ylabel("Overall Stream Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(noise_levels)
    plt.ylim(30, 60)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("noise_level_robustness.png", dpi=300)
    print("Plot saved as noise_level_robustness.png")

if __name__ == "__main__":
    main()
