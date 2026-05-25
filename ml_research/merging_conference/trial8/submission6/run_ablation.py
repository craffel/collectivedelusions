import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import os
import numpy as np

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

from evaluate_ttmm import (
    get_resnet18_1channel,
    precompute_offline_prototypes,
    evaluate_method
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Disable cuDNN for stability
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN for stability")
        
    # Load expert model weights
    mnist_path = "expert_mnist.pth"
    kmnist_path = "expert_kmnist.pth"
    
    if not os.path.exists(mnist_path) or not os.path.exists(kmnist_path):
        print("Error: Expert models not found!")
        exit(1)
        
    sd_mnist = torch.load(mnist_path, map_location=device)
    sd_kmnist = torch.load(kmnist_path, map_location=device)
    
    # Prepare test stream dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root=".", train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root=".", train=False, download=False, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root=".", train=False, download=False, transform=transform)
    
    # Create non-stationary test stream: 90 sequential batches
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=True)
    
    stream_batches = []
    
    # Get 30 batches of each
    mnist_iter = iter(mnist_loader)
    for _ in range(30):
        imgs, lbls = next(mnist_iter)
        stream_batches.append((imgs, lbls, "MNIST"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(30):
        imgs, lbls = next(kmnist_iter)
        stream_batches.append((imgs, lbls, "KMNIST"))
        
    fashion_iter = iter(fashion_loader)
    for _ in range(30):
        imgs, lbls = next(fashion_iter)
        stream_batches.append((imgs, lbls, "FashionMNIST"))
        
    print(f"Built non-stationary test stream with {len(stream_batches)} batches.")
    
    # Precompute offline prototypes (used by KT-Fisher and PROTO-TTMM)
    mu_static, class_prototypes0, class_prototypes1 = precompute_offline_prototypes(sd_mnist, sd_kmnist, device)
    
    # Dictionary to store ablation results
    ablation_results = {}
    
    # 1. Base Configuration (Original FDF-DPA)
    print("\n--- Running Base FDF-DPA Configuration ---")
    ablation_results["Base"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.70, lr=0.005, beta_damping=0.5, anchor_layers=["conv1", "bn1", "layer1", "layer2"]
    )
    
    # 2. Damping Parameter Beta Ablation
    print("\n--- Running Beta Ablation (Beta = 0.1) ---")
    ablation_results["Beta_0.1"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.70, lr=0.005, beta_damping=0.1, anchor_layers=["conv1", "bn1", "layer1", "layer2"]
    )
    
    print("\n--- Running Beta Ablation (Beta = 0.9) ---")
    ablation_results["Beta_0.9"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.70, lr=0.005, beta_damping=0.9, anchor_layers=["conv1", "bn1", "layer1", "layer2"]
    )
    
    # 3. Novelty Threshold Tau Ablation
    print("\n--- Running Novelty Threshold Ablation (Tau = 0.55) ---")
    ablation_results["Tau_0.55"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.55, lr=0.005, beta_damping=0.5, anchor_layers=["conv1", "bn1", "layer1", "layer2"]
    )
    
    print("\n--- Running Novelty Threshold Ablation (Tau = 0.85) ---")
    ablation_results["Tau_0.85"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.85, lr=0.005, beta_damping=0.5, anchor_layers=["conv1", "bn1", "layer1", "layer2"]
    )
    
    # 4. Feature Anchoring Strategy Ablation
    print("\n--- Running Anchoring Ablation (No Anchoring, adapt all layers) ---")
    ablation_results["No_Anchoring"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.70, lr=0.005, beta_damping=0.5, anchor_layers=[]
    )
    
    print("\n--- Running Anchoring Ablation (Late Anchoring, freeze middle/head, adapt early) ---")
    ablation_results["Late_Anchoring"] = evaluate_method(
        "FDF-DPA", sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device,
        tau_entropy=0.70, lr=0.005, beta_damping=0.5, anchor_layers=["layer3", "layer4", "fc"]
    )
    
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Configuration':<25} | {'MNIST':<8} | {'KMNIST':<8} | {'Fashion':<8} | {'Overall':<8}")
    print("-"*70)
    for config, res in ablation_results.items():
        print(f"{config:<25} | {res['mnist']:<8.2f}% | {res['kmnist']:<8.2f}% | {res['fashion']:<8.2f}% | {res['overall']:<8.2f}%")
    print("="*70)
    
    # Save ablation results
    with open("ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
    print("Ablation results successfully saved to ablation_results.json")

if __name__ == "__main__":
    main()
