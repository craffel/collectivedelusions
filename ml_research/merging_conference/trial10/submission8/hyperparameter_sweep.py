import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluate_ttmm import SimpleCNN, precompute_prototypes, compute_routing_priors, run_test_stream

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Expert Models
    model0 = SimpleCNN()
    model1 = SimpleCNN()
    model0.load_state_dict(torch.load('checkpoints/expert0.pth', map_location=device))
    model1.load_state_dict(torch.load('checkpoints/expert1.pth', map_location=device))
    model0.to(device)
    model1.to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_cal = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_cal = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_cal_loader = DataLoader(Subset(mnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    fmnist_cal_loader = DataLoader(Subset(fmnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    
    proto0 = precompute_prototypes(model0, mnist_cal_loader, device)
    proto1 = precompute_prototypes(model1, fmnist_cal_loader, device)
    
    # Construct 50-batch target test stream
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    stream_batches = []
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Phase 0: Clean MNIST
    for _ in range(10):
        stream_batches.append(next(mnist_iter))
    # Phase 1: Noisy MNIST
    for _ in range(10):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((images_noisy, labels))
    # Phase 2: Clean FashionMNIST
    for _ in range(10):
        stream_batches.append(next(fmnist_iter))
    # Phase 3: Noisy FashionMNIST
    for _ in range(10):
        images, labels = next(fmnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((images_noisy, labels))
    # Phase 4: Novel KMNIST
    for _ in range(10):
        stream_batches.append(next(kmnist_iter))
        
    # Hyperparameter grids
    lrs = [0.001, 0.005, 0.01, 0.02, 0.05]
    rhos = [0.01, 0.02, 0.05, 0.10, 0.20]
    
    results_sam = []
    results_sw_sam = []
    
    print("Starting Sweep for SAM_TTMM and SW_SAM_TTMM...")
    for lr in lrs:
        for rho in rhos:
            print(f"Sweeping lr={lr}, rho={rho}...")
            # SAM-TTMM
            sam_acc, _ = run_test_stream(model0, model1, proto0, proto1, stream_batches, device, method_name="sam_ttmm", lr=lr, rho=rho)
            results_sam.append({"lr": lr, "rho": rho, "accuracy": sam_acc})
            
            # SW-SAM-TTMM
            sw_sam_acc, _ = run_test_stream(model0, model1, proto0, proto1, stream_batches, device, method_name="sw_sam_ttmm", lr=lr, rho=rho)
            results_sw_sam.append({"lr": lr, "rho": rho, "accuracy": sw_sam_acc})
            
    df_sam = pd.DataFrame(results_sam)
    df_sw_sam = pd.DataFrame(results_sw_sam)
    
    df_sam.to_csv("sweep_sam_results.csv", index=False)
    df_sw_sam.to_csv("sweep_sw_sam_results.csv", index=False)
    
    print("Sweep complete! Saved results to CSV files.")
    
    # Plot 1: Accuracy vs Learning Rate (for fixed rho=0.05)
    plt.figure(figsize=(10, 5))
    sam_fixed_rho = df_sam[df_sam['rho'] == 0.05]
    sw_sam_fixed_rho = df_sw_sam[df_sw_sam['rho'] == 0.05]
    
    plt.plot(sam_fixed_rho['lr'], sam_fixed_rho['accuracy'] * 100, marker='o', label='SAM-TTMM', color='blue', linestyle='--')
    plt.plot(sw_sam_fixed_rho['lr'], sw_sam_fixed_rho['accuracy'] * 100, marker='s', label='SW-SAM-TTMM (Ours)', color='red')
    plt.xlabel('Learning Rate (eta)')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Performance comparison across Learning Rates (at fixed rho = 0.05)')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy_vs_lr.png')
    plt.close()
    
    # Plot 2: Accuracy vs Perturbation Scale (rho) (for fixed lr=0.005)
    plt.figure(figsize=(10, 5))
    sam_fixed_lr = df_sam[df_sam['lr'] == 0.005]
    sw_sam_fixed_lr = df_sw_sam[df_sw_sam['lr'] == 0.005]
    
    plt.plot(sam_fixed_lr['rho'], sam_fixed_lr['accuracy'] * 100, marker='o', label='SAM-TTMM', color='blue', linestyle='--')
    plt.plot(sw_sam_fixed_lr['rho'], sw_sam_fixed_lr['accuracy'] * 100, marker='s', label='SW-SAM-TTMM (Ours)', color='red')
    plt.xlabel('Perturbation Scale (rho)')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Performance comparison across Perturbation Scales (at fixed lr = 0.005)')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy_vs_rho.png')
    plt.close()
    
    print("Plots saved successfully as accuracy_vs_lr.png and accuracy_vs_rho.png!")

if __name__ == "__main__":
    main()
