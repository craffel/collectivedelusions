import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Import helper functions from eval_ttmm
from eval_ttmm import (
    modify_resnet18_for_grayscale,
    FeatureExtractorResNet18,
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for batch size sweep: {device}")
    
    # Set seed for exact reproduction
    torch.manual_seed(2026)
    np.random.seed(2026)
    
    # Load experts
    print("Loading pre-trained experts...")
    expert_mnist = models.resnet18()
    expert_mnist = modify_resnet18_for_grayscale(expert_mnist)
    expert_mnist.load_state_dict(torch.load("checkpoints/expert_mnist.pth", map_location=device))
    expert_mnist = expert_mnist.to(device).eval()
    
    expert_kmnist = models.resnet18()
    expert_kmnist = modify_resnet18_for_grayscale(expert_kmnist)
    expert_kmnist.load_state_dict(torch.load("checkpoints/expert_kmnist.pth", map_location=device))
    expert_kmnist = expert_kmnist.to(device).eval()
    
    expert_fmnist = models.resnet18()
    expert_fmnist = modify_resnet18_for_grayscale(expert_fmnist)
    expert_fmnist.load_state_dict(torch.load("checkpoints/expert_fashionmnist.pth", map_location=device))
    expert_fmnist = expert_fmnist.to(device).eval()
    
    expert_mnist_fe = FeatureExtractorResNet18(expert_mnist).to(device).eval()
    expert_kmnist_fe = FeatureExtractorResNet18(expert_kmnist).to(device).eval()
    expert_fmnist_fe = FeatureExtractorResNet18(expert_fmnist).to(device).eval()
    
    # Retrieve class centroids
    centroids_mnist = expert_mnist.fc.weight.data.clone() # (10, 512)
    centroids_kmnist = expert_kmnist.fc.weight.data.clone() # (10, 512)
    
    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading test datasets...")
    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_kmnist = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    # Fixed number of total evaluation samples per domain (1920)
    samples_per_domain = 1920
    
    # Pre-draw indices for consistency across sweeps
    mnist_indices = np.random.choice(len(test_mnist), samples_per_domain, replace=False)
    kmnist_indices = np.random.choice(len(test_kmnist), samples_per_domain, replace=False)
    fmnist_indices = np.random.choice(len(test_fmnist), samples_per_domain, replace=False)
    
    mnist_subset = Subset(test_mnist, mnist_indices)
    kmnist_subset = Subset(test_kmnist, kmnist_indices)
    fmnist_subset = Subset(test_fmnist, fmnist_indices)
    
    # Batch sizes to sweep
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    
    # Fixed L-GMM parameters
    sigma_sq = 0.1
    gamma = 0.8
    
    results = []
    
    print("Starting Batch Size Sweep...")
    for B in batch_sizes:
        print(f"Evaluating Batch Size: B = {B}")
        
        # Build dataloaders for this batch size
        mnist_loader = DataLoader(mnist_subset, batch_size=B, shuffle=False)
        kmnist_loader = DataLoader(kmnist_subset, batch_size=B, shuffle=False)
        fmnist_loader = DataLoader(fmnist_subset, batch_size=B, shuffle=False)
        
        stream_batches = []
        for x, y in mnist_loader:
            stream_batches.append((x, y, "MNIST"))
        num_mnist_batches = len(stream_batches)
        
        for x, y in kmnist_loader:
            stream_batches.append((x, y, "KMNIST"))
        num_kmnist_batches = len(stream_batches) - num_mnist_batches
        
        for x, y in fmnist_loader:
            stream_batches.append((x, y, "FashionMNIST"))
        num_fmnist_batches = len(stream_batches) - num_mnist_batches - num_kmnist_batches
        
        # 1. Calibrate threshold using the first batch
        with torch.no_grad():
            x_cal, _, _ = stream_batches[0]
            x_cal = x_cal.to(device)
            _, feat_cal = expert_mnist_fe(x_cal)
            
            feat_cal_norm = F.normalize(feat_cal, p=2, dim=1)
            centroids_mnist_norm = F.normalize(centroids_mnist, p=2, dim=1)
            dist_cal = torch.cdist(feat_cal_norm, centroids_mnist_norm)
            ll_cal = torch.logsumexp(-dist_cal**2 / (2 * sigma_sq), dim=1) - np.log(10)
            mean_ll_cal = ll_cal.mean().item()
            threshold = mean_ll_cal - gamma
            
        # 2. Run stream evaluation
        novelty_detected = []
        lgmm_accs = []
        
        for idx, (x, y, domain) in enumerate(stream_batches):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, feat_m = expert_mnist_fe(x)
                _, feat_k = expert_kmnist_fe(x)
                
                feat_m_norm = F.normalize(feat_m, p=2, dim=1)
                feat_k_norm = F.normalize(feat_k, p=2, dim=1)
                centroids_mnist_norm = F.normalize(centroids_mnist, p=2, dim=1)
                centroids_kmnist_norm = F.normalize(centroids_kmnist, p=2, dim=1)
                
                dist_m = torch.cdist(feat_m_norm, centroids_mnist_norm)
                ll_m = (torch.logsumexp(-dist_m**2 / (2 * sigma_sq), dim=1) - np.log(10)).mean().item()
                
                dist_k = torch.cdist(feat_k_norm, centroids_kmnist_norm)
                ll_k = (torch.logsumexp(-dist_k**2 / (2 * sigma_sq), dim=1) - np.log(10)).mean().item()
                
                max_ll = max(ll_m, ll_k)
                
                if max_ll < threshold:
                    is_novel = True
                else:
                    is_novel = False
                    
                novelty_detected.append(is_novel)
                
        # Compute metrics
        mnist_novel = novelty_detected[:num_mnist_batches]
        kmnist_novel = novelty_detected[num_mnist_batches:num_mnist_batches+num_kmnist_batches]
        fmnist_novel = novelty_detected[num_mnist_batches+num_kmnist_batches:]
        
        fpr_mnist = sum(mnist_novel) / len(mnist_novel) * 100
        fpr_kmnist = sum(kmnist_novel) / len(kmnist_novel) * 100
        avg_fpr = (fpr_mnist + fpr_kmnist) / 2
        ndr = sum(fmnist_novel) / len(fmnist_novel) * 100
        
        print(f"B = {B} -> NDR = {ndr:.2f}%, FPR = {avg_fpr:.2f}% (MNIST FPR = {fpr_mnist:.2f}%, KMNIST FPR = {fpr_kmnist:.2f}%)")
        results.append({
            'batch_size': B,
            'ndr': ndr,
            'fpr': avg_fpr,
            'fpr_mnist': fpr_mnist,
            'fpr_kmnist': fpr_kmnist
        })
        
    # Save sweep results
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/batch_size_results.txt", "w") as bf:
        bf.write("batch_size,ndr,fpr,fpr_mnist,fpr_kmnist\n")
        for r in results:
            bf.write(f"{r['batch_size']},{r['ndr']:.2f},{r['fpr']:.2f},{r['fpr_mnist']:.2f},{r['fpr_kmnist']:.2f}\n")
            
    # Generate Plot
    plt.figure(figsize=(7, 5))
    bs = [r['batch_size'] for r in results]
    ndrs = [r['ndr'] for r in results]
    fprs = [r['fpr'] for r in results]
    
    plt.plot(bs, ndrs, marker='o', color='green', linewidth=2, label='Novelty Detection Rate (NDR)')
    plt.plot(bs, fprs, marker='s', color='red', linewidth=2, label='False Positive Rate (FPR)')
    
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, [str(b) for b in batch_sizes])
    plt.xlabel('Streaming Batch Size (B)', fontsize=11)
    plt.ylabel('Rate (%)', fontsize=11)
    plt.title('Impact of Streaming Batch Size on L-GMM Routing', fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim(-5, 105)
    plt.legend(loc='center right', fontsize=10)
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/lgmm_batch_size_sweep.png")
    plt.close()
    print("Batch size sweep complete. Plot saved in plots/lgmm_batch_size_sweep.png.")

if __name__ == "__main__":
    main()
