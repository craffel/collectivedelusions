import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Import models, helpers and loaders from eval_ttmm
from eval_ttmm import (
    modify_resnet18_for_grayscale,
    FeatureExtractorResNet18,
)
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for sweep: {device}")
    
    # Set seed for exact reproduction of stream
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
    
    # Reconstruct test stream
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_kmnist = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    batch_size = 64
    stream_batches = []
    
    mnist_indices = np.random.choice(len(test_mnist), 30 * batch_size, replace=False)
    mnist_subset = Subset(test_mnist, mnist_indices)
    mnist_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
    for x, y in mnist_loader:
        stream_batches.append((x, y, "MNIST"))
        
    kmnist_indices = np.random.choice(len(test_kmnist), 30 * batch_size, replace=False)
    kmnist_subset = Subset(test_kmnist, kmnist_indices)
    kmnist_loader = DataLoader(kmnist_subset, batch_size=batch_size, shuffle=False)
    for x, y in kmnist_loader:
        stream_batches.append((x, y, "KMNIST"))
        
    fmnist_indices = np.random.choice(len(test_fmnist), 30 * batch_size, replace=False)
    fmnist_subset = Subset(test_fmnist, fmnist_indices)
    fmnist_loader = DataLoader(fmnist_subset, batch_size=batch_size, shuffle=False)
    for x, y in fmnist_loader:
        stream_batches.append((x, y, "FashionMNIST"))
        
    # Define create_merged_model locally
    def create_merged_model(lambdas, experts):
        merged = models.resnet18()
        merged = modify_resnet18_for_grayscale(merged)
        merged.load_state_dict(experts[0].state_dict())
        merged = merged.to(device)
        
        merged_sd = merged.state_dict()
        expert_sds = [exp.state_dict() for exp in experts]
        
        for k in merged_sd.keys():
            if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                continue
            tmp = torch.zeros_like(merged_sd[k]).float()
            for idx, lam in enumerate(lambdas):
                tmp += lam * expert_sds[idx][k].float()
            merged_sd[k].copy_(tmp)
            
        for k in merged_sd.keys():
            if 'running_mean' in k or 'running_var' in k:
                tmp = torch.zeros_like(merged_sd[k]).float()
                for idx, lam in enumerate(lambdas):
                    tmp += lam * expert_sds[idx][k].float()
                merged_sd[k].copy_(tmp)
                
        merged.load_state_dict(merged_sd)
        return merged
    
    # Hyperparameter ranges
    sigma_sq_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    gamma_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    
    results = {}
    
    print("Starting hyperparameter sweep...")
    for sigma_sq in sigma_sq_list:
        results[sigma_sq] = []
        for gamma in gamma_list:
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
                        lambdas = [0.0, 0.0, 1.0]
                    else:
                        is_novel = False
                        if ll_m > ll_k:
                            lambdas = [1.0, 0.0, 0.0]
                        else:
                            lambdas = [0.0, 1.0, 0.0]
                            
                    merged_model = create_merged_model(lambdas, [expert_mnist, expert_kmnist, expert_fmnist])
                    merged_model.eval()
                    out = merged_model(x)
                    _, pred = out.max(1)
                    acc = pred.eq(y).sum().item() / y.size(0)
                    lgmm_accs.append(acc)
                    novelty_detected.append(is_novel)
                    
            # Compute metrics
            mnist_novel = novelty_detected[:30]
            kmnist_novel = novelty_detected[30:60]
            fmnist_novel = novelty_detected[60:]
            
            fpr_mnist = sum(mnist_novel) / len(mnist_novel) * 100
            fpr_kmnist = sum(kmnist_novel) / len(kmnist_novel) * 100
            avg_fpr = (fpr_mnist + fpr_kmnist) / 2
            ndr = sum(fmnist_novel) / len(fmnist_novel) * 100
            
            mnist_acc = np.mean(lgmm_accs[:30]) * 100
            kmnist_acc = np.mean(lgmm_accs[30:60]) * 100
            fmnist_acc = np.mean(lgmm_accs[60:]) * 100
            
            print(f"sigma_sq={sigma_sq}, gamma={gamma} -> NDR={ndr:.1f}%, FPR={avg_fpr:.1f}%, Accs: MNIST={mnist_acc:.1f}%, KMNIST={kmnist_acc:.1f}%, Fashion={fmnist_acc:.1f}%")
            results[sigma_sq].append({
                'gamma': gamma,
                'ndr': ndr,
                'fpr': avg_fpr,
                'mnist_acc': mnist_acc,
                'kmnist_acc': kmnist_acc,
                'fmnist_acc': fmnist_acc
            })
            
    # Save sweep results to results file
    with open("checkpoints/sweep_results.txt", "w") as sf:
        sf.write("sigma_sq,gamma,ndr,fpr,mnist_acc,kmnist_acc,fmnist_acc\n")
        for sigma_sq, r_list in results.items():
            for r in r_list:
                sf.write(f"{sigma_sq},{r['gamma']},{r['ndr']:.2f},{r['fpr']:.2f},{r['mnist_acc']:.2f},{r['kmnist_acc']:.2f},{r['fmnist_acc']:.2f}\n")
                
    # Create the figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot NDR
    for sigma_sq, r_list in results.items():
        gammas = [r['gamma'] for r in r_list]
        ndrs = [r['ndr'] for r in r_list]
        axes[0].plot(gammas, ndrs, marker='o', label=rf'$\sigma^2={sigma_sq}$')
    axes[0].set_title(r'Novelty Detection Rate (NDR) vs. $\gamma$')
    axes[0].set_xlabel(r'Safety Margin ($\gamma$)')
    axes[0].set_ylabel('NDR (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot FPR
    for sigma_sq, r_list in results.items():
        gammas = [r['gamma'] for r in r_list]
        fprs = [r['fpr'] for r in r_list]
        axes[1].plot(gammas, fprs, marker='s', label=rf'$\sigma^2={sigma_sq}$')
    axes[1].set_title(r'False Positive Rate (FPR) vs. $\gamma$')
    axes[1].set_xlabel(r'Safety Margin ($\gamma$)')
    axes[1].set_ylabel('FPR (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle('Sensitivity Analysis of L-GMM Routing Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/lgmm_hyperparameter_sweep.png")
    plt.close()
    print("Sweep complete. Plot saved in plots/lgmm_hyperparameter_sweep.png.")

if __name__ == "__main__":
    main()
