import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from train import get_dataset, evaluate
from merge_and_evaluate import (
    get_backbone_and_head,
    merge_weight_averaging,
    create_full_model,
    compute_l2_norm_dim0
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def merge_hybrid_ucpr(progenitor_backbone, expert_backbones, alpha, clamp_min=0.1, clamp_max=10.0):
    merged_backbone = {}
    keys = progenitor_backbone.keys()
    K = len(expert_backbones)
    
    # Standard Weight Averaging backbone
    wa_backbone = merge_weight_averaging(expert_backbones)
    
    for k in keys:
        prog_tensor = progenitor_backbone[k]
        wa_tensor = wa_backbone[k]
        
        # If it's not a floating point or is a running buffer, use averaged WA
        if not prog_tensor.is_floating_point() or "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            merged_backbone[k] = wa_tensor.clone()
            continue
            
        # Task vectors
        task_vectors = [expert[k] - prog_tensor for expert in expert_backbones]
        merged_vector = wa_tensor - prog_tensor
        
        # 1. Compute layer-wise (Frobenius) scaling factor S_l
        expert_norms_fro = [torch.norm(tv, p="fro") for tv in task_vectors]
        avg_expert_norm_fro = torch.stack(expert_norms_fro).mean()
        merged_norm_fro = torch.norm(merged_vector, p="fro")
        S_l = avg_expert_norm_fro / (merged_norm_fro + 1e-8)
        S_l = torch.clamp(S_l, clamp_min, clamp_max)
        
        # 2. Compute channel-wise (dim 0) scaling factor S_lc
        expert_norms_list = [compute_l2_norm_dim0(tv) for tv in task_vectors]
        avg_expert_norm_c = torch.stack(expert_norms_list, dim=0).mean(dim=0)
        merged_norm_c = compute_l2_norm_dim0(merged_vector)
        S_lc = avg_expert_norm_c / (merged_norm_c + 1e-8)
        S_lc = torch.clamp(S_lc, clamp_min, clamp_max)
        
        # 3. Interpolate scaling factors
        S_hybrid = alpha * S_lc + (1.0 - alpha) * S_l
        
        # Apply scaling channel-by-channel along dimension 0
        scaled_merged_vector = merged_vector.clone()
        for c in range(scaled_merged_vector.size(0)):
            scaled_merged_vector[c] = scaled_merged_vector[c] * S_hybrid[c]
            
        # Reconstruct weights
        merged_backbone[k] = prog_tensor + scaled_merged_vector
        
    return merged_backbone

def main():
    set_seed(42)
    device = torch.device("cpu")
    print("Running Hybrid UCPR Sweep on CPU...")
    
    prog_path = "progenitor.pt"
    mnist_path = "mnist_expert.pt"
    fmnist_path = "fmnist_expert.pt"
    cifar10_path = "cifar10_expert.pt"
    
    prog_sd = torch.load(prog_path, map_location=device)
    mnist_sd = torch.load(mnist_path, map_location=device)
    fmnist_sd = torch.load(fmnist_path, map_location=device)
    cifar10_sd = torch.load(cifar10_path, map_location=device)
    
    prog_backbone, _ = get_backbone_and_head(prog_sd)
    mnist_backbone, mnist_head = get_backbone_and_head(mnist_sd)
    fmnist_backbone, fmnist_head = get_backbone_and_head(fmnist_sd)
    cifar10_backbone, cifar10_head = get_backbone_and_head(cifar10_sd)
    
    expert_backbones = [mnist_backbone, fmnist_backbone, cifar10_backbone]
    heads = {
        "mnist": mnist_head,
        "fmnist": fmnist_head,
        "cifar10": cifar10_head
    }
    
    _, mnist_test = get_dataset("mnist", download=False)
    _, fmnist_test = get_dataset("fmnist", download=False)
    _, cifar10_test = get_dataset("cifar10", download=False)
    
    test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=4),
        "fmnist": DataLoader(fmnist_test, batch_size=128, shuffle=False, num_workers=4),
        "cifar10": DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=4)
    }
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("\n" + "="*60)
    print(f"{'Alpha':<8} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*60)
    
    best_avg = 0.0
    best_alpha = 0.0
    best_accs = {}
    
    for alpha in alphas:
        hybrid_backbone = merge_hybrid_ucpr(prog_backbone, expert_backbones, alpha)
        accs = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            model = create_full_model(hybrid_backbone, heads[task], device)
            accs[task] = evaluate(model, test_loaders[task], device)
            
        avg = np.mean(list(accs.values()))
        print(f"{alpha:<8.1f} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")
        
        if avg > best_avg:
            best_avg = avg
            best_alpha = alpha
            best_accs = accs
            
    print("="*60)
    print(f"Best Alpha: {best_alpha:.1f} with Average Accuracy: {best_avg:.2f}%")
    print(f"MNIST: {best_accs['mnist']:.2f}%, F-MNIST: {best_accs['fmnist']:.2f}%, CIFAR-10: {best_accs['cifar10']:.2f}%")

if __name__ == "__main__":
    main()
