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
from sweep_alpha import merge_hybrid_ucpr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device("cpu")
    print("Running Hybrid UCPR Clamp Sweep on CPU...")
    
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
    
    clamp_mins = [0.05, 0.1, 0.2, 0.3, 0.5]
    clamp_maxs = [1.5, 2.0, 3.0, 5.0, 10.0]
    alpha = 0.6
    
    print("\n" + "="*70)
    print(f"{'Clamp Min':<10} | {'Clamp Max':<10} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*70)
    
    best_avg = 0.0
    best_min = 0.0
    best_max = 0.0
    best_accs = {}
    
    for c_min in clamp_mins:
        for c_max in clamp_maxs:
            hybrid_backbone = merge_hybrid_ucpr(prog_backbone, expert_backbones, alpha, clamp_min=c_min, clamp_max=c_max)
            accs = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                model = create_full_model(hybrid_backbone, heads[task], device)
                accs[task] = evaluate(model, test_loaders[task], device)
                
            avg = np.mean(list(accs.values()))
            print(f"{c_min:<10.2f} | {c_max:<10.2f} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")
            
            if avg > best_avg:
                best_avg = avg
                best_min = c_min
                best_max = c_max
                best_accs = accs
                
    print("="*70)
    print(f"Best Clamping: [{best_min:.2f}, {best_max:.2f}] with Average Accuracy: {best_avg:.2f}%")
    print(f"MNIST: {best_accs['mnist']:.2f}%, F-MNIST: {best_accs['fmnist']:.2f}%, CIFAR-10: {best_accs['cifar10']:.2f}%")

if __name__ == "__main__":
    main()
