import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from train import get_dataset
from merge_and_evaluate import (
    get_backbone_and_head,
    merge_weight_averaging,
    merge_task_arithmetic,
    merge_u_ipr,
    merge_hns,
    create_full_model,
    merge_depth_adaptive_hpr
)
from sweep_alpha import merge_hybrid_ucpr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_with_noise(model, dataloader, device, noise_std):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Add Gaussian noise
            if noise_std > 0.0:
                noise = torch.randn_like(inputs) * noise_std
                inputs = inputs + noise
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def main():
    set_seed(42)
    device = torch.device("cpu")
    print("Running Robustness Evaluation under Gaussian Noise on CPU...")
    
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
    
    noise_stds = [0.0, 0.1, 0.2, 0.5]
    
    # Create merged backbones
    backbones = {
        "WA": merge_weight_averaging(expert_backbones),
        "TA (l=0.3)": merge_task_arithmetic(prog_backbone, expert_backbones, 0.3),
        "U-IPR": merge_u_ipr(prog_backbone, expert_backbones),
        "H-UCPR (a=0.6)": merge_hybrid_ucpr(prog_backbone, expert_backbones, 0.6),
        "DA-HPR (S=1.0, D=0.1)": merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1)
    }
    
    # Store results: method -> noise_std -> avg_acc
    results = {m: {} for m in backbones.keys()}
    
    print("\n" + "="*80)
    print(f"{'Method':<15} | {'Noise Std':<10} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*80)
    
    for method_name, backbone in backbones.items():
        for std in noise_stds:
            accs = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                model = create_full_model(backbone, heads[task], device)
                accs[task] = evaluate_with_noise(model, test_loaders[task], device, std)
                
            avg = np.mean(list(accs.values()))
            results[method_name][std] = avg
            print(f"{method_name:<15} | {std:<10.1f} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")
            
    print("="*80)
    
    # Summary of relative degradation
    print("\n--- Relative Performance Retention under severe noise (std=0.2) ---")
    for method_name in backbones.keys():
        acc_clean = results[method_name][0.0]
        acc_noisy = results[method_name][0.2]
        drop = acc_clean - acc_noisy
        pct_retained = (acc_noisy / acc_clean) * 100 if acc_clean > 0.0 else 0.0
        print(f"{method_name:<15}: Clean {acc_clean:.2f}% | Noisy {acc_noisy:.2f}% | Drop: -{drop:.2f}% | Retained: {pct_retained:.1f}%")

if __name__ == "__main__":
    main()
