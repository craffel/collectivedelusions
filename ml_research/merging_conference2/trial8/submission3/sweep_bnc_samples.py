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
    merge_u_ipr,
    merge_depth_adaptive_hpr,
    create_full_model,
    calibrate_batchnorm
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Load datasets
    print("Loading datasets...")
    mnist_train, mnist_test = get_dataset("mnist", download=False)
    fmnist_train, fmnist_test = get_dataset("fmnist", download=False)
    cifar10_train, cifar10_test = get_dataset("cifar10", download=False)
    
    test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=4),
        "fmnist": DataLoader(fmnist_test, batch_size=128, shuffle=False, num_workers=4),
        "cifar10": DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=4)
    }
    
    train_loaders = {
        "mnist": DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=4),
        "fmnist": DataLoader(fmnist_train, batch_size=128, shuffle=True, num_workers=4),
        "cifar10": DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=4)
    }
    
    # Define backbones
    backbones = {
        "WA": merge_weight_averaging(expert_backbones),
        "U-IPR": merge_u_ipr(prog_backbone, expert_backbones),
        "DA-HPR (S=1.0, D=0.1)": merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1)
    }
    
    sample_sizes = [0, 16, 32, 64, 128, 256, 512, 1024]
    
    print("\n" + "="*80)
    print(f"{'Method':<22} | {'Samples/Task':<12} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*80)
    
    for name, backbone in backbones.items():
        for size in sample_sizes:
            if size == 0:
                # No BNC
                cal_backbone = backbone
            else:
                cal_backbone = calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=size, batch_size=min(size, 64))
            
            accs = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                model = create_full_model(cal_backbone, heads[task], device)
                accs[task] = evaluate(model, test_loaders[task], device)
                
            avg = np.mean(list(accs.values()))
            print(f"{name:<22} | {size:<12} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")
            
        print("-"*80)
    print("="*80)

if __name__ == "__main__":
    main()
