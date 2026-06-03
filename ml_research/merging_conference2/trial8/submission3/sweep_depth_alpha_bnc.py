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
    merge_depth_adaptive_hpr,
    create_full_model
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=512, batch_size=64):
    """Calibrates the BatchNorm statistics of the merged backbone on a mixed dataset."""
    first_task = list(heads.keys())[0]
    cal_model = create_full_model(backbone, heads[first_task], device)
    
    cal_model.train()
    
    for m in cal_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = 0.1
            
    mixed_inputs = []
    for task_name, loader in train_loaders.items():
        dataset = loader.dataset
        num_samples = min(num_samples_per_task, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            img, _ = dataset[idx]
            mixed_inputs.append(img)
            
    random.shuffle(mixed_inputs)
    cal_loader = DataLoader(mixed_inputs, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        for epoch in range(5):
            for x in cal_loader:
                x = x.to(device)
                _ = cal_model(x)
                
    cal_backbone = {}
    for k, v in cal_model.state_dict().items():
        if not k.startswith("fc."):
            cal_backbone[k] = v.clone()
            
    return cal_backbone

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
    
    # Grid search over alpha_shallow and alpha_deep
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\n" + "="*80)
    print(f"{'Alpha_sh':<10} | {'Alpha_dp':<10} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*80)
    
    best_avg = 0.0
    best_config = None
    
    for alpha_sh in alphas:
        for alpha_dp in alphas:
            # Create DA-HPR backbone
            backbone = merge_depth_adaptive_hpr(prog_backbone, expert_backbones, alpha_sh, alpha_dp)
            # Calibrate BN
            cal_backbone = calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=512, batch_size=64)
            
            # Evaluate
            accs = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                model = create_full_model(cal_backbone, heads[task], device)
                accs[task] = evaluate(model, test_loaders[task], device)
                
            avg = np.mean(list(accs.values()))
            print(f"{alpha_sh:<10.1f} | {alpha_dp:<10.1f} | {accs['mnist']:<8.2f}% | {accs['fmnist']:<8.2f}% | {accs['cifar10']:<8.2f}% | {avg:<8.2f}%")
            
            if avg > best_avg:
                best_avg = avg
                best_config = (alpha_sh, alpha_dp, accs)
                
    print("="*80)
    print(f"\nBest Config with BNC: Alpha_sh={best_config[0]:.1f}, Alpha_dp={best_config[1]:.1f}")
    print(f"MNIST: {best_config[2]['mnist']:.2f}%, F-MNIST: {best_config[2]['fmnist']:.2f}%, CIFAR-10: {best_config[2]['cifar10']:.2f}%, Average: {best_avg:.2f}%")

if __name__ == "__main__":
    main()
