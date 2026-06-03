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
    merge_task_arithmetic,
    merge_u_ipr,
    merge_ucpr,
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
    
    # Define backbones
    backbones = {
        "WA": merge_weight_averaging(expert_backbones),
        "TA (l=0.3)": merge_task_arithmetic(prog_backbone, expert_backbones, 0.3),
        "U-IPR": merge_u_ipr(prog_backbone, expert_backbones),
        "UCPR": merge_ucpr(prog_backbone, expert_backbones),
        "DA-HPR (S=1.0, D=0.1)": merge_depth_adaptive_hpr(prog_backbone, expert_backbones, 1.0, 0.1)
    }
    
    # Evaluate with and without BNC
    print("\n" + "="*90)
    print(f"{'Method':<22} | {'Calib?':<6} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*90)
    
    for name, backbone in backbones.items():
        # Clean (No Calibration)
        accs_clean = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            model = create_full_model(backbone, heads[task], device)
            accs_clean[task] = evaluate(model, test_loaders[task], device)
        avg_clean = np.mean(list(accs_clean.values()))
        print(f"{name:<22} | {'No':<6} | {accs_clean['mnist']:<8.2f}% | {accs_clean['fmnist']:<8.2f}% | {accs_clean['cifar10']:<8.2f}% | {avg_clean:<8.2f}%")
        
        # Calibrated
        cal_backbone = calibrate_batchnorm(backbone, heads, train_loaders, device, num_samples_per_task=512, batch_size=64)
        accs_cal = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            model = create_full_model(cal_backbone, heads[task], device)
            accs_cal[task] = evaluate(model, test_loaders[task], device)
        avg_cal = np.mean(list(accs_cal.values()))
        print(f"{name:<22} | {'Yes':<6} | {accs_cal['mnist']:<8.2f}% | {accs_cal['fmnist']:<8.2f}% | {accs_cal['cifar10']:<8.2f}% | {avg_cal:<8.2f}%")
        print("-"*90)

if __name__ == "__main__":
    main()
