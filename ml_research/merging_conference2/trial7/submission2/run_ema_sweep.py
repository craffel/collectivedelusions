import os
import json
import time
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# Import existing helpers from run_experiments
from run_experiments import (
    get_dataloaders,
    check_and_train_experts,
    merge_models,
    evaluate_backbone
)

def calibrate_model_offline_ema(merged_backbone_state, calib_loader, device, num_samples, momentum):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(merged_backbone_state, strict=False)
    model = model.to(device)
    
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.reset_running_stats()
            m.momentum = momentum
            
    samples_accumulated = 0
    with torch.no_grad():
        for inputs, _ in calib_loader:
            if samples_accumulated >= num_samples:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            samples_accumulated += inputs.size(0)
            
    calibrated_state = {}
    model_state = model.state_dict()
    for k, v in model_state.items():
        if not k.startswith("fc."):
            calibrated_state[k] = v.cpu()
            
    return calibrated_state

def calibrate_model_offline_ecc(merged_backbone_state, calib_loader, device, num_samples):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(merged_backbone_state, strict=False)
    model = model.to(device)
    
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.reset_running_stats()
            
    samples_accumulated = 0
    step = 0
    with torch.no_grad():
        for inputs, _ in calib_loader:
            if samples_accumulated >= num_samples:
                break
            momentum = 1.0 / (step + 1)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum
            inputs = inputs.to(device)
            _ = model(inputs)
            samples_accumulated += inputs.size(0)
            step += 1
            
    calibrated_state = {}
    model_state = model.state_dict()
    for k, v in model_state.items():
        if not k.startswith("fc."):
            calibrated_state[k] = v.cpu()
            
    return calibrated_state

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for EMA sweep: {device}")
    
    loaders = get_dataloaders()
    expert_states = check_and_train_experts(loaders, device)
    
    # Load Progenitor W_0 state dict
    progenitor = resnet18(weights=ResNet18_Weights.DEFAULT)
    progenitor_state = progenitor.state_dict()
    
    # Standard batch size 64 eval loaders
    eval_loaders_64 = {
        "mnist": DataLoader(torchvision.datasets.MNIST(root="./data", train=False, download=False, 
                            transform=loaders["mnist"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4),
        "fmnist": DataLoader(torchvision.datasets.FashionMNIST(root="./data", train=False, download=False,
                            transform=loaders["fmnist"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4),
        "cifar10": DataLoader(torchvision.datasets.CIFAR10(root="./data", train=False, download=False,
                            transform=loaders["cifar10"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4)
    }
    
    # Weight Averaging backbone
    merged_backbone = merge_models(progenitor_state, expert_states, merge_method="wa", lambda_val=0.3)
    
    sample_sizes = [128, 256, 512, 1024, 2560]
    momentums = [0.01, 0.05, 0.1, 0.2]
    
    results = {}
    
    print("\n--- Starting EMA Momentum vs. ECC Sweep ---")
    
    # 1. Run ECC baseline for each sample size
    results["ecc"] = {}
    for num_samples in sample_sizes:
        print(f"\nRunning ECC with {num_samples} samples...")
        oracle_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            oracle_backbones[task] = calibrate_model_offline_ecc(
                merged_backbone, loaders[task][0], device, num_samples=num_samples
            )
        
        ecc_res = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            res = evaluate_backbone(oracle_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
            ecc_res[task] = res[task]
        ecc_avg = np.mean(list(ecc_res.values()))
        results["ecc"][num_samples] = ecc_avg
        print(f"ECC {num_samples} samples: Avg Acc = {ecc_avg:.2f}%")
        
    # 2. Sweep over constant momentums
    for m in momentums:
        results[f"ema_{m}"] = {}
        print(f"\n--- Sweeping EMA Momentum = {m} ---")
        for num_samples in sample_sizes:
            print(f"Running EMA (m={m}) with {num_samples} samples...")
            ema_backbones = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                ema_backbones[task] = calibrate_model_offline_ema(
                    merged_backbone, loaders[task][0], device, num_samples=num_samples, momentum=m
                )
            
            ema_res = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                res = evaluate_backbone(ema_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
                ema_res[task] = res[task]
            ema_avg = np.mean(list(ema_res.values()))
            results[f"ema_{m}"][num_samples] = ema_avg
            print(f"EMA (m={m}) {num_samples} samples: Avg Acc = {ema_avg:.2f}%")
            
    # Save sweep results
    with open("ema_sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Print markdown table
    print("\n\n" + "="*70)
    print("EMA MOMENTUM VS. ECC-MERGE SWEEP TABLE (MARKDOWN)")
    print("="*70)
    print("| Calibration Samples | EMA (m=0.01) | EMA (m=0.05) | EMA (m=0.1) | EMA (m=0.2) | ECC-Merge (Ours) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for num_samples in sample_sizes:
        print(f"| {num_samples} | "
              f"{results['ema_0.01'][num_samples]:.2f}% | "
              f"{results['ema_0.05'][num_samples]:.2f}% | "
              f"{results['ema_0.1'][num_samples]:.2f}% | "
              f"{results['ema_0.2'][num_samples]:.2f}% | "
              f"{results['ecc'][num_samples]:.2f}% |")

if __name__ == "__main__":
    main()
