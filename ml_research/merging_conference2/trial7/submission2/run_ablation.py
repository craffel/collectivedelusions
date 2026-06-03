import os
import json
import time
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Import existing helpers from run_experiments
from run_experiments import (
    get_dataloaders,
    check_and_train_experts,
    merge_models,
    calibrate_model_offline_tspc,
    calibrate_model_offline_real_data,
    evaluate_backbone
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for ablation: {device}")
    
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
    
    sample_sizes = [64, 128, 256, 512, 1024, 2048, 2560]
    ablation_results = {}
    
    print("\n--- Starting Sample Efficiency Ablation ---")
    
    for num_samples in sample_sizes:
        print(f"\nEvaluating with {num_samples} calibration samples...")
        ablation_results[num_samples] = {}
        
        # 1. TSPC Calibration
        start = time.time()
        tspc_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            tspc_backbones[task] = calibrate_model_offline_tspc(
                merged_backbone, task, device, num_samples=num_samples, batch_size=min(num_samples, 128)
            )
        tspc_calib_time = time.time() - start
        
        # 2. Oracle Calibration (Real Data)
        start = time.time()
        oracle_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            oracle_backbones[task] = calibrate_model_offline_real_data(
                merged_backbone, loaders[task][0], device, num_samples=num_samples
            )
        oracle_calib_time = time.time() - start
        
        # Evaluate TSPC
        tspc_res = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            res = evaluate_backbone(tspc_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
            tspc_res[task] = res[task]
        tspc_avg = np.mean(list(tspc_res.values()))
        
        # Evaluate Oracle
        oracle_res = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            res = evaluate_backbone(oracle_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
            oracle_res[task] = res[task]
        oracle_avg = np.mean(list(oracle_res.values()))
        
        ablation_results[num_samples] = {
            "tspc": {
                "mnist": tspc_res["mnist"],
                "fmnist": tspc_res["fmnist"],
                "cifar10": tspc_res["cifar10"],
                "avg": tspc_avg,
                "time": tspc_calib_time
            },
            "oracle": {
                "mnist": oracle_res["mnist"],
                "fmnist": oracle_res["fmnist"],
                "cifar10": oracle_res["cifar10"],
                "avg": oracle_avg,
                "time": oracle_calib_time
            }
        }
        
        print(f"TSPC (Data-Free): Avg Acc = {tspc_avg:.2f}% | Calibration Time = {tspc_calib_time:.4f}s")
        print(f"Oracle (Real Data): Avg Acc = {oracle_avg:.2f}% | Calibration Time = {oracle_calib_time:.4f}s")
        
    # Save ablation results
    with open("ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    # Print markdown table
    print("\n\n" + "="*50)
    print("SAMPLE EFFICIENCY ABLATION TABLE (MARKDOWN)")
    print("="*50)
    print("| Calibration Samples | TSPC Avg Acc | TSPC Time (s) | Oracle Avg Acc | Oracle Time (s) |")
    print("| --- | --- | --- | --- | --- |")
    for num_samples in sample_sizes:
        res = ablation_results[num_samples]
        print(f"| {num_samples} | {res['tspc']['avg']:.2f}% | {res['tspc']['time']:.4f}s | {res['oracle']['avg']:.2f}% | {res['oracle']['time']:.4f}s |")

if __name__ == "__main__":
    main()
