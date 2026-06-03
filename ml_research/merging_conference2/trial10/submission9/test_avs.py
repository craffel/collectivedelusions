import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from main import (
    MultiTaskResNet, 
    get_datasets, 
    quantize_weights, 
    run_naive_mixed_calibration, 
    evaluate_model, 
    merge_models, 
    set_seed
)

def apply_variance_scaling(model, beta):
    scaled_model = copy.deepcopy(model)
    for name, module in scaled_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.running_var is not None:
                module.running_var.copy_(module.running_var * beta)
    return scaled_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for AVS evaluation")
    set_seed(42)

    # 1. Load Datasets
    train_loaders, test_loaders = get_datasets(batch_size=256, dry_run=False)

    # 2. Re-create model structure
    heads = {
        "MNIST": nn.Linear(512, 10),
        "FMNIST": nn.Linear(512, 10),
        "CIFAR10": nn.Linear(512, 10)
    }
    
    # 3. Load pre-trained models from checkpoints
    checkpoint_dir = "./checkpoints"
    print("Loading pre-trained progenitor and experts...")
    
    progenitor = MultiTaskResNet(heads)
    progenitor.load_state_dict(torch.load(f"{checkpoint_dir}/progenitor.pt", map_location=device))
    
    experts = {}
    for task_name in heads.keys():
        expert_model = copy.deepcopy(progenitor)
        expert_model.load_state_dict(torch.load(f"{checkpoint_dir}/{task_name}_expert.pt", map_location=device))
        experts[task_name] = expert_model

    # Merge model with optimal lambda=0.4
    merged_model_opt = merge_models(progenitor, experts, lam=0.4)
    
    # Run Naive Mixed Calibration at N=64
    naive_cal_model = run_naive_mixed_calibration(merged_model_opt, train_loaders, N=64, device=device)
    
    beta_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = {
        "FP32": [],
        "PC-INT4": [],
        "Noisy_FP32": []
    }
    
    for beta in beta_values:
        print(f"\nEvaluating Adaptive Variance Scaling (beta = {beta}) ...")
        scaled_model = apply_variance_scaling(naive_cal_model, beta)
        
        # 1. Clean FP32
        res_fp = evaluate_model(scaled_model, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        print(f"FP32 Acc: {res_fp['Average']:.2f}% (MNIST: {res_fp['MNIST']:.2f}%, FMNIST: {res_fp['FMNIST']:.2f}%, CIFAR10: {res_fp['CIFAR10']:.2f}%)")
        results["FP32"].append(res_fp["Average"])
        
        # 2. Noisy FP32 (std=0.1)
        res_noise = evaluate_model(scaled_model, test_loaders, calibration_type="None", noise_std=0.1, device=device)
        print(f"Noisy FP32 Acc: {res_noise['Average']:.2f}% (MNIST: {res_noise['MNIST']:.2f}%, FMNIST: {res_noise['FMNIST']:.2f}%, CIFAR10: {res_noise['CIFAR10']:.2f}%)")
        results["Noisy_FP32"].append(res_noise["Average"])
        
        # 3. PC-INT4
        scaled_q4 = quantize_weights(scaled_model, bits=4, per_channel=True)
        res_q4 = evaluate_model(scaled_q4, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        print(f"PC-INT4 Acc: {res_q4['Average']:.2f}% (MNIST: {res_q4['MNIST']:.2f}%, FMNIST: {res_q4['FMNIST']:.2f}%, CIFAR10: {res_q4['CIFAR10']:.2f}%)")
        results["PC-INT4"].append(res_q4["Average"])
        
    print("\n--- Final Summary of Adaptive Variance Scaling ---")
    print(f"{'Beta':<10} | {'FP32 Acc':<12} | {'Noisy FP32 Acc':<16} | {'PC-INT4 Acc':<12}")
    print("-" * 60)
    for i, beta in enumerate(beta_values):
        print(f"{beta:<10.2f} | {results['FP32'][i]:<12.2f} | {results['Noisy_FP32'][i]:<16.2f} | {results['PC-INT4'][i]:<12.2f}")
        
    # Write results to file
    with open("avs_results.json", "w") as f:
        json.dump({"beta": beta_values, "results": results}, f, indent=4)

if __name__ == "__main__":
    main()
