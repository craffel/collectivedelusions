import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import helpers from the existing experiment script
from experiment import (
    device,
    get_dataset,
    get_dataloader,
    get_resnet18_backbone_and_head,
    evaluate_model,
    calibrate_backbone_bn
)

def run_evaluation(merged_backbone, calib_loader, experts, tasks, name_str, use_expert_affine=True):
    results = {}
    for task in tasks:
        # Calibrate using the provided calib_loader
        expert_backbone = experts[task][0] if use_expert_affine else None
        calibrated_backbone = calibrate_backbone_bn(
            merged_backbone, calib_loader, device, expert_backbone=expert_backbone, use_expert_affine=use_expert_affine
        )
        
        # Evaluate on the target task
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        acc = evaluate_model(calibrated_backbone, experts[task][1], test_loader)
        results[task] = acc
    return results

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    print(f"Running Data-Free & Cross-Domain Calibration Study on device: {device}")

    # Load experts
    experts = {}
    for task in tasks:
        backbone_path = f"models/backbone_{task}.pt"
        head_path = f"models/head_{task}.pt"

        backbone, head = get_resnet18_backbone_and_head()
        backbone = backbone.to(device)
        head = head.to(device)
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        head.load_state_dict(torch.load(head_path, map_location=device))
        experts[task] = (backbone, head)

    # Reconstruct Task Arithmetic merged model (lam = 0.4)
    base_backbone, _ = get_resnet18_backbone_and_head()
    base_backbone = base_backbone.to(device)
    base_state = base_backbone.state_dict()
    
    merged_backbone_ta, _ = get_resnet18_backbone_and_head()
    merged_backbone_ta = merged_backbone_ta.to(device)
    ta_state = {}
    lam = 0.4
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            task_vectors = []
            for t in tasks:
                vec = experts[t][0].state_dict()[key] - base_state[key]
                task_vectors.append(vec)
            ta_state[key] = base_state[key] + lam * torch.stack(task_vectors).sum(dim=0)
        else:
            ta_state[key] = base_state[key].clone()
    merged_backbone_ta.load_state_dict(ta_state)

    # 1. Uncalibrated Baselines
    print("\n--- Uncalibrated Baselines ---")
    baselines = {}
    for task in tasks:
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        acc = evaluate_model(merged_backbone_ta, experts[task][1], test_loader)
        baselines[task] = acc
        print(f"Baseline {task.upper()}: {acc:.2f}%")

    # 2. Clean Calibration (In-Domain, N = 128)
    print("\n--- Clean In-Domain Calibration (N=128) ---")
    clean_results = {}
    for task in tasks:
        calib_dataset = get_dataset(task, is_train=True)
        calib_subset = Subset(calib_dataset, list(range(128)))
        calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
        
        expert_backbone = experts[task][0]
        calibrated_backbone = calibrate_backbone_bn(
            merged_backbone_ta, calib_loader, device, expert_backbone=expert_backbone, use_expert_affine=True
        )
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        acc = evaluate_model(calibrated_backbone, experts[task][1], test_loader)
        clean_results[task] = acc
        print(f"Clean TCAC {task.upper()}: {acc:.2f}%")

    # 3. Data-Free Calibration (Gaussian Noise, N = 128)
    print("\n--- Data-Free Gaussian Noise Calibration (N=128) ---")
    # Generate random noise samples
    noise_data = torch.randn(128, 3, 32, 32)
    # Target label is dummy
    noise_targets = torch.zeros(128, dtype=torch.long)
    noise_dataset = TensorDataset(noise_data, noise_targets)
    noise_loader = DataLoader(noise_dataset, batch_size=64, shuffle=False)
    
    noise_results = {}
    for task in tasks:
        expert_backbone = experts[task][0]
        calibrated_backbone = calibrate_backbone_bn(
            merged_backbone_ta, noise_loader, device, expert_backbone=expert_backbone, use_expert_affine=True
        )
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        acc = evaluate_model(calibrated_backbone, experts[task][1], test_loader)
        noise_results[task] = acc
        print(f"Noise-Calibrated TCAC {task.upper()}: {acc:.2f}%")

    # 4. Cross-Domain Calibration (Out-of-Domain, N = 128)
    # For MNIST: calibrate using CIFAR-10
    # For Fashion-MNIST: calibrate using CIFAR-10
    # For CIFAR-10: calibrate using MNIST
    print("\n--- Cross-Domain Out-of-Domain Calibration (N=128) ---")
    cross_results = {}
    for task in tasks:
        cross_task = "cifar10" if task in ["mnist", "fashion"] else "mnist"
        print(f"Calibrating {task.upper()} using {cross_task.upper()} dataset...")
        calib_dataset = get_dataset(cross_task, is_train=True)
        calib_subset = Subset(calib_dataset, list(range(128)))
        calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
        
        expert_backbone = experts[task][0]
        calibrated_backbone = calibrate_backbone_bn(
            merged_backbone_ta, calib_loader, device, expert_backbone=expert_backbone, use_expert_affine=True
        )
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        acc = evaluate_model(calibrated_backbone, experts[task][1], test_loader)
        cross_results[task] = acc
        print(f"Cross-Calibrated TCAC {task.upper()} (via {cross_task.upper()} data): {acc:.2f}%")

    # Save results
    results_dict = {
        "baselines": baselines,
        "clean_results": clean_results,
        "noise_results": noise_results,
        "cross_results": cross_results
    }
    with open("results/data_free_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print("\nResults successfully saved to results/data_free_results.json!")

    # Plot comparisons
    plt.figure(figsize=(10, 6))
    x_indices = np.arange(len(tasks))
    width = 0.2
    
    plt.bar(x_indices - 1.5 * width, [baselines[t] for t in tasks], width, label='Uncalibrated Baseline', color='#e74c3c')
    plt.bar(x_indices - 0.5 * width, [noise_results[t] for t in tasks], width, label='Data-Free (Gaussian Noise) TCAC', color='#f1c40f')
    plt.bar(x_indices + 0.5 * width, [cross_results[t] for t in tasks], width, label='Cross-Domain (Out-of-Domain) TCAC', color='#2ecc71')
    plt.bar(x_indices + 1.5 * width, [clean_results[t] for t in tasks], width, label='Clean In-Domain TCAC', color='#3498db')
    
    plt.xticks(x_indices, [t.upper() for t in tasks], fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('TCAC Robustness under Data-Free and Out-of-Domain Calibration Settings', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=True)
    
    # Add accuracy values on top of bars
    for i, t in enumerate(tasks):
        plt.text(i - 1.5 * width, baselines[t] + 1, f"{baselines[t]:.1f}", ha='center', fontsize=8, fontweight='bold', color='#c0392b')
        plt.text(i - 0.5 * width, noise_results[t] + 1, f"{noise_results[t]:.1f}", ha='center', fontsize=8, fontweight='bold', color='#d35400')
        plt.text(i + 0.5 * width, cross_results[t] + 1, f"{cross_results[t]:.1f}", ha='center', fontsize=8, fontweight='bold', color='#27ae60')
        plt.text(i + 1.5 * width, clean_results[t] + 1, f"{clean_results[t]:.1f}", ha='center', fontsize=8, fontweight='bold', color='#2980b9')
        
    plt.tight_layout()
    plt.savefig('plots/data_free_robustness.png', dpi=300)
    print("Plot successfully saved to plots/data_free_robustness.png!")

if __name__ == "__main__":
    main()
