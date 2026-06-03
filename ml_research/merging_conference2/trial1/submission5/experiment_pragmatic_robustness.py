import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import helpers from the existing experiment script
from experiment import (
    device,
    get_dataset,
    get_dataloader,
    get_resnet18_backbone_and_head,
    MultiTaskModel,
    evaluate_model
)

# Reuse the flexible calibration function
from experiment_extensions import calibrate_backbone_bn_flexible

def get_calibration_indices(dataset, n, seed=42, imbalance_mode="balanced", imbalance_ratio=1.0):
    """
    Returns a list of indices of size n from the dataset based on different imbalance modes.
    - balanced: equal samples per class
    - exponential: exponential decay of sample counts across classes
    - single_class: only class 0 is present
    - two_classes: only class 0 and 1 are present
    """
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if torch.is_tensor(targets):
            targets = targets.tolist()
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
        
    num_classes = 10
    class_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
        
    np.random.seed(seed)
    # Shuffle indices for each class
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        
    if imbalance_mode == "balanced":
        samples_per_class = n // num_classes
        remainder = n % num_classes
        selected_indices = []
        for c in range(num_classes):
            take = samples_per_class + (1 if c < remainder else 0)
            selected_indices.extend(class_indices[c][:take])
            
    elif imbalance_mode == "exponential":
        if imbalance_ratio <= 1.0:
            samples_per_class = n // num_classes
            remainder = n % num_classes
            selected_indices = []
            for c in range(num_classes):
                take = samples_per_class + (1 if c < remainder else 0)
                selected_indices.extend(class_indices[c][:take])
        else:
            b = (1.0 / imbalance_ratio) ** (1.0 / 9.0)
            weights = np.array([b**i for i in range(num_classes)])
            weights = weights / weights.sum()
            counts = np.round(weights * n).astype(int)
            # Adjust to sum exactly to n
            diff = n - counts.sum()
            if diff > 0:
                counts[0] += diff
            elif diff < 0:
                for i in range(abs(diff)):
                    counts[i % num_classes] -= 1
            selected_indices = []
            for c in range(num_classes):
                take = max(0, counts[c])
                selected_indices.extend(class_indices[c][:take])
                
    elif imbalance_mode == "single_class":
        selected_indices = class_indices[0][:n]
        
    elif imbalance_mode == "two_classes":
        half = n // 2
        selected_indices = class_indices[0][:half] + class_indices[1][:n - half]
        
    else:
        raise ValueError(f"Unknown imbalance_mode {imbalance_mode}")
        
    return selected_indices

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    print(f"Running pragmatic robustness experiments on device: {device}")
    
    # Load expert models
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
        
    # Get base backbone
    base_backbone, _ = get_resnet18_backbone_and_head()
    base_backbone = base_backbone.to(device)
    base_state = base_backbone.state_dict()
    
    # Test loaders for evaluation
    test_loaders = {task: get_dataloader(task, batch_size=256, is_train=False) for task in tasks}
    
    # Perform Task Arithmetic merging with lambda = 0.4
    print("Merging models with Task Arithmetic (lambda = 0.4)...")
    merged_backbone_ta, _ = get_resnet18_backbone_and_head()
    merged_backbone_ta = merged_backbone_ta.to(device)
    ta_state = {}
    lam_val = 0.4
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            task_vectors = [experts[t][0].state_dict()[key] - base_state[key] for t in tasks]
            ta_state[key] = base_state[key] + lam_val * torch.stack(task_vectors).sum(dim=0)
        else:
            ta_state[key] = base_state[key].clone()
    merged_backbone_ta.load_state_dict(ta_state)
    
    # -------------------------------------------------------------
    # PRAGMATIC STUDY 1: Calibration Set Class-Imbalance Robustness
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STUDY 1: Calibration Set Class-Imbalance Robustness (N = 128)")
    print("=======================================================")
    
    imbalance_scenarios = {
        "Balanced (1:1 Ratio)": ("balanced", 1.0),
        "Moderate Imbalance (10:1 Ratio)": ("exponential", 10.0),
        "Severe Imbalance (100:1 Ratio)": ("exponential", 100.0),
        "Extreme Imbalance (2 Classes Only)": ("two_classes", 1.0),
        "Extreme Imbalance (1 Class Only)": ("single_class", 1.0),
    }
    
    imbalance_results = {}
    n_calib = 128
    
    for scenario_name, (mode, ratio) in imbalance_scenarios.items():
        print(f"Evaluating scenario: {scenario_name}...")
        task_accs = {}
        for task in tasks:
            dataset = get_dataset(task, is_train=True)
            indices = get_calibration_indices(dataset, n_calib, seed=42, imbalance_mode=mode, imbalance_ratio=ratio)
            calib_subset = Subset(dataset, indices)
            calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
            
            # Calibrate using Task-Specific Affine TCAC
            calibrated_backbone = calibrate_backbone_bn_flexible(
                merged_backbone_ta, calib_loader, device,
                expert_backbone=experts[task][0], use_expert_affine=True
            )
            acc = evaluate_model(calibrated_backbone, experts[task][1], test_loaders[task])
            task_accs[task] = acc
            
        avg_acc = sum(task_accs.values()) / len(tasks)
        imbalance_results[scenario_name] = {
            "tasks": task_accs,
            "average": avg_acc
        }
        print(f"  -> Average Accuracy: {avg_acc:.2f}% (MNIST: {task_accs['mnist']:.2f}%, Fashion: {task_accs['fashion']:.2f}%, CIFAR: {task_accs['cifar10']:.2f}%)")

    # -------------------------------------------------------------
    # PRAGMATIC STUDY 2: Calibration Seed Statistical Robustness
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STUDY 2: Calibration Seed Statistical Robustness (5 Random Seeds)")
    print("=======================================================")
    
    seeds = [100, 200, 300, 400, 500]
    sweep_ns = [16, 32, 128]
    seed_results = {str(n): {} for n in sweep_ns}
    
    for n in sweep_ns:
        print(f"\nEvaluating N = {n} across seeds...")
        seed_runs = []
        for seed in seeds:
            task_accs = {}
            for task in tasks:
                dataset = get_dataset(task, is_train=True)
                indices = get_calibration_indices(dataset, n, seed=seed, imbalance_mode="balanced")
                calib_subset = Subset(dataset, indices)
                calib_loader = DataLoader(calib_subset, batch_size=min(64, n), shuffle=False)
                
                calibrated_backbone = calibrate_backbone_bn_flexible(
                    merged_backbone_ta, calib_loader, device,
                    expert_backbone=experts[task][0], use_expert_affine=True
                )
                acc = evaluate_model(calibrated_backbone, experts[task][1], test_loaders[task])
                task_accs[task] = acc
            avg_acc = sum(task_accs.values()) / len(tasks)
            seed_runs.append(avg_acc)
            print(f"  Seed {seed} -> Average Accuracy: {avg_acc:.2f}%")
            
        mean_acc = np.mean(seed_runs)
        std_acc = np.std(seed_runs)
        seed_results[str(n)] = {
            "runs": seed_runs,
            "mean": mean_acc,
            "std": std_acc
        }
        print(f"N = {n} Summary -> Mean: {mean_acc:.3f}%, Std Dev: {std_acc:.3f}%")

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/pragmatic_robustness_results.json", "w") as f:
        json.dump({
            "imbalance_study": imbalance_results,
            "seed_study": seed_results
        }, f, indent=4)
    print("\nPragmatic robustness results successfully saved to results/pragmatic_robustness_results.json!")
    
    # -------------------------------------------------------------
    # Plotting Study 1: Imbalance Robustness
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    scenarios = list(imbalance_results.keys())
    mnist_vals = [imbalance_results[sc]["tasks"]["mnist"] for sc in scenarios]
    fashion_vals = [imbalance_results[sc]["tasks"]["fashion"] for sc in scenarios]
    cifar_vals = [imbalance_results[sc]["tasks"]["cifar10"] for sc in scenarios]
    avg_vals = [imbalance_results[sc]["average"] for sc in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.18
    
    plt.bar(x - 1.5*width, mnist_vals, width, label='MNIST', color='cornflowerblue')
    plt.bar(x - 0.5*width, fashion_vals, width, label='Fashion-MNIST', color='lightcoral')
    plt.bar(x + 0.5*width, cifar_vals, width, label='CIFAR-10', color='mediumaquamarine')
    plt.bar(x + 1.5*width, avg_vals, width, label='Multi-Task Average', color='gold', edgecolor='black')
    
    plt.ylabel('Evaluation Accuracy (%)', fontsize=11)
    plt.title('TCAC Robustness Under Severe Calibration Class Imbalance ($N=128$)', fontsize=12, fontweight='bold')
    plt.xticks(x, [sc.replace(" (", "\n(") for sc in scenarios], rotation=0, fontsize=9)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("plots/imbalance_robustness.png", dpi=300)
    print("Saved plots/imbalance_robustness.png!")
    
    # Plotting Study 2: Seed Stability
    plt.figure(figsize=(6, 5))
    ns_str = [str(n) for n in sweep_ns]
    means = [seed_results[n]["mean"] for n in ns_str]
    stds = [seed_results[n]["std"] for n in ns_str]
    
    # Draw error bar plot
    plt.errorbar(ns_str, means, yerr=stds, fmt='o-', color='darkviolet', ecolor='plum', elinewidth=3, capsize=8, linewidth=2, label='TCAC (Mean $\pm$ Std)')
    plt.xlabel('Calibration Dataset Size ($N$)', fontsize=11)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=11)
    plt.title('TCAC Statistical Stability Across 5 Random Calibration Seeds', fontsize=12, fontweight='bold')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/seed_stability.png", dpi=300)
    print("Saved plots/seed_stability.png!")

if __name__ == "__main__":
    main()
