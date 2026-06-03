import os
import json
import torch
torch.backends.cudnn.enabled = False
import copy
import matplotlib.pyplot as plt
from torch.utils.data import Subset, ConcatDataset, DataLoader
import numpy as np

# Add src to system path to import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.merge import (
    get_expert_models,
    get_datasets,
    compute_fisher_information,
    collect_expert_statistics,
    build_merged_model,
    apply_calibration,
    evaluate_multi_task,
    run_reda
)

def run_calibration_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sweep running on device: {device}")
    
    data_dir = "./data"
    experts_dir = "./experts"
    
    # Load expert models and datasets
    print("Loading expert models...")
    experts = get_expert_models(experts_dir, device)
    all_datasets = get_datasets(data_dir)
    
    # Define calibration sizes to evaluate
    cal_sizes = [4, 8, 16, 32, 64, 128, 256]
    methods = ['none', 'sp-taac', 'taac', 'r-taac', 's-tcac', 'fwas', 'reda']
    
    # Initialize dictionary to store results
    sweep_results = {method: {size: [] for size in cal_sizes} for method in methods}
    sweep_results['expert_baselines'] = {}
    
    # Evaluate individual experts
    print("Evaluating individual expert baselines...")
    tasks = list(experts.keys())
    for task, model in experts.items():
        _, test_ds = all_datasets[task]
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100.0 * correct / total
        sweep_results['expert_baselines'][task] = acc
        print(f"  {task.upper()} Expert Accuracy: {acc:.2f}%")
    
    # Run sweep over calibration set sizes
    for cal_size in cal_sizes:
        print(f"\n================ Running Sweep for Cal Size: {cal_size} ================")
        
        # Prepare task-specific calibration sets
        calibration_sets = {}
        test_datasets = {}
        for task in experts.keys():
            train_ds, test_ds = all_datasets[task]
            # Calibration subset: deterministic, using generator with fixed seed
            g = torch.Generator().manual_seed(100)
            cal_indices = torch.randperm(len(train_ds), generator=g)[:cal_size].tolist()
            calibration_sets[task] = Subset(train_ds, cal_indices)
            test_datasets[task] = test_ds
            
        joint_cal_set = ConcatDataset([calibration_sets[t] for t in experts.keys()])
        
        # 1. Compute Fisher Information on-the-fly using the calibration sets
        fisher_info = compute_fisher_information(experts, calibration_sets, device)
        
        # 2. Collect expert statistics
        expert_stats = collect_expert_statistics(experts, calibration_sets, device)
        
        for method in methods:
            if method == 'reda':
                # REDA is run separately as it fine-tunes the heads
                merged_model = build_merged_model(experts, method='wa')
                scores = run_reda(merged_model, experts, calibration_sets, test_datasets, device)
            elif method == 'fwas':
                # For FWAS, we sweep over multiple lambda values and choose the best one!
                best_score = None
                best_lambda = None
                fwas_lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
                for f_lam in fwas_lambdas:
                    merged_model = build_merged_model(experts, method='wa')
                    cal_hooks = apply_calibration(
                        merged_model=merged_model,
                        expert_stats=expert_stats,
                        joint_cal_set=joint_cal_set,
                        method='fwas',
                        fisher_info=fisher_info,
                        fwas_lambda=f_lam,
                        device=device
                    )
                    scores = evaluate_multi_task(merged_model, experts, test_datasets, device)
                    for h in cal_hooks:
                        h.remove()
                    print(f"  FWAS lambda={f_lam} -> Avg Acc: {scores['avg']:.2f}%")
                    if best_score is None or scores['avg'] > best_score['avg']:
                        best_score = scores
                        best_lambda = f_lam
                scores = best_score
                print(f"Method [FWAS] (Best Lambda={best_lambda}) with N={cal_size} -> Avg Acc: {scores['avg']:.2f}%")
            else:
                # Standard activation calibration methods
                merged_model = build_merged_model(experts, method='wa')
                cal_hooks = apply_calibration(
                    merged_model=merged_model,
                    expert_stats=expert_stats,
                    joint_cal_set=joint_cal_set,
                    method=method,
                    fisher_info=fisher_info,
                    fwas_lambda=0.5, # Default regularization
                    device=device
                )
                scores = evaluate_multi_task(merged_model, experts, test_datasets, device)
                # Clean up hooks
                for h in cal_hooks:
                    h.remove()
                print(f"Method [{method.upper()}] with N={cal_size} -> Avg Acc: {scores['avg']:.2f}%")
                    
            sweep_results[method][cal_size] = scores
            
    # Save results to a file
    results_path = "./results.json"
    with open(results_path, "w") as f:
        json.dump(sweep_results, f, indent=4)
    print(f"\nSaved sweep results to {results_path}")
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    
    # Map friendly names and colors
    method_styles = {
        'none': ('No Calibration', 'o-', 'gray'),
        'sp-taac': ('SP-TAAC', 's--', 'blue'),
        'taac': ('TAAC', 'd--', 'orange'),
        'r-taac': ('R-TAAC', 'v-.', 'green'),
        's-tcac': ('S-TCAC', '^-.', 'purple'),
        'reda': ('REDA', 'x:', 'brown'),
        'fwas': ('FWAS (Ours)', '*-', 'red')
    }
    
    for method in methods:
        label, marker, color = method_styles[method]
        avg_accuracies = [sweep_results[method][size]['avg'] for size in cal_sizes]
        plt.plot(cal_sizes, avg_accuracies, marker, label=label, color=color, linewidth=2, markersize=8)
        
    # Plot average expert baseline for reference
    avg_expert = sum(sweep_results['expert_baselines'].values()) / len(sweep_results['expert_baselines'])
    plt.axhline(y=avg_expert, color='black', linestyle=':', label='Average Expert Baseline')
    
    plt.xscale('log')
    plt.xticks(cal_sizes, cal_sizes)
    plt.xlabel('Calibration Set Size ($N$ per task)', fontsize=12)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=12)
    plt.title('Activation Calibration Sweep: Accuracy vs Calibration Size', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    
    plot_path = "./calibration_sweep.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved sweep plot to {plot_path}")

if __name__ == "__main__":
    run_calibration_sweep()
