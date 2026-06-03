import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os
import json
import random
import numpy as np

# Import all necessary components from run_experiments
from run_experiments import (
    get_dataloaders,
    load_or_train_experts,
    merge_experts,
    collect_activation_stats,
    apply_smacs_hooks,
    evaluate_model,
    run_head_adaptation,
    DEVICE
)

def run_head_sweep(datasets, experts, seed=42, calibration_size=128):
    print(f"\n==========================================")
    print(f"Executing Head Hyperparam Sweep - Seed {seed} - N={calibration_size}")
    print(f"==========================================")
    
    # Set random seed for reproducibility in calibration sampling
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    # Sample calibration datasets and collect statistics
    calib_loaders = {}
    stats_pre_expert = {}
    stats_pre_merged = {}
    stats_post_expert = {}
    stats_post_merged = {}
    
    merged_model = merge_experts(experts)
    
    for task in tasks:
        train_data = datasets[task]['train']
        indices = random.sample(range(len(train_data)), calibration_size)
        sub_dataset = Subset(train_data, indices)
        calib_loader = DataLoader(sub_dataset, batch_size=calibration_size, shuffle=False)
        calib_loaders[task] = calib_loader
        
        # Collect statistics for Pre-BN
        stats_pre_expert[task] = collect_activation_stats(experts[task], calib_loader, task, DEVICE, mode="pre_bn")
        stats_pre_merged[task] = collect_activation_stats(merged_model, calib_loader, task, DEVICE, mode="pre_bn")
        
        # Collect statistics for Post-BN
        stats_post_expert[task] = collect_activation_stats(experts[task], calib_loader, task, DEVICE, mode="post_bn")
        stats_post_merged[task] = collect_activation_stats(merged_model, calib_loader, task, DEVICE, mode="post_bn")
        
    results = {}
    
    lrs = [1e-2, 1e-3, 1e-4]
    epochs_list = [5, 10, 20]
    
    for lr in lrs:
        for epochs in epochs_list:
            config_key = f"lr={lr:.0e}_epochs={epochs}"
            results[config_key] = {}
            print(f"\nEvaluating: {config_key}")
            
            # 1. Pure Head-only Adaptation (No calibration)
            accs_head_only = []
            for task in tasks:
                restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task, epochs=epochs, lr=lr)
                acc = evaluate_model(merged_model, datasets[task]['test'], task)
                accs_head_only.append(acc)
                restore_fn()
            avg_head_only = sum(accs_head_only) / len(accs_head_only)
            results[config_key]["Head-only Adaptation"] = {
                "mnist": accs_head_only[0],
                "fmnist": accs_head_only[1],
                "cifar10": accs_head_only[2],
                "average": avg_head_only
            }
            print(f"  Head-only: {avg_head_only:.2f}%")
            
            # 2. Pre-BN SMACS (tau=0.50) + Head Adaptation
            accs_pre_smacs = []
            for task in tasks:
                # Apply SMACS Pre-BN hooks
                handles = apply_smacs_hooks(merged_model, stats_pre_expert[task], stats_pre_merged[task], tau=0.50, mode="pre_bn")
                restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task, epochs=epochs, lr=lr)
                acc = evaluate_model(merged_model, datasets[task]['test'], task)
                accs_pre_smacs.append(acc)
                restore_fn()
                for h in handles:
                    h.remove()
            avg_pre_smacs = sum(accs_pre_smacs) / len(accs_pre_smacs)
            results[config_key]["Pre-BN SMACS (tau=0.50) + Head Adapt"] = {
                "mnist": accs_pre_smacs[0],
                "fmnist": accs_pre_smacs[1],
                "cifar10": accs_pre_smacs[2],
                "average": avg_pre_smacs
            }
            print(f"  Pre-BN SMACS + Head: {avg_pre_smacs:.2f}%")
            
            # 3. Post-BN LSC + Head Adaptation
            accs_post_lsc = []
            for task in tasks:
                # Apply LSC Post-BN hooks (tau=1.1)
                handles = apply_smacs_hooks(merged_model, stats_post_expert[task], stats_post_merged[task], tau=1.1, mode="post_bn")
                restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task, epochs=epochs, lr=lr)
                acc = evaluate_model(merged_model, datasets[task]['test'], task)
                accs_post_lsc.append(acc)
                restore_fn()
                for h in handles:
                    h.remove()
            avg_post_lsc = sum(accs_post_lsc) / len(accs_post_lsc)
            results[config_key]["Post-BN LSC + Head Adapt"] = {
                "mnist": accs_post_lsc[0],
                "fmnist": accs_post_lsc[1],
                "cifar10": accs_post_lsc[2],
                "average": avg_post_lsc
            }
            print(f"  Post-BN LSC + Head: {avg_post_lsc:.2f}%")
            
            # 4. Post-BN SMACS (tau=0.95) + Head Adaptation
            accs_post_smacs = []
            for task in tasks:
                # Apply SMACS Post-BN hooks
                handles = apply_smacs_hooks(merged_model, stats_post_expert[task], stats_post_merged[task], tau=0.95, mode="post_bn")
                restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task, epochs=epochs, lr=lr)
                acc = evaluate_model(merged_model, datasets[task]['test'], task)
                accs_post_smacs.append(acc)
                restore_fn()
                for h in handles:
                    h.remove()
            avg_post_smacs = sum(accs_post_smacs) / len(accs_post_smacs)
            results[config_key]["Post-BN SMACS (tau=0.95) + Head Adapt"] = {
                "mnist": accs_post_smacs[0],
                "fmnist": accs_post_smacs[1],
                "cifar10": accs_post_smacs[2],
                "average": avg_post_smacs
            }
            print(f"  Post-BN SMACS + Head: {avg_post_smacs:.2f}%")
            
    return results

if __name__ == "__main__":
    datasets = get_dataloaders()
    experts = load_or_train_experts(datasets)
    
    seeds = [42, 43, 44]
    results_dict = {}
    
    for seed in seeds:
        results_dict[str(seed)] = run_head_sweep(datasets, experts, seed=seed, calibration_size=128)
        
        # Save incrementally
        with open("results_head_hyperparams.json", "w") as f:
            json.dump(results_dict, f, indent=4)
            
    print("\nHead hyperparameter sweep complete! Results saved to results_head_hyperparams.json.")
