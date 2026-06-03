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

def run_experiments_pipeline_with_n(datasets, experts, seed=42, mode="post_bn", calibration_size=128):
    mode_label = "Post-BN" if mode == "post_bn" else "Pre-BN"
    print(f"\n==========================================")
    print(f"Executing Sweep - Seed {seed} ({mode_label}) - N={calibration_size}")
    print(f"==========================================")
    
    # Set random seed for this run's calibration subset sampling
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    merged_model = merge_experts(experts)
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    # Sample calibration datasets
    calib_loaders = {}
    calib_stats_expert = {}
    calib_stats_merged = {}
    
    for task in tasks:
        train_data = datasets[task]['train']
        # Randomly sample N indices
        indices = random.sample(range(len(train_data)), calibration_size)
        sub_dataset = Subset(train_data, indices)
        calib_loader = DataLoader(sub_dataset, batch_size=calibration_size, shuffle=False)
        calib_loaders[task] = calib_loader
        
        # Collect statistics
        print(f"Collecting calibration stats for {task.upper()} (Seed {seed}, {mode_label}, N={calibration_size})...")
        calib_stats_expert[task] = collect_activation_stats(experts[task], calib_loader, task, DEVICE, mode=mode)
        calib_stats_merged[task] = collect_activation_stats(merged_model, calib_loader, task, DEVICE, mode=mode)
        
    results = {}
    
    # Sweep of threshold tau (representing SMACS)
    tau_sweep = [1.1, 0.95, 0.90, 0.70, 0.50, 0.30, 0.10, -0.1]
    
    for tau in tau_sweep:
        if tau > 1.0:
            method_name = f"{mode_label} LSC"
        elif tau < 0:
            method_name = f"{mode_label} TCAC/SAC"
        else:
            method_name = f"{mode_label} SMACS (tau={tau:.2f})"
            
        results[method_name] = {}
        print(f"Evaluating {method_name}...")
        
        accs = []
        for task in tasks:
            # Apply SMACS hooks
            handles = apply_smacs_hooks(merged_model, calib_stats_expert[task], calib_stats_merged[task], tau, mode=mode)
            
            # Evaluate test set
            acc = evaluate_model(merged_model, datasets[task]['test'], task)
            results[method_name][task] = acc
            accs.append(acc)
            
            # Remove hooks
            for h in handles:
                h.remove()
                
        results[method_name]['average'] = sum(accs) / len(accs)
        print(f"--> Average Acc: {results[method_name]['average']:.2f}%")
        
    # Evaluate Uncalibrated Model
    results['Uncalibrated'] = {}
    accs = []
    for task in tasks:
        acc = evaluate_model(merged_model, datasets[task]['test'], task)
        results['Uncalibrated'][task] = acc
        accs.append(acc)
    results['Uncalibrated']['average'] = sum(accs) / len(accs)
    
    # Evaluate Head-only Adaptation
    print(f"Evaluating Head-only Adaptation (N={calibration_size})...")
    results['Head-only Adaptation'] = {}
    accs = []
    for task in tasks:
        restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task)
        acc = evaluate_model(merged_model, datasets[task]['test'], task)
        results['Head-only Adaptation'][task] = acc
        accs.append(acc)
        restore_fn() # Restore classification head and backbone requires_grad
    results['Head-only Adaptation']['average'] = sum(accs) / len(accs)
    print(f"--> Head-only Adaptation Average Acc: {results['Head-only Adaptation']['average']:.2f}%")
    
    # Evaluate Joint SMACS (best tau) + Head Adaptation
    # First, let's identify the best tau from the sweep
    best_tau = None
    best_avg_acc = 0.0
    for tau in tau_sweep:
        if 0 <= tau <= 1.0:
            method_name = f"{mode_label} SMACS (tau={tau:.2f})"
            if results[method_name]['average'] > best_avg_acc:
                best_avg_acc = results[method_name]['average']
                best_tau = tau
                
    if best_tau is not None:
        joint_method_name = f"{mode_label} SMACS (tau={best_tau:.2f}) + Head Adaptation"
        print(f"Evaluating Joint {joint_method_name}...")
        results[joint_method_name] = {}
        accs = []
        for task in tasks:
            # Apply SMACS hooks
            handles = apply_smacs_hooks(merged_model, calib_stats_expert[task], calib_stats_merged[task], best_tau, mode=mode)
            # Run Head Adaptation
            restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task)
            
            acc = evaluate_model(merged_model, datasets[task]['test'], task)
            results[joint_method_name][task] = acc
            accs.append(acc)
            
            # Cleanup
            restore_fn()
            for h in handles:
                h.remove()
        results[joint_method_name]['average'] = sum(accs) / len(accs)
        print(f"--> Joint Average Acc: {results[joint_method_name]['average']:.2f}%")
        
    return results

if __name__ == "__main__":
    datasets = get_dataloaders()
    experts = load_or_train_experts(datasets)
    
    seeds = [42, 43, 44]
    n_values = [32, 128, 512]
    
    sweep_results = {}
    
    for n in n_values:
        sweep_results[str(n)] = {}
        for seed in seeds:
            sweep_results[str(n)][str(seed)] = {}
            
            # Run pre_bn and post_bn calibration pipelines
            res_pre = run_experiments_pipeline_with_n(datasets, experts, seed, mode="pre_bn", calibration_size=n)
            res_post = run_experiments_pipeline_with_n(datasets, experts, seed, mode="post_bn", calibration_size=n)
            
            # Merge the dictionaries
            combined_res = {}
            combined_res.update(res_pre)
            combined_res.update(res_post)
            
            sweep_results[str(n)][str(seed)] = combined_res
            
            # Save progress incrementally so we don't lose anything if interrupted
            with open("results_n_sweep.json", "w") as f:
                json.dump(sweep_results, f, indent=4)
                
    print("\nN Sweep complete! Results saved to results_n_sweep.json.")
