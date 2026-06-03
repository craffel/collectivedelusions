import os
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Import helpers from additional_analysis
from additional_analysis import (
    set_seed,
    get_dataloaders,
    create_model,
    train_expert,
    evaluate,
    calibrate_model
)
from train_eval import analyze_drift

def merge_models_ta(experts, init_model, lmbda):
    """Merges models using Task Arithmetic with a scaling parameter lambda."""
    merged = create_model()
    merged_state = merged.state_dict()
    expert_states = {t: expert.state_dict() for t, expert in experts.items()}
    init_state = init_model.state_dict()

    for name in merged_state.keys():
        if 'fc' in name:
            continue
        if torch.is_floating_point(merged_state[name]) or torch.is_complex(merged_state[name]):
            # W_merged = W_init + lmbda * sum_k (W_k - W_init)
            task_updates = [expert_states[t][name] - init_state[name] for t in experts.keys()]
            merged_state[name] = init_state[name] + lmbda * torch.stack(task_updates).sum(dim=0)
        else:
            first_task = list(experts.keys())[0]
            merged_state[name] = expert_states[first_task][name].clone()

    merged.load_state_dict(merged_state)
    return merged

def run_robustness_and_scaling(device):
    print("=====================================================================")
    print("Starting Methodological Robustness and Scaling Audit...")
    print("=====================================================================")

    # 1. Task Vector Scaling Baseline Audit
    # We will use the same loaders and seed as the main experiments (seed=42)
    set_seed(42)
    loaders = get_dataloaders()
    init_model = create_model().to(device)

    # Train experts for Scenario A (Low Reg) and Scenario C (High Reg)
    scenarios_config = {
        'A_low_reg': {'weight_decay': 0.0, 'l2_sp_lambda': 0.0},
        'C_high_reg': {'weight_decay': 1e-2, 'l2_sp_lambda': 0.0}
    }
    
    scenarios_experts = {}
    for sc_name, config in scenarios_config.items():
        print(f"\nTraining experts for Scaling Audit ({sc_name})...")
        experts = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            set_seed(42)
            model = create_model()
            model = train_expert(
                model=model,
                init_model=init_model,
                train_loader=loaders[task]['train'],
                device=device,
                epochs=5,
                lr=1e-4,
                weight_decay=config['weight_decay'],
                l2_sp_lambda=config['l2_sp_lambda']
            )
            experts[task] = model
        scenarios_experts[sc_name] = experts

    # Sweep lambda
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scaling_results = {}

    for sc_name in scenarios_experts.keys():
        experts = scenarios_experts[sc_name]
        sc_results = []
        for lmbda in lambdas:
            print(f"Scaling Audit {sc_name} | lambda={lmbda}")
            merged_base = merge_models_ta(experts, init_model, lmbda)
            
            # Evaluate uncalibrated
            uncal_accs = []
            for task in ['mnist', 'fmnist', 'cifar']:
                merged_base.fc = experts[task].fc
                uncal_accs.append(evaluate(merged_base, loaders[task]['test'], device))
            uncal_avg = sum(uncal_accs) / len(uncal_accs)
            
            # Evaluate Calibrated (r=4, reg=0.5)
            try:
                merged_cal = copy.deepcopy(merged_base)
                merged_cal = calibrate_model(
                    merged=merged_cal,
                    experts=experts,
                    cal_loaders=loaders,
                    device=device,
                    method='hybrid',
                    r=4,
                    reg=0.5
                )
                cal_accs = []
                for task in ['mnist', 'fmnist', 'cifar']:
                    merged_cal.fc = experts[task].fc
                    cal_accs.append(evaluate(merged_cal, loaders[task]['test'], device))
                cal_avg = sum(cal_accs) / len(cal_accs)
            except Exception as e:
                print(f"  Calibrated (r=4, reg=0.5) failed due to instability: {e}")
                cal_avg = 0.0

            # Evaluate Calibrated (r=8, reg=0.1)
            try:
                merged_cal_opt = copy.deepcopy(merged_base)
                merged_cal_opt = calibrate_model(
                    merged=merged_cal_opt,
                    experts=experts,
                    cal_loaders=loaders,
                    device=device,
                    method='hybrid',
                    r=8,
                    reg=0.1
                )
                cal_opt_accs = []
                for task in ['mnist', 'fmnist', 'cifar']:
                    merged_cal_opt.fc = experts[task].fc
                    cal_opt_accs.append(evaluate(merged_cal_opt, loaders[task]['test'], device))
                cal_opt_avg = sum(cal_opt_accs) / len(cal_opt_accs)
            except Exception as e:
                print(f"  Calibrated (r=8, reg=0.1) failed due to instability: {e}")
                cal_opt_avg = 0.0
            
            print(f"  Uncalibrated Avg Acc: {uncal_avg:.2f}% | Calibrated (r=4, reg=0.5): {cal_avg:.2f}% | Calibrated (r=8, reg=0.1): {cal_opt_avg:.2f}%")
            sc_results.append({
                'lambda': lmbda,
                'uncalibrated': uncal_avg,
                'calibrated_r4_reg0.5': cal_avg,
                'calibrated_r8_reg0.1': cal_opt_avg
            })
        scaling_results[sc_name] = sc_results

    # 2. Multi-Seed Robustness Evaluation (seeds = 42, 43, 44)
    # We will compute the mean and std of cosine similarity and accuracy
    seeds = [42, 43, 44]
    robustness_results = {}

    for sc_name, config in scenarios_config.items():
        print(f"\n=================================================")
        print(f"Running Multi-Seed Robustness for {sc_name}")
        print(f"=================================================")
        
        seed_drift_similarities = []
        seed_uncal_accuracies = []
        seed_cal_r4_accuracies = []
        seed_cal_r8_accuracies = []
        
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)
            # Dataloaders under this seed (subset splits differ by seed)
            seed_loaders = get_dataloaders()
            seed_init_model = create_model().to(device)
            
            # Train experts
            seed_experts = {}
            for task in ['mnist', 'fmnist', 'cifar']:
                set_seed(seed)
                model = create_model()
                model = train_expert(
                    model=model,
                    init_model=seed_init_model,
                    train_loader=seed_loaders[task]['train'],
                    device=device,
                    epochs=5,
                    lr=1e-4,
                    weight_decay=config['weight_decay'],
                    l2_sp_lambda=config['l2_sp_lambda']
                )
                seed_experts[task] = model
            
            # Analyze update similarity (cosine similarity between tasks)
            drift = analyze_drift(seed_experts, seed_init_model)
            # Average of the three pairwise similarities
            pairwise_sims = [
                drift['cos_sim_mnist_fmnist'],
                drift['cos_sim_mnist_cifar'],
                drift['cos_sim_fmnist_cifar']
            ]
            avg_sim = sum(pairwise_sims) / len(pairwise_sims)
            seed_drift_similarities.append(avg_sim)
            print(f"  Average Update Cosine Similarity: {avg_sim:.4f}")
            
            # Merge models (Weight Averaging)
            seed_merged_base = merge_models_ta(seed_experts, seed_init_model, 1.0 / 3.0)
            
            # Evaluate uncalibrated
            uncal_accs = []
            for task in ['mnist', 'fmnist', 'cifar']:
                seed_merged_base.fc = seed_experts[task].fc
                uncal_accs.append(evaluate(seed_merged_base, seed_loaders[task]['test'], device))
            uncal_avg = sum(uncal_accs) / len(uncal_accs)
            seed_uncal_accuracies.append(uncal_avg)
            print(f"  Uncalibrated Avg Acc: {uncal_avg:.2f}%")
            
            # Calibrate and evaluate (r=4, reg=0.5)
            seed_merged_cal = copy.deepcopy(seed_merged_base)
            seed_merged_cal = calibrate_model(
                merged=seed_merged_cal,
                experts=seed_experts,
                cal_loaders=seed_loaders,
                device=device,
                method='hybrid',
                r=4,
                reg=0.5
            )
            cal_accs = []
            for task in ['mnist', 'fmnist', 'cifar']:
                seed_merged_cal.fc = seed_experts[task].fc
                cal_accs.append(evaluate(seed_merged_cal, seed_loaders[task]['test'], device))
            cal_avg = sum(cal_accs) / len(cal_accs)
            seed_cal_r4_accuracies.append(cal_avg)
            print(f"  Calibrated (r=4, reg=0.5) Avg Acc: {cal_avg:.2f}%")

            # Calibrate and evaluate (r=8, reg=0.1)
            seed_merged_cal_opt = copy.deepcopy(seed_merged_base)
            seed_merged_cal_opt = calibrate_model(
                merged=seed_merged_cal_opt,
                experts=seed_experts,
                cal_loaders=seed_loaders,
                device=device,
                method='hybrid',
                r=8,
                reg=0.1
            )
            cal_opt_accs = []
            for task in ['mnist', 'fmnist', 'cifar']:
                seed_merged_cal_opt.fc = seed_experts[task].fc
                cal_opt_accs.append(evaluate(seed_merged_cal_opt, seed_loaders[task]['test'], device))
            cal_opt_avg = sum(cal_opt_accs) / len(cal_opt_accs)
            seed_cal_r8_accuracies.append(cal_opt_avg)
            print(f"  Calibrated (r=8, reg=0.1) Avg Acc: {cal_opt_avg:.2f}%")
            
        robustness_results[sc_name] = {
            'sim_mean': float(np.mean(seed_drift_similarities)),
            'sim_std': float(np.std(seed_drift_similarities)),
            'uncal_mean': float(np.mean(seed_uncal_accuracies)),
            'uncal_std': float(np.std(seed_uncal_accuracies)),
            'cal_r4_mean': float(np.mean(seed_cal_r4_accuracies)),
            'cal_r4_std': float(np.std(seed_cal_r4_accuracies)),
            'cal_r8_mean': float(np.mean(seed_cal_r8_accuracies)),
            'cal_r8_std': float(np.std(seed_cal_r8_accuracies)),
            'sims_all': seed_drift_similarities,
            'uncal_all': seed_uncal_accuracies,
            'cal_r4_all': seed_cal_r4_accuracies,
            'cal_r8_all': seed_cal_r8_accuracies
        }

    # Save all results
    all_results = {
        'scaling_audit': scaling_results,
        'robustness_audit': robustness_results
    }
    
    with open('robustness_and_scaling_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\n=============================================================")
    print("Methodological Robustness and Scaling Audit completed successfully!")
    print("Results saved to robustness_and_scaling_results.json")
    print("=============================================================")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running robustness and scaling on device:", device)
    run_robustness_and_scaling(device)
