import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from data import get_multi_task_datasets, get_calibration_subset
from methods import (
    get_merged_state_dict,
    apply_slr_wbc,
)
from evaluate import load_expert_model, evaluate_model_on_task
from sweep import get_task_prototypes_custom, ssr_merge_eval_custom

def run_ta_scaling_sweep(device='cuda'):
    print("==================================================")
    print("🚀 RUNNING TASK ARITHMETIC SCALING SWEEP (LAMBDA) 🚀")
    print("==================================================")
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = {}
    expert_heads = {}
    expert_state_dicts = []
    
    for t in tasks:
        model = load_expert_model(t, device=device)
        expert_models[t] = model
        expert_heads[t] = copy.deepcopy(model.fc)
        expert_state_dicts.append(copy.deepcopy(model.state_dict()))
        
    try:
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        base_model = resnet18(pretrained=True)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        base_model.fc
    )
    base_state_dict = copy.deepcopy(base_model.state_dict())
    
    # Use seed 42 and N=128 for calibration
    seed = 42
    N = 128
    
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed)
    test_loaders = {t: DataLoader(test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
    
    cal_subsets = {t: get_calibration_subset(train_datasets[t], N, seed=seed) for t in tasks}
    cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
    
    # We will test lambdas from 0.1 to 0.5
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    for lam in lambdas:
        print(f"\nEvaluating scaling factor lambda = {lam}...")
        
        # 1. Uncalibrated Task Arithmetic Model
        ta_state_dict = get_merged_state_dict(expert_state_dicts, mode='ta', lam=lam, base_state_dict=base_state_dict)
        ta_model = copy.deepcopy(base_model).to(device)
        ta_model.load_state_dict(ta_state_dict)
        
        ta_accs = {}
        for t in tasks:
            acc = evaluate_model_on_task(ta_model, test_loaders[t], expert_heads[t], device=device)
            ta_accs[t] = acc
        ta_avg = sum(ta_accs.values()) / 3.0
        print(f"  Uncalibrated TA Average Accuracy: {ta_avg:.2f}% (MNIST: {ta_accs['mnist']:.2f}%, F-MNIST: {ta_accs['fmnist']:.2f}%, CIFAR-10: {ta_accs['cifar10']:.2f}%)")
        
        # 2. SSR-Merge (Ours) on top of this TA model
        try:
            # Extract task prototypes from Layer 2
            prototypes = get_task_prototypes_custom(ta_model, cal_loaders, layer_name='layer2', normalize=True, device=device)
            
            # Calibrate task-specific models using optimal hyperparameters (rank=8, reg=0.01)
            task_specific_models = {}
            for t in tasks:
                ts_model = copy.deepcopy(ta_model)
                task_cal_batches = []
                for inputs, _ in cal_loaders[t]:
                    task_cal_batches.append(inputs)
                task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
                
                # Use optimal hyperparameters (rank=8, reg=0.01) for SSR-Merge
                apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=8, reg=0.01, device=device)
                ts_model.fc = expert_heads[t]
                task_specific_models[t] = ts_model
                
            # Evaluate SSR-Merge (using Manhattan routing)
            ssr_accs = {}
            for t in tasks:
                acc = ssr_merge_eval_custom(ta_model, task_specific_models, prototypes, test_loaders[t], layer_name='layer2', metric='manhattan', device=device)
                ssr_accs[t] = acc
            ssr_avg = sum(ssr_accs.values()) / 3.0
            print(f"  SSR-Merge Average Accuracy: {ssr_avg:.2f}% (MNIST: {ssr_accs['mnist']:.2f}%, F-MNIST: {ssr_accs['fmnist']:.2f}%, CIFAR-10: {ssr_accs['cifar10']:.2f}%)")
            
            results[str(lam)] = {
                'ta_individual': ta_accs,
                'ta_avg': ta_avg,
                'ssr_individual': ssr_accs,
                'ssr_avg': ssr_avg,
                'status': 'Success'
            }
        except Exception as e:
            print(f"  SSR-Merge calibration failed for lambda = {lam}: {str(e)}")
            results[str(lam)] = {
                'ta_individual': ta_accs,
                'ta_avg': ta_avg,
                'ssr_individual': None,
                'ssr_avg': None,
                'status': f"Failed: {type(e).__name__}"
            }
        
    # Save results to JSON
    with open('ta_scaling_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n=========================================")
    print("📊 TA SCALING SWEEP RESULTS SUMMARY 📊")
    print("=========================================")
    print(f"{'Lambda':<10} | {'Uncalibrated TA Avg (%)':<25} | {'SSR-Merge Avg (%)':<20} | {'Absolute Gain (%)':<20}")
    print("-" * 83)
    for lam in lambdas:
        ta_a = results[str(lam)]['ta_avg']
        ssr_a = results[str(lam)]['ssr_avg']
        if ssr_a is not None:
            gain = ssr_a - ta_a
            print(f"{lam:<10.1f} | {ta_a:<25.2f} | {ssr_a:<20.2f} | {gain:<20.2f}")
        else:
            print(f"{lam:<10.1f} | {ta_a:<25.2f} | {'Failed (Singular)':<20} | {'N/A':<20}")
    print("=========================================")
    print("=========================================")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_ta_scaling_sweep(device)
