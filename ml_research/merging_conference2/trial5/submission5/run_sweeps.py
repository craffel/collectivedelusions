import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import os
import json
import numpy as np
import copy
from data import get_splits
from calibration import register_hooks, merge_models, fuse_calibration_to_bn
from evaluate import evaluate_model, collect_expert_stats, run_calibration, copy_model, load_expert

def run_comprehensive_sweeps(device='cuda'):
    print("Initializing comprehensive empirical sweeps...")
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    
    # Load base model template
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    base_state_dict = base_model.state_dict()
    
    # Load expert models
    expert_models = {name: load_expert(name, device=device) for name in datasets}
    expert_state_dicts = {name: model.state_dict() for name, model in expert_models.items()}
    
    # Sweep variables
    N_list = [16, 64, 128]
    lambda_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    calibration_methods = ['None', 'SP-TAAC', 'TAAC', 'ZIO-CF', 'FDSA', 'JSSC']
    
    sweep_results = {}
    
    for N in N_list:
        print(f"\n==================== SWEEP: Calibration Size N = {N} ====================")
        sweep_results[N] = {}
        
        # Sliced calibration datasets for this N
        cal_datasets = {}
        for name in datasets:
            _, full_cal_ds, _ = get_splits(name)
            cal_datasets[name] = Subset(full_cal_ds, list(range(N)))
            
        # Collect expert statistics for this specific N
        target_spatial, target_spectral = collect_expert_stats(expert_models, cal_datasets, device=device)
        
        # 1. Weight Averaging (WA) Sweep
        print("\n--- Running sweeps for Weight Averaging (WA) ---")
        sweep_results[N]['WA'] = {}
        
        merged_state_wa = merge_models([e.state_dict() for e in expert_models.values()], mode='WA')
        merged_wa_template = models.resnet18()
        merged_wa_template.fc = nn.Linear(512, 10)
        merged_wa_template.load_state_dict(merged_state_wa)
        merged_wa_template = merged_wa_template.to(device)
        
        for cal_method in calibration_methods:
            print(f"Evaluating WA + {cal_method} (N={N})...")
            cal_model, handles = run_calibration(
                merged_wa_template, target_spatial, target_spectral, cal_datasets, mode=cal_method, device=device
            )
            
            task_accs = {}
            with torch.no_grad():
                for task_name in datasets:
                    # Save original classification head
                    orig_weight = cal_model.fc.weight.data.clone()
                    orig_bias = cal_model.fc.bias.data.clone()
                    
                    # Overwrite with task-specific expert head
                    cal_model.fc.weight.copy_(expert_state_dicts[task_name]['fc.weight'])
                    cal_model.fc.bias.copy_(expert_state_dicts[task_name]['fc.bias'])
                    
                    acc = evaluate_model(cal_model, task_name, device=device)
                    task_accs[task_name] = acc
                    
                    # Restore original head
                    cal_model.fc.fc_weight_restored = cal_model.fc.weight.copy_(orig_weight)
                    cal_model.fc.fc_bias_restored = cal_model.fc.bias.copy_(orig_bias)
                
            avg_acc = np.mean(list(task_accs.values()))
            print(f"  Avg Accuracy: {avg_acc:.2f}%")
            
            sweep_results[N]['WA'][cal_method] = {
                'tasks': task_accs,
                'average': avg_acc
            }
            
            for h in handles:
                h.remove()
                
        # 2. Task Arithmetic (TA) Sweep over Lambda
        sweep_results[N]['TA'] = {}
        for lam in lambda_list:
            print(f"\n--- Running sweeps for Task Arithmetic (TA) with lambda = {lam} ---")
            sweep_results[N]['TA'][lam] = {}
            
            merged_state_ta = merge_models(
                [e.state_dict() for e in expert_models.values()], 
                base_state_dict=base_state_dict, 
                mode='TA', 
                lambda_val=lam
            )
            merged_ta_template = models.resnet18()
            merged_ta_template.fc = nn.Linear(512, 10)
            merged_ta_template.load_state_dict(merged_state_ta)
            merged_ta_template = merged_ta_template.to(device)
            
            for cal_method in calibration_methods:
                print(f"Evaluating TA (lambda={lam}) + {cal_method} (N={N})...")
                cal_model, handles = run_calibration(
                    merged_ta_template, target_spatial, target_spectral, cal_datasets, mode=cal_method, device=device
                )
                
                task_accs = {}
                with torch.no_grad():
                    for task_name in datasets:
                        # Save original classification head
                        orig_weight = cal_model.fc.weight.data.clone()
                        orig_bias = cal_model.fc.bias.data.clone()
                        
                        # Overwrite with task-specific expert head
                        cal_model.fc.weight.copy_(expert_state_dicts[task_name]['fc.weight'])
                        cal_model.fc.bias.copy_(expert_state_dicts[task_name]['fc.bias'])
                        
                        acc = evaluate_model(cal_model, task_name, device=device)
                        task_accs[task_name] = acc
                        
                        # Restore original head
                        cal_model.fc.fc_weight_restored = cal_model.fc.weight.copy_(orig_weight)
                        cal_model.fc.fc_bias_restored = cal_model.fc.bias.copy_(orig_bias)
                    
                avg_acc = np.mean(list(task_accs.values()))
                print(f"  Avg Accuracy: {avg_acc:.2f}%")
                
                sweep_results[N]['TA'][lam][cal_method] = {
                    'tasks': task_accs,
                    'average': avg_acc
                }
                
                for h in handles:
                    h.remove()
                    
    # Save sweep results to json
    with open('sweep_results.json', 'w') as f:
        json.dump(sweep_results, f, indent=4)
    print("\nSweep results successfully saved to sweep_results.json")

if __name__ == '__main__':
    run_comprehensive_sweeps(device='cuda' if torch.cuda.is_available() else 'cpu')
