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

# Import helpers from additional_analysis and train_eval
from additional_analysis import (
    set_seed,
    get_dataloaders,
    create_model,
    train_expert,
    evaluate,
    calibrate_model
)

def merge_models_wa(experts, init_model):
    """Averages expert weights to create a merged model."""
    merged = create_model()
    merged_state = merged.state_dict()
    expert_states = {t: expert.state_dict() for t, expert in experts.items()}
    
    for name in merged_state.keys():
        if 'fc' in name:
            continue
        if torch.is_floating_point(merged_state[name]) or torch.is_complex(merged_state[name]):
            merged_state[name] = torch.stack([expert_states[t][name] for t in experts.keys()]).mean(dim=0)
        else:
            first_task = list(experts.keys())[0]
            merged_state[name] = expert_states[first_task][name].clone()
        
    merged.load_state_dict(merged_state)
    return merged

def merge_models_ties(experts, init_model, p=0.2, scaling=1.0):
    """
    TIES-Merging:
    1. Trim: Keep top-p% updates by magnitude, set the rest to 0.
    2. Elect sign: Sign consensus across active experts.
    3. Merge matching signs.
    """
    merged = create_model()
    merged_state = merged.state_dict()
    expert_states = {t: expert.state_dict() for t, expert in experts.items()}
    init_state = init_model.state_dict()

    for name in merged_state.keys():
        if 'fc' in name:
            continue
        if torch.is_floating_point(merged_state[name]) or torch.is_complex(merged_state[name]):
            task_updates = []
            for t in experts.keys():
                update = expert_states[t][name] - init_state[name]
                flat_update = update.flatten()
                k = max(1, int(p * flat_update.numel()))
                threshold = torch.topk(flat_update.abs(), k).values[-1]
                mask = flat_update.abs() >= threshold
                pruned_flat = flat_update.clone()
                pruned_flat[~mask] = 0.0
                task_updates.append(pruned_flat.view(update.shape))
            
            stacked_updates = torch.stack(task_updates) # (K, *shape)
            signs = torch.sign(stacked_updates) # (K, *shape)
            sum_signs = signs.sum(dim=0) # (*shape)
            consensus_sign = torch.sign(sum_signs) # (*shape)
            
            matching_mask = (signs == consensus_sign.unsqueeze(0)) & (consensus_sign.unsqueeze(0) != 0)
            
            sum_matching = (stacked_updates * matching_mask.float()).sum(dim=0)
            count_matching = matching_mask.float().sum(dim=0)
            
            merged_update = torch.where(count_matching > 0, sum_matching / count_matching, torch.zeros_like(sum_matching))
            merged_state[name] = init_state[name] + scaling * merged_update
        else:
            first_task = list(experts.keys())[0]
            merged_state[name] = expert_states[first_task][name].clone()

    merged.load_state_dict(merged_state)
    return merged

def merge_models_dare(experts, init_model, p=0.8):
    """
    DARE (Drop and Rescale):
    1. Drop: Randomly drop parameters with probability p (set to 0).
    2. Rescale: Scale remaining parameters by 1 / (1 - p).
    """
    merged = create_model()
    merged_state = merged.state_dict()
    expert_states = {t: expert.state_dict() for t, expert in experts.items()}
    init_state = init_model.state_dict()

    for name in merged_state.keys():
        if 'fc' in name:
            continue
        if torch.is_floating_point(merged_state[name]) or torch.is_complex(merged_state[name]):
            task_updates = []
            for t in experts.keys():
                update = expert_states[t][name] - init_state[name]
                mask = torch.rand_like(update) >= p
                pruned = torch.where(mask, update / (1.0 - p), torch.zeros_like(update))
                task_updates.append(pruned)
                
            merged_update = torch.stack(task_updates).mean(dim=0)
            merged_state[name] = init_state[name] + merged_update
        else:
            first_task = list(experts.keys())[0]
            merged_state[name] = expert_states[first_task][name].clone()

    merged.load_state_dict(merged_state)
    return merged

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running TIES and DARE Analysis on: {device}")
    
    set_seed(42)
    loaders = get_dataloaders()
    init_model = create_model().to(device)

    scenarios_config = {
        'A_low_reg': {'weight_decay': 0.0, 'l2_sp_lambda': 0.0},
        'C_high_reg': {'weight_decay': 1e-2, 'l2_sp_lambda': 0.0}
    }
    
    scenarios_experts = {}
    for sc_name, config in scenarios_config.items():
        print(f"\nTraining experts for TIES/DARE Audit ({sc_name})...")
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

    results = {}
    
    for sc_name in scenarios_experts.keys():
        experts = scenarios_experts[sc_name]
        results[sc_name] = {}
        
        # Define the merge types
        merge_fns = {
            'WA': lambda: merge_models_wa(experts, init_model),
            'TIES': lambda: merge_models_ties(experts, init_model, p=0.2, scaling=1.0),
            'DARE': lambda: merge_models_dare(experts, init_model, p=0.8)
        }
        
        for m_name, merge_fn in merge_fns.items():
            print(f"\n--- Scenario: {sc_name} | Merge Method: {m_name} ---")
            merged_base = merge_fn().to(device)
            
            # 1. Evaluate Uncalibrated
            uncal_accs = {}
            for task in ['mnist', 'fmnist', 'cifar']:
                merged_base.fc = experts[task].fc
                uncal_accs[task] = evaluate(merged_base, loaders[task]['test'], device)
            uncal_avg = sum(uncal_accs.values()) / len(uncal_accs)
            print(f"  Uncalibrated Acc: {uncal_avg:.2f}% (MNIST: {uncal_accs['mnist']:.2f}%, F-MNIST: {uncal_accs['fmnist']:.2f}%, CIFAR: {uncal_accs['cifar']:.2f}%)")
            
            # 2. Evaluate Calibrated (r=8, reg=0.1)
            try:
                merged_cal = copy.deepcopy(merged_base)
                merged_cal = calibrate_model(
                    merged=merged_cal,
                    experts=experts,
                    cal_loaders=loaders,
                    device=device,
                    method='hybrid',
                    r=8,
                    reg=0.1
                )
                cal_accs = {}
                for task in ['mnist', 'fmnist', 'cifar']:
                    merged_cal.fc = experts[task].fc
                    cal_accs[task] = evaluate(merged_cal, loaders[task]['test'], device)
                cal_avg = sum(cal_accs.values()) / len(cal_accs)
                print(f"  Calibrated (r=8, reg=0.1) Acc: {cal_avg:.2f}% (MNIST: {cal_accs['mnist']:.2f}%, F-MNIST: {cal_accs['fmnist']:.2f}%, CIFAR: {cal_accs['cifar']:.2f}%)")
            except Exception as e:
                print(f"  Calibration failed: {e}")
                cal_accs = {k: 0.0 for k in ['mnist', 'fmnist', 'cifar']}
                cal_avg = 0.0
                
            results[sc_name][m_name] = {
                'uncalibrated_avg': uncal_avg,
                'uncalibrated': uncal_accs,
                'calibrated_avg': cal_avg,
                'calibrated': cal_accs
            }
            
    # Save the results
    with open('ties_dare_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to ties_dare_results.json successfully!")

if __name__ == '__main__':
    main()
