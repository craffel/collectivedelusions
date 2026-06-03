import sys
import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on the GPU node
torch.backends.cudnn.enabled = False

from data import get_multi_task_datasets, get_calibration_subset
from methods import (
    get_merged_state_dict,
    apply_slr_wbc,
)
from evaluate import load_expert_model, evaluate_model_on_task

def run_slr_wbc_sweep(device='cuda'):
    print("==================================================")
    print("🚀 Running SLR-WBC Hyperparameter Sweep 🚀")
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
    
    wa_state_dict = get_merged_state_dict(expert_state_dicts, mode='wa')
    wa_model = copy.deepcopy(base_model).to(device)
    wa_model.load_state_dict(wa_state_dict)
    
    N_sweep = 128
    seed_sweep = 42
    
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed_sweep)
    test_loaders = {t: DataLoader(test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
    cal_subsets = {t: get_calibration_subset(train_datasets[t], N_sweep, seed=seed_sweep) for t in tasks}
    cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
    
    # Pool samples from all tasks to form joint calibration batches
    joint_cal_samples = []
    for t in tasks:
        for inputs, _ in cal_loaders[t]:
            joint_cal_samples.append(inputs)
    joint_cal_batches = [torch.cat(joint_cal_samples, dim=0)]
    
    em_list = [expert_models[t] for t in tasks]
    
    ranks = [1, 2, 4, 8]
    regs = [0.01, 0.1, 0.5]
    
    sweep_results = {}
    
    for rank in ranks:
        sweep_results[str(rank)] = {}
        for reg in regs:
            print(f"\nEvaluating Joint SLR-WBC with Rank r={rank}, Reg={reg}...")
            
            # Copy merged model to calibrate
            slr_model = copy.deepcopy(wa_model)
            
            # Apply SVD-based Joint Calibration
            apply_slr_wbc(slr_model, em_list, joint_cal_batches, rank=rank, reg=reg, device=device)
            
            # Evaluate across tasks
            slr_accs = {}
            for t in tasks:
                acc = evaluate_model_on_task(slr_model, test_loaders[t], expert_heads[t], device=device)
                slr_accs[t] = acc
                
            avg_acc = sum(slr_accs.values()) / 3.0
            print(f"  MNIST: {slr_accs['mnist']:.2f}%, F-MNIST: {slr_accs['fmnist']:.2f}%, CIFAR-10: {slr_accs['cifar10']:.2f}%")
            print(f"  ==> Average Accuracy: {avg_acc:.2f}%")
            
            sweep_results[str(rank)][str(reg)] = avg_acc
            
    # Write results to json
    output_path = 'slr_wbc_sweep_results.json'
    with open(output_path, 'w') as f:
        json.dump(sweep_results, f, indent=4)
        
    print(f"\n✅ Sweep completed! Results written to {output_path}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_slr_wbc_sweep(device=device)
