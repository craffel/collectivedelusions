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
    calibrate_model,
    merge_models_wa
)

def get_biased_cal_loader(base_cal_dataset, task, num_samples=128):
    """Filter the calibration subset to only contain specific classes."""
    indices = []
    for idx in range(len(base_cal_dataset)):
        _, y = base_cal_dataset[idx]
        if task in ['mnist', 'fmnist']:
            # Even classes only (0, 2, 4, 6, 8)
            if y % 2 == 0:
                indices.append(idx)
        elif task == 'cifar':
            # First 5 classes only (0, 1, 2, 3, 4)
            if y < 5:
                indices.append(idx)
                
    # Select up to num_samples
    biased_indices = indices[:num_samples]
    biased_subset = Subset(base_cal_dataset, biased_indices)
    return DataLoader(biased_subset, batch_size=num_samples, shuffle=False)

def evaluate_by_class_split(model, loader, task, device):
    """Evaluate accuracy separately on In-Distribution (ID) and Out-of-Distribution (OOD) classes."""
    model.eval()
    id_correct = 0
    id_total = 0
    ood_correct = 0
    ood_total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            
            for pred, target in zip(predicted, y):
                is_id = False
                if task in ['mnist', 'fmnist']:
                    if target.item() % 2 == 0:
                        is_id = True
                elif task == 'cifar':
                    if target.item() < 5:
                        is_id = True
                
                if is_id:
                    id_total += 1
                    if pred == target:
                        id_correct += 1
                else:
                    ood_total += 1
                    if pred == target:
                        ood_correct += 1
                        
    id_acc = 100.0 * id_correct / id_total if id_total > 0 else 0.0
    ood_acc = 100.0 * ood_correct / ood_total if ood_total > 0 else 0.0
    total_acc = 100.0 * (id_correct + ood_correct) / (id_total + ood_total) if (id_total + ood_total) > 0 else 0.0
    return id_acc, ood_acc, total_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Class Bias Generalization Analysis on: {device}")
    
    set_seed(42)
    loaders = get_dataloaders()
    init_model = create_model().to(device)

    scenarios_config = {
        'A_low_reg': {'weight_decay': 0.0, 'l2_sp_lambda': 0.0},
        'C_high_reg': {'weight_decay': 1e-2, 'l2_sp_lambda': 0.0}
    }
    
    scenarios_experts = {}
    for sc_name, config in scenarios_config.items():
        print(f"\nTraining experts for Class Bias Generalization Audit ({sc_name})...")
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

    # Build biased loaders
    biased_loaders = copy.deepcopy(loaders)
    for task in ['mnist', 'fmnist', 'cifar']:
        biased_loaders[task]['cal'] = get_biased_cal_loader(
            loaders[task]['cal'].dataset,
            task,
            num_samples=128
        )
        print(f"Task: {task} | Biased calibration size: {len(biased_loaders[task]['cal'].dataset)}")

    results = {}
    
    for sc_name in scenarios_experts.keys():
        experts = scenarios_experts[sc_name]
        results[sc_name] = {}
        
        # Create uncalibrated merged model
        print(f"\nMerging experts via Weight Averaging for {sc_name}...")
        merged_base = merge_models_wa(experts, init_model).to(device)
        
        # Evaluate Uncalibrated
        uncal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            merged_base.fc = experts[task].fc
            id_acc, ood_acc, total_acc = evaluate_by_class_split(merged_base, loaders[task]['test'], task, device)
            uncal_accs[task] = {
                'id_acc': id_acc,
                'ood_acc': ood_acc,
                'total_acc': total_acc
            }
        print(f"  Uncalibrated Average:")
        for t in ['mnist', 'fmnist', 'cifar']:
            print(f"    {t.upper()} - ID: {uncal_accs[t]['id_acc']:.2f}%, OOD: {uncal_accs[t]['ood_acc']:.2f}%, Total: {uncal_accs[t]['total_acc']:.2f}%")
            
        results[sc_name]['Uncalibrated'] = uncal_accs

        # Test configurations
        configs = [
            {'name': 'Unbiased_reg_0.1', 'loaders': loaders, 'reg': 0.1},
            {'name': 'Unbiased_reg_1.0', 'loaders': loaders, 'reg': 1.0},
            {'name': 'Biased_reg_0.1', 'loaders': biased_loaders, 'reg': 0.1},
            {'name': 'Biased_reg_1.0', 'loaders': biased_loaders, 'reg': 1.0},
            {'name': 'Biased_reg_5.0', 'loaders': biased_loaders, 'reg': 5.0}
        ]
        
        for cfg in configs:
            cfg_name = cfg['name']
            reg_val = cfg['reg']
            test_loaders = cfg['loaders']
            print(f"\nEvaluating Post-hoc Hybrid Calibration ({cfg_name})...")
            
            try:
                merged_cal = copy.deepcopy(merged_base)
                merged_cal = calibrate_model(
                    merged=merged_cal,
                    experts=experts,
                    cal_loaders=test_loaders,
                    device=device,
                    method='hybrid',
                    r=8,
                    reg=reg_val
                )
                
                cal_accs = {}
                for task in ['mnist', 'fmnist', 'cifar']:
                    merged_cal.fc = experts[task].fc
                    id_acc, ood_acc, total_acc = evaluate_by_class_split(merged_cal, loaders[task]['test'], task, device)
                    cal_accs[task] = {
                        'id_acc': id_acc,
                        'ood_acc': ood_acc,
                        'total_acc': total_acc
                    }
                print(f"  Calibrated Average:")
                for t in ['mnist', 'fmnist', 'cifar']:
                    print(f"    {t.upper()} - ID: {cal_accs[t]['id_acc']:.2f}%, OOD: {cal_accs[t]['ood_acc']:.2f}%, Total: {cal_accs[t]['total_acc']:.2f}%")
                
                results[sc_name][cfg_name] = cal_accs
            except Exception as e:
                print(f"  Calibration failed: {e}")
                results[sc_name][cfg_name] = "Failed"
                
    # Save the results
    with open('class_bias_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults successfully saved to class_bias_results.json!")

if __name__ == '__main__':
    main()