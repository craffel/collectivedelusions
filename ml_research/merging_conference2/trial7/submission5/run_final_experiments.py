import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models, generate_harmonic_patterns, generate_pink_noise

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calibrate_standard(model, calibration_data, epochs=10, device='cpu'):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = 0.1
    loader = DataLoader(calibration_data, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            _ = model.backbone(x)
    model.eval()

def calibrate_rba(model, calibration_data, epochs=30, momentum=0.2, device='cpu'):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
    loader = DataLoader(calibration_data, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            _ = model.backbone(x)
    model.eval()

def evaluate_on_task(model, test_dataset, task_name, device):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, task_name)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    task_names = ['mnist', 'fmnist', 'cifar10']
    test_datasets_dict = {task: datasets_dict[task][1] for task in task_names}
    
    # Load experts
    expert_state_dicts = {}
    heads_state_dicts = {}
    oracle_accuracies = {}
    for task in task_names:
        checkpoint = torch.load(f"./checkpoints/expert_{task}.pt", map_location=device)
        expert_state_dicts[task] = checkpoint['state_dict']
        heads_state_dicts[task] = checkpoint['head_state_dict']
        oracle_accuracies[task] = checkpoint['accuracy']
        
    progenitor = MultiTaskResNet18().to(device)
    progenitor_state_dict = {f"backbone.{k}": v.cpu().clone() for k, v in progenitor.backbone.state_dict().items()}
    
    # Prepare calibration datasets of 256 samples
    task_real_samples = {}
    for task in task_names:
        train_sub, _ = datasets_dict[task]
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train_sub), generator=g)[:256]
        task_real_samples[task] = torch.stack([train_sub[idx][0] for idx in indices], dim=0)
        
    # Joint real dataset (85 + 85 + 86 = 256 samples)
    joint_samples = torch.cat([
        task_real_samples['mnist'][:85],
        task_real_samples['fmnist'][:85],
        task_real_samples['cifar10'][:86]
    ], dim=0)
    
    configurations = [
        ('WA', 0.5),
        ('TA', 0.3),
        ('TA', 0.5),
        ('TA', 0.7)
    ]
    
    results = {}
    
    for merge_type, lam in configurations:
        cfg_name = f"{merge_type}_{lam}"
        results[cfg_name] = {}
        print(f"\nEvaluating Configuration: {cfg_name}")
        
        # Merge backbone
        merged_backbone = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type=merge_type, lam=lam)
        
        # 1. Uncalibrated Baseline
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone, strict=False)
        cfg_uncal = {}
        for task in task_names:
            model.heads[task].load_state_dict(heads_state_dicts[task])
            cfg_uncal[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        cfg_uncal['average'] = sum(cfg_uncal.values()) / 3
        results[cfg_name]['Uncalibrated'] = cfg_uncal
        print(f"  Uncalibrated: {cfg_uncal['average']:.2f}%")
        
        # 2. Joint Standard Calibration (Reset, 10 epochs, joint real data)
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone, strict=False)
        calibrate_standard(model, joint_samples, epochs=10, device=device)
        cfg_joint_std = {}
        for task in task_names:
            model.heads[task].load_state_dict(heads_state_dicts[task])
            cfg_joint_std[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        cfg_joint_std['average'] = sum(cfg_joint_std.values()) / 3
        results[cfg_name]['Joint_Standard_Cal'] = cfg_joint_std
        print(f"  Joint Standard Cal (Reset): {cfg_joint_std['average']:.2f}%")
        
        # 3. Task-Specific Standard Calibration (Reset, 10 epochs)
        cfg_ts_std = {}
        ts_std_accs = []
        for task in task_names:
            model = MultiTaskResNet18().to(device)
            model.load_state_dict(merged_backbone, strict=False)
            model.heads[task].load_state_dict(heads_state_dicts[task])
            calibrate_standard(model, task_real_samples[task], epochs=10, device=device)
            acc = evaluate_on_task(model, test_datasets_dict[task], task, device)
            cfg_ts_std[task] = acc
            ts_std_accs.append(acc)
        cfg_ts_std['average'] = sum(ts_std_accs) / 3
        results[cfg_name]['TS_Standard_Cal'] = cfg_ts_std
        print(f"  TS Standard Cal (Reset): {cfg_ts_std['average']:.2f}%")
        
        # 4. Joint RBA (No Reset, 30 epochs, joint real data)
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone, strict=False)
        calibrate_rba(model, joint_samples, epochs=30, momentum=0.2, device=device)
        cfg_joint_rba = {}
        for task in task_names:
            model.heads[task].load_state_dict(heads_state_dicts[task])
            cfg_joint_rba[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        cfg_joint_rba['average'] = sum(cfg_joint_rba.values()) / 3
        results[cfg_name]['Joint_RBA'] = cfg_joint_rba
        print(f"  Joint RBA (No Reset): {cfg_joint_rba['average']:.2f}%")
        
        # 5. Task-Specific RBA (No Reset, 30 epochs, task-specific real data)
        cfg_ts_rba = {}
        ts_rba_accs = []
        for task in task_names:
            model = MultiTaskResNet18().to(device)
            model.load_state_dict(merged_backbone, strict=False)
            model.heads[task].load_state_dict(heads_state_dicts[task])
            calibrate_rba(model, task_real_samples[task], epochs=30, momentum=0.2, device=device)
            acc = evaluate_on_task(model, test_datasets_dict[task], task, device)
            cfg_ts_rba[task] = acc
            ts_rba_accs.append(acc)
        cfg_ts_rba['average'] = sum(ts_rba_accs) / 3
        results[cfg_name]['TS_RBA'] = cfg_ts_rba
        print(f"  TS RBA (No Reset): {cfg_ts_rba['average']:.2f}%")

    # Save results
    with open('results_final.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nSaved final results to 'results_final.json'")
    
    # Plot results
    configs = [f"{m}_{l}" for m, l in configurations]
    plt.figure(figsize=(10, 6))
    
    methods = {
        'Uncalibrated': 'Uncalibrated Baseline',
        'Joint_Standard_Cal': 'Joint Standard Cal (Reset)',
        'TS_Standard_Cal': 'Task-Specific Standard Cal (Reset)',
        'Joint_RBA': 'Joint RBA (Ours)',
        'TS_RBA': 'Task-Specific RBA (Ours)'
    }
    
    for m, label in methods.items():
        accs = [results[cfg][m]['average'] for cfg in configs]
        plt.plot(configs, accs, marker='o', label=label, linewidth=2.5)
        
    oracle_avg = sum(oracle_accuracies.values()) / 3
    plt.axhline(y=oracle_avg, color='r', linestyle='--', label='Oracle Experts (80.32%)', alpha=0.7)
    
    plt.title('Multi-Task Merged Model Performance: Standard Calibration vs. RBA', fontsize=12, fontweight='bold')
    plt.xlabel('Merging Configuration (Method_Lambda)', fontsize=11)
    plt.ylabel('Average Test Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig('calibration_comparison_final.png', dpi=300)
    print("Saved beautiful comparison plot to 'calibration_comparison_final.png'")

if __name__ == '__main__':
    main()
