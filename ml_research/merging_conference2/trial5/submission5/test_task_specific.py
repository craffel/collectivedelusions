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

def run_sequential_spatial_calibration(calibrated_model, target_spatial, cal_x, device='cuda', mode='TAAC'):
    calibrated_model.eval()
    hooks, handles = register_hooks(calibrated_model, mode='none')
    
    for layer_name, hook in hooks.items():
        hook.mode = 'collect_spatial'
        hook.reset()
        
        with torch.no_grad():
            _ = calibrated_model(cal_x)
            
        merged_mean = hook.spatial_means[0]
        merged_std = hook.spatial_stds[0]
        
        target_mean = target_spatial[layer_name]['mean']
        target_std = target_spatial[layer_name]['std']
        
        if mode == 'SP-TAAC':
            merged_std_g = merged_std.mean()
            target_std_g = target_std.mean()
            gamma = target_std_g / (merged_std_g + 1e-5)
            gamma = torch.clamp(gamma, 1.0/hook.gamma_max, hook.gamma_max)
            s = torch.ones_like(merged_std) * gamma
            b_cal = torch.zeros_like(merged_mean)
        else:
            s = target_std / (merged_std + 1e-5)
            s = torch.clamp(s, 1.0/hook.gamma_max, hook.gamma_max)
            b_cal = target_mean - s * merged_mean
            
        hook.set_calibration_params(s, b_cal, None)
        hook.mode = 'apply_spatial'
        
    return hooks, handles

def run_task_specific_jssc(calibrated_model, target_spatial, target_spectral, cal_x, device='cuda'):
    hooks, handles = run_sequential_spatial_calibration(calibrated_model, target_spatial, cal_x, device=device, mode='TAAC')
    fused_model = fuse_calibration_to_bn(calibrated_model, hooks)
    for h in handles:
        h.remove()
        
    for name, module in fused_model.named_modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
            
    hooks, handles = register_hooks(fused_model, mode='collect_spectral')
    with torch.no_grad():
        _ = fused_model(cal_x)
        
    for layer_name, hook in hooks.items():
        merged_spectral = hook.spectral_mags[0]
        target_spectral_l = target_spectral[layer_name]
        
        gamma = target_spectral_l / (merged_spectral + 1e-5)
        gamma_star = torch.clamp(gamma, 1.0/hook.gamma_max, hook.gamma_max)
        
        hook.set_calibration_params(None, None, gamma_star)
        hook.mode = 'apply_spectral'
        
    return fused_model, handles

def run_evaluation_test(device='cpu'):
    print("Running task-specific calibration test...")
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    
    expert_models = {name: load_expert(name, device=device) for name in datasets}
    expert_state_dicts = {name: model.state_dict() for name, model in expert_models.items()}
    
    N = 64
    cal_datasets = {}
    for name in datasets:
        _, full_cal_ds, _ = get_splits(name)
        cal_datasets[name] = Subset(full_cal_ds, list(range(N)))
        
    # Collect individual expert statistics (task-specific!)
    expert_spatial = {}
    expert_spectral = {}
    for name in datasets:
        cal_ds = cal_datasets[name]
        cal_loader = DataLoader(cal_ds, batch_size=len(cal_ds), shuffle=False)
        x_cal, _ = next(iter(cal_loader))
        x_cal = x_cal.to(device)
        
        model = expert_models[name]
        # Spatial
        hooks, handles = register_hooks(model, mode='collect_spatial')
        with torch.no_grad():
            _ = model(x_cal)
        spatial_stats = {}
        for layer_name, hook in hooks.items():
            spatial_stats[layer_name] = {'mean': hook.spatial_means[0], 'std': hook.spatial_stds[0]}
        expert_spatial[name] = spatial_stats
        for h in handles:
            h.remove()
            
        # Spectral
        hooks, handles = register_hooks(model, mode='collect_spectral')
        with torch.no_grad():
            _ = model(x_cal)
        spectral_stats = {}
        for layer_name, hook in hooks.items():
            spectral_stats[layer_name] = hook.spectral_mags[0]
        expert_spectral[name] = spectral_stats
        for h in handles:
            h.remove()
            
    # Merge using Task Arithmetic (lambda = 0.3)
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    base_state_dict = base_model.state_dict()
    
    merged_state_ta = merge_models(
        [e.state_dict() for e in expert_models.values()], 
        base_state_dict=base_state_dict, 
        mode='TA', 
        lambda_val=0.3
    )
    
    # Evaluate Task-Specific TAAC & JSSC
    ta_taac_accs = {}
    ta_jssc_accs = {}
    
    for name in datasets:
        # Load calibration data for this task
        cal_ds = cal_datasets[name]
        cal_loader = DataLoader(cal_ds, batch_size=len(cal_ds), shuffle=False)
        x_cal, _ = next(iter(cal_loader))
        x_cal = x_cal.to(device)
        
        # 1. TAAC
        cal_model = models.resnet18()
        cal_model.fc = nn.Linear(512, 10)
        cal_model.load_state_dict(merged_state_ta)
        cal_model = cal_model.to(device)
        
        hooks, handles = run_sequential_spatial_calibration(
            cal_model, expert_spatial[name], x_cal, device=device, mode='TAAC'
        )
        
        eval_model = copy_model(cal_model)
        with torch.no_grad():
            eval_model.fc.weight.copy_(expert_state_dicts[name]['fc.weight'])
            eval_model.fc.bias.copy_(expert_state_dicts[name]['fc.bias'])
        acc = evaluate_model(eval_model, name, device=device)
        ta_taac_accs[name] = acc
        for h in handles:
            h.remove()
            
        # 2. JSSC
        cal_model = models.resnet18()
        cal_model.fc = nn.Linear(512, 10)
        cal_model.load_state_dict(merged_state_ta)
        cal_model = cal_model.to(device)
        
        fused_model, handles = run_task_specific_jssc(
            cal_model, expert_spatial[name], expert_spectral[name], x_cal, device=device
        )
        
        eval_model = copy_model(fused_model)
        with torch.no_grad():
            eval_model.fc.weight.copy_(expert_state_dicts[name]['fc.weight'])
            eval_model.fc.bias.copy_(expert_state_dicts[name]['fc.bias'])
        acc = evaluate_model(eval_model, name, device=device)
        ta_jssc_accs[name] = acc
        for h in handles:
            h.remove()
            
    print("\n=== Task-Specific Sequential Results ===")
    print(f"TAAC: {ta_taac_accs}, Avg: {np.mean(list(ta_taac_accs.values())):.2f}%")
    print(f"JSSC: {ta_jssc_accs}, Avg: {np.mean(list(ta_jssc_accs.values())):.2f}%")

if __name__ == '__main__':
    run_evaluation_test(device='cuda' if torch.cuda.is_available() else 'cpu')
