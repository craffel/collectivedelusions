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

def run_sequential_spatial_calibration(calibrated_model, target_spatial, joint_x, device='cuda', mode='TAAC'):
    calibrated_model.eval()
    
    # 1. Register hooks in 'none' mode
    hooks, handles = register_hooks(calibrated_model, mode='none')
    
    # 2. Iterate and calibrate sequentially
    for layer_name, hook in hooks.items():
        hook.mode = 'collect_spatial'
        hook.reset()
        
        with torch.no_grad():
            _ = calibrated_model(joint_x)
            
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

def run_test_jssc(calibrated_model, target_spatial, target_spectral, joint_x, device='cuda', spectral_gamma_max=2.0, only_late=False):
    # Step A: Sequential Spatial Calibration
    hooks, handles = run_sequential_spatial_calibration(calibrated_model, target_spatial, joint_x, device=device, mode='TAAC')
    
    # Step B: Fuse spatial calibration
    fused_model = fuse_calibration_to_bn(calibrated_model, hooks)
    for h in handles:
        h.remove()
        
    for name, module in fused_model.named_modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        
    # Step C: Collect Spectral Stats (Parallel)
    hooks, handles = register_hooks(fused_model, mode='collect_spectral', gamma_max=spectral_gamma_max)
    with torch.no_grad():
        _ = fused_model(joint_x)
        
    for layer_name, hook in hooks.items():
        # Check if we only calibrate late layers (e.g. layer3 and layer4)
        if only_late and not any(k in layer_name for k in ['layer3', 'layer4']):
            hook.mode = 'none'
            continue
            
        merged_spectral = hook.spectral_mags[0]
        target_spectral_l = target_spectral[layer_name]
        
        gamma = target_spectral_l / (merged_spectral + 1e-5)
        gamma_star = torch.clamp(gamma, 1.0/hook.gamma_max, hook.gamma_max)
        
        hook.set_calibration_params(None, None, gamma_star)
        hook.mode = 'apply_spectral'
        
    return fused_model, handles

def run_sequential_jssc(calibrated_model, target_spatial, target_spectral, joint_x, device='cuda', spectral_gamma_max=2.0, only_late=False):
    # Step A: Sequential Spatial Calibration
    hooks, handles = run_sequential_spatial_calibration(calibrated_model, target_spatial, joint_x, device=device, mode='TAAC')
    
    # Step B: Fuse spatial calibration
    fused_model = fuse_calibration_to_bn(calibrated_model, hooks)
    for h in handles:
        h.remove()
        
    for name, module in fused_model.named_modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        
    # Step C: Sequential Spectral Calibration
    hooks, handles = register_hooks(fused_model, mode='none', gamma_max=spectral_gamma_max)
    
    for layer_name, hook in hooks.items():
        if only_late and not any(k in layer_name for k in ['layer3', 'layer4']):
            continue
            
        hook.mode = 'collect_spectral'
        hook.reset()
        
        with torch.no_grad():
            _ = fused_model(joint_x)
            
        merged_spectral = hook.spectral_mags[0]
        target_spectral_l = target_spectral[layer_name]
        
        gamma = target_spectral_l / (merged_spectral + 1e-5)
        gamma_star = torch.clamp(gamma, 1.0/hook.gamma_max, hook.gamma_max)
        
        hook.set_calibration_params(None, None, gamma_star)
        hook.mode = 'apply_spectral'
        
    return fused_model, handles

def run_spectral_first_jssc(calibrated_model, target_spatial, target_spectral, joint_x, device='cuda', spectral_gamma_max=2.0, only_late=False):
    # Step A: Sequential Spectral Calibration
    spec_hooks, spec_handles = register_hooks(calibrated_model, mode='none', gamma_max=spectral_gamma_max)
    
    for layer_name, hook in spec_hooks.items():
        if only_late and not any(k in layer_name for k in ['layer3', 'layer4']):
            continue
            
        hook.mode = 'collect_spectral'
        hook.reset()
        
        with torch.no_grad():
            _ = calibrated_model(joint_x)
            
        merged_spectral = hook.spectral_mags[0]
        target_spectral_l = target_spectral[layer_name]
        
        gamma = target_spectral_l / (merged_spectral + 1e-5)
        gamma_star = torch.clamp(gamma, 1.0/hook.gamma_max, hook.gamma_max)
        
        hook.set_calibration_params(None, None, gamma_star)
        hook.mode = 'apply_spectral'
        
    # Step B: Sequential Spatial Calibration on top of Spectrally Calibrated Model
    spat_hooks, spat_handles = register_hooks(calibrated_model, mode='none')
    
    for layer_name, hook in spat_hooks.items():
        hook.mode = 'collect_spatial'
        hook.reset()
        
        with torch.no_grad():
            _ = calibrated_model(joint_x)
            
        merged_mean = hook.spatial_means[0]
        merged_std = hook.spatial_stds[0]
        
        target_mean = target_spatial[layer_name]['mean']
        target_std = target_spatial[layer_name]['std']
        
        s = target_std / (merged_std + 1e-5)
        s = torch.clamp(s, 1.0/hook.gamma_max, hook.gamma_max)
        b_cal = target_mean - s * merged_mean
        
        hook.set_calibration_params(s, b_cal, None)
        hook.mode = 'apply_spatial'
        
    # Step C: Create Fused Model and apply weights & spectral hooks cleanly
    fused_model = copy_model(calibrated_model)
    
    # Remove all hooks from fused_model first to make a clean start
    for name, module in list(fused_model.named_modules()):
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
            
    # Apply Fused Spatial params directly into BN parameters
    with torch.no_grad():
        for name, module in fused_model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in spat_hooks:
                hook = spat_hooks[name]
                if hook.s is not None and hook.b_cal is not None:
                    s = hook.s.to(module.weight.device)
                    b_cal = hook.b_cal.to(module.bias.device)
                    module.weight.copy_(s * module.weight)
                    module.bias.copy_(s * module.bias + b_cal)
                    
    # Clean up handles of calibrated_model
    for h in spec_handles:
        h.remove()
    for h in spat_handles:
        h.remove()
        
    # Register Spectral Hooks on Fused Model
    new_spec_hooks, new_spec_handles = register_hooks(fused_model, mode='none', gamma_max=spectral_gamma_max)
    for name, hook in new_spec_hooks.items():
        if name in spec_hooks and spec_hooks[name].gamma_star is not None:
            hook.set_calibration_params(None, None, spec_hooks[name].gamma_star)
            hook.mode = 'apply_spectral'
            
    return fused_model, new_spec_handles

def run_evaluation_test(device='cpu'):
    print("Running sweeps over JSSC hyperparameters...")
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    
    expert_models = {name: load_expert(name, device=device) for name in datasets}
    expert_state_dicts = {name: model.state_dict() for name, model in expert_models.items()}
    
    N = 64
    cal_datasets = {}
    for name in datasets:
        _, full_cal_ds, _ = get_splits(name)
        cal_datasets[name] = Subset(full_cal_ds, list(range(N)))
        
    target_spatial, target_spectral = collect_expert_stats(expert_models, cal_datasets, device=device)
    
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
    
    # Get joint calibration data
    joint_x = []
    for name, ds in cal_datasets.items():
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        x, _ = next(iter(loader))
        joint_x.append(x)
    joint_x = torch.cat(joint_x, dim=0).to(device)
    
    # Sweep configurations
    configs = [
        {'spectral_gamma_max': 5.0, 'only_late': False},
        {'spectral_gamma_max': 2.0, 'only_late': False},
        {'spectral_gamma_max': 1.5, 'only_late': False},
        {'spectral_gamma_max': 1.2, 'only_late': False},
        {'spectral_gamma_max': 1.1, 'only_late': False},
        {'spectral_gamma_max': 5.0, 'only_late': True},
        {'spectral_gamma_max': 2.0, 'only_late': True},
        {'spectral_gamma_max': 1.5, 'only_late': True},
    ]
    
    for config in configs:
        # 1. Parallel JSSC
        cal_model_p = models.resnet18()
        cal_model_p.fc = nn.Linear(512, 10)
        cal_model_p.load_state_dict(merged_state_ta)
        cal_model_p = cal_model_p.to(device)
        
        fused_model_p, handles_p = run_test_jssc(
            cal_model_p, target_spatial, target_spectral, joint_x, device=device, 
            spectral_gamma_max=config['spectral_gamma_max'], only_late=config['only_late']
        )
        
        task_accs_p = {}
        with torch.no_grad():
            for task_name in datasets:
                eval_model = copy_model(fused_model_p)
                eval_model.fc.weight.copy_(expert_state_dicts[task_name]['fc.weight'])
                eval_model.fc.bias.copy_(expert_state_dicts[task_name]['fc.bias'])
                acc = evaluate_model(eval_model, task_name, device=device)
                task_accs_p[task_name] = acc
        avg_p = np.mean(list(task_accs_p.values()))
        print(f"Parallel   JSSC Config: {config} -> Accs: {task_accs_p}, Avg: {avg_p:.2f}%")
        for h in handles_p:
            h.remove()

        # 2. Sequential JSSC
        cal_model_s = models.resnet18()
        cal_model_s.fc = nn.Linear(512, 10)
        cal_model_s.load_state_dict(merged_state_ta)
        cal_model_s = cal_model_s.to(device)
        
        fused_model_s, handles_s = run_sequential_jssc(
            cal_model_s, target_spatial, target_spectral, joint_x, device=device, 
            spectral_gamma_max=config['spectral_gamma_max'], only_late=config['only_late']
        )
        
        task_accs_s = {}
        with torch.no_grad():
            for task_name in datasets:
                eval_model = copy_model(fused_model_s)
                eval_model.fc.weight.copy_(expert_state_dicts[task_name]['fc.weight'])
                eval_model.fc.bias.copy_(expert_state_dicts[task_name]['fc.bias'])
                acc = evaluate_model(eval_model, task_name, device=device)
                task_accs_s[task_name] = acc
        avg_s = np.mean(list(task_accs_s.values()))
        print(f"Sequential JSSC Config: {config} -> Accs: {task_accs_s}, Avg: {avg_s:.2f}%")
        print(f"Improvement over Parallel: {avg_s - avg_p:+.2f}%\n")
        for h in handles_s:
            h.remove()

        # 3. Spectral-First JSSC
        cal_model_sf = models.resnet18()
        cal_model_sf.fc = nn.Linear(512, 10)
        cal_model_sf.load_state_dict(merged_state_ta)
        cal_model_sf = cal_model_sf.to(device)
        
        fused_model_sf, handles_sf = run_spectral_first_jssc(
            cal_model_sf, target_spatial, target_spectral, joint_x, device=device, 
            spectral_gamma_max=config['spectral_gamma_max'], only_late=config['only_late']
        )
        
        task_accs_sf = {}
        with torch.no_grad():
            for task_name in datasets:
                eval_model = copy_model(fused_model_sf)
                eval_model.fc.weight.copy_(expert_state_dicts[task_name]['fc.weight'])
                eval_model.fc.bias.copy_(expert_state_dicts[task_name]['fc.bias'])
                acc = evaluate_model(eval_model, task_name, device=device)
                task_accs_sf[task_name] = acc
        avg_sf = np.mean(list(task_accs_sf.values()))
        print(f"Spectral-First JSSC Config: {config} -> Accs: {task_accs_sf}, Avg: {avg_sf:.2f}%")
        print(f"Improvement over Parallel: {avg_sf - avg_p:+.2f}%\n")
        print("-" * 50)
        for h in handles_sf:
            h.remove()

if __name__ == '__main__':
    run_evaluation_test(device='cuda' if torch.cuda.is_available() else 'cpu')
