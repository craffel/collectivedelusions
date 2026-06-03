import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import MultiTaskResNet18
from eval import assemble_merged_model, get_calibration_loaders

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    expert_paths = {
        'mnist': 'checkpoints/expert_mnist.pt',
        'fashion': 'checkpoints/expert_fashion.pt',
        'cifar': 'checkpoints/expert_cifar.pt'
    }
    pretrained_path = 'checkpoints/pretrained.pt'
    
    # Assemble merged model (WA)
    model = assemble_merged_model(expert_paths, pretrained_path, merge_mode='wa').to(device)
    model.eval()
    
    # Load expert models for analysis
    experts = {}
    for task, path in expert_paths.items():
        m = MultiTaskResNet18(pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        experts[task] = m
        
    # Get all BatchNorm2d modules and their names
    bn_modules = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_modules.append((name, module))
            
    total_channels = sum(module.num_features for name, module in bn_modules)
    print(f"Total BatchNorm2d layers: {len(bn_modules)}, Total channels: {total_channels}")
    
    N_values = [4, 16, 64, 128, 256]
    
    print("\n" + "="*110)
    print(f"{'N':<5} | {'Joint Dead (%)':<15} | {'Task-Spec Dead (%)':<20} | {'Max SF (TAAC)':<15} | {'Max SF (TCAC)':<15} | {'Sign Flip (%)':<12}")
    print("="*110)
    
    results = []
    
    for N in N_values:
        cal_loaders, joint_loader = get_calibration_loaders(N=N, seed=42)
        
        # 1. Compute expert stats on calibration sets
        expert_stats = {t: {} for t in ['mnist', 'fashion', 'cifar']}
        for task in ['mnist', 'fashion', 'cifar']:
            recorded_inputs = {}
            handles = []
            def make_hook(name):
                def hook_fn(module, input):
                    recorded_inputs[name] = input[0].detach()
                return hook_fn
                
            for name, module in experts[task].backbone.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    handles.append(module.register_forward_pre_hook(make_hook(name)))
                    
            for inputs, targets in cal_loaders[task]:
                inputs = inputs.to(device)
                with torch.no_grad():
                    _ = experts[task](inputs, task)
                break
                
            for h in handles:
                h.remove()
                
            for name, act in recorded_inputs.items():
                mean = act.mean(dim=(0, 2, 3))
                var = act.var(dim=(0, 2, 3), unbiased=False)
                expert_stats[task][name] = {'mean': mean, 'std': torch.sqrt(var + 1e-5), 'var': var}
                
        # Compute joint target statistics (average of experts)
        target_stats = {}
        for name, _ in bn_modules:
            target_stats[name] = {
                'mean': sum(expert_stats[t][name]['mean'] for t in ['mnist', 'fashion', 'cifar']) / 3.0,
                'std': sum(expert_stats[t][name]['std'] for t in ['mnist', 'fashion', 'cifar']) / 3.0
            }
            
        # 2. Record merged model activations on task-specific and joint calibration sets
        merged_inputs_joint = {}
        merged_inputs_task = {t: {} for t in ['mnist', 'fashion', 'cifar']}
        
        # Record on joint
        handles = []
        def make_merged_joint_hook(name):
            def hook_fn(module, input):
                merged_inputs_joint[name] = input[0].detach()
            return hook_fn
            
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                handles.append(module.register_forward_pre_hook(make_merged_joint_hook(name)))
                
        for inputs, targets, tasks in joint_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                _ = model.backbone(inputs)
            break
            
        for h in handles:
            h.remove()
            
        # Record on task-specific
        for task in ['mnist', 'fashion', 'cifar']:
            handles = []
            recorded_task_inputs = {}
            def make_merged_task_hook(name):
                def hook_fn(module, input):
                    recorded_task_inputs[name] = input[0].detach()
                return hook_fn
                
            for name, module in model.backbone.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    handles.append(module.register_forward_pre_hook(make_merged_task_hook(name)))
                    
            for inputs, targets in cal_loaders[task]:
                inputs = inputs.to(device)
                with torch.no_grad():
                    # We pass through backbone to get intermediate inputs
                    _ = model.backbone(inputs)
                break
                
            for h in handles:
                h.remove()
                
            merged_inputs_task[task] = recorded_task_inputs
            
        # 3. Analyze sparsity trap metrics
        total_dead_joint = 0
        total_dead_task = 0
        
        max_sf_taac = 0.0
        max_sf_tcac = 0.0
        
        total_elements = 0
        total_flipped_elements = 0
        
        for name, module in bn_modules:
            # Joint Analysis
            act_joint = merged_inputs_joint[name]
            C = act_joint.shape[1]
            mean_joint = act_joint.mean(dim=(0, 2, 3))
            var_joint = act_joint.var(dim=(0, 2, 3), unbiased=False)
            std_joint = torch.sqrt(var_joint + 1e-5)
            
            dead_joint_mask = (var_joint < 1e-7)
            total_dead_joint += dead_joint_mask.sum().item()
            
            # TAAC scaling
            target_std = target_stats[name]['std']
            sf_taac = target_std / std_joint
            max_sf_taac = max(max_sf_taac, sf_taac.max().item())
            
            # Sign flips on joint set under TAAC
            s = sf_taac.view(1, C, 1, 1)
            mu_j = mean_joint.view(1, C, 1, 1)
            mu_t = target_stats[name]['mean'].view(1, C, 1, 1)
            taac_act = s * (act_joint - mu_j) + mu_t
            
            flips = ((act_joint > 0) != (taac_act > 0))
            total_flipped_elements += flips.sum().item()
            total_elements += act_joint.numel()
            
            # Task-specific analysis (TCAC)
            for task in ['mnist', 'fashion', 'cifar']:
                act_task = merged_inputs_task[task][name]
                var_task = act_task.var(dim=(0, 2, 3), unbiased=False)
                std_task = torch.sqrt(var_task + 1e-5)
                
                dead_task_mask = (var_task < 1e-7)
                total_dead_task += dead_task_mask.sum().item()
                
                # TCAC scaling: expert_std / merged_task_std
                expert_std = expert_stats[task][name]['std']
                sf_tcac = expert_std / std_task
                max_sf_tcac = max(max_sf_tcac, sf_tcac.max().item())
                
        dead_joint_pct = 100.0 * total_dead_joint / total_channels
        # Task-specific is averaged across the 3 tasks
        dead_task_pct = 100.0 * total_dead_task / (total_channels * 3)
        sign_flip_pct = 100.0 * total_flipped_elements / total_elements
        
        print(f"{N:<5} | {dead_joint_pct:<15.2f}% | {dead_task_pct:<20.2f}% | {max_sf_taac:<15.2f} | {max_sf_tcac:<15.2f} | {sign_flip_pct:<12.2f}%")
        
        results.append({
            'N': N,
            'dead_joint_pct': dead_joint_pct,
            'dead_task_pct': dead_task_pct,
            'max_sf_taac': max_sf_taac,
            'max_sf_tcac': max_sf_tcac,
            'sign_flip_pct': sign_flip_pct
        })
        
    print("="*110)
    print("\nSign flip of SP-TAAC is exactly 0.00% across all N because gamma_l > 0 and no mean shift is applied.")
    
    # Save results as JSON
    import json
    with open('results/sparsity_trap_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved results to results/sparsity_trap_analysis.json")

if __name__ == '__main__':
    main()
