import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import json
import numpy as np
from data import get_splits
from calibration import register_hooks, merge_models, fuse_calibration_to_bn

def evaluate_model(model, dataset_name, device='cuda'):
    model.eval()
    _, _, test_ds = get_splits(dataset_name)
    loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    acc = 100.0 * correct / total
    return acc

def collect_expert_stats(expert_models, cal_datasets, device='cuda'):
    """
    Collects spatial and spectral target statistics from the experts.
    """
    print("Collecting statistics from expert models...")
    expert_spatial_stats = {}
    expert_spectral_stats = {}
    
    for name, model in expert_models.items():
        model.eval()
        cal_ds = cal_datasets[name]
        cal_loader = DataLoader(cal_ds, batch_size=len(cal_ds), shuffle=False)
        x_cal, _ = next(iter(cal_loader))
        x_cal = x_cal.to(device)
        
        # 1. Collect spatial stats
        hooks, handles = register_hooks(model, mode='collect_spatial')
        with torch.no_grad():
            _ = model(x_cal)
        
        spatial_stats = {}
        for layer_name, hook in hooks.items():
            spatial_stats[layer_name] = {
                'mean': hook.spatial_means[0],
                'std': hook.spatial_stds[0]
            }
        expert_spatial_stats[name] = spatial_stats
        for h in handles:
            h.remove()
            
        # 2. Collect spectral stats
        hooks, handles = register_hooks(model, mode='collect_spectral')
        with torch.no_grad():
            _ = model(x_cal)
            
        spectral_stats = {}
        for layer_name, hook in hooks.items():
            spectral_stats[layer_name] = hook.spectral_mags[0]
        expert_spectral_stats[name] = spectral_stats
        for h in handles:
            h.remove()
            
    # Compute Target Spatial and Spectral Stats (averages across experts)
    target_spatial = {}
    target_spectral = {}
    layer_names = list(expert_spatial_stats['mnist'].keys())
    
    for l_name in layer_names:
        # Spatial Target
        means = [expert_spatial_stats[name][l_name]['mean'] for name in expert_models]
        stds = [expert_spatial_stats[name][l_name]['std'] for name in expert_models]
        target_spatial[l_name] = {
            'mean': torch.stack(means).mean(dim=0),
            'std': torch.stack(stds).mean(dim=0)
        }
        
        # Spectral Target
        mags = [expert_spectral_stats[name][l_name] for name in expert_models]
        target_spectral[l_name] = torch.stack(mags).mean(dim=0)
        
    return target_spatial, target_spectral

def run_calibration(merged_model, target_spatial, target_spectral, cal_datasets, mode='none', device='cuda'):
    """
    Calibrates the merged model using the specified mode.
    Returns: calibrated_model, handles
    """
    calibrated_model = copy_model(merged_model)
    calibrated_model.eval()
    
    if mode == 'none':
        return calibrated_model, []
        
    # Get joint calibration data
    joint_x = []
    for name, ds in cal_datasets.items():
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        x, _ = next(iter(loader))
        joint_x.append(x)
    joint_x = torch.cat(joint_x, dim=0).to(device)
    
    # 1. SP-TAAC (Static global layer-wise scaling, run sequentially for stability)
    if mode == 'SP-TAAC':
        hooks, handles = register_hooks(calibrated_model, mode='none')
        
        for layer_name, hook in hooks.items():
            hook.mode = 'collect_spatial'
            hook.reset()
            
            with torch.no_grad():
                _ = calibrated_model(joint_x)
                
            merged_std = hook.spatial_stds[0]
            merged_std_g = merged_std.mean()
            
            target_std = target_spatial[layer_name]['std']
            target_std_g = target_std.mean()
            
            gamma = target_std_g / (merged_std_g + 1e-5)
            gamma = torch.clamp(gamma, 1.0/hook.gamma_max, hook.gamma_max)
            
            s = torch.ones_like(merged_std) * gamma
            b_cal = torch.zeros_like(hook.spatial_means[0])
            
            hook.set_calibration_params(s, b_cal, None)
            hook.mode = 'apply_spatial'
            
        return calibrated_model, handles
        
    # 2. TAAC (Channel-wise Affine Calibration, run sequentially with clamping)
    elif mode == 'TAAC':
        hooks, handles = register_hooks(calibrated_model, mode='none')
        
        for layer_name, hook in hooks.items():
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
            
        return calibrated_model, handles
        
    # 3. ZIO-CF (Fused TAAC into BatchNorm weights/biases, run sequentially with clamping)
    elif mode == 'ZIO-CF':
        hooks, handles = register_hooks(calibrated_model, mode='none')
        
        for layer_name, hook in hooks.items():
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
            
        # Fuse directly into BatchNorm
        fused_model = fuse_calibration_to_bn(calibrated_model, hooks)
        
        # Remove hooks from calibrated_model
        for h in handles:
            h.remove()
            
        # Crucially, clear copied hooks on fused_model to prevent double application!
        for name, module in fused_model.named_modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
                
        return fused_model, []
        
    # 4. FDSA (Frequency-domain Spectral Alignment, run sequentially)
    elif mode == 'FDSA':
        hooks, handles = register_hooks(calibrated_model, mode='none')
        
        for layer_name, hook in hooks.items():
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
            
        return calibrated_model, handles
        
    # 5. JSSC (Joint Spatial-Spectral Calibration, run sequentially with clamping)
    elif mode == 'JSSC':
        # Step A: Sequential Spatial Calibration
        hooks, handles = register_hooks(calibrated_model, mode='none')
        
        for layer_name, hook in hooks.items():
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
            
        # Step B: Fuse spatial calibration to make JSSC zero spatial overhead
        fused_model = fuse_calibration_to_bn(calibrated_model, hooks)
        
        # Remove hooks from calibrated_model
        for h in handles:
            h.remove()
            
        # Crucially, clear copied hooks on fused_model to prevent double application!
        for name, module in fused_model.named_modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            
        # Step C: Sequential Spectral Calibration on Fused Model
        hooks, handles = register_hooks(fused_model, mode='none')
        
        for layer_name, hook in hooks.items():
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
        
    return calibrated_model, []

def copy_model(model):
    return copy.deepcopy(model)

def load_expert(dataset_name, device='cuda'):
    # Load ResNet-18 expert
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    path = f"./models/{dataset_name}_expert.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    return model

def run_evaluation_suite(device='cuda'):
    print("Initializing evaluation suite...")
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    
    # Load datasets and experts
    cal_datasets = {}
    expert_models = {}
    
    for name in datasets:
        _, cal_ds, _ = get_splits(name)
        cal_datasets[name] = cal_ds
        expert_models[name] = load_expert(name, device=device)
        
    # Collect expert target stats
    target_spatial, target_spectral = collect_expert_stats(expert_models, cal_datasets, device=device)
    
    # Load baseline model (pre-trained ResNet-18)
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10) # dummy head to match expert state keys
    base_state_dict = base_model.state_dict()
    
    # Merging Methods: Weight Averaging (WA) and Task Arithmetic (TA)
    mergers = {
        'WA': lambda experts: merge_models([e.state_dict() for e in experts], mode='WA'),
        'TA': lambda experts: merge_models([e.state_dict() for e in experts], base_state_dict=base_state_dict, mode='TA', lambda_val=0.3)
    }
    
    calibration_methods = ['None', 'SP-TAAC', 'TAAC', 'ZIO-CF', 'FDSA', 'JSSC']
    results = {}
    
    expert_state_dicts = {name: model.state_dict() for name, model in expert_models.items()}
    
    for merge_name, merge_fn in mergers.items():
        print(f"\n==================== Merging Mode: {merge_name} ====================")
        results[merge_name] = {}
        
        # Merge state dict and load into a model template
        merged_state = merge_fn(list(expert_models.values()))
        merged_template = models.resnet18()
        merged_template.fc = nn.Linear(512, 10)
        merged_template.load_state_dict(merged_state)
        merged_template = merged_template.to(device)
        
        for cal_method in calibration_methods:
            print(f"\nRunning Calibration: {cal_method}...")
            # Run calibration
            cal_model, handles = run_calibration(
                merged_template, target_spatial, target_spectral, cal_datasets, mode=cal_method, device=device
            )
            
            # Evaluate on each dataset using target heads
            task_accs = {}
            with torch.no_grad():
                for task_name in datasets:
                    # Load correct classification head from the expert
                    # Since the merged model has some classification head, we overwrite its head with the task expert's head
                    eval_model = copy_model(cal_model)
                    expert_head_state = expert_state_dicts[task_name]['fc.weight']
                    expert_bias_state = expert_state_dicts[task_name]['fc.bias']
                    eval_model.fc.weight.copy_(expert_head_state)
                    eval_model.fc.bias.copy_(expert_bias_state)
                    
                    acc = evaluate_model(eval_model, task_name, device=device)
                    task_accs[task_name] = acc
                    print(f"  {task_name} Accuracy: {acc:.2f}%")
                
            avg_acc = np.mean(list(task_accs.values()))
            print(f"  Average Accuracy ({cal_method}): {avg_acc:.2f}%")
            
            results[merge_name][cal_method] = {
                'tasks': task_accs,
                'average': avg_acc
            }
            
            # Remove hooks
            for h in handles:
                h.remove()
                
    # Save results to json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults successfully saved to evaluation_results.json")
    
    # Print formatted Markdown table
    print("\n### Summary Table (Test Accuracy %)")
    print("| Merge Mode | Calibration | MNIST | F-MNIST | CIFAR-10 | Average |")
    print("|---|---|---|---|---|---|")
    for merge_name in results:
        for cal_method in results[merge_name]:
            res = results[merge_name][cal_method]
            print(f"| {merge_name} | {cal_method} | {res['tasks']['mnist']:.2f}% | {res['tasks']['fashion_mnist']:.2f}% | {res['tasks']['cifar10']:.2f}% | {res['average']:.2f}% |")

if __name__ == '__main__':
    import copy
    run_evaluation_suite(device='cuda' if torch.cuda.is_available() else 'cpu')
