import os
import json
import torch
import torch.nn as nn
from dataset import get_dataloaders
from models import MultiTaskResNet18
from eval import (
    get_calibration_loaders,
    assemble_merged_model,
    evaluate_model,
    record_expert_stds,
    register_sp_taac_hooks
)

def get_layer_group(name):
    """Determine the layer group based on standard ResNet-18 block naming."""
    if name.startswith('bn1') or name.startswith('layer1.'):
        return 'shallow'
    elif name.startswith('layer2.') or name.startswith('layer3.'):
        return 'middle'
    elif name.startswith('layer4.'):
        return 'deep'
    else:
        return 'unknown'

def calibrate_sp_taac_ablation(merged_model, expert_stds_all, joint_loader, active_group, device):
    """Perform sequential SP-TAAC, but only calibrating layers in the active_group."""
    merged_model.eval()
    
    bn_names = []
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_names.append(name)
            
    # Target stds are the average of the experts' stds
    target_stds = {}
    for name in bn_names:
        target_stds[name] = sum(expert_stds_all[t][name] for t in ['mnist', 'fashion', 'cifar']) / 3.0
        
    gammas = {}
    active_handles = []
    
    for target_name in bn_names:
        recorded_activations = []
        
        def record_hook(module, input):
            recorded_activations.append(input[0].detach())
            
        target_module = dict(merged_model.backbone.named_modules())[target_name]
        temp_handle = target_module.register_forward_pre_hook(record_hook)
        
        # Pass joint calibration batch through backbone
        for inputs, targets, tasks in joint_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                _ = merged_model.backbone(inputs)
            break
            
        temp_handle.remove()
        
        x = torch.cat(recorded_activations, dim=0)
        var = x.var()
        std_merged = torch.sqrt(var + 1e-5).item()
        
        # Compute task-agnostic global scaling factor
        gamma = target_stds[target_name] / std_merged
        
        # If the layer group is not active, set gamma to 1.0 (no calibration)
        group = get_layer_group(target_name)
        if active_group != 'all' and group != active_group:
            gamma = 1.0
            
        gammas[target_name] = gamma
        
        # Register permanent scaling hook for the calibration pass
        # to stabilize representations for deeper layers sequentially
        def make_scaling_hook(g):
            def scaling_hook(module, input):
                return (input[0] * g,)
            return scaling_hook
            
        h = target_module.register_forward_pre_hook(make_scaling_hook(gamma))
        active_handles.append(h)
        
    # Clean up calibration hooks
    for h in active_handles:
        h.remove()
        
    return gammas

def main():
    os.makedirs('results', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Calibration Depth Ablation Study on device: {device.upper()}")
    
    # Load test dataloaders
    print("Loading test dataloaders...")
    _, test_loaders = get_dataloaders(batch_size=128)
    
    # Get calibration loaders (N=128)
    print("Loading calibration loaders (N=128)...")
    cal_loaders, joint_loader = get_calibration_loaders(N=128, seed=42)
    
    expert_paths = {
        'mnist': 'checkpoints/expert_mnist.pt',
        'fashion': 'checkpoints/expert_fashion.pt',
        'cifar': 'checkpoints/expert_cifar.pt'
    }
    pretrained_path = 'checkpoints/pretrained.pt'
    
    # Pre-compute expert stds
    print("Pre-computing expert standard deviations...")
    expert_stds_all = {}
    for task in ['mnist', 'fashion', 'cifar']:
        expert_stds_all[task] = record_expert_stds(expert_paths[task], cal_loaders[task], task, device)
        
    # Assemble Weight Averaging model
    print("Assembling Weight Averaging model...")
    model_wa = assemble_merged_model(expert_paths, pretrained_path, 'wa', 0.2).to(device)
    
    groups = ['none', 'shallow', 'middle', 'deep', 'all']
    results = {}
    
    for group in groups:
        print(f"\nEvaluating ablation setting: {group.upper()} LAYERS CALIBRATED...")
        
        if group == 'none':
            # Uncalibrated baseline
            res = evaluate_model(
                model_wa, test_loaders, cal_loaders, joint_loader, expert_paths,
                cal_method='none', device=device
            )
        else:
            # Calibrate only the selected group
            gammas = calibrate_sp_taac_ablation(model_wa, expert_stds_all, joint_loader, group, device)
            res = evaluate_model(
                model_wa, test_loaders, cal_loaders, joint_loader, expert_paths,
                cal_method='sp_taac', sp_taac_gammas=gammas, device=device
            )
            
        results[group] = res
        print(f"  MNIST: {res['mnist']:.2f}% | Fashion: {res['fashion']:.2f}% | CIFAR-10: {res['cifar']:.2f}% | AVG: {res['avg']:.2f}%")
        
    # Save the ablation results
    output_path = 'results/ablation_depth.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAblation study complete! Results saved to {output_path}")
    
    print("\n" + "="*70)
    print("CALIBRATION DEPTH ABLATION STUDY SUMMARY (Avg Acc %):")
    print("="*70)
    print(f"{'Active Layer Group':<20} | {'MNIST':<10} | {'F-MNIST':<10} | {'CIFAR-10':<10} | {'Average':<10}")
    print("-" * 70)
    for group in groups:
        res = results[group]
        print(f"{group.upper():<20} | {res['mnist']:<10.2f} | {res['fashion']:<10.2f} | {res['cifar']:<10.2f} | {res['avg']:<10.2f}")
    print("="*70)

if __name__ == '__main__':
    main()
