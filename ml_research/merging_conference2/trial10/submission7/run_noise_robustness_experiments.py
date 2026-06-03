import os
import copy
import torch
import torch.nn as nn
import torchvision.models as models
from utils import (
    get_dataloaders,
    calibrate_bn,
    apply_weight_quantization,
    collect_activation_scales,
    apply_structured_pruning_mask,
    get_target_layers_mapping
)

def load_checkpoint(model, path, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333):
    merged_model = copy.deepcopy(progenitor).to(device)
    merged_state_dict = merged_model.state_dict()
    progenitor_state_dict = progenitor.state_dict()
    
    expert_state_dicts = []
    for path in expert_paths:
        expert_state_dicts.append(torch.load(path, map_location='cpu'))
        
    with torch.no_grad():
        for key in merged_state_dict.keys():
            if 'fc.' not in key:
                proj_w = progenitor_state_dict[key].cpu()
                update_sum = torch.zeros_like(proj_w)
                for exp_sd in expert_state_dicts:
                    update_sum += (exp_sd[key].cpu() - proj_w)
                merged_state_dict[key] = (proj_w + lambda_factor * update_sum).to(device)
                
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def get_weight_l1_masks(model, target_layers, prune_ratio):
    masks = {}
    with torch.no_grad():
        for layer_name in target_layers:
            module = dict(model.named_modules())[layer_name]
            weight = module.weight
            out_channels = weight.shape[0]
            l1_norms = torch.sum(torch.abs(weight.view(out_channels, -1)), dim=1)
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(l1_norms, k)[0]
                mask = (l1_norms > threshold).float()
            else:
                mask = torch.ones(out_channels, device=weight.device)
            masks[layer_name] = mask
    return masks

def get_activation_task_specific_masks(activation_scales, task_name, target_layers, prune_ratio, device):
    masks = {}
    with torch.no_grad():
        for layer_name in target_layers:
            task_scales = activation_scales[task_name][layer_name].to(device)
            out_channels = task_scales.shape[0]
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(task_scales, k)[0]
                mask = (task_scales > threshold).float().to(device)
            else:
                mask = torch.ones(out_channels, device=device)
            masks[layer_name] = mask
    return masks

def evaluate_model_with_noise(model, test_loader, device, noise_std):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if noise_std > 0:
                # Add Gaussian noise to normalized inputs (simulating sensor/channel noise)
                noise = torch.randn_like(inputs) * noise_std
                inputs = inputs + noise
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def run_evaluation_noise(progenitor, expert_paths, loaders, device, prune_method, prune_ratio, num_bits, per_channel, calibrate, noise_std, activation_scales_cache=None):
    merged_model = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
    target_layers = list(get_target_layers_mapping().keys())
    
    task_accuracies = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    joint_masks = None
    if prune_ratio > 0 and prune_method == 'weight_l1':
        joint_masks = get_weight_l1_masks(merged_model, target_layers, prune_ratio)
            
    for task in tasks:
        task_model = copy.deepcopy(merged_model).to(device)
        task_model = load_checkpoint(task_model, f"./checkpoints/{task}_expert.pth", device)
        
        if prune_ratio > 0:
            if prune_method == 'weight_l1':
                for layer_name, mask in joint_masks.items():
                    apply_structured_pruning_mask(task_model, layer_name, mask)
            elif prune_method == 'activation_task_specific' and activation_scales_cache is not None:
                task_masks = get_activation_task_specific_masks(activation_scales_cache, task, target_layers, prune_ratio, device)
                for layer_name, mask in task_masks.items():
                    apply_structured_pruning_mask(task_model, layer_name, mask)
                    
        if calibrate:
            calibrate_bn(task_model, loaders[task]['cal'], device)
            
        if num_bits is not None:
            apply_weight_quantization(task_model, num_bits, per_channel=per_channel)
            
        test_acc = evaluate_model_with_noise(task_model, loaders[task]['test'], device, noise_std)
        task_accuracies[task] = test_acc
        
    avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies, avg_acc

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for noise robustness sweep: {device}")
    
    # Use fewer workers on CPU to avoid multiprocessing overhead
    loaders = get_dataloaders(batch_size=256, num_workers=2)
    
    expert_paths = [
        "./checkpoints/mnist_expert.pth",
        "./checkpoints/fmnist_expert.pth",
        "./checkpoints/cifar10_expert.pth"
    ]
    
    progenitor = models.resnet18()
    progenitor.fc = nn.Linear(512, 10)
    progenitor = load_checkpoint(progenitor, "./checkpoints/progenitor.pth", device)
    
    target_layers = list(get_target_layers_mapping().keys())
    
    # Collect activation scales using 'variance' (our state-of-the-art metric)
    print("\nCollecting activation scales (Variance)...")
    activation_scales_cache = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        merged_temp = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
        merged_temp = load_checkpoint(merged_temp, f"./checkpoints/{task}_expert.pth", device)
        calibrate_bn(merged_temp, loaders[task]['cal'], device)
        scales = collect_activation_scales(merged_temp, loaders[task]['cal'], device, target_layers, metric='variance')
        activation_scales_cache[task] = scales
        
    # Noise standard deviation levels
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    prune_ratio = 0.3
    
    # We compare:
    # 1. No Pruning + DE-BN
    # 2. Weight L1 (30%)
    # 3. Dynamic ACP (30%)
    # Across both INT8 and INT4 quantization
    
    precision_modes = [
        {'bits': 8, 'name': 'INT8'},
        {'bits': 4, 'name': 'INT4'}
    ]
    
    results = {}
    
    for bits_cfg in precision_modes:
        bits = bits_cfg['bits']
        prec_name = bits_cfg['name']
        results[prec_name] = {}
        
        for method in ['No Pruning + DE-BN', 'Weight L1', 'Dynamic ACP (Variance)']:
            results[prec_name][method] = []
            print(f"\nEvaluating {method} ({prec_name}) across noise levels...")
            
            for noise_std in noise_levels:
                if method == 'No Pruning + DE-BN':
                    p_method = 'none'
                    p_ratio = 0.0
                    cache = None
                elif method == 'Weight L1':
                    p_method = 'weight_l1'
                    p_ratio = prune_ratio
                    cache = None
                else: # Dynamic ACP (Variance)
                    p_method = 'activation_task_specific'
                    p_ratio = prune_ratio
                    cache = activation_scales_cache
                    
                accs, avg = run_evaluation_noise(
                    progenitor, expert_paths, loaders, device,
                    prune_method=p_method, prune_ratio=p_ratio,
                    num_bits=bits, per_channel=True, calibrate=True,
                    noise_std=noise_std, activation_scales_cache=cache
                )
                
                results[prec_name][method].append({
                    'noise': noise_std, 'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
                })
                print(f"  Noise: {noise_std:.2f} | Avg Acc: {avg:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%)")

    # Display results summary
    print("\n" + "="*80)
    print("NOISE ROBUSTNESS EMPIRICAL STUDY SUMMARY (30% Pruning)")
    print("="*80)
    for prec_name in ['INT8', 'INT4']:
        print(f"\n--- {prec_name} PRECISION REGIME ---")
        for method in ['No Pruning + DE-BN', 'Weight L1', 'Dynamic ACP (Variance)']:
            print(f"\nMethod: {method}")
            print(f"  {'Noise Std':<10} | {'MNIST (%)':<10} | {'F-MNIST (%)':<12} | {'CIFAR-10 (%)':<13} | {'Average (%)':<12}")
            print("  " + "-"*65)
            for r in results[prec_name][method]:
                print(f"  {r['noise']:<10.2f} | {r['mnist']:<10.2f} | {r['fmnist']:<12.2f} | {r['cifar10']:<13.2f} | {r['avg']:<12.2f}")

if __name__ == '__main__':
    main()
