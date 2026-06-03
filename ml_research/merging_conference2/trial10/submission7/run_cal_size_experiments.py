import os
import copy
import torch
import torch.nn as nn
import torchvision.models as models
from utils import (
    get_dataloaders,
    evaluate_model,
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

def run_evaluation_with_cal_size(progenitor, expert_paths, loaders, device, prune_ratio, num_bits, per_channel, metric, activation_scales_cache):
    merged_model = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
    target_layers = list(get_target_layers_mapping().keys())
    
    task_accuracies = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    for task in tasks:
        task_model = copy.deepcopy(merged_model).to(device)
        task_model = load_checkpoint(task_model, f"./checkpoints/{task}_expert.pth", device)
        
        # Pruning
        task_masks = get_activation_task_specific_masks(activation_scales_cache, task, target_layers, prune_ratio, device)
        for layer_name, mask in task_masks.items():
            apply_structured_pruning_mask(task_model, layer_name, mask)
            
        # Calibration
        calibrate_bn(task_model, loaders[task]['cal'], device)
        
        # Quantization
        if num_bits is not None:
            apply_weight_quantization(task_model, num_bits, per_channel=per_channel)
            
        test_acc = evaluate_model(task_model, loaders[task]['test'], device)
        task_accuracies[task] = test_acc
        
    avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies, avg_acc

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for calibration size sweep: {device}")
    
    expert_paths = [
        "./checkpoints/mnist_expert.pth",
        "./checkpoints/fmnist_expert.pth",
        "./checkpoints/cifar10_expert.pth"
    ]
    
    progenitor = models.resnet18()
    progenitor.fc = nn.Linear(512, 10)
    progenitor = load_checkpoint(progenitor, "./checkpoints/progenitor.pth", device)
    
    cal_sizes = [16, 32, 64, 128]
    prune_ratio = 0.3
    num_bits = 4
    metrics = ['l1', 'variance']
    target_layers = list(get_target_layers_mapping().keys())
    
    print("\n" + "="*50)
    print("RUNNING CALIBRATION SIZE SWEEPS (30% Pruning, INT4)")
    print("="*50)
    
    results = {}
    for metric in metrics:
        results[metric] = []
        for cal_size in cal_sizes:
            print(f"\nEvaluating metric: {metric.upper()} with calibration size N = {cal_size}...")
            # Load loaders with this specific calibration size
            loaders = get_dataloaders(batch_size=256, num_workers=4, cal_size=cal_size)
            
            # Collect activation scales
            activation_scales_cache = {}
            for task in ['mnist', 'fmnist', 'cifar10']:
                merged_temp = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
                merged_temp = load_checkpoint(merged_temp, f"./checkpoints/{task}_expert.pth", device)
                calibrate_bn(merged_temp, loaders[task]['cal'], device)
                scales = collect_activation_scales(merged_temp, loaders[task]['cal'], device, target_layers, metric=metric)
                activation_scales_cache[task] = scales
                
            accs, avg = run_evaluation_with_cal_size(
                progenitor, expert_paths, loaders, device,
                prune_ratio=prune_ratio, num_bits=num_bits, per_channel=True,
                metric=metric, activation_scales_cache=activation_scales_cache
            )
            results[metric].append({
                'cal_size': cal_size,
                'mnist': accs['mnist'],
                'fmnist': accs['fmnist'],
                'cifar10': accs['cifar10'],
                'avg': avg
            })
            print(f"  N = {cal_size} | Avg Acc: {avg:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%)")
            
    print("\n" + "="*50)
    print("FINAL SUMMARY OF CALIBRATION SIZE SENSITIVITY")
    print("="*50)
    for metric in metrics:
        print(f"\nMetric: {metric.upper()}")
        print("| N | MNIST (%) | F-MNIST (%) | CIFAR-10 (%) | Average (%) |")
        print("|---|---|---|---|---|")
        for r in results[metric]:
            print(f"| {r['cal_size']} | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |")

if __name__ == '__main__':
    main()
