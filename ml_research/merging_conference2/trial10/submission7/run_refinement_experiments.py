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

def get_random_masks(model, target_layers, prune_ratio, seed=42):
    """
    Computes structured pruning masks based on random channel selection.
    """
    masks = {}
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    with torch.no_grad():
        for layer_name in target_layers:
            module = dict(model.named_modules())[layer_name]
            out_channels = module.weight.shape[0]
            scores = torch.rand(out_channels, generator=g)
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(scores, k)[0]
                mask = (scores > threshold).float().to(module.weight.device)
            else:
                mask = torch.ones(out_channels, device=module.weight.device)
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

def run_evaluation(progenitor, expert_paths, loaders, device, prune_method, prune_ratio, num_bits, per_channel, calibrate, activation_scales_cache=None):
    merged_model = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
    target_layers = list(get_target_layers_mapping().keys())
    
    task_accuracies = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    joint_masks = None
    if prune_ratio > 0 and prune_method == 'random':
        joint_masks = get_random_masks(merged_model, target_layers, prune_ratio, seed=42)
            
    for task in tasks:
        task_model = copy.deepcopy(merged_model).to(device)
        task_model = load_checkpoint(task_model, f"./checkpoints/{task}_expert.pth", device)
        
        if prune_ratio > 0:
            if prune_method == 'random':
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
            
        test_acc = evaluate_model(task_model, loaders[task]['test'], device)
        task_accuracies[task] = test_acc
        
    avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies, avg_acc

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for refinement: {device}")
    
    loaders = get_dataloaders(batch_size=256, num_workers=4)
    
    expert_paths = [
        "./checkpoints/mnist_expert.pth",
        "./checkpoints/fmnist_expert.pth",
        "./checkpoints/cifar10_expert.pth"
    ]
    
    progenitor = models.resnet18()
    progenitor.fc = nn.Linear(512, 10)
    progenitor = load_checkpoint(progenitor, "./checkpoints/progenitor.pth", device)
    
    # -------------------------------------------------------------
    # Step 1: Run Random Structured Pruning sweeps
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("RUNNING RANDOM STRUCTURED PRUNING BASELINE")
    print("="*50)
    
    random_results = []
    pruning_ratios = [0.1, 0.3, 0.5]
    precision_modes = [
        {'bits': None, 'name': 'FP32'},
        {'bits': 8, 'name': 'INT8'},
        {'bits': 4, 'name': 'INT4'}
    ]
    
    for bits_cfg in precision_modes:
        bits = bits_cfg['bits']
        prec_name = bits_cfg['name']
        print(f"\nRandom Pruning under {prec_name} precision:")
        for ratio in pruning_ratios:
            accs, avg = run_evaluation(
                progenitor, expert_paths, loaders, device,
                prune_method='random', prune_ratio=ratio,
                num_bits=bits, per_channel=True, calibrate=True
            )
            random_results.append({
                'ratio': ratio, 'bits': prec_name,
                'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
            })
            print(f"  Random | Ratio: {ratio*100:.0f}% | Avg Acc: {avg:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%)")
            
    # -------------------------------------------------------------
    # Step 2: Run Activation Metric Ablations (L2 and Variance) under INT4 Dynamic ACP
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("RUNNING ACTIVATION METRIC ABLATIONS UNDER INT4")
    print("="*50)
    
    ablation_results = {}
    target_layers = list(get_target_layers_mapping().keys())
    
    metrics = ['l2', 'variance']
    for metric in metrics:
        print(f"\nCollecting activation scales using '{metric}' metric...")
        activation_scales_cache = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            merged_temp = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
            merged_temp = load_checkpoint(merged_temp, f"./checkpoints/{task}_expert.pth", device)
            calibrate_bn(merged_temp, loaders[task]['cal'], device)
            scales = collect_activation_scales(merged_temp, loaders[task]['cal'], device, target_layers, metric=metric)
            activation_scales_cache[task] = scales
            
        ablation_results[metric] = []
        for ratio in pruning_ratios:
            accs, avg = run_evaluation(
                progenitor, expert_paths, loaders, device,
                prune_method='activation_task_specific', prune_ratio=ratio,
                num_bits=4, per_channel=True, calibrate=True,
                activation_scales_cache=activation_scales_cache
            )
            ablation_results[metric].append({
                'ratio': ratio, 'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
            })
            print(f"  Dynamic ACP ({metric.upper()}) | Ratio: {ratio*100:.0f}% | Avg Acc: {avg:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%)")

    # -------------------------------------------------------------
    # Step 3: Write outputs in a markdown-friendly summary
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("SUMMARY OF NEW RESULTS")
    print("="*50)
    print("\nRandom Pruning Main Table Extensions:")
    for r in random_results:
        print(f"Random | {r['ratio']*100:.0f}% | {r['bits']} | Yes | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |")
        
    print("\nActivation Metric Ablation Table (INT4, Dynamic ACP):")
    for metric in metrics:
        for r in ablation_results[metric]:
            print(f"Dynamic ACP ({metric.upper()}) | {r['ratio']*100:.0f}% | INT4 | Yes | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |")

if __name__ == '__main__':
    main()
