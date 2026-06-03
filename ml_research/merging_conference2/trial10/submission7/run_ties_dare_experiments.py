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

def merge_experts_ta(progenitor, expert_paths, device, lambda_factor=0.3333):
    """
    Standard Task Arithmetic merging.
    """
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
                if not torch.is_floating_point(proj_w):
                    merged_state_dict[key] = proj_w.to(device)
                    continue
                update_sum = torch.zeros_like(proj_w)
                for exp_sd in expert_state_dicts:
                    update_sum += (exp_sd[key].cpu() - proj_w)
                merged_state_dict[key] = (proj_w + lambda_factor * update_sum).to(device)
                
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def merge_experts_ties(progenitor, expert_paths, device, lambda_factor=0.3333, reset_threshold=0.50):
    """
    TIES-Merging: Sparsifies task updates, resolves sign conflicts, and averages agreeing updates.
    """
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
                if not torch.is_floating_point(proj_w):
                    merged_state_dict[key] = proj_w.to(device)
                    continue
                task_vectors = []
                for exp_sd in expert_state_dicts:
                    task_vectors.append(exp_sd[key].cpu() - proj_w)
                
                # Stack: (num_tasks, *param_shape)
                stacked = torch.stack(task_vectors)
                original_shape = stacked.shape
                # Flatten across parameter dimension
                stacked_flat = stacked.view(original_shape[0], -1)
                
                # 1. Sparsification
                k = int(stacked_flat.shape[1] * reset_threshold)
                sparsified_flat = torch.zeros_like(stacked_flat)
                if k > 0:
                    for i in range(original_shape[0]):
                        task_vec = stacked_flat[i]
                        abs_vals = torch.abs(task_vec)
                        threshold = torch.kthvalue(abs_vals, stacked_flat.shape[1] - k + 1)[0]
                        mask = (abs_vals >= threshold)
                        sparsified_flat[i][mask] = task_vec[mask]
                else:
                    sparsified_flat = stacked_flat
                    
                # 2. Sign Election
                signs = torch.sign(sparsified_flat)
                signs_sum = torch.sum(signs, dim=0)
                majority_sign = torch.sign(signs_sum)
                
                # Disagreement Resolution
                filtered_flat = torch.zeros_like(sparsified_flat)
                for i in range(original_shape[0]):
                    agree_mask = (torch.sign(sparsified_flat[i]) == majority_sign) & (majority_sign != 0)
                    filtered_flat[i][agree_mask] = sparsified_flat[i][agree_mask]
                    
                # 3. Disagreement-Aware Averaging
                counts = torch.sum((filtered_flat != 0).float(), dim=0)
                sum_updates = torch.sum(filtered_flat, dim=0)
                merged_update = sum_updates / torch.clamp(counts, min=1.0)
                
                # Add back to progenitor
                merged_state_dict[key] = (proj_w + lambda_factor * merged_update.view(original_shape[1:])).to(device)
                
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def merge_experts_dare(progenitor, expert_paths, device, lambda_factor=0.3333, drop_rate=0.50):
    """
    DARE-Merging: Randomly drops updates, rescales the remaining ones, and averages.
    """
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
                if not torch.is_floating_point(proj_w):
                    merged_state_dict[key] = proj_w.to(device)
                    continue
                task_vectors = []
                for exp_sd in expert_state_dicts:
                    task_vectors.append(exp_sd[key].cpu() - proj_w)
                    
                stacked = torch.stack(task_vectors)
                original_shape = stacked.shape
                stacked_flat = stacked.view(original_shape[0], -1)
                
                # Random masking & rescaling
                # Generate mask on CPU deterministically with seed to ensure reproducibility
                torch.manual_seed(42)
                mask = (torch.rand_like(stacked_flat) > drop_rate).float()
                sparsified_flat = stacked_flat * mask / (1.0 - drop_rate)
                
                merged_update = torch.mean(sparsified_flat, dim=0)
                merged_state_dict[key] = (proj_w + lambda_factor * merged_update.view(original_shape[1:])).to(device)
                
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

def get_activation_joint_masks(activation_scales, target_layers, prune_ratio, device):
    masks = {}
    with torch.no_grad():
        for layer_name in target_layers:
            scales_all_tasks = []
            for task in activation_scales:
                scales_all_tasks.append(activation_scales[task][layer_name])
            stacked_scales = torch.stack(scales_all_tasks)
            max_scales = torch.max(stacked_scales, dim=0)[0].to(device)
            out_channels = max_scales.shape[0]
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(max_scales, k)[0]
                mask = (max_scales > threshold).float()
            else:
                mask = torch.ones(out_channels, device=device)
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
                mask = (task_scales > threshold).float()
            else:
                mask = torch.ones(out_channels, device=device)
            masks[layer_name] = mask
    return masks

def run_evaluation(progenitor, expert_paths, loaders, device, merge_method, prune_method, prune_ratio, num_bits, per_channel, calibrate, activation_scales_cache=None):
    # 1. Merge models using requested algorithm
    if merge_method == 'ta':
        merged_model = merge_experts_ta(progenitor, expert_paths, device, lambda_factor=0.3333)
    elif merge_method == 'ties':
        merged_model = merge_experts_ties(progenitor, expert_paths, device, lambda_factor=0.3333, reset_threshold=0.5)
    elif merge_method == 'dare':
        merged_model = merge_experts_dare(progenitor, expert_paths, device, lambda_factor=0.3333, drop_rate=0.5)
    else:
        raise ValueError(f"Unknown merge method {merge_method}")
        
    target_layers = list(get_target_layers_mapping().keys())
    
    task_accuracies = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    # Pre-compute masks if using joint methods
    joint_masks = None
    if prune_ratio > 0:
        if prune_method == 'weight_l1':
            joint_masks = get_weight_l1_masks(merged_model, target_layers, prune_ratio)
        elif prune_method == 'activation_joint' and activation_scales_cache is not None:
            joint_masks = get_activation_joint_masks(activation_scales_cache, target_layers, prune_ratio, device)
            
    # For each task, evaluate the merged model
    for task in tasks:
        task_model = copy.deepcopy(merged_model).to(device)
        task_model = load_checkpoint(task_model, f"./checkpoints/{task}_expert.pth", device)
        
        # Apply pruning masks
        if prune_ratio > 0:
            if prune_method == 'weight_l1':
                for layer_name, mask in joint_masks.items():
                    apply_structured_pruning_mask(task_model, layer_name, mask)
            elif prune_method == 'activation_joint' and activation_scales_cache is not None:
                for layer_name, mask in joint_masks.items():
                    apply_structured_pruning_mask(task_model, layer_name, mask)
            elif prune_method == 'activation_task_specific' and activation_scales_cache is not None:
                task_masks = get_activation_task_specific_masks(activation_scales_cache, task, target_layers, prune_ratio, device)
                for layer_name, mask in task_masks.items():
                    apply_structured_pruning_mask(task_model, layer_name, mask)
                    
        # Apply DE-BN Calibration
        if calibrate:
            calibrate_bn(task_model, loaders[task]['cal'], device)
            
        # Apply Quantization
        if num_bits is not None:
            apply_weight_quantization(task_model, num_bits, per_channel=per_channel)
            
        test_acc = evaluate_model(task_model, loaders[task]['test'], device)
        task_accuracies[task] = test_acc
        
    avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies, avg_acc

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loaders = get_dataloaders(batch_size=256, num_workers=4)
    
    expert_paths = [
        "./checkpoints/mnist_expert.pth",
        "./checkpoints/fmnist_expert.pth",
        "./checkpoints/cifar10_expert.pth"
    ]
    
    progenitor = models.resnet18()
    progenitor.fc = nn.Linear(512, 10)
    progenitor = load_checkpoint(progenitor, "./checkpoints/progenitor.pth", device)
    
    target_layers = list(get_target_layers_mapping().keys())
    
    # We will evaluate at 30% pruning ratio across TA, TIES, and DARE under FP32, INT8, and INT4 precision levels.
    # Pruning methods: Weight L1 (Baseline), Joint ACP, Dynamic ACP (Ours)
    prune_ratio = 0.30
    merge_methods = ['ta', 'ties', 'dare']
    prune_methods = ['weight_l1', 'activation_joint', 'activation_task_specific']
    precision_modes = [
        {'bits': None, 'name': 'FP32'},
        {'bits': 8, 'name': 'INT8'},
        {'bits': 4, 'name': 'INT4'}
    ]
    
    results = []
    
    # Run loop
    for m_method in merge_methods:
        print("\n" + "="*50)
        print(f"COLLECTING ACTIVATION SCALES FOR MERGE METHOD: {m_method.upper()}")
        print("="*50)
        
        activation_scales_cache = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            if m_method == 'ta':
                merged_temp = merge_experts_ta(progenitor, expert_paths, device, lambda_factor=0.3333)
            elif m_method == 'ties':
                merged_temp = merge_experts_ties(progenitor, expert_paths, device, lambda_factor=0.3333, reset_threshold=0.5)
            elif m_method == 'dare':
                merged_temp = merge_experts_dare(progenitor, expert_paths, device, lambda_factor=0.3333, drop_rate=0.5)
                
            merged_temp = load_checkpoint(merged_temp, f"./checkpoints/{task}_expert.pth", device)
            calibrate_bn(merged_temp, loaders[task]['cal'], device)
            scales = collect_activation_scales(merged_temp, loaders[task]['cal'], device, target_layers)
            activation_scales_cache[task] = scales
            
        print(f"\nRunning evaluations for {m_method.upper()}...")
        for bits_cfg in precision_modes:
            bits = bits_cfg['bits']
            prec_name = bits_cfg['name']
            
            for p_method in prune_methods:
                p_method_name = {
                    'weight_l1': 'Weight L1',
                    'activation_joint': 'Joint ACP (Ours)',
                    'activation_task_specific': 'Dynamic ACP (Ours)'
                }[p_method]
                
                accs, avg = run_evaluation(
                    progenitor, expert_paths, loaders, device,
                    merge_method=m_method, prune_method=p_method, prune_ratio=prune_ratio,
                    num_bits=bits, per_channel=True, calibrate=True,
                    activation_scales_cache=activation_scales_cache
                )
                
                results.append({
                    'merge': m_method.upper(), 'prune': p_method_name, 'ratio': prune_ratio, 'bits': prec_name,
                    'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
                })
                print(f"  {m_method.upper()} | {p_method_name} | {prec_name} | Avg Acc: {avg:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%)")

    # Print out results as markdown table
    print("\n" + "="*80)
    print("CROSS-ALGORITHM EVALUATION RESULTS (30% PRUNING RATIO)")
    print("="*80)
    print("| Merging Algorithm | Pruning Method | Precision | MNIST (%) | F-MNIST (%) | CIFAR-10 (%) | Average (%) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['merge']} | {r['prune']} | {r['bits']} | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |")

    # Save to file
    os.makedirs('./results', exist_ok=True)
    with open('./results/cross_algorithm_results.md', 'w') as f:
        f.write("| Merging Algorithm | Pruning Method | Precision | MNIST (%) | F-MNIST (%) | CIFAR-10 (%) | Average (%) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| {r['merge']} | {r['prune']} | {r['bits']} | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |\n")
            
    print("\nResults saved to ./results/cross_algorithm_results.md")

if __name__ == '__main__':
    main()
