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
    """
    Merges task-specific experts with a progenitor model using Task Arithmetic:
    theta_merged = theta_init + lambda * sum(theta_t - theta_init)
    """
    merged_model = copy.deepcopy(progenitor).to(device)
    merged_state_dict = merged_model.state_dict()
    progenitor_state_dict = progenitor.state_dict()
    
    # Load all expert state dicts on CPU to perform safe device-agnostic arithmetic
    expert_state_dicts = []
    for path in expert_paths:
        expert_state_dicts.append(torch.load(path, map_location='cpu'))
        
    # Merge backbone parameters (exclude heads since they are task-specific and kept separate)
    with torch.no_grad():
        for key in merged_state_dict.keys():
            # Only merge backbone weights, do not merge 'fc.' parameters which are task-specific heads
            if 'fc.' not in key:
                proj_w = progenitor_state_dict[key].cpu()
                update_sum = torch.zeros_like(proj_w)
                for exp_sd in expert_state_dicts:
                    update_sum += (exp_sd[key].cpu() - proj_w)
                merged_state_dict[key] = (proj_w + lambda_factor * update_sum).to(device)
                
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def get_weight_l1_masks(model, target_layers, prune_ratio):
    """
    Computes structured pruning masks based on the L1 norm of the convolutional weights.
    """
    masks = {}
    with torch.no_grad():
        for layer_name in target_layers:
            module = dict(model.named_modules())[layer_name]
            weight = module.weight  # shape: (out_channels, in_channels, K, K)
            out_channels = weight.shape[0]
            
            # Compute L1 norm along out_channels (dim 0)
            l1_norms = torch.sum(torch.abs(weight.view(out_channels, -1)), dim=1)
            
            # Find threshold for pruning
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(l1_norms, k)[0]
                mask = (l1_norms > threshold).float()
            else:
                mask = torch.ones(out_channels, device=weight.device)
            masks[layer_name] = mask
    return masks

def get_activation_joint_masks(activation_scales, target_layers, prune_ratio, device):
    """
    Computes structured pruning masks based on the maximum activation scale across all tasks (Joint ACP).
    """
    masks = {}
    with torch.no_grad():
        for layer_name in target_layers:
            # Get activation scales for this layer across all tasks
            scales_all_tasks = []
            for task in activation_scales:
                scales_all_tasks.append(activation_scales[task][layer_name])
                
            # Compute maximum scale across tasks
            stacked_scales = torch.stack(scales_all_tasks) # shape: (num_tasks, out_channels)
            max_scales = torch.max(stacked_scales, dim=0)[0].to(device)
            out_channels = max_scales.shape[0]
            
            # Find threshold for pruning
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(max_scales, k)[0]
                mask = (max_scales > threshold).float()
            else:
                mask = torch.ones(out_channels, device=device)
            masks[layer_name] = mask
    return masks

def get_activation_task_specific_masks(activation_scales, task_name, target_layers, prune_ratio, device):
    """
    Computes structured pruning masks based on task-specific activation scale (Dynamic Task-Specific ACP).
    """
    masks = {}
    with torch.no_grad():
        for layer_name in target_layers:
            task_scales = activation_scales[task_name][layer_name].to(device)
            out_channels = task_scales.shape[0]
            
            # Find threshold for pruning
            k = int(out_channels * prune_ratio)
            if k > 0:
                threshold = torch.kthvalue(task_scales, k)[0]
                mask = (task_scales > threshold).float()
            else:
                mask = torch.ones(out_channels, device=device)
            masks[layer_name] = mask
    return masks

def run_evaluation(progenitor, expert_paths, loaders, device, prune_method, prune_ratio, num_bits, per_channel, calibrate, activation_scales_cache=None):
    """
    Runs model merging, applies pruning, applies fake quantization, optionally calibrates BN,
    and returns task-specific and average test accuracies.
    """
    # 1. Merge models using Task Arithmetic (lambda = 0.3333 is equivalent to Weight Averaging)
    merged_model = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
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
        # Clone the merged model so we don't pollute the weights for the next task
        task_model = copy.deepcopy(merged_model).to(device)
        
        # Load task-specific classification head
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
                    
        # Apply Data-Efficient BatchNorm Calibration if requested
        if calibrate:
            calibrate_bn(task_model, loaders[task]['cal'], device)
            
        # Apply Post-Training Quantization (Fake Quantization)
        if num_bits is not None:
            apply_weight_quantization(task_model, num_bits, per_channel=per_channel)
            
        # Evaluate performance on test set
        test_acc = evaluate_model(task_model, loaders[task]['test'], device)
        task_accuracies[task] = test_acc
        
    avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies, avg_acc

def main():
    # Disable cuDNN to bypass driver compatibility issues as noted in Paper 9
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data loaders
    loaders = get_dataloaders(batch_size=256, num_workers=4)
    
    # 2. Check that expert checkpoints exist
    expert_paths = [
        "./checkpoints/mnist_expert.pth",
        "./checkpoints/fmnist_expert.pth",
        "./checkpoints/cifar10_expert.pth"
    ]
    for p in expert_paths + ["./checkpoints/progenitor.pth"]:
        if not os.path.exists(p):
            print(f"Error: Checkpoint {p} not found. Please run train_experts.py first!")
            return
            
    progenitor = models.resnet18()
    progenitor.fc = nn.Linear(512, 10)
    progenitor = load_checkpoint(progenitor, "./checkpoints/progenitor.pth", device)
    
    # 3. Collect activation scales for our proposed activation-aware pruning methods
    print("\nCollecting activation scales for ACP-QMM on calibration sets...")
    activation_scales_cache = {}
    target_layers = list(get_target_layers_mapping().keys())
    
    for task in ['mnist', 'fmnist', 'cifar10']:
        # To collect activations accurately, we load the merged model with the task's expert head and calibrate BN
        merged_temp = merge_experts(progenitor, expert_paths, device, lambda_factor=0.3333)
        merged_temp = load_checkpoint(merged_temp, f"./checkpoints/{task}_expert.pth", device)
        calibrate_bn(merged_temp, loaders[task]['cal'], device)
        
        # Collect activations on the calibration batch
        scales = collect_activation_scales(merged_temp, loaders[task]['cal'], device, target_layers)
        activation_scales_cache[task] = scales
    print("Activation scales successfully collected!")
    
    # 4. Experimental Run Configurations
    # We evaluate combinations of:
    # - Pruning Method: None, Weight L1 (baseline), Activation Joint (ours), Activation Task-Specific (ours)
    # - Pruning Ratios: 0.1, 0.3, 0.5
    # - Quantization: FP32, INT8 (per-channel), INT4 (per-channel)
    # - Calibration: DE-BN (Yes vs No)
    
    # We will format this into a highly readable table.
    results = []
    
    print("\nStarting experimental sweeps...")
    
    # Baseline 1: Unpruned, Unquantized, No BN Calibration
    print("\nEvaluating Baseline (No Pruning, FP32, No Calibration)...")
    accs, avg = run_evaluation(progenitor, expert_paths, loaders, device, prune_method='none', prune_ratio=0.0, num_bits=None, per_channel=False, calibrate=False)
    results.append({
        'method': 'No Pruning (WA)', 'ratio': 0.0, 'bits': 'FP32', 'calibrate': 'No',
        'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
    })
    
    # Baseline 2: Unpruned, Unquantized, With DE-BN Calibration
    print("Evaluating Baseline (No Pruning, FP32, With DE-BN Calibration)...")
    accs, avg = run_evaluation(progenitor, expert_paths, loaders, device, prune_method='none', prune_ratio=0.0, num_bits=None, per_channel=False, calibrate=True)
    results.append({
        'method': 'No Pruning + DE-BN', 'ratio': 0.0, 'bits': 'FP32', 'calibrate': 'Yes',
        'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
    })
    
    # Sweep configurations
    pruning_ratios = [0.1, 0.3, 0.5]
    pruning_methods = ['weight_l1', 'activation_joint', 'activation_task_specific']
    precision_modes = [
        {'bits': None, 'name': 'FP32'},
        {'bits': 8, 'name': 'INT8'},
        {'bits': 4, 'name': 'INT4'}
    ]
    
    for bits_cfg in precision_modes:
        bits = bits_cfg['bits']
        prec_name = bits_cfg['name']
        print(f"\nEvaluating sweeps under {prec_name} precision...")
        
        for ratio in pruning_ratios:
            for method in pruning_methods:
                method_name = {
                    'weight_l1': 'Weight L1',
                    'activation_joint': 'Joint ACP (Ours)',
                    'activation_task_specific': 'Dynamic ACP (Ours)'
                }[method]
                
                # We always run with DE-BN = True as it is a crucial pragmatist standard for merged models
                accs, avg = run_evaluation(
                    progenitor, expert_paths, loaders, device,
                    prune_method=method, prune_ratio=ratio,
                    num_bits=bits, per_channel=True, calibrate=True,
                    activation_scales_cache=activation_scales_cache
                )
                
                results.append({
                    'method': method_name, 'ratio': ratio, 'bits': prec_name, 'calibrate': 'Yes',
                    'mnist': accs['mnist'], 'fmnist': accs['fmnist'], 'cifar10': accs['cifar10'], 'avg': avg
                })
                print(f"  {method_name} | Ratio: {ratio} | Bits: {prec_name} | Avg Acc: {avg:.2f}%")
                
    # Print Markdown Table of Results
    print("\n" + "="*80)
    print("FINAL EXPERIMENTAL RESULTS")
    print("="*80)
    
    # Print table header
    print(f"| Pruning Method | Pruning Ratio | Precision | DE-BN | MNIST (%) | F-MNIST (%) | CIFAR-10 (%) | Average (%) |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        ratio_str = f"{r['ratio']*100:.0f}%" if r['ratio'] > 0 else "-"
        print(f"| {r['method']} | {ratio_str} | {r['bits']} | {r['calibrate']} | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |")
        
    # Also write results to a text file for paper compilation
    os.makedirs('./results', exist_ok=True)
    with open('./results/results_table.md', 'w') as f:
        f.write(f"| Pruning Method | Pruning Ratio | Precision | DE-BN | MNIST (%) | F-MNIST (%) | CIFAR-10 (%) | Average (%) |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            ratio_str = f"{r['ratio']*100:.0f}%" if r['ratio'] > 0 else "-"
            f.write(f"| {r['method']} | {ratio_str} | {r['bits']} | {r['calibrate']} | {r['mnist']:.2f} | {r['fmnist']:.2f} | {r['cifar10']:.2f} | {r['avg']:.2f} |\n")
    print("\nResults table saved to ./results/results_table.md")

if __name__ == '__main__':
    main()
