import os
import sys
import copy
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import (
    get_datasets,
    get_resnet18_progenitor,
    evaluate_model,
    estimate_fisher_bn
)
from src.models import patch_bn_to_test_time

def load_expert_state(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict

def merge_backbone_weights(progenitor_state, expert_states, paradigm='wa', lam=0.5):
    merged_state = copy.deepcopy(progenitor_state)
    keys = list(progenitor_state.keys())
    keys_to_merge = [k for k in keys if not k.startswith('fc.')]
    
    for k in keys_to_merge:
        if not torch.is_floating_point(progenitor_state[k]):
            merged_state[k] = copy.deepcopy(progenitor_state[k])
            continue
            
        if paradigm == 'wa':
            stacked_weights = torch.stack([expert_states[i][k] for i in range(len(expert_states))], dim=0)
            merged_state[k] = stacked_weights.mean(dim=0)
        elif paradigm == 'ta':
            task_vectors = torch.stack([expert_states[i][k].to(progenitor_state[k].dtype) - progenitor_state[k] for i in range(len(expert_states))], dim=0)
            merged_state[k] = progenitor_state[k] + lam * task_vectors.sum(dim=0)
            
    return merged_state

def merge_bn_buffers_temp(merged_model, expert_states, tasks, fisher_dict_raw, strategy, temperature, device):
    """
    fisher_dict_raw: dict mapping task -> layer_name -> tensor [C]
    """
    for name, module in merged_model.named_modules():
        if module.__class__.__name__ == 'TestTimeBatchNorm2d' or isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            
            # Retrieve running mean and variance from each expert
            running_means = {t: expert_states[t][f"{name}.running_mean"].to(device) for t in tasks}
            running_vars = {t: expert_states[t][f"{name}.running_var"].to(device) for t in tasks}
            
            if temperature == 0.0:
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += running_means[t]
                    merged_var += running_vars[t]
                module.running_mean.copy_(merged_mean / len(tasks))
                module.running_var.copy_(merged_var / len(tasks))
            else:
                raw_weights = {}
                for t in tasks:
                    # Raw Fisher value
                    f_val = fisher_dict_raw[t][name].to(device)
                    # Apply temperature scaling: (F + eps)^T
                    raw_weights[t] = (f_val + 1e-8) ** temperature
                
                # Normalize weights
                weight_sum = sum(raw_weights.values()) + 1e-8
                norm_weights = {t: raw_weights[t] / weight_sum for t in tasks}
                
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += norm_weights[t] * running_means[t]
                    merged_var += norm_weights[t] * running_vars[t]
                    
                module.running_mean.copy_(merged_mean)
                module.running_var.copy_(merged_var)

def evaluate_model_robustness(model, dataloader, device, noise_std=0.0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            if noise_std > 0.0:
                inputs = inputs + torch.randn_like(inputs) * noise_std
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if total >= 2000:
                break
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Sweep temperature parameter T for Fisher weighting")
    parser.add_argument('--output', type=str, default='results/temperature_sweep.json', help="Path to save result JSON")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable non-cudnn if on Hopper to prevent errors
    torch.backends.cudnn.enabled = False
    
    loaders = get_datasets(batch_size=128)
    tasks = ['mnist', 'fashion', 'cifar10']
    
    progenitor_state = load_expert_state('checkpoints/progenitor.pt', device)
    
    expert_states = {}
    expert_models = {}
    for t in tasks:
        ckpt_path = f'checkpoints/expert_{t}.pt'
        expert_states[t] = load_expert_state(ckpt_path, device)
        
        model = get_resnet18_progenitor(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(expert_states[t])
        model = model.to(device)
        expert_models[t] = model
        
    print("\n--- Calculating raw Fisher values ---")
    raw_fisher = {
        'fisher_real': {t: {} for t in tasks},
        'fisher_syn': {t: {} for t in tasks}
    }
    
    for t, model in expert_models.items():
        print(f"Calculating raw Fisher for task {t}...")
        loader = loaders[t]['train']
        
        # Real Fisher
        f_real = estimate_fisher_bn(model, loader, device, num_batches=15, use_synthetic=False)
        for bn_name, f_params in f_real.items():
            raw_fisher['fisher_real'][t][bn_name] = (f_params['weight'] + f_params['bias']).cpu()
            
        # Synthetic Fisher
        f_syn = estimate_fisher_bn(model, None, device, num_batches=15, use_synthetic=True)
        for bn_name, f_params in f_syn.items():
            raw_fisher['fisher_syn'][t][bn_name] = (f_params['weight'] + f_params['bias']).cpu()
            
    # Set up the merged model
    merged_model = get_resnet18_progenitor(pretrained=False)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = patch_bn_to_test_time(merged_model)
    merged_model = merged_model.to(device)
    
    # Merge backbone using TA (lambda=0.70)
    print("\nMerging weights using paradigm=TA (lambda=0.70)...")
    merged_backbone = merge_backbone_weights(
        progenitor_state,
        [expert_states[t] for t in tasks],
        paradigm='ta',
        lam=0.70
    )
    merged_model.load_state_dict(merged_backbone, strict=False)
    expert_fcs = {t: {k: v for k, v in expert_states[t].items() if k.startswith('fc.')} for t in tasks}
    
    # Define Sweep Parameters
    strategies = ['fisher_syn', 'fisher_real']
    temperatures = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    batch_sizes = [1, 4, 16, 64, 256]
    alphas = [0.0, 0.4, 0.8, 1.0]
    
    sweep_results = []
    
    for b in batch_sizes:
        print(f"\nEvaluating batch size B={b}...")
        eval_loaders = {
            t: DataLoader(loaders[t]['test'].dataset, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
            for t in tasks
        }
        
        for strat in strategies:
            for temp in temperatures:
                # Merge BN buffers with this temperature
                merge_bn_buffers_temp(merged_model, expert_states, tasks, raw_fisher[strat], strat, temp, device)
                
                for alpha in alphas:
                    task_accs = {}
                    for t in tasks:
                        fc_state = {k.replace('fc.', ''): v for k, v in expert_fcs[t].items()}
                        merged_model.fc.load_state_dict(fc_state)
                        
                        for module in merged_model.modules():
                            if module.__class__.__name__ == 'TestTimeBatchNorm2d':
                                module.alpha = alpha
                                
                        acc = evaluate_model_robustness(merged_model, eval_loaders[t], device, noise_std=0.0)
                        task_accs[t] = acc
                        
                    avg_acc = sum(task_accs.values()) / len(task_accs)
                    
                    entry = {
                        'strategy': strat,
                        'temperature': temp,
                        'test_batch_size': b,
                        'alpha': alpha,
                        'task_accuracies': task_accs,
                        'avg_accuracy': avg_acc
                    }
                    sweep_results.append(entry)
                    
                    # Log interesting entries to console (e.g., B=1 or B=256, and alpha=0.0 or alpha=0.8)
                    if b in [1, 256] and alpha in [0.0, 0.8] and temp in [0.0, 0.1, 0.2, 1.0]:
                        print(f"  [{strat}, T={temp:.2f}, B={b:3d}, alpha={alpha:.1f}] Acc: {avg_acc:.2f}% (MNIST={task_accs['mnist']:.1f}%, Fashion={task_accs['fashion']:.1f}%, CIFAR={task_accs['cifar10']:.1f}%)")
                        
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(sweep_results, f, indent=4)
        
    print(f"\nSaved temperature sweep results to {args.output}")

if __name__ == '__main__':
    main()
