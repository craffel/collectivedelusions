import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import argparse
import os
import json
import copy
from torch.utils.data import DataLoader

from src.utils import (
    get_datasets,
    get_resnet18_progenitor,
    evaluate_model,
    estimate_fisher_bn,
    estimate_activation_variance_bn,
    estimate_grad_norm_bn
)
from src.models import patch_bn_to_test_time

def load_expert_state(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict

def merge_backbone_weights(progenitor_state, expert_states, paradigm='wa', lam=0.5):
    """
    Merges backbone weights from multiple experts.
    progenitor_state: dict
    expert_states: list of dicts
    paradigm: 'wa' (Weight Averaging) or 'ta' (Task Arithmetic)
    lam: scaling factor for TA
    """
    merged_state = copy.deepcopy(progenitor_state)
    keys = list(progenitor_state.keys())
    
    # Identify keys to merge (everything except 'fc.')
    keys_to_merge = [k for k in keys if not k.startswith('fc.')]
    
    for k in keys_to_merge:
        # Check if the progenitor weight is floating point; if not, keep it as is
        if not torch.is_floating_point(progenitor_state[k]):
            merged_state[k] = copy.deepcopy(progenitor_state[k])
            continue
            
        if paradigm == 'wa':
            # Simple weight averaging: W_merged = sum(W_i) / M
            stacked_weights = torch.stack([expert_states[i][k] for i in range(len(expert_states))], dim=0)
            merged_state[k] = stacked_weights.mean(dim=0)
            
        elif paradigm == 'ta':
            # Task arithmetic: W_merged = W_0 + lam * sum(W_i - W_0)
            task_vectors = torch.stack([expert_states[i][k].to(progenitor_state[k].dtype) - progenitor_state[k] for i in range(len(expert_states))], dim=0)
            merged_state[k] = progenitor_state[k] + lam * task_vectors.sum(dim=0)
            
    return merged_state

def merge_bn_buffers(merged_model, expert_states, tasks, strategy, weightings=None):
    """
    merged_model: PyTorch model with patched TestTimeBatchNorm2d
    expert_states: dict mapping task_name -> state_dict
    tasks: list of task names
    strategy: 'uniform', 'variance', 'fisher_real', 'fisher_syn', 'grad_norm'
    weightings: dict mapping strategy -> task -> bn_layer_name -> tensor [C]
    """
    M = len(tasks)
    
    # We iterate over modules in the merged model
    for name, module in merged_model.named_modules():
        if module.__class__.__name__ == 'TestTimeBatchNorm2d' or isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            device = module.running_mean.device
            
            # Retrieve running mean and variance from each expert
            running_means = {t: expert_states[t][f"{name}.running_mean"].to(device) for t in tasks}
            running_vars = {t: expert_states[t][f"{name}.running_var"].to(device) for t in tasks}
            
            if strategy == 'uniform':
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += running_means[t]
                    merged_var += running_vars[t]
                module.running_mean.copy_(merged_mean / M)
                module.running_var.copy_(merged_var / M)
                
            elif strategy in ['variance', 'fisher_real', 'fisher_syn', 'grad_norm']:
                # weightings[strategy][task][name] is a tensor of size [C]
                raw_weights = {t: weightings[strategy][t][name].to(device) for t in tasks}
                # Add epsilon to sum
                weight_sum = sum(raw_weights.values()) + 1e-8
                norm_weights = {t: raw_weights[t] / weight_sum for t in tasks}
                
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += norm_weights[t] * running_means[t]
                    merged_var += norm_weights[t] * running_vars[t]
                    
                module.running_mean.copy_(merged_mean)
                module.running_var.copy_(merged_var)

def calculate_weightings(expert_models, loaders, device):
    """
    Computes channel-wise weightings for 'variance', 'fisher_real', 'fisher_syn', and 'grad_norm'
    across tasks.
    Returns a dict mapping strategy_name -> task_name -> layer_name -> weight_tensor [C]
    """
    strategies = ['variance', 'fisher_real', 'fisher_syn', 'grad_norm']
    weightings = {s: {t: {} for t in expert_models} for s in strategies}
    
    for t, model in expert_models.items():
        print(f"Calculating weightings for task {t}...")
        loader = loaders[t]['train']
        
        # Estimate activation variance
        print(f"  Estimating activation variance...")
        var_dict = estimate_activation_variance_bn(model, loader, device, num_batches=15)
        for bn_name, v in var_dict.items():
            weightings['variance'][t][bn_name] = v.cpu()
            
        # Estimate Fisher real
        print(f"  Estimating Fisher (real data)...")
        fisher_real_dict = estimate_fisher_bn(model, loader, device, num_batches=15, use_synthetic=False)
        for bn_name, f_params in fisher_real_dict.items():
            combined = f_params['weight'] + f_params['bias']
            weightings['fisher_real'][t][bn_name] = combined.cpu()
            
        # Estimate Fisher synthetic
        print(f"  Estimating Fisher (synthetic noise)...")
        fisher_syn_dict = estimate_fisher_bn(model, None, device, num_batches=15, use_synthetic=True)
        for bn_name, f_params in fisher_syn_dict.items():
            combined = f_params['weight'] + f_params['bias']
            weightings['fisher_syn'][t][bn_name] = combined.cpu()
            
        # Estimate Grad Norm
        print(f"  Estimating gradient norm...")
        grad_norm_dict = estimate_grad_norm_bn(model, loader, device, num_batches=15)
        for bn_name, g in grad_norm_dict.items():
            weightings['grad_norm'][t][bn_name] = g.cpu()
            
    return weightings

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
    parser = argparse.ArgumentParser(description="Multi-Task Model Merging and test-time BN Calibration sweeps")
    parser.add_argument('--strategy', type=str, required=True, 
                        choices=['uniform', 'variance', 'fisher_real', 'fisher_syn', 'grad_norm'],
                        help="BatchNorm statistics merging strategy")
    parser.add_argument('--paradigm', type=str, default='all', choices=['wa', 'ta', 'all'],
                        help="Merging paradigm to run")
    parser.add_argument('--output', type=str, required=True, help="Path to save result JSON")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load loaders
    loaders = get_datasets(batch_size=128)
    tasks = ['mnist', 'fashion', 'cifar10']
    
    # 2. Load progenitor state dict
    progenitor_state = load_expert_state('checkpoints/progenitor.pt', device)
    
    # 3. Load expert models and state dicts
    expert_states = {}
    expert_models = {}
    for t in tasks:
        ckpt_path = f'checkpoints/expert_{t}.pt'
        expert_states[t] = load_expert_state(ckpt_path, device)
        
        # Instantiate model structure for weight estimation
        model = get_resnet18_progenitor(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(expert_states[t])
        model = model.to(device)
        expert_models[t] = model
        
    # Evaluate individual expert baselines
    print("\n--- Expert Baseline Accuracies ---")
    baselines = {}
    for t in tasks:
        acc = evaluate_model(expert_models[t], loaders[t]['test'], device)
        baselines[t] = acc
        print(f"Expert {t.upper()} accuracy: {acc:.2f}%")
        
    # 4. Calculate Weightings for statistics merging if necessary
    weightings = None
    if args.strategy != 'uniform':
        print("\n--- Calculating non-uniform statistics weightings ---")
        weightings = calculate_weightings(expert_models, loaders, device)
        
    # 5. Define Sweeps
    paradigms = ['wa', 'ta'] if args.paradigm == 'all' else [args.paradigm]
    lambdas = [0.3, 0.5, 0.7, 0.9]
    batch_sizes = [1, 4, 16, 64, 256]
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    noise_levels = [0.0, 0.1, 0.2]  # Test-time noise robustness
    
    results = {
        'baselines': baselines,
        'strategy': args.strategy,
        'sweep': []
    }
    
    # We instantiate a single merged model skeleton, patch it to TestTimeBatchNorm2d
    merged_model = get_resnet18_progenitor(pretrained=False)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = patch_bn_to_test_time(merged_model)
    merged_model = merged_model.to(device)
    
    # Run loops
    for pad in paradigms:
        # For WA, lambda does not affect parameters, so we set it to None or just run once
        lams = lambdas if pad == 'ta' else [1.0]
        for lam in lams:
            print(f"\nMerging weights using paradigm={pad.upper()} (lambda={lam if pad == 'ta' else 'N/A'})...")
            
            # Merge the backbone weights
            merged_backbone = merge_backbone_weights(
                progenitor_state, 
                [expert_states[t] for t in tasks], 
                paradigm=pad, 
                lam=lam
            )
            merged_model.load_state_dict(merged_backbone, strict=False)
            
            # Merge the BatchNorm buffers according to strategy
            merge_bn_buffers(merged_model, expert_states, tasks, args.strategy, weightings)
            
            # Collect the task classification heads' weight states from the experts
            expert_fcs = {t: {k: v for k, v in expert_states[t].items() if k.startswith('fc.')} for t in tasks}
            
            # Evaluate across batch sizes, alphas, and noise levels
            for b in batch_sizes:
                # Pre-create evaluation loaders for this batch size to avoid recreating them in the inner loop
                eval_loaders = {
                    t: DataLoader(loaders[t]['test'].dataset, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
                    for t in tasks
                }
                for alpha in alphas:
                    for noise in noise_levels:
                        # Evaluate on all tasks
                        task_accs = {}
                        for t in tasks:
                            # Re-map fc head keys to match the local 'fc' of merged_model
                            fc_state = {k.replace('fc.', ''): v for k, v in expert_fcs[t].items()}
                            
                            # Attach head, set alpha, add noise, evaluate
                            # We can load the head using merged_model.fc.load_state_dict
                            merged_model.fc.load_state_dict(fc_state)
                            
                            # Set alpha
                            for module in merged_model.modules():
                                if module.__class__.__name__ == 'TestTimeBatchNorm2d':
                                    module.alpha = alpha
                                    
                            acc = evaluate_model_robustness(merged_model, eval_loaders[t], device, noise_std=noise)
                            task_accs[t] = acc
                            
                        avg_acc = sum(task_accs.values()) / len(task_accs)
                        
                        entry = {
                            'paradigm': pad,
                            'lambda': lam if pad == 'ta' else None,
                            'test_batch_size': b,
                            'alpha': alpha,
                            'noise_std': noise,
                            'task_accuracies': task_accs,
                            'avg_accuracy': avg_acc
                        }
                        results['sweep'].append(entry)
                        
                        # Quick console update for standard no-noise settings
                        if noise == 0.0 and b in [1, 64] and alpha in [0.0, 0.4, 1.0]:
                            print(f"  [B={b:3d}, alpha={alpha:.1f}] Acc: {avg_acc:.2f}% (MNIST={task_accs['mnist']:.1f}%, Fashion={task_accs['fashion']:.1f}%, CIFAR={task_accs['cifar10']:.1f}%)")
                            
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nCompleted sweep for strategy {args.strategy}. Results saved to {args.output}")

if __name__ == '__main__':
    main()
