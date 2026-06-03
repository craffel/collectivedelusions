import os
import sys
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils import (
    get_datasets,
    get_resnet18_progenitor,
    estimate_fisher_bn,
    estimate_activation_variance_bn,
    estimate_grad_norm_bn
)
from src.models import patch_bn_to_test_time

def load_expert_state(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict

def merge_backbone_weights(progenitor_state, expert_states, paradigm='ta', lam=0.70):
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

def apply_bn_merging(merged_model, expert_states, tasks, strategy, weightings, temperature=1.0, device='cuda'):
    """
    merged_model: PyTorch model with patched TestTimeBatchNorm2d
    expert_states: dict mapping task_name -> state_dict
    tasks: list of task names
    strategy: 'uniform', 'variance', 'fisher_real', 'fisher_syn', 'grad_norm'
    weightings: dict mapping strategy -> task_name -> layer_name -> tensor [C]
    temperature: only used for 'fisher_real_soft' or 'fisher_syn_soft'
    """
    for name, module in merged_model.named_modules():
        if module.__class__.__name__ == 'TestTimeBatchNorm2d' or isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            
            # Retrieve running mean and variance from each expert
            running_means = {t: expert_states[t][f"{name}.running_mean"].to(device) for t in tasks}
            running_vars = {t: expert_states[t][f"{name}.running_var"].to(device) for t in tasks}
            
            if strategy == 'uniform':
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += running_means[t]
                    merged_var += running_vars[t]
                module.running_mean.copy_(merged_mean / len(tasks))
                module.running_var.copy_(merged_var / len(tasks))
                
            elif strategy in ['variance', 'grad_norm']:
                raw_weights = {t: weightings[strategy][t][name].to(device) for t in tasks}
                weight_sum = sum(raw_weights.values()) + 1e-8
                norm_weights = {t: raw_weights[t] / weight_sum for t in tasks}
                
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += norm_weights[t] * running_means[t]
                    merged_var += norm_weights[t] * running_vars[t]
                module.running_mean.copy_(merged_mean)
                module.running_var.copy_(merged_var)
                
            elif strategy in ['fisher_real', 'fisher_syn']:
                raw_weights = {}
                for t in tasks:
                    f_val = weightings[strategy][t][name].to(device)
                    raw_weights[t] = (f_val + 1e-8) ** temperature
                    
                weight_sum = sum(raw_weights.values()) + 1e-8
                norm_weights = {t: raw_weights[t] / weight_sum for t in tasks}
                
                merged_mean = torch.zeros(C, device=device)
                merged_var = torch.zeros(C, device=device)
                for t in tasks:
                    merged_mean += norm_weights[t] * running_means[t]
                    merged_var += norm_weights[t] * running_vars[t]
                module.running_mean.copy_(merged_mean)
                module.running_var.copy_(merged_var)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable non-cudnn to prevent errors
    torch.backends.cudnn.enabled = False
    
    loaders = get_datasets(batch_size=128)
    tasks = ['mnist', 'fashion', 'cifar10']
    
    # Load expert models and progenitor
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
        
    print("\n--- Calculating Raw Weightings ---")
    raw_weightings = {
        'variance': {t: {} for t in tasks},
        'grad_norm': {t: {} for t in tasks},
        'fisher_real': {t: {} for t in tasks},
        'fisher_syn': {t: {} for t in tasks}
    }
    
    for t, model in expert_models.items():
        print(f"Calculating weightings for task {t}...")
        loader = loaders[t]['train']
        
        # Activation Variance
        v_dict = estimate_activation_variance_bn(model, loader, device, num_batches=15)
        for bn_name, v_tensor in v_dict.items():
            raw_weightings['variance'][t][bn_name] = v_tensor.cpu()
            
        # Gradient Norm
        gn_dict = estimate_grad_norm_bn(model, loader, device, num_batches=15)
        for bn_name, gn_tensor in gn_dict.items():
            raw_weightings['grad_norm'][t][bn_name] = gn_tensor.cpu()
            
        # Real Fisher
        f_real = estimate_fisher_bn(model, loader, device, num_batches=15, use_synthetic=False)
        for bn_name, f_params in f_real.items():
            raw_weightings['fisher_real'][t][bn_name] = (f_params['weight'] + f_params['bias']).cpu()
            
        # Synthetic Fisher
        f_syn = estimate_fisher_bn(model, None, device, num_batches=15, use_synthetic=True)
        for bn_name, f_params in f_syn.items():
            raw_weightings['fisher_syn'][t][bn_name] = (f_params['weight'] + f_params['bias']).cpu()
            
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
    
    # We will evaluate activation variance using CIFAR-10 test set images.
    # Feed a batch of size 128
    cifar_test_loader = DataLoader(loaders['cifar10']['test'].dataset, batch_size=128, shuffle=False)
    eval_batch, _ = next(iter(cifar_test_loader))
    eval_batch = eval_batch.to(device)
    
    # Define strategies we want to analyze
    # mapping label -> (strategy_name, temperature)
    analysis_configs = {
        'Uniform': ('uniform', 1.0),
        'Activation Variance': ('variance', 1.0),
        'Gradient Norm': ('grad_norm', 1.0),
        'Real Fisher (T=1.0)': ('fisher_real', 1.0),
        'Real Fisher (T=0.1)': ('fisher_real', 0.1),
        'Synthetic Fisher (T=1.0)': ('fisher_syn', 1.0),
        'Synthetic Fisher (T=0.2)': ('fisher_syn', 0.2)
    }
    
    layer_variance_results = {label: [] for label in analysis_configs}
    layer_names = []
    
    # We will identify layer names in forward order
    test_bn_modules = []
    for name, module in merged_model.named_modules():
        if module.__class__.__name__ == 'TestTimeBatchNorm2d':
            layer_names.append(name)
            test_bn_modules.append((name, module))
            
    print(f"Found {len(layer_names)} TestTimeBatchNorm2d layers.")
    
    for label, (strat, temp) in analysis_configs.items():
        print(f"\nAnalyzing strategy: {label}...")
        
        # Apply the merged statistics to the BatchNorm layers
        apply_bn_merging(merged_model, expert_states, tasks, strat, raw_weightings, temperature=temp, device=device)
        
        # Ensure we are in static evaluation mode (alpha = 0.0) so we measure representation variance
        # without test-time calibration updating it.
        for module in merged_model.modules():
            if module.__class__.__name__ == 'TestTimeBatchNorm2d':
                module.alpha = 0.0
                
        # Register forward hooks to capture variances
        current_variances = {}
        hooks = []
        
        def make_hook(lyr_name):
            def hook(module, inp, out):
                with torch.no_grad():
                    # Compute channel-wise variances and average across channels
                    # out has shape [B, C, H, W]
                    ch_vars = out.var(dim=(0, 2, 3), unbiased=False)
                    avg_var = ch_vars.mean().item()
                    current_variances[lyr_name] = avg_var
            return hook
            
        for name, module in test_bn_modules:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
            
        # Perform single forward pass
        merged_model.eval()
        with torch.no_grad():
            _ = merged_model(eval_batch)
            
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Store results in ordered list
        for name in layer_names:
            layer_variance_results[label].append(current_variances[name])
            
    # Save the results to JSON
    output_data = {
        'layer_names': layer_names,
        'layer_indices': list(range(1, len(layer_names) + 1)),
        'variances': layer_variance_results
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/layer_variance.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    print("\nSaved results/layer_variance.json")
    
    # Let's generate a beautiful plot!
    plt.figure(figsize=(8, 5))
    
    # We will use distinct styles and colors for clarity
    plot_styles = {
        'Uniform': ('C0', '-', 'o'),
        'Activation Variance': ('C1', '--', 's'),
        'Gradient Norm': ('C4', '-.', 'd'),
        'Real Fisher (T=1.0)': ('C2', ':', '^'),
        'Real Fisher (T=0.1)': ('green', '-', '*'),
        'Synthetic Fisher (T=1.0)': ('C3', ':', 'x'),
        'Synthetic Fisher (T=0.2)': ('red', '-', 'P')
    }
    
    indices = output_data['layer_indices']
    
    for label, variances in layer_variance_results.items():
        color, linestyle, marker = plot_styles[label]
        plt.plot(indices, variances, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6)
        
    plt.xlabel('BatchNorm Layer Index (Depth $\\rightarrow$)', fontsize=11, fontweight='bold')
    plt.ylabel('Average Activation Variance', fontsize=11, fontweight='bold')
    plt.title('Layer-Wise Activation Variance Deconstruction of Merged Models', fontsize=12, fontweight='bold', pad=12)
    plt.yscale('log') # Log scale is essential due to exponential decay
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(indices, [str(i) for i in indices])
    plt.legend(loc='lower left', frameon=True, fontsize=9, ncol=2)
    plt.tight_layout()
    
    plt.savefig('fig_layer_variance.png', dpi=300)
    plt.savefig('fig_layer_variance.pdf', dpi=300)
    plt.close()
    print("Saved fig_layer_variance.png and fig_layer_variance.pdf")

if __name__ == '__main__':
    main()
