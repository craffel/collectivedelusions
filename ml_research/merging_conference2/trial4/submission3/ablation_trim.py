import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import helper utilities from existing scripts
from experiment import (
    set_seed,
    get_datasets,
    evaluate_model,
    get_bn_modules,
    merge_models,
    get_calibration_sets,
    CalibrationHook,
    resnet18
)
from robust_experiments import CorruptedDataset

def run_ablation_calibration_and_fusion(base_model, expert_models, merged_model, calib_sets, N=128, mode="TAAC", trim_fraction=0.25, device="cuda"):
    expert_names = list(calib_sets.keys())
    M = len(expert_models)
    
    # Copy merged model to perform ZIO-CF weight fusion
    fused_model = copy.deepcopy(merged_model).to(device)
    fused_model.eval()
    
    # Copy merged model for online hooks to evaluate mathematical equivalence
    hooked_model = copy.deepcopy(merged_model).to(device)
    hooked_model.eval()
    online_hooks = []
    
    # Get BatchNorm modules in sequential order
    bn_fused = get_bn_modules(fused_model)
    bn_hooked = get_bn_modules(hooked_model)
    
    # Place experts on device for statistic collection
    dev_experts = [copy.deepcopy(m).to(device).eval() for m in expert_models]
    
    # Setup loaders for calibration
    cal_loaders = {name: DataLoader(subset, batch_size=N, shuffle=False) for name, subset in calib_sets.items()}
    
    # Pooled joint dataset
    joint_dataset = torch.utils.data.ConcatDataset([calib_sets[name] for name in expert_names])
    joint_loader = DataLoader(joint_dataset, batch_size=N * M, shuffle=False)
    
    epsilon = 1e-5
    
    # Sequential calibration loop
    for i in range(len(bn_fused)):
        name, bn_layer_fused = bn_fused[i]
        _, bn_layer_hooked = bn_hooked[i]
        
        expert_stds = []
        expert_means = []
        
        # Capture intermediate activations
        activation_store = {}
        def get_hook(key):
            def hook(module, input, output):
                activation_store[key] = output.detach()
            return hook
            
        expert_hook_handles = []
        for m_idx, exp_model in enumerate(dev_experts):
            exp_bn = get_bn_modules(exp_model)[i][1]
            h = exp_bn.register_forward_hook(get_hook(f"expert_{m_idx}"))
            expert_hook_handles.append(h)
            
        # Run forward pass on experts
        for m_idx, name_exp in enumerate(expert_names):
            exp_loader = cal_loaders[name_exp]
            bx, _ = next(iter(exp_loader))
            bx = bx.to(device)
            dev_experts[m_idx].fc = expert_models[m_idx].fc.to(device)
            with torch.no_grad():
                _ = dev_experts[m_idx](bx)
                
        for h in expert_hook_handles:
            h.remove()
            
        # Compute expert target statistics with SLF
        for m_idx in range(M):
            act_exp = activation_store[f"expert_{m_idx}"]
            B = act_exp.size(0)
            
            if trim_fraction > 0.0:
                # Score each sample by its mean squared activation
                sample_scores = torch.mean(act_exp ** 2, dim=(1, 2, 3))
                _, sorted_indices = torch.sort(sample_scores)
                
                k = int(B * trim_fraction)
                k = max(0, min(k, B - 1))
                
                clean_indices = sorted_indices[: B - k]
                clean_act_exp = act_exp[clean_indices]
                
                mean = torch.mean(clean_act_exp, dim=(0, 2, 3))
                var = torch.var(clean_act_exp, dim=(0, 2, 3), unbiased=False)
                std = torch.sqrt(var + epsilon)
            else:
                mean = torch.mean(act_exp, dim=(0, 2, 3))
                var = torch.var(act_exp, dim=(0, 2, 3), unbiased=False)
                std = torch.sqrt(var + epsilon)
                
            expert_means.append(mean)
            expert_stds.append(std)
            
        target_mean = torch.stack(expert_means).mean(dim=0)
        target_std = torch.stack(expert_stds).mean(dim=0)
        
        # Collect statistics from current state of fused/calibrated merged model
        fused_hook_handle = bn_layer_fused.register_forward_hook(get_hook("merged"))
        
        bx_joint, _ = next(iter(joint_loader))
        bx_joint = bx_joint.to(device)
        with torch.no_grad():
            _ = fused_model(bx_joint)
            
        fused_hook_handle.remove()
        act_merged = activation_store["merged"]
        B = act_merged.size(0)
        
        if trim_fraction > 0.0:
            sample_scores_merged = torch.mean(act_merged ** 2, dim=(1, 2, 3))
            _, sorted_indices_merged = torch.sort(sample_scores_merged)
            
            k = int(B * trim_fraction)
            k = max(0, min(k, B - 1))
            
            clean_indices_merged = sorted_indices_merged[: B - k]
            clean_act_merged = act_merged[clean_indices_merged]
            
            merged_mean = torch.mean(clean_act_merged, dim=(0, 2, 3))
            merged_var = torch.var(clean_act_merged, dim=(0, 2, 3), unbiased=False)
            merged_std = torch.sqrt(merged_var + epsilon)
        else:
            merged_mean = torch.mean(act_merged, dim=(0, 2, 3))
            merged_var = torch.var(act_merged, dim=(0, 2, 3), unbiased=False)
            merged_std = torch.sqrt(merged_var + epsilon)
            
        scale = target_std / merged_std
        shift = target_mean - scale * merged_mean
        
        with torch.no_grad():
            bn_layer_fused.weight.copy_(scale * bn_layer_fused.weight)
            bn_layer_fused.bias.copy_(scale * bn_layer_fused.bias + shift)
            
        hook = CalibrationHook(bn_layer_hooked, scale=scale, shift=shift)
        online_hooks.append(hook)
        
    return fused_model, hooked_model, online_hooks

def main():
    print("=== STARTING PRAGMATIST ABLATION STUDY FOR SLF TRIM FRACTION ===")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets = get_datasets()
    task_names = ['mnist', 'fmnist', 'cifar10']
    
    # Load cached expert models
    expert_models = []
    expert_accs = {}
    
    from torchvision.models import ResNet18_Weights
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    
    for task in task_names:
        ckpt_path = f"./checkpoints/expert_{task}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint not found at {ckpt_path}. Run experiment.py first.")
        
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        test_ds = datasets[task][1]
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        acc = evaluate_model(model, test_loader, device)
        expert_models.append(model)
        expert_accs[task] = acc
        print(f"Expert {task.upper()} Acc: {acc:.2f}%")
        
    train_datasets_dict = {task: datasets[task][0] for task in task_names}
    merged_base = merge_models(base_model, expert_models, mode="WA")
    merged_base = merged_base.to(device)
    
    # Ablation settings
    trim_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    corruption_levels = [0.2, 0.4]
    N = 128
    
    results = {
        "trim_fractions": trim_fractions,
        "results_p_0.2": [],
        "results_p_0.4": [],
        "parity_verified": []
    }
    
    for p in corruption_levels:
        p_key = f"results_p_{p}"
        print(f"\n--- Sweeping Trim Fraction alpha under Corruption p = {p} ---")
        
        for alpha in trim_fractions:
            # Construct corrupted calibration sets
            clean_calib_sets = get_calibration_sets(train_datasets_dict, N=N)
            corrupted_calib_sets = {}
            for task, subset in clean_calib_sets.items():
                corrupted_calib_sets[task] = CorruptedDataset(subset, corruption_prob=p, seed=42)
                
            # Run calibration with the specific alpha
            fused_model, hooked_model, hooks = run_ablation_calibration_and_fusion(
                base_model, expert_models, merged_base, corrupted_calib_sets, N=N, mode="TAAC", trim_fraction=alpha, device=device
            )
            
            # Evaluate clean test sets
            test_accs = {}
            for idx, task in enumerate(task_names):
                _, test_ds = datasets[task]
                test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                fused_model.fc = expert_models[idx].fc.to(device)
                acc = evaluate_model(fused_model, test_loader, device)
                test_accs[task] = acc
            avg_acc = np.mean(list(test_accs.values()))
            
            # Verify mathematical parity with online hooked model
            hooked_accs = {}
            for idx, task in enumerate(task_names):
                _, test_ds = datasets[task]
                test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                hooked_model.fc = expert_models[idx].fc.to(device)
                acc = evaluate_model(hooked_model, test_loader, device)
                hooked_accs[task] = acc
            hooked_avg_acc = np.mean(list(hooked_accs.values()))
            
            parity = abs(avg_acc - hooked_avg_acc) < 1e-4
            
            for h in hooks:
                h.remove()
                
            print(f"Alpha: {alpha:.2f} | Fused Acc: {avg_acc:.2f}% | Hooked Acc: {hooked_avg_acc:.2f}% | Parity: {parity}")
            results[p_key].append(avg_acc)
            results["parity_verified"].append(parity)
            
    # Save to JSON
    with open("ablation_trim_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(trim_fractions, results["results_p_0.2"], '-o', label="p = 0.2 (20% Outliers)", color="#1b9e77", linewidth=2)
    plt.plot(trim_fractions, results["results_p_0.4"], '-s', label="p = 0.4 (40% Outliers)", color="#d95f02", linewidth=2)
    
    plt.title("SLF Ablation: Multi-Task Accuracy vs. Trimming Fraction", fontsize=11, fontweight='bold')
    plt.xlabel("Trimming Fraction ($\alpha$)", fontsize=10)
    plt.ylabel("Multi-Task Test Accuracy (%)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(trim_fractions)
    plt.ylim(50, 62)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("slf_ablation_trim.png", dpi=300)
    plt.close()
    print("Ablation results plotted and saved to slf_ablation_trim.png.")

if __name__ == "__main__":
    main()
