import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Import necessary modules and functions from experiment.py
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

class CorruptedDataset(Dataset):
    """
    Wraps a dataset and corrupts a fraction of the images by replacing them
    with standard Gaussian noise (complete out-of-distribution outliers).
    """
    def __init__(self, dataset, corruption_prob=0.2, seed=42):
        self.dataset = dataset
        self.corruption_prob = corruption_prob
        self.seed = seed
        
        # Deterministic corruption mask
        np.random.seed(seed)
        n = len(dataset)
        self.corrupt_mask = np.random.rand(n) < corruption_prob
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.corrupt_mask[idx]:
            # Replace image with pure standard Gaussian noise (OOD outlier)
            img = torch.randn_like(img)
        return img, label

def run_robust_calibration_and_fusion(base_model, expert_models, merged_model, calib_sets, N=128, mode="TAAC", use_robust=False, device="cuda"):
    print(f"Running {mode} (Robust={use_robust}) Calibration & In-place Weight Fusion...")
    
    expert_names = list(calib_sets.keys())
    M = len(expert_models)
    
    # We make a copy of the merged model to perform ZIO-CF weight-fusion
    fused_model = copy.deepcopy(merged_model).to(device)
    fused_model.eval()
    
    # We also keep a list of online hooks for the baseline hooked model to evaluate mathematical equivalence
    hooked_model = copy.deepcopy(merged_model).to(device)
    hooked_model.eval()
    online_hooks = []
    
    # Get BatchNorm modules in sequential order
    bn_fused = get_bn_modules(fused_model)
    bn_hooked = get_bn_modules(hooked_model)
    
    # We also need experts on device for statistic collection
    dev_experts = [copy.deepcopy(m).to(device).eval() for m in expert_models]
    
    # Setup loaders for calibration
    cal_loaders = {name: DataLoader(subset, batch_size=N, shuffle=False) for name, subset in calib_sets.items()}
    
    # For Joint Task-Agnostic Calibration, we need a pooled joint dataset
    joint_dataset = torch.utils.data.ConcatDataset([calib_sets[name] for name in expert_names])
    joint_loader = DataLoader(joint_dataset, batch_size=N * M, shuffle=False)
    
    epsilon = 1e-5
    trim_fraction = 0.25 if use_robust else 0.0
    
    # Sequential calibration loop
    for i in range(len(bn_fused)):
        name, bn_layer_fused = bn_fused[i]
        _, bn_layer_hooked = bn_hooked[i]
        
        # 1. Collect statistics from experts on this BatchNorm layer output
        expert_stds = []
        expert_means = []
        
        # To capture intermediate activations
        activation_store = {}
        def get_hook(key):
            def hook(module, input, output):
                activation_store[key] = output.detach()
            return hook
            
        # Register temporary hooks on experts
        expert_hook_handles = []
        for m_idx, exp_model in enumerate(dev_experts):
            exp_bn = get_bn_modules(exp_model)[i][1]
            h = exp_bn.register_forward_hook(get_hook(f"expert_{m_idx}"))
            expert_hook_handles.append(h)
            
        # Run forward pass on experts with their corresponding calibration sets
        for m_idx, name_exp in enumerate(expert_names):
            exp_loader = cal_loaders[name_exp]
            bx, _ = next(iter(exp_loader))
            bx = bx.to(device)
            dev_experts[m_idx].fc = expert_models[m_idx].fc.to(device)
            with torch.no_grad():
                _ = dev_experts[m_idx](bx)
                
        # Remove expert hooks
        for h in expert_hook_handles:
            h.remove()
            
        # Compute expert target statistics
        for m_idx in range(M):
            act_exp = activation_store[f"expert_{m_idx}"]  # shape [N, C, H, W]
            B = act_exp.size(0)
            
            if use_robust:
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
        
        # 2. Collect statistics from current state of fused/calibrated merged model
        fused_hook_handle = bn_layer_fused.register_forward_hook(get_hook("merged"))
        
        bx_joint, _ = next(iter(joint_loader))
        bx_joint = bx_joint.to(device)
        with torch.no_grad():
            _ = fused_model(bx_joint)
            
        fused_hook_handle.remove()
        act_merged = activation_store["merged"]
        B = act_merged.size(0)
        
        # Compute merged statistics
        if use_robust:
            # Score each sample by its mean squared activation
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
            
        # 3. Compute calibration factors
        scale = target_std / merged_std
        shift = target_mean - scale * merged_mean
        
        # --- ZIO-CF Weight Fusion ---
        with torch.no_grad():
            bn_layer_fused.weight.copy_(scale * bn_layer_fused.weight)
            bn_layer_fused.bias.copy_(scale * bn_layer_fused.bias + shift)
            
        # --- Online Hook Setup ---
        hook = CalibrationHook(bn_layer_hooked, scale=scale, shift=shift)
        online_hooks.append(hook)
        
    return fused_model, hooked_model, online_hooks

def main():
    print("=== STARTING PRAGMATIST ROBUSTNESS EXPERIMENTS ===")
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
            raise FileNotFoundError(f"Expert checkpoint not found at {ckpt_path}. Please run experiment.py first.")
        
        print(f"Loading cached expert for {task.upper()}...")
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
        
    oracle_avg = np.mean(list(expert_accs.values()))
    print(f"Oracle (Single Experts) Average: {oracle_avg:.2f}%")
    
    train_datasets_dict = {task: datasets[task][0] for task in task_names}
    
    # Merging Base (WA is used as the standard benchmark)
    merged_base = merge_models(base_model, expert_models, mode="WA")
    merged_base = merged_base.to(device)
    
    # Evaluate clean uncalibrated baseline
    uncal_accs = {}
    for idx, task in enumerate(task_names):
        _, test_ds = datasets[task]
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        merged_base.fc = expert_models[idx].fc.to(device)
        acc = evaluate_model(merged_base, test_loader, device)
        uncal_accs[task] = acc
    uncal_avg = np.mean(list(uncal_accs.values()))
    print(f"Uncalibrated Merged Average Acc: {uncal_avg:.2f}%")
    
    # Corruption settings
    corruption_probs = [0.0, 0.1, 0.2, 0.3, 0.4]
    N = 128
    
    results = {
        "corruption_probs": corruption_probs,
        "uncalibrated": [uncal_avg] * len(corruption_probs),
        "standard_taac": [],
        "robust_taac": [],
        "parity_verified": []
    }
    
    for p in corruption_probs:
        print(f"\n--- Evaluating Corruption Probability p = {p} ---")
        
        # Get clean calibration set
        clean_calib_sets = get_calibration_sets(train_datasets_dict, N=N)
        
        # Apply corruption to calibration sets
        corrupted_calib_sets = {}
        for task, subset in clean_calib_sets.items():
            # Wrap standard subset in CorruptedDataset
            corrupted_calib_sets[task] = CorruptedDataset(subset, corruption_prob=p, seed=42)
            
        # 1. Standard TAAC Calibration (L2 Mean & Std)
        fused_std, hooked_std, hooks_std = run_robust_calibration_and_fusion(
            base_model, expert_models, merged_base, corrupted_calib_sets, N=N, mode="TAAC", use_robust=False, device=device
        )
        
        # Evaluate clean test set with standard calibration
        std_accs = {}
        for idx, task in enumerate(task_names):
            _, test_ds = datasets[task]
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            fused_std.fc = expert_models[idx].fc.to(device)
            acc = evaluate_model(fused_std, test_loader, device)
            std_accs[task] = acc
        std_avg = np.mean(list(std_accs.values()))
        
        # Verify parity
        hooked_std_accs = {}
        for idx, task in enumerate(task_names):
            _, test_ds = datasets[task]
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            hooked_std.fc = expert_models[idx].fc.to(device)
            acc = evaluate_model(hooked_std, test_loader, device)
            hooked_std_accs[task] = acc
        hooked_std_avg = np.mean(list(hooked_std_accs.values()))
        std_parity = abs(std_avg - hooked_std_avg) < 1e-4
        
        for h in hooks_std:
            h.remove()
            
        print(f"Standard TAAC - Fused Clean Acc: {std_avg:.2f}% | Hooked Clean Acc: {hooked_std_avg:.2f}% | Parity: {std_parity}")
        
        # 2. Robust TAAC Calibration (Trimmed-Mean & Trimmed-Std)
        fused_robust, hooked_robust, hooks_robust = run_robust_calibration_and_fusion(
            base_model, expert_models, merged_base, corrupted_calib_sets, N=N, mode="TAAC", use_robust=True, device=device
        )
        
        # Evaluate clean test set with robust calibration
        robust_accs = {}
        for idx, task in enumerate(task_names):
            _, test_ds = datasets[task]
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            fused_robust.fc = expert_models[idx].fc.to(device)
            acc = evaluate_model(fused_robust, test_loader, device)
            robust_accs[task] = acc
        robust_avg = np.mean(list(robust_accs.values()))
        
        # Verify parity
        hooked_robust_accs = {}
        for idx, task in enumerate(task_names):
            _, test_ds = datasets[task]
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            hooked_robust.fc = expert_models[idx].fc.to(device)
            acc = evaluate_model(hooked_robust, test_loader, device)
            hooked_robust_accs[task] = acc
        hooked_robust_avg = np.mean(list(hooked_robust_accs.values()))
        robust_parity = abs(robust_avg - hooked_robust_avg) < 1e-4
        
        for h in hooks_robust:
            h.remove()
            
        print(f"Robust TAAC - Fused Clean Acc: {robust_avg:.2f}% | Hooked Clean Acc: {hooked_robust_avg:.2f}% | Parity: {robust_parity}")
        
        results["standard_taac"].append(std_avg)
        results["robust_taac"].append(robust_avg)
        results["parity_verified"].append(bool(std_parity and robust_parity))
        
    print("\n=== Robustness Experiment Summary ===")
    for idx, p in enumerate(corruption_probs):
        print(f"p = {p:.1f} | Standard TAAC: {results['standard_taac'][idx]:.2f}% | Robust TAAC: {results['robust_taac'][idx]:.2f}% | Parity: {results['parity_verified'][idx]}")
        
    # Save results to robust_results.json
    with open("robust_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plot
    plt.figure(figsize=(7, 5))
    plt.plot(corruption_probs, results["uncalibrated"], '--', label="Uncalibrated WA (Clean)", color="#d95f02", linewidth=2)
    plt.plot(corruption_probs, results["standard_taac"], '-o', label="Standard TAAC (Mean/Std)", color="#7570b3", linewidth=2)
    plt.plot(corruption_probs, results["robust_taac"], '-s', label="Robust TAAC (SLF Trim-25%, Ours)", color="#1b9e77", linewidth=2)
    
    plt.title("Calibration Robustness under Outlier Corruption", fontsize=12, fontweight='bold')
    plt.xlabel("Calibration Data Outlier Fraction ($p$)", fontsize=10)
    plt.ylabel("Multi-Task Test Accuracy (%)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(25, 75)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("robustness_vs_corruption.png", dpi=300)
    plt.close()
    print("Robustness plot saved to robustness_vs_corruption.png.")

if __name__ == "__main__":
    main()
