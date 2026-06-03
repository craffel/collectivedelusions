import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import methods from methods.py
from methods import (
    set_seed,
    get_datasets,
    load_expert,
    load_base_model,
    merge_experts_wa,
    merge_experts_ta,
    ActivationHookManager,
    get_calibration_sets,
    run_evaluation,
    capture_activations
)

def sequential_calibration(merged_model, experts, train_datasets, cal_size, method, device, c_reg=0.3, gamma_reg=0.1, cal_seed=42):
    # Get all BN layers
    bn_layers = []
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
            
    early_layers = []
    deep_layers = []
    for name, module in bn_layers:
        if 'layer3' in name or 'layer4' in name:
            deep_layers.append((name, module))
        else:
            early_layers.append((name, module))
            
    cal_subsets = get_calibration_sets(train_datasets, size=cal_size, seed=cal_seed)
    joint_cal_dataset = torch.utils.data.ConcatDataset(cal_subsets)
    joint_cal_loader = DataLoader(joint_cal_dataset, batch_size=32, shuffle=False)
    
    # We sequentially calibrate each early layer using SP-TAAC
    print(f"--- Calibrating Early Layers via SP-TAAC (N={cal_size}) ---")
    for name, module in early_layers:
        # Capture target expert activations: only corresponding task subsets are passed to each expert
        target_acts = []
        for i, exp in enumerate(experts):
            subset_dataset = cal_subsets[i]
            subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
            target_acts.append(capture_activations(exp, subset_loader, name, device))
        # Concatenate along batch dimension: shape [3 * cal_size, C, H, W]
        target_acts_cat = torch.cat(target_acts, dim=0)
        
        # Capture merged activations on joint calibration set
        merged_acts = capture_activations(merged_model, joint_cal_loader, name, device)
        
        # Compute standard deviations channel-wise (dim 0, 2, 3)
        target_std = torch.std(target_acts_cat, dim=(0, 2, 3), keepdim=False)
        merged_std = torch.std(merged_acts, dim=(0, 2, 3), keepdim=False)
        
        # Scaling factor gamma = target_std / (merged_std + epsilon)
        eps = 1e-5
        gamma = target_std / (merged_std + eps)
        # Clamp scaling factor for stability
        gamma = torch.clamp(gamma, 0.1, 10.0)
        
        # Apply scaling to the merged model's BN weights and biases in-place!
        with torch.no_grad():
            module.weight.copy_(module.weight * gamma.to(device))
            if module.bias is not None:
                module.bias.copy_(module.bias * gamma.to(device))
                
    # Now, for deep layers, we apply the chosen method
    if method == "none":
        print("No deep layer calibration applied.")
        return None
        
    print(f"--- Calibrating Deep Layers via {method.upper()} (N={cal_size}) ---")
    deep_layer_names = [name for name, _ in deep_layers]
    eval_manager = ActivationHookManager(merged_model, deep_layer_names, mode=method)
    
    # We sequentially compute the calibration parameters for each deep layer
    for name, module in deep_layers:
        # Capture target expert activations: only corresponding task subsets are passed to each expert
        target_acts = []
        for i, exp in enumerate(experts):
            subset_dataset = cal_subsets[i]
            subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
            target_acts.append(capture_activations(exp, subset_loader, name, device))
        target_acts_cat = torch.cat(target_acts, dim=0) # [3 * cal_size, C, H, W]
        
        # Capture merged activations on joint calibration set
        merged_acts = capture_activations(merged_model, joint_cal_loader, name, device) # [3 * cal_size, C, H, W]
        
        if method == "sptaac":
            # SP-TAAC for deep layers too
            target_std = torch.std(target_acts_cat, dim=(0, 2, 3), keepdim=False)
            merged_std = torch.std(merged_acts, dim=(0, 2, 3), keepdim=False)
            eps = 1e-5
            scale = torch.clamp(target_std / (merged_std + eps), 0.1, 10.0)
            eval_manager.calibration_params[name] = scale.to(device)
            
        elif method == "repair":
            # REPAIR spatial calibration (mean and variance alignment)
            target_mean = torch.mean(target_acts_cat, dim=(0, 2, 3), keepdim=False)
            target_std = torch.std(target_acts_cat, dim=(0, 2, 3), keepdim=False)
            merged_mean = torch.mean(merged_acts, dim=(0, 2, 3), keepdim=False)
            merged_std = torch.std(merged_acts, dim=(0, 2, 3), keepdim=False)
            eps = 1e-5
            scale = torch.clamp(target_std / (merged_std + eps), 0.1, 10.0)
            eval_manager.calibration_params[name] = (merged_mean.to(device), scale.to(device), target_mean.to(device))
            
        elif method == "wrsa":
            # WRSA Calibration: Wiener-Regularized Spectral Alignment (channel-agnostic spectral scale map)
            expert_fft = torch.fft.fft2(target_acts_cat, dim=(-2, -1))
            expert_mag = torch.abs(expert_fft) # [3 * cal_size, C, H, W]
            # Average over batch and channels to get channel-agnostic magnitude profile M_T: shape [H, W]
            M_T = expert_mag.mean(dim=(0, 1))
            
            merged_fft = torch.fft.fft2(merged_acts, dim=(-2, -1))
            merged_mag = torch.abs(merged_fft) # [3 * cal_size, C, H, W]
            M_M = merged_mag.mean(dim=(0, 1)) # [H, W]
            
            # Compute WRSA scaling map:
            # gamma(u, v) = (M_T * M_M) / (M_M^2 + c_reg^2 * M_T^2)
            denom = M_M**2 + (c_reg**2) * (M_T**2)
            gamma = (M_T * M_M) / (denom + 1e-8)
            # Bound the scaling factor by 1 / (2 * c_reg) as proved by Theorem 3.2
            max_bound = 1.0 / (2.0 * c_reg)
            gamma = torch.clamp(gamma, 0.0, max_bound)
            
            eval_manager.calibration_params[name] = gamma.to(device)
            
        elif method == "nra":
            # Proposed Neural Resonance Alignment: channel-specific complex scaling map
            expert_fft = torch.fft.fft2(target_acts_cat, dim=(-2, -1))
            merged_fft = torch.fft.fft2(merged_acts, dim=(-2, -1))
            
            # Compute numerator: E[ X_exp * conj(X_m) ] over the batch dimension (dim 0)
            num = (expert_fft * torch.conj(merged_fft)).mean(dim=0) # [C, H, W]
            
            # Compute denominator: E[ |X_m|^2 ] + gamma_reg^2 * E[ |X_exp|^2 ] over batch dimension (dim 0)
            denom_m = (torch.abs(merged_fft)**2).mean(dim=0) # [C, H, W]
            denom_exp = (torch.abs(expert_fft)**2).mean(dim=0) # [C, H, W]
            denom = denom_m + (gamma_reg**2) * denom_exp
            
            # Complex Resonance Scaling Factor g
            g = num / (denom + 1e-8)
            
            # For numerical safety and stability, clamp the magnitude of g
            mag_g = torch.abs(g)
            max_bound = 1.0 / (2.0 * gamma_reg)
            scale_factor = torch.clamp(mag_g, max=max_bound) / (mag_g + 1e-8)
            g = g * scale_factor
            
            eval_manager.calibration_params[name] = g.to(device)
            
        elif method == "tcnra":
            # Task-Conditional Neural Resonance Alignment (TC-NRA)
            # We compute a separate complex scaling factor g_k for each task k
            g_list = []
            for k in range(len(experts)):
                exp_act = target_acts[k] # [cal_size, C, H, W] expert activations on S_k
                subset_dataset = cal_subsets[k]
                subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
                m_act = capture_activations(merged_model, subset_loader, name, device) # [cal_size, C, H, W] merged model activations on S_k
                
                # Compute FFT
                exp_fft = torch.fft.fft2(exp_act, dim=(-2, -1))
                m_fft = torch.fft.fft2(m_act, dim=(-2, -1))
                
                # Compute numerator: E[ X_exp * conj(X_m) ] over the batch dimension (dim 0)
                num_k = (exp_fft * torch.conj(m_fft)).mean(dim=0) # [C, H, W]
                
                # Compute denominator: E[ |X_m|^2 ] + gamma_reg^2 * E[ |X_exp|^2 ] over batch dimension (dim 0)
                denom_m_k = (torch.abs(m_fft)**2).mean(dim=0) # [C, H, W]
                denom_exp_k = (torch.abs(exp_fft)**2).mean(dim=0) # [C, H, W]
                denom_k = denom_m_k + (gamma_reg**2) * denom_exp_k
                
                # Complex Resonance Scaling Factor g_k
                g_k = num_k / (denom_k + 1e-8)
                
                # For numerical safety and stability, clamp the magnitude of g_k
                mag_g_k = torch.abs(g_k)
                max_bound = 1.0 / (2.0 * gamma_reg)
                scale_factor = torch.clamp(mag_g_k, max=max_bound) / (mag_g_k + 1e-8)
                g_k = g_k * scale_factor
                
                g_list.append(g_k.to(device))
            eval_manager.calibration_params[name] = g_list
            
    return eval_manager

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on device: {device}")
    
    # Load Datasets
    (mnist_tr, mnist_te), (fmnist_tr, fmnist_te), (cifar_tr, cifar_te) = get_datasets()
    
    # Load Experts
    experts = [
        load_expert('MNIST', device),
        load_expert('FashionMNIST', device),
        load_expert('CIFAR10', device)
    ]
    
    # Base model (pretrained ImageNet) for Task Arithmetic
    base_model = load_base_model(device)
    
    # Test loaders
    test_loaders = {
        'MNIST': DataLoader(mnist_te, batch_size=128, shuffle=False, num_workers=2),
        'FashionMNIST': DataLoader(fmnist_te, batch_size=128, shuffle=False, num_workers=2),
        'CIFAR10': DataLoader(cifar_te, batch_size=128, shuffle=False, num_workers=2)
    }
    
    train_datasets = [mnist_tr, fmnist_tr, cifar_tr]
    
    # Verify individual experts
    print("\n--- Verification of Experts ---")
    for name, loader in test_loaders.items():
        # Load corresponding expert
        exp = load_expert(name, device)
        acc = run_evaluation(exp, loader, device)
        print(f"Expert {name} Test Accuracy: {acc:.2f}%")
    results = []
    
    # Loop over calibration sizes N
    cal_sizes = [16, 64, 128]
    methods = ["none", "repair", "sptaac", "wrsa", "nra", "tcnra"]
    merge_modes = ["WA", "TA"]
    seeds = [42, 43, 44]
    
    for cal_size in cal_sizes:
        for mode in merge_modes:
            for method in methods:
                seed_results = []
                for seed in seeds:
                    set_seed(42) # Ensure deterministic merging and model base state
                    
                    # 1. Merge models
                    if mode == "WA":
                        merged_model = merge_experts_wa(experts)
                    else:
                        merged_model = merge_experts_ta(experts, base_model, lam=0.3)
                    merged_model.to(device)
                    
                    # 2. Apply sequential calibration
                    eval_manager = sequential_calibration(
                        merged_model, experts, train_datasets, cal_size, 
                        method, device, c_reg=0.3, gamma_reg=0.2, cal_seed=seed
                    )
                    
                    # 3. Evaluate on the three test sets
                    accs = {}
                    for task_idx, (name, loader) in enumerate(test_loaders.items()):
                        if eval_manager is not None and method == "tcnra":
                            eval_manager.current_task = task_idx
                        accs[name] = run_evaluation(merged_model, loader, device)
                    
                    # Clean up hooks
                    if eval_manager is not None:
                        eval_manager.remove_hooks()
                        
                    avg_acc = np.mean(list(accs.values()))
                    seed_results.append({
                        "MNIST": accs["MNIST"],
                        "FashionMNIST": accs["FashionMNIST"],
                        "CIFAR10": accs["CIFAR10"],
                        "Average": avg_acc
                    })
                
                # Compute mean and standard deviation across seeds
                mnist_vals = [s["MNIST"] for s in seed_results]
                fmnist_vals = [s["FashionMNIST"] for s in seed_results]
                cifar_vals = [s["CIFAR10"] for s in seed_results]
                avg_vals = [s["Average"] for s in seed_results]
                
                method_name_map = {
                    "none": "Uncalibrated",
                    "repair": "REPAIR",
                    "sptaac": "SPTAAC",
                    "wrsa": "WRSA",
                    "nra": "NRA",
                    "tcnra": "TC-NRA"
                }
                res_dict = {
                    "N": cal_size,
                    "Mode": mode,
                    "Method": method_name_map[method],
                    "MNIST_mean": np.mean(mnist_vals),
                    "MNIST_std": np.std(mnist_vals),
                    "FashionMNIST_mean": np.mean(fmnist_vals),
                    "FashionMNIST_std": np.std(fmnist_vals),
                    "CIFAR10_mean": np.mean(cifar_vals),
                    "CIFAR10_std": np.std(cifar_vals),
                    "Average_mean": np.mean(avg_vals),
                    "Average_std": np.std(avg_vals)
                }
                results.append(res_dict)
                print(f"[{mode} | {res_dict['Method']} | N={cal_size}] MNIST: {res_dict['MNIST_mean']:.2f}±{res_dict['MNIST_std']:.2f}% | F-MNIST: {res_dict['FashionMNIST_mean']:.2f}±{res_dict['FashionMNIST_std']:.2f}% | CIFAR10: {res_dict['CIFAR10_mean']:.2f}±{res_dict['CIFAR10_std']:.2f}% | Average: {res_dict['Average_mean']:.2f}±{res_dict['Average_std']:.2f}%")
                
    # Print formatted markdown table of results
    print("\n\n--- SUMMARY OF RESULTS ---")
    print("| Merge Mode | Calibration Method | N | MNIST | F-MNIST | CIFAR-10 | Average |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(f"| {r['Mode']} | {r['Method']} | {r['N']} | {r['MNIST_mean']:.2f}±{r['MNIST_std']:.2f}% | {r['FashionMNIST_mean']:.2f}±{r['FashionMNIST_std']:.2f}% | {r['CIFAR10_mean']:.2f}±{r['CIFAR10_std']:.2f}% | {r['Average_mean']:.2f}±{r['Average_std']:.2f}% |")
        
    # Generate Plots
    print("\nGenerating comparative plots...")
    
    # Set style for professional look
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    plot_methods = ["Uncalibrated", "REPAIR", "SPTAAC", "WRSA", "NRA", "TC-NRA"]
    colors = {
        "Uncalibrated": "#7f7f7f",  # Gray
        "REPAIR": "#9467bd",        # Purple
        "SPTAAC": "#1f77b4",        # Blue
        "WRSA": "#ff7f0e",          # Orange
        "NRA": "#d62728",           # Red
        "TC-NRA": "#2ca02c"         # Green (Highlight)
    }
    markers = {
        "Uncalibrated": "x",
        "REPAIR": "d",
        "SPTAAC": "s",
        "WRSA": "^",
        "NRA": "o",
        "TC-NRA": "*"
    }
    
    for mode in ["WA", "TA"]:
        plt.figure(figsize=(6.5, 4.8))
        for method in plot_methods:
            y_vals = []
            y_errs = []
            for size in cal_sizes:
                match = [r for r in results if r["Mode"] == mode and r["Method"] == method and r["N"] == size]
                if match:
                    y_vals.append(match[0]["Average_mean"])
                    y_errs.append(match[0]["Average_std"])
            
            if y_vals:
                label_name = f"{method} (Ours)" if method in ["NRA", "TC-NRA"] else method
                plt.errorbar(cal_sizes, y_vals, yerr=y_errs, marker=markers[method], color=colors[method], linewidth=2.5, markersize=9, label=label_name, capsize=5)
        
        title_str = "Weight Averaging (WA) Calibration" if mode == "WA" else "Task Arithmetic (TA) Calibration"
        plt.xlabel("Calibration Dataset Size ($N$ per task)", fontsize=11)
        plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=11)
        plt.title(title_str, fontsize=12, fontweight='bold')
        plt.xticks(cal_sizes)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")
        plt.tight_layout()
        filename = f"calibration_size_comparison_{mode.lower()}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
