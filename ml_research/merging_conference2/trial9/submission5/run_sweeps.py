import torch
import torch.nn as nn
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from models import ResNet18CIFAR, MLPCIFAR
from merge_and_evaluate import (
    get_model,
    evaluate_multi_task,
    merge_weight_averaging,
    merge_task_arithmetic,
    merge_qr_wcpr,
    merge_wcpr
)

def run_de_bn_sweep(device):
    print("\n" + "="*40)
    print("RUNNING DE-BN SAMPLE COUNT SWEEP")
    print("="*40)
    
    # Load ResNet-18 models
    progenitor = get_model("resnet18")
    progenitor.load_state_dict(torch.load("checkpoints/progenitor_resnet18.pt", map_location="cpu"))
    progenitor = progenitor.to(device)
    
    experts = []
    datasets = ["mnist", "fmnist", "cifar10"]
    for ds in datasets:
        model = get_model("resnet18")
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_resnet18.pt", map_location="cpu"))
        experts.append(model.to(device))
        
    sample_counts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    methods = {
        "WA": lambda: merge_weight_averaging(progenitor, experts),
        "TA (l=0.3)": lambda: merge_task_arithmetic(progenitor, experts, lambd=0.3),
        "QR-WCPR": lambda: merge_qr_wcpr(progenitor, experts, lambd=0.5, gamma=1.5)
    }
    
    print("Pre-merging models for ResNet-18 sweep...")
    merged_models = {}
    for m_name, merge_fn in methods.items():
        print(f"  Merging {m_name}...")
        merged_models[m_name] = merge_fn().to(device)
    print("Pre-merging complete!")
    
    seeds = [42, 43, 44]
    results = {m: {"clean": {"mean": [], "std": [], "raw": []}, "int4": {"mean": [], "std": [], "raw": []}} for m in methods}
    
    for count in sample_counts:
        print(f"\nEvaluating DE-BN sample count: {count}")
        for m_name in methods.keys():
            merged_model = merged_models[m_name]
            
            clean_accs = []
            int4_accs = []
            
            # If count is 0, we don't need to run multiple seeds
            run_seeds = [42] if count == 0 else seeds
            
            for seed in run_seeds:
                # 1. Evaluate FP32 Clean
                res_clean = evaluate_multi_task(
                    merged_model, 
                    datasets, 
                    device, 
                    num_bits=0, 
                    bn_calib_samples=count, 
                    seed=seed,
                    batch_size=1024,
                    num_workers=4
                )
                clean_accs.append(res_clean["average_acc"])
                
                # 2. Evaluate INT4
                res_int4 = evaluate_multi_task(
                    merged_model, 
                    datasets, 
                    device, 
                    num_bits=4, 
                    bn_calib_samples=count, 
                    seed=seed,
                    batch_size=1024,
                    num_workers=4
                )
                int4_accs.append(res_int4["average_acc"])
                
            # Compute mean and std
            mean_clean = float(np.mean(clean_accs))
            std_clean = float(np.std(clean_accs)) if len(clean_accs) > 1 else 0.0
            
            mean_int4 = float(np.mean(int4_accs))
            std_int4 = float(np.std(int4_accs)) if len(int4_accs) > 1 else 0.0
            
            results[m_name]["clean"]["mean"].append(mean_clean)
            results[m_name]["clean"]["std"].append(std_clean)
            results[m_name]["clean"]["raw"].append(clean_accs)
            
            results[m_name]["int4"]["mean"].append(mean_int4)
            results[m_name]["int4"]["std"].append(std_int4)
            results[m_name]["int4"]["raw"].append(int4_accs)
            
            print(f"  [{m_name}]: Clean={mean_clean*100:.2f}% (std={std_clean*100:.2f}%) | INT4={mean_int4*100:.2f}% (std={std_int4*100:.2f}%)")
            
    return sample_counts, results


def run_gamma_sweep(device):
    print("\n" + "="*40)
    print("RUNNING QR-WCPR GAMMA SWEEP")
    print("="*40)
    
    gamma_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    datasets = ["mnist", "fmnist", "cifar10"]
    
    # 1. ResNet-18 INT4 + DE-BN (16)
    r18_progenitor = get_model("resnet18")
    r18_progenitor.load_state_dict(torch.load("checkpoints/progenitor_resnet18.pt", map_location="cpu"))
    r18_progenitor = r18_progenitor.to(device)
    
    r18_experts = []
    for ds in datasets:
        model = get_model("resnet18")
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_resnet18.pt", map_location="cpu"))
        r18_experts.append(model.to(device))
        
    r18_accs = []
    print("\nResNet-18 (INT4 + DE-BN 16):")
    for gamma in gamma_vals:
        merged = merge_qr_wcpr(r18_progenitor, r18_experts, lambd=0.5, gamma=gamma).to(device)
        res = evaluate_multi_task(
            merged, 
            datasets, 
            device, 
            num_bits=4, 
            bn_calib_samples=16, 
            seed=42,
            batch_size=1024,
            num_workers=4
        )
        acc = res["average_acc"]
        r18_accs.append(acc)
        print(f"  Gamma={gamma:4.1f} | Avg Acc={acc*100:.2f}% | MNIST: {res['mnist']['acc']*100:.1f}% | FMNIST: {res['fmnist']['acc']*100:.1f}% | CIFAR10: {res['cifar10']['acc']*100:.1f}%")
        
    # Also evaluate standard WCPR for reference
    wcpr_merged = merge_wcpr(r18_progenitor, r18_experts, lambd=0.5).to(device)
    res_wcpr = evaluate_multi_task(wcpr_merged, datasets, device, num_bits=4, bn_calib_samples=16, seed=42, batch_size=1024, num_workers=4)
    wcpr_r18_acc = res_wcpr["average_acc"]
    print(f"  Standard WCPR baseline: {wcpr_r18_acc*100:.2f}%")
        
    # 2. MLP INT4
    mlp_progenitor = get_model("mlp")
    mlp_progenitor.load_state_dict(torch.load("checkpoints/progenitor_mlp.pt", map_location="cpu"))
    mlp_progenitor = mlp_progenitor.to(device)
    
    mlp_experts = []
    for ds in datasets:
        model = get_model("mlp")
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_mlp.pt", map_location="cpu"))
        mlp_experts.append(model.to(device))
        
    mlp_accs = []
    print("\nMLP (INT4):")
    for gamma in gamma_vals:
        merged = merge_qr_wcpr(mlp_progenitor, mlp_experts, lambd=0.5, gamma=gamma).to(device)
        res = evaluate_multi_task(
            merged, 
            datasets, 
            device, 
            num_bits=4, 
            bn_calib_samples=0, 
            seed=42,
            batch_size=1024,
            num_workers=4
        )
        acc = res["average_acc"]
        mlp_accs.append(acc)
        print(f"  Gamma={gamma:4.1f} | Avg Acc={acc*100:.2f}% | MNIST: {res['mnist']['acc']*100:.1f}% | FMNIST: {res['fmnist']['acc']*100:.1f}% | CIFAR10: {res['cifar10']['acc']*100:.1f}%")
        
    mlp_wcpr_merged = merge_wcpr(mlp_progenitor, mlp_experts, lambd=0.5).to(device)
    res_mlp_wcpr = evaluate_multi_task(mlp_wcpr_merged, datasets, device, num_bits=4, bn_calib_samples=0, seed=42, batch_size=1024, num_workers=4)
    wcpr_mlp_acc = res_mlp_wcpr["average_acc"]
    print(f"  Standard WCPR baseline: {wcpr_mlp_acc*100:.2f}%")
        
    return gamma_vals, r18_accs, wcpr_r18_acc, mlp_accs, wcpr_mlp_acc


def plot_results(sample_counts, de_bn_results, gamma_vals, r18_gamma_accs, wcpr_r18, mlp_gamma_accs, wcpr_mlp):
    # Setup plotting style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = False # Tectonic is not easily connected to matplotlib direct rendering
    
    # Plot 1: DE-BN Sweep (Clean vs INT4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    colors = {"WA": "#1f77b4", "TA (l=0.3)": "#ff7f0e", "QR-WCPR": "#2ca02c"}
    markers = {"WA": "o", "TA (l=0.3)": "s", "QR-WCPR": "^"}
    
    # 1.1 Clean
    for m, data in de_bn_results.items():
        mean = np.array(data["clean"]["mean"]) * 100
        std = np.array(data["clean"]["std"]) * 100
        ax1.plot(sample_counts, mean, label=m, color=colors[m], marker=markers[m], linewidth=1.8)
        # Fill only if there is a non-zero standard deviation
        if np.any(std > 0):
            ax1.fill_between(sample_counts, mean - std, mean + std, color=colors[m], alpha=0.15)
            
    ax1.set_xscale('symlog', linthresh=1)
    ax1.set_xticks(sample_counts)
    ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%g'))
    ax1.set_xlabel("DE-BN Calibration Samples", fontsize=11)
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax1.set_title("ResNet-18 Clean FP32", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)
    ax1.set_ylim(8, 50)
    
    # 1.2 INT4
    for m, data in de_bn_results.items():
        mean = np.array(data["int4"]["mean"]) * 100
        std = np.array(data["int4"]["std"]) * 100
        ax2.plot(sample_counts, mean, label=m, color=colors[m], marker=markers[m], linewidth=1.8)
        if np.any(std > 0):
            ax2.fill_between(sample_counts, mean - std, mean + std, color=colors[m], alpha=0.15)
            
    ax2.set_xscale('symlog', linthresh=1)
    ax2.set_xticks(sample_counts)
    ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%g'))
    ax2.set_xlabel("DE-BN Calibration Samples", fontsize=11)
    ax2.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax2.set_title("ResNet-18 Quantized INT4", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)
    ax2.set_ylim(8, 50)
    
    plt.tight_layout()
    plt.savefig("plot_de_bn_sweep.png", dpi=300)
    plt.close()
    print("Saved plot_de_bn_sweep.png!")
    
    # Plot 2: Gamma Sweep
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(gamma_vals, np.array(r18_gamma_accs)*100, label="QR-WCPR (ResNet-18 INT4)", color="#2ca02c", marker="o", linewidth=2)
    ax.axhline(wcpr_r18*100, label="WCPR Baseline (ResNet-18 INT4)", color="#2ca02c", linestyle="--", alpha=0.7)
    
    ax.plot(gamma_vals, np.array(mlp_gamma_accs)*100, label="QR-WCPR (MLP INT4)", color="#9467bd", marker="s", linewidth=2)
    ax.axhline(wcpr_mlp*100, label="WCPR Baseline (MLP INT4)", color="#9467bd", linestyle="--", alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_xticks(gamma_vals)
    ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%g'))
    ax.set_xlabel(r"QR-WCPR Robust Clamping Threshold $\gamma$", fontsize=11)
    ax.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax.set_title("Robust Clamping Threshold Sensitivity in INT4", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig("plot_gamma_sweep.png", dpi=300)
    plt.close()
    print("Saved plot_gamma_sweep.png!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    print(f"Sweeps running on device: {device}")
    
    # 1. Run DE-BN Sample Count Sweep
    sample_counts, de_bn_results = run_de_bn_sweep(device)
    
    # 2. Run Gamma Threshold Sweep
    gamma_vals, r18_gamma, wcpr_r18, mlp_gamma, wcpr_mlp = run_gamma_sweep(device)
    
    # 3. Save all results to sweep_results.json
    results_dict = {
        "de_bn_sweep": {
            "sample_counts": sample_counts,
            "results": de_bn_results
        },
        "gamma_sweep": {
            "gamma_vals": gamma_vals,
            "r18_gamma_accs": r18_gamma,
            "wcpr_r18_acc": wcpr_r18,
            "mlp_gamma_accs": mlp_gamma,
            "wcpr_mlp_acc": wcpr_mlp
        }
    }
    with open("sweep_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Saved sweep_results.json!")
    
    # 4. Plot results
    plot_results(sample_counts, de_bn_results, gamma_vals, r18_gamma, wcpr_r18, mlp_gamma, wcpr_mlp)

if __name__ == "__main__":
    main()
