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

def run_lambda_sweep(device):
    print("\n" + "="*40)
    print("RUNNING LAMBDA SWEEP (0.1 TO 1.0)")
    print("="*40)
    
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    datasets = ["mnist", "fmnist", "cifar10"]
    
    # 1. ResNet-18 Clean + DE-BN (16)
    r18_progenitor = get_model("resnet18").to(device)
    r18_progenitor.load_state_dict(torch.load("checkpoints/progenitor_resnet18.pt", map_location="cpu"))
    
    r18_experts = []
    for ds in datasets:
        model = get_model("resnet18").to(device)
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_resnet18.pt", map_location="cpu"))
        r18_experts.append(model)
        
    r18_results = {"TA": [], "WCPR": [], "QR-WCPR": []}
    
    print("\nResNet-18 Clean + DE-BN 16:")
    for l in lambdas:
        print(f"Evaluating Lambda={l:.1f}")
        # TA
        merged_ta = merge_task_arithmetic(r18_progenitor, r18_experts, lambd=l).to(device)
        res_ta = evaluate_multi_task(merged_ta, datasets, device, num_bits=0, bn_calib_samples=16, seed=42, batch_size=1024, num_workers=4)
        r18_results["TA"].append(res_ta["average_acc"])
        
        # WCPR
        merged_wcpr = merge_wcpr(r18_progenitor, r18_experts, lambd=l).to(device)
        res_wcpr = evaluate_multi_task(merged_wcpr, datasets, device, num_bits=0, bn_calib_samples=16, seed=42, batch_size=1024, num_workers=4)
        r18_results["WCPR"].append(res_wcpr["average_acc"])
        
        # QR-WCPR
        merged_qr = merge_qr_wcpr(r18_progenitor, r18_experts, lambd=l, gamma=1.5).to(device)
        res_qr = evaluate_multi_task(merged_qr, datasets, device, num_bits=0, bn_calib_samples=16, seed=42, batch_size=1024, num_workers=4)
        r18_results["QR-WCPR"].append(res_qr["average_acc"])
        
        print(f"  Lambda={l:.1f} | TA: {res_ta['average_acc']*100:.2f}% | WCPR: {res_wcpr['average_acc']*100:.2f}% | QR-WCPR: {res_qr['average_acc']*100:.2f}%")
        
    # 2. MLP Clean (no DE-BN)
    mlp_progenitor = get_model("mlp").to(device)
    mlp_progenitor.load_state_dict(torch.load("checkpoints/progenitor_mlp.pt", map_location="cpu"))
    
    mlp_experts = []
    for ds in datasets:
        model = get_model("mlp").to(device)
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_mlp.pt", map_location="cpu"))
        mlp_experts.append(model)
        
    mlp_results = {"TA": [], "WCPR": [], "QR-WCPR": []}
    
    print("\nMLP Clean (No DE-BN):")
    for l in lambdas:
        print(f"Evaluating Lambda={l:.1f}")
        # TA
        merged_ta = merge_task_arithmetic(mlp_progenitor, mlp_experts, lambd=l).to(device)
        res_ta = evaluate_multi_task(merged_ta, datasets, device, num_bits=0, bn_calib_samples=0, seed=42, batch_size=1024, num_workers=4)
        mlp_results["TA"].append(res_ta["average_acc"])
        
        # WCPR
        merged_wcpr = merge_wcpr(mlp_progenitor, mlp_experts, lambd=l).to(device)
        res_wcpr = evaluate_multi_task(merged_wcpr, datasets, device, num_bits=0, bn_calib_samples=0, seed=42, batch_size=1024, num_workers=4)
        mlp_results["WCPR"].append(res_wcpr["average_acc"])
        
        # QR-WCPR
        merged_qr = merge_qr_wcpr(mlp_progenitor, mlp_experts, lambd=l, gamma=1.5).to(device)
        res_qr = evaluate_multi_task(merged_qr, datasets, device, num_bits=0, bn_calib_samples=0, seed=42, batch_size=1024, num_workers=4)
        mlp_results["QR-WCPR"].append(res_qr["average_acc"])
        
        print(f"  Lambda={l:.1f} | TA: {res_ta['average_acc']*100:.2f}% | WCPR: {res_wcpr['average_acc']*100:.2f}% | QR-WCPR: {res_qr['average_acc']*100:.2f}%")
        
    return lambdas, r18_results, mlp_results


def run_bitwidth_sweep(device):
    print("\n" + "="*40)
    print("RUNNING BITWIDTH SWEEP (2 TO 8 BITS)")
    print("="*40)
    
    bitwidths = [2, 3, 4, 5, 6, 8]
    datasets = ["mnist", "fmnist", "cifar10"]
    
    # 1. ResNet-18 + DE-BN (16)
    r18_progenitor = get_model("resnet18").to(device)
    r18_progenitor.load_state_dict(torch.load("checkpoints/progenitor_resnet18.pt", map_location="cpu"))
    
    r18_experts = []
    for ds in datasets:
        model = get_model("resnet18").to(device)
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_resnet18.pt", map_location="cpu"))
        r18_experts.append(model)
        
    # We pre-merge models for Bitwidth Sweep (since model weights are static, only quantization bitwidth changes!)
    print("Pre-merging ResNet-18 models for Bitwidth Sweep...")
    r18_merged = {
        "WA": merge_weight_averaging(r18_progenitor, r18_experts).to(device),
        "TA": merge_task_arithmetic(r18_progenitor, r18_experts, lambd=0.3).to(device),
        "WCPR": merge_wcpr(r18_progenitor, r18_experts, lambd=0.5).to(device),
        "QR-WCPR": merge_qr_wcpr(r18_progenitor, r18_experts, lambd=0.5, gamma=1.5).to(device)
    }
    
    r18_results = {"WA": [], "TA": [], "WCPR": [], "QR-WCPR": []}
    
    print("\nResNet-18 (with DE-BN 16):")
    for b in bitwidths:
        print(f"Evaluating Bitwidth={b}")
        for name, model in r18_merged.items():
            res = evaluate_multi_task(model, datasets, device, num_bits=b, bn_calib_samples=16, seed=42, batch_size=1024, num_workers=4)
            r18_results[name].append(res["average_acc"])
            print(f"  [{name}]: {res['average_acc']*100:.2f}%")
            
    # 2. MLP (no DE-BN)
    mlp_progenitor = get_model("mlp").to(device)
    mlp_progenitor.load_state_dict(torch.load("checkpoints/progenitor_mlp.pt", map_location="cpu"))
    
    mlp_experts = []
    for ds in datasets:
        model = get_model("mlp").to(device)
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_mlp.pt", map_location="cpu"))
        mlp_experts.append(model)
        
    print("Pre-merging MLP models for Bitwidth Sweep...")
    mlp_merged = {
        "WA": merge_weight_averaging(mlp_progenitor, mlp_experts).to(device),
        "TA": merge_task_arithmetic(mlp_progenitor, mlp_experts, lambd=0.5).to(device),
        "WCPR": merge_wcpr(mlp_progenitor, mlp_experts, lambd=0.5).to(device),
        "QR-WCPR": merge_qr_wcpr(mlp_progenitor, mlp_experts, lambd=0.5, gamma=1.5).to(device)
    }
    
    mlp_results = {"WA": [], "TA": [], "WCPR": [], "QR-WCPR": []}
    
    print("\nMLP (No DE-BN):")
    for b in bitwidths:
        print(f"Evaluating Bitwidth={b}")
        for name, model in mlp_merged.items():
            res = evaluate_multi_task(model, datasets, device, num_bits=b, bn_calib_samples=0, seed=42, batch_size=1024, num_workers=4)
            mlp_results[name].append(res["average_acc"])
            print(f"  [{name}]: {res['average_acc']*100:.2f}%")
            
    return bitwidths, r18_results, mlp_results


def plot_results(lambdas, r18_lambdas, mlp_lambdas, bitwidths, r18_bits, mlp_bits):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    
    # Plot 1: Lambda Sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    colors = {"TA": "#ff7f0e", "WCPR": "#1f77b4", "QR-WCPR": "#2ca02c"}
    markers = {"TA": "s", "WCPR": "o", "QR-WCPR": "^"}
    
    # ResNet-18 Clean + DE-BN 16
    for m in ["TA", "WCPR", "QR-WCPR"]:
        ax1.plot(lambdas, np.array(r18_lambdas[m])*100, label=m, color=colors[m], marker=markers[m], linewidth=1.8)
    ax1.set_xlabel(r"Merging Scaling Coefficient $\lambda$", fontsize=11)
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax1.set_title("ResNet-18 (FP32 Clean + DE-BN 16)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)
    ax1.set_ylim(8, 50)
    
    # MLP Clean
    for m in ["TA", "WCPR", "QR-WCPR"]:
        ax2.plot(lambdas, np.array(mlp_lambdas[m])*100, label=m, color=colors[m], marker=markers[m], linewidth=1.8)
    ax2.set_xlabel(r"Merging Scaling Coefficient $\lambda$", fontsize=11)
    ax2.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax2.set_title("MLP (FP32 Clean, No DE-BN)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)
    ax2.set_ylim(8, 50)
    
    plt.tight_layout()
    plt.savefig("plot_lambda_sweep.png", dpi=300)
    plt.close()
    print("Saved plot_lambda_sweep.png!")
    
    # Plot 2: Bitwidth Sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    colors_bits = {"WA": "#7f7f7f", "TA": "#ff7f0e", "WCPR": "#1f77b4", "QR-WCPR": "#2ca02c"}
    markers_bits = {"WA": "x", "TA": "s", "WCPR": "o", "QR-WCPR": "^"}
    
    # ResNet-18 Quantization
    for m in ["WA", "TA", "WCPR", "QR-WCPR"]:
        ax1.plot(bitwidths, np.array(r18_bits[m])*100, label=m, color=colors_bits[m], marker=markers_bits[m], linewidth=1.8)
    ax1.set_xlabel("Quantization Bitwidth", fontsize=11)
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax1.set_title("ResNet-18 (with DE-BN 16) vs Bitwidth", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)
    ax1.set_ylim(8, 50)
    
    # MLP Quantization
    for m in ["WA", "TA", "WCPR", "QR-WCPR"]:
        ax2.plot(bitwidths, np.array(mlp_bits[m])*100, label=m, color=colors_bits[m], marker=markers_bits[m], linewidth=1.8)
    ax2.set_xlabel("Quantization Bitwidth", fontsize=11)
    ax2.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax2.set_title("MLP (No DE-BN) vs Bitwidth", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)
    ax2.set_ylim(8, 50)
    
    plt.tight_layout()
    plt.savefig("plot_bitwidth_sweep.png", dpi=300)
    plt.close()
    print("Saved plot_bitwidth_sweep.png!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    print(f"Additional sweeps running on device: {device}")
    
    # 1. Run Lambda Sweep
    lambdas, r18_lambdas, mlp_lambdas = run_lambda_sweep(device)
    
    # 2. Run Bitwidth Sweep
    bitwidths, r18_bits, mlp_bits = run_bitwidth_sweep(device)
    
    # 3. Save all results
    additional_results = {
        "lambda_sweep": {
            "lambdas": lambdas,
            "resnet18": r18_lambdas,
            "mlp": mlp_lambdas
        },
        "bitwidth_sweep": {
            "bitwidths": bitwidths,
            "resnet18": r18_bits,
            "mlp": mlp_bits
        }
    }
    
    with open("additional_sweep_results.json", "w") as f:
        json.dump(additional_results, f, indent=4)
    print("Saved additional_sweep_results.json!")
    
    # 4. Plot results
    plot_results(lambdas, r18_lambdas, mlp_lambdas, bitwidths, r18_bits, mlp_bits)
    print("Additional sweeps complete and plots saved!")

if __name__ == "__main__":
    main()
