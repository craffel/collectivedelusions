import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
from torch.utils.data import Subset

# Import everything from tta_eval
from tta_eval import (
    device,
    test_subset_A,
    build_merged_model,
    load_teacher,
    run_tta,
    LoRAConv2d,
    apply_corruption
)

def extract_singular_values(model):
    """
    Extracts the singular values of W_update = B_flat @ A_flat for all LoRAConv2d layers,
    and returns their average across layers.
    """
    all_s_vals = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            A = module.lora_A.weight  # shape [r, in_channels, K, K]
            B = module.lora_B.weight  # shape [out_channels, r, 1, 1]
            
            A_flat = A.view(A.size(0), -1)  # shape [r, in_channels * K * K]
            B_flat = B.view(B.size(0), -1)  # shape [out_channels, r]
            
            # W_update shape [out_channels, in_channels * K * K]
            W_update = torch.matmul(B_flat, A_flat)
            
            # SVD of W_update
            W_numpy = W_update.detach().cpu().numpy()
            _, s, _ = np.linalg.svd(W_numpy, full_matrices=False)
            
            # We are interested in the first 8 singular values (as rank is 8)
            s_8 = s[:8]
            # Pad with zeros if less than 8
            if len(s_8) < 8:
                s_8 = np.pad(s_8, (0, 8 - len(s_8)), 'constant')
                
            all_s_vals.append(s_8)
            
    # Average across all 8 LoRA layers
    avg_s_vals = np.mean(all_s_vals, axis=0)
    return avg_s_vals

def main():
    print("Starting Spectral Analysis of LoRA Updates...")
    
    # We use SAM-trained experts as initialization, where representation collapse is most severe
    expert_type = "sam"
    
    # 1. Load Teachers
    print("Loading teacher and head...")
    teacher_A = load_teacher("A", expert_type)
    head_state_A = torch.load(f"./checkpoints/head_A_{expert_type}.pt", map_location=device)
    
    # 2. Setup 512-image TTA calibration pool for Task A (same as in tta_eval.py)
    torch.manual_seed(42)
    indices_A = torch.randperm(len(test_subset_A))[:512]
    tta_pool_A = Subset(test_subset_A, indices_A)
    
    corruption = "blur"  # We use blur as a representative corruption where SAT-SyMerge collapses
    
    # Methods to evaluate
    methods = ["baseline", "symerge", "sat-symerge", "o-lorta"]
    method_labels = {
        "baseline": "Baseline (No TTA)",
        "symerge": "SyMerge (Standard TTA)",
        "sat-symerge": "SAT-SyMerge (SAM TTA)",
        "o-lorta": "O-LoRTA (Ours)"
    }
    
    results = {}
    
    for method in methods:
        print(f"\nRunning TTA for method: {method.upper()}...")
        # Fresh merged model
        model = build_merged_model(expert_type)
        
        if method != "baseline":
            # Run TTA (which modifies the model parameters in-place)
            _ = run_tta(
                merged_model=model,
                teacher_model=teacher_A,
                tta_data_pool=tta_pool_A,
                head_state=head_state_A,
                corruption_type=corruption,
                method=method,
                lr=1e-3,
                steps=10,
                rho=0.05,
                beta=0.1
            )
            
        # Extract average singular values of LoRA layers
        s_vals = extract_singular_values(model)
        results[method] = s_vals
        print(f"Method {method.upper()} Average Singular Values: {s_vals}")
        
    # Save the results to a JSON file
    results_list = {k: v.tolist() for k, v in results.items()}
    with open("spectrum_results.json", "w") as f:
        json.dump(results_list, f, indent=4)
    print("Saved spectrum results to spectrum_results.json.")
    
    # Let's generate a beautiful visualization plot
    plt.figure(figsize=(7, 4.5), dpi=300)
    x = np.arange(1, 9)  # 8 ranks
    
    colors = {
        "baseline": "#7F8C8D",       # Gray (Baseline)
        "symerge": "#3498DB",        # Blue (SyMerge)
        "sat-symerge": "#E74C3C",    # Red (SAT-SyMerge)
        "o-lorta": "#2ECC71"         # Green (O-LoRTA)
    }
    
    for method in methods:
        # We plot normalized singular values to show relative energy / anisotropy
        s_norm = results[method] / np.sum(results[method])
        plt.plot(x, s_norm, marker='o', linewidth=2.5, markersize=7, color=colors[method], label=method_labels[method])
        
    plt.title("LoRA Update Singular Value Spectrum (Isotropy vs. Collapse)", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Singular Value Index (Rank Dimension)", fontsize=10)
    plt.ylabel("Normalized Singular Value Energy", fontsize=10)
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=9)
    plt.tight_layout()
    plt.savefig("spectrum_plot.png", dpi=300)
    print("Saved spectrum plot as spectrum_plot.png.")

if __name__ == "__main__":
    main()
