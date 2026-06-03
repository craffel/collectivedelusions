import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from main import (
    MultiTaskResNet, 
    get_datasets, 
    quantize_weights, 
    run_task_specific_de_bn, 
    run_naive_mixed_calibration, 
    run_centroid_aligned_unified_calibration, 
    evaluate_model, 
    merge_models, 
    set_seed
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for sweeps")
    set_seed(42)

    # 1. Load Datasets
    # Use standard batch size of 256
    train_loaders, test_loaders = get_datasets(batch_size=256, dry_run=False)

    # 2. Re-create model structure
    heads = {
        "MNIST": nn.Linear(512, 10),
        "FMNIST": nn.Linear(512, 10),
        "CIFAR10": nn.Linear(512, 10)
    }
    
    # 3. Load pre-trained models from checkpoints
    checkpoint_dir = "./checkpoints"
    print("Loading pre-trained progenitor and experts...")
    
    progenitor = MultiTaskResNet(heads)
    progenitor.load_state_dict(torch.load(f"{checkpoint_dir}/progenitor.pt", map_location=device))
    
    experts = {}
    for task_name in heads.keys():
        expert_model = copy.deepcopy(progenitor)
        expert_model.load_state_dict(torch.load(f"{checkpoint_dir}/{task_name}_expert.pt", map_location=device))
        experts[task_name] = expert_model

    sweep_results = {}

    # ==========================================
    # SWEEP 1: Calibration Sample Size (N) Sensitivity
    # We sweep N in [16, 32, 64, 128, 256] under:
    #   - PC-INT4 Quantization
    #   - Noisy FP32 (std=0.1)
    # ==========================================
    print("\n--- Starting SWEEP 1: Calibration Sample Size N ---")
    N_list = [16, 32, 64, 128, 256]
    sweep_results["N_sensitivity"] = {
        "PC-INT4": {N: {} for N in N_list},
        "Noisy_FP32": {N: {} for N in N_list}
    }
    
    # Pre-merge model with optimal lambda=0.4
    merged_model_opt = merge_models(progenitor, experts, lam=0.4)
    
    for N in N_list:
        print(f"Evaluating Calibration N = {N} ...")
        
        # A. Calibration
        # Task-Specific DE-BN
        task_specific_stats = {}
        for task_name in train_loaders.keys():
            captured = run_task_specific_de_bn(merged_model_opt, task_name, train_loaders[task_name], N=N, device=device)
            task_specific_stats[task_name] = captured
            
        # Naive Mixed Calibration
        naive_cal_model = run_naive_mixed_calibration(merged_model_opt, train_loaders, N=N, device=device)
        # Proposed CA-UC
        ca_uc_model = run_centroid_aligned_unified_calibration(merged_model_opt, train_loaders, N=N, device=device)
        
        # PC-INT4 Evaluation
        naive_model_q4 = quantize_weights(naive_cal_model, bits=4, per_channel=True)
        ca_model_q4 = quantize_weights(ca_uc_model, bits=4, per_channel=True)
        base_merged_q4 = quantize_weights(merged_model_opt, bits=4, per_channel=True)
        
        res_debn_q4 = evaluate_model(base_merged_q4, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats, noise_std=0.0, device=device)
        res_naive_q4 = evaluate_model(naive_model_q4, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        res_cauc_q4 = evaluate_model(ca_model_q4, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        
        sweep_results["N_sensitivity"]["PC-INT4"][N] = {
            "DE-BN (Oracle, routed)": res_debn_q4["Average"],
            "Naive Mixed Cal": res_naive_q4["Average"],
            "Proposed CA-UC": res_cauc_q4["Average"]
        }
        
        # Noisy FP32 Evaluation (std=0.1)
        res_debn_noise = evaluate_model(merged_model_opt, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats, noise_std=0.1, device=device)
        res_naive_noise = evaluate_model(naive_cal_model, test_loaders, calibration_type="None", noise_std=0.1, device=device)
        res_cauc_noise = evaluate_model(ca_uc_model, test_loaders, calibration_type="None", noise_std=0.1, device=device)
        
        sweep_results["N_sensitivity"]["Noisy_FP32"][N] = {
            "DE-BN (Oracle, routed)": res_debn_noise["Average"],
            "Naive Mixed Cal": res_naive_noise["Average"],
            "Proposed CA-UC": res_cauc_noise["Average"]
        }

    # ==========================================
    # SWEEP 2: Robustness to Noise Scaling (sigma)
    # We sweep input noise std in [0.0, 0.05, 0.1, 0.15, 0.2]
    # under optimal lam=0.4, N=64
    # ==========================================
    print("\n--- Starting SWEEP 2: Noise Robustness Sweep ---")
    noise_list = [0.0, 0.05, 0.1, 0.15, 0.2]
    sweep_results["noise_robustness"] = {str(n): {} for n in noise_list}
    
    # Compute base calibration stats at N=64
    task_specific_stats_64 = {}
    for task_name in train_loaders.keys():
        task_specific_stats_64[task_name] = run_task_specific_de_bn(merged_model_opt, task_name, train_loaders[task_name], N=64, device=device)
        
    naive_cal_model_64 = run_naive_mixed_calibration(merged_model_opt, train_loaders, N=64, device=device)
    ca_uc_model_64 = run_centroid_aligned_unified_calibration(merged_model_opt, train_loaders, N=64, device=device)
    
    for noise in noise_list:
        print(f"Evaluating Noise std = {noise} ...")
        
        res_uncal = evaluate_model(merged_model_opt, test_loaders, calibration_type="None", noise_std=noise, device=device)
        res_debn = evaluate_model(merged_model_opt, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats_64, noise_std=noise, device=device)
        res_naive = evaluate_model(naive_cal_model_64, test_loaders, calibration_type="None", noise_std=noise, device=device)
        res_cauc = evaluate_model(ca_uc_model_64, test_loaders, calibration_type="None", noise_std=noise, device=device)
        
        sweep_results["noise_robustness"][str(noise)] = {
            "Uncalibrated": res_uncal["Average"],
            "DE-BN (Oracle, routed)": res_debn["Average"],
            "Naive Mixed Cal": res_naive["Average"],
            "Proposed CA-UC": res_cauc["Average"]
        }

    # ==========================================
    # SWEEP 3: Model Merging Scale (lambda) Sweep
    # We sweep lam in [0.2, 0.3, 0.4, 0.5, 0.6]
    # under N=64, comparing PC-INT4 and FP32
    # ==========================================
    print("\n--- Starting SWEEP 3: Merging Lambda Sweep ---")
    lam_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    sweep_results["lambda_sweep"] = {
        "FP32": {str(l): {} for l in lam_list},
        "PC-INT4": {str(l): {} for l in lam_list}
    }
    
    for l in lam_list:
        print(f"Evaluating Merging Lambda = {l} ...")
        merged_l = merge_models(progenitor, experts, lam=l)
        
        # Calibrate with N=64
        task_specific_stats_l = {}
        for task_name in train_loaders.keys():
            task_specific_stats_l[task_name] = run_task_specific_de_bn(merged_l, task_name, train_loaders[task_name], N=64, device=device)
            
        naive_cal_l = run_naive_mixed_calibration(merged_l, train_loaders, N=64, device=device)
        ca_uc_l = run_centroid_aligned_unified_calibration(merged_l, train_loaders, N=64, device=device)
        
        # 1. FP32
        res_debn_fp = evaluate_model(merged_l, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats_l, noise_std=0.0, device=device)
        res_naive_fp = evaluate_model(naive_cal_l, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        res_cauc_fp = evaluate_model(ca_uc_l, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        
        sweep_results["lambda_sweep"]["FP32"][str(l)] = {
            "DE-BN (Oracle, routed)": res_debn_fp["Average"],
            "Naive Mixed Cal": res_naive_fp["Average"],
            "Proposed CA-UC": res_cauc_fp["Average"]
        }
        
        # 2. PC-INT4
        merged_l_q4 = quantize_weights(merged_l, bits=4, per_channel=True)
        naive_cal_l_q4 = quantize_weights(naive_cal_l, bits=4, per_channel=True)
        ca_uc_l_q4 = quantize_weights(ca_uc_l, bits=4, per_channel=True)
        
        res_debn_q4 = evaluate_model(merged_l_q4, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats_l, noise_std=0.0, device=device)
        res_naive_q4 = evaluate_model(naive_cal_l_q4, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        res_cauc_q4 = evaluate_model(ca_uc_l_q4, test_loaders, calibration_type="None", noise_std=0.0, device=device)
        
        sweep_results["lambda_sweep"]["PC-INT4"][str(l)] = {
            "DE-BN (Oracle, routed)": res_debn_q4["Average"],
            "Naive Mixed Cal": res_naive_q4["Average"],
            "Proposed CA-UC": res_cauc_q4["Average"]
        }

    # Save results to JSON
    with open("sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
    print("\nSuccessfully saved all sweep results to sweep_results.json!")

    # ==========================================
    # PLOTTING SWEEP RESULTS
    # ==========================================
    methods = ["DE-BN (Oracle, routed)", "Naive Mixed Cal", "Proposed CA-UC"]
    colors = {"DE-BN (Oracle, routed)": "#3498db", "Naive Mixed Cal": "#e74c3c", "Proposed CA-UC": "#2ecc71", "Uncalibrated": "#7f8c8d"}
    markers = {"DE-BN (Oracle, routed)": "o", "Naive Mixed Cal": "s", "Proposed CA-UC": "^", "Uncalibrated": "x"}
    
    # Plot 1: N Sensitivity under PC-INT4 and Noisy FP32
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for m in methods:
        y_q4 = [sweep_results["N_sensitivity"]["PC-INT4"][N][m] for N in N_list]
        axes[0].plot(N_list, y_q4, marker=markers[m], color=colors[m], label=m, linewidth=2)
    axes[0].set_xlabel("Calibration Sample Size (N)", fontsize=11)
    axes[0].set_ylabel("Multi-task Average Accuracy (%)", fontsize=11)
    axes[0].set_title("N Sensitivity under 4-bit PTQ", fontsize=12, fontweight="bold")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(N_list)
    axes[0].set_xticklabels([str(N) for N in N_list])
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    for m in methods:
        y_noise = [sweep_results["N_sensitivity"]["Noisy_FP32"][N][m] for N in N_list]
        axes[1].plot(N_list, y_noise, marker=markers[m], color=colors[m], label=m, linewidth=2)
    axes[1].set_xlabel("Calibration Sample Size (N)", fontsize=11)
    axes[1].set_ylabel("Multi-task Average Accuracy (%)", fontsize=11)
    axes[1].set_title("N Sensitivity under Input Noise (std=0.1)", fontsize=12, fontweight="bold")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(N_list)
    axes[1].set_xticklabels([str(N) for N in N_list])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("sweep_sample_size.png", dpi=300)
    plt.close()

    # Plot 2: Noise Robustness Sweep
    plt.figure(figsize=(7, 5))
    methods_noise = ["Uncalibrated", "DE-BN (Oracle, routed)", "Naive Mixed Cal", "Proposed CA-UC"]
    for m in methods_noise:
        y_noise = [sweep_results["noise_robustness"][str(n)][m] for n in noise_list]
        plt.plot(noise_list, y_noise, marker=markers.get(m, "x"), color=colors.get(m, "#7f8c8d"), label=m, linewidth=2)
    plt.xlabel("Input Noise Standard Deviation (std)", fontsize=11)
    plt.ylabel("Multi-task Average Accuracy (%)", fontsize=11)
    plt.title("Robustness to Environmental Input Noise Scaling", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sweep_noise.png", dpi=300)
    plt.close()

    # Plot 3: Lambda Sweep
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for m in methods:
        y_fp = [sweep_results["lambda_sweep"]["FP32"][str(l)][m] for l in lam_list]
        axes[0].plot(lam_list, y_fp, marker=markers[m], color=colors[m], label=m, linewidth=2)
    axes[0].set_xlabel("Backbone Merge Scaling Factor (lambda)", fontsize=11)
    axes[0].set_ylabel("Multi-task Average Accuracy (%)", fontsize=11)
    axes[0].set_title("Merge Scale lambda Sweep under Clean FP32", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    for m in methods:
        y_q4 = [sweep_results["lambda_sweep"]["PC-INT4"][str(l)][m] for l in lam_list]
        axes[1].plot(lam_list, y_q4, marker=markers[m], color=colors[m], label=m, linewidth=2)
    axes[1].set_xlabel("Backbone Merge Scaling Factor (lambda)", fontsize=11)
    axes[1].set_ylabel("Multi-task Average Accuracy (%)", fontsize=11)
    axes[1].set_title("Merge Scale lambda Sweep under 4-bit PTQ", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("sweep_lambda.png", dpi=300)
    plt.close()
    
    print("All sweep plots successfully generated and saved!")

if __name__ == "__main__":
    main()
