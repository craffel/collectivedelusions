import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Import from merge_and_calibrate
import merge_and_calibrate as mc

# Set style for publication-ready figures
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

def generate_performance_plot():
    print("Generating performance comparison plot...")
    # Data from progress.md
    methods = ["Uncalibrated", "TCAC", "N-TAAC", "SP-TAAC", "L-FDSA (High Latency)", "Hybrid SLR-WBC (Ours)"]
    n_values = [16, 64, 128]
    
    # Accuracy values for each N (rows corresponding to N=[16, 64, 128])
    # [Uncalibrated, TCAC, N-TAAC, SP-TAAC, L-FDSA, SLR-WBC]
    accs = {
        16: [32.84, 9.71, 36.08, 53.57, 55.26, 46.58],
        64: [32.84, 32.34, 43.32, 53.18, 55.07, 53.17],
        128: [32.84, 58.41, 52.27, 53.11, 55.24, 55.16]
    }
    
    oracle_avg = 81.95
    uncal_avg = 32.84
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    
    # Width of a bar
    width = 0.12
    
    # Positions of groups on x-axis
    ind = np.arange(len(n_values))
    
    colors = [
        '#7f7f7f', # Uncalibrated - Gray
        '#bcbd22', # TCAC - Olive
        '#d62728', # N-TAAC - Red
        '#2ca02c', # SP-TAAC - Green
        '#ff7f0e', # L-FDSA - Orange
        '#1f77b4'  # SLR-WBC - Deep Blue
    ]
    
    # Plotting each method's bars
    for i, method in enumerate(methods):
        bar_vals = [accs[n][i] for n in n_values]
        # Offset position for each bar in the group
        pos = ind - (len(methods) * width)/2 + (i + 0.5) * width
        
        # Draw bars
        bars = ax.bar(pos, bar_vals, width, label=method, color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add values on top of our SLR-WBC bars
        if method == "Hybrid SLR-WBC (Ours)":
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),  # 2 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
                
    # Add horizontal line for Oracle
    ax.axhline(y=oracle_avg, color='purple', linestyle='--', linewidth=1.2, label="Oracle Experts (Upper Bound: 81.95%)")
    # Add horizontal line for Uncalibrated weight averaging
    ax.axhline(y=uncal_avg, color='gray', linestyle=':', linewidth=1.2, label="Uncalibrated Merging (32.84%)")
    
    ax.set_xlabel("Calibration Budget N (Samples per Task)", fontweight='bold')
    ax.set_ylabel("Average Multi-Task Accuracy (%)", fontweight='bold')
    ax.set_title("Performance Comparison under Varying Calibration Budgets", fontweight='bold')
    ax.set_xticks(ind)
    ax.set_xticklabels([f"N={n}" for n in n_values])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Put legend below the plot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=True)
    
    plt.tight_layout()
    os.makedirs("template", exist_ok=True)
    plt.savefig("template/perf_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Performance comparison plot saved successfully to template/perf_comparison.png!")

def generate_variance_and_spectrum_plot():
    print("Loading experts on CPU...")
    # Setup CPU device
    device = torch.device("cpu")
    mc.device = device
    
    experts = mc.load_experts()
    base_merged = mc.create_merged_model(experts)
    
    # Prepare Calibration set with N = 128
    N = 128
    calibration_sets = {}
    for ds in ["mnist", "fashion", "cifar10"]:
        full_train = mc.get_dataset(ds, train=True)
        calibration_sets[ds] = Subset(full_train, list(range(5000, 5000 + N)))
        
    print("Calibrating merged model via Hybrid SLR-WBC (Rank=4, Reg=0.01) on CPU...")
    calibrated_model = mc.calibrate_slr_wbc(base_merged, experts, calibration_sets, rank=4, reg=0.01)
    
    # Extract 128 samples from joint dataset to feed through models
    joint_dataset = torch.utils.data.ConcatDataset(list(calibration_sets.values()))
    loader = DataLoader(joint_dataset, batch_size=128, shuffle=False)
    x_joint = next(iter(loader))[0].to(device)
    
    # Dictionary to hold the collected activation statistics at model.layer4[1].bn2
    activations = {}
    
    # Helper to capture forward activations
    def get_layer4_activations(model, inputs):
        out_list = []
        target_layer = model.layer4[1].bn2
        handle = target_layer.register_forward_hook(lambda m, i, o: out_list.append(o.detach()))
        with torch.no_grad():
            model(inputs)
        handle.remove()
        return out_list[0]
        
    print("Collecting activations from experts, uncalibrated merged, and SLR-WBC models...")
    # 1. Experts (run each model on its task-specific subset)
    expert_outs = []
    for ds in ["mnist", "fashion", "cifar10"]:
        ds_loader = DataLoader(calibration_sets[ds], batch_size=128, shuffle=False)
        x_task = next(iter(ds_loader))[0].to(device)
        act = get_layer4_activations(experts[ds], x_task)
        expert_outs.append(act)
        
    # Standard target activation is average of experts' activations
    # We will average the channel-wise standard deviations of the three experts as target
    expert_stds = []
    expert_singular_values = []
    for act in expert_outs:
        # Shape: (B, C, H, W) -> (C, B * H * W)
        B, C, H, W = act.shape
        act_matrix = act.transpose(0, 1).reshape(C, -1)
        
        # Standard deviation per channel
        std = torch.std(act_matrix, dim=1)
        expert_stds.append(std)
        
        # Singular values
        # Center the matrix first for PCA spectrum
        act_centered = act_matrix - act_matrix.mean(dim=1, keepdim=True)
        _, S, _ = torch.linalg.svd(act_centered, full_matrices=False)
        expert_singular_values.append(S)
        
    # Average across experts
    target_std = torch.stack(expert_stds).mean(dim=0).numpy()
    target_S = torch.stack(expert_singular_values).mean(dim=0).numpy()
    
    # 2. Uncalibrated merged model on joint dataset
    act_uncal = get_layer4_activations(base_merged, x_joint)
    B, C, H, W = act_uncal.shape
    act_uncal_matrix = act_uncal.transpose(0, 1).reshape(C, -1)
    uncal_std = torch.std(act_uncal_matrix, dim=1).numpy()
    act_uncal_centered = act_uncal_matrix - act_uncal_matrix.mean(dim=1, keepdim=True)
    _, S_uncal, _ = torch.linalg.svd(act_uncal_centered, full_matrices=False)
    uncal_S = S_uncal.numpy()
    
    # 3. Calibrated merged model (Hybrid SLR-WBC) on joint dataset
    act_cal = get_layer4_activations(calibrated_model, x_joint)
    act_cal_matrix = act_cal.transpose(0, 1).reshape(C, -1)
    cal_std = torch.std(act_cal_matrix, dim=1).numpy()
    act_cal_centered = act_cal_matrix - act_cal_matrix.mean(dim=1, keepdim=True)
    _, S_cal, _ = torch.linalg.svd(act_cal_centered, full_matrices=False)
    cal_S = S_cal.numpy()
    
    # Sort for beautiful visualization
    target_std_sorted = np.sort(target_std)[::-1]
    uncal_std_sorted = np.sort(uncal_std)[::-1]
    cal_std_sorted = np.sort(cal_std)[::-1]
    
    # Normalize singular values to plot relative spectrum decay
    target_S_norm = target_S / target_S[0]
    uncal_S_norm = uncal_S / uncal_S[0]
    cal_S_norm = cal_S / cal_S[0]
    
    print("Plotting figures...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    
    # Plot 1: Channel-wise standard deviations
    channels = np.arange(C)
    ax1.plot(channels, target_std_sorted, color='purple', linestyle='--', linewidth=1.5, label="Oracle Target (Experts Avg)")
    ax1.plot(channels, uncal_std_sorted, color='gray', linestyle='-', linewidth=1.5, label="Uncalibrated Merging")
    ax1.plot(channels, cal_std_sorted, color='#1f77b4', linestyle='-', linewidth=1.8, label="Hybrid SLR-WBC (Ours)")
    ax1.set_xlabel("Channels (Sorted by std)", fontweight='bold')
    ax1.set_ylabel("Standard Deviation of Activations", fontweight='bold')
    ax1.set_title("(a) Representation Variance Collapse at layer4.1.bn2", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)
    
    # Plot 2: Singular value spectrum decay
    singular_indices = np.arange(len(target_S_norm))
    ax2.plot(singular_indices, target_S_norm, color='purple', linestyle='--', linewidth=1.5, label="Oracle Target (Experts Avg)")
    ax2.plot(singular_indices, uncal_S_norm, color='gray', linestyle='-', linewidth=1.5, label="Uncalibrated Merging")
    ax2.plot(singular_indices, cal_S_norm, color='#1f77b4', linestyle='-', linewidth=1.8, label="Hybrid SLR-WBC (Ours)")
    ax2.set_xlabel("Singular Value Rank", fontweight='bold')
    ax2.set_ylabel("Normalized Singular Values ($S_i / S_0$)", fontweight='bold')
    ax2.set_title("(b) Feature Spectrum Decay (Dimensionality)", fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig("template/variance_collapse.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Variance and spectrum plots saved successfully to template/variance_collapse.png!")

if __name__ == "__main__":
    generate_performance_plot()
    generate_variance_and_spectrum_plot()
    print("\nAll plots generated successfully!")
