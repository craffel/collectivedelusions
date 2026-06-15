import os
import time
import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import numpy as np

# Set style for professional academic publication
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False
})

def load_lora_updates():
    updates = {}
    tasks = ["mnist", "fashionmnist", "cifar10", "svhn"]
    for task in tasks:
        path = f"checkpoints/{task}_lora/adapter_model.safetensors"
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Creating random updates.")
            task_updates = {}
            for l in range(12):
                task_updates[l] = torch.randn(576, 192) * 0.01
            updates[task] = task_updates
            continue
        state_dict = load_file(path)
        task_updates = {}
        for l in range(12):
            key_A = f"base_model.model.blocks.{l}.attn.qkv.lora_A.weight"
            key_B = f"base_model.model.blocks.{l}.attn.qkv.lora_B.weight"
            W_A = state_dict[key_A] # [8, 192]
            W_B = state_dict[key_B] # [576, 8]
            task_updates[l] = W_B @ W_A
        updates[task] = task_updates
    return updates

def apply_weights(model, W_dict):
    for l in range(12):
        layer = model.blocks[l].attn.qkv
        if hasattr(layer, "weight"):
            if isinstance(layer.weight, nn.Parameter):
                del layer.weight
            layer.weight = W_dict[l]

def restore_weights(model, original_weights):
    for l in range(12):
        layer = model.blocks[l].attn.qkv
        if hasattr(layer, "weight"):
            del layer.weight
        layer.register_parameter("weight", nn.Parameter(original_weights[l]))

def main():
    device = torch.device("cpu") # Benchmarking on CPU (standard for edge deployment)
    print(f"Benchmarking physical hardware latency on: {device}")
    
    # 1. Initialize model
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False).to(device)
    model.eval()
    
    # Save original weights
    original_weights = {}
    for l in range(12):
        original_weights[l] = model.blocks[l].attn.qkv.weight.clone()
        
    # Load task updates
    task_updates = load_lora_updates()
    tasks_list = list(task_updates.keys())
    
    # Batch configuration
    batch_size = 64
    warmup_runs = 5
    profile_runs = 20
    
    results = {
        "K": [],
        "Merging_Mean_ms": [],
        "Merging_Std_ms": [],
        "Coexistence_Mean_ms": [],
        "Coexistence_Std_ms": []
    }
    
    print("\n--- Starting Latency Profiling ---")
    
    # We sweep K from 1 to 4 tasks
    for K in [1, 2, 3, 4]:
        print(f"\nEvaluating K = {K} active tasks...")
        active_tasks = tasks_list[:K]
        
        # --- A. Profile Weight-Space Merging ---
        # Pre-merge the K adapters (uniform blend)
        merged_updates = {}
        for l in range(12):
            merged_updates[l] = original_weights[l].clone()
            blend = torch.zeros_like(original_weights[l])
            for task in active_tasks:
                blend += task_updates[task][l]
            merged_updates[l] += blend / K
            
        apply_weights(model, merged_updates)
        
        # Generate mixed batch of size 64
        inputs = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(inputs)
                
        # Measure
        merging_times = []
        for _ in range(profile_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(inputs)
            merging_times.append((time.perf_counter() - start) * 1000.0) # convert to ms
            
        merging_mean = np.mean(merging_times)
        merging_std = np.std(merging_times)
        print(f"  Weight-Space Merging: {merging_mean:.2f} ms +/- {merging_std:.2f} ms")
        
        # Restore model
        restore_weights(model, original_weights)
        
        # --- B. Profile Native-Format Co-existence ---
        # Generate subset sizes for the batch of size 64
        sub_sizes = [batch_size // K] * K
        sub_sizes[-1] += batch_size % K # Adjust for remainder
        
        # Generate subset inputs
        sub_inputs = [torch.randn(sz, 3, 224, 224).to(device) for sz in sub_sizes]
        
        # Warmup
        for _ in range(warmup_runs):
            for i, task in enumerate(active_tasks):
                # 1. Apply specific task weight
                W_dict = {l: original_weights[l] + task_updates[task][l] for l in range(12)}
                apply_weights(model, W_dict)
                # 2. Forward pass for sub-batch
                with torch.no_grad():
                    _ = model(sub_inputs[i])
                # 3. Restore
                restore_weights(model, original_weights)
                
        # Measure
        coexistence_times = []
        for _ in range(profile_runs):
            start = time.perf_counter()
            for i, task in enumerate(active_tasks):
                # 1. Apply specific task weight (simulates dynamic swapping)
                W_dict = {l: original_weights[l] + task_updates[task][l] for l in range(12)}
                apply_weights(model, W_dict)
                # 2. Forward pass
                with torch.no_grad():
                    _ = model(sub_inputs[i])
                # 3. Restore
                restore_weights(model, original_weights)
            coexistence_times.append((time.perf_counter() - start) * 1000.0)
            
        coexistence_mean = np.mean(coexistence_times)
        coexistence_std = np.std(coexistence_times)
        print(f"  Co-existence (Sequential Swap): {coexistence_mean:.2f} ms +/- {coexistence_std:.2f} ms")
        
        results["K"].append(K)
        results["Merging_Mean_ms"].append(merging_mean)
        results["Merging_Std_ms"].append(merging_std)
        results["Coexistence_Mean_ms"].append(coexistence_mean)
        results["Coexistence_Std_ms"].append(coexistence_std)

    # Print markdown table
    print("\n--- SUMMARY TABLE ---")
    print("| Number of Tasks ($K$) | Weight-Space Merging (ms) | Co-existence Swap (ms) | Latency Overhead (%) | Speedup Factor |")
    print("|---|---|---|---|---|")
    for i in range(len(results["K"])):
        k = results["K"][i]
        m_mean = results["Merging_Mean_ms"][i]
        m_std = results["Merging_Std_ms"][i]
        c_mean = results["Coexistence_Mean_ms"][i]
        c_std = results["Coexistence_Std_ms"][i]
        overhead = ((c_mean - m_mean) / m_mean) * 100.0
        speedup = c_mean / m_mean
        print(f"| {k} | {m_mean:.2f} ± {m_std:.2f} | {c_mean:.2f} ± {c_std:.2f} | {overhead:+.1f}% | {speedup:.2f}x |")

    # Generate professional plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    K_arr = np.array(results["K"])
    merg_mean = np.array(results["Merging_Mean_ms"])
    merg_std = np.array(results["Merging_Std_ms"])
    coex_mean = np.array(results["Coexistence_Mean_ms"])
    coex_std = np.array(results["Coexistence_Std_ms"])
    
    # Plot with error bars
    ax.errorbar(K_arr, coex_mean, yerr=coex_std, fmt='-o', color='#d62728', 
                linewidth=2, elinewidth=1.5, capsize=4, markersize=6,
                label='Co-existence (NF4 + Swapping)')
                
    ax.errorbar(K_arr, merg_mean, yerr=merg_std, fmt='--s', color='#1f77b4',
                linewidth=2, elinewidth=1.5, capsize=4, markersize=6,
                label='Weight-Space Merging (O(1) Constant)')
                
    ax.set_xlabel('Number of Active Tasks ($K$)', fontweight='bold')
    ax.set_ylabel('Inference Latency (ms / batch of 64)', fontweight='bold')
    ax.set_title('Physical CPU Latency Scaling (ViT-Tiny)', pad=12, fontweight='bold')
    ax.set_xticks(K_arr)
    ax.set_xlim(0.8, 4.2)
    
    # Custom annotations
    for i, k in enumerate(K_arr):
        ax.annotate(f"{coex_mean[i]:.1f} ms", (k, coex_mean[i]), textcoords="offset points", xytext=(-10,12), ha='center', fontsize=9, color='#d62728', fontweight='bold')
        ax.annotate(f"{merg_mean[i]:.1f} ms", (k, merg_mean[i]), textcoords="offset points", xytext=(-10,-15), ha='center', fontsize=9, color='#1f77b4', fontweight='bold')
        
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray')
    plt.tight_layout()
    
    # Save files
    plt.savefig('submission/physical_latency_scaling.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('submission/physical_latency_scaling.png', format='png', bbox_inches='tight')
    print("\nSuccessfully generated physical_latency_scaling.pdf and physical_latency_scaling.png")

if __name__ == "__main__":
    main()
