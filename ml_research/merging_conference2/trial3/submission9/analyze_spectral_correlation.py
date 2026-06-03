import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from merge import load_model, dct_2d

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED
torch.backends.cudnn.enabled = False

def cosine_similarity(t1, t2):
    denom = torch.norm(t1) * torch.norm(t2)
    if denom == 0:
        return 0.0
    return torch.sum(t1 * t2).item() / denom.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Load base model and experts
    base_model = load_model("checkpoints/base.pt")
    base_state = base_model.state_dict()
    
    experts = {task: load_model(f"checkpoints/{task}.pt") for task in tasks}
    expert_states = {task: experts[task].state_dict() for task in tasks}
    
    # Define layer groups
    groups = {
        "conv1": ["conv1.weight"],
        "layer1": [
            "layer1.0.conv1.weight", "layer1.0.conv2.weight",
            "layer1.1.conv1.weight", "layer1.1.conv2.weight"
        ],
        "layer2": [
            "layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer2.0.downsample.0.weight",
            "layer2.1.conv1.weight", "layer2.1.conv2.weight"
        ],
        "layer3": [
            "layer3.0.conv1.weight", "layer3.0.conv2.weight", "layer3.0.downsample.0.weight",
            "layer3.1.conv1.weight", "layer3.1.conv2.weight"
        ],
        "layer4": [
            "layer4.0.conv1.weight", "layer4.0.conv2.weight", "layer4.0.downsample.0.weight",
            "layer4.1.conv1.weight", "layer4.1.conv2.weight"
        ],
    }
    
    low_freq_ratio = 0.1
    print(f"Analyzing Spectral Task Correlation (low_freq_ratio={low_freq_ratio}):")
    print("-" * 100)
    
    # Pairs of tasks
    pairs = [("mnist", "fashion"), ("mnist", "cifar10"), ("fashion", "cifar10")]
    
    group_spatial_sims = {g: [] for g in groups}
    group_low_sims = {g: [] for g in groups}
    group_high_sims = {g: [] for g in groups}
    
    for g_name, keys in groups.items():
        for key in keys:
            if key not in base_state:
                continue
            base_w = base_state[key].float()
            
            # Compute updates for each task
            updates = {}
            for task in tasks:
                expert_w = expert_states[task][key].float()
                updates[task] = (expert_w - base_w).to(device)
            
            for t1, t2 in pairs:
                u1 = updates[t1]
                u2 = updates[t2]
                
                # Flatten or reshape to 2D
                shape = u1.shape
                M = shape[0]
                N = 1
                for s in shape[1:]:
                    N *= s
                
                u1_2d = u1.view(M, N)
                u2_2d = u2.view(M, N)
                
                # Compute spatial similarity
                sp_sim = cosine_similarity(u1_2d, u2_2d)
                group_spatial_sims[g_name].append(sp_sim)
                
                # Compute DCT
                D1, _, _ = dct_2d(u1_2d)
                D2, _, _ = dct_2d(u2_2d)
                
                # Compute distance grid
                u_grid = torch.arange(M, device=device).unsqueeze(1).expand(M, N)
                v_grid = torch.arange(N, device=device).unsqueeze(0).expand(M, N)
                dist = u_grid.float() / M + v_grid.float() / N
                
                threshold = torch.quantile(dist, low_freq_ratio)
                low_mask = dist <= threshold
                high_mask = dist > threshold
                
                # Project onto low and high components
                D1_low = D1 * low_mask
                D2_low = D2 * low_mask
                
                D1_high = D1 * high_mask
                D2_high = D2 * high_mask
                
                low_sim = cosine_similarity(D1_low, D2_low)
                high_sim = cosine_similarity(D1_high, D2_high)
                
                group_low_sims[g_name].append(low_sim)
                group_high_sims[g_name].append(high_sim)
                
    # Print results
    print(f"{'Layer Group':<15} | {'Spatial Sim':<15} | {'Low-Freq Sim':<15} | {'High-Freq Sim':<15}")
    print("-" * 68)
    
    g_names = list(groups.keys())
    avg_spatials = []
    avg_lows = []
    avg_highs = []
    
    for g_name in g_names:
        sp_mean = np.mean(group_spatial_sims[g_name])
        low_mean = np.mean(group_low_sims[g_name])
        high_mean = np.mean(group_high_sims[g_name])
        
        avg_spatials.append(sp_mean)
        avg_lows.append(low_mean)
        avg_highs.append(high_mean)
        
        print(f"{g_name:<15} | {sp_mean:14.4f} | {low_mean:14.4f} | {high_mean:14.4f}")
        
    overall_spatial = np.mean(avg_spatials)
    overall_low = np.mean(avg_lows)
    overall_high = np.mean(avg_highs)
    print("-" * 68)
    print(f"{'OVERALL AVERAGE':<15} | {overall_spatial:14.4f} | {overall_low:14.4f} | {overall_high:14.4f}")
    
    # Save a plot visualizing this
    plt.figure(figsize=(8, 5))
    x = np.arange(len(g_names))
    width = 0.25
    
    plt.bar(x - width, avg_spatials, width, label='Spatial Domain', color='#7f7f7f', alpha=0.8)
    plt.bar(x, avg_lows, width, label='Low-Frequency (Spectral)', color='#1f77b4', alpha=0.9)
    plt.bar(x + width, avg_highs, width, label='High-Frequency (Spectral)', color='#d62728', alpha=0.9)
    
    plt.xlabel('Layer Group', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Task Update Similarity: Spatial vs Spectral Bands', fontsize=14, fontweight='bold')
    plt.xticks(x, g_names)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = "spectral_similarity_comparison.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.savefig("spectral_similarity_comparison.png", bbox_inches='tight', dpi=300)
    print(f"Saved spectral similarity comparison plots to {plot_path}")

if __name__ == "__main__":
    main()
