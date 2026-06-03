import os
import torch
import torch.nn as nn
import numpy as np
from merge import load_model, dct_2d

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED
torch.backends.cudnn.enabled = False

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
    print(f"Analyzing Spectral Energy Partition (low_freq_ratio={low_freq_ratio}):")
    print("-" * 80)
    
    group_results = {g: {t: [] for t in tasks} for g in groups}
    
    for g_name, keys in groups.items():
        for key in keys:
            if key not in base_state:
                continue
            base_w = base_state[key].float()
            
            for task in tasks:
                expert_w = expert_states[task][key].float()
                update_w = expert_w - base_w
                
                # Reshape to 2D
                shape = update_w.shape
                M = shape[0]
                N = 1
                for s in shape[1:]:
                    N *= s
                
                update_2d = update_w.view(M, N)
                
                # Compute DCT
                D, _, _ = dct_2d(update_2d)
                
                # Compute distance grid
                u_grid = torch.arange(M, device=D.device).unsqueeze(1).expand(M, N)
                v_grid = torch.arange(N, device=D.device).unsqueeze(0).expand(M, N)
                dist = u_grid.float() / M + v_grid.float() / N
                
                threshold = torch.quantile(dist, low_freq_ratio)
                low_mask = dist <= threshold
                high_mask = dist > threshold
                
                # Calculate energies
                total_energy = torch.sum(D ** 2).item()
                low_energy = torch.sum((D * low_mask) ** 2).item()
                high_energy = torch.sum((D * high_mask) ** 2).item()
                
                if total_energy > 0:
                    high_ratio = high_energy / total_energy
                    group_results[g_name][task].append(high_ratio)
                    
    # Print and summarize results
    print(f"{'Layer Group':<15} | {'MNIST High-Freq %':<18} | {'Fashion High-Freq %':<20} | {'CIFAR-10 High-Freq %':<20} | {'Average High-Freq %':<20}")
    print("-" * 101)
    
    for g_name in groups.keys():
        row = []
        avg_vals = []
        for task in tasks:
            vals = group_results[g_name][task]
            avg_val = np.mean(vals) * 100.0 if vals else 0.0
            row.append(f"{avg_val:.2f}%")
            avg_vals.append(avg_val)
        overall_avg = np.mean(avg_vals)
        print(f"{g_name:<15} | {row[0]:<18} | {row[1]:<20} | {row[2]:<20} | {overall_avg:.2f}%")

if __name__ == "__main__":
    main()
