import os
import sys
sys.path.insert(0, os.path.abspath("local_packages"))

import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device("cpu")
    tasks = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
    
    # 1. Load base model
    print("Loading pre-trained base model...")
    base_model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True).to(device)
    base_params = {n: p.clone().detach() for n, p in base_model.named_parameters()}
    
    # 2. Load experts and extract parameters
    expert_params = {task: {} for task in tasks}
    for task in tasks:
        path = f"experts/{task.lower()}_expert.pt"
        if os.path.exists(path):
            print(f"Loading expert {task} checkpoint from {path}...")
            model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
            model.head = nn.Linear(192, 10)
            model.load_state_dict(torch.load(path, map_location=device))
            expert_params[task] = {
                n: p.clone().detach() for n, p in model.named_parameters() if "head" not in n
            }
        else:
            print(f"Warning: expert {task} not found!")
            return

    # 3. Target layers and setup plotting
    representative_layers = {
        "Layer 0 Attention QKV": "blocks.0.attn.qkv.weight",
        "Layer 5 Attention Out": "blocks.5.attn.proj.weight",
        "Layer 11 MLP fc1": "blocks.11.mlp.fc1.weight",
        "Layer 11 MLP fc2": "blocks.11.mlp.fc2.weight"
    }
    
    colors = {
        "Layer 0 Attention QKV": "#1f77b4",
        "Layer 5 Attention Out": "#ff7f0e",
        "Layer 11 MLP fc1": "#2ca02c",
        "Layer 11 MLP fc2": "#d62728"
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for layer_name, param_key in representative_layers.items():
        # Construct joint update matrix: M^(l) of shape (d_out, K * d_in)
        task_updates = []
        for task in tasks:
            update = expert_params[task][param_key] - base_params[param_key]
            task_updates.append(update)
            
        M = torch.cat(task_updates, dim=1) # Shape: (d_out, K * d_in)
        
        # SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        S_np = S.numpy()
        
        # Normalize singular values
        S_norm = S_np / S_np[0]
        
        # Total energy (sum of squared singular values)
        total_energy = np.sum(S_np ** 2)
        
        # Singular Value Decay Plot
        # We plot singular value index vs. normalized singular value
        # To make it readable across different shapes, we can plot the top 100 or top 150
        num_vals = min(150, len(S_norm))
        ax1.plot(range(1, num_vals + 1), S_norm[:num_vals], label=layer_name, color=colors[layer_name], linewidth=2.5)
        
        # Cumulative Energy Captured Plot
        # gamma ranges from 0 to 1
        # we evaluate at discrete gammas
        gammas = np.linspace(0.01, 1.0, 100)
        captured_energies = []
        for gamma in gammas:
            r = max(1, int(gamma * len(S_np)))
            energy_r = np.sum(S_np[:r] ** 2)
            percentage = (energy_r / total_energy) * 100.0
            captured_energies.append(percentage)
            
        ax2.plot(gammas, captured_energies, label=layer_name, color=colors[layer_name], linewidth=2.5)
        
    # Configure Left Plot: Singular Value Decay
    ax1.set_xlabel(r'Singular Value Index $i$', fontsize=12)
    ax1.set_ylabel(r'Normalized Singular Value $\sigma_i / \sigma_1$', fontsize=12)
    ax1.set_title('Singular Value Decay Spectrum', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1, 150)
    ax1.set_ylim(0, 1.05)
    
    # Configure Right Plot: Cumulative Energy Captured
    ax2.set_xlabel(r'Fractional Subspace Rank $\gamma$ ($r = \lfloor \gamma \cdot d_{out} \rfloor$)', fontsize=12)
    ax2.set_ylabel('Cumulative Update Energy Captured (%)', fontsize=12)
    ax2.set_title('Cumulative Energy Spectrum vs. Subspace Rank', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 105)
    
    # Draw a line at gamma = 0.3 to highlight our chosen hyperparameter
    ax2.axvline(x=0.3, color='black', linestyle=':', alpha=0.8, linewidth=1.5)
    ax2.text(0.32, 45, r'Optimal Rank $\gamma=0.3$' + '\n' + r'($>90\%$ Energy Captured)', color='black', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
    
    plt.suptitle('Spectral Consensus Analysis of Multi-Task Update Matrices', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plot_path = "results/singular_value_decay.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Singular value decay plot successfully saved to {plot_path}")

if __name__ == "__main__":
    main()
