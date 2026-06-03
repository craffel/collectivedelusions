import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os
from merge import merge_models

# Ensure we have the same setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_layer_spectral_decay():
    # 1. Load pretrained base model
    base_model = models.resnet18()
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_state = base_model.state_dict()
    
    # 2. Load task checkpoints
    task_names = ['cifar10', 'svhn', 'fashionmnist']
    task_states = []
    for name in task_names:
        chk_path = f"checkpoint_{name}.pt"
        if not os.path.exists(chk_path):
            raise FileNotFoundError(f"Checkpoint not found: {chk_path}")
        state = torch.load(chk_path, map_location='cpu')
        task_states.append(state)
        
    weights = [1.0 / len(task_names)] * len(task_names)
    
    # Merge using all three methods
    methods = ['task_arithmetic', 'isotropic', 'wsa']
    merged_states = {}
    for method in methods:
        print(f"Merging with {method}...")
        merged_states[method] = merge_models(
            base_state, task_states, weights, method, scaling=1.0, eps=1e-8, normalize_barycenter=False
        )
        
    # Choose a large convolutional layer to analyze
    target_key = 'layer4.1.conv2.weight'
    print(f"\nAnalyzing spectral decay of layer: {target_key}")
    
    # Compute base update and merged updates
    W0 = base_state[target_key].float()
    orig_shape = W0.shape
    d1 = orig_shape[0]
    d2 = W0.numel() // d1
    W0_2d = W0.view(d1, d2)
    
    # Compute the singular values for the task updates
    plt.figure(figsize=(8, 5))
    
    # Plot individual task updates for context
    for name, t_state in zip(task_names, task_states):
        W_task = t_state[target_key].float().view(d1, d2)
        T_task = W_task - W0_2d
        _, S_task, _ = torch.linalg.svd(T_task, full_matrices=False)
        plt.plot(S_task.cpu().numpy(), label=f'Task: {name.upper()}', linestyle='--', alpha=0.6)
        print(f"  {name.upper()} update trace energy: {torch.sum(S_task**2).item():.4f}")
        
    # Plot merged updates
    colors = {'task_arithmetic': 'red', 'isotropic': 'blue', 'wsa': 'green'}
    labels = {'task_arithmetic': 'Task Arithmetic', 'isotropic': 'Isotropic Merging', 'wsa': 'WSA (Ours)'}
    
    for method in methods:
        W_merged = merged_states[method][target_key].float().view(d1, d2)
        T_merged = W_merged - W0_2d
        _, S_merged, _ = torch.linalg.svd(T_merged, full_matrices=False)
        plt.plot(S_merged.cpu().numpy(), color=colors[method], linewidth=2.5, label=labels[method])
        trace_energy = torch.sum(S_merged**2).item()
        print(f"  Merged {labels[method]} update trace energy: {trace_energy:.4f}")
        
    plt.title('Singular Value Spectrum (Spectral Decay) of Merged Layer Weights', fontsize=12)
    plt.xlabel('Singular Value Index', fontsize=10)
    plt.ylabel('Singular Value Magnitude', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_path = 'spectral_decay.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nSpectral decay plot saved to {plot_path}")

if __name__ == '__main__':
    analyze_layer_spectral_decay()
