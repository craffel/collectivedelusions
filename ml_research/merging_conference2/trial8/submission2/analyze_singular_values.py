import torch
import os
import numpy as np

checkpoint_dir = "checkpoints"
progenitor_path = os.path.join(checkpoint_dir, "progenitor.pt")
expert_paths = {
    "MNIST": os.path.join(checkpoint_dir, "expert_mnist.pt"),
    "Fashion-MNIST": os.path.join(checkpoint_dir, "expert_fashion.pt"),
    "CIFAR-10": os.path.join(checkpoint_dir, "expert_cifar.pt"),
}

print("Loading checkpoints...")
progenitor_state = torch.load(progenitor_path, map_location="cpu")
expert_states = {k: torch.load(v, map_location="cpu") for k, v in expert_paths.items()}

# We select a few representative convolutional layers across different depths of ResNet-18
target_layers = [
    "layer1.0.conv1.weight",
    "layer2.0.conv1.weight",
    "layer3.0.conv1.weight",
    "layer4.0.conv1.weight",
]

print("\n--- Singular Value Decay Analysis ---")
for layer in target_layers:
    if layer not in progenitor_state:
        print(f"Layer {layer} not found in progenitor state.")
        continue
    
    W_init = progenitor_state[layer].float()
    orig_shape = W_init.shape
    R = orig_shape[0]
    D = W_init.numel() // R
    min_dim = min(R, D)
    
    print(f"\nLayer: {layer} | Shape: {orig_shape} | Matrix Dim: {R}x{D} (min_dim={min_dim})")
    
    for task, state in expert_states.items():
        W_expert = state[layer].float()
        T = W_expert - W_init
        
        # Flatten to 2D
        T_2d = T.view(R, D)
        
        # Compute SVD
        _, S, _ = torch.linalg.svd(T_2d, full_matrices=False)
        S_np = S.numpy()
        
        # Calculate total energy (Frobenius norm squared)
        total_energy = np.sum(S_np**2)
        
        # Cumulative energy fractions
        cum_energy = np.cumsum(S_np**2) / total_energy
        
        # Find index for 50%, 80%, 90%, 95% of energy
        idx_50 = np.where(cum_energy >= 0.50)[0][0] + 1
        idx_80 = np.where(cum_energy >= 0.80)[0][0] + 1
        idx_90 = np.where(cum_energy >= 0.90)[0][0] + 1
        idx_95 = np.where(cum_energy >= 0.95)[0][0] + 1
        
        # Energy at top 10% and top 20% singular vectors
        k_10 = max(1, int(0.10 * min_dim))
        k_20 = max(1, int(0.20 * min_dim))
        energy_10 = cum_energy[k_10 - 1]
        energy_20 = cum_energy[k_20 - 1]
        
        print(f"  Task: {task:15s} | Frobenius Norm: {np.sqrt(total_energy):.4f}")
        print(f"    Top 10% singular vectors (k={k_10}/{min_dim}) capture {energy_10*100:5.2f}% of energy.")
        print(f"    Top 20% singular vectors (k={k_20}/{min_dim}) capture {energy_20*100:5.2f}% of energy.")
        print(f"    k to capture 50% energy: {idx_50}/{min_dim} ({idx_50/min_dim*100:.1f}%)")
        print(f"    k to capture 80% energy: {idx_80}/{min_dim} ({idx_80/min_dim*100:.1f}%)")
        print(f"    k to capture 90% energy: {idx_90}/{min_dim} ({idx_90/min_dim*100:.1f}%)")
        print(f"    k to capture 95% energy: {idx_95}/{min_dim} ({idx_95/min_dim*100:.1f}%)")
        print(f"    Top 5 singular values: {S_np[:5]}")
