import torch
import numpy as np
import math

def compute_entropy(S):
    # S is a tensor of singular values
    # Normalize S^2 to represent a probability distribution
    S_sq = S ** 2
    sum_sq = torch.sum(S_sq)
    if sum_sq == 0:
        return 0.0
    p = S_sq / sum_sq
    # Avoid log(0)
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log(p)).item()
    return entropy

# Load checkpoints
experts = []
for t in ['A', 'B', 'C']:
    checkpoint = torch.load(f"expert_{t}.pt", map_location='cpu')
    experts.append(checkpoint)

# We also need the base model weights to compute the task vectors. Let's find W_0.
# Since we don't have base_model.pt saved separately, let's load SimpleCNN from train_and_merge.py
from train_and_merge import SimpleCNN
base_model = SimpleCNN(num_classes=10)
# SimpleCNN starts with random weights. But wait, train_and_merge.py trains experts starting from base_model.
# Since base_model was randomly initialized, let's check if we can reconstruct the base model.
# Ah, the seed in train_and_merge.py is 42. So if we initialize SimpleCNN, it will have the exact same random weights as W_0!
torch.manual_seed(42)
np.random.seed(42)
base_model = SimpleCNN(num_classes=10)
base_state = base_model.state_dict()

# Target layer
key = "classifier.3.weight"
W_0 = base_state[key]

# Task vectors
task_vectors = [expert[key] - W_0 for expert in experts]
joint_update = torch.stack(task_vectors).sum(dim=0)

# 2D representation
orig_shape = joint_update.shape
flat_update = joint_update.view(orig_shape[0], -1)

U, S_orig, Vh = torch.linalg.svd(flat_update, full_matrices=False)

# Let's define the statistical functions
def apply_fd_cap(S, T, mu):
    # energies are -S
    energies = -S
    exponent = (energies - mu) / T
    exponent = torch.clamp(exponent, -50.0, 50.0)
    return 1.0 / (torch.exp(exponent) + 1.0)

def apply_fd_suppress(S, T, mu):
    exponent = (S - mu) / T
    exponent = torch.clamp(exponent, -50.0, 50.0)
    return 1.0 / (torch.exp(exponent) + 1.0)

def apply_be(S, T, mu):
    energies = -S
    exponent = (energies - mu) / T
    exponent = torch.clamp(exponent, 1e-5, 50.0)
    return 1.0 / (torch.exp(exponent) - 1.0)

# Compute entropies
print("Method | Temperature | Entropy")
print("-" * 35)

# 1. Original (Task Arithmetic)
h_orig = compute_entropy(S_orig)
print(f"Task Arithmetic | - | {h_orig:.4f}")

# 2. Isotropic
S_iso = torch.ones_like(S_orig)
h_iso = compute_entropy(S_iso)
print(f"Isotropic | - | {h_iso:.4f}")

# 3. FD-Cap
for T in [0.1, 1.0, 5.0, 10.0]:
    energies = -S_orig
    mu = energies.mean().item()
    S_new = apply_fd_cap(S_orig, T, mu)
    h = compute_entropy(S_new)
    print(f"FD-Cap | T={T} | {h:.4f}")

# 4. FD-Suppress
for T in [0.1, 1.0, 5.0, 10.0]:
    mu = S_orig.mean().item()
    S_new = apply_fd_suppress(S_orig, T, mu)
    h = compute_entropy(S_new)
    print(f"FD-Suppress | T={T} | {h:.4f}")

# 5. Bose-Einstein
for T in [0.1, 1.0, 5.0, 10.0]:
    energies = -S_orig
    mu = energies.min().item() - 1.0
    S_new = apply_be(S_orig, T, mu)
    h = compute_entropy(S_new)
    print(f"Bose-Einstein | T={T} | {h:.4f}")
