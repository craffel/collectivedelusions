import torch
import torch.nn as nn
import numpy as np
from simulate import generate_expert_heads, generate_data, set_seed, train_router, compute_cghr_coefficients

D = 192
K = 4
d = D // K
C = 10
optimal_metric = "max"
optimal_th = 0.85

set_seed(42)
W = generate_expert_heads()
train_z, train_tasks, train_classes = generate_data(W, 16, [42]) # N=64
test_z, test_tasks, test_classes = generate_data(W, 250, [42]) # 1000 test samples

router = train_router(train_z, train_tasks, wd=0.1, epochs=150, lr=0.01)

# Let's inspect what happens to alpha and k_star for the first batch
test_z_b = test_z[:256]
test_tasks_b = test_tasks[:256]
test_classes_b = test_classes[:256]

alpha, u_prime = compute_cghr_coefficients(test_z_b, W, router, optimal_metric, optimal_th)
k_star = torch.argmax(u_prime, dim=1)

# Check task classification accuracy of PFSR (u_prime)
correct_tasks = (k_star == test_tasks_b).sum().item()
print(f"PFSR Task Classification Accuracy on this batch: {correct_tasks / 256 * 100:.2f}%")

# Now let's see what happens to alpha_avg_g under 0% error
for g in range(K):
    mask = (k_star == g)
    if mask.any():
        alpha_avg = torch.mean(alpha[mask], dim=0)
        print(f"Group {g} (Size {mask.sum().item()}): alpha_avg = {alpha_avg.numpy()}")

# Let's see what happens to alpha_avg_g under 75% error
np.random.seed(42)
corrupted_k_star = k_star.clone()
for b in range(256):
    if np.random.rand() < 0.75:
        other_tasks = [t for t in range(K) if t != k_star[b].item()]
        corrupted_k_star[b] = np.random.choice(other_tasks)

print("\nUnder 75% Error:")
for g in range(K):
    mask = (corrupted_k_star == g)
    if mask.any():
        alpha_avg = torch.mean(alpha[mask], dim=0)
        print(f"Group {g} (Size {mask.sum().item()}): alpha_avg = {alpha_avg.numpy()}")
