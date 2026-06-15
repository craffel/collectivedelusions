import torch
from simulate import generate_expert_heads, set_seed
import numpy as np

D = 1024
K = 4
d = 48
C = 10

set_seed(42)
W = generate_expert_heads()

# Let's check with orthogonal matrices R
R = []
for k in range(K):
    R_k = torch.zeros(D, d)
    R_k[k*d : (k+1)*d, :] = torch.eye(d)
    R.append(R_k)

# Generate a single test sample for task 0, class 5
t = 0
c_b = 5
z_t = W[t][c_b] + 0.05 * torch.randn(d)
z_t = z_t / torch.norm(z_t)

# Active projected
z_t_global = torch.matmul(R[t], z_t)

# Inactive projected
z_inactive_global = torch.zeros(D)
inactive_z_locals = []
for k in range(K):
    if k != t:
        eps_k = torch.randn(d)
        z_k = eps_k / torch.norm(eps_k)
        inactive_z_locals.append(z_k)
        z_inactive_global += torch.matmul(R[k], z_k)
    else:
        inactive_z_locals.append(z_t) # just placeholder

z_global = z_t_global + z_inactive_global

# Now, U joint in the orthogonal case should be exactly R stacked together:
U_joint = torch.cat(R, dim=1) # [D, K*d] = [1024, 192]
U_joint_pinv = torch.pinverse(U_joint) # [192, 1024]

z_reconstructed = torch.matmul(U_joint_pinv, z_global)

print("Original z_t norm:", torch.norm(z_t).item())
print("Reconstructed first block norm:", torch.norm(z_reconstructed[0:48]).item())
print("Reconstruction error (active task):", torch.norm(z_reconstructed[0:48] - z_t).item())
print("Reconstruction error (total):")
for k in range(K):
    if k == t:
        print(f"Block {k} (active) error:", torch.norm(z_reconstructed[k*d:(k+1)*d] - z_t).item())
    else:
        # Inactive clean
        print(f"Block {k} error:", torch.norm(z_reconstructed[k*d:(k+1)*d] - R[k].t() @ z_global).item())
