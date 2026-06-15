import torch
from simulate import generate_expert_heads, set_seed
import numpy as np

D = 1024
K = 4
d = 48
C = 10

set_seed(42)
W = generate_expert_heads()

# Generate overlapping R_k
R = []
for k in range(K):
    R_k = torch.randn(D, d)
    Q, _ = torch.linalg.qr(R_k)
    R.append(Q)

# Generate one sample for task 0
t = 0
c_b = 5
z_t = W[t][c_b] + 0.05 * torch.randn(d)
z_t = z_t / torch.norm(z_t)

# Global representations
z_global = torch.matmul(R[t], z_t)
for k in range(K):
    if k != t:
        eps_k = torch.randn(d)
        z_k = eps_k / torch.norm(eps_k)
        z_global += torch.matmul(R[k], z_k)

# Compute P_k = R_k R_k^T
P = []
for k in range(K):
    P_k = torch.matmul(R[k], R[k].t())
    P.append(P_k)

print("z_global norm:", torch.norm(z_global).item())
for k in range(K):
    z_proj = torch.matmul(P[k], z_global)
    print(f"Subspace {k} projection norm:", torch.norm(z_proj).item())
