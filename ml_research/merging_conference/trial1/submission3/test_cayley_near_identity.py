import torch
import scipy.linalg
import numpy as np

d = 128
I = torch.eye(d)

# Case 1: R is near Identity (rotation of small angle)
A = torch.eye(d) + 0.05 * torch.randn(d, d)
U, S, V = torch.linalg.svd(A)
R = U @ V.T

print("=== Cayley Near Identity ===")
try:
    Q = torch.linalg.solve(R + I, R - I)
    print("Computed Q. Skew-symmetric check (norm of Q + Q^T):", torch.norm(Q + Q.T).item())
    
    R_rec = torch.linalg.solve(I - Q, I + Q)
    print("Recovered R. Orthogonality check (norm of R_rec.T @ R_rec - I):", torch.norm(R_rec.T @ R_rec - I).item())
    print("Reconstruction error (norm of R_rec - R):", torch.norm(R_rec - R).item())
except Exception as e:
    print("Cayley error:", e)

print("\n=== SciPy Matrix Log/Exp (General case) ===")
try:
    # Use random orthogonal matrix (not necessarily near identity)
    A_rand = torch.randn(d, d)
    U_rand, S_rand, V_rand = torch.linalg.svd(A_rand)
    R_rand = U_rand @ V_rand.T
    
    # SciPy logm
    R_np = R_rand.cpu().numpy()
    Q_np = scipy.linalg.logm(R_np)
    # Check skew-symmetry
    print("SciPy Q skew-symmetric check (norm of Q + Q^T):", np.linalg.norm(Q_np + Q_np.T))
    
    # SciPy expm
    R_rec_np = scipy.linalg.expm(Q_np)
    print("SciPy reconstruction error:", np.linalg.norm(R_rec_np - R_np))
except Exception as e:
    print("SciPy error:", e)
