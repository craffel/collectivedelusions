import torch

# Generate a random orthogonal matrix R
d = 128
A = torch.randn(d, d)
U, S, V = torch.linalg.svd(A)
R = U @ V.T  # This is orthogonal: R^T R = I

I = torch.eye(d)
# Inverse Cayley transform: Q = (R - I)(R + I)^-1
# Wait, let's make sure it is stable
try:
    Q = torch.linalg.solve(R + I, R - I)
    print("Computed Q. Skew-symmetric check (norm of Q + Q^T):", torch.norm(Q + Q.T).item())
    
    # Cayley transform back: R_recovered = (I - Q)^-1 (I + Q)
    # Let's verify
    R_rec = torch.linalg.solve(I - Q, I + Q)
    print("Recovered R. Orthogonality check (norm of R^T R - I):", torch.norm(R_rec.T @ R_rec - I).item())
    print("Reconstruction error (norm of R_rec - R):", torch.norm(R_rec - R).item())
except Exception as e:
    print("Error:", e)
