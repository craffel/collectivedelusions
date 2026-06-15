import numpy as np
import torch

D = 192
K = 4
d_block = 48
C = 10

prototypes = []
for k in range(K):
    W = np.random.randn(C, d_block)
    q, r = np.linalg.qr(W.T)
    prototypes.append(q.T)

# Let's generate an OOD sample block with orth_part + noise
k = 0
v = np.random.randn(d_block)
Phi = prototypes[k].T
proj = Phi @ np.linalg.inv(Phi.T @ Phi) @ Phi.T @ v
orth_part = v - proj
orth_part = orth_part / np.linalg.norm(orth_part)

noise = np.random.randn(d_block) * 0.1
X_block = orth_part + noise

# Let's compute cosine similarity
X_block_norm = X_block / np.linalg.norm(X_block)
protos = prototypes[k]
protos_norm = protos / np.linalg.norm(protos, axis=1, keepdims=True)
sims = protos_norm @ X_block_norm
print("OOD block similarity (max):", sims.max())

# Let's compute with noise scale 0.0
X_block_0 = orth_part
X_block_norm_0 = X_block_0 / np.linalg.norm(X_block_0)
sims_0 = protos_norm @ X_block_norm_0
print("OOD block similarity with zero noise (max):", sims_0.max())
