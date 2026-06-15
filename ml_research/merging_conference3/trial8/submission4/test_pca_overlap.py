import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
num_classes = 10
block_size = D // K  # 48
overlap_size = 12

task_dims = {}
for k in range(K):
    task_dims[k] = list(range(k*block_size, (k+1)*block_size)) + list(range(((k+1)%K)*block_size, ((k+1)%K)*block_size + overlap_size))

# Print task dimensions to verify overlap
for k in range(K):
    print(f"Task {k} dims: {task_dims[k][:5]}...{task_dims[k][-5:]} (total: {len(task_dims[k])})")

class_prototypes = {}
for k in range(K):
    subspace_size = len(task_dims[k])
    U, S, V = torch.svd(torch.randn(subspace_size, num_classes))
    prototypes = torch.zeros(num_classes, D)
    for idx, d_idx in enumerate(task_dims[k]):
        prototypes[:, d_idx] = U.t()[:num_classes, idx]
    class_prototypes[k] = prototypes

print("Prototypes successfully created.")
