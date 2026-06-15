import torch
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
L = 14
num_classes = 10
block_size = D // K  # 48

# Generate class prototypes for each task
# For task k, prototypes lie in dimensions [48*k ... 48*(k+1)-1]
class_prototypes = {}
for k in range(K):
    # Generate 10 random orthogonal-like vectors of size 48
    U, S, V = torch.svd(torch.randn(block_size, num_classes))
    # Pad with zeros to size 192
    prototypes = torch.zeros(num_classes, D)
    prototypes[:, k*block_size : (k+1)*block_size] = U.t()[:num_classes]
    class_prototypes[k] = prototypes

# Classification heads for each expert
W_head = {}
for k in range(K):
    # Head maps 192-dim representation to 10 classes, using only task k's subspace
    head = torch.zeros(D, num_classes)
    head[k*block_size : (k+1)*block_size, :] = class_prototypes[k][:, k*block_size : (k+1)*block_size].t()
    W_head[k] = head

# Print verification
print("Prototypes and heads generated.")
print("MNIST (Task 0) prototype shape:", class_prototypes[0].shape)
print("FashionMNIST (Task 1) prototype shape:", class_prototypes[1].shape)
