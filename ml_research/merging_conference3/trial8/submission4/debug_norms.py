import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
num_classes = 10
block_size = 48
overlap_size = 0

# Set up orthogonal dimensions
task_dims = {}
for k in range(K):
    task_dims[k] = list(range(k*block_size, (k+1)*block_size))

class_prototypes = {}
for k in range(K):
    subspace_size = len(task_dims[k])
    U, S, V = torch.svd(torch.randn(subspace_size, num_classes))
    prototypes = torch.zeros(num_classes, D)
    for idx, d_idx in enumerate(task_dims[k]):
        prototypes[:, d_idx] = U.t()[:num_classes, idx]
    class_prototypes[k] = prototypes

# Noise levels
noise_levels = [0.01, 0.05, 0.28, 1.35]

# Generate calibration samples
calib_x = []
calib_y = []
calib_class_y = []
for k in range(K):
    for i in range(16):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        calib_x.append(x)
        calib_y.append(k)
        calib_class_y.append(c)
calib_x = torch.stack(calib_x)
calib_y = torch.tensor(calib_y)
calib_class_y = torch.tensor(calib_class_y)

# Extract Layer 3 features
W_base = {}
for l in range(1, 14):
    W_base[l] = 0.05 * torch.eye(D)

h_calib = calib_x.clone()
for l in range(1, 4):
    h_calib = h_calib + torch.relu(h_calib @ W_base[l])
z_calib = h_calib.clone()

# SVD on raw calibration samples
V_pca_raw = {}
for k in range(K):
    mask = (calib_y == k)
    z_k = z_calib[mask]
    U_k, S_k, V_k = torch.svd(z_k)
    V_pca_raw[k] = V_k[:, :16]

# SVD on class centroids
V_pca_cent = {}
for k in range(K):
    class_centroids = []
    for c in range(num_classes):
        mask = (calib_y == k) & (calib_class_y == c)
        if mask.sum() > 0:
            class_centroids.append(z_calib[mask].mean(dim=0))
        else:
            class_centroids.append(torch.zeros(D))
    class_centroids = torch.stack(class_centroids)
    U_k, S_k, V_k = torch.svd(class_centroids)
    S_weights = S_k[:10] / (S_k[0] + 1e-8)
    V_pca_cent[k] = V_k[:, :10] * S_weights.unsqueeze(0)

# Print norms for a few samples
print("--- COMPARISON OF NORMS (ORTHOGONAL SETTING) ---")
for task_id in [0, 3]: # MNIST and SVHN
    # Select first sample of this task
    sample_idx = task_id * 16
    z_sample = z_calib[sample_idx]
    
    # 1. Block norms
    block_norms = [z_sample[b*block_size : (b+1)*block_size].norm().item() for b in range(K)]
    
    # 2. Raw PCA norms
    pca_raw_norms = [(z_sample @ V_pca_raw[b]).norm().item() for b in range(K)]
    
    # 3. Centroid PCA norms
    pca_cent_norms = [(z_sample @ V_pca_cent[b]).norm().item() for b in range(K)]
    
    print(f"\nTask {task_id} sample (class {calib_class_y[sample_idx].item()}):")
    print(f"  Block Norms:         {block_norms}")
    print(f"  Raw PCA Norms:       {pca_raw_norms}")
    print(f"  Centroid PCA Norms:  {pca_cent_norms}")
