import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
num_classes = 10
block_size = D // K  # 48

# Generate class prototypes for each task
class_prototypes = {}
for k in range(K):
    U, S, V = torch.svd(torch.randn(block_size, num_classes))
    prototypes = torch.zeros(num_classes, D)
    prototypes[:, k*block_size : (k+1)*block_size] = U.t()[:num_classes]
    class_prototypes[k] = prototypes

# Shared base layers (Layers 1-13)
W_base = {}
for l in range(1, 14):
    W_base[l] = 0.05 * torch.eye(D)

# Generate calibration set (16 samples per task, 64 total)
noise_levels = [0.01, 0.05, 0.28, 1.35]
calib_x = []
calib_y = []

for k in range(K):
    for i in range(16):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        calib_x.append(x)
        calib_y.append(k)

calib_x = torch.stack(calib_x)  # (64, D)
calib_y = torch.tensor(calib_y)  # (64,)

# Extract Layer 3 features for calibration set
h = calib_x.clone()
for l in range(1, 4):
    h = h + torch.relu(h @ W_base[l])
z_calib = h.clone()  # (64, D)

# Compute centroids and dispersion scales on calibration set
centroids = {}
dispersion = {}
for k in range(K):
    mask = (calib_y == k)
    z_k = z_calib[mask]  # (16, D)
    mu_k = z_k.mean(dim=0)
    centroids[k] = mu_k
    
    # Cosine similarity similarity to centroid
    z_k_norm = z_k / (z_k.norm(dim=1, keepdim=True) + 1e-8)
    mu_k_norm = mu_k / (mu_k.norm() + 1e-8)
    cos_sims = (z_k_norm @ mu_k_norm)
    dispersion[k] = cos_sims.mean().item()

print("Centroids and dispersion computed.")
for k in range(K):
    print(f"Task {k} dispersion scale s_k: {dispersion[k]:.4f}")

# Compute dispersion-calibrated similarity coordinates for calibration set
# u_calib of shape (64, 4)
u_calib = torch.zeros(64, K)
for k in range(K):
    mu_k_norm = centroids[k] / (centroids[k].norm() + 1e-8)
    z_calib_norm = z_calib / (z_calib.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = z_calib_norm @ mu_k_norm
    u_calib[:, k] = cos_sim / dispersion[k]

# Optimize PAC-Bayesian bound to find optimal temperatures
theta = torch.nn.Parameter(torch.log(torch.ones(K) * 0.05))
optimizer = torch.optim.Adam([theta], lr=0.1)
N = 64

for epoch in range(200):
    optimizer.zero_grad()
    tau = torch.exp(theta)
    logits = u_calib / tau
    q = torch.softmax(logits, dim=1)
    
    # Empirical risk (mean expected classification error on correct task)
    risk = 1.0 - q[range(N), calib_y].mean()
    
    # Shannon entropy of the routing policy
    entropy = - (q * torch.log(q + 1e-8)).sum(dim=1).mean()
    
    # KL divergence and PAC-Bayesian bound (aligned with parameter-space L2 norm)
    kl = (theta ** 2).sum() / 2.0
    
    bound = risk + torch.sqrt((kl + np.log(2.0 * np.sqrt(N) / 0.05)) / (2.0 * N))
    
    bound.backward()
    optimizer.step()

opt_tau = torch.exp(theta).detach().numpy()
print("Optimized PAC-Bayesian temperatures:")
for k in range(K):
    print(f"Task {k} (MNIST/F-MNIST/CIFAR/SVHN) optimal temperature: {opt_tau[k]:.6f}")
