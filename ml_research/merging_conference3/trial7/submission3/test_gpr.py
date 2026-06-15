import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Global variables
L = 14       # number of layer groups
D = 192      # representation dimension
K = 4        # number of tasks
d_block = 48 # dimension of each block
C = 10       # number of classes per task

# Setup class-specific prototypes
prototypes = []
for k in range(K):
    W = np.random.randn(C, d_block)
    q, r = np.linalg.qr(W.T)
    prototypes.append(q.T)

noise_scales = [0.01, 0.18, 0.25, 0.85]
bg_noise_scale = 0.5

def generate_data_coupled(num_samples_per_task, noise_scales, prototypes, bg_noise_scale=0.5, coupling=0.0):
    X_list = []
    y_list = []
    task_labels_list = []
    for k in range(K):
        task_noise = noise_scales[k]
        task_protos = prototypes[k]
        for _ in range(num_samples_per_task):
            class_idx = np.random.randint(0, C)
            z = np.zeros(D)
            active_feature = task_protos[class_idx]
            z[k*d_block:(k+1)*d_block] = active_feature + np.random.randn(d_block) * task_noise
            for j in range(K):
                if j != k:
                    leak = coupling * active_feature
                    z[j*d_block:(j+1)*d_block] = leak + np.random.randn(d_block) * bg_noise_scale
            X_list.append(z)
            y_list.append(k * C + class_idx)
            task_labels_list.append(k)
    return torch.tensor(np.array(X_list), dtype=torch.float32), \
           torch.tensor(np.array(y_list), dtype=torch.long), \
           torch.tensor(np.array(task_labels_list), dtype=torch.long)

X_train_expert, y_train_expert, task_train_expert = generate_data_coupled(1000, noise_scales, prototypes, bg_noise_scale, coupling=0.0)
expert_heads = []
for k in range(K):
    mask = (task_train_expert == k)
    X_k = X_train_expert[mask][:, k*d_block:(k+1)*d_block]
    y_k = y_train_expert[mask] % C
    head = nn.Linear(d_block, C, bias=False)
    optimizer = optim.AdamW(head.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset_k = TensorDataset(X_k, y_k)
    loader_k = DataLoader(dataset_k, batch_size=64, shuffle=True)
    for epoch in range(10): # short training
        for inputs, targets in loader_k:
            optimizer.zero_grad()
            outputs = head(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    expert_heads.append(head)

def project_subspace_coords(X, expert_heads, prototypes):
    B_size = X.shape[0]
    u = torch.zeros(B_size, K)
    for k in range(K):
        X_block = X[:, k*d_block:(k+1)*d_block]
        X_block_norm = X_block / (torch.norm(X_block, p=2, dim=1, keepdim=True) + 1e-8)
        protos = torch.tensor(prototypes[k], dtype=torch.float32)
        protos_norm = protos / (torch.norm(protos, p=2, dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(X_block_norm, protos_norm.T)
        u[:, k] = sims.max(dim=1)[0]
    cal_factor = np.sqrt(2.0 * np.log(10) / 48)
    u_cal = u / cal_factor
    norm = torch.norm(u_cal, p=2, dim=1, keepdim=True)
    psi = torch.zeros_like(u_cal)
    mask = (norm.squeeze(1) > 1e-5)
    psi[mask] = u_cal[mask] / norm[mask]
    return psi

coupling_val = 0.50
X_cal, y_cal, task_cal = generate_data_coupled(16, noise_scales, prototypes, bg_noise_scale, coupling=coupling_val)
psi_cal = project_subspace_coords(X_cal, expert_heads, prototypes)

# Let's print some calibration points
print("Calibration points (first 5):")
print(psi_cal[:5])

# Let's check GPR for RBF kernel
lengthscale = 0.5
sq_dist = torch.cdist(psi_cal, psi_cal, p=2) ** 2
K_gram = torch.exp(-sq_dist / (2.0 * (lengthscale ** 2)))
print("K_gram shape:", K_gram.shape)
print("K_gram eigenvalues:", torch.linalg.eigvalsh(K_gram))

M = torch.inverse(K_gram + (1e-2 ** 2) * torch.eye(64))

# Let's pick an OOD point psi_star
psi_star = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

# Compute k_star
sq_dist_star = torch.cdist(psi_star, psi_cal, p=2) ** 2
k_star = torch.exp(-sq_dist_star / (2.0 * (lengthscale ** 2)))
print("Max correlation with calibration:", k_star.max().item())
print("Mean correlation with calibration:", k_star.mean().item())

k_star_M = torch.matmul(k_star, M)
post_var = torch.clamp(1.0 - (k_star_M * k_star).sum(dim=1), min=0.0)
print("Posterior variance for psi_star [0.5, 0.5, 0.5, 0.5]:", post_var.item())
