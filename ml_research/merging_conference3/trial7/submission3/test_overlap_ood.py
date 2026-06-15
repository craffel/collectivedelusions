import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Global variables
L = 14       # number of layer groups
D = 192      # representation dimension
K = 4        # number of tasks
d_block = 48 # dimension of each block
C = 10       # number of classes per task
N_cal = 64   # calibration dataset size (16 per task)
N_test = 1000 # test dataset size (250 per task)

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
            
            # Fill the k-th block with prototype + noise
            active_feature = task_protos[class_idx]
            z[k*d_block:(k+1)*d_block] = active_feature + np.random.randn(d_block) * task_noise
            
            # Fill other blocks with background noise + coupled features
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

print("Training specialized experts...")
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
    for epoch in range(50):
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

class GPDRRouter:
    def __init__(self, psi_train, y_train, K, sigma_f=1.0, lengthscale=1.0, sigma_n=1e-2, kernel_type='rbf'):
        self.psi_train = psi_train
        self.K = K
        self.sigma_f = sigma_f
        self.lengthscale = lengthscale
        self.sigma_n = sigma_n
        self.kernel_type = kernel_type
        self.N = psi_train.shape[0]
        
        self.Y_targets = torch.zeros(self.N, K)
        for i in range(self.N):
            self.Y_targets[i, y_train[i]] = 1.0
            
        self.prior_mean = 1.0 / K
        self.K_gram = self.kernel(self.psi_train, self.psi_train)
        self.M = torch.inverse(self.K_gram + (self.sigma_n ** 2) * torch.eye(self.N))
        self.W_gp = torch.matmul(self.M, self.Y_targets - self.prior_mean)
        
    def kernel(self, x1, x2):
        if self.kernel_type == 'rbf':
            sq_dist = torch.cdist(x1, x2, p=2) ** 2
            return (self.sigma_f ** 2) * torch.exp(-sq_dist / (2.0 * (self.lengthscale ** 2)))
        elif self.kernel_type == 'cosine':
            norm1 = torch.norm(x1, p=2, dim=1, keepdim=True)
            norm2 = torch.norm(x2, p=2, dim=1, keepdim=True)
            sims = torch.matmul(x1, x2.T) / (torch.matmul(norm1, norm2.T) + 1e-12)
            mask1 = (norm1 < 1e-5)
            mask2 = (norm2 < 1e-5)
            zero_mask = mask1 | mask2.T
            dists = torch.cdist(x1, x2, p=2)
            is_same = (dists < 1e-5)
            final_sims = torch.zeros_like(sims)
            final_sims[~zero_mask] = sims[~zero_mask]
            final_sims[zero_mask & is_same] = 1.0
            p = 3
            return (self.sigma_f ** 2) * (final_sims ** p)
        
    def forward(self, psi_test, theta_ood=0.9):
        B_size = psi_test.shape[0]
        k_star = self.kernel(psi_test, self.psi_train)
        mu = self.prior_mean + torch.matmul(k_star, self.W_gp)
        k_star_M = torch.matmul(k_star, self.M)
        # Compute posterior variance with a non-negative clamping safeguard to prevent numerical instabilities
        post_var = torch.clamp((self.sigma_f ** 2) - (k_star_M * k_star).sum(dim=1), min=0.0)
        return post_var

def get_min_cosine_dist(psi_test, psi_cal):
    sims = torch.matmul(psi_test, psi_cal.T)
    return (1.0 - sims).min(dim=1)[0]

def generate_overlapping_ood(num_samples, beta, prototypes, noise_scale=0.1):
    X_list = []
    for _ in range(num_samples):
        z = np.zeros(D)
        for k in range(K):
            v = np.random.randn(d_block)
            Phi = prototypes[k].T
            proj = Phi @ np.linalg.inv(Phi.T @ Phi) @ Phi.T @ v
            orth_part = v - proj
            orth_part = orth_part / (np.linalg.norm(orth_part) + 1e-8)
            
            class_idx = np.random.randint(0, C)
            proto_part = prototypes[k][class_idx]
            
            mixed_block = beta * proto_part + (1.0 - beta) * orth_part
            z[k*d_block:(k+1)*d_block] = mixed_block + np.random.randn(d_block) * noise_scale
        X_list.append(z)
    return torch.tensor(np.array(X_list), dtype=torch.float32)

coupling_val = 0.50
X_cal_ood, y_cal_ood, task_cal_ood = generate_data_coupled(16, noise_scales, prototypes, bg_noise_scale, coupling=coupling_val)
psi_cal_ood = project_subspace_coords(X_cal_ood, expert_heads, prototypes)

X_test_id, y_test_id, task_test_id = generate_data_coupled(250, noise_scales, prototypes, bg_noise_scale, coupling=coupling_val)
psi_test_id = project_subspace_coords(X_test_id, expert_heads, prototypes)

router_gp_rbf = GPDRRouter(psi_cal_ood, task_cal_ood, K, sigma_f=1.0, lengthscale=0.5, sigma_n=0.01, kernel_type='rbf')
vars_id_rbf = router_gp_rbf.forward(psi_test_id)

print(f"ID psi shape: {psi_test_id.shape}")
print(f"ID psi sample[0]: {psi_test_id[0]}")
print(f"ID GP-RBF var: mean={vars_id_rbf.mean().item():.4f}, std={vars_id_rbf.std().item():.4f}, min={vars_id_rbf.min().item():.4f}, max={vars_id_rbf.max().item():.4f}")

X_test_ood = generate_overlapping_ood(250, 0.0, prototypes)
psi_test_ood = project_subspace_coords(X_test_ood, expert_heads, prototypes)
vars_ood_rbf = router_gp_rbf.forward(psi_test_ood)

print(f"OOD psi (beta=0.0) shape: {psi_test_ood.shape}")
print(f"OOD psi sample[0]: {psi_test_ood[0]}")
print(f"OOD GP-RBF var: mean={vars_ood_rbf.mean().item():.4f}, std={vars_ood_rbf.std().item():.4f}, min={vars_ood_rbf.min().item():.4f}, max={vars_ood_rbf.max().item():.4f}")
