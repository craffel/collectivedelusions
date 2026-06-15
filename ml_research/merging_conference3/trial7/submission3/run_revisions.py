import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time

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

# 1. Dataset Generation under Coupled representations (Critical Flaw 1)
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
            
            # Fill other blocks with background noise + coupled features (Critical Flaw 1)
            for j in range(K):
                if j != k:
                    # Coupled feature leak representing feature shared across tasks
                    leak = coupling * active_feature
                    z[j*d_block:(j+1)*d_block] = leak + np.random.randn(d_block) * bg_noise_scale
            
            X_list.append(z)
            y_list.append(k * C + class_idx)
            task_labels_list.append(k)
            
    return torch.tensor(np.array(X_list), dtype=torch.float32), \
           torch.tensor(np.array(y_list), dtype=torch.long), \
           torch.tensor(np.array(task_labels_list), dtype=torch.long)

# Train specialized experts (same as run_experiments.py to remain consistent)
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

W_experts_joint = torch.zeros(K * C, D)
for k in range(K):
    W_k = expert_heads[k].weight.data
    W_experts_joint[k*C:(k+1)*C, k*d_block:(k+1)*d_block] = W_k

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

def compute_logits(X, alpha, W_experts_joint):
    logits = torch.zeros(X.shape[0], K * C)
    for k in range(K):
        outputs_k = torch.matmul(X[:, k*d_block:(k+1)*d_block], expert_heads[k].weight.data.T)
        logits[:, k*C:(k+1)*C] = outputs_k * alpha[:, k:k+1]
    return logits

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
            # Cosine/Inner-Product Kernel: k(x1, x2) = sigma_f^2 * (x1 . x2)^p
            # We assume x1 and x2 are batches of shapes (B1, d) and (B2, d)
            # Compute norms to handle any mapping of OOD to origin
            norm1 = torch.norm(x1, p=2, dim=1, keepdim=True) # (B1, 1)
            norm2 = torch.norm(x2, p=2, dim=1, keepdim=True) # (B2, 1)
            
            sims = torch.matmul(x1, x2.T) / (torch.matmul(norm1, norm2.T) + 1e-12)
            
            # Mask out cases where either norm is close to zero (e.g. OOD mapped to origin)
            mask1 = (norm1 < 1e-5) # (B1, 1)
            mask2 = (norm2 < 1e-5) # (B2, 1)
            zero_mask = mask1 | mask2.T # (B1, B2)
            
            # Where zero_mask is True, we check if the inputs are identical (for variance k(x_star, x_star))
            dists = torch.cdist(x1, x2, p=2) # (B1, B2)
            is_same = (dists < 1e-5)
            
            final_sims = torch.zeros_like(sims)
            final_sims[~zero_mask] = sims[~zero_mask]
            final_sims[zero_mask & is_same] = 1.0
            
            p = 3
            return (self.sigma_f ** 2) * (final_sims ** p)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
    def forward(self, psi_test, theta_ood=0.9):
        B_size = psi_test.shape[0]
        k_star = self.kernel(psi_test, self.psi_train)
        mu = self.prior_mean + torch.matmul(k_star, self.W_gp)
        
        k_star_M = torch.matmul(k_star, self.M)
        # Compute posterior variance with a non-negative clamping safeguard to prevent numerical instabilities
        post_var = torch.clamp((self.sigma_f ** 2) - (k_star_M * k_star).sum(dim=1), min=0.0)
        
        alpha = torch.zeros(B_size, self.K)
        for b in range(B_size):
            if post_var[b] > theta_ood:
                alpha[b] = torch.ones(self.K) * self.prior_mean
            else:
                alpha[b] = mu[b]
                
        alpha = torch.clamp(alpha, min=1e-5, max=1.0)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha, post_var

def pfsr_routing(X, expert_heads, prototypes):
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
    alpha = torch.softmax(u_cal / 0.001, dim=1)
    return alpha


# =====================================================================
# EXPERIMENT 1: Coupled Representations (Critical Flaw 1 Revisions)
# =====================================================================
print("\n=== EXPERIMENT 1: Coupled Representations ===")
coupling_factors = [0.0, 0.25, 0.50, 0.75]
results_coupled = {
    'Uniform': [],
    'PFSR': [],
    'GP-DR (Ours)': []
}

for gamma in coupling_factors:
    print(f"Evaluating representation coupling level: gamma = {gamma:.2f}")
    # Generate coupled test set
    X_test_c, y_test_c, task_test_c = generate_data_coupled(250, noise_scales, prototypes, bg_noise_scale, coupling=gamma)
    psi_test_c = project_subspace_coords(X_test_c, expert_heads, prototypes)
    
    # Static Uniform
    alpha_unif = torch.ones(X_test_c.shape[0], K) / K
    logits_unif = compute_logits(X_test_c, alpha_unif, W_experts_joint)
    acc_unif = (logits_unif.argmax(dim=1) == y_test_c).float().mean().item() * 100.0
    results_coupled['Uniform'].append(acc_unif)
    
    # PFSR
    alpha_pf = pfsr_routing(X_test_c, expert_heads, prototypes)
    logits_pf = compute_logits(X_test_c, alpha_pf, W_experts_joint)
    acc_pf = (logits_pf.argmax(dim=1) == y_test_c).float().mean().item() * 100.0
    results_coupled['PFSR'].append(acc_pf)
    
    # GP-DR (Ours)
    # Calibrate on independent calibration set (same as training)
    X_cal_c, y_cal_c, task_cal_c = generate_data_coupled(16, noise_scales, prototypes, bg_noise_scale, coupling=0.0)
    psi_cal_c = project_subspace_coords(X_cal_c, expert_heads, prototypes)
    router_gp_c = GPDRRouter(psi_cal_c, task_cal_c, K, sigma_f=1.0, lengthscale=0.5, sigma_n=0.01)
    alpha_gp, _ = router_gp_c.forward(psi_test_c, theta_ood=0.90)
    logits_gp = compute_logits(X_test_c, alpha_gp, W_experts_joint)
    acc_gp = (logits_gp.argmax(dim=1) == y_test_c).float().mean().item() * 100.0
    results_coupled['GP-DR (Ours)'].append(acc_gp)
    
    print(f"  Uniform: {acc_unif:.2f}% | PFSR: {acc_pf:.2f}% | GP-DR: {acc_gp:.2f}%")


# =====================================================================
# EXPERIMENT 2: Comparison against Distance-Based OOD Baselines (Critical Flaw 2)
# =====================================================================
print("\n=== EXPERIMENT 2: Distance-Based OOD Baselines ===")

# Distance metrics functions
def get_min_euclidean_dist(psi_test, psi_cal):
    # Pairwise L2 distance: shape (B_test, N_cal)
    dists = torch.cdist(psi_test, psi_cal, p=2)
    return dists.min(dim=1)[0]

def get_knn_dist(psi_test, psi_cal, k=5):
    dists = torch.cdist(psi_test, psi_cal, p=2)
    topk_dists, _ = torch.topk(dists, k, dim=1, largest=False)
    return topk_dists.mean(dim=1)

def get_min_cosine_dist(psi_test, psi_cal):
    # psi_test and psi_cal are normalized, so cosine sim is just matmul
    # cos_dist = 1.0 - cos_sim
    sims = torch.matmul(psi_test, psi_cal.T)
    return (1.0 - sims).min(dim=1)[0]

def get_frr_at_100_rejection(scores_id, scores_ood):
    min_ood_score = scores_ood.min().item()
    false_rejections = (scores_id >= min_ood_score).float().mean().item() * 100.0
    return false_rejections

for coupling_val in [0.0, 0.50]:
    print(f"\n--- OOD Evaluation under Representational Coupling gamma = {coupling_val:.2f} ---")
    
    # Setup calibration and test sets for OOD evaluation under coupling_val
    X_cal_ood, y_cal_ood, task_cal_ood = generate_data_coupled(16, noise_scales, prototypes, bg_noise_scale, coupling=coupling_val)
    psi_cal_ood = project_subspace_coords(X_cal_ood, expert_heads, prototypes)

    # Test ID data (all tasks combined)
    X_test_id, y_test_id, task_test_id = generate_data_coupled(250, noise_scales, prototypes, bg_noise_scale, coupling=coupling_val)
    psi_test_id = project_subspace_coords(X_test_id, expert_heads, prototypes)

    # True OOD data
    X_test_ood_c = []
    for _ in range(250):
        z = np.zeros(D)
        for k in range(K):
            v = np.random.randn(d_block)
            Phi = prototypes[k].T
            proj = Phi @ np.linalg.inv(Phi.T @ Phi) @ Phi.T @ v
            z[k*d_block:(k+1)*d_block] = v - proj
        X_test_ood_c.append(z)
    X_test_ood_c = torch.tensor(np.array(X_test_ood_c), dtype=torch.float32)
    psi_test_ood_c = project_subspace_coords(X_test_ood_c, expert_heads, prototypes)

    # Instantiate routers (RBF and Cosine)
    router_gp_rbf = GPDRRouter(psi_cal_ood, task_cal_ood, K, sigma_f=1.0, lengthscale=0.5, sigma_n=0.01, kernel_type='rbf')
    _, vars_id_rbf = router_gp_rbf.forward(psi_test_id, theta_ood=2.0) # set theta_ood high to get raw variance
    _, vars_ood_rbf = router_gp_rbf.forward(psi_test_ood_c, theta_ood=2.0)

    router_gp_cos = GPDRRouter(psi_cal_ood, task_cal_ood, K, sigma_f=1.0, sigma_n=0.01, kernel_type='cosine')
    _, vars_id_cos = router_gp_cos.forward(psi_test_id, theta_ood=2.0)
    _, vars_ood_cos = router_gp_cos.forward(psi_test_ood_c, theta_ood=2.0)

    # Distance metrics
    dist_euclid_id = get_min_euclidean_dist(psi_test_id, psi_cal_ood)
    dist_euclid_ood = get_min_euclidean_dist(psi_test_ood_c, psi_cal_ood)

    dist_knn_id = get_knn_dist(psi_test_id, psi_cal_ood, k=5)
    dist_knn_ood = get_knn_dist(psi_test_ood_c, psi_cal_ood, k=5)

    dist_cos_id = get_min_cosine_dist(psi_test_id, psi_cal_ood)
    dist_cos_ood = get_min_cosine_dist(psi_test_ood_c, psi_cal_ood)

    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.zeros(len(psi_test_id)), np.ones(len(psi_test_ood_c))])

    auroc_gp_rbf = roc_auc_score(y_true, torch.cat([vars_id_rbf, vars_ood_rbf]).cpu().numpy())
    auroc_gp_cos = roc_auc_score(y_true, torch.cat([vars_id_cos, vars_ood_cos]).cpu().numpy())
    auroc_euclid = roc_auc_score(y_true, torch.cat([dist_euclid_id, dist_euclid_ood]).cpu().numpy())
    auroc_knn = roc_auc_score(y_true, torch.cat([dist_knn_id, dist_knn_ood]).cpu().numpy())
    auroc_cos = roc_auc_score(y_true, torch.cat([dist_cos_id, dist_cos_ood]).cpu().numpy())

    frr_gp_rbf = get_frr_at_100_rejection(vars_id_rbf, vars_ood_rbf)
    frr_gp_cos = get_frr_at_100_rejection(vars_id_cos, vars_ood_cos)
    frr_euclid = get_frr_at_100_rejection(dist_euclid_id, dist_euclid_ood)
    frr_knn = get_frr_at_100_rejection(dist_knn_id, dist_knn_ood)
    frr_cos = get_frr_at_100_rejection(dist_cos_id, dist_cos_ood)

    print(f"OOD Rejection AUROC Score Board:")
    print(f"  GP Posterior Var (RBF):        {auroc_gp_rbf * 100.0:.2f}%")
    print(f"  GP Posterior Var (Cosine):     {auroc_gp_cos * 100.0:.2f}%")
    print(f"  Min Euclidean Distance:        {auroc_euclid * 100.0:.2f}%")
    print(f"  5-NN Euclidean Distance:       {auroc_knn * 100.0:.2f}%")
    print(f"  Min Cosine Distance:           {auroc_cos * 100.0:.2f}%")

    print(f"False Rejection Rate (FRR) on ID tasks at 100% True OOD Rejection:")
    print(f"  GP Posterior Var (RBF):        {frr_gp_rbf:.2f}%")
    print(f"  GP Posterior Var (Cosine):     {frr_gp_cos:.2f}%")
    print(f"  Min Euclidean Distance:        {frr_euclid:.2f}%")
    print(f"  5-NN Euclidean Distance:       {frr_knn:.2f}%")
    print(f"  Min Cosine Distance:           {frr_cos:.2f}%")
    
    # Overlapping OOD Sweep (Reviewer Actionable Suggestion 1)
    print(f"\n  --- Overlapping OOD Sweep (Unit-Sphere Coordinate Mixture) ---")
    for beta in [0.25, 0.50, 0.75]:
        # Generate OOD coordinates as a mixture of ID calibration sample + random unit-sphere noise
        noise = torch.randn(len(psi_test_id), K)
        noise = noise / (torch.norm(noise, p=2, dim=1, keepdim=True) + 1e-8)
        
        indices = np.random.choice(len(psi_cal_ood), size=len(psi_test_id))
        psi_id_basis = psi_cal_ood[indices]
        
        psi_test_ood_overlap = beta * psi_id_basis + (1.0 - beta) * noise
        psi_test_ood_overlap = psi_test_ood_overlap / (torch.norm(psi_test_ood_overlap, p=2, dim=1, keepdim=True) + 1e-8)
        
        # Forward OOD
        _, vars_ood_overlap_rbf = router_gp_rbf.forward(psi_test_ood_overlap, theta_ood=2.0)
        _, vars_ood_overlap_cos = router_gp_cos.forward(psi_test_ood_overlap, theta_ood=2.0)
        
        dist_euclid_overlap_ood = get_min_euclidean_dist(psi_test_ood_overlap, psi_cal_ood)
        dist_knn_overlap_ood = get_knn_dist(psi_test_ood_overlap, psi_cal_ood, k=5)
        dist_cos_overlap_ood = get_min_cosine_dist(psi_test_ood_overlap, psi_cal_ood)
        
        y_true_overlap = np.concatenate([np.zeros(len(psi_test_id)), np.ones(len(psi_test_ood_overlap))])
        
        auroc_overlap_gp_rbf = roc_auc_score(y_true_overlap, torch.cat([vars_id_rbf, vars_ood_overlap_rbf]).cpu().numpy())
        auroc_overlap_gp_cos = roc_auc_score(y_true_overlap, torch.cat([vars_id_cos, vars_ood_overlap_cos]).cpu().numpy())
        auroc_overlap_euclid = roc_auc_score(y_true_overlap, torch.cat([dist_euclid_id, dist_euclid_overlap_ood]).cpu().numpy())
        auroc_overlap_knn = roc_auc_score(y_true_overlap, torch.cat([dist_knn_id, dist_knn_overlap_ood]).cpu().numpy())
        auroc_overlap_cos = roc_auc_score(y_true_overlap, torch.cat([dist_cos_id, dist_cos_overlap_ood]).cpu().numpy())
        
        frr_overlap_gp_rbf = get_frr_at_100_rejection(vars_id_rbf, vars_ood_overlap_rbf)
        frr_overlap_gp_cos = get_frr_at_100_rejection(vars_id_cos, vars_ood_overlap_cos)
        frr_overlap_euclid = get_frr_at_100_rejection(dist_euclid_id, dist_euclid_overlap_ood)
        frr_overlap_knn = get_frr_at_100_rejection(dist_knn_id, dist_knn_overlap_ood)
        frr_overlap_cos = get_frr_at_100_rejection(dist_cos_id, dist_cos_overlap_ood)
        
        print(f"    Beta = {beta:.2f} | GP RBF AUROC: {auroc_overlap_gp_rbf*100.0:.1f}% (FRR: {frr_overlap_gp_rbf:.1f}%) | GP Cos AUROC: {auroc_overlap_gp_cos*100.0:.1f}% (FRR: {frr_overlap_gp_cos:.1f}%) | Min Cos Dist AUROC: {auroc_overlap_cos*100.0:.1f}% (FRR: {frr_overlap_cos:.1f}%)")
    
    # Save router_gp_ood with RBF (which has lengthscale 0.5) for Experiment 3 benchmarking
    if coupling_val == 0.0:
        router_gp_ood = router_gp_rbf


# =====================================================================
# EXPERIMENT 3: MBH Latency and Throughput Benchmarking (Critical Flaw 3)
# =====================================================================
print("\n=== EXPERIMENT 3: MBH Latency and Throughput Benchmarking ===")
batch_sizes = [32, 64, 128, 256, 512]
latency_no_mbh = []
latency_mbh = []
throughput_no_mbh = []
throughput_mbh = []

# Create a heterogeneous dataset stream
X_stream, y_stream, task_stream = generate_data_coupled(1000, noise_scales, prototypes, bg_noise_scale, coupling=0.0)

# Create a small dummy model forward loop representing the modular model forward pass
class ModularForward:
    def __init__(self, W_experts_joint, expert_heads):
        self.W_experts = W_experts_joint
        self.expert_heads = expert_heads
        
    def forward_no_mbh(self, batch_X, alpha_samples):
        # Monolithic batch forwarding
        B_size = batch_X.shape[0]
        alpha_mean = alpha_samples.mean(dim=0, keepdim=True).repeat(B_size, 1)
        logits = compute_logits(batch_X, alpha_mean, self.W_experts)
        return logits
        
    def forward_mbh(self, batch_X, alpha_samples, post_var=None, theta_ood=0.90):
        # MBH sequential micro-batch forwarding
        B_size = batch_X.shape[0]
        if post_var is not None:
            k_star = torch.zeros(B_size, dtype=torch.long)
            for b in range(B_size):
                if post_var[b] > theta_ood:
                    k_star[b] = -1
                else:
                    k_star[b] = alpha_samples[b].argmax()
        else:
            k_star = alpha_samples.argmax(dim=1)
            
        final_logits = torch.zeros(B_size, K * C)
        active_groups = torch.unique(k_star)
        
        for g in active_groups:
            mask_g = (k_star == g)
            if mask_g.sum() == 0:
                continue
            alpha_g_mean = alpha_samples[mask_g].mean(dim=0, keepdim=True).repeat(mask_g.sum(), 1)
            logits_g = compute_logits(batch_X[mask_g], alpha_g_mean, self.W_experts)
            final_logits[mask_g] = logits_g
        return final_logits

    def forward_mbh_cuda_streams(self, batch_X, alpha_samples, post_var=None, theta_ood=0.90):
        # MBH concurrent micro-batch forwarding using CUDA streams for GPU execution optimization
        B_size = batch_X.shape[0]
        if post_var is not None:
            k_star = torch.zeros(B_size, dtype=torch.long)
            for b in range(B_size):
                if post_var[b] > theta_ood:
                    k_star[b] = -1
                else:
                    k_star[b] = alpha_samples[b].argmax()
        else:
            k_star = alpha_samples.argmax(dim=1)
            
        final_logits = torch.zeros(B_size, K * C)
        active_groups = torch.unique(k_star)
        
        if torch.cuda.is_available():
            streams = [torch.cuda.Stream() for _ in active_groups]
            results = {}
            for i, g in enumerate(active_groups):
                mask_g = (k_star == g)
                if mask_g.sum() == 0:
                    continue
                with torch.cuda.stream(streams[i]):
                    alpha_g_mean = alpha_samples[mask_g].mean(dim=0, keepdim=True).repeat(mask_g.sum(), 1)
                    results[g] = compute_logits(batch_X[mask_g].cuda(), alpha_g_mean.cuda(), self.W_experts.cuda())
            
            torch.cuda.synchronize()
            for g in active_groups:
                mask_g = (k_star == g)
                final_logits[mask_g] = results[g].cpu()
        else:
            final_logits = self.forward_mbh(batch_X, alpha_samples, post_var, theta_ood)
            
        return final_logits

forward_runner = ModularForward(W_experts_joint, expert_heads)

for B in batch_sizes:
    # Prepare DataLoader
    loader = DataLoader(TensorDataset(X_stream, y_stream), batch_size=B, shuffle=True)
    
    # Warm-up
    for batch_X, _ in loader:
        B_size = batch_X.shape[0]
        batch_psi = project_subspace_coords(batch_X, expert_heads, prototypes)
        alpha, post_var = router_gp_ood.forward(batch_psi, theta_ood=0.90)
        _ = forward_runner.forward_no_mbh(batch_X, alpha)
        _ = forward_runner.forward_mbh(batch_X, alpha, post_var)
        break
        
    # Benchmarking No MBH
    start_time = time.perf_counter()
    count = 0
    for batch_X, _ in loader:
        B_size = batch_X.shape[0]
        batch_psi = project_subspace_coords(batch_X, expert_heads, prototypes)
        alpha, post_var = router_gp_ood.forward(batch_psi, theta_ood=0.90)
        _ = forward_runner.forward_no_mbh(batch_X, alpha)
        count += B_size
    end_time = time.perf_counter()
    total_time_no_mbh = (end_time - start_time) * 1000.0 # ms
    latency_no_mbh.append(total_time_no_mbh / (len(loader)))
    throughput_no_mbh.append(count / (end_time - start_time))
    
    # Benchmarking With MBH
    start_time = time.perf_counter()
    count = 0
    for batch_X, _ in loader:
        B_size = batch_X.shape[0]
        batch_psi = project_subspace_coords(batch_X, expert_heads, prototypes)
        alpha, post_var = router_gp_ood.forward(batch_psi, theta_ood=0.90)
        _ = forward_runner.forward_mbh(batch_X, alpha, post_var)
        count += B_size
    end_time = time.perf_counter()
    total_time_mbh = (end_time - start_time) * 1000.0 # ms
    latency_mbh.append(total_time_mbh / (len(loader)))
    throughput_mbh.append(count / (end_time - start_time))
    
    print(f"Batch Size: {B:3d} | No MBH Latency: {latency_no_mbh[-1]:6.2f} ms, Throughput: {throughput_no_mbh[-1]:7.1f} samples/s | MBH Latency: {latency_mbh[-1]:6.2f} ms, Throughput: {throughput_mbh[-1]:7.1f} samples/s")

# Plot 4: MBH Latency & Throughput Benchmark Plot
plt.figure(figsize=(10, 5))

# Subplot 1: Latency
plt.subplot(1, 2, 1)
plt.plot(batch_sizes, latency_no_mbh, 'r-o', label='Standard Forwarding (No MBH)')
plt.plot(batch_sizes, latency_mbh, 'g-^', label='Micro-Batch Homogenization (MBH)')
plt.xlabel('Batch Size (B)')
plt.ylabel('Mean Forward Latency (ms / batch)')
plt.title('Forwarding Latency Comparison')
plt.grid(True)
plt.legend()

# Subplot 2: Throughput
plt.subplot(1, 2, 2)
plt.plot(batch_sizes, throughput_no_mbh, 'r-o', label='Standard Forwarding (No MBH)')
plt.plot(batch_sizes, throughput_mbh, 'g-^', label='Micro-Batch Homogenization (MBH)')
plt.xlabel('Batch Size (B)')
plt.ylabel('Throughput (samples / sec)')
plt.title('Streaming Throughput Comparison')
plt.grid(True)
plt.legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/fig4_mbh_latency_benchmarks.png")
plt.close()
print("Saved MBH Latency Benchmark figure to results/fig4_mbh_latency_benchmarks.png")
