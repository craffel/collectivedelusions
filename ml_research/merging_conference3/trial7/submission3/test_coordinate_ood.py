import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

K = 4        # number of tasks
N_cal = 64   # calibration dataset size
N_test = 250 # test size

# Generate synthetic ID subspace coordinates psi_cal and psi_test_id
# For ID, one coordinate is dominant, others are small (representing task-specific activations)
def generate_id_psi(num_samples, K, coupling=0.0):
    psi_list = []
    labels = []
    for _ in range(num_samples):
        k = np.random.randint(0, K)
        u = np.zeros(K)
        u[k] = 1.0
        for j in range(K):
            if j != k:
                u[j] = coupling + np.random.randn() * 0.05
        # normalize to unit sphere
        u = np.maximum(u, 1e-5)
        u = u / np.linalg.norm(u)
        psi_list.append(u)
        labels.append(k)
    return torch.tensor(np.array(psi_list), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

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
        
    def forward(self, psi_test):
        k_star = self.kernel(psi_test, self.psi_train)
        k_star_M = torch.matmul(k_star, self.M)
        # Compute posterior variance with a non-negative clamping safeguard to prevent numerical instabilities
        post_var = torch.clamp((self.sigma_f ** 2) - (k_star_M * k_star).sum(dim=1), min=0.0)
        return post_var

def get_min_euclidean_dist(psi_test, psi_cal):
    return torch.cdist(psi_test, psi_cal, p=2).min(dim=1)[0]

def get_knn_dist(psi_test, psi_cal, k=5):
    dists = torch.cdist(psi_test, psi_cal, p=2)
    topk_dists, _ = torch.topk(dists, k, dim=1, largest=False)
    return topk_dists.mean(dim=1)

def get_min_cosine_dist(psi_test, psi_cal):
    sims = torch.matmul(psi_test, psi_cal.T)
    return (1.0 - sims).min(dim=1)[0]

def get_frr_at_100_rejection(scores_id, scores_ood):
    min_ood_score = scores_ood.min().item()
    false_rejections = (scores_id >= min_ood_score).float().mean().item() * 100.0
    return false_rejections

# Evaluate uncoupled and coupled spaces
for coupling_val in [0.0, 0.50]:
    print(f"\n==================================================")
    print(f"EVALUATION UNDER COUPLING gamma = {coupling_val:.2f}")
    print(f"==================================================")
    
    psi_cal, y_cal = generate_id_psi(64, K, coupling=coupling_val)
    psi_test_id, y_test_id = generate_id_psi(250, K, coupling=coupling_val)
    
    # Initialize GP-DR routers
    router_gp_rbf = GPDRRouter(psi_cal, y_cal, K, sigma_f=1.0, lengthscale=0.5, sigma_n=0.01, kernel_type='rbf')
    vars_id_rbf = router_gp_rbf.forward(psi_test_id)
    
    router_gp_cos = GPDRRouter(psi_cal, y_cal, K, sigma_f=1.0, sigma_n=0.01, kernel_type='cosine')
    vars_id_cos = router_gp_cos.forward(psi_test_id)
    
    dist_euclid_id = get_min_euclidean_dist(psi_test_id, psi_cal)
    dist_knn_id = get_knn_dist(psi_test_id, psi_cal, k=5)
    dist_cos_id = get_min_cosine_dist(psi_test_id, psi_cal)
    
    for beta in [0.0, 0.25, 0.50, 0.75, 0.90]:
        print(f"\n--- Overlapping OOD Sweep with beta = {beta:.2f} ---")
        
        # Generate OOD coordinates as a mixture of ID calibration sample + random unit-sphere noise
        # Random noise on the unit sphere
        noise = torch.randn(250, K)
        noise = noise / torch.norm(noise, p=2, dim=1, keepdim=True)
        
        # Select random calibration samples as basis
        indices = np.random.choice(len(psi_cal), size=250)
        psi_id_basis = psi_cal[indices]
        
        psi_test_ood = beta * psi_id_basis + (1.0 - beta) * noise
        psi_test_ood = psi_test_ood / torch.norm(psi_test_ood, p=2, dim=1, keepdim=True)
        
        # Forward OOD
        vars_ood_rbf = router_gp_rbf.forward(psi_test_ood)
        vars_ood_cos = router_gp_cos.forward(psi_test_ood)
        
        dist_euclid_ood = get_min_euclidean_dist(psi_test_ood, psi_cal)
        dist_knn_ood = get_knn_dist(psi_test_ood, psi_cal, k=5)
        dist_cos_ood = get_min_cosine_dist(psi_test_ood, psi_cal)
        
        # Compute AUROC
        y_true = np.concatenate([np.zeros(len(psi_test_id)), np.ones(len(psi_test_ood))])
        
        auroc_gp_rbf = roc_auc_score(y_true, torch.cat([vars_id_rbf, vars_ood_rbf]).cpu().numpy())
        auroc_gp_cos = roc_auc_score(y_true, torch.cat([vars_id_cos, vars_ood_cos]).cpu().numpy())
        auroc_euclid = roc_auc_score(y_true, torch.cat([dist_euclid_id, dist_euclid_ood]).cpu().numpy())
        auroc_knn = roc_auc_score(y_true, torch.cat([dist_knn_id, dist_knn_ood]).cpu().numpy())
        auroc_cos = roc_auc_score(y_true, torch.cat([dist_cos_id, dist_cos_ood]).cpu().numpy())
        
        # Compute FRR
        frr_gp_rbf = get_frr_at_100_rejection(vars_id_rbf, vars_ood_rbf)
        frr_gp_cos = get_frr_at_100_rejection(vars_id_cos, vars_ood_cos)
        frr_euclid = get_frr_at_100_rejection(dist_euclid_id, dist_euclid_ood)
        frr_knn = get_frr_at_100_rejection(dist_knn_id, dist_knn_ood)
        frr_cos = get_frr_at_100_rejection(dist_cos_id, dist_cos_ood)
        
        print(f"AUROC Scores:")
        print(f"  GP Posterior Var (RBF):    {auroc_gp_rbf * 100.0:.2f}% | FRR: {frr_gp_rbf:.2f}%")
        print(f"  GP Posterior Var (Cosine): {auroc_gp_cos * 100.0:.2f}% | FRR: {frr_gp_cos:.2f}%")
        print(f"  Min Euclidean Distance:    {auroc_euclid * 100.0:.2f}% | FRR: {frr_euclid:.2f}%")
        print(f"  5-NN Euclidean Distance:   {auroc_knn * 100.0:.2f}% | FRR: {frr_knn:.2f}%")
        print(f"  Min Cosine Distance:       {auroc_cos * 100.0:.2f}% | FRR: {frr_cos:.2f}%")
