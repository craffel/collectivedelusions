import os
import torch
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

warnings.filterwarnings('ignore')

TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

class ShrunkFullGMM(GaussianMixture):
    def __init__(self, n_components=2, random_state=42, **kwargs):
        super().__init__(n_components=n_components, covariance_type='full', random_state=random_state, **kwargs)
        
    def fit(self, X, y=None):
        # 1. Standard EM fit
        super().fit(X, y)
        
        # 2. Extract statistics and responsibilities
        resp = self.predict_proba(X)
        n_samples, n_features = X.shape
        shrunk_covariances = np.zeros_like(self.covariances_) # [n_components, n_features, n_features]
        
        for m in range(self.n_components):
            sigma = self.covariances_[m] # [n_features, n_features]
            mean = self.means_[m] # [n_features]
            
            w = resp[:, m]
            W = np.sum(w)
            if W < 1e-5:
                shrunk_covariances[m] = sigma
                continue
                
            # Centered data
            Z = X - mean # [n_samples, n_features]
            
            # Target is spherical diagonal
            nu = np.trace(sigma) / n_features
            T = nu * np.identity(n_features)
            
            # Compute variance of sample covariance elements (Ledoit-Wolf approach)
            sum_var = 0.0
            for i in range(n_features):
                for j in range(n_features):
                    dev = Z[:, i] * Z[:, j] - sigma[i, j]
                    sum_var += np.sum(w**2 * dev**2)
                    
            sum_var = sum_var / (W**2 + 1e-8)
            sum_diff = np.sum((sigma - T)**2)
            
            if sum_diff < 1e-8:
                alpha_opt = 1.0
            else:
                alpha_opt = sum_var / sum_diff
            alpha_opt = np.clip(alpha_opt, 0.0, 1.0)
            
            shrunk_covariances[m] = (1.0 - alpha_opt) * sigma + alpha_opt * T
            
        self.covariances_ = shrunk_covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(shrunk_covariances, 'full')
        return self

# Let's see if we can load the features and evaluate it!
features = torch.load("extracted_features.pt", map_location="cpu")

# Simple centroid computing and coordinate mapping
centroids = {}
for task in TASKS:
    centroids[task] = torch.mean(features[task]["train"][:64], dim=0)

def map_to_coords(feats):
    coords_list = []
    for task in TASKS:
        centroid = centroids[task]
        norm_feats = torch.norm(feats, p=2, dim=1, keepdim=True)
        norm_centroid = torch.norm(centroid, p=2)
        sim = torch.mm(feats, centroid.view(-1, 1)) / (norm_feats * norm_centroid + 1e-8)
        coords_list.append(sim)
    return torch.cat(coords_list, dim=1).numpy()

# Run evaluation under 0.05 noise
noise_var = 0.05
task_aucs_full = []
task_aucs_diag = []

for task_idx, task in enumerate(TASKS):
    # Train
    X_train = map_to_coords(features[task]["train"][:64])
    
    # Test (with noise)
    id_test_perturbed = features[task]["test"] + torch.randn_like(features[task]["test"]) * np.sqrt(noise_var)
    X_test_id = map_to_coords(id_test_perturbed)
    
    # OOD Test
    ood_coords_list = []
    for ood_task in TASKS:
        if ood_task == task:
            continue
        ood_test_perturbed = features[ood_task]["test"] + torch.randn_like(features[ood_task]["test"]) * np.sqrt(noise_var)
        ood_coords_list.append(map_to_coords(ood_test_perturbed))
    X_test_ood = np.concatenate(ood_coords_list, axis=0)
    
    # Labels
    y_true = np.concatenate([np.ones(len(X_test_id)), np.zeros(len(X_test_ood))])
    
    # Fit Full Shrunk GMM
    gmm_full = ShrunkFullGMM(n_components=2, random_state=42)
    gmm_full.fit(X_train)
    scores_full = np.concatenate([gmm_full.score_samples(X_test_id), gmm_full.score_samples(X_test_ood)])
    task_aucs_full.append(roc_auc_score(y_true, scores_full))
    
    # Fit Diagonal Shrunk GMM
    # Import diagonal ShrunkGMM from run_experiments.py
    from run_experiments import ShrunkGMM
    gmm_diag = ShrunkGMM(n_components=2, target_type='global_diagonal', random_state=42)
    gmm_diag.fit(X_train)
    scores_diag = np.concatenate([gmm_diag.score_samples(X_test_id), gmm_diag.score_samples(X_test_ood)])
    task_aucs_diag.append(roc_auc_score(y_true, scores_diag))

print("Average AUC under noise 0.05:")
print(f"Full Shrunk GMM: {np.mean(task_aucs_full):.4f}")
print(f"Diagonal Shrunk GMM: {np.mean(task_aucs_diag):.4f}")
