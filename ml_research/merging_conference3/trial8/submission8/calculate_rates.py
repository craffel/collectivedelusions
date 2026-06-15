import os
import torch
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve

warnings.filterwarnings('ignore')

TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

class ShrunkGMM(GaussianMixture):
    def __init__(self, n_components=2, target_type='global_diagonal', random_state=42, **kwargs):
        super().__init__(n_components=n_components, covariance_type='diag', random_state=random_state, **kwargs)
        self.target_type = target_type
        
    def fit(self, X, y=None):
        super().fit(X, y)
        resp = self.predict_proba(X)
        n_samples, n_features = X.shape
        shrunk_covariances = np.zeros_like(self.covariances_)
        
        if self.target_type == 'global_diagonal':
            global_target = np.var(X, axis=0)
            global_target = np.clip(global_target, 1e-5, None)
            
        for m in range(self.n_components):
            sigmas = self.covariances_[m]
            
            if self.target_type == 'spherical':
                T = np.mean(sigmas) * np.ones_like(sigmas)
            elif self.target_type == 'global_diagonal':
                T = global_target
            elif self.target_type == 'identity':
                T = np.ones_like(sigmas)
            else:
                raise ValueError(f"Unknown target type: {self.target_type}")
                
            w = resp[:, m]
            W = np.sum(w)
            if W < 1e-5:
                shrunk_covariances[m] = sigmas
                continue
                
            diffs = (X - self.means_[m])**2
            var_of_vars = np.zeros(n_features)
            for j in range(n_features):
                var_of_vars[j] = np.sum(w**2 * (diffs[:, j] - sigmas[j])**2) / (W**2 + 1e-8)
                
            sum_var = np.sum(var_of_vars)
            sum_diff = np.sum((sigmas - T)**2)
            
            if sum_diff < 1e-8:
                alpha_opt = 1.0
            else:
                alpha_opt = sum_var / sum_diff
            alpha_opt = np.clip(alpha_opt, 0.0, 1.0)
            
            shrunk_covariances[m] = (1.0 - alpha_opt) * sigmas + alpha_opt * T
            
        self.covariances_ = shrunk_covariances
        self.precisions_cholesky_ = 1.0 / np.sqrt(shrunk_covariances)
        return self


def tune_ridge_gmm(X_calib, n_components, seed, reg_covar=1e-5):
    """
    Perform 3-fold cross-validation over calibration coordinates X_calib
    to select the optimal Ridge regularizer gamma from [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].
    """
    N = len(X_calib)
    if N < 2 * n_components:
        return 1e-4
        
    n_splits = 3
    candidates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    best_gamma = 1e-4
    best_score = -np.inf
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, n_splits)
    
    for gamma in candidates:
        scores = []
        for fold_idx in range(n_splits):
            val_idx = folds[fold_idx]
            train_idx = np.array([idx for idx in indices if idx not in val_idx])
            
            if len(train_idx) < n_components:
                continue
                
            X_train = X_calib[train_idx]
            X_val = X_calib[val_idx]
            
            try:
                gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=seed, reg_covar=reg_covar)
                gmm.fit(X_train)
                gmm.covariances_ = gmm.covariances_ + gamma
                gmm.precisions_cholesky_ = 1.0 / np.sqrt(gmm.covariances_)
                
                score = np.mean(gmm.score_samples(X_val))
                scores.append(score)
            except Exception:
                pass
                
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_gamma = gamma
                
    return best_gamma


def get_extracted_features():
    return torch.load("extracted_features.pt", map_location="cpu")

class ExperimentRunner:
    def __init__(self, features):
        self.features = features
        self.centroids = {}
        
    def compute_centroids(self, N_calib, seed=42):
        for task in TASKS:
            rng = np.random.default_rng(seed)
            train_feats = self.features[task]["train"]
            num_train = len(train_feats)
            indices = rng.permutation(num_train)[:N_calib]
            selected_train_feats = train_feats[indices]
            centroid = torch.mean(selected_train_feats, dim=0)
            self.centroids[task] = centroid
            
    def map_to_coordinates(self, feats):
        coords_list = []
        for task in TASKS:
            centroid = self.centroids[task]
            norm_feats = torch.norm(feats, p=2, dim=1, keepdim=True)
            norm_centroid = torch.norm(centroid, p=2)
            sim = torch.mm(feats, centroid.view(-1, 1)) / (norm_feats * norm_centroid + 1e-8)
            coords_list.append(sim)
        return torch.cat(coords_list, dim=1).numpy()

    def add_noise_to_features(self, feats, noise_var, seed=None):
        if noise_var == 0.0:
            return feats
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn_like(feats) * np.sqrt(noise_var)
        return feats + noise

    def get_rates(self, N_calib, noise_var, seed=42):
        self.compute_centroids(N_calib, seed=seed)
        calib_coords = {}
        for task in TASKS:
            rng = np.random.default_rng(seed)
            train_feats = self.features[task]["train"]
            num_train = len(train_feats)
            indices = rng.permutation(num_train)[:N_calib]
            selected_train_feats = train_feats[indices]
            calib_coords[task] = self.map_to_coordinates(selected_train_feats)
            
        models = {
            "Unreg GMM": {},
            "Ridge GMM": {},
            "Tuned Ridge GMM": {},
            "SRC-DE": {}
        }
        for task in TASKS:
            X_calib = calib_coords[task]
            
            # Unreg
            gmm_unreg = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
            gmm_unreg.fit(X_calib)
            models["Unreg GMM"][task] = gmm_unreg
            
            # Ridge
            gmm_ridge = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
            gmm_ridge.fit(X_calib)
            gmm_ridge.covariances_ = gmm_ridge.covariances_ + 1e-4
            gmm_ridge.precisions_cholesky_ = 1.0 / np.sqrt(gmm_ridge.covariances_)
            models["Ridge GMM"][task] = gmm_ridge

            # Tuned Ridge
            gamma_opt = tune_ridge_gmm(X_calib, n_components=2, seed=seed)
            gmm_tuned_ridge = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
            gmm_tuned_ridge.fit(X_calib)
            gmm_tuned_ridge.covariances_ = gmm_tuned_ridge.covariances_ + gamma_opt
            gmm_tuned_ridge.precisions_cholesky_ = 1.0 / np.sqrt(gmm_tuned_ridge.covariances_)
            models["Tuned Ridge GMM"][task] = gmm_tuned_ridge
            
            # SRC-DE
            gmm_shrunk = ShrunkGMM(n_components=2, random_state=seed, reg_covar=1e-5)
            gmm_shrunk.fit(X_calib)
            models["SRC-DE"][task] = gmm_shrunk

        model_fprs = {name: [] for name in models}
        
        for task_idx, task in enumerate(TASKS):
            id_feats_perturbed = self.add_noise_to_features(self.features[task]["test"], noise_var, seed=seed+task_idx)
            id_coords = self.map_to_coordinates(id_feats_perturbed)
            
            ood_coords_list = []
            for ood_task_idx, ood_task in enumerate(TASKS):
                if ood_task == task:
                    continue
                ood_feats_perturbed = self.add_noise_to_features(self.features[ood_task]["test"], noise_var, seed=seed+ood_task_idx+10)
                ood_coords_list.append(self.map_to_coordinates(ood_feats_perturbed))
            ood_coords = np.concatenate(ood_coords_list, axis=0)
            
            y_true = np.concatenate([np.ones(len(id_coords)), np.zeros(len(ood_coords))])
            
            for model_name in models:
                gmm_model = models[model_name][task]
                id_scores = gmm_model.score_samples(id_coords)
                ood_scores = gmm_model.score_samples(ood_coords)
                scores = np.concatenate([id_scores, ood_scores])
                
                fpr, tpr, thresholds = roc_curve(y_true, scores)
                # Find FPR at TPR >= 0.90
                idx = np.where(tpr >= 0.90)[0]
                if len(idx) > 0:
                    fpr_at_target = fpr[idx[0]]
                else:
                    fpr_at_target = 1.0
                model_fprs[model_name].append(fpr_at_target)
                
        return {name: np.mean(model_fprs[name]) for name in model_fprs}

def main():
    features = get_extracted_features()
    runner = ExperimentRunner(features)
    seeds = list(range(42, 62))
    
    noise = 0.05
    N = 64
    
    all_fprs = {name: [] for name in ["Unreg GMM", "Ridge GMM", "Tuned Ridge GMM", "SRC-DE"]}
    for s in seeds:
        fprs = runner.get_rates(N_calib=N, noise_var=noise, seed=s)
        for name in fprs:
            all_fprs[name].append(fprs[name])
            
    print(f"At N={N}, sigma^2={noise}:")
    for name in all_fprs:
        mean_fpr = np.mean(all_fprs[name])
        std_fpr = np.std(all_fprs[name])
        print(f"  {name:16s} : FPR at TPR=0.90 is {mean_fpr:.4f} +- {std_fpr:.4f}")

if __name__ == "__main__":
    main()
