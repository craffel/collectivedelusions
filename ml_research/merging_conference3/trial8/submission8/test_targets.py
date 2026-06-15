import os
import torch
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

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

    def run_evaluation(self, N_calib, noise_var, target_type, seed=42):
        self.compute_centroids(N_calib, seed=seed)
        calib_coords = {}
        for task in TASKS:
            rng = np.random.default_rng(seed)
            train_feats = self.features[task]["train"]
            num_train = len(train_feats)
            indices = rng.permutation(num_train)[:N_calib]
            selected_train_feats = train_feats[indices]
            calib_coords[task] = self.map_to_coordinates(selected_train_feats)
            
        models = {}
        for task in TASKS:
            X_calib = calib_coords[task]
            gmm = ShrunkGMM(n_components=2, target_type=target_type, random_state=seed, reg_covar=1e-5)
            gmm.fit(X_calib)
            models[task] = gmm

        task_aucs = []
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
            gmm_model = models[task]
            id_scores = gmm_model.score_samples(id_coords)
            ood_scores = gmm_model.score_samples(ood_coords)
            scores = np.concatenate([id_scores, ood_scores])
            task_aucs.append(roc_auc_score(y_true, scores))
            
        return np.mean(task_aucs)

def main():
    features = get_extracted_features()
    runner = ExperimentRunner(features)
    seeds = [42, 43, 44, 45, 46]
    
    print("--- EXPERIMENT 1: Robustness vs Noise (N=64) ---")
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
    for noise in noise_levels:
        print(f"Noise sigma^2 = {noise:.2f}:")
        for target in ['spherical', 'global_diagonal']:
            aucs = []
            for s in seeds:
                auc = runner.run_evaluation(N_calib=64, noise_var=noise, target_type=target, seed=s)
                aucs.append(auc)
            print(f"  {target:18s} : AUC = {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
            
    print("\n--- EXPERIMENT 2: Sample Complexity (sigma^2 = 0.05) ---")
    sample_sizes = [8, 16, 32, 64, 128, 256]
    for N in sample_sizes:
        print(f"Calibration size N = {N}:")
        for target in ['spherical', 'global_diagonal']:
            aucs = []
            for s in seeds:
                auc = runner.run_evaluation(N_calib=N, noise_var=0.05, target_type=target, seed=s)
                aucs.append(auc)
            print(f"  {target:18s} : AUC = {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")

if __name__ == "__main__":
    main()
