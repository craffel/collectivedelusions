import os
import torch
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from run_experiments import ShrunkGMM

warnings.filterwarnings('ignore')

TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

class ShrunkFullGMM(GaussianMixture):
    def __init__(self, n_components=2, random_state=42, **kwargs):
        super().__init__(n_components=n_components, covariance_type='full', random_state=random_state, **kwargs)
        
    def fit(self, X, y=None):
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
        super().fit(X, y)
        resp = self.predict_proba(X)
        n_samples, n_features = X.shape
        shrunk_covariances = np.zeros_like(self.covariances_)
        
        for m in range(self.n_components):
            sigma = self.covariances_[m]
            mean = self.means_[m]
            w = resp[:, m]
            W = np.sum(w)
            if W < 1e-5:
                shrunk_covariances[m] = sigma
                continue
                
            Z = X - mean
            nu = np.trace(sigma) / n_features
            T = nu * np.identity(n_features)
            
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

def build_physical_registry(features, C, seed=42):
    """
    Partition each of the 4 tasks into C sub-tasks using KMeans.
    Returns:
      centroids: list of K centroids (tensors) of shape [192]
      subtask_info: list of dicts with keys 'task', 'cluster_id', 'train_feats', 'test_feats'
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    centroids = []
    subtasks = []
    
    # K is 4 * C
    for task in TASKS:
        train_feats = features[task]["train"].numpy()
        test_feats = features[task]["test"].numpy()
        
        # Fit KMeans to group the 256 training samples into C sub-tasks
        if C > 1:
            kmeans = KMeans(n_clusters=C, random_state=seed, n_init=10)
            train_labels = kmeans.fit_predict(train_feats)
            test_labels = kmeans.predict(test_feats)
        else:
            train_labels = np.zeros(len(train_feats), dtype=int)
            test_labels = np.zeros(len(test_feats), dtype=int)
            
        for c in range(C):
            # Training samples in this sub-task
            train_idx = np.where(train_labels == c)[0]
            # If a cluster is empty (unlikely with N=256), fallback
            if len(train_idx) == 0:
                train_idx = np.arange(16)
                
            sub_train = features[task]["train"][train_idx]
            sub_centroid = torch.mean(sub_train, dim=0)
            centroids.append(sub_centroid)
            
            # Test samples matching this sub-task
            test_idx = np.where(test_labels == c)[0]
            if len(test_idx) == 0:
                test_idx = np.arange(16)
            sub_test = features[task]["test"][test_idx]
            
            subtasks.append({
                "task": task,
                "cluster_id": c,
                "train_feats": sub_train.numpy(),
                "test_feats": sub_test.numpy(),
            })
            
    return centroids, subtasks

def map_to_coordinate_space(feats, centroids):
    """Map representations to K-dimensional cosine similarity coordinates."""
    coords_list = []
    for centroid in centroids:
        feats_tensor = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
        feats_tensor = feats_tensor.float()
        norm_feats = torch.norm(feats_tensor, p=2, dim=1, keepdim=True)
        norm_centroid = torch.norm(centroid, p=2)
        sim = torch.mm(feats_tensor, centroid.view(-1, 1)) / (norm_feats * norm_centroid + 1e-8)
        coords_list.append(sim)
    return torch.cat(coords_list, dim=1).numpy()

def evaluate_physical_registry(features, C, noise_var=0.05, seed=42):
    centroids, subtasks = build_physical_registry(features, C, seed=seed)
    K = len(centroids)
    
    # Define models
    models_to_test = {
        "Unreg GMM M=1": lambda: GaussianMixture(n_components=1, covariance_type='diag', random_state=seed),
        "Unreg GMM M=2": lambda: GaussianMixture(n_components=2, covariance_type='diag', random_state=seed),
        "Ridge GMM M=1": lambda: GaussianMixture(n_components=1, covariance_type='diag', reg_covar=1e-4, random_state=seed),
        "Ridge GMM M=2": lambda: GaussianMixture(n_components=2, covariance_type='diag', reg_covar=1e-4, random_state=seed),
        "SRC-DE M=1": lambda: ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=seed),
        "SRC-DE M=2": lambda: ShrunkGMM(n_components=2, target_type='global_diagonal', random_state=seed),
        "Full Shrunk GMM M=2": lambda: ShrunkFullGMM(n_components=2, random_state=seed)
    }
    
    aucs = {name: [] for name in models_to_test}
    
    # Evaluate each subtask as the in-distribution target
    for subtask_idx, subtask in enumerate(subtasks):
        # 1. Calibration features
        # Limit calibration to N=64 samples from this sub-task's train features
        train_feats = subtask["train_feats"]
        n_calib = min(64, len(train_feats))
        rng = np.random.default_rng(seed + subtask_idx)
        calib_idx = rng.permutation(len(train_feats))[:n_calib]
        X_train_physical = train_feats[calib_idx]
        
        # Map training features to K-dimensional coordinates
        X_train_coords = map_to_coordinate_space(X_train_physical, centroids)
        
        # Fit models
        fitted_models = {}
        for name, model_fn in models_to_test.items():
            try:
                model = model_fn()
                model.fit(X_train_coords)
                fitted_models[name] = model
            except Exception as e:
                fitted_models[name] = None
                
        # 2. In-distribution (ID) test features (with symmetric noise)
        test_feats = subtask["test_feats"]
        # Limit to 100 test samples to keep things balanced
        n_test = min(100, len(test_feats))
        test_feats = test_feats[:n_test]
        
        id_test_perturbed = test_feats + np.random.normal(loc=0.0, scale=np.sqrt(noise_var), size=test_feats.shape)
        X_test_id = map_to_coordinate_space(id_test_perturbed, centroids)
        
        # 3. Out-of-distribution (OOD) test features (with symmetric noise)
        # OOD tasks are other physical datasets (excluding the current base dataset)
        ood_feats_list = []
        for other_task in TASKS:
            if other_task == subtask["task"]:
                continue
            # Extract test features of other physical task
            other_feats = features[other_task]["test"].numpy()
            n_other = min(100, len(other_feats))
            other_feats = other_feats[:n_other]
            
            other_perturbed = other_feats + np.random.normal(loc=0.0, scale=np.sqrt(noise_var), size=other_feats.shape)
            ood_feats_list.append(other_perturbed)
            
        X_test_ood_physical = np.concatenate(ood_feats_list, axis=0)
        X_test_ood = map_to_coordinate_space(X_test_ood_physical, centroids)
        
        y_true = np.concatenate([np.ones(len(X_test_id)), np.zeros(len(X_test_ood))])
        
        # Score models
        for name, model in fitted_models.items():
            if model is None:
                aucs[name].append(0.5)
                continue
            try:
                id_scores = model.score_samples(X_test_id)
                ood_scores = model.score_samples(X_test_ood)
                scores = np.concatenate([id_scores, ood_scores])
                auc = roc_auc_score(y_true, scores)
                aucs[name].append(auc)
            except Exception as e:
                aucs[name].append(0.5)
                
    return {name: np.mean(aucs[name]) for name in aucs}

def main():
    print("Loading features...")
    features = torch.load("extracted_features.pt", map_location="cpu")
    
    C_values = [1, 2, 3, 4]
    noise_var = 0.05
    
    print(f"\n--- Physical Task Registry Scaling Audit (Noise={noise_var}) ---")
    print("-" * 140)
    print(f"{'Dim K (4*C)':12s} | {'Unreg M=1':12s} | {'Unreg M=2':12s} | {'Ridge M=1':12s} | {'Ridge M=2':12s} | {'SRC-DE M=1':12s} | {'SRC-DE M=2':12s} | {'Full Shrunk M=2':15s}")
    print("-" * 140)
    
    for C in C_values:
        K = 4 * C
        res = evaluate_physical_registry(features, C, noise_var=noise_var, seed=42)
        print(f"{K:<12d} | "
              f"{res['Unreg GMM M=1']:<12.4f} | "
              f"{res['Unreg GMM M=2']:<12.4f} | "
              f"{res['Ridge GMM M=1']:<12.4f} | "
              f"{res['Ridge GMM M=2']:<12.4f} | "
              f"{res['SRC-DE M=1']:<12.4f} | "
              f"{res['SRC-DE M=2']:<12.4f} | "
              f"{res['Full Shrunk GMM M=2']:<15.4f}")
    print("-" * 140)

if __name__ == "__main__":
    main()
