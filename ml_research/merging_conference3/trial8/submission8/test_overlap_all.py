import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class ShrunkGMM(GaussianMixture):
    def __init__(self, n_components=1, target_type='global_diagonal', random_state=None, **kwargs):
        super().__init__(n_components=n_components, covariance_type='diag', random_state=random_state, **kwargs)
        self.target_type = target_type

    def fit(self, X, y=None):
        super().fit(X, y)
        N, D = X.shape
        M = self.n_components
        
        # Calculate component posterior responsibilities gamma
        gamma = self.predict_proba(X) # Shape: (N, M)
        
        # Empirical and shrunk covariances
        empirical_covariances = self.covariances_.copy() # Shape: (M, D)
        shrunk_covariances = np.zeros_like(empirical_covariances)
        
        # Global target
        global_vars = np.var(X, axis=0) # Shape: (D,)
        
        for m in range(M):
            sigmas = empirical_covariances[m] # Shape: (D,)
            W_m = np.sum(gamma[:, m])
            if W_m <= 1:
                shrunk_covariances[m] = sigmas
                continue
                
            # Compute responsibility-weighted fourth central moments for estimator variance
            diff = (X - self.means_[m]) ** 2 # Shape: (N, D)
            var_estimates = np.sum(gamma[:, [m]] * (diff - sigmas) ** 2, axis=0) / (W_m ** 2)
            
            # Select target
            if self.target_type == 'global_diagonal':
                T = global_vars
            elif self.target_type == 'spherical':
                T = np.ones(D) * np.mean(sigmas)
            else:
                T = np.zeros(D)
                
            # Analytical shrinkage optimal intensity alpha_opt
            sum_diff = np.sum((sigmas - T) ** 2)
            sum_var = np.sum(var_estimates)
            
            if sum_diff == 0:
                alpha_opt = 1.0
            else:
                alpha_opt = sum_var / sum_diff
            alpha_opt = np.clip(alpha_opt, 0.0, 1.0)
            
            shrunk_covariances[m] = (1.0 - alpha_opt) * sigmas + alpha_opt * T
            
        self.covariances_ = shrunk_covariances
        self.precisions_cholesky_ = 1.0 / np.sqrt(shrunk_covariances)
        return self

def generate_data_with_overlap(N, K, task_id, is_ood=False, noise_var=0.0, overlap_prob=0.3):
    # Inactive similarity: mean 0.15, scale 0.08
    coords = np.random.normal(loc=0.15, scale=0.08, size=(N, K))
    
    if not is_ood:
        coords[:, task_id] = np.random.normal(loc=0.7, scale=0.1, size=N)
    else:
        for i in range(N):
            if np.random.rand() < overlap_prob and K >= 2:
                # Overlapping OOD query: active similarity on task_id and another task
                other_candidates = [j for j in range(K) if j != task_id]
                other_idx = np.random.choice(other_candidates)
                coords[i, task_id] = np.random.normal(loc=0.7, scale=0.1)
                coords[i, other_idx] = np.random.normal(loc=0.7, scale=0.1)
            else:
                # Pure OOD query
                pass
                
    coords = np.clip(coords, -1.0, 1.0)
    
    if noise_var > 0:
        coords = coords / (1.0 + noise_var) + np.random.normal(loc=0.0, scale=np.sqrt(noise_var), size=coords.shape)
        coords = np.clip(coords, -1.0, 1.0)
        
    return coords

def evaluate_overlap_all(K, N_calib=64, noise_var=0.05, overlap_prob=0.4):
    np.random.seed(42)
    
    aucs = {
        "Raw Cosine": [],
        "Full Unreg M=1": [],
        "Full SRC-DE M=1": [],
        "Full SRC-DE M=1 (Noise-Adapted)": [],
        "1D Unreg GMM M=2": [],
        "1D SRC-DE GMM M=2": []
    }
    
    num_eval_tasks = min(K, 8)
    
    for task_id in range(num_eval_tasks):
        # Generate Calibration Data
        X_calib = generate_data_with_overlap(N_calib, K, task_id, is_ood=False, noise_var=0.0)
        X_calib_1d = X_calib[:, [task_id]]
        
        # Fit Full GMM Models
        model_full_unreg = GaussianMixture(n_components=1, covariance_type='diag', random_state=42).fit(X_calib)
        model_full_src = ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42).fit(X_calib)
        
        # Fit 1D GMM Models
        model_1d_unreg = GaussianMixture(n_components=2, covariance_type='diag', random_state=42).fit(X_calib_1d)
        model_1d_src = ShrunkGMM(n_components=2, target_type='global_diagonal', random_state=42).fit(X_calib_1d)
        
        # Generate Test Data
        X_id_test = generate_data_with_overlap(100, K, task_id, is_ood=False, noise_var=noise_var)
        X_ood_test = generate_data_with_overlap(100, K, task_id, is_ood=True, noise_var=noise_var, overlap_prob=overlap_prob)
        y_true = np.concatenate([np.ones(len(X_id_test)), np.zeros(len(X_ood_test))])
        
        # 1. Raw Cosine
        scores_cos = np.concatenate([X_id_test[:, task_id], X_ood_test[:, task_id]])
        aucs["Raw Cosine"].append(roc_auc_score(y_true, scores_cos))
        
        # 2. Full Unreg M=1
        id_full_unreg = model_full_unreg.score_samples(X_id_test)
        ood_full_unreg = model_full_unreg.score_samples(X_ood_test)
        aucs["Full Unreg M=1"].append(roc_auc_score(y_true, np.concatenate([id_full_unreg, ood_full_unreg])))
        
        # 3. Full SRC-DE M=1
        id_full_src = model_full_src.score_samples(X_id_test)
        ood_full_src = model_full_src.score_samples(X_ood_test)
        aucs["Full SRC-DE M=1"].append(roc_auc_score(y_true, np.concatenate([id_full_src, ood_full_src])))
        
        # 4. Full SRC-DE M=1 (Noise-Adapted)
        adapted_model = ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42)
        adapted_model.means_ = model_full_src.means_.copy()
        adapted_model.covariances_ = model_full_src.covariances_.copy() + noise_var
        adapted_model.precisions_cholesky_ = 1.0 / np.sqrt(adapted_model.covariances_)
        adapted_model.weights_ = model_full_src.weights_.copy()
        
        id_full_adapted = adapted_model.score_samples(X_id_test)
        ood_full_adapted = adapted_model.score_samples(X_ood_test)
        aucs["Full SRC-DE M=1 (Noise-Adapted)"].append(roc_auc_score(y_true, np.concatenate([id_full_adapted, ood_full_adapted])))
        
        # 5. 1D Unreg GMM M=2
        X_id_test_1d = X_id_test[:, [task_id]]
        X_ood_test_1d = X_ood_test[:, [task_id]]
        id_1d_unreg = model_1d_unreg.score_samples(X_id_test_1d)
        ood_1d_unreg = model_1d_unreg.score_samples(X_ood_test_1d)
        aucs["1D Unreg GMM M=2"].append(roc_auc_score(y_true, np.concatenate([id_1d_unreg, ood_1d_unreg])))
        
        # 6. 1D SRC-DE GMM M=2
        id_1d_src = model_1d_src.score_samples(X_id_test_1d)
        ood_1d_src = model_1d_src.score_samples(X_ood_test_1d)
        aucs["1D SRC-DE GMM M=2"].append(roc_auc_score(y_true, np.concatenate([id_1d_src, ood_1d_src])))
        
    return {name: np.mean(aucs[name]) for name in aucs}

def main():
    K_values = [4, 8, 16, 32, 64]
    print("Evaluating Overlap Scenario (p_overlap=0.4, noise_var=0.05, N_calib=64)")
    print("-" * 130)
    print(f"{'Dim K':5s} | {'Raw Cosine':10s} | {'Full Unreg':10s} | {'Full SRC-DE':11s} | {'Full SRC-DE (NA)':17s} | {'1D Unreg GMM':12s} | {'1D SRC-DE GMM':13s}")
    print("-" * 130)
    for K in K_values:
        res = evaluate_overlap_all(K, N_calib=64, noise_var=0.05, overlap_prob=0.4)
        print(f"{K:<5d} | {res['Raw Cosine']:<10.4f} | {res['Full Unreg M=1']:<10.4f} | {res['Full SRC-DE M=1']:<11.4f} | {res['Full SRC-DE M=1 (Noise-Adapted)']:<17.4f} | {res['1D Unreg GMM M=2']:<12.4f} | {res['1D SRC-DE GMM M=2']:<13.4f}")
    print("-" * 130)

if __name__ == "__main__":
    main()
