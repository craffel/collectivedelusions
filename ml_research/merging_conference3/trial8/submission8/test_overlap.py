import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class ShrunkGMM(GaussianMixture):
    def __init__(self, n_components=2, target_type='spherical', random_state=42, **kwargs):
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

def generate_data_with_overlap(N, K, task_id, is_ood=False, noise_var=0.0, overlap_prob=0.3):
    # Inactive similarity: mean 0.15, scale 0.08
    coords = np.random.normal(loc=0.15, scale=0.08, size=(N, K))
    
    if not is_ood:
        # ID: active similarity for task_id
        coords[:, task_id] = np.random.normal(loc=0.7, scale=0.1, size=N)
    else:
        # OOD: Some are overlapping task mixtures, others are pure OOD
        for i in range(N):
            if np.random.rand() < overlap_prob and K >= 2:
                # Overlapping OOD query: has two active dimensions (including task_id and another task)
                active_idx = [task_id]
                other_candidates = [j for j in range(K) if j != task_id]
                other_idx = np.random.choice(other_candidates)
                coords[i, task_id] = np.random.normal(loc=0.6, scale=0.1)
                coords[i, other_idx] = np.random.normal(loc=0.6, scale=0.1)
            else:
                # Normal OOD query: inactive similarities everywhere (including task_id)
                pass
                
    coords = np.clip(coords, -1.0, 1.0)
    
    # Symmetric noise injection to model covariate shift representation drift
    if noise_var > 0:
        coords = coords / (1.0 + noise_var) + np.random.normal(loc=0.0, scale=np.sqrt(noise_var), size=coords.shape)
        coords = np.clip(coords, -1.0, 1.0)
        
    return coords

def evaluate_overlap_scenario(K, N_calib=64, noise_var=0.05, overlap_prob=0.3):
    np.random.seed(42)
    
    models_to_test = {
        "Raw Cosine": None,
        "Unreg M=1": lambda: GaussianMixture(n_components=1, covariance_type='diag', random_state=42),
        "SRC-DE M=1": lambda: ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42),
        "SRC-DE M=1 (Noise-Adapted)": lambda: ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42),
    }
    
    aucs = {name: [] for name in models_to_test}
    num_eval_tasks = min(K, 8)
    
    for task_id in range(num_eval_tasks):
        # 1. Calibration features
        X_calib = generate_data_with_overlap(N_calib, K, task_id, is_ood=False, noise_var=0.0)
        
        # 2. Fit models
        fitted_models = {}
        for name, model_fn in models_to_test.items():
            if name == "Raw Cosine":
                continue
            try:
                model = model_fn()
                model.fit(X_calib)
                fitted_models[name] = model
            except Exception as e:
                fitted_models[name] = None
                
        # 3. Test features
        X_id_test = generate_data_with_overlap(100, K, task_id, is_ood=False, noise_var=noise_var)
        X_ood_test = generate_data_with_overlap(100, K, task_id, is_ood=True, noise_var=noise_var, overlap_prob=overlap_prob)
        y_true = np.concatenate([np.ones(len(X_id_test)), np.zeros(len(X_ood_test))])
        
        # Eval Raw Cosine
        scores_cos = np.concatenate([X_id_test[:, task_id], X_ood_test[:, task_id]])
        aucs["Raw Cosine"].append(roc_auc_score(y_true, scores_cos))
        
        for name, model in fitted_models.items():
            if model is None:
                aucs[name].append(0.5)
                continue
            try:
                if "Noise-Adapted" in name:
                    # Adapt GMM covariance by adding noise variance to the diagonal
                    adapted_model = ShrunkGMM(n_components=model.n_components, target_type=model.target_type, random_state=model.random_state)
                    adapted_model.means_ = model.means_.copy()
                    adapted_model.covariances_ = model.covariances_.copy() + noise_var
                    adapted_model.precisions_cholesky_ = 1.0 / np.sqrt(adapted_model.covariances_)
                    adapted_model.weights_ = model.weights_.copy()
                    
                    id_scores = adapted_model.score_samples(X_id_test)
                    ood_scores = adapted_model.score_samples(X_ood_test)
                else:
                    id_scores = model.score_samples(X_id_test)
                    ood_scores = model.score_samples(X_ood_test)
                    
                scores = np.concatenate([id_scores, ood_scores])
                auc = roc_auc_score(y_true, scores)
                aucs[name].append(auc)
            except Exception as e:
                import traceback
                traceback.print_exc()
                aucs[name].append(0.5)
                
    return {name: np.mean(aucs[name]) for name in aucs}

def main():
    K_values = [4, 8, 16, 32, 64]
    print("Evaluating Overlap Scenario with overlap_prob=0.4")
    print("-" * 90)
    print(f"{'Dimension K':12s} | {'Raw Cosine':12s} | {'Unreg M=1':12s} | {'SRC-DE M=1':12s} | {'SRC-DE M=1 (Noise-Adapted)':28s}")
    print("-" * 90)
    for K in K_values:
        res = evaluate_overlap_scenario(K, N_calib=64, noise_var=0.05, overlap_prob=0.4)
        print(f"{K:<12d} | {res['Raw Cosine']:<12.4f} | {res['Unreg M=1']:<12.4f} | {res['SRC-DE M=1']:<12.4f} | {res['SRC-DE M=1 (Noise-Adapted)']:<28.4f}")
    print("-" * 90)

if __name__ == "__main__":
    main()
