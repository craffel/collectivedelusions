import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import warnings

# Suppress sklearn convergence warnings
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

def generate_data(N, K, task_id, is_ood=False, noise_var=0.0):
    # Inactive similarity: mean 0.15, scale 0.08
    coords = np.random.normal(loc=0.15, scale=0.08, size=(N, K))
    
    # Active similarity for task_id (if not OOD)
    if not is_ood:
        coords[:, task_id] = np.random.normal(loc=0.7, scale=0.1, size=N)
        
    coords = np.clip(coords, -1.0, 1.0)
    
    # Symmetric noise injection to model covariate shift representation drift
    if noise_var > 0:
        coords = coords / (1.0 + noise_var) + np.random.normal(loc=0.0, scale=np.sqrt(noise_var), size=coords.shape)
        coords = np.clip(coords, -1.0, 1.0)
        
    return coords

def evaluate_dimension(K, N_calib=64, noise_var=0.05):
    np.random.seed(42)
    models_to_test = {
        "Unreg M=1": lambda: GaussianMixture(n_components=1, covariance_type='diag', random_state=42),
        "Unreg M=2": lambda: GaussianMixture(n_components=2, covariance_type='diag', random_state=42),
        "SRC-DE M=1 (Spherical)": lambda: ShrunkGMM(n_components=1, target_type='spherical', random_state=42),
        "SRC-DE M=2 (Spherical)": lambda: ShrunkGMM(n_components=2, target_type='spherical', random_state=42),
        "SRC-DE M=1 (GlobalDiag)": lambda: ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42),
        "SRC-DE M=2 (GlobalDiag)": lambda: ShrunkGMM(n_components=2, target_type='global_diagonal', random_state=42)
    }
    
    aucs = {name: [] for name in models_to_test}
    
    # Evaluate across multiple tasks to get a stable average
    num_eval_tasks = min(K, 8)
    for task_id in range(num_eval_tasks):
        # 1. Calibration features
        X_calib = generate_data(N_calib, K, task_id, is_ood=False, noise_var=0.0)
        
        # 2. Fit models
        fitted_models = {}
        for name, model_fn in models_to_test.items():
            try:
                model = model_fn()
                model.fit(X_calib)
                fitted_models[name] = model
            except Exception as e:
                # Handle potential singular matrices under extreme overfitting
                fitted_models[name] = None
                
        # 3. Test features
        X_id_test = generate_data(100, K, task_id, is_ood=False, noise_var=noise_var)
        X_ood_test = generate_data(100, K, task_id, is_ood=True, noise_var=noise_var)
        y_true = np.concatenate([np.ones(len(X_id_test)), np.zeros(len(X_ood_test))])
        
        for name, model in fitted_models.items():
            if model is None:
                aucs[name].append(0.5) # Failed model behaves as random guessing
                continue
            try:
                id_scores = model.score_samples(X_id_test)
                ood_scores = model.score_samples(X_ood_test)
                scores = np.concatenate([id_scores, ood_scores])
                auc = roc_auc_score(y_true, scores)
                aucs[name].append(auc)
            except Exception as e:
                aucs[name].append(0.5)
                
    return {name: np.mean(aucs[name]) for name in aucs}

def main():
    K_values = [4, 8, 16, 32, 64]
    print("Running Scaling Simulation: Dimension K vs OOD Rejection AUC")
    print("-" * 110)
    print(f"{'Dim K':5s} | {'Unreg M=1':10s} | {'Unreg M=2':10s} | {'SRC-DE M=1 (Sph)':15s} | {'SRC-DE M=2 (Sph)':15s} | {'SRC-DE M=1 (Glob)':15s} | {'SRC-DE M=2 (Glob)':15s}")
    print("-" * 110)
    
    for K in K_values:
        results = evaluate_dimension(K, N_calib=64, noise_var=0.05)
        print(f"{K:<5d} | {results['Unreg M=1']:<10.4f} | {results['Unreg M=2']:<10.4f} | {results['SRC-DE M=1 (Spherical)']:<15.4f} | {results['SRC-DE M=2 (Spherical)']:<15.4f} | {results['SRC-DE M=1 (GlobalDiag)']:<15.4f} | {results['SRC-DE M=2 (GlobalDiag)']:<15.4f}")
    print("-" * 110)

if __name__ == "__main__":
    main()
