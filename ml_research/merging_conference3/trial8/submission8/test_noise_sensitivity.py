import numpy as np
from sklearn.metrics import roc_auc_score
from test_overlap_all import ShrunkGMM, generate_data_with_overlap
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')

def evaluate_noise_sensitivity():
    K = 4
    N_calib = 64
    noise_var = 0.05
    overlap_prob = 0.4
    np.random.seed(42)
    
    # We sweep beta, where estimated_noise = beta * noise_var
    beta_values = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    
    results = {beta: [] for beta in beta_values}
    
    num_eval_tasks = 4
    
    for task_id in range(num_eval_tasks):
        # Generate Calibration Data
        X_calib = generate_data_with_overlap(N_calib, K, task_id, is_ood=False, noise_var=0.0)
        
        # Fit Full SRC-DE Model
        model_full_src = ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42).fit(X_calib)
        
        # Generate Test Data
        X_id_test = generate_data_with_overlap(100, K, task_id, is_ood=False, noise_var=noise_var)
        X_ood_test = generate_data_with_overlap(100, K, task_id, is_ood=True, noise_var=noise_var, overlap_prob=overlap_prob)
        y_true = np.concatenate([np.ones(len(X_id_test)), np.zeros(len(X_ood_test))])
        
        # Sweep beta
        for beta in beta_values:
            adapted_model = ShrunkGMM(n_components=1, target_type='global_diagonal', random_state=42)
            adapted_model.means_ = model_full_src.means_.copy()
            # Covariance adaptation with estimated noise
            estimated_noise = beta * noise_var
            adapted_model.covariances_ = model_full_src.covariances_.copy() + estimated_noise
            adapted_model.precisions_cholesky_ = 1.0 / np.sqrt(adapted_model.covariances_)
            adapted_model.weights_ = model_full_src.weights_.copy()
            
            id_full_adapted = adapted_model.score_samples(X_id_test)
            ood_full_adapted = adapted_model.score_samples(X_ood_test)
            auc = roc_auc_score(y_true, np.concatenate([id_full_adapted, ood_full_adapted]))
            results[beta].append(auc)
            
    print(f"{'Beta':<10s} | {'Estimated Noise Var':<20s} | {'Mean OOD Rejection AUC':<25s}")
    print("-" * 65)
    for beta in beta_values:
        mean_auc = np.mean(results[beta])
        std_auc = np.std(results[beta])
        print(f"{beta:<10.1f} | {beta*noise_var:<20.4f} | {mean_auc:.4f} ± {std_auc:.4f}")

if __name__ == "__main__":
    evaluate_noise_sensitivity()
