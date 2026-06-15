import os
import math
import time
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

# Reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Global Constants
L = 14
gamma = 0.12

# --- EXPERIMENT 1: Expert Population Scaling ---
def run_scaling_experiment():
    print("--- EXPERIMENT 1: EXPERT POPULATION SCALING ---")
    seeds = range(42, 47) # 5 seeds for stability and speed
    K_list = [4, 8, 12, 16, 20, 24]
    
    # Base configuration templates
    base_sigmas = [0.05, 0.15, 0.40, 1.20]
    base_nu_vals = [0.01, 0.01, 0.36, 3.74]
    base_biases = [0.0, 0.0, 0.0, -1.50]
    
    results = {}
    
    for K_val in K_list:
        subspace_dim = 48
        D_val = 1152 # unified max dimension to support up to 24 orthogonal tasks
        L = 14
        gamma = 0.12
        
        # Build orthogonal task signatures
        v_val = torch.zeros(K_val, D_val)
        for k in range(K_val):
            v_val[k, k*subspace_dim : (k+1)*subspace_dim] = 1.0 / math.sqrt(subspace_dim)
            
        sigmas_val = [base_sigmas[i % 4] for i in range(K_val)]
        nu_vals_val = [base_nu_vals[i % 4] for i in range(K_val)]
        biases_val = [base_biases[i % 4] for i in range(K_val)]
        
        seed_accs = []
        seed_latencies = []
        seed_ood_rejs = []
        
        for seed in seeds:
            set_seed(seed)
            
            # Calibration data (64 samples per task)
            cal_data = []
            for k in range(K_val):
                eps = torch.randn(64, D_val) * sigmas_val[k]
                h_0 = v_val[k].unsqueeze(0) + eps
                cal_data.append(h_0)
                
            # Extract centroids and IDC scales
            centroids = torch.zeros(K_val, D_val)
            s_k = torch.zeros(K_val)
            for k in range(K_val):
                h_3_cal = cal_data[k]
                centroids[k] = h_3_cal.mean(dim=0)
                norm_h = torch.norm(h_3_cal, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[k], p=2).clamp(min=1e-8)
                similarities = torch.matmul(h_3_cal, centroids[k].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                s_k[k] = similarities.mean().item()
                
            # GMM OOD Fitting
            gmms = []
            for k in range(K_val):
                # get coordinates
                coords_k = torch.zeros(64, K_val)
                h_cal_k = cal_data[k]
                for j in range(K_val):
                    norm_h = torch.norm(h_cal_k, p=2, dim=1).clamp(min=1e-8)
                    norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                    coords_k[:, j] = torch.matmul(h_cal_k, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                
                gmm_k = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
                gmm_k.fit(coords_k.numpy())
                gmms.append(gmm_k)
                
            # Calibrate threshold on joint calibration split (5% FPR)
            all_cal_h3 = torch.cat(cal_data, dim=0)
            all_cal_coords = torch.zeros(all_cal_h3.shape[0], K_val)
            for j in range(K_val):
                norm_h = torch.norm(all_cal_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                all_cal_coords[:, j] = torch.matmul(all_cal_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                
            all_cal_scores = np.zeros((all_cal_coords.shape[0], K_val))
            for k in range(K_val):
                all_cal_scores[:, k] = gmms[k].score_samples(all_cal_coords.numpy())
            ood_threshold = np.percentile(all_cal_scores.max(axis=1), 5)
            
            # Generate test set (250 samples per task)
            test_data = []
            for k in range(K_val):
                eps = torch.randn(250, D_val) * sigmas_val[k]
                test_data.append(v_val[k].unsqueeze(0) + eps)
            all_test_h3 = torch.cat(test_data, dim=0)
            
            # Generate OOD test dataset
            ood_test_h3 = torch.randn(250, D_val) * 1.5
            
            # Evaluate OOD
            ood_coords = torch.zeros(250, K_val)
            for j in range(K_val):
                norm_h = torch.norm(ood_test_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                ood_coords[:, j] = torch.matmul(ood_test_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            all_ood_scores = np.zeros((250, K_val))
            for k in range(K_val):
                all_ood_scores[:, k] = gmms[k].score_samples(ood_coords.numpy())
            ood_rejection_rate = (all_ood_scores.max(axis=1) < ood_threshold).mean()
            seed_ood_rejs.append(ood_rejection_rate)
            
            # Evaluate routing latency and accuracy under C_budget = 0.4
            c_budget = 0.4
            M_val = max(1, int(K_val * c_budget))
            theta_min = 0.001
            theta_max = 0.20
            theta_val = theta_min + (1.0 - c_budget) * (theta_max - theta_min)
            
            # Time the router
            t_start = time.perf_counter()
            test_coords = torch.zeros(all_test_h3.shape[0], K_val)
            for j in range(K_val):
                norm_h = torch.norm(all_test_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                test_coords[:, j] = torch.matmul(all_test_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            all_test_scores = np.zeros((all_test_h3.shape[0], K_val))
            for k in range(K_val):
                all_test_scores[:, k] = gmms[k].score_samples(test_coords.numpy())
            max_test_scores = all_test_scores.max(axis=1)
            is_id = torch.tensor(max_test_scores >= ood_threshold, dtype=torch.float32)
            
            # Softmax & pruning
            u_prime = test_coords / s_k.unsqueeze(0)
            alpha_hat = torch.softmax(u_prime / 0.05, dim=1)
            
            alpha_final = torch.zeros_like(alpha_hat)
            for b in range(all_test_h3.shape[0]):
                coeffs = alpha_hat[b].clone()
                top_m_vals, top_m_indices = torch.topk(coeffs, k=M_val)
                masked_coeffs = torch.zeros_like(coeffs)
                masked_coeffs[top_m_indices] = coeffs[top_m_indices]
                sum_top = masked_coeffs.sum()
                if sum_top > 0:
                    masked_coeffs = masked_coeffs / sum_top
                coeffs = masked_coeffs
                pruned_coeffs = torch.where(coeffs >= theta_val, coeffs, torch.zeros_like(coeffs))
                sum_pruned = pruned_coeffs.sum()
                if sum_pruned > 0:
                    pruned_coeffs = pruned_coeffs / sum_pruned
                alpha_final[b] = pruned_coeffs
                
            rb_alpha = alpha_final * is_id.unsqueeze(1)
            router_latency_ms = (time.perf_counter() - t_start) * 1000.0 / all_test_h3.shape[0]
            seed_latencies.append(router_latency_ms)
            
            # Propagate layers
            h = all_test_h3.clone()
            for l in range(4, L + 1):
                update = torch.zeros_like(h)
                for k in range(K_val):
                    diff = v_val[k].unsqueeze(0) - h
                    update += rb_alpha[:, k:k+1] * gamma * diff
                h = h + update
                
            # Accuracy computation
            task_accs = []
            for k in range(K_val):
                start = k * 250
                end = (k + 1) * 250
                h_L_k = h[start:end]
                num_s = h_L_k.shape[0]
                logits = torch.zeros(num_s, K_val)
                for j in range(K_val):
                    proj = torch.matmul(h_L_k, v_val[j])
                    logit_noise = torch.randn(num_s) * nu_vals_val[k]
                    logits[:, j] = proj + logit_noise + biases_val[k] if j == k else proj + logit_noise
                preds = torch.argmax(logits, dim=1)
                acc = (preds == k).float().mean().item()
                task_accs.append(acc)
            seed_accs.append(np.mean(task_accs))
            
        print(f"K = {K_val:2d}: Joint Accuracy = {np.mean(seed_accs)*100:.2f}% +- {np.std(seed_accs)*100:.2f}%, Router Latency = {np.mean(seed_latencies)*1000.0:.3f} us/sample, OOD Rejection = {np.mean(seed_ood_rejs)*100:.2f}%")
        results[K_val] = {
            'accuracy': np.mean(seed_accs)*100,
            'accuracy_std': np.std(seed_accs)*100,
            'latency_us': np.mean(seed_latencies)*1000.0,
            'ood_rejection': np.mean(seed_ood_rejs)*100
        }

# --- EXPERIMENT 2: GMM Components Sweep ---
def run_gmm_components_experiment():
    print("\n--- EXPERIMENT 2: GMM COMPONENTS SENSITIVITY SWEEP ---")
    seeds = range(42, 52) # 10 seeds
    K_val = 4
    subspace_dim = 48
    D_val = 192
    
    base_sigmas = [0.05, 0.15, 0.40, 1.20]
    v_val = torch.zeros(K_val, D_val)
    for k in range(K_val):
        v_val[k, k*subspace_dim : (k+1)*subspace_dim] = 1.0 / math.sqrt(subspace_dim)
        
    for C_components in [1, 2, 3]:
        seed_ood_rejs = []
        seed_test_fprs = []
        seed_latencies = []
        
        for seed in seeds:
            set_seed(seed)
            
            # Calibration data (64 samples per task)
            cal_data = []
            for k in range(K_val):
                eps = torch.randn(64, D_val) * base_sigmas[k]
                h_0 = v_val[k].unsqueeze(0) + eps
                cal_data.append(h_0)
                
            # Extract centroids
            centroids = torch.zeros(K_val, D_val)
            for k in range(K_val):
                centroids[k] = cal_data[k].mean(dim=0)
                
            # GMM OOD Fitting
            gmms = []
            for k in range(K_val):
                coords_k = torch.zeros(64, K_val)
                h_cal_k = cal_data[k]
                for j in range(K_val):
                    norm_h = torch.norm(h_cal_k, p=2, dim=1).clamp(min=1e-8)
                    norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                    coords_k[:, j] = torch.matmul(h_cal_k, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                
                gmm_k = GaussianMixture(n_components=C_components, covariance_type='diag', random_state=seed, reg_covar=1e-5)
                gmm_k.fit(coords_k.numpy())
                gmms.append(gmm_k)
                
            # Calibrate threshold on joint calibration split (5% FPR)
            all_cal_h3 = torch.cat(cal_data, dim=0)
            all_cal_coords = torch.zeros(all_cal_h3.shape[0], K_val)
            for j in range(K_val):
                norm_h = torch.norm(all_cal_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                all_cal_coords[:, j] = torch.matmul(all_cal_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                
            all_cal_scores = np.zeros((all_cal_coords.shape[0], K_val))
            for k in range(K_val):
                all_cal_scores[:, k] = gmms[k].score_samples(all_cal_coords.numpy())
            ood_threshold = np.percentile(all_cal_scores.max(axis=1), 5)
            
            # Generate test set (250 samples per task)
            test_data = []
            for k in range(K_val):
                eps = torch.randn(250, D_val) * base_sigmas[k]
                test_data.append(v_val[k].unsqueeze(0) + eps)
            all_test_h3 = torch.cat(test_data, dim=0)
            
            # Generate OOD test dataset
            ood_test_h3 = torch.randn(250, D_val) * 1.5
            
            # Measure evaluation latency per sample
            t_start = time.perf_counter()
            test_coords = torch.zeros(all_test_h3.shape[0], K_val)
            for j in range(K_val):
                norm_h = torch.norm(all_test_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                test_coords[:, j] = torch.matmul(all_test_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            all_test_scores = np.zeros((all_test_h3.shape[0], K_val))
            for k in range(K_val):
                all_test_scores[:, k] = gmms[k].score_samples(test_coords.numpy())
            _ = all_test_scores.max(axis=1) < ood_threshold
            lat_us = (time.perf_counter() - t_start) * 1e6 / all_test_h3.shape[0]
            seed_latencies.append(lat_us)
            
            # Test-set FPR (in-distribution samples flagged as OOD)
            test_scores = np.zeros((all_test_h3.shape[0], K_val))
            for k in range(K_val):
                test_scores[:, k] = gmms[k].score_samples(test_coords.numpy())
            test_fpr = (test_scores.max(axis=1) < ood_threshold).mean()
            seed_test_fprs.append(test_fpr)
            
            # Evaluate OOD rejection
            ood_coords = torch.zeros(250, K_val)
            for j in range(K_val):
                norm_h = torch.norm(ood_test_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                ood_coords[:, j] = torch.matmul(ood_test_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            all_ood_scores = np.zeros((250, K_val))
            for k in range(K_val):
                all_ood_scores[:, k] = gmms[k].score_samples(ood_coords.numpy())
            ood_rejection_rate = (all_ood_scores.max(axis=1) < ood_threshold).mean()
            seed_ood_rejs.append(ood_rejection_rate)
            
        print(f"Components = {C_components}: OOD Rejection = {np.mean(seed_ood_rejs)*100:.2f}% +- {np.std(seed_ood_rejs)*100:.2f}%, Test-set FPR = {np.mean(seed_test_fprs)*100:.2f}% +- {np.std(seed_test_fprs)*100:.2f}%, Eval Latency = {np.mean(seed_latencies):.3f} us/sample")

# --- EXPERIMENT 3: Regularized GMM Calibration (N=256, 5-fold CV) vs. Baseline (N=64) ---
def run_gmm_regularization_experiment():
    print("\n--- EXPERIMENT 3: GMM CALIBRATION GENERALIZATION ANALYSIS ---")
    seeds = range(42, 52) # 10 seeds
    K_val = 4
    subspace_dim = 48
    D_val = 192
    
    base_sigmas = [0.05, 0.15, 0.40, 1.20]
    base_nu_vals = [0.01, 0.01, 0.36, 3.74]
    base_biases = [0.0, 0.0, 0.0, -1.50]
    v_val = torch.zeros(K_val, D_val)
    for k in range(K_val):
        v_val[k, k*subspace_dim : (k+1)*subspace_dim] = 1.0 / math.sqrt(subspace_dim)
        
    for config_name, N_cal, use_cv in [("Baseline", 64, False), ("Regularized", 256, True)]:
        seed_test_fprs = []
        seed_ood_rejs = []
        seed_joint_accs_cb04 = []
        seed_joint_accs_cb00 = []
        
        for seed in seeds:
            set_seed(seed)
            
            # Generate calibration split
            cal_data = []
            for k in range(K_val):
                eps = torch.randn(N_cal, D_val) * base_sigmas[k]
                h_0 = v_val[k].unsqueeze(0) + eps
                cal_data.append(h_0)
                
            # Centroids
            centroids = torch.zeros(K_val, D_val)
            s_k = torch.zeros(K_val)
            for k in range(K_val):
                h_3_cal = cal_data[k]
                centroids[k] = h_3_cal.mean(dim=0)
                norm_h = torch.norm(h_3_cal, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[k], p=2).clamp(min=1e-8)
                similarities = torch.matmul(h_3_cal, centroids[k].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                s_k[k] = similarities.mean().item()
                
            # Fit GMM
            gmms = []
            for k in range(K_val):
                coords_k = torch.zeros(N_cal, K_val)
                h_cal_k = cal_data[k]
                for j in range(K_val):
                    norm_h = torch.norm(h_cal_k, p=2, dim=1).clamp(min=1e-8)
                    norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                    coords_k[:, j] = torch.matmul(h_cal_k, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                
                gmm_k = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
                gmm_k.fit(coords_k.numpy())
                gmms.append(gmm_k)
                
            # Threshold calibration
            all_cal_h3 = torch.cat(cal_data, dim=0)
            all_cal_coords = torch.zeros(all_cal_h3.shape[0], K_val)
            for j in range(K_val):
                norm_h = torch.norm(all_cal_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                all_cal_coords[:, j] = torch.matmul(all_cal_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                
            if not use_cv:
                # Standard threshold percentile
                all_cal_scores = np.zeros((all_cal_coords.shape[0], K_val))
                for k in range(K_val):
                    all_cal_scores[:, k] = gmms[k].score_samples(all_cal_coords.numpy())
                ood_threshold = np.percentile(all_cal_scores.max(axis=1), 5)
            else:
                # 5-fold CV threshold calibration
                # Divide each task's calibration samples into 5 folds
                cv_scores = []
                for fold in range(5):
                    train_coords_list = []
                    val_coords_list = []
                    for k in range(K_val):
                        fold_size = N_cal // 5
                        val_start = fold * fold_size
                        val_end = (fold + 1) * fold_size
                        
                        h_val_k = cal_data[k][val_start:val_end]
                        h_train_k = torch.cat([cal_data[k][:val_start], cal_data[k][val_end:]], dim=0)
                        
                        # Project train coords
                        coords_train_k = torch.zeros(h_train_k.shape[0], K_val)
                        for j in range(K_val):
                            norm_h = torch.norm(h_train_k, p=2, dim=1).clamp(min=1e-8)
                            norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                            coords_train_k[:, j] = torch.matmul(h_train_k, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                        
                        # Project val coords
                        coords_val_k = torch.zeros(h_val_k.shape[0], K_val)
                        for j in range(K_val):
                            norm_h = torch.norm(h_val_k, p=2, dim=1).clamp(min=1e-8)
                            norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                            coords_val_k[:, j] = torch.matmul(h_val_k, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
                            
                        val_coords_list.append(coords_val_k)
                        train_coords_list.append(coords_train_k)
                        
                    # Fit GMMs on train splits
                    gmms_fold = []
                    for k in range(K_val):
                        gmm_kf = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
                        gmm_kf.fit(train_coords_list[k].numpy())
                        gmms_fold.append(gmm_kf)
                        
                    # Score val splits
                    all_val_coords = torch.cat(val_coords_list, dim=0)
                    all_val_scores = np.zeros((all_val_coords.shape[0], K_val))
                    for k in range(K_val):
                        all_val_scores[:, k] = gmms_fold[k].score_samples(all_val_coords.numpy())
                    cv_scores.extend(all_val_scores.max(axis=1))
                
                ood_threshold = np.percentile(cv_scores, 5)
                
            # Generate test set (250 samples per task)
            test_data = []
            for k in range(K_val):
                eps = torch.randn(250, D_val) * base_sigmas[k]
                test_data.append(v_val[k].unsqueeze(0) + eps)
            all_test_h3 = torch.cat(test_data, dim=0)
            
            # Generate OOD test dataset
            ood_test_h3 = torch.randn(250, D_val) * 1.5
            
            # OOD Rejection rate on OOD test
            ood_coords = torch.zeros(250, K_val)
            for j in range(K_val):
                norm_h = torch.norm(ood_test_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                ood_coords[:, j] = torch.matmul(ood_test_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            all_ood_scores = np.zeros((250, K_val))
            for k in range(K_val):
                all_ood_scores[:, k] = gmms[k].score_samples(ood_coords.numpy())
            ood_rejection_rate = (all_ood_scores.max(axis=1) < ood_threshold).mean()
            seed_ood_rejs.append(ood_rejection_rate)
            
            # Test-set FPR
            test_coords = torch.zeros(all_test_h3.shape[0], K_val)
            for j in range(K_val):
                norm_h = torch.norm(all_test_h3, p=2, dim=1).clamp(min=1e-8)
                norm_c = torch.norm(centroids[j], p=2).clamp(min=1e-8)
                test_coords[:, j] = torch.matmul(all_test_h3, centroids[j].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            all_test_scores = np.zeros((all_test_h3.shape[0], K_val))
            for k in range(K_val):
                all_test_scores[:, k] = gmms[k].score_samples(test_coords.numpy())
            test_fpr = (all_test_scores.max(axis=1) < ood_threshold).mean()
            seed_test_fprs.append(test_fpr)
            
            # Downstream joint ensembling accuracy at C_budget = 0.4 and 0.0
            for cb, acc_list in [(0.4, seed_joint_accs_cb04), (0.0, seed_joint_accs_cb00)]:
                M_val = max(1, int(K_val * cb))
                theta_min = 0.001
                theta_max = 0.20
                theta_val = theta_min + (1.0 - cb) * (theta_max - theta_min)
                
                is_id = torch.tensor(all_test_scores.max(axis=1) >= ood_threshold, dtype=torch.float32)
                
                # Softmax routing
                u_prime = test_coords / s_k.unsqueeze(0)
                alpha_hat = torch.softmax(u_prime / 0.05, dim=1)
                
                alpha_final = torch.zeros_like(alpha_hat)
                for b in range(all_test_h3.shape[0]):
                    coeffs = alpha_hat[b].clone()
                    top_m_vals, top_m_indices = torch.topk(coeffs, k=M_val)
                    masked_coeffs = torch.zeros_like(coeffs)
                    masked_coeffs[top_m_indices] = coeffs[top_m_indices]
                    sum_top = masked_coeffs.sum()
                    if sum_top > 0:
                        masked_coeffs = masked_coeffs / sum_top
                    coeffs = masked_coeffs
                    pruned_coeffs = torch.where(coeffs >= theta_val, coeffs, torch.zeros_like(coeffs))
                    sum_pruned = pruned_coeffs.sum()
                    if sum_pruned > 0:
                        pruned_coeffs = pruned_coeffs / sum_pruned
                    alpha_final[b] = pruned_coeffs
                    
                rb_alpha = alpha_final * is_id.unsqueeze(1)
                
                # Propagate layers
                h = all_test_h3.clone()
                for l in range(4, L + 1):
                    update = torch.zeros_like(h)
                    for k in range(K_val):
                        diff = v_val[k].unsqueeze(0) - h
                        update += rb_alpha[:, k:k+1] * gamma * diff
                    h = h + update
                    
                # Accuracy computation
                task_accs = []
                for k in range(K_val):
                    start = k * 250
                    end = (k + 1) * 250
                    h_L_k = h[start:end]
                    num_s = h_L_k.shape[0]
                    logits = torch.zeros(num_s, K_val)
                    for j in range(K_val):
                        proj = torch.matmul(h_L_k, v_val[j])
                        logit_noise = torch.randn(num_s) * base_nu_vals[k]
                        logits[:, j] = proj + logit_noise + base_biases[k] if j == k else proj + logit_noise
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == k).float().mean().item()
                    task_accs.append(acc)
                acc_list.append(np.mean(task_accs))
                
        print(f"Configuration: {config_name}")
        print(f"  Test-set FPR = {np.mean(seed_test_fprs)*100:.2f}% +- {np.std(seed_test_fprs)*100:.2f}%")
        print(f"  OOD Rejection Rate = {np.mean(seed_ood_rejs)*100:.2f}% +- {np.std(seed_ood_rejs)*100:.2f}%")
        print(f"  Accuracy ($C_{{budget}} = 0.4$) = {np.mean(seed_joint_accs_cb04)*100:.2f}%")
        print(f"  Accuracy ($C_{{budget}} = 0.0$) = {np.mean(seed_joint_accs_cb00)*100:.2f}%")

if __name__ == "__main__":
    run_scaling_experiment()
    run_gmm_components_experiment()
    run_gmm_regularization_experiment()
