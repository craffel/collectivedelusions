import os
import math
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Dimensions and Sandbox configuration
D = 192
K = 4
subspace_dim = 48
L = 14
gamma = 0.12
shrink = (1.0 - gamma) ** 11

tasks = ["MNIST", "Fashion-MNIST", "CIFAR-10", "SVHN"]
sigmas = [0.05, 0.15, 0.40, 1.20]
nu_vals = [0.01, 0.01, 0.36, 3.74]
biases = [0.0, 0.0, 0.0, -1.50]

v = torch.zeros(K, D)
for k in range(K):
    v[k, k*subspace_dim : (k+1)*subspace_dim] = 1.0 / math.sqrt(subspace_dim)

def generate_samples(task_idx, num_samples, sigma):
    eps = torch.randn(num_samples, D) * sigma
    h_0 = v[task_idx].unsqueeze(0) + eps
    return h_0

def generate_ood_samples(num_samples):
    return torch.randn(num_samples, D) * 1.5

def propagate_layers(h_3, alpha):
    h = h_3.clone()
    for l in range(4, L + 1):
        update = torch.zeros_like(h)
        for k in range(K):
            diff = v[k].unsqueeze(0) - h
            update += alpha[:, k:k+1] * gamma * diff
        h = h + update
    return h

def compute_logits_and_acc(h_L, true_task_idx, nu, bias):
    num_samples = h_L.shape[0]
    logits = torch.zeros(num_samples, K)
    for j in range(K):
        proj = torch.matmul(h_L, v[j])
        logit_noise = torch.randn(num_samples) * nu
        logits[:, j] = proj + logit_noise + bias if j == true_task_idx else proj + logit_noise
    preds = torch.argmax(logits, dim=1)
    acc = (preds == true_task_idx).float().mean().item()
    return preds, acc

def get_coordinates(h_3, centroids):
    B = h_3.shape[0]
    coords = torch.zeros(B, K)
    for k in range(K):
        norm_h = torch.norm(h_3, p=2, dim=1).clamp(min=1e-8)
        norm_c = torch.norm(centroids[k], p=2).clamp(min=1e-8)
        coords[:, k] = torch.matmul(h_3, centroids[k].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
    return coords

def compute_routing_coefficients_realistic(coords, s_k, tau, M=None, theta=None):
    B = coords.shape[0]
    u_prime = coords / s_k.unsqueeze(0)
    alpha_hat = torch.softmax(u_prime / tau, dim=1)
    
    if M is None and theta is None:
        return alpha_hat, torch.full((B,), K, dtype=torch.float32)
        
    alpha_final = torch.zeros_like(alpha_hat)
    active_experts = torch.zeros(B)
    
    for b in range(B):
        coeffs = alpha_hat[b].clone()
        if M is not None:
            top_m_vals, top_m_indices = torch.topk(coeffs, k=M)
            masked_coeffs = torch.zeros_like(coeffs)
            masked_coeffs[top_m_indices] = coeffs[top_m_indices]
            sum_top = masked_coeffs.sum()
            if sum_top > 0:
                masked_coeffs = masked_coeffs / sum_top
            coeffs = masked_coeffs
            
        if theta is not None:
            pruned_coeffs = torch.where(coeffs >= theta, coeffs, torch.zeros_like(coeffs))
            sum_pruned = pruned_coeffs.sum()
            if sum_pruned > 0:
                pruned_coeffs = pruned_coeffs / sum_pruned
            coeffs = pruned_coeffs
            
        alpha_final[b] = coeffs
        active_experts[b] = (coeffs > 0).float().sum()
        
    return alpha_final, active_experts

def run_evaluation_realistic(N_cal, use_cv):
    seeds = range(42, 52)
    
    methods = ['Oracle', 'Uniform', 'SABLE', 'SPS-ZCA', 'Q-SPS']
    budgets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for b in budgets:
        methods.append(f'RB-TopM_C{b:.1f}')
        
    # seed -> method -> (task_accuracies list, active_experts mean)
    runs = {m: [] for m in methods}
    ood_rejections = []
    test_fprs = []
    
    for seed in seeds:
        set_seed(seed)
        
        # 1. Calibration set
        cal_data = []
        for k in range(K):
            cal_data.append(generate_samples(k, N_cal, sigmas[k]))
            
        # 2. Extract centroids & s_k
        centroids = torch.zeros(K, D)
        s_k = torch.zeros(K)
        for k in range(K):
            h_3_cal = cal_data[k]
            centroids[k] = h_3_cal.mean(dim=0)
            norm_h = torch.norm(h_3_cal, p=2, dim=1).clamp(min=1e-8)
            norm_c = torch.norm(centroids[k], p=2).clamp(min=1e-8)
            similarities = torch.matmul(h_3_cal, centroids[k].unsqueeze(1)).squeeze(1) / (norm_h * norm_c)
            s_k[k] = similarities.mean().item()
            
        # 3. Fit GMMs
        gmms = []
        for k in range(K):
            coords_k = get_coordinates(cal_data[k], centroids)
            gmm_k = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
            gmm_k.fit(coords_k.numpy())
            gmms.append(gmm_k)
            
        # 4. Calibrate OOD threshold
        all_cal_h3 = torch.cat(cal_data, dim=0)
        all_cal_coords = get_coordinates(all_cal_h3, centroids)
        
        if not use_cv:
            all_cal_scores = np.zeros((all_cal_coords.shape[0], K))
            for k in range(K):
                all_cal_scores[:, k] = gmms[k].score_samples(all_cal_coords.numpy())
            ood_threshold = np.percentile(all_cal_scores.max(axis=1), 5)
        else:
            cv_scores = []
            for fold in range(5):
                train_coords_list = []
                val_coords_list = []
                for k in range(K):
                    fold_size = N_cal // 5
                    val_start = fold * fold_size
                    val_end = (fold + 1) * fold_size
                    
                    h_val_k = cal_data[k][val_start:val_end]
                    h_train_k = torch.cat([cal_data[k][:val_start], cal_data[k][val_end:]], dim=0)
                    
                    coords_train_k = get_coordinates(h_train_k, centroids)
                    coords_val_k = get_coordinates(h_val_k, centroids)
                        
                    val_coords_list.append(coords_val_k)
                    train_coords_list.append(coords_train_k)
                    
                gmms_fold = []
                for k in range(K):
                    gmm_kf = GaussianMixture(n_components=2, covariance_type='diag', random_state=seed, reg_covar=1e-5)
                    gmm_kf.fit(train_coords_list[k].numpy())
                    gmms_fold.append(gmm_kf)
                    
                all_val_coords = torch.cat(val_coords_list, dim=0)
                all_val_scores = np.zeros((all_val_coords.shape[0], K))
                for k in range(K):
                    all_val_scores[:, k] = gmms_fold[k].score_samples(all_val_coords.numpy())
                cv_scores.extend(all_val_scores.max(axis=1))
            
            ood_threshold = np.percentile(cv_scores, 5)
            
        # 5. Generate Test dataset
        test_data = []
        for k in range(K):
            test_data.append(generate_samples(k, 250, sigmas[k]))
        all_test_h3 = torch.cat(test_data, dim=0)
        
        # 6. OOD test rejection
        ood_test_h3 = generate_ood_samples(250)
        ood_coords = get_coordinates(ood_test_h3, centroids)
        all_ood_scores = np.zeros((ood_coords.shape[0], K))
        for k in range(K):
            all_ood_scores[:, k] = gmms[k].score_samples(ood_coords.numpy())
        ood_rejection_rate = (all_ood_scores.max(axis=1) < ood_threshold).mean()
        ood_rejections.append(ood_rejection_rate)
        
        # 7. Test FPR
        test_coords = get_coordinates(all_test_h3, centroids)
        all_test_scores = np.zeros((test_coords.shape[0], K))
        for k in range(K):
            all_test_scores[:, k] = gmms[k].score_samples(test_coords.numpy())
        test_fpr = (all_test_scores.max(axis=1) < ood_threshold).mean()
        test_fprs.append(test_fpr)
        
        is_id = torch.tensor(all_test_scores.max(axis=1) >= ood_threshold, dtype=torch.float32)
        
        # Evaluation helper
        def evaluate_alpha(alpha, active_exps):
            h_L = propagate_layers(all_test_h3, alpha)
            task_accs = []
            for k in range(K):
                start = k * 250
                end = (k + 1) * 250
                _, acc = compute_logits_and_acc(h_L[start:end], k, nu_vals[k], biases[k])
                task_accs.append(acc)
            return task_accs, active_exps.mean().item()
            
        # --- BASELINE 1: Expert Oracle ---
        oracle_alpha = torch.zeros(1000, K)
        for k in range(K):
            oracle_alpha[k*250 : (k+1)*250, k] = 1.0
        runs['Oracle'].append(evaluate_alpha(oracle_alpha, torch.ones(1000)))
        
        # --- BASELINE 2: Uniform Merging ---
        uniform_alpha = torch.full((1000, K), 0.25)
        runs['Uniform'].append(evaluate_alpha(uniform_alpha, torch.full((1000,), 4.0)))
        
        # --- BASELINE 3: SABLE ---
        sable_alpha_raw, sable_active = compute_routing_coefficients_realistic(test_coords, s_k, tau=0.05)
        sable_alpha = sable_alpha_raw * is_id.unsqueeze(1)
        runs['SABLE'].append(evaluate_alpha(sable_alpha, sable_active * is_id))
        
        # --- BASELINE 4: SPS-ZCA ---
        sps_alpha_raw, sps_active = compute_routing_coefficients_realistic(test_coords, s_k, tau=0.001)
        sps_alpha = sps_alpha_raw * is_id.unsqueeze(1)
        runs['SPS-ZCA'].append(evaluate_alpha(sps_alpha, sps_active * is_id))
        
        # --- BASELINE 5: Q-SPS ---
        qsps_alpha_raw, qsps_active = compute_routing_coefficients_realistic(test_coords, s_k, tau=0.001, M=2, theta=0.01)
        qsps_alpha = qsps_alpha_raw * is_id.unsqueeze(1)
        runs['Q-SPS'].append(evaluate_alpha(qsps_alpha, qsps_active * is_id))
        
        # --- RB-TopM Sweep ---
        for cb in budgets:
            M_val = max(1, int(4 * cb))
            theta_min = 0.001
            theta_max = 0.20
            theta_val = theta_min + (1.0 - cb) * (theta_max - theta_min)
            
            rb_alpha_raw, rb_active = compute_routing_coefficients_realistic(test_coords, s_k, tau=0.05, M=M_val, theta=theta_val)
            rb_alpha = rb_alpha_raw * is_id.unsqueeze(1)
            runs[f'RB-TopM_C{cb:.1f}'].append(evaluate_alpha(rb_alpha, rb_active * is_id))
            
    print(f"\n--- CONFIG SUMMARY (N_cal={N_cal}, use_cv={use_cv}) ---")
    print(f"OOD Rejection Rate: {np.mean(ood_rejections)*100:.2f}% +- {np.std(ood_rejections)*100:.2f}%")
    print(f"Test-set FPR:       {np.mean(test_fprs)*100:.2f}% +- {np.std(test_fprs)*100:.2f}%")
    
    print("-" * 105)
    print(f"{'Method':<20} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'SVHN':<8} | {'Joint Mean':<15} | {'Active Exps':<12}")
    print("-" * 105)
    for m in methods:
        task_accs_list = np.array([r[0] for r in runs[m]])
        active_exps_list = np.array([r[1] for r in runs[m]])
        
        mean_tasks = task_accs_list.mean(axis=0)
        joint_means = task_accs_list.mean(axis=1)
        mean_joint = joint_means.mean()
        std_joint = joint_means.std()
        
        mean_active = active_exps_list.mean()
        
        task_str = " | ".join([f"{mean_tasks[k]*100:6.2f}%" for k in range(K)])
        print(f"{m:<20} | {task_str} | {mean_joint*100:6.2f}% +- {std_joint*100:4.2f}% | {mean_active:.3f}")
    print("-" * 105)

if __name__ == "__main__":
    # Run unregularized baseline (N_cal=64, use_cv=False)
    run_evaluation_realistic(64, False)
    # Run regularized calibration (N_cal=256, use_cv=True)
    run_evaluation_realistic(256, True)
