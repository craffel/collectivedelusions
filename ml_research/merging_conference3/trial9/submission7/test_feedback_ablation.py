import numpy as np
from run_experiments import generate_data, cos_sim, K, D, L, gamma, targets_un, ceilings

def run_serving_simulation_ablation(seed, p_fail=0.20, use_lyapunov_feedback=True):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=0.0)
    
    # 1. Compute Centroids
    static_centroids = np.zeros((K, D))
    for k in range(K):
        mask = (y_cal == k)
        mean_h = np.mean(X_cal[mask], axis=0)
        static_centroids[k] = mean_h / np.linalg.norm(mean_h)
    centroids = np.zeros((L + 1, K, D))
    for l in range(3, L + 1):
        centroids[l] = static_centroids
        
    C_init = np.ones((len(X_test), K)) * 0.25
        
    # L-ARC (Lyapunov Adaptive Control)
    h = X_test.copy()
    C = C_init.copy()
    for l in range(4, L + 1):
        sims = np.dot(h, centroids[l - 1].T)
        k_rate = np.exp(sims / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        if p_fail > 0.0:
            np.random.seed(seed * 100 + l)
            fail_mask = np.random.rand(len(h)) < p_fail
            if np.sum(fail_mask) > 0:
                k_rate[fail_mask] = 1.0 / K
        # Compute normalized routing entropy to detect transient failures or complete routing confusion
        H = - np.sum(k_rate * np.log(k_rate + 1e-9), axis=1) / np.log(K)
        delta = np.ones(len(h)) * 1.5
        # Entropy-Gated Concentration Gating (ECG-Reset)
        delta[H > 0.95] = 0.0
        
        C_next = C + delta[:, np.newaxis] * (k_rate * (1.0 - C) - 0.3 * C)
        C = np.clip(C_next, 0.0, 1.0)
        alpha_lar = C / np.sum(C, axis=1, keepdims=True)
            
        # Adaptive closed-loop Lyapunov Controller
        bar_mu = np.dot(alpha_lar, centroids[l - 1])
        
        if use_lyapunov_feedback:
            norm_bar = np.linalg.norm(bar_mu, axis=1, keepdims=True)
            bar_mu_norm = bar_mu / (norm_bar + 1e-9)
            sims_mu_centroids = np.dot(bar_mu_norm, centroids[l - 1].T)
            sims_h_centroids = sims
            sims_h_bar_mu = np.einsum('ij,ij->i', h, bar_mu_norm)[:, np.newaxis]
            A = np.sum(C * (sims_mu_centroids - sims_h_centroids * sims_h_bar_mu), axis=1)
            eta = np.zeros(len(h))
            mask = (A > 0.04)
            eta[mask] = np.minimum(0.15, 1.0 * A[mask])
            h_warped = h + eta[:, np.newaxis] * (bar_mu - h)
            h_warped = h_warped / np.linalg.norm(h_warped, axis=1, keepdims=True)
        else:
            h_warped = h.copy()
        
        expert_update = np.zeros_like(h_warped)
        for k in range(K):
            expert_update += alpha_lar[:, k:k+1] * (v[k] - h_warped)
        h = h_warped + gamma * expert_update
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
    
    # Compute downstream classification accuracy and representation similarity
    task_accs = []
    task_sims = []
    for k in range(K):
        mask = (y_test == k)
        alpha_correct = alpha_lar[mask, k]
        p_correct = targets_un[k] + (alpha_correct - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        task_accs.append(np.mean(p_correct))
        s_correct = np.sum(h[mask] * v[k], axis=1)
        task_sims.append(np.mean(s_correct))
        
    return np.mean(task_accs), np.mean(task_sims)

accs_feedback, sims_feedback = [], []
accs_nofeedback, sims_nofeedback = [], []

for seed in range(10):
    acc, sim = run_serving_simulation_ablation(seed, p_fail=0.20, use_lyapunov_feedback=True)
    accs_feedback.append(acc)
    sims_feedback.append(sim)
    
    acc, sim = run_serving_simulation_ablation(seed, p_fail=0.20, use_lyapunov_feedback=False)
    accs_nofeedback.append(acc)
    sims_nofeedback.append(sim)

print(f"Setting C (Transient Failures):")
print(f"  Only ECG-Reset (No Feedback): Acc = {np.mean(accs_nofeedback)*100:.4f}% ± {np.std(accs_nofeedback)*100:.4f}%, Similarity = {np.mean(sims_nofeedback):.5f} ± {np.std(sims_nofeedback):.5f}")
print(f"  Full L-ARC (Feedback + ECG-Reset): Acc = {np.mean(accs_feedback)*100:.4f}% ± {np.std(accs_feedback)*100:.4f}%, Similarity = {np.mean(sims_feedback):.5f} ± {np.std(sims_feedback):.5f}")

# Let us also test under Setting A (Static Centroids, Clean Serving)
accs_feedback_a, sims_feedback_a = [], []
accs_nofeedback_a, sims_nofeedback_a = [], []
for seed in range(10):
    acc, sim = run_serving_simulation_ablation(seed, p_fail=0.0, use_lyapunov_feedback=True)
    accs_feedback_a.append(acc)
    sims_feedback_a.append(sim)
    
    acc, sim = run_serving_simulation_ablation(seed, p_fail=0.0, use_lyapunov_feedback=False)
    accs_nofeedback_a.append(acc)
    sims_nofeedback_a.append(sim)

print(f"\nSetting A (Clean Serving):")
print(f"  Only ECG-Reset (No Feedback): Acc = {np.mean(accs_nofeedback_a)*100:.4f}% ± {np.std(accs_nofeedback_a)*100:.4f}%, Similarity = {np.mean(sims_nofeedback_a):.5f} ± {np.std(sims_nofeedback_a):.5f}")
print(f"  Full L-ARC (Feedback + ECG-Reset): Acc = {np.mean(accs_feedback_a)*100:.4f}% ± {np.std(accs_feedback_a)*100:.4f}%, Similarity = {np.mean(sims_feedback_a):.5f} ± {np.std(sims_feedback_a):.5f}")
