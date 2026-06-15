import os
import numpy as np
from sklearn.linear_model import LogisticRegression

D = 192
L = 14
K = 4
sigmas = [0.05, 0.15, 0.40, 1.20]
ceilings = np.array([1.0, 1.0, 0.924, 0.228])
targets_un = np.array([0.985, 0.850, 0.450, 0.141])
gamma = 0.35

def cos_sim(A, B):
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    dot = np.dot(A, B.T)
    return dot / (norm_A * norm_B.T + 1e-9)

def generate_data(seed, num_samples, rho=0.0):
    np.random.seed(seed)
    v_orth = np.zeros((K, D))
    for k in range(K):
        block = np.random.normal(size=48)
        v_orth[k, 48*k:48*(k+1)] = block / np.linalg.norm(block)
    v = np.zeros((K, D))
    v_shared = np.random.normal(size=D)
    v_shared /= np.linalg.norm(v_shared)
    for k in range(K):
        v[k] = np.sqrt(1.0 - rho) * v_orth[k] + np.sqrt(rho) * v_shared
        v[k] /= np.linalg.norm(v[k])
    X_cal, y_cal = [], []
    for k in range(K):
        for _ in range(64):
            noise = np.random.normal(0, sigmas[k], size=D)
            h0 = v[k] + noise
            h0 /= np.linalg.norm(h0)
            X_cal.append(h0)
            y_cal.append(k)
    X_test, y_test = [], []
    for k in range(K):
        for _ in range(num_samples):
            noise = np.random.normal(0, sigmas[k], size=D)
            h0 = v[k] + noise
            h0 /= np.linalg.norm(h0)
            X_test.append(h0)
            y_test.append(k)
    return np.array(X_cal), np.array(y_cal), np.array(X_test), np.array(y_test), v

def run_test(seed, use_reset=True):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=0.0)
    static_centroids = np.zeros((K, D))
    for k in range(K):
        mask = (y_cal == k)
        mean_h = np.mean(X_cal[mask], axis=0)
        static_centroids[k] = mean_h / np.linalg.norm(mean_h)
    centroids = np.zeros((L + 1, K, D))
    for l in range(3, L + 1):
        centroids[l] = static_centroids

    C_init = np.ones((len(X_test), K)) * 0.25
    h = X_test.copy()
    C = C_init.copy()
    p_fail = 0.20
    for l in range(4, L + 1):
        sims = cos_sim(h, centroids[l - 1])
        k_rate = np.exp(sims / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        
        # Check if failure occurs
        np.random.seed(seed * 100 + l)
        fail_mask = np.random.rand(len(h)) < p_fail
        if np.sum(fail_mask) > 0:
            k_rate[fail_mask] = 1.0 / K
            
        # Entropy-gated reset
        if use_reset:
            H = - np.sum(k_rate * np.log(k_rate + 1e-9), axis=1) / np.log(K)
            C[H > 0.99] = 1.0 / K
            
        C_next = C + 1.5 * (k_rate * (1.0 - C) - 0.3 * C)
        C = np.clip(C_next, 0.0, 1.0)
        alpha_lar = C / np.sum(C, axis=1, keepdims=True)
        
        bar_mu = np.dot(alpha_lar, centroids[l - 1])
        sims_mu_centroids = cos_sim(bar_mu, centroids[l - 1])
        sims_h_centroids = cos_sim(h, centroids[l - 1])
        sims_h_bar_mu = np.sum(h * bar_mu, axis=1, keepdims=True) / (np.linalg.norm(h, axis=1, keepdims=True) * np.linalg.norm(bar_mu, axis=1, keepdims=True) + 1e-9)
        
        A = np.sum(C * (sims_mu_centroids - sims_h_centroids * sims_h_bar_mu), axis=1)
        eta = np.zeros(len(h))
        mask = (A > 0.04)
        eta[mask] = np.minimum(0.15, 1.0 * A[mask])
        
        h_warped = h + eta[:, np.newaxis] * (bar_mu - h)
        h_warped = h_warped / np.linalg.norm(h_warped, axis=1, keepdims=True)
        
        expert_update = np.zeros_like(h_warped)
        for k in range(K):
            expert_update += alpha_lar[:, k:k+1] * (v[k] - h_warped)
        h = h_warped + gamma * expert_update
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        
    # Accuracy & similarity
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

accs_no_reset, sims_no_reset = [], []
accs_reset, sims_reset = [], []
for seed in range(10):
    acc, sim = run_test(seed, use_reset=False)
    accs_no_reset.append(acc)
    sims_no_reset.append(sim)
    
    acc, sim = run_test(seed, use_reset=True)
    accs_reset.append(acc)
    sims_reset.append(sim)

print(f"No Reset: Joint Mean Acc: {np.mean(accs_no_reset)*100:.2f}% ± {np.std(accs_no_reset)*100:.2f}%, Similarity: {np.mean(sims_no_reset):.4f} ± {np.std(sims_no_reset):.4f}")
print(f"With Reset: Joint Mean Acc: {np.mean(accs_reset)*100:.2f}% ± {np.std(accs_reset)*100:.2f}%, Similarity: {np.mean(sims_reset):.4f} ± {np.std(sims_reset):.4f}")
