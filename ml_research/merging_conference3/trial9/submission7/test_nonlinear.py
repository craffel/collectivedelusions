import os
import numpy as np

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

def run_test_nonlinear(seed, use_entropy_gating=True, non_linear=True):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=0.0)
    static_centroids = np.zeros((K, D))
    for k in range(K):
        mask = (y_cal == k)
        mean_h = np.mean(X_cal[mask], axis=0)
        static_centroids[k] = mean_h / np.linalg.norm(mean_h)
    centroids = np.zeros((L + 1, K, D))
    for l in range(3, L + 1):
        centroids[l] = static_centroids

    # Create random matrices for non-linear layers to simulate non-linear network transformations
    np.random.seed(seed + 999)
    W = [np.eye(D) + 0.05 * np.random.normal(size=(D, D)) for _ in range(L + 1)]
    # Normalize W to keep scales stable
    for l in range(len(W)):
        W[l] /= np.linalg.norm(W[l], ord=2)

    C_init = np.ones((len(X_test), K)) * 0.25
    
    # 1. Evaluate ChemMerge (No Reset)
    h_cm = X_test.copy()
    C_cm = C_init.copy()
    for l in range(4, L + 1):
        sims = cos_sim(h_cm, centroids[l - 1])
        k_rate = np.exp(sims / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        C_next = C_cm + 1.5 * (k_rate * (1.0 - C_cm) - 0.3 * C_cm)
        C_cm = np.clip(C_next, 0.0, 1.0)
        alpha_cm = C_cm / np.sum(C_cm, axis=1, keepdims=True)
        
        expert_update = np.zeros_like(h_cm)
        for k in range(K):
            expert_update += alpha_cm[:, k:k+1] * (v[k] - h_cm)
        
        # Apply step and non-linearity
        h_cm = h_cm + gamma * expert_update
        if non_linear:
            # Pass through random mapping and activation (tanh)
            h_cm = np.tanh(np.dot(h_cm, W[l].T))
        h_cm = h_cm / np.linalg.norm(h_cm, axis=1, keepdims=True)
        
    # 2. Evaluate L-ARC (With and Without Entropy Gating)
    h_lar = X_test.copy()
    C_lar = C_init.copy()
    for l in range(4, L + 1):
        sims = cos_sim(h_lar, centroids[l - 1])
        k_rate = np.exp(sims / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        
        # Compute normalized entropy
        H = - np.sum(k_rate * np.log(k_rate + 1e-9), axis=1) / np.log(K)
        
        delta = np.ones(len(h_lar)) * 1.5
        if use_entropy_gating:
            delta = delta * ((1.0 - H) ** 2)
            
        C_next = C_lar + delta[:, np.newaxis] * (k_rate * (1.0 - C_lar) - 0.3 * C_lar)
        C_lar = np.clip(C_next, 0.0, 1.0)
        alpha_lar = C_lar / np.sum(C_lar, axis=1, keepdims=True)
        
        # Adaptive closed-loop Lyapunov Controller
        bar_mu = np.dot(alpha_lar, centroids[l - 1])
        sims_mu_centroids = cos_sim(bar_mu, centroids[l - 1])
        sims_h_centroids = cos_sim(h_lar, centroids[l - 1])
        sims_h_bar_mu = np.sum(h_lar * bar_mu, axis=1, keepdims=True) / (np.linalg.norm(h_lar, axis=1, keepdims=True) * np.linalg.norm(bar_mu, axis=1, keepdims=True) + 1e-9)
        
        A = np.sum(C_lar * (sims_mu_centroids - sims_h_centroids * sims_h_bar_mu), axis=1)
        eta = np.zeros(len(h_lar))
        mask = (A > 0.04)
        eta[mask] = np.minimum(0.15, 1.0 * A[mask])
        
        h_warped = h_lar + eta[:, np.newaxis] * (bar_mu - h_lar)
        h_warped = h_warped / np.linalg.norm(h_warped, axis=1, keepdims=True)
        
        expert_update = np.zeros_like(h_warped)
        for k in range(K):
            expert_update += alpha_lar[:, k:k+1] * (v[k] - h_warped)
            
        h_lar = h_warped + gamma * expert_update
        if non_linear:
            h_lar = np.tanh(np.dot(h_lar, W[l].T))
        h_lar = h_lar / np.linalg.norm(h_lar, axis=1, keepdims=True)
        
    # Accuracies
    cm_accs = []
    lar_accs = []
    for k in range(K):
        mask = (y_test == k)
        
        alpha_cm_correct = alpha_cm[mask, k]
        p_cm = targets_un[k] + (alpha_cm_correct - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        cm_accs.append(np.mean(p_cm))
        
        alpha_lar_correct = alpha_lar[mask, k]
        p_lar = targets_un[k] + (alpha_lar_correct - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        lar_accs.append(np.mean(p_lar))
        
    return np.mean(cm_accs), np.mean(lar_accs)

accs_cm, accs_lar = [], []
for seed in range(10):
    ac_cm, ac_lar = run_test_nonlinear(seed, use_entropy_gating=True, non_linear=True)
    accs_cm.append(ac_cm)
    accs_lar.append(ac_lar)
    
print(f"Non-linear Sandbox (Setting D) -> ChemMerge: {np.mean(accs_cm)*100:.2f}%, L-ARC (Ours): {np.mean(accs_lar)*100:.2f}%")
