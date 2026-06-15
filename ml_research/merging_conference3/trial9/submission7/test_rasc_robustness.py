import os
import numpy as np

D = 192
L = 14
K = 4
sigmas = [0.05, 0.15, 0.40, 1.20]
ceilings = np.array([1.0, 1.0, 0.924, 0.228])
targets_un = np.array([0.985, 0.850, 0.450, 0.141])
gamma = 0.35
LOG_K = np.log(K)

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

def run_simulation(seed, p_fail=0.0, bias_scale=0.0, use_rasc=True):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 100, rho=0.0)
    static_centroids = np.zeros((K, D))
    for k in range(K):
        mask = (y_cal == k)
        mean_h = np.mean(X_cal[mask], axis=0)
        static_centroids[k] = mean_h / np.linalg.norm(mean_h)
    centroids = np.zeros((L + 1, K, D))
    for l in range(3, L + 1):
        centroids[l] = static_centroids

    # L-ARC
    h = X_test.copy()
    C = np.ones((len(X_test), K)) * 0.25
    for l in range(4, L + 1):
        sims = np.dot(h, centroids[l - 1].T)
        sims_routing = sims.copy()
        if bias_scale > 0.0:
            for i in range(len(h)):
                incorrect_k = (y_test[i] + 1) % K
                sims_routing[i, incorrect_k] += bias_scale
        k_rate = np.exp(sims_routing / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        
        # Apply failure mask under Setting C
        if p_fail > 0.0:
            np.random.seed(seed * 100 + l)
            fail_mask = np.random.rand(len(h)) < p_fail
            if np.sum(fail_mask) > 0:
                k_rate[fail_mask] = 1.0 / K
                
        # Unbiased representation-space similarity distribution
        p_sims = np.exp(sims / 0.05)
        p_sims = p_sims / np.sum(p_sims, axis=1, keepdims=True)
        
        # Mismatch / Disagreement
        k_pred = np.argmax(k_rate, axis=1)
        sims_pred = np.argmax(sims, axis=1)
        disagree = (k_pred != sims_pred)
        
        # Entropy-gated reset delta
        H = - np.einsum('ij,ij->i', k_rate, np.log(k_rate + 1e-9)) / LOG_K
        delta = 1.5 * (H <= 0.95)
        
        # Apply RASC override if they disagree and rasc is enabled
        k_rate_used = k_rate.copy()
        if use_rasc:
            k_rate_used[disagree] = p_sims[disagree]
            
        C_next = C + delta[:, np.newaxis] * (k_rate_used * (1.0 - C) - 0.3 * C)
        C = np.clip(C_next, 0.0, 1.0)
        alpha_lar = C / np.sum(C, axis=1, keepdims=True)
        
        bar_mu = np.dot(alpha_lar, centroids[l - 1])
        eta = np.zeros(len(h))
        active_guard_mask = (H >= 0.15) & (H <= 0.95)
        
        if np.any(active_guard_mask):
            norm_bar = np.sqrt(np.sum(bar_mu**2, axis=1, keepdims=True) + 1e-9)
            bar_mu_norm = bar_mu / norm_bar
            sims_mu_centroids = np.dot(bar_mu_norm, centroids[l - 1].T)
            sims_h_centroids = sims
            sims_h_bar_mu = np.einsum('ij,ij->i', h, bar_mu_norm)
            
            A = np.einsum('ij,ij->i', C, sims_mu_centroids) - sims_h_bar_mu * np.einsum('ij,ij->i', C, sims_h_centroids)
            mask = (A > 0.04) & active_guard_mask
            eta[mask] = np.minimum(0.15, 1.0 * A[mask])
            
        h_warped = h.copy()
        if np.any(eta > 0.0):
            h_warped = h + eta[:, np.newaxis] * (bar_mu - h)
            h_warped = h_warped / np.sqrt(np.sum(h_warped**2, axis=1, keepdims=True) + 1e-9)
            
        expert_update = np.dot(alpha_lar, v) - h_warped
        h = h_warped + gamma * expert_update
        h = h / np.sqrt(np.sum(h**2, axis=1, keepdims=True) + 1e-9)
        
    accs_larc = []
    for k in range(K):
        mask = (y_test == k)
        alpha_l = alpha_lar[mask, k]
        p_correct_l = targets_un[k] + (alpha_l - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_larc.append(np.mean(p_correct_l))
    return np.mean(accs_larc)

# 1. Sweep Bias Scale under zero transient failure
print("=== STRESS TEST 1: SWEEPING BIAS SCALE (No Transient Failures) ===")
bias_scales = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
for bias in bias_scales:
    accs_no_rasc = []
    accs_with_rasc = []
    for seed in range(5):
        accs_no_rasc.append(run_simulation(seed, p_fail=0.0, bias_scale=bias, use_rasc=False))
        accs_with_rasc.append(run_simulation(seed, p_fail=0.0, bias_scale=bias, use_rasc=True))
    gain = np.mean(accs_with_rasc) - np.mean(accs_no_rasc)
    print(f"Bias={bias:.2f} | Standard L-ARC: {np.mean(accs_no_rasc)*100:.2f}% | RASC L-ARC: {np.mean(accs_with_rasc)*100:.2f}% | Gain: {gain*100:+.2f}%")

# 2. Sweep Failure Probability under zero bias
print("\n=== STRESS TEST 2: SWEEPING FAILURE PROBABILITY (No Router Bias) ===")
fail_probs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
for p in fail_probs:
    accs_no_rasc = []
    accs_with_rasc = []
    for seed in range(5):
        accs_no_rasc.append(run_simulation(seed, p_fail=p, bias_scale=0.0, use_rasc=False))
        accs_with_rasc.append(run_simulation(seed, p_fail=p, bias_scale=0.0, use_rasc=True))
    gain = np.mean(accs_with_rasc) - np.mean(accs_no_rasc)
    print(f"p_fail={p:.2f} | Standard L-ARC: {np.mean(accs_no_rasc)*100:.2f}% | RASC L-ARC: {np.mean(accs_with_rasc)*100:.2f}% | Gain: {gain*100:+.2f}%")
