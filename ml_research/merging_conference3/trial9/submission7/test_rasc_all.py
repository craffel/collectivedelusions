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

def run_serving_simulation(seed, rho=0.0, use_layer_centroids=False, p_fail=0.0, bias_scale=0.0, rasc=True):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=rho)
    
    # Compute Centroids
    centroids = np.zeros((L + 1, K, D))
    if use_layer_centroids:
        for k in range(K):
            mask = (y_cal == k)
            mean_h = np.mean(X_cal[mask], axis=0)
            centroids[3, k] = mean_h / np.linalg.norm(mean_h)
        for l in range(4, L + 1):
            for k in range(K):
                mask = (y_cal == k)
                h_cal = X_cal[mask].copy()
                for j in range(4, l):
                    h_cal = h_cal + gamma * (v[k] - h_cal)
                    h_cal = h_cal / np.linalg.norm(h_cal, axis=1, keepdims=True)
                mean_h = np.mean(h_cal, axis=0)
                centroids[l - 1, k] = mean_h / np.linalg.norm(mean_h)
    else:
        static_centroids = np.zeros((K, D))
        for k in range(K):
            mask = (y_cal == k)
            mean_h = np.mean(X_cal[mask], axis=0)
            static_centroids[k] = mean_h / np.linalg.norm(mean_h)
        for l in range(3, L + 1):
            centroids[l] = static_centroids
            
    # SABLE (Stateless baseline)
    h_sable = X_test.copy()
    for l in range(4, L + 1):
        sims = np.dot(h_sable, centroids[l - 1].T)
        sims_routing = sims.copy()
        if bias_scale > 0.0:
            for i in range(len(h_sable)):
                incorrect_k = (y_test[i] + 1) % K
                sims_routing[i, incorrect_k] += bias_scale
        alpha = np.exp(sims_routing / 0.05)
        alpha = alpha / np.sum(alpha, axis=1, keepdims=True)
        if p_fail > 0.0:
            np.random.seed(seed * 100 + l)
            fail_mask = np.random.rand(len(h_sable)) < p_fail
            if np.sum(fail_mask) > 0:
                alpha[fail_mask] = 1.0 / K
        expert_update = np.dot(alpha, v) - h_sable
        h_sable = h_sable + gamma * expert_update
        h_sable = h_sable / np.linalg.norm(h_sable, axis=1, keepdims=True)
        
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
        if rasc:
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
        
    accs_sable, sims_sable = [], []
    accs_larc, sims_larc = [], []
    for k in range(K):
        mask = (y_test == k)
        
        # SABLE
        alpha_s = alpha[mask, k]
        p_correct_s = targets_un[k] + (alpha_s - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_sable.append(np.mean(p_correct_s))
        sims_sable.append(np.mean(np.sum(h_sable[mask] * v[k], axis=1)))
        
        # L-ARC
        alpha_l = alpha_lar[mask, k]
        p_correct_l = targets_un[k] + (alpha_l - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_larc.append(np.mean(p_correct_l))
        sims_larc.append(np.mean(np.sum(h[mask] * v[k], axis=1)))
        
    return np.mean(accs_sable), np.mean(sims_sable), np.mean(accs_larc), np.mean(sims_larc)

# Run simulations across 10 seeds
print("=== RASC ALL SETTINGS TEST (CORRECTED FAULT) ===")
for rasc_enabled in [False, True]:
    print(f"\nRASC Enabled = {rasc_enabled}")
    
    # Setting A
    accs_a_s, sims_a_s, accs_a_l, sims_a_l = [], [], [], []
    for s in range(10):
        as_a, ss_a, al_a, sl_a = run_serving_simulation(s, use_layer_centroids=False, rasc=rasc_enabled)
        accs_a_s.append(as_a); sims_a_s.append(ss_a)
        accs_a_l.append(al_a); sims_a_l.append(sl_a)
    print(f"Setting A (Static):")
    print(f"  SABLE: Acc = {np.mean(accs_a_s)*100:.2f}%, Sim = {np.mean(sims_a_s):.4f}")
    print(f"  L-ARC: Acc = {np.mean(accs_a_l)*100:.2f}%, Sim = {np.mean(sims_a_l):.4f}")
    
    # Setting C
    accs_c_s, sims_c_s, accs_c_l, sims_c_l = [], [], [], []
    for s in range(10):
        as_c, ss_c, al_c, sl_c = run_serving_simulation(s, use_layer_centroids=False, p_fail=0.2, rasc=rasc_enabled)
        accs_c_s.append(as_c); sims_c_s.append(ss_c)
        accs_c_l.append(al_c); sims_c_l.append(sl_c)
    print(f"Setting C (Fault-Tolerant):")
    print(f"  SABLE: Acc = {np.mean(accs_c_s)*100:.2f}%, Sim = {np.mean(sims_c_s):.4f}")
    print(f"  L-ARC: Acc = {np.mean(accs_c_l)*100:.2f}%, Sim = {np.mean(sims_c_l):.4f}")

    # Setting D
    accs_d_s, sims_d_s, accs_d_l, sims_d_l = [], [], [], []
    for s in range(10):
        as_d, ss_d, al_d, sl_d = run_serving_simulation(s, use_layer_centroids=False, bias_scale=0.15, rasc=rasc_enabled)
        accs_d_s.append(as_d); sims_d_s.append(ss_d)
        accs_d_l.append(al_d); sims_d_l.append(sl_d)
    print(f"Setting D (Bias):")
    print(f"  SABLE: Acc = {np.mean(accs_d_s)*100:.2f}%, Sim = {np.mean(sims_d_s):.4f}")
    print(f"  L-ARC: Acc = {np.mean(accs_d_l)*100:.2f}%, Sim = {np.mean(sims_d_l):.4f}")
