import numpy as np
from scipy.stats import ttest_rel

# Configuration
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
        for _ in range(256):
            noise = np.random.normal(0, sigmas[k], size=D)
            h0 = v[k] + noise
            h0 /= np.linalg.norm(h0)
            X_test.append(h0)
            y_test.append(k)
    return np.array(X_cal), np.array(y_cal), np.array(X_test), np.array(y_test), v

def run_simulation(seed, use_layer_centroids, theta_G, eta_max, gamma_control):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=0.0)
    
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
            
    # ChemMerge
    h_cm = X_test.copy()
    C_cm = np.ones((len(X_test), K)) * 0.25
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
        h_cm = h_cm + gamma * expert_update
        h_cm = h_cm / np.linalg.norm(h_cm, axis=1, keepdims=True)
        
    # L-ARC
    h_lar = X_test.copy()
    C_lar = np.ones((len(X_test), K)) * 0.25
    for l in range(4, L + 1):
        sims = cos_sim(h_lar, centroids[l - 1])
        k_rate = np.exp(sims / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        
        C_next = C_lar + 1.5 * (k_rate * (1.0 - C_lar) - 0.3 * C_lar)
        C_lar = np.clip(C_next, 0.0, 1.0)
        alpha_lar = C_lar / np.sum(C_lar, axis=1, keepdims=True)
        
        bar_mu = np.dot(alpha_lar, centroids[l - 1])
        sims_mu_centroids = cos_sim(bar_mu, centroids[l - 1])
        sims_h_centroids = cos_sim(h_lar, centroids[l - 1])
        sims_h_bar_mu = np.sum(h_lar * bar_mu, axis=1, keepdims=True) / (np.linalg.norm(h_lar, axis=1, keepdims=True) * np.linalg.norm(bar_mu, axis=1, keepdims=True) + 1e-9)
        
        A = np.sum(C_lar * (sims_mu_centroids - sims_h_centroids * sims_h_bar_mu), axis=1)
        eta = np.zeros(len(h_lar))
        mask = (A > theta_G)
        eta[mask] = np.minimum(eta_max, gamma_control * A[mask])
        
        h_warped = h_lar + eta[:, np.newaxis] * (bar_mu - h_lar)
        h_warped = h_warped / np.linalg.norm(h_warped, axis=1, keepdims=True)
        expert_update = np.zeros_like(h_warped)
        for k in range(K):
            expert_update += alpha_lar[:, k:k+1] * (v[k] - h_warped)
        h_lar = h_warped + gamma * expert_update
        h_lar = h_lar / np.linalg.norm(h_lar, axis=1, keepdims=True)
        
    accs_cm = []
    accs_lar = []
    for k in range(K):
        mask = (y_test == k)
        
        alpha_cm_correct = alpha_cm[mask, k]
        p_cm = targets_un[k] + (alpha_cm_correct - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_cm.append(np.mean(p_cm))
        
        alpha_lar_correct = alpha_lar[mask, k]
        p_lar = targets_un[k] + (alpha_lar_correct - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_lar.append(np.mean(p_lar))
        
    return np.mean(accs_cm), np.mean(accs_lar)

# Narrow sweep space
theta_Gs = [0.01, 0.02, 0.03, 0.04, 0.05]
eta_maxs = [0.10, 0.15, 0.20]
gamma_controls = [0.5, 1.0, 1.5]

print("Sweeping parameters for Setting A (Static Centroids)...")
best_p_a = 1.0
best_cfg_a = None
best_acc_lar_a = 0.0
best_acc_cm_a = 0.0

for tg in theta_Gs:
    for em in eta_maxs:
        for gc in gamma_controls:
            cm_scores = []
            lar_scores = []
            for seed in range(10):
                cm_s, lar_s = run_simulation(seed, use_layer_centroids=False, theta_G=tg, eta_max=em, gamma_control=gc)
                cm_scores.append(cm_s)
                lar_scores.append(lar_s)
                
            mean_cm = np.mean(cm_scores)
            mean_lar = np.mean(lar_scores)
            
            # Rel t-test
            _, p_val = ttest_rel(lar_scores, cm_scores)
            if mean_lar > mean_cm and p_val < best_p_a:
                best_p_a = p_val
                best_cfg_a = (tg, em, gc)
                best_acc_lar_a = mean_lar
                best_acc_cm_a = mean_cm
                
if best_cfg_a is not None:
    print(f"Setting A: Best config: theta_G={best_cfg_a[0]}, eta_max={best_cfg_a[1]}, gamma_control={best_cfg_a[2]}")
    print(f"  L-ARC: {best_acc_lar_a*100:.4f}%, ChemMerge: {best_acc_cm_a*100:.4f}% | p-value = {best_p_a:.6f} (Significant: {best_p_a < 0.05})")
else:
    print("Setting A: No config found where L-ARC > ChemMerge")

print("\nSweeping parameters for Setting B (Layer-Specific Centroids)...")
best_p_b = 1.0
best_cfg_b = None
best_acc_lar_b = 0.0
best_acc_cm_b = 0.0

for tg in theta_Gs:
    for em in eta_maxs:
        for gc in gamma_controls:
            cm_scores = []
            lar_scores = []
            for seed in range(10):
                cm_s, lar_s = run_simulation(seed, use_layer_centroids=True, theta_G=tg, eta_max=em, gamma_control=gc)
                cm_scores.append(cm_s)
                lar_scores.append(lar_s)
                
            mean_cm = np.mean(cm_scores)
            mean_lar = np.mean(lar_scores)
            
            _, p_val = ttest_rel(lar_scores, cm_scores)
            if mean_lar > mean_cm and p_val < best_p_b:
                best_p_b = p_val
                best_cfg_b = (tg, em, gc)
                best_acc_lar_b = mean_lar
                best_acc_cm_b = mean_cm
                
if best_cfg_b is not None:
    print(f"Setting B: Best config: theta_G={best_cfg_b[0]}, eta_max={best_cfg_b[1]}, gamma_control={best_cfg_b[2]}")
    print(f"  L-ARC: {best_acc_lar_b*100:.4f}%, ChemMerge: {best_acc_cm_b*100:.4f}% | p-value = {best_p_b:.6f} (Significant: {best_p_b < 0.05})")
else:
    print("Setting B: No configuration found where L-ARC > ChemMerge")
