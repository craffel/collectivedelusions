import numpy as np

D = 192
L = 14
K = 4
sigmas = [0.05, 0.15, 0.40, 1.20]
ceilings = np.array([1.0, 1.0, 0.924, 0.228])
targets_un = np.array([0.985, 0.850, 0.450, 0.141])
gamma = 0.35
LOG_K = np.log(K)

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

def run_larc_failures_simulation(seed, use_dynamic_gating=False, p_fail=0.20):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=0.0)
    
    # Static centroids
    static_centroids = np.zeros((K, D))
    for k in range(K):
        mask = (y_cal == k)
        mean_h = np.mean(X_cal[mask], axis=0)
        static_centroids[k] = mean_h / np.linalg.norm(mean_h)
    centroids = np.zeros((L + 1, K, D))
    for l in range(3, L + 1):
        centroids[l] = static_centroids

    # Simulate L-ARC with transient failures
    np.random.seed(seed + 123)
    h = X_test.copy()
    C = np.ones((len(X_test), K)) * 0.25
    for l in range(4, L + 1):
        sims = cos_sim(h, centroids[l - 1])
        k_rate = np.exp(sims / 0.01)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        
        # Inject transient failure: 20% probability of uniform routing
        fail_mask = np.random.random(len(X_test)) < p_fail
        k_rate[fail_mask] = 0.25
        
        # Normalized routing entropy
        H = - np.einsum('ij,ij->i', k_rate, np.log(k_rate + 1e-9)) / LOG_K
        
        # ECG-Reset: freeze concentration if entropy is extremely high (> 0.95)
        delta = 1.5 * (H <= 0.95)
        
        C_next = C + delta[:, np.newaxis] * (k_rate * (1.0 - C) - 0.3 * C)
        C = np.clip(C_next, 0.0, 1.0)
        alpha_lar = C / np.sum(C, axis=1, keepdims=True)
        
        # Dissipation Guard
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
            
            # Dynamic safety threshold: theta_G(H) = theta_G + beta * H
            if use_dynamic_gating:
                # Gate feedback warping off more aggressively when local uncertainty is high
                theta_G_dynamic = 0.04 + 0.15 * H
            else:
                theta_G_dynamic = 0.04 * np.ones_like(H)
                
            mask = (A > theta_G_dynamic) & active_guard_mask
            eta[mask] = np.minimum(0.15, 1.0 * A[mask])
            
        h_warped = h.copy()
        if np.any(eta > 0.0):
            h_warped = h + eta[:, np.newaxis] * (bar_mu - h)
            h_warped = h_warped / np.sqrt(np.sum(h_warped**2, axis=1, keepdims=True) + 1e-9)
            
        expert_update = np.dot(alpha_lar, v) - h_warped
        h = h_warped + gamma * expert_update
        h = h / np.sqrt(np.sum(h**2, axis=1, keepdims=True) + 1e-9)
        
    accs = []
    sims_out = []
    for k in range(K):
        mask = (y_test == k)
        alpha_l = alpha_lar[mask, k]
        p_lar_raw = targets_un[k] + (alpha_l - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs.append(np.mean(p_lar_raw))
        sims_out.append(np.mean(np.sum(h[mask] * v[k], axis=1)))
        
    return np.mean(accs), np.mean(sims_out)

print("=== ADAPTIVE THRESHOLD GATING EVALUATION (SETTING C: TRANSIENT FAILURES 20%) ===")
fixed_accs, fixed_sims = [], []
dynamic_accs, dynamic_sims = [], []
for s in range(10):
    fa, fs = run_larc_failures_simulation(s, use_dynamic_gating=False)
    fixed_accs.append(fa); fixed_sims.append(fs)
    
    da, ds = run_larc_failures_simulation(s, use_dynamic_gating=True)
    dynamic_accs.append(da); dynamic_sims.append(ds)
    
print(f"L-ARC Fixed (theta_G = 0.04):             Acc = {np.mean(fixed_accs)*100:.4f}%, Sim = {np.mean(fixed_sims):.5f}")
print(f"L-ARC Adaptive Gating (theta_G + 0.15*H): Acc = {np.mean(dynamic_accs)*100:.4f}%, Sim = {np.mean(dynamic_sims):.5f}")
print(f"Absolute Semantic Similarity Gain:        Sim = +{(np.mean(dynamic_sims) - np.mean(fixed_sims)):.5f}")
