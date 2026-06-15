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

def run_serving_simulation(seed, tau, bias_scale=0.0, rasc=True, use_threshold_accuracy=False, threshold_val=0.7):
    X_cal, y_cal, X_test, y_test, v = generate_data(seed, 256, rho=0.0)
    
    # Compute Centroids (Static centroids at Layer 3)
    static_centroids = np.zeros((K, D))
    for k in range(K):
        mask = (y_cal == k)
        mean_h = np.mean(X_cal[mask], axis=0)
        static_centroids[k] = mean_h / np.linalg.norm(mean_h)
    centroids = np.zeros((L + 1, K, D))
    for l in range(3, L + 1):
        centroids[l] = static_centroids
        
    # 1. ChemMerge (Decoupled, eta=0)
    h_cm = X_test.copy()
    C_cm = np.ones((len(X_test), K)) * 0.25
    for l in range(4, L + 1):
        sims = cos_sim(h_cm, centroids[l - 1])
        sims_routing = sims.copy()
        if bias_scale > 0.0:
            for i in range(len(h_cm)):
                incorrect_k = (y_test[i] + 1) % K
                sims_routing[i, incorrect_k] += bias_scale
        k_rate = np.exp(sims_routing / tau)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        C_next = C_cm + 1.5 * (k_rate * (1.0 - C_cm) - 0.3 * C_cm)
        C_cm = np.clip(C_next, 0.0, 1.0)
        alpha_cm = C_cm / np.sum(C_cm, axis=1, keepdims=True)
        
        expert_update = np.zeros_like(h_cm)
        for k in range(K):
            expert_update += alpha_cm[:, k:k+1] * (v[k] - h_cm)
        h_cm = h_cm + gamma * expert_update
        h_cm = h_cm / np.linalg.norm(h_cm, axis=1, keepdims=True)

    # 2. L-ARC (Ours)
    h = X_test.copy()
    C = np.ones((len(X_test), K)) * 0.25
    for l in range(4, L + 1):
        sims = cos_sim(h, centroids[l - 1])
        sims_routing = sims.copy()
        if bias_scale > 0.0:
            for i in range(len(h)):
                incorrect_k = (y_test[i] + 1) % K
                sims_routing[i, incorrect_k] += bias_scale
        k_rate = np.exp(sims_routing / tau)
        k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
        
        p_sims = np.exp(sims / 0.05)
        p_sims = p_sims / np.sum(p_sims, axis=1, keepdims=True)
        
        k_pred = np.argmax(k_rate, axis=1)
        sims_pred = np.argmax(sims, axis=1)
        disagree = (k_pred != sims_pred)
        
        H = - np.einsum('ij,ij->i', k_rate, np.log(k_rate + 1e-9)) / LOG_K
        delta = 1.5 * (H <= 0.95)
        
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
        
    accs_cm, sims_cm_out = [], []
    accs_larc, sims_larc_out = [], []
    for k in range(K):
        mask = (y_test == k)
        
        # ChemMerge
        alpha_c = alpha_cm[mask, k]
        if use_threshold_accuracy:
            # Model non-linear threshold effect (e.g. step or steep sigmoid at threshold_val)
            # If ensembling weight is above threshold, we get ceiling performance. If below, we get base performance.
            p_cm_raw = (alpha_c >= threshold_val) * ceilings[k] + (alpha_c < threshold_val) * targets_un[k]
        else:
            p_cm_raw = targets_un[k] + (alpha_c - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_cm.append(np.mean(p_cm_raw))
        sims_cm_out.append(np.mean(np.sum(h_cm[mask] * v[k], axis=1)))
        
        # L-ARC
        alpha_l = alpha_lar[mask, k]
        if use_threshold_accuracy:
            p_lar_raw = (alpha_l >= threshold_val) * ceilings[k] + (alpha_l < threshold_val) * targets_un[k]
        else:
            p_lar_raw = targets_un[k] + (alpha_l - 0.25) * (ceilings[k] - targets_un[k]) / 0.75
        accs_larc.append(np.mean(p_lar_raw))
        sims_larc_out.append(np.mean(np.sum(h[mask] * v[k], axis=1)))
        
    return np.mean(accs_cm), np.mean(sims_cm_out), np.mean(accs_larc), np.mean(sims_larc_out)

print("=== SWEEP ARRHENIUS TEMPERATURE (tau) ===")
print("Evaluating Setting A (Clean Static centoids):")
taus = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
for t in taus:
    cm_a, cm_s, lar_a, lar_s = [], [], [], []
    for s in range(10):
        c_a, c_s, l_a, l_s = run_serving_simulation(s, tau=t, bias_scale=0.0, rasc=True)
        cm_a.append(c_a); cm_s.append(c_s)
        lar_a.append(l_a); lar_s.append(l_s)
    print(f"  tau = {t:5f} | ChemMerge: Acc = {np.mean(cm_a)*100:.2f}%, Sim = {np.mean(cm_s):.4f} | L-ARC: Acc = {np.mean(lar_a)*100:.2f}%, Sim = {np.mean(lar_s):.4f}")

print("\nEvaluating Setting D (Systematic Bias 0.15, RASC enabled):")
for t in taus:
    cm_a, cm_s, lar_a, lar_s = [], [], [], []
    for s in range(10):
        c_a, c_s, l_a, l_s = run_serving_simulation(s, tau=t, bias_scale=0.15, rasc=True)
        cm_a.append(c_a); cm_s.append(c_s)
        lar_a.append(l_a); lar_s.append(l_s)
    print(f"  tau = {t:5f} | ChemMerge: Acc = {np.mean(cm_a)*100:.2f}%, Sim = {np.mean(cm_s):.4f} | L-ARC: Acc = {np.mean(lar_a)*100:.2f}%, Sim = {np.mean(lar_s):.4f}")

print("\n=== SWEEP NON-LINEAR THRESHOLD EFFECT ON DOWNSTREAM ACCURACY ===")
# Let's see if the ensembling-weight-to-accuracy proxy holds when task competence requires a minimum adapter activation threshold
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("Evaluating Setting D (Systematic Bias 0.15, RASC enabled, tau=0.01):")
for thresh in thresholds:
    cm_a, _, lar_a, _ = [], [], [], []
    for s in range(10):
        c_a, _, l_a, _ = run_serving_simulation(s, tau=0.01, bias_scale=0.15, rasc=True, use_threshold_accuracy=True, threshold_val=thresh)
        cm_a.append(c_a)
        lar_a.append(l_a)
    print(f"  Threshold = {thresh:.1f} | ChemMerge (RASC): Acc = {np.mean(cm_a)*100:.2f}% | L-ARC (Ours): Acc = {np.mean(lar_a)*100:.2f}% (Gain = +{(np.mean(lar_a)-np.mean(cm_a))*100:.2f}%)")
