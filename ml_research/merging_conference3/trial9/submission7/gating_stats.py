import numpy as np
from run_experiments import generate_data, cos_sim

D = 192
L = 14
K = 4
sigmas = [0.05, 0.15, 0.40, 1.20]
gamma = 0.35

def get_gating_percentage(rho=0.0, use_layer_centroids=False):
    gated_count = 0
    total_count = 0
    active_etas = []
    
    for seed in range(10):
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
                
        h = X_test.copy()
        C_init = np.ones((len(X_test), K)) * 0.25
        C = C_init.copy()
        
        for l in range(4, L + 1):
            sims = cos_sim(h, centroids[l - 1])
            k_rate = np.exp(sims / 0.01)
            k_rate = k_rate / np.sum(k_rate, axis=1, keepdims=True)
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
            
            # Count gated (dissipation strength below safety threshold)
            gated_count += np.sum(A <= 0.04)
            total_count += len(h)
            
            active_etas.extend(eta[mask])
            
            h_warped = h + eta[:, np.newaxis] * (bar_mu - h)
            h_warped = h_warped / np.linalg.norm(h_warped, axis=1, keepdims=True)
            
            expert_update = np.zeros_like(h_warped)
            for k in range(K):
                expert_update += alpha_lar[:, k:k+1] * (v[k] - h_warped)
            h = h_warped + gamma * expert_update
            h = h / np.linalg.norm(h, axis=1, keepdims=True)
            
    percentage = (gated_count / total_count) * 100
    mean_active_eta = np.mean(active_etas) if active_etas else 0.0
    return percentage, mean_active_eta

for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pct, m_eta = get_gating_percentage(rho=r, use_layer_centroids=False)
    print(f"Rho = {r:.1f} | Gating percentage = {pct:.2f}% | Mean Active Eta = {m_eta:.4f}")
