import numpy as np

def run_test(E, gamma_val, lambda_val, l_ns, tau_sable, tau_chem, tau_mom, best_beta):
    np.random.seed(42)
    D = 192
    L = 14
    K = 4
    L_frozen = 3
    block_dim = D // K
    num_samples = 400
    
    signatures = []
    for k in range(K):
        v = np.zeros(D)
        start_idx = k * block_dim
        end_idx = (k + 1) * block_dim
        v[start_idx:end_idx] = np.random.normal(1.0, 0.1, block_dim)
        v = v / np.linalg.norm(v)
        signatures.append(v)
    signatures = np.array(signatures)
    
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    centroids = []
    for k in range(K):
        cal_samples = [signatures[k] + np.random.normal(0, noise_scales[k], D) for _ in range(64)]
        centroid = np.mean(cal_samples, axis=0)
        centroids.append(centroid / np.linalg.norm(centroid))
    centroids = np.array(centroids)
    
    task_indices = np.random.choice(K, size=num_samples)
    stream_samples = [signatures[k] + np.random.normal(0, noise_scales[k], D) for k in task_indices]
    
    gammas = {l: gamma_val for l in range(L_frozen + 1, L + 1)}
    
    methods = ["Expert Ceiling", "Uniform Merging", "SABLE", "ChemMerge", "Momentum-Merge"]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        # 1. Expert Ceiling
        h = h_init.copy()
        for l in range(L_frozen + 1, L + 1):
            h = (1 - gammas[l]) * h + gammas[l] * signatures[k_true]
            h += np.random.normal(0, l_ns, D)
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Expert Ceiling"] += 1
            
        # 2. Uniform Merging
        h = h_init.copy()
        for l in range(L_frozen + 1, L + 1):
            h = (1 - gammas[l]) * h + gammas[l] * np.mean(signatures, axis=0)
            h += np.random.normal(0, l_ns, D)
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Uniform Merging"] += 1
            
        # 3. SABLE
        h = h_init.copy()
        alpha = np.full(K, 1.0 / K)
        jitter_sable = 0.0
        for l in range(L_frozen + 1, L + 1):
            similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(K)])
            exp_sims = np.exp(similarities / tau_sable)
            alpha_prev = alpha.copy()
            alpha = exp_sims / np.sum(exp_sims)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, l_ns, D)
            jitter_sable += np.sum((alpha - alpha_prev)**2)
        jitter_sable /= (L - L_frozen - 1)
        total_jitter["SABLE"] += jitter_sable
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["SABLE"] += 1
            
        # 4. ChemMerge
        h = h_init.copy()
        C = np.full(K, 1.0 / K)
        dt = 1.5
        k_decay = 0.3
        jitter_chem = 0.0
        for l in range(L_frozen + 1, L + 1):
            similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(K)])
            exp_sims = np.exp(similarities / tau_chem)
            k_rates = exp_sims / np.sum(exp_sims)
            lambdas = np.exp(-(k_rates + k_decay) * dt)
            C_star = k_rates / (k_rates + k_decay)
            C = lambdas * C + (1.0 - lambdas) * C_star
            alpha_prev = alpha.copy() if l > L_frozen + 1 else np.full(K, 1.0/K)
            alpha = C / np.sum(C)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, l_ns, D)
            jitter_chem += np.sum((alpha - alpha_prev)**2)
        jitter_chem /= (L - L_frozen - 1)
        total_jitter["ChemMerge"] += jitter_chem
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["ChemMerge"] += 1
            
        # 5. Momentum-Merge
        h = h_init.copy()
        alpha = np.full(K, 1.0 / K)
        jitter_mom = 0.0
        for l in range(L_frozen + 1, L + 1):
            similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, l_ns, D)
            jitter_mom += np.sum((alpha - alpha_prev)**2)
        jitter_mom /= (L - L_frozen - 1)
        total_jitter["Momentum-Merge"] += jitter_mom
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge"] += 1

    acc = {m: correct[m] / num_samples for m in methods}
    return acc, {m: total_jitter[m] / num_samples for m in methods}

target_ec = 0.7905
target_um = 0.6065
target_sable = 0.6810
target_chem = 0.7811

best_diff = float('inf')
best_params = None

for e_svhn in [0.45, 0.48, 0.50, 0.52]:
    for e1 in [0.08, 0.10, 0.12]:
        for e2 in [0.18, 0.20, 0.22]:
            E = [0.005, e1, e2, e_svhn]
            for gamma in [0.28, 0.30, 0.32]:
                for lam in [0.15, 0.20, 0.25]:
                    for l_ns in [0.004, 0.006, 0.008]:
                        for tau_s in [0.08, 0.10, 0.12]:
                            acc, jit = run_test(E, gamma, lam, l_ns, tau_s, 0.01, 0.01, 0.40)
                            diff = (abs(acc["Expert Ceiling"] - target_ec) + 
                                    abs(acc["Uniform Merging"] - target_um) + 
                                    abs(acc["SABLE"] - target_sable) + 
                                    abs(acc["ChemMerge"] - target_chem))
                            if diff < best_diff:
                                best_diff = diff
                                best_params = (E, gamma, lam, l_ns, tau_s, acc, jit)

print("Best calibration:")
print(f"E: {best_params[0]}")
print(f"Gamma: {best_params[1]:.3f}, Lambda: {best_params[2]:.3f}, Layer Noise: {best_params[3]:.3f}")
print(f"T_SABLE: {best_params[4]:.3f}")
print("Accuracies:")
for m, a in best_params[5].items():
    print(f"  {m}: {a*100:.2f}% (Jitter: {best_params[6][m]:.6f})")
