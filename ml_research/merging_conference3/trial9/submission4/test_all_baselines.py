import os
import numpy as np
import json

def set_seed(seed):
    np.random.seed(seed)

class AnalyticalCoordinateSandbox:
    def __init__(self, D=192, L=14, K=4, L_frozen=3):
        self.D = D
        self.L = L
        self.K = K
        self.L_frozen = L_frozen
        self.block_dim = D // K
        
    def generate_task_signatures(self):
        signatures = []
        for k in range(self.K):
            v = np.zeros(self.D)
            start_idx = k * self.block_dim
            end_idx = (k + 1) * self.block_dim
            v[start_idx:end_idx] = np.random.normal(1.0, 0.1, self.block_dim)
            v = v / np.linalg.norm(v)
            signatures.append(v)
        return np.array(signatures)

    def generate_sample(self, task_idx, signatures, noise_scales):
        v = signatures[task_idx]
        sigma = noise_scales[task_idx]
        epsilon = np.random.normal(0, sigma, self.D)
        return v + epsilon

def run_comparisons_seed(seed, num_samples=1000):
    set_seed(seed)
    sandbox = AnalyticalCoordinateSandbox()
    signatures = sandbox.generate_task_signatures()
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    # Pre-compute Global Centroids
    global_centroids = []
    for k in range(sandbox.K):
        cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
        centroid = np.mean(cal_samples, axis=0)
        global_centroids.append(centroid / np.linalg.norm(centroid))
    global_centroids = np.array(global_centroids)
    
    # Save RNG state right before layer calibration to preserve stream generation
    rng_before_layer_cal = np.random.get_state()
    
    gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
    layer_centroids = {}
    for k in range(sandbox.K):
        cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
        flows = []
        for h_init in cal_samples:
            h = h_init.copy()
            flow = [h.copy()]
            for l in range(sandbox.L_frozen + 1, sandbox.L):
                h = (1 - gammas[l]) * h + gammas[l] * signatures[k]
                h += np.random.normal(0, 0.015, sandbox.D)
                flow.append(h.copy())
            flows.append(flow)
        
        for idx_l, l in enumerate(range(sandbox.L_frozen + 1, sandbox.L + 1)):
            layer_reps = np.array([f[idx_l] for f in flows])
            mean_rep = np.mean(layer_reps, axis=0)
            layer_centroids[(l, k)] = mean_rep / np.linalg.norm(mean_rep)
            
    # Restore RNG state so the serving stream is generated exactly as before
    np.random.set_state(rng_before_layer_cal)
    
    # Generate serving stream
    task_indices = np.random.choice(sandbox.K, size=num_samples)
    stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
    
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    tau_mom = 0.005
    tau_sable_opt = 0.200 # tuned optimal for SABLE + LC
    tau_chem_opt = 0.050  # tuned optimal for ChemMerge
    
    methods = [
        "Expert Ceiling", 
        "Uniform Merging", 
        "SABLE", 
        "SABLE + Layer Centroids",
        "ChemMerge", 
        "ChemMerge + Layer Centroids",
        "Momentum-Merge (Base)",
        "Momentum-Merge (Advanced)"
    ]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        init_rng_state = np.random.get_state()
        
        # 1. Expert Ceiling
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            h = (1 - gammas[l]) * h + gammas[l] * signatures[k_true]
            h += np.random.normal(0, 0.015, sandbox.D)
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Expert Ceiling"] += 1
            
        # 2. Uniform Merging
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            h = (1 - gammas[l]) * h + gammas[l] * np.mean(signatures, axis=0)
            h += np.random.normal(0, 0.015, sandbox.D)
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Uniform Merging"] += 1
            
        # 3. SABLE
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_sable = 0.0
        tau_sable = 0.005 # SABLE at baseline temperature
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_sable)
            alpha_prev = alpha.copy()
            alpha = exp_sims / np.sum(exp_sims)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_sable += np.sum((alpha - alpha_prev)**2)
        jitter_sable /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["SABLE"] += jitter_sable
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["SABLE"] += 1
            
        # 4. SABLE + Layer Centroids
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_sable_lc = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, layer_centroids[(l, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_sable_opt)
            alpha_prev = alpha.copy()
            alpha = exp_sims / np.sum(exp_sims)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_sable_lc += np.sum((alpha - alpha_prev)**2)
        jitter_sable_lc /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["SABLE + Layer Centroids"] += jitter_sable_lc
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["SABLE + Layer Centroids"] += 1
            
        # 5. ChemMerge (Global Centroids)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        C = np.full(sandbox.K, 1.0 / sandbox.K)
        dt = 1.5
        k_decay = 0.3
        jitter_chem = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_chem_opt)
            k_rates = exp_sims / np.sum(exp_sims)
            lambdas = np.exp(-(k_rates + k_decay) * dt)
            C_star = k_rates / (k_rates + k_decay)
            C = lambdas * C + (1.0 - lambdas) * C_star
            alpha_prev = alpha.copy() if l > sandbox.L_frozen + 1 else np.full(sandbox.K, 1.0 / sandbox.K)
            alpha = C / np.sum(C)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_chem += np.sum((alpha - alpha_prev)**2)
        jitter_chem /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["ChemMerge"] += jitter_chem
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["ChemMerge"] += 1

        # 6. ChemMerge + Layer Centroids
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        C = np.full(sandbox.K, 1.0 / sandbox.K)
        dt = 1.5
        k_decay = 0.3
        jitter_chem_lc = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, layer_centroids[(l, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_chem_opt)
            k_rates = exp_sims / np.sum(exp_sims)
            lambdas = np.exp(-(k_rates + k_decay) * dt)
            C_star = k_rates / (k_rates + k_decay)
            C = lambdas * C + (1.0 - lambdas) * C_star
            alpha_prev = alpha.copy() if l > sandbox.L_frozen + 1 else np.full(sandbox.K, 1.0 / sandbox.K)
            alpha = C / np.sum(C)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_chem_lc += np.sum((alpha - alpha_prev)**2)
        jitter_chem_lc /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["ChemMerge + Layer Centroids"] += jitter_chem_lc
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["ChemMerge + Layer Centroids"] += 1
            
        # 7. Momentum-Merge (Base)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_mom = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / 0.100) # optimal tau for MM Base
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_mom += np.sum((alpha - alpha_prev)**2)
        jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Base)"] += jitter_mom
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Base)"] += 1
            
        # 8. Momentum-Merge (Advanced)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        l_first = sandbox.L_frozen + 1
        sims_first = np.array([np.dot(h, layer_centroids[(l_first, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
        exp_sims_first = np.exp(sims_first / tau_mom)
        w_first = exp_sims_first / np.sum(exp_sims_first)
        
        alpha = w_first.copy()
        jitter_adv = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, layer_centroids[(l, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            if l == l_first:
                alpha = w_first.copy()
            else:
                alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_adv += np.sum((alpha - alpha_prev)**2)
        jitter_adv /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Advanced)"] += jitter_adv
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Advanced)"] += 1

    seed_results = {}
    for m in methods:
        seed_results[m] = {
            "accuracy": correct[m] / num_samples,
            "jitter": total_jitter[m] / num_samples
        }
    return seed_results

def run_simulation(seeds=10):
    methods = [
        "Expert Ceiling", 
        "Uniform Merging", 
        "SABLE", 
        "SABLE + Layer Centroids",
        "ChemMerge", 
        "ChemMerge + Layer Centroids",
        "Momentum-Merge (Base)",
        "Momentum-Merge (Advanced)"
    ]
    raw_results = {m: {"accuracy": [], "jitter": []} for m in methods}
    
    for seed in range(seeds):
        res = run_comparisons_seed(seed=seed)
        for m in methods:
            raw_results[m]["accuracy"].append(res[m]["accuracy"])
            raw_results[m]["jitter"].append(res[m]["jitter"])
            
    print("\n" + "="*115)
    print(f"{'Method':<35} | {'Accuracy Mean (%)':<15} | {'Accuracy Std (%)':<15} | {'Jitter Mean (MSE)':<15} | {'Jitter Std':<15}")
    print("="*115)
    for m in methods:
        accs = np.array(raw_results[m]["accuracy"]) * 100
        jits = np.array(raw_results[m]["jitter"])
        print(f"{m:<35} | {np.mean(accs):<15.2f} | {np.std(accs):<15.2f} | {np.mean(jits):<15.6f} | {np.std(jits):<15.6f}")
    print("="*115)

if __name__ == "__main__":
    run_simulation(seeds=10)
