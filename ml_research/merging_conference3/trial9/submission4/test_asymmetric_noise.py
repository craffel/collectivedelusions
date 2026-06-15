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

def run_asymmetric_noise_seed(seed, layer_noise_scales, num_samples=1000):
    set_seed(seed)
    sandbox = AnalyticalCoordinateSandbox()
    signatures = sandbox.generate_task_signatures()
    noise_scales = [0.05, 0.15, 0.40, 1.20] # input sample noise
    
    # Pre-compute Global Centroids
    global_centroids = []
    for k in range(sandbox.K):
        cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
        centroid = np.mean(cal_samples, axis=0)
        global_centroids.append(centroid / np.linalg.norm(centroid))
    global_centroids = np.array(global_centroids)
    
    # Pre-compute Layer-wise Centroids (Eq 9) with asymmetric layer noise
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
                # Use task-asymmetric layer noise during calibration flow
                h += np.random.normal(0, layer_noise_scales[k], sandbox.D)
                flow.append(h.copy())
            flows.append(flow)
        for idx_l, l in enumerate(range(sandbox.L_frozen + 1, sandbox.L + 1)):
            layer_reps = np.array([f[idx_l] for f in flows])
            mean_rep = np.mean(layer_reps, axis=0)
            layer_centroids[(l, k)] = mean_rep / np.linalg.norm(mean_rep)
            
    # Generate serving stream
    task_indices = np.random.choice(sandbox.K, size=num_samples)
    stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
    
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    tau_mom = 0.005
    
    methods = [
        "SABLE", 
        "ChemMerge", 
        "Momentum-Merge (Base)",
        "Momentum-Merge (Advanced)"
    ]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        init_rng_state = np.random.get_state()
        sig_layer_noise = layer_noise_scales[k_true]
        
        # 1. SABLE
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_sable = 0.0
        tau_sable = 0.15
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_sable)
            alpha_prev = alpha.copy()
            alpha = exp_sims / np.sum(exp_sims)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, sig_layer_noise, sandbox.D)
            jitter_sable += np.sum((alpha - alpha_prev)**2)
        jitter_sable /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["SABLE"] += jitter_sable
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["SABLE"] += 1
            
        # 2. ChemMerge
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        C = np.full(sandbox.K, 1.0 / sandbox.K)
        dt = 1.5
        k_decay = 0.3
        jitter_chem = 0.0
        tau_chem = 0.005
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_chem)
            k_rates = exp_sims / np.sum(exp_sims)
            lambdas = np.exp(-(k_rates + k_decay) * dt)
            C_star = k_rates / (k_rates + k_decay)
            C = lambdas * C + (1.0 - lambdas) * C_star
            alpha_prev = alpha.copy() if l > sandbox.L_frozen + 1 else np.full(sandbox.K, 1.0 / sandbox.K)
            alpha = C / np.sum(C)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, sig_layer_noise, sandbox.D)
            jitter_chem += np.sum((alpha - alpha_prev)**2)
        jitter_chem /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["ChemMerge"] += jitter_chem
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["ChemMerge"] += 1
            
        # 3. Momentum-Merge (Base)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_mom = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, sig_layer_noise, sandbox.D)
            jitter_mom += np.sum((alpha - alpha_prev)**2)
        jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Base)"] += jitter_mom
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Base)"] += 1
            
        # 4. Momentum-Merge (Advanced) [Eq 9 + Eq 10]
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        l_first = sandbox.L_frozen + 1
        sims_first = np.array([np.dot(h, layer_centroids[(l_first, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
        exp_sims_first = np.exp(sims_first / tau_mom)
        w_first = exp_sims_first / np.sum(exp_sims_first)
        
        alpha = w_first.copy()
        jitter_both = 0.0
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
            h += np.random.normal(0, sig_layer_noise, sandbox.D)
            jitter_both += np.sum((alpha - alpha_prev)**2)
        jitter_both /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Advanced)"] += jitter_both
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

def evaluate_scenarios(seeds=5):
    scenarios = {
        "Symmetric (Baseline)": [0.015, 0.015, 0.015, 0.015],
        "Asymmetric Low-Noise Focus": [0.002, 0.005, 0.015, 0.050],
        "Asymmetric High-Noise SVHN": [0.005, 0.010, 0.020, 0.100],
        "Extreme Noise Asymmetry": [0.001, 0.002, 0.010, 0.200]
    }
    
    for name, noise_scales in scenarios.items():
        print(f"\n--- Scenario: {name} (Layer Noise Scales: {noise_scales}) ---")
        raw_results = {m: {"accuracy": [], "jitter": []} for m in [
            "SABLE", 
            "ChemMerge", 
            "Momentum-Merge (Base)",
            "Momentum-Merge (Advanced)"
        ]}
        for seed in range(seeds):
            res = run_asymmetric_noise_seed(seed=seed, layer_noise_scales=noise_scales)
            for m in raw_results.keys():
                raw_results[m]["accuracy"].append(res[m]["accuracy"])
                raw_results[m]["jitter"].append(res[m]["jitter"])
                
        print("="*95)
        print(f"{'Method':<30} | {'Accuracy Mean (%)':<15} | {'Accuracy Std (%)':<15} | {'Jitter Mean (MSE)':<15}")
        print("="*95)
        for m in raw_results.keys():
            accs = np.array(raw_results[m]["accuracy"])
            jits = np.array(raw_results[m]["jitter"])
            print(f"{m:<30} | {np.mean(accs)*100:<15.2f} | {np.std(accs)*100:<15.2f} | {np.mean(jits):<15.6f}")
        print("="*95)

if __name__ == "__main__":
    evaluate_scenarios(seeds=10)
