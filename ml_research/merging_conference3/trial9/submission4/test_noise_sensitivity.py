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

def run_noise_sensitivity_seed(seed, layer_noise, num_samples=200):
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
    
    # Generate serving stream
    task_indices = np.random.choice(sandbox.K, size=num_samples)
    stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
    
    gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    tau_mom = 0.005
    
    methods = [
        "Momentum-Merge (Base - Uniform Init)",
        "Momentum-Merge (Raw Boundary Init)"
    ]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        init_rng_state = np.random.get_state()
        
        # 1. Uniform Init (Base)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_base = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, layer_noise, sandbox.D)
            jitter_base += np.sum((alpha - alpha_prev)**2)
        jitter_base /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Base - Uniform Init)"] += jitter_base
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Base - Uniform Init)"] += 1
            
        # 2. Raw Boundary Init (Eq 10)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        l_first = sandbox.L_frozen + 1
        sims_first = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
        exp_sims_first = np.exp(sims_first / tau_mom)
        w_first = exp_sims_first / np.sum(exp_sims_first)
        
        alpha = w_first.copy()
        jitter_raw = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            if l == l_first:
                alpha = w_first.copy()
            else:
                alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, layer_noise, sandbox.D)
            jitter_raw += np.sum((alpha - alpha_prev)**2)
        jitter_raw /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Raw Boundary Init)"] += jitter_raw
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Raw Boundary Init)"] += 1

    seed_results = {}
    for m in methods:
        seed_results[m] = {
            "accuracy": correct[m] / num_samples,
            "jitter": total_jitter[m] / num_samples
        }
    return seed_results

def sweep_noise(seeds=10):
    noise_scales = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060]
    methods = [
        "Momentum-Merge (Base - Uniform Init)",
        "Momentum-Merge (Raw Boundary Init)"
    ]
    
    print("\n" + "="*115)
    print(f"{'Layer Noise (sigma_layer)':<25} | {'Uniform Acc (%)':<15} | {'Raw Init Acc (%)':<15} | {'Uniform Jitter':<15} | {'Raw Init Jitter':<15}")
    print("="*115)
    
    for sigma in noise_scales:
        raw_results = {m: {"accuracy": [], "jitter": []} for m in methods}
        for seed in range(seeds):
            res = run_noise_sensitivity_seed(seed=seed, layer_noise=sigma)
            for m in methods:
                raw_results[m]["accuracy"].append(res[m]["accuracy"])
                raw_results[m]["jitter"].append(res[m]["jitter"])
                
        uniform_acc = np.mean(raw_results["Momentum-Merge (Base - Uniform Init)"]["accuracy"]) * 100
        raw_acc = np.mean(raw_results["Momentum-Merge (Raw Boundary Init)"]["accuracy"]) * 100
        uniform_jit = np.mean(raw_results["Momentum-Merge (Base - Uniform Init)"]["jitter"])
        raw_jit = np.mean(raw_results["Momentum-Merge (Raw Boundary Init)"]["jitter"])
        
        print(f"{sigma:<25.3f} | {uniform_acc:<15.2f} | {raw_acc:<15.2f} | {uniform_jit:<15.6f} | {raw_jit:<15.6f}")
    print("="*115)

if __name__ == "__main__":
    sweep_noise(seeds=10)
