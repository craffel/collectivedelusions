import os
import numpy as np
import json
from run_experiments import AnalyticalCoordinateSandbox, set_seed

def run_synchronized_simulation(seeds=10, num_samples=200):
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    
    methods = ["Expert Ceiling", "Uniform Merging", "SABLE", "ChemMerge", "Momentum-Merge", "Momentum-Merge (Advanced)"]
    raw_results = {m: {"accuracy": [], "jitter": []} for m in methods}
    
    for seed in range(seeds):
        set_seed(seed)
        sandbox = AnalyticalCoordinateSandbox()
        signatures = sandbox.generate_task_signatures()
        
        # Pre-compute Global Centroids
        centroids = []
        for k in range(sandbox.K):
            cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
            centroid = np.mean(cal_samples, axis=0)
            centroids.append(centroid / np.linalg.norm(centroid))
        centroids = np.array(centroids)
        
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
        
        task_indices = np.random.choice(sandbox.K, size=num_samples)
        stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
        
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
            tau_sable = 0.15
            for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
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
                
            # 4. ChemMerge
            np.random.set_state(init_rng_state)
            h = h_init.copy()
            C = np.full(sandbox.K, 1.0 / sandbox.K)
            dt = 1.5
            k_decay = 0.3
            jitter_chem = 0.0
            tau_chem = 0.005
            for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                exp_sims = np.exp(similarities / tau_chem)
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
                
            # 5. Momentum-Merge
            np.random.set_state(init_rng_state)
            h = h_init.copy()
            alpha = np.full(sandbox.K, 1.0 / sandbox.K)
            jitter_mom = 0.0
            tau_mom = 0.005
            for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                exp_sims = np.exp(similarities / tau_mom)
                w = exp_sims / np.sum(exp_sims)
                alpha_prev = alpha.copy()
                alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
                blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                h += np.random.normal(0, 0.015, sandbox.D)
                jitter_mom += np.sum((alpha - alpha_prev)**2)
            jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
            total_jitter["Momentum-Merge"] += jitter_mom
            d = np.linalg.norm(h - signatures[k_true])**2
            p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
            if np.random.random() < p_correct:
                correct["Momentum-Merge"] += 1

            # 6. Momentum-Merge (Advanced)
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
                h = (1 - gammas[l]) * h + blended_sig * gammas[l]
                h += np.random.normal(0, 0.015, sandbox.D)
                jitter_adv += np.sum((alpha - alpha_prev)**2)
            jitter_adv /= (sandbox.L - sandbox.L_frozen - 1)
            total_jitter["Momentum-Merge (Advanced)"] += jitter_adv
            d = np.linalg.norm(h - signatures[k_true])**2
            p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
            if np.random.random() < p_correct:
                correct["Momentum-Merge (Advanced)"] += 1
                
        for m in methods:
            raw_results[m]["accuracy"].append(correct[m] / num_samples)
            raw_results[m]["jitter"].append(total_jitter[m] / num_samples)
            
    print("="*95)
    print(f"{'Method / Variant':<35} | {'Accuracy Mean (%)':<15} | {'Accuracy Std (%)':<15} | {'Jitter Mean (MSE)':<15}")
    print("="*95)
    for m in methods:
        accs = np.array(raw_results[m]["accuracy"])
        jits = np.array(raw_results[m]["jitter"])
        print(f"{m:<35} | {np.mean(accs)*100:<15.2f} | {np.std(accs)*100:<15.2f} | {np.mean(jits):<15.6f}")
    print("="*95)

if __name__ == "__main__":
    run_synchronized_simulation()
