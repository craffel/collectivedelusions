import os
import numpy as np
import json
from run_experiments import AnalyticalCoordinateSandbox, set_seed

def run_synchronized_temp_sweep(seeds=10, num_samples=1000):
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    
    taus = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    methods = ["SABLE", "ChemMerge", "Momentum-Merge"]
    
    # Structure of results: method -> tau -> accuracy & jitter lists
    raw_results = {m: {str(tau): {"accuracy": [], "jitter": []} for tau in taus} for m in methods}
    
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
        
        # Generate stream
        task_indices = np.random.choice(sandbox.K, size=num_samples)
        stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
        gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
        
        for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
            init_rng_state = np.random.get_state()
            
            for m in methods:
                for tau in taus:
                    np.random.set_state(init_rng_state)
                    h = h_init.copy()
                    
                    if m == "SABLE":
                        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
                        jitter = 0.0
                        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                            similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                            exp_sims = np.exp(similarities / tau)
                            alpha_prev = alpha.copy()
                            alpha = exp_sims / np.sum(exp_sims)
                            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                            h += np.random.normal(0, 0.015, sandbox.D)
                            jitter += np.sum((alpha - alpha_prev)**2)
                        jitter /= (sandbox.L - sandbox.L_frozen - 1)
                        raw_results[m][str(tau)]["jitter"].append(jitter)
                        
                    elif m == "ChemMerge":
                        C = np.full(sandbox.K, 1.0 / sandbox.K)
                        dt = 1.5
                        k_decay = 0.3
                        jitter = 0.0
                        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                            similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                            exp_sims = np.exp(similarities / tau)
                            k_rates = exp_sims / np.sum(exp_sims)
                            lambdas = np.exp(-(k_rates + k_decay) * dt)
                            C_star = k_rates / (k_rates + k_decay)
                            C = lambdas * C + (1.0 - lambdas) * C_star
                            alpha_prev = alpha.copy() if l > sandbox.L_frozen + 1 else np.full(sandbox.K, 1.0 / sandbox.K)
                            alpha = C / np.sum(C)
                            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                            h += np.random.normal(0, 0.015, sandbox.D)
                            jitter += np.sum((alpha - alpha_prev)**2)
                        jitter /= (sandbox.L - sandbox.L_frozen - 1)
                        raw_results[m][str(tau)]["jitter"].append(jitter)
                        
                    elif m == "Momentum-Merge":
                        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
                        jitter = 0.0
                        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                            similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                            exp_sims = np.exp(similarities / tau)
                            w = exp_sims / np.sum(exp_sims)
                            alpha_prev = alpha.copy()
                            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
                            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                            h += np.random.normal(0, 0.015, sandbox.D)
                            jitter += np.sum((alpha - alpha_prev)**2)
                        jitter /= (sandbox.L - sandbox.L_frozen - 1)
                        raw_results[m][str(tau)]["jitter"].append(jitter)
                    
                    d = np.linalg.norm(h - signatures[k_true])**2
                    p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
                    is_correct = 1 if np.random.random() < p_correct else 0
                    raw_results[m][str(tau)]["accuracy"].append(is_correct)
                    
    # Now compute means and stds across samples / seeds
    sweep_results = {m: {} for m in methods}
    for m in methods:
        for tau in taus:
            # We want to aggregate by seed
            # Each seed had num_samples samples
            accs_by_seed = []
            jits_by_seed = []
            for s in range(seeds):
                start_idx = s * num_samples
                end_idx = (s + 1) * num_samples
                seed_accs = raw_results[m][str(tau)]["accuracy"][start_idx:end_idx]
                seed_jits = raw_results[m][str(tau)]["jitter"][start_idx:end_idx]
                accs_by_seed.append(np.mean(seed_accs))
                jits_by_seed.append(np.mean(seed_jits))
                
            sweep_results[m][str(tau)] = {
                "accuracy_mean": float(np.mean(accs_by_seed)),
                "accuracy_std": float(np.std(accs_by_seed)),
                "jitter_mean": float(np.mean(jits_by_seed)),
                "jitter_std": float(np.std(jits_by_seed))
            }
            print(f"Method: {m:15s} | Tau: {tau:5.3f} | Accuracy: {np.mean(accs_by_seed)*100:6.2f}% | Jitter: {np.mean(jits_by_seed):8.6f}")
            
    os.makedirs("results", exist_ok=True)
    with open("results/temperature_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    print("Sweep complete. Results saved to results/temperature_sweep.json")

if __name__ == "__main__":
    run_synchronized_temp_sweep()
