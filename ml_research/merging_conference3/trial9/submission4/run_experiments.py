import os
import numpy as np
import json
import matplotlib.pyplot as plt

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

def run_simulation_seed(seed, num_samples=200):
    set_seed(seed)
    sandbox = AnalyticalCoordinateSandbox()
    signatures = sandbox.generate_task_signatures()
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    # Pre-compute task centroids (UNC) using 64 calibration samples
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
    
    # Generate serving stream (heterogeneous shuffled stream)
    task_indices = np.random.choice(sandbox.K, size=num_samples)
    stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
    
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    
    methods = ["Expert Ceiling", "Uniform Merging", "SABLE", "ChemMerge", "Momentum-Merge", "Momentum-Merge (Advanced)"]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    best_beta = 0.60
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        # Record the RNG state right before running the ensembling methods on this sample
        # to ensure perfect synchronization across all compared methods!
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

def run_multi_seed_simulation(seeds=10):
    raw_results = {m: {"accuracy": [], "jitter": []} for m in ["Expert Ceiling", "Uniform Merging", "SABLE", "ChemMerge", "Momentum-Merge", "Momentum-Merge (Advanced)"]}
    
    for seed in range(seeds):
        res = run_simulation_seed(seed=seed)
        for m in raw_results.keys():
            raw_results[m]["accuracy"].append(res[m]["accuracy"])
            raw_results[m]["jitter"].append(res[m]["jitter"])
            
    real_results = {}
    for m in raw_results.keys():
        accs = np.array(raw_results[m]["accuracy"])
        jits = np.array(raw_results[m]["jitter"])
        
        real_results[m] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "jitter_mean": float(np.mean(jits)),
            "jitter_std": float(np.std(jits)),
            "all_accuracy": accs.tolist(),
            "all_jitter": jits.tolist()
        }
        
    return real_results

def run_beta_sweep_seeds(seeds=5):
    betas = np.linspace(0.0, 1.0, 11)
    sweep_acc = {beta: [] for beta in betas}
    sweep_jit = {beta: [] for beta in betas}
    
    for seed in range(seeds):
        set_seed(seed)
        sandbox = AnalyticalCoordinateSandbox()
        signatures = sandbox.generate_task_signatures()
        noise_scales = [0.05, 0.15, 0.40, 1.20]
        
        centroids = []
        for k in range(sandbox.K):
            cal_samples = [signatures[k] + np.random.normal(0, noise_scales[k], sandbox.D) for _ in range(64)]
            centroid = np.mean(cal_samples, axis=0)
            centroids.append(centroid / np.linalg.norm(centroid))
        centroids = np.array(centroids)
        
        num_samples = 200
        task_indices = np.random.choice(sandbox.K, size=num_samples)
        stream_samples = [signatures[k] + np.random.normal(0, noise_scales[k], sandbox.D) for k in task_indices]
        
        gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
        E = [0.005, 0.03, 0.08, 0.50]
        lambda_val = 0.40
        
        correct = {beta: 0 for beta in betas}
        total_jitter = {beta: 0.0 for beta in betas}
        tau_mom = 0.005
        
        for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
            init_rng_state = np.random.get_state()
            
            for beta in betas:
                np.random.set_state(init_rng_state)
                h = h_init.copy()
                alpha = np.full(sandbox.K, 1.0 / sandbox.K)
                jitter_mom = 0.0
                
                for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                    similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                    exp_sims = np.exp(similarities / tau_mom)
                    w = exp_sims / np.sum(exp_sims)
                    
                    alpha_prev = alpha.copy()
                    alpha = (1.0 - beta) * w + beta * alpha_prev
                    
                    blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                    h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                    h += np.random.normal(0, 0.015, sandbox.D)
                    jitter_mom += np.sum((alpha - alpha_prev)**2)
                    
                jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
                total_jitter[beta] += jitter_mom
                
                d = np.linalg.norm(h - signatures[k_true])**2
                p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
                if np.random.random() < p_correct:
                    correct[beta] += 1
                    
        for beta in betas:
            sweep_acc[beta].append(correct[beta] / num_samples)
            sweep_jit[beta].append(total_jitter[beta] / num_samples)
            
    real_sweep = []
    for beta in betas:
        accs = np.array(sweep_acc[beta])
        jits = np.array(sweep_jit[beta])
        
        real_sweep.append({
            "beta": float(beta),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "jitter_mean": float(np.mean(jits)),
            "jitter_std": float(np.std(jits))
        })
        
    return real_sweep

def generate_plots(results, sweep_results):
    os.makedirs("results", exist_ok=True)
    
    # 1. Performance Comparison Bar Chart (Accuracy & Jitter)
    methods = ["Uniform Merging", "SABLE", "ChemMerge", "Momentum-Merge", "Momentum-Merge (Advanced)", "Expert Ceiling"]
    acc_means = [results[m]["accuracy_mean"] * 100 for m in methods]
    acc_stds = [results[m]["accuracy_std"] * 100 for m in methods]
    jit_means = [results[m]["jitter_mean"] for m in methods]
    jit_stds = [results[m]["jitter_std"] for m in methods]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = "tab:blue"
    ax1.set_xlabel("Ensembling Method", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Joint Classification Accuracy (%)", color=color, fontweight="bold", fontsize=12)
    bars = ax1.bar(methods, acc_means, yerr=acc_stds, color=color, alpha=0.6, label="Accuracy", width=0.4, capsize=5)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 100)
    
    # Grid lines
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = "tab:red"
    ax2.set_ylabel("Layer-to-Layer Routing Jitter (MSE)", color=color, fontweight="bold", fontsize=12)
    dots = ax2.errorbar(methods, jit_means, yerr=jit_stds, color=color, fmt="o-", linewidth=2, label="Routing Jitter", capsize=5)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.01, 0.08)
    
    plt.title("Performance Comparison inside Analytical Coordinate Sandbox (ICS)", fontweight="bold", fontsize=14, pad=15)
    fig.tight_layout()  
    plt.savefig("results/performance_comparison.png", dpi=300)
    plt.close()
    
    # 2. Beta Pareto Sweep
    betas = [res["beta"] for res in sweep_results]
    sweep_accs = [res["accuracy_mean"] * 100 for res in sweep_results]
    sweep_acc_stds = [res["accuracy_std"] * 100 for res in sweep_results]
    sweep_jits = [res["jitter_mean"] for res in sweep_results]
    sweep_jit_stds = [res["jitter_std"] for res in sweep_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = "tab:blue"
    ax1.set_xlabel("Momentum Coefficient ($\\beta$)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Joint Classification Accuracy (%)", color=color, fontweight="bold", fontsize=12)
    ax1.errorbar(betas, sweep_accs, yerr=sweep_acc_stds, color=color, fmt="o-", linewidth=2, label="Accuracy", capsize=3)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(50, 85)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Layer-to-Layer Routing Jitter (MSE)", color=color, fontweight="bold", fontsize=12)
    ax2.errorbar(betas, sweep_jits, yerr=sweep_jit_stds, color=color, fmt="s-", linewidth=2, label="Routing Jitter", capsize=3)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.01, 0.08)
    
    plt.title("Stability-Accuracy Pareto Sweep over Momentum Coefficient ($\\beta$)", fontweight="bold", fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig("results/beta_pareto_sweep.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("--- Running 10-Seed Analytical Coordinate Sandbox Simulation ---")
    results = run_multi_seed_simulation(seeds=10)
    
    print("\n--- Running Momentum-Merge Beta Coefficient Sweep ---")
    sweep_results = run_beta_sweep_seeds(seeds=5)
    
    print("\n--- Saving Metrics and Generating Figures ---")
    generate_plots(results, sweep_results)
    
    # Save results as JSON
    final_metrics = {
        "multi_seed_results": results,
        "beta_sweep": sweep_results
    }
    with open("results/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    print("\nSimulation Completed Successfully! Metrics saved to 'results/metrics.json'")
    print("Figures generated:")
    print(" - 'results/performance_comparison.png'")
    print(" - 'results/beta_pareto_sweep.png'")
    
    # Print clean results table
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Accuracy Mean (%)':<15} | {'Accuracy Std (%)':<15} | {'Jitter Mean (MSE)':<15}")
    print("="*80)
    for m, res in results.items():
        print(f"{m:<20} | {res['accuracy_mean']*100:<15.2f} | {res['accuracy_std']*100:<15.2f} | {res['jitter_mean']:<15.6f}")
    print("="*80)
