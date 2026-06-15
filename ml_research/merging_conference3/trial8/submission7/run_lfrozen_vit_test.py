import os
import numpy as np
import matplotlib.pyplot as plt
from run_real_vit_test import (
    FeatureExtractor,
    generate_shape_image,
    evaluate_routing_accuracy,
    compute_routing_jitter,
    STAGES
)

def run_chemmerge_routing_lfrozen(features, centroids, temp=0.01, delta_t=1.5, k_decay=0.3, L_frozen=3):
    """
    ChemMerge with adjustable frozen early layer boundary L_frozen.
    For l < L_frozen (0-indexed layers 0 to L_frozen-1), ensembling weights are kept uniform (1/K).
    For l >= L_frozen, concentrations evolve based on our non-equilibrium kinetics.
    """
    num_samples = len(features)
    L = len(STAGES)
    K = len(centroids[STAGES[0]])
    
    weights = []
    for i in range(num_samples):
        # Initialize concentrations at L_frozen to uniform
        C_layer = np.ones(K) / K
        sample_weights = []
        
        for l, stage in enumerate(STAGES):
            if l < L_frozen:
                # Keep weights uniform
                alpha = np.ones(K) / K
                sample_weights.append(alpha)
            else:
                h = features[i][stage]
                # 1. Cosine similarity
                sims = np.array([np.dot(h, centroids[stage][k]) for k in range(K)])
                # 2. Subtract max
                sims_stable = sims - np.max(sims)
                # 3. Arrhenius rate equation
                exp_u = np.exp(sims_stable / temp)
                k_rate = exp_u / np.sum(exp_u)
                
                # 4. Discretized Euler step update
                C_next = C_layer + delta_t * (k_rate * (1.0 - C_layer) - k_decay * C_layer)
                C_layer = np.clip(C_next, 0.0, 1.0)
                
                # 5. Law of Mass Action normalization
                alpha = C_layer / np.sum(C_layer)
                sample_weights.append(alpha)
                
        weights.append(sample_weights)
    return np.array(weights)

def run_lfrozen_sweep():
    print("====================================================")
    print("RUNNING L_frozen SENSITIVITY STUDY ON PRE-TRAINED ViT")
    print("====================================================")
    
    extractor = FeatureExtractor()
    shape_types = ["circle", "square", "triangle", "cross"]
    K = len(shape_types)
    
    seeds = list(range(42, 47)) # 5 independent seeds
    num_cal = 32
    num_eval = 25
    
    lfrozen_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    results = {lf: {"acc": [], "jitter": []} for lf in lfrozen_vals}
    
    # We extract features for each seed once and cache them
    cached_data = {}
    
    for seed in seeds:
        print(f"\nExtracting and caching features for Seed {seed}...")
        np.random.seed(seed)
        
        # Calibration features
        cal_features = []
        cal_y = []
        for k, shape in enumerate(shape_types):
            for _ in range(num_cal):
                img = generate_shape_image(shape, seed=seed + _ + k*100)
                feats = extractor.extract(img)
                cal_features.append(feats)
                cal_y.append(k)
                
        # Centroids and expected similarities
        centroids = {stage: [] for stage in STAGES}
        for stage in STAGES:
            for k in range(K):
                k_feats = [cal_features[i][stage] for i in range(len(cal_y)) if cal_y[i] == k]
                centroid = np.mean(k_feats, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                centroids[stage].append(centroid)
            centroids[stage] = np.array(centroids[stage])
            
        # Evaluation features
        eval_features = []
        eval_y = []
        for k, shape in enumerate(shape_types):
            for _ in range(num_eval):
                img = generate_shape_image(shape, seed=seed + _ + k*200 + 1000)
                feats = extractor.extract(img)
                eval_features.append(feats)
                eval_y.append(k)
                
        eval_y = np.array(eval_y)
        
        cached_data[seed] = {
            "centroids": centroids,
            "eval_features": eval_features,
            "eval_y": eval_y
        }
        
    extractor.close()
    
    # Sweep L_frozen across cached data
    print("\nRunning L_frozen sweeps...")
    for lf in lfrozen_vals:
        for seed in seeds:
            data = cached_data[seed]
            centroids = data["centroids"]
            eval_features = data["eval_features"]
            eval_y = data["eval_y"]
            
            weights = run_chemmerge_routing_lfrozen(
                eval_features, centroids, temp=0.01, delta_t=1.5, k_decay=0.3, L_frozen=lf
            )
            
            acc = evaluate_routing_accuracy(weights, eval_y)
            jitter = compute_routing_jitter(weights)
            
            results[lf]["acc"].append(acc)
            results[lf]["jitter"].append(jitter)
            
    # Print results
    print("\n====================================================")
    print("RESULTS: L_frozen SENSITIVITY SWEEP")
    print("====================================================")
    print(f"{'L_frozen':<10} | {'Routing Accuracy (%)':<22} | {'Routing Jitter':<20}")
    print("-" * 60)
    for lf in lfrozen_vals:
        mean_acc = np.mean(results[lf]["acc"]) * 100
        std_acc = np.std(results[lf]["acc"]) * 100
        mean_jit = np.mean(results[lf]["jitter"])
        std_jit = np.std(results[lf]["jitter"])
        print(f"{lf:<10} | {mean_acc:5.2f}% +/- {std_acc:4.2f}% | {mean_jit:6.4f} +/- {std_jit:6.4f}")
    print("====================================================")
    
    # Plot results
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    lf_means_acc = [np.mean(results[lf]["acc"]) * 100 for lf in lfrozen_vals]
    lf_stds_acc = [np.std(results[lf]["acc"]) * 100 for lf in lfrozen_vals]
    lf_means_jit = [np.mean(results[lf]["jitter"]) for lf in lfrozen_vals]
    lf_stds_jit = [np.std(results[lf]["jitter"]) for lf in lfrozen_vals]
    
    color = "#1f77b4"
    ax1.set_xlabel(r"Frozen Depth Boundary ($L_{\text{frozen}}$)", fontsize=12)
    ax1.set_ylabel("Routing Accuracy (%)", color=color, fontsize=12)
    ax1.errorbar(lfrozen_vals, lf_means_acc, yerr=lf_stds_acc, fmt="o-", color=color, linewidth=2.5, capsize=4, label="Accuracy")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel("Routing Jitter (Lower is Better)", color=color, fontsize=12)
    ax2.errorbar(lfrozen_vals, lf_means_jit, yerr=lf_stds_jit, fmt="s-", color=color, linewidth=2, capsize=4, label="Jitter")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(r"Sensitivity of ChemMerge Routing to Frozen Layer Boundary ($L_{\text{frozen}}$)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission/results", exist_ok=True)
    plt.savefig("results/lfrozen_sensitivity.png", dpi=150)
    plt.savefig("submission/results/lfrozen_sensitivity.png", dpi=150)
    plt.close()
    print("Plots saved successfully to results/lfrozen_sensitivity.png and submission/results/lfrozen_sensitivity.png")

if __name__ == "__main__":
    run_lfrozen_sweep()
