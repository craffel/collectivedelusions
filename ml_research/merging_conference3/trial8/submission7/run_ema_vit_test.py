import os
import numpy as np
import matplotlib.pyplot as plt
from run_real_vit_test import (
    FeatureExtractor,
    generate_shape_image,
    evaluate_routing_accuracy,
    compute_routing_jitter,
    run_sable_routing,
    run_sps_zca_routing,
    run_chemmerge_routing,
    STAGES
)

def run_static_ema_routing(features, centroids, beta=0.3, temp=0.05):
    """
    Static EMA routing baseline.
    alpha^(l) = beta * w_stateless + (1 - beta) * alpha^(l-1)
    """
    num_samples = len(features)
    L = len(STAGES)
    K = len(centroids[STAGES[0]])
    
    weights = []
    for i in range(num_samples):
        # Initialize ensembling weights at layer 1: alpha = 1/K
        alpha = np.ones(K) / K
        sample_weights = []
        
        for l, stage in enumerate(STAGES):
            h = features[i][stage]
            # Compare to layer-specific centroids
            sims = np.array([np.dot(h, centroids[stage][k]) for k in range(K)])
            sims_stable = sims - np.max(sims)
            exp_sims = np.exp(sims_stable / temp)
            w = exp_sims / np.sum(exp_sims)
            
            # EMA step
            alpha_next = beta * w + (1.0 - beta) * alpha
            alpha = alpha_next
            sample_weights.append(alpha)
        weights.append(sample_weights)
    return np.array(weights) # Shape: [N, L, K]

def run_ema_vit_comparison():
    print("Evaluating Static EMA Routing Baseline vs. ChemMerge on pre-trained ViT-B/16 features...")
    extractor = FeatureExtractor()
    shape_types = ["circle", "square", "triangle", "cross"]
    K = len(shape_types)
    
    seeds = list(range(42, 47)) # 5 independent evaluation seeds
    num_cal = 32
    num_eval = 25
    
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = {
        "SABLE (temp=0.05)": {"acc": [], "jitter": []},
        "SABLE (temp=0.01)": {"acc": [], "jitter": []},
        "ChemMerge (temp=0.01)": {"acc": [], "jitter": []},
        "SPS-ZCA": {"acc": [], "jitter": []},
    }
    for b in betas:
        results[f"Static EMA (b={b}, temp=0.01)"] = {"acc": [], "jitter": []}
        
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        np.random.seed(seed)
        
        # 1. Generate Calibration Dataset
        cal_features = []
        cal_y = []
        for k, shape in enumerate(shape_types):
            for _ in range(num_cal):
                img = generate_shape_image(shape, seed=seed + _ + k*100)
                feats = extractor.extract(img)
                cal_features.append(feats)
                cal_y.append(k)
                
        # 2. Extract Layer-Specific Centroids
        centroids = {stage: [] for stage in STAGES}
        for stage in STAGES:
            for k in range(K):
                k_feats = [cal_features[i][stage] for i in range(len(cal_y)) if cal_y[i] == k]
                centroid = np.mean(k_feats, axis=0)
                centroid = centroid / np.linalg.norm(centroid) # normalize
                centroids[stage].append(centroid)
            centroids[stage] = np.array(centroids[stage])
            
        expected_sims = {stage: [] for stage in STAGES}
        for stage in STAGES:
            for k in range(K):
                k_feats = [cal_features[i][stage] for i in range(len(cal_y)) if cal_y[i] == k]
                sims = [np.dot(f, centroids[stage][k]) for f in k_feats]
                expected_sims[stage].append(np.mean(sims))
            expected_sims[stage] = np.array(expected_sims[stage])
            
        # 3. Generate Evaluation Dataset
        eval_features = []
        eval_y = []
        for k, shape in enumerate(shape_types):
            for _ in range(num_eval):
                img = generate_shape_image(shape, seed=seed + _ + k*200 + 1000)
                feats = extractor.extract(img)
                eval_features.append(feats)
                eval_y.append(k)
                
        eval_y = np.array(eval_y)
        
        # 4. Evaluate Routers
        # SABLE default (temp=0.05)
        weights_sable_05 = run_sable_routing(eval_features, centroids, temp=0.05)
        results["SABLE (temp=0.05)"]["acc"].append(evaluate_routing_accuracy(weights_sable_05, eval_y))
        results["SABLE (temp=0.05)"]["jitter"].append(compute_routing_jitter(weights_sable_05))
        
        # SABLE high-selectivity (temp=0.01)
        weights_sable_01 = run_sable_routing(eval_features, centroids, temp=0.01)
        results["SABLE (temp=0.01)"]["acc"].append(evaluate_routing_accuracy(weights_sable_01, eval_y))
        results["SABLE (temp=0.01)"]["jitter"].append(compute_routing_jitter(weights_sable_01))
        
        # SPS-ZCA
        weights_zca = run_sps_zca_routing(eval_features, centroids, expected_sims, temp=0.001)
        results["SPS-ZCA"]["acc"].append(evaluate_routing_accuracy(weights_zca, eval_y))
        results["SPS-ZCA"]["jitter"].append(compute_routing_jitter(weights_zca))
        
        # ChemMerge (temp=0.01)
        weights_chem = run_chemmerge_routing(eval_features, centroids, temp=0.01, delta_t=1.5, k_decay=0.3)
        results["ChemMerge (temp=0.01)"]["acc"].append(evaluate_routing_accuracy(weights_chem, eval_y))
        results["ChemMerge (temp=0.01)"]["jitter"].append(compute_routing_jitter(weights_chem))
        
        # Static EMA baselines
        for b in betas:
            weights_ema = run_static_ema_routing(eval_features, centroids, beta=b, temp=0.01)
            results[f"Static EMA (b={b}, temp=0.01)"]["acc"].append(evaluate_routing_accuracy(weights_ema, eval_y))
            results[f"Static EMA (b={b}, temp=0.01)"]["jitter"].append(compute_routing_jitter(weights_ema))
            
    extractor.close()
    
    print("\n====================================================")
    print("AGGREGATED PERFORMANCE: EMA BASELINES VS CHEMMERGE")
    print("====================================================")
    print(f"{'Method':<30} | {'Routing Accuracy (%)':<22} | {'Routing Jitter':<20}")
    print("-" * 80)
    for m in results:
        m_acc = np.mean(results[m]["acc"]) * 100
        std_acc = np.std(results[m]["acc"]) * 100
        m_jit = np.mean(results[m]["jitter"])
        std_jit = np.std(results[m]["jitter"])
        print(f"{m:<30} | {m_acc:5.2f}% +/- {std_acc:4.2f}% | {m_jit:6.4f} +/- {std_jit:6.4f}")
    print("====================================================")

if __name__ == "__main__":
    run_ema_vit_comparison()
