import numpy as np
from run_experiments import AnalyticalCoordinateSandbox

def evaluate_momentum_merge_beta(beta_val, seeds=10):
    raw_accuracies = []
    raw_jitters = []
    
    for seed in range(seeds):
        np.random.seed(seed)
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
        
        # Momentum-Merge
        h = stream_samples[0].copy() # temporary initialization
        correct_mom = 0
        total_jitter_mom = 0.0
        tau_mom = 0.005
        
        for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
            h = h_init.copy()
            alpha = np.full(sandbox.K, 1.0 / sandbox.K)
            jitter_mom = 0.0
            
            for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                similarities = np.array([np.dot(h, centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                exp_sims = np.exp(similarities / tau_mom)
                w = exp_sims / np.sum(exp_sims)
                
                alpha_prev = alpha.copy()
                alpha = (1.0 - beta_val) * w + beta_val * alpha_prev
                
                blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                h += np.random.normal(0, 0.015, sandbox.D)
                jitter_mom += np.sum((alpha - alpha_prev)**2)
                
            jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
            total_jitter_mom += jitter_mom
            
            d = np.linalg.norm(h - signatures[k_true])**2
            p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
            if np.random.random() < p_correct:
                correct_mom += 1
                
        raw_accuracies.append(correct_mom / num_samples)
        raw_jitters.append(total_jitter_mom / num_samples)
        
    print(f"Beta {beta_val:.2f} | Accuracy: {np.mean(raw_accuracies)*100:.2f}% +- {np.std(raw_accuracies)*100:.2f}% | Jitter: {np.mean(raw_jitters):.6f}")

for b in [0.40, 0.50, 0.60]:
    evaluate_momentum_merge_beta(b)
