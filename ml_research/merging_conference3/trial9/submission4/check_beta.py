import numpy as np
from run_experiments import run_beta_sweep_seeds

print("Running 5 seeds beta sweep uncalibrated...")
# We temporarily modify the return inside run_beta_sweep_seeds to not calibrate,
# or we can write a clean loop to do it ourselves in this script.

def run_real_beta_sweep(seeds=5):
    import numpy as np
    from run_experiments import AnalyticalCoordinateSandbox
    betas = np.linspace(0.0, 1.0, 11)
    sweep_acc = {beta: [] for beta in betas}
    sweep_jit = {beta: [] for beta in betas}
    
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
        
        for beta in betas:
            correct = 0
            total_jitter = 0.0
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
                    alpha = (1.0 - beta) * w + beta * alpha_prev
                    
                    blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                    h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                    h += np.random.normal(0, 0.015, sandbox.D)
                    jitter_mom += np.sum((alpha - alpha_prev)**2)
                    
                jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
                total_jitter += jitter_mom
                
                d = np.linalg.norm(h - signatures[k_true])**2
                p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
                if np.random.random() < p_correct:
                    correct += 1
                    
            sweep_acc[beta].append(correct / num_samples)
            sweep_jit[beta].append(total_jitter / num_samples)
            
    print("\n" + "="*80)
    print(f"{'Beta':<10} | {'Accuracy Mean (%)':<18} | {'Accuracy Std (%)':<18} | {'Jitter Mean (MSE)':<18}")
    print("="*80)
    for beta in betas:
        accs = np.array(sweep_acc[beta])
        jits = np.array(sweep_jit[beta])
        print(f"{beta:<10.2f} | {np.mean(accs)*100:<18.2f} | {np.std(accs)*100:<18.2f} | {np.mean(jits):<18.6f}")
    print("="*80)

run_real_beta_sweep()
