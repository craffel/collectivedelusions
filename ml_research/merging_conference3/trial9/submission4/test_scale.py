import os
import numpy as np

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

def run_scale_simulation(K=10, seeds=5):
    # D must be divisible by K, so let's set D = 240 for K = 10 (block_dim = 24)
    D = 240 if K == 10 else 192
    L = 14
    L_frozen = 3
    
    # We will sweep beta to find the optimal beta for K=10
    betas = np.linspace(0.0, 1.0, 6)
    
    sweep_acc = {beta: [] for beta in betas}
    sweep_jit = {beta: [] for beta in betas}
    
    for seed in range(seeds):
        set_seed(seed)
        sandbox = AnalyticalCoordinateSandbox(D=D, L=L, K=K, L_frozen=L_frozen)
        signatures = sandbox.generate_task_signatures()
        
        # Noise scales: let's generate a linearly spaced noise scale from 0.05 to 1.20
        noise_scales = np.linspace(0.05, 1.20, K).tolist()
        
        # Global centroids calibration
        centroids = []
        for k in range(sandbox.K):
            cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
            centroid = np.mean(cal_samples, axis=0)
            centroids.append(centroid / np.linalg.norm(centroid))
        centroids = np.array(centroids)
        
        num_samples = 200
        task_indices = np.random.choice(sandbox.K, size=num_samples)
        stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
        
        gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
        
        # E error ceiling: linearly space error from 0.005 to 0.50
        E = np.linspace(0.005, 0.50, K).tolist()
        lambda_val = 0.40
        tau_mom = 0.005
        
        for beta in betas:
            correct = 0
            total_jitter = 0.0
            
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
            
    print(f"\n--- Scale Analysis results for K = {K} ---")
    print(f"{'Beta':<10} | {'Accuracy Mean (%)':<20} | {'Jitter Mean (MSE)':<20}")
    print("-" * 60)
    for beta in betas:
        acc_mean = np.mean(sweep_acc[beta]) * 100
        jit_mean = np.mean(sweep_jit[beta])
        print(f"{beta:<10.2f} | {acc_mean:<20.2f} | {jit_mean:<20.6f}")

if __name__ == "__main__":
    print("Running Scale Analysis for K = 10 (10 experts)...")
    run_scale_simulation(K=10, seeds=5)
    print("\nRunning Scale Analysis for K = 4 (4 experts) as baseline...")
    run_scale_simulation(K=4, seeds=5)
