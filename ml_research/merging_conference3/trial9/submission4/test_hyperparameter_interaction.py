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

def run_interaction_seed(seed, beta, tau, num_samples=1000):
    set_seed(seed)
    sandbox = AnalyticalCoordinateSandbox()
    signatures = sandbox.generate_task_signatures()
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
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
    
    correct = 0
    total_jitter = 0.0
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_val = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - beta) * w + beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_val += np.sum((alpha - alpha_prev)**2)
        jitter_val /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter += jitter_val
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct += 1
            
    return correct / num_samples, total_jitter / num_samples

def sweep_interaction(seeds=5):
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    taus = [0.005, 0.010, 0.050, 0.100, 0.150, 0.200, 0.300]
    
    acc_matrix = np.zeros((len(betas), len(taus)))
    jit_matrix = np.zeros((len(betas), len(taus)))
    
    for idx_b, beta in enumerate(betas):
        for idx_t, tau in enumerate(taus):
            accs, jits = [], []
            for seed in range(seeds):
                acc, jit = run_interaction_seed(seed=seed, beta=beta, tau=tau)
                accs.append(acc)
                jits.append(jit)
            acc_matrix[idx_b, idx_t] = np.mean(accs) * 100
            jit_matrix[idx_b, idx_t] = np.mean(jits)
            
    print("\n" + "="*115)
    print("JOINT HYPERPARAMETER SWEEP: MEAN JOINT ACCURACY (%)")
    print("="*115)
    header = f"{'Beta \\ Temperature':<20} | " + " | ".join([f"{t:<8.3f}" for t in taus])
    print(header)
    print("-"*115)
    for idx_b, beta in enumerate(betas):
        row = f"{beta:<20.1f} | " + " | ".join([f"{acc_matrix[idx_b, idx_t]:<8.2f}" for idx_t, _ in enumerate(taus)])
        print(row)
    print("="*115)
    
    print("\n" + "="*115)
    print("JOINT HYPERPARAMETER SWEEP: ROUTING JITTER (MSE)")
    print("="*115)
    print(header)
    print("-"*115)
    for idx_b, beta in enumerate(betas):
        row = f"{beta:<20.1f} | " + " | ".join([f"{jit_matrix[idx_b, idx_t]:<8.6f}" for idx_t, _ in enumerate(taus)])
        print(row)
    print("="*115)

if __name__ == "__main__":
    sweep_interaction(seeds=10)
