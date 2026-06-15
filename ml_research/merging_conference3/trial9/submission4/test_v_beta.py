import numpy as np

def set_seed(seed):
    np.random.seed(seed)

class AnalyticalCoordinateSandbox:
    def __init__(self, D=192, L=14, K=4, L_frozen=3):
        self.D = D
        self.L = L
        self.K = K
        self.L_frozen = L_frozen
        
    def generate_task_signatures(self):
        signatures = []
        block_dim = self.D // self.K
        for k in range(self.K):
            sig = np.zeros(self.D)
            start_idx = k * block_dim
            end_idx = (k + 1) * block_dim
            sig[start_idx:end_idx] = np.random.normal(1.0, 0.1, block_dim)
            sig = sig / np.linalg.norm(sig)
            signatures.append(sig)
        return np.array(signatures)
        
    def generate_sample(self, task_idx, signatures, noise_scales):
        # Generate with task-specific representation noise scale
        sig = signatures[task_idx]
        noise = np.random.normal(0, noise_scales[task_idx], self.D)
        return sig + noise

def run_v_beta_test(seeds=10, num_samples=1000):
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    tau_mom = 0.005
    
    # Pre-calculate V-shaped beta schedule for layers 4 to 14
    # Layers: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    # From 4 to 9: 0.8 down to 0.4
    # From 9 to 14: 0.4 up to 0.8
    v_betas = {}
    for l in range(4, 15):
        if l <= 9:
            v_betas[l] = 0.8 - (l - 4) * 0.08
        else:
            v_betas[l] = 0.4 + (l - 9) * 0.08
            
    methods = ["Momentum-Merge (Constant beta=0.60)", "Momentum-Merge (V-shaped beta)"]
    acc_results = {m: [] for m in methods}
    jit_results = {m: [] for m in methods}
    
    for seed in range(seeds):
        set_seed(seed)
        sandbox = AnalyticalCoordinateSandbox()
        signatures = sandbox.generate_task_signatures()
        
        # Pre-compute Global Centroids
        global_centroids = []
        for k in range(sandbox.K):
            cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
            centroid = np.mean(cal_samples, axis=0)
            global_centroids.append(centroid / np.linalg.norm(centroid))
        global_centroids = np.array(global_centroids)
        
        # Generate stream
        task_indices = np.random.choice(sandbox.K, size=num_samples)
        stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
        
        correct = {m: 0 for m in methods}
        total_jitter = {m: 0.0 for m in methods}
        gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
        
        for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
            init_rng_state = np.random.get_state()
            
            # 1. Constant beta=0.60
            np.random.set_state(init_rng_state)
            h = h_init.copy()
            alpha = np.full(sandbox.K, 1.0 / sandbox.K)
            jitter_const = 0.0
            for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                exp_sims = np.exp(similarities / tau_mom)
                w = exp_sims / np.sum(exp_sims)
                alpha_prev = alpha.copy()
                alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
                blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                h += np.random.normal(0, 0.015, sandbox.D)
                jitter_const += np.sum((alpha - alpha_prev)**2)
            jitter_const /= (sandbox.L - sandbox.L_frozen - 1)
            total_jitter["Momentum-Merge (Constant beta=0.60)"] += jitter_const
            d = np.linalg.norm(h - signatures[k_true])**2
            p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
            if np.random.random() < p_correct:
                correct["Momentum-Merge (Constant beta=0.60)"] += 1
                
            # 2. V-shaped beta
            np.random.set_state(init_rng_state)
            h = h_init.copy()
            alpha = np.full(sandbox.K, 1.0 / sandbox.K)
            jitter_v = 0.0
            for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
                similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
                exp_sims = np.exp(similarities / tau_mom)
                w = exp_sims / np.sum(exp_sims)
                alpha_prev = alpha.copy()
                beta_l = v_betas[l]
                alpha = (1.0 - beta_l) * w + beta_l * alpha_prev
                blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
                h = (1 - gammas[l]) * h + gammas[l] * blended_sig
                h += np.random.normal(0, 0.015, sandbox.D)
                jitter_v += np.sum((alpha - alpha_prev)**2)
            jitter_v /= (sandbox.L - sandbox.L_frozen - 1)
            total_jitter["Momentum-Merge (V-shaped beta)"] += jitter_v
            d = np.linalg.norm(h - signatures[k_true])**2
            p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
            if np.random.random() < p_correct:
                correct["Momentum-Merge (V-shaped beta)"] += 1
                
        for m in methods:
            acc_results[m].append(correct[m] / num_samples)
            jit_results[m].append(total_jitter[m] / num_samples)
            
    print("\n--- RESULTS OVER 10 SEEDS ---")
    for m in methods:
        mean_acc = np.mean(acc_results[m]) * 100
        std_acc = np.std(acc_results[m]) * 100
        mean_jit = np.mean(jit_results[m])
        print(f"{m:<40} | Acc: {mean_acc:.2f}% +- {std_acc:.2f}% | Jitter: {mean_jit:.6f}")

if __name__ == "__main__":
    run_v_beta_test()
