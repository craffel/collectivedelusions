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

def run_simulation_seed_all(seed, num_samples=200):
    set_seed(seed)
    sandbox = AnalyticalCoordinateSandbox()
    signatures = sandbox.generate_task_signatures()
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    # 1. Pre-compute Global Centroids (exactly as in run_experiments.py)
    global_centroids = []
    for k in range(sandbox.K):
        cal_samples = [sandbox.generate_sample(k, signatures, noise_scales) for _ in range(64)]
        centroid = np.mean(cal_samples, axis=0)
        global_centroids.append(centroid / np.linalg.norm(centroid))
    global_centroids = np.array(global_centroids)
    
    # Generate serving stream (exactly as in run_experiments.py)
    task_indices = np.random.choice(sandbox.K, size=num_samples)
    stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
    
    gammas = {l: 0.30 for l in range(sandbox.L_frozen + 1, sandbox.L + 1)}
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    
    methods = [
        "Expert Ceiling", 
        "Uniform Merging", 
        "SABLE", 
        "ChemMerge", 
        "Momentum-Merge (Base)",
        "Momentum-Merge + Eq 9 (Layer Centroids)",
        "Momentum-Merge + Eq 10 (Raw Boundary)",
        "Momentum-Merge + Eq 9 + Eq 10 (Both)"
    ]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    best_beta = 0.60
    
    # Pre-compute Layer-wise Centroids (Eq 9) for the variants that use it
    # We do this calibration AFTER generating the main stream, so as not to disrupt the RNG state
    # for the baseline methods. Wait! If we run calibration, does it change the RNG state for the main loop?
    # Yes! To keep SABLE, ChemMerge, and MM (Base) EXACTLY identical to the original run_experiments.py,
    # we must save the RNG state right before calibration, run calibration, and then restore the RNG state
    # right before running the serving stream! This is a beautiful trick.
    rng_state_before_cal = np.random.get_state()
    
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
            
    # Restore the RNG state so the main loop runs EXACTLY with the same noise sequence as before
    np.random.set_state(rng_state_before_cal)
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        # To make sure each method gets the exact same noise sequence for its layer noise,
        # we can save and restore the RNG state for each method within the loop!
        # This guarantees that the layer-wise noise is identical across all methods.
        # But wait, run_experiments.py did not restore RNG state between methods;
        # it ran them sequentially, which means SABLE got noise set A, ChemMerge got noise set B, etc.
        # Let's run SABLE, ChemMerge, and MM Base exactly sequentially as before to match the original.
        # Let's save RNG state before the loop step to run the Eq 9 and 10 variants.
        
        rng_before_step = np.random.get_state()
        
        # 1. Expert Ceiling
        h = h_init.copy()
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            h = (1 - gammas[l]) * h + gammas[l] * signatures[k_true]
            h += np.random.normal(0, 0.015, sandbox.D)
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Expert Ceiling"] += 1
            
        # 2. Uniform Merging
        h = h_init.copy()
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            h = (1 - gammas[l]) * h + gammas[l] * np.mean(signatures, axis=0)
            h += np.random.normal(0, 0.015, sandbox.D)
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Uniform Merging"] += 1
            
        # 3. SABLE
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_sable = 0.0
        tau_sable = 0.15
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
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
        h = h_init.copy()
        C = np.full(sandbox.K, 1.0 / sandbox.K)
        dt = 1.5
        k_decay = 0.3
        jitter_chem = 0.0
        tau_chem = 0.005
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
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
            
        # 5. Momentum-Merge (Base)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_mom = 0.0
        tau_mom = 0.005
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_mom += np.sum((alpha - alpha_prev)**2)
        jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Base)"] += jitter_mom
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Base)"] += 1
            
        # To evaluate the other variants with the exact same RNG trajectory state as MM (Base)
        # after it starts, we can use a dedicated RNG save/restore or just run them.
        # Actually, let's capture the RNG state right before MM (Base) and restore it for each variant
        # so that each variant gets the exact same layer noise as MM (Base)!
        # Let's save RNG state right before MM (Base):
        # (Wait, MM Base was run after ChemMerge, so it has whatever RNG state was left by ChemMerge).
        # We can just record the state right before MM (Base) and restore it for each new variant.
        # This is incredibly elegant and ensures perfect comparative parity.
        
    # Let's write a cleaner loop where we run each method with the exact same layer noise
    # as its baseline. But actually, let's just run them with seed reset or sequential.
    # To keep it extremely simple and match the 10-seed run exactly, let's run the main 10 seeds
    # with the exact sequential structure of run_experiments.py, but now with the new variants
    # added at the end. To make sure the new variants get their own independent evaluations,
    # let's just append them. Let's see how much they get.

def run_clean_comparisons_seed(seed, num_samples=200):
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
    
    # Pre-compute Layer-wise Centroids (Eq 9)
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
            
    # Generate serving stream
    task_indices = np.random.choice(sandbox.K, size=num_samples)
    stream_samples = [sandbox.generate_sample(k, signatures, noise_scales) for k in task_indices]
    
    E = [0.005, 0.03, 0.08, 0.50]
    lambda_val = 0.40
    best_beta = 0.60
    tau_mom = 0.005
    
    methods = [
        "Expert Ceiling", 
        "Uniform Merging", 
        "SABLE", 
        "SABLE + Eq 9",
        "ChemMerge", 
        "Momentum-Merge (Base)",
        "Momentum-Merge + Eq 9",
        "Momentum-Merge + Eq 10",
        "Momentum-Merge + Eq 9 + Eq 10"
    ]
    correct = {m: 0 for m in methods}
    total_jitter = {m: 0.0 for m in methods}
    
    for idx, (k_true, h_init) in enumerate(zip(task_indices, stream_samples)):
        # We will reset the RNG state before running EACH method for this sample
        # so they all experience the EXACT same layer-wise noise sequence!
        # This is the most mathematically rigorous way to run a comparative ablation!
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
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
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
            
        # 3b. SABLE + Eq 9 (Layer Centroids)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_sable_eq9 = 0.0
        tau_sable = 0.15
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, layer_centroids[(l, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_sable)
            alpha_prev = alpha.copy()
            alpha = exp_sims / np.sum(exp_sims)
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_sable_eq9 += np.sum((alpha - alpha_prev)**2)
        jitter_sable_eq9 /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["SABLE + Eq 9"] += jitter_sable_eq9
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["SABLE + Eq 9"] += 1
            
        # 4. ChemMerge
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        C = np.full(sandbox.K, 1.0 / sandbox.K)
        dt = 1.5
        k_decay = 0.3
        jitter_chem = 0.0
        tau_chem = 0.005
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
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
            
        # 5. Momentum-Merge (Base)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_mom = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_mom += np.sum((alpha - alpha_prev)**2)
        jitter_mom /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge (Base)"] += jitter_mom
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge (Base)"] += 1
            
        # 6. Momentum-Merge + Eq 9 (Layer Centroids)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        alpha = np.full(sandbox.K, 1.0 / sandbox.K)
        jitter_eq9 = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, layer_centroids[(l, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
            exp_sims = np.exp(similarities / tau_mom)
            w = exp_sims / np.sum(exp_sims)
            alpha_prev = alpha.copy()
            alpha = (1.0 - best_beta) * w + best_beta * alpha_prev
            blended_sig = np.sum([alpha[j] * signatures[j] for j in range(sandbox.K)], axis=0)
            h = (1 - gammas[l]) * h + gammas[l] * blended_sig
            h += np.random.normal(0, 0.015, sandbox.D)
            jitter_eq9 += np.sum((alpha - alpha_prev)**2)
        jitter_eq9 /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge + Eq 9"] += jitter_eq9
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge + Eq 9"] += 1
            
        # 7. Momentum-Merge + Eq 10 (Raw Boundary)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        l_first = sandbox.L_frozen + 1
        sims_first = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
        exp_sims_first = np.exp(sims_first / tau_mom)
        w_first = exp_sims_first / np.sum(exp_sims_first)
        
        alpha = w_first.copy()
        jitter_eq10 = 0.0
        for l in range(sandbox.L_frozen + 1, sandbox.L + 1):
            similarities = np.array([np.dot(h, global_centroids[j]) / np.linalg.norm(h) for j in range(sandbox.K)])
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
            jitter_eq10 += np.sum((alpha - alpha_prev)**2)
        jitter_eq10 /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge + Eq 10"] += jitter_eq10
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge + Eq 10"] += 1
            
        # 8. Momentum-Merge + Eq 9 + Eq 10 (Both)
        np.random.set_state(init_rng_state)
        h = h_init.copy()
        sims_first = np.array([np.dot(h, layer_centroids[(l_first, j)]) / np.linalg.norm(h) for j in range(sandbox.K)])
        exp_sims_first = np.exp(sims_first / tau_mom)
        w_first = exp_sims_first / np.sum(exp_sims_first)
        
        alpha = w_first.copy()
        jitter_both = 0.0
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
            jitter_both += np.sum((alpha - alpha_prev)**2)
        jitter_both /= (sandbox.L - sandbox.L_frozen - 1)
        total_jitter["Momentum-Merge + Eq 9 + Eq 10"] += jitter_both
        d = np.linalg.norm(h - signatures[k_true])**2
        p_correct = (1.0 - E[k_true]) * np.exp(-lambda_val * d)
        if np.random.random() < p_correct:
            correct["Momentum-Merge + Eq 9 + Eq 10"] += 1

    seed_results = {}
    for m in methods:
        seed_results[m] = {
            "accuracy": correct[m] / num_samples,
            "jitter": total_jitter[m] / num_samples
        }
    return seed_results

def run_clean_simulation_all_seeds(seeds=10):
    raw_results = {m: {"accuracy": [], "jitter": []} for m in [
        "Expert Ceiling", 
        "Uniform Merging", 
        "SABLE", 
        "SABLE + Eq 9",
        "ChemMerge", 
        "Momentum-Merge (Base)",
        "Momentum-Merge + Eq 9",
        "Momentum-Merge + Eq 10",
        "Momentum-Merge + Eq 9 + Eq 10"
    ]}
    
    for seed in range(seeds):
        res = run_clean_comparisons_seed(seed=seed)
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
        }
        
    return real_results

if __name__ == "__main__":
    print("--- Running 10-Seed Mathematically Rigorous Comparative Simulation ---")
    results = run_clean_simulation_all_seeds(seeds=10)
    
    print("\n" + "="*95)
    print(f"{'Method / Variant':<35} | {'Accuracy Mean (%)':<15} | {'Accuracy Std (%)':<15} | {'Jitter Mean (MSE)':<15}")
    print("="*95)
    for m, res in results.items():
        print(f"{m:<35} | {res['accuracy_mean']*100:<15.2f} | {res['accuracy_std']*100:<15.2f} | {res['jitter_mean']:<15.6f}")
    print("="*95)
