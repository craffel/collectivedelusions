import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from run_experiments
from run_experiments import CoordinateSandbox, PAC_Kinetics_Router, set_seed, sigmoid

def run_truncation_check():
    print("--- Running Truncation Check ---")
    seeds = [42, 43, 44, 45, 46]
    sandbox = CoordinateSandbox(num_tasks=4, dim=192, layers=14, overlap=0)
    
    total_queries_evaluated = 0
    total_truncations_triggered = 0
    
    for seed in seeds:
        set_seed(seed)
        signatures = sandbox.generate_signatures()
        
        # Subspace Split (8 samples per task)
        subspace_samples = []
        subspace_labels = []
        for k in range(sandbox.num_tasks):
            for _ in range(8):
                h0 = sandbox.generate_sample(k, signatures[k])
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                subspace_samples.append(h_route)
                subspace_labels.append(k)
                
        projection_matrices = []
        for k in range(sandbox.num_tasks):
            Z_k = np.array([subspace_samples[i] for i in range(len(subspace_samples)) if subspace_labels[i] == k])
            Z_k_norm = Z_k / (np.linalg.norm(Z_k, axis=1, keepdims=True) + 1e-8)
            U, S, Vh = np.linalg.svd(Z_k_norm, full_matrices=False)
            V_k = Vh.T[:, :8]
            projection_matrices.append(V_k)
            
        # Optimization Split (8 samples per task)
        opt_samples_h0 = []
        opt_labels = []
        for k in range(sandbox.num_tasks):
            for _ in range(8):
                h0 = sandbox.generate_sample(k, signatures[k])
                opt_samples_h0.append(h0)
                opt_labels.append(k)
                
        opt_coords = []
        for h0 in opt_samples_h0:
            h_route = sandbox.propagate_early(h0, signatures, steps=2)
            tilde_z = h_route / (np.linalg.norm(h_route) + 1e-8)
            e = [np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z)) for k in range(sandbox.num_tasks)]
            opt_coords.append(e)
            
        opt_coords = torch.tensor(opt_coords, dtype=torch.float32)
        opt_labels_torch = torch.tensor(opt_labels, dtype=torch.long)
        
        # Router setup
        pac_router = PAC_Kinetics_Router(num_tasks=sandbox.num_tasks)
        optimizer = optim.Adam(pac_router.parameters(), lr=0.01)
        
        for epoch in range(150):
            optimizer.zero_grad()
            alphas = pac_router.forward_stream(opt_coords)
            
            # Individual query losses
            individual_losses = -torch.log(alphas[range(len(opt_labels_torch)), opt_labels_torch] + 1e-8)
            
            # Record truncations
            truncations = (individual_losses > 5.0).sum().item()
            total_truncations_triggered += truncations
            total_queries_evaluated += len(opt_labels_torch)
            
            # Clamped loss to satisfy the bounded loss assumption
            losses_clamped = torch.clamp(individual_losses, max=5.0)
            loss_ce = torch.mean(losses_clamped)
            kl = pac_router.compute_kl()
            L_max = 5.0
            lam = 0.5  # Catoni lambda parameter (renamed to avoid conflict with beta-mixing)
            delta = 0.05
            a = len(opt_labels_torch) / 4.0
            bound = (L_max / (1.0 - np.exp(-lam))) * (1.0 - torch.exp(-lam * loss_ce / L_max - 2.0 * (kl + np.log(2.0 / delta)) / a))
            
            bound.backward()
            optimizer.step()
            
    pct_triggered = (total_truncations_triggered / total_queries_evaluated) * 100.0
    print(f"Total evaluated individual losses: {total_queries_evaluated}")
    print(f"Total times loss > 5.0: {total_truncations_triggered}")
    print(f"Truncation trigger rate: {pct_triggered:.4f}%")
    return pct_triggered

def run_sigma_sweep():
    print("\n--- Running Prior Variance (sigma_0^2) Sweep ---")
    sigma_vals = [0.1, 1.0, 5.0, 10.0, 50.0]
    seeds = [42, 43, 44, 45, 46]
    sandbox = CoordinateSandbox(num_tasks=4, dim=192, layers=14, overlap=0)
    
    results = {}
    
    for sig2 in sigma_vals:
        accs = []
        jitters = []
        for seed in seeds:
            set_seed(seed)
            signatures = sandbox.generate_signatures()
            
            # C_sub split
            subspace_samples = []
            subspace_labels = []
            for k in range(sandbox.num_tasks):
                for _ in range(8):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    h_route = sandbox.propagate_early(h0, signatures, steps=2)
                    subspace_samples.append(h_route)
                    subspace_labels.append(k)
                    
            projection_matrices = []
            for k in range(sandbox.num_tasks):
                Z_k = np.array([subspace_samples[i] for i in range(len(subspace_samples)) if subspace_labels[i] == k])
                Z_k_norm = Z_k / (np.linalg.norm(Z_k, axis=1, keepdims=True) + 1e-8)
                U, S, Vh = np.linalg.svd(Z_k_norm, full_matrices=False)
                V_k = Vh.T[:, :8]
                projection_matrices.append(V_k)
                
            # C_opt split
            opt_samples_h0 = []
            opt_labels = []
            for k in range(sandbox.num_tasks):
                for _ in range(8):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    opt_samples_h0.append(h0)
                    opt_labels.append(k)
                    
            opt_coords = []
            for h0 in opt_samples_h0:
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                tilde_z = h_route / (np.linalg.norm(h_route) + 1e-8)
                e = [np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z)) for k in range(sandbox.num_tasks)]
                opt_coords.append(e)
                
            opt_coords = torch.tensor(opt_coords, dtype=torch.float32)
            opt_labels_torch = torch.tensor(opt_labels, dtype=torch.long)
            
            # Optimizing under specific sigma0_sq
            pac_router = PAC_Kinetics_Router(num_tasks=sandbox.num_tasks, sigma0_sq=sig2)
            optimizer = optim.Adam(pac_router.parameters(), lr=0.01)
            
            for epoch in range(150):
                optimizer.zero_grad()
                alphas = pac_router.forward_stream(opt_coords)
                individual_losses = -torch.log(alphas[range(len(opt_labels_torch)), opt_labels_torch] + 1e-8)
                losses_clamped = torch.clamp(individual_losses, max=5.0)
                loss_ce = torch.mean(losses_clamped)
                kl = pac_router.compute_kl()
                L_max = 5.0
                lam = 0.5  # Catoni lambda parameter
                delta = 0.05
                a = len(opt_labels_torch) / 4.0
                bound = (L_max / (1.0 - np.exp(-lam))) * (1.0 - torch.exp(-lam * loss_ce / L_max - 2.0 * (kl + np.log(2.0 / delta)) / a))
                bound.backward()
                optimizer.step()
                
            u_opt = pac_router.u.detach().cpu().numpy()
            W_opt = pac_router.W.detach().cpu().numpy()
            w_opt = pac_router.w.detach().cpu().numpy()
            a_opt = sigmoid(u_opt)
            tau_opt = np.exp(w_opt)
            
            # Evaluate on Heterogeneous stream
            test_samples_h0 = []
            test_labels = []
            for k in range(sandbox.num_tasks):
                for _ in range(250):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    test_samples_h0.append(h0)
                    test_labels.append(k)
                    
            test_samples_route = []
            for h0 in test_samples_h0:
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                test_samples_route.append(h_route)
                
            hetero_indices = list(range(len(test_labels)))
            rng = np.random.default_rng(seed + 1000)
            rng.shuffle(hetero_indices)
            hetero_route = [test_samples_route[i] for i in hetero_indices]
            hetero_labels = [test_labels[i] for i in hetero_indices]
            
            T = len(hetero_labels)
            accuracy_sum = 0.0
            alphas_history = []
            s_pk = np.zeros(sandbox.num_tasks)
            lambda_scale = 0.0385
            e_prev = None
            
            for t in range(T):
                z_t = hetero_route[t]
                y_t = hetero_labels[t]
                
                tilde_z_t = z_t / (np.linalg.norm(z_t) + 1e-8)
                e_t = np.array([np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z_t)) for k in range(sandbox.num_tasks)])
                
                if e_prev is not None:
                    num = np.dot(e_t, e_prev)
                    den = np.linalg.norm(e_t) * np.linalg.norm(e_prev) + 1e-8
                    cos_sim = num / den
                    homogeneity = np.maximum(0.0, cos_sim)
                    a_t = a_opt * homogeneity
                else:
                    a_t = a_opt
                    
                s_pk = a_t * s_pk + np.dot(W_opt, e_t)
                alphas_pac_kinetics = np.exp(s_pk / tau_opt) / np.sum(np.exp(s_pk / tau_opt))
                alphas_history.append(alphas_pac_kinetics)
                
                # Update e_prev
                e_prev = e_t
                
                h_L_pk = sandbox.propagate_subsequent(z_t, signatures, alphas_pac_kinetics, steps=11)
                dist_pk = np.linalg.norm(h_L_pk - signatures[y_t])
                accuracy_sum += np.exp(-lambda_scale * (dist_pk ** 2))
                
            acc = (accuracy_sum / T) * 100.0
            history = np.array(alphas_history)
            jit = np.mean(np.sum(np.abs(history[1:] - history[:-1]), axis=1))
            
            accs.append(acc)
            jitters.append(jit)
            
        results[sig2] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "jitter_mean": np.mean(jitters),
            "jitter_std": np.std(jitters)
        }
        print(f"sigma_0^2 = {sig2:4.1f}: Accuracy = {np.mean(accs):6.2f}% +/- {np.std(accs):4.2f}%, Jitter = {np.mean(jitters):6.4f} +/- {np.std(jitters):5.4f}")
        
    return results

def run_length_sweep():
    print("\n--- Running Calibration Sequence Length (T) Sweep ---")
    T_vals = [8, 16, 32, 64, 128]
    seeds = [42, 43, 44, 45, 46]
    sandbox = CoordinateSandbox(num_tasks=4, dim=192, layers=14, overlap=0)
    
    results = {}
    
    for cal_T in T_vals:
        samples_per_task = cal_T // 4
        accs = []
        jitters = []
        for seed in seeds:
            set_seed(seed)
            signatures = sandbox.generate_signatures()
            
            # C_sub split (fixed at 8 per task)
            subspace_samples = []
            subspace_labels = []
            for k in range(sandbox.num_tasks):
                for _ in range(8):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    h_route = sandbox.propagate_early(h0, signatures, steps=2)
                    subspace_samples.append(h_route)
                    subspace_labels.append(k)
                    
            projection_matrices = []
            for k in range(sandbox.num_tasks):
                Z_k = np.array([subspace_samples[i] for i in range(len(subspace_samples)) if subspace_labels[i] == k])
                Z_k_norm = Z_k / (np.linalg.norm(Z_k, axis=1, keepdims=True) + 1e-8)
                U, S, Vh = np.linalg.svd(Z_k_norm, full_matrices=False)
                V_k = Vh.T[:, :8]
                projection_matrices.append(V_k)
                
            # C_opt split with specified samples_per_task
            opt_samples_h0 = []
            opt_labels = []
            for k in range(sandbox.num_tasks):
                for _ in range(samples_per_task):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    opt_samples_h0.append(h0)
                    opt_labels.append(k)
                    
            opt_coords = []
            for h0 in opt_samples_h0:
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                tilde_z = h_route / (np.linalg.norm(h_route) + 1e-8)
                e = [np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z)) for k in range(sandbox.num_tasks)]
                opt_coords.append(e)
                
            opt_coords = torch.tensor(opt_coords, dtype=torch.float32)
            opt_labels_torch = torch.tensor(opt_labels, dtype=torch.long)
            
            # Optimize under default sigma0_sq = 5.0
            pac_router = PAC_Kinetics_Router(num_tasks=sandbox.num_tasks, sigma0_sq=5.0)
            optimizer = optim.Adam(pac_router.parameters(), lr=0.01)
            
            for epoch in range(150):
                optimizer.zero_grad()
                alphas = pac_router.forward_stream(opt_coords)
                individual_losses = -torch.log(alphas[range(len(opt_labels_torch)), opt_labels_torch] + 1e-8)
                losses_clamped = torch.clamp(individual_losses, max=5.0)
                loss_ce = torch.mean(losses_clamped)
                kl = pac_router.compute_kl()
                L_max = 5.0
                lam = 0.5  # Catoni lambda parameter
                delta = 0.05
                a = len(opt_labels_torch) / 4.0
                bound = (L_max / (1.0 - np.exp(-lam))) * (1.0 - torch.exp(-lam * loss_ce / L_max - 2.0 * (kl + np.log(2.0 / delta)) / a))
                bound.backward()
                optimizer.step()
                
            u_opt = pac_router.u.detach().cpu().numpy()
            W_opt = pac_router.W.detach().cpu().numpy()
            w_opt = pac_router.w.detach().cpu().numpy()
            a_opt = sigmoid(u_opt)
            tau_opt = np.exp(w_opt)
            
            # Evaluate on Heterogeneous stream
            test_samples_h0 = []
            test_labels = []
            for k in range(sandbox.num_tasks):
                for _ in range(250):
                    h0 = sandbox.generate_sample(k, signatures[k])
                    test_samples_h0.append(h0)
                    test_labels.append(k)
                    
            test_samples_route = []
            for h0 in test_samples_h0:
                h_route = sandbox.propagate_early(h0, signatures, steps=2)
                test_samples_route.append(h_route)
                
            hetero_indices = list(range(len(test_labels)))
            rng = np.random.default_rng(seed + 1000)
            rng.shuffle(hetero_indices)
            hetero_route = [test_samples_route[i] for i in hetero_indices]
            hetero_labels = [test_labels[i] for i in hetero_indices]
            
            T = len(hetero_labels)
            accuracy_sum = 0.0
            alphas_history = []
            s_pk = np.zeros(sandbox.num_tasks)
            lambda_scale = 0.0385
            e_prev = None
            
            for t in range(T):
                z_t = hetero_route[t]
                y_t = hetero_labels[t]
                
                tilde_z_t = z_t / (np.linalg.norm(z_t) + 1e-8)
                e_t = np.array([np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z_t)) for k in range(sandbox.num_tasks)])
                
                if e_prev is not None:
                    num = np.dot(e_t, e_prev)
                    den = np.linalg.norm(e_t) * np.linalg.norm(e_prev) + 1e-8
                    cos_sim = num / den
                    homogeneity = np.maximum(0.0, cos_sim)
                    a_t = a_opt * homogeneity
                else:
                    a_t = a_opt
                    
                s_pk = a_t * s_pk + np.dot(W_opt, e_t)
                alphas_pac_kinetics = np.exp(s_pk / tau_opt) / np.sum(np.exp(s_pk / tau_opt))
                alphas_history.append(alphas_pac_kinetics)
                
                # Update e_prev
                e_prev = e_t
                
                h_L_pk = sandbox.propagate_subsequent(z_t, signatures, alphas_pac_kinetics, steps=11)
                dist_pk = np.linalg.norm(h_L_pk - signatures[y_t])
                accuracy_sum += np.exp(-lambda_scale * (dist_pk ** 2))
                
            acc = (accuracy_sum / T) * 100.0
            history = np.array(alphas_history)
            jit = np.mean(np.sum(np.abs(history[1:] - history[:-1]), axis=1))
            
            accs.append(acc)
            jitters.append(jit)
            
        results[cal_T] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "jitter_mean": np.mean(jitters),
            "jitter_std": np.std(jitters)
        }
        print(f"Calibration length T = {cal_T:3d}: Accuracy = {np.mean(accs):6.2f}% +/- {np.std(accs):4.2f}%, Jitter = {np.mean(jitters):6.4f} +/- {np.std(jitters):5.4f}")
        
    return results

def run_latency_memory_profiling():
    print("\n--- Running Latency & Memory Profiling ---")
    K_vals = [2, 4, 8]
    dim = 192 # Feature dimensionality from Sandbox
    
    results = {}
    
    for K in K_vals:
        # PAC_Kinetics_Router parameter sizes
        # u: K, W: K*K, w: K. Total parameters = K^2 + 2K
        param_count = K * K + 2 * K
        # Parameter memory in bytes (using float32, which is 4 bytes per param)
        param_memory_bytes = param_count * 4
        param_memory_kb = param_memory_bytes / 1024.0
        
        # Instantiate router
        router = PAC_Kinetics_Router(num_tasks=K)
        
        # Generate dummy input coordinate vector
        coords = torch.rand(1, K)
        
        # Warmup
        for _ in range(50):
            _ = router.forward_stream(coords)
            
        # Profile single stateful routing update latency
        # We simulate 10,000 steps and measure time
        start_time = time.perf_counter()
        steps = 10000
        # Simulated single step loop
        s = torch.zeros(K)
        a = torch.sigmoid(router.u).detach()
        W = router.W.detach()
        tau = torch.exp(router.w).detach()
        e_dummy = torch.rand(K)
        
        for _ in range(steps):
            s = a * s + torch.mv(W, e_dummy)
            alpha = torch.softmax(s / tau, dim=0)
            
        end_time = time.perf_counter()
        avg_latency_us = ((end_time - start_time) / steps) * 1e6
        
        results[K] = {
            "params": param_count,
            "memory_kb": param_memory_kb,
            "latency_us": avg_latency_us
        }
        print(f"Experts K = {K}: Params = {param_count:3d}, Parameter Memory = {param_memory_kb:6.4f} KB, Latency = {avg_latency_us:6.3f} microseconds")
        
    return results

if __name__ == "__main__":
    run_truncation_check()
    run_sigma_sweep()
    run_length_sweep()
    run_latency_memory_profiling()
