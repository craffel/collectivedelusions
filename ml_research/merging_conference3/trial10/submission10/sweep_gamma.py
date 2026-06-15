import numpy as np
import torch
import run_experiments

def run_simulation_sweep(seed, gamma, overlapping=False, homogeneous=True):
    run_experiments.set_seed(seed)
    v = run_experiments.get_signatures(overlapping=overlapping)
    y = run_experiments.get_query_stream(homogeneous=homogeneous)
    
    # We will simulate 2D-STEM specifically
    stem_alpha = {l: torch.ones(run_experiments.K) / run_experiments.K for l in range(run_experiments.L_frozen, run_experiments.L + 1)}
    
    accuracies = []
    coefficients = []
    
    for t in range(run_experiments.T):
        target_k = y[t]
        target_v = v[target_k]
        
        # 1. Input generation with observation noise
        h_0 = target_v + torch.randn(run_experiments.D) * run_experiments.sigma_noise
        
        # 2. Propagation through frozen layers l=1, 2, 3
        h = h_0.clone()
        for l in range(1, run_experiments.L_frozen + 1):
            h = h + torch.randn(run_experiments.D) * run_experiments.sigma_layer_noise
            
        h_3 = h.clone() # Activation at early routing layer 3
        
        # 3. Coordinate signals
        e_t = torch.zeros(run_experiments.K)
        for k in range(run_experiments.K):
            e_t[k] = torch.max(torch.tensor(0.0), torch.dot(h_3, v[k]) / (torch.norm(h_3) * torch.norm(v[k]) + 1e-6))
            
        # Dynamic stream similarity Sim_t
        if t == 0:
            Sim_t = torch.tensor(1.0)
            e_prev = e_t.clone()
        else:
            Sim_t = torch.dot(e_t, e_prev) / (torch.norm(e_t) * torch.norm(e_prev) + 1e-6)
            Sim_t = torch.clamp(Sim_t, 0.0, 1.0)
            e_prev = e_t.clone()
            
        h_l = h_3.clone()
        m_coeffs = []
        
        # Propagate through adapted layers l=4 to 14
        for l in range(run_experiments.L_frozen + 1, run_experiments.L + 1):
            S = torch.zeros(run_experiments.K)
            for k in range(run_experiments.K):
                S[k] = torch.dot(h_l, v[k]) / (torch.norm(h_l) * torch.norm(v[k]) + 1e-6)
            S_noise = torch.randn(run_experiments.K) * 0.04
            w_l_t = torch.softmax((S + S_noise) / run_experiments.tau, dim=0)
            
            beta_depth = 0.40
            beta_temp_0 = 0.40
            # Power-Law Gating
            beta_temp_t = beta_temp_0 * (Sim_t.item() ** gamma) if t > 0 else 0.0
            
            if l == run_experiments.L_frozen + 1:
                alpha_prev_depth = w_l_t.clone()
                
            alpha_prev_temp = stem_alpha[l]
            alpha = beta_depth * alpha_prev_depth + beta_temp_t * alpha_prev_temp + (1.0 - beta_depth - beta_temp_t) * w_l_t
            alpha_prev_depth = alpha.clone()
            stem_alpha[l] = alpha.clone()
            
            m_coeffs.append(alpha.clone())
            h_l = h_l + run_experiments.g_scale * torch.matmul(alpha, v - h_l) + torch.randn(run_experiments.D) * run_experiments.sigma_layer_noise
            
        dist_sq = torch.sum((h_l - target_v)**2)
        acc = torch.exp(-run_experiments.kappa_scale * dist_sq).item()
        accuracies.append(acc)
        coefficients.append(m_coeffs[-1].numpy())
        
    coeffs = np.array(coefficients)
    jitters = np.sum(np.abs(coeffs[1:] - coeffs[:-1]), axis=1)
    return np.mean(accuracies) * 100.0, np.mean(jitters)

# Sweep gamma on Overlapping, Heterogeneous stream across 5 seeds
seeds = [42, 43, 44, 45, 46]
gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

print("Overlapping Heterogeneous Stream Sweep:")
for g in gammas:
    accs, jits = [], []
    for s in seeds:
        acc, jit = run_simulation_sweep(s, g, overlapping=True, homogeneous=False)
        accs.append(acc)
        jits.append(jit)
    print(f"Gamma: {g:.1f} | Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}% | Jitter: {np.mean(jits):.4f}")

print("\nOrthogonal Heterogeneous Stream Sweep:")
for g in gammas:
    accs, jits = [], []
    for s in seeds:
        acc, jit = run_simulation_sweep(s, g, overlapping=False, homogeneous=False)
        accs.append(acc)
        jits.append(jit)
    print(f"Gamma: {g:.1f} | Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}% | Jitter: {np.mean(jits):.4f}")

print("\nOverlapping Homogeneous Stream Sweep (to ensure high-noise smoothing is preserved):")
for g in gammas:
    accs, jits = [], []
    for s in seeds:
        acc, jit = run_simulation_sweep(s, g, overlapping=True, homogeneous=True)
        accs.append(acc)
        jits.append(jit)
    print(f"Gamma: {g:.1f} | Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}% | Jitter: {np.mean(jits):.4f}")
