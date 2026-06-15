import time
import torch
import numpy as np

# Simulation Parameters matching run_experiments.py
D = 192               # Representation dimension
K = 4                 # Number of task experts
L = 14                # Total layers
L_frozen = 3          # First 3 layers are frozen
T = 5000              # Number of steps for stable profiling
tau = 0.10            # Softmax temperature
g_scale = 0.35         # LoRA adapter projection scale

# Generate mock signatures and activations
v = torch.randn(K, D)
v = v / (torch.norm(v, p=2, dim=-1, keepdim=True) + 1e-6)

# Inputs for simulation
h_inputs = [torch.randn(D) for _ in range(T)]

# 1. Benchmark SABLE
print("Profiling SABLE...")
start_time = time.perf_counter()
for t in range(T):
    h_3 = h_inputs[t]
    h_l = h_3.clone()
    for l in range(L_frozen + 1, L + 1):
        S = torch.zeros(K)
        for k in range(K):
            S[k] = torch.dot(h_l, v[k]) / (torch.norm(h_l) * torch.norm(v[k]) + 1e-6)
        S_noise = torch.randn(K) * 0.04
        alpha = torch.softmax((S + S_noise) / tau, dim=0)
        h_l = h_l + g_scale * torch.matmul(alpha, v - h_l)
sable_time = (time.perf_counter() - start_time) / T * 1e6 # in microseconds

# 2. Benchmark PAC-Kinetics
print("Profiling PAC-Kinetics...")
pk_A = torch.eye(K) * 0.85
pk_W = torch.eye(K) * 0.15
pk_s = torch.zeros(K)
start_time = time.perf_counter()
for t in range(T):
    h_3 = h_inputs[t]
    e_t = torch.zeros(K)
    for k in range(K):
        e_t[k] = torch.max(torch.tensor(0.0), torch.dot(h_3, v[k]) / (torch.norm(h_3) * torch.norm(v[k]) + 1e-6))
    if t == 0:
        pk_s = e_t.clone()
    else:
        pk_s = torch.matmul(pk_A, pk_s) + torch.matmul(pk_W, e_t)
    alpha_pk = torch.softmax(pk_s / tau, dim=0)
    
    h_l = h_3.clone()
    for l in range(L_frozen + 1, L + 1):
        S = torch.zeros(K)
        for k in range(K):
            S[k] = torch.dot(h_l, v[k]) / (torch.norm(h_l) * torch.norm(v[k]) + 1e-6)
        # SABLE routing layer updates
        alpha = alpha_pk.clone()
        h_l = h_l + g_scale * torch.matmul(alpha, v - h_l)
pk_time = (time.perf_counter() - start_time) / T * 1e6 # in microseconds

# 3. Benchmark ChemMerge (Dynamic ODE)
print("Profiling ChemMerge (Dynamic ODE)...")
cmd_alpha = {l: torch.ones(K) / K for l in range(L_frozen, L + 1)}
start_time = time.perf_counter()
for t in range(T):
    h_3 = h_inputs[t]
    h_l = h_3.clone()
    
    for l in range(L_frozen + 1, L + 1):
        S = torch.zeros(K)
        for k in range(K):
            S[k] = torch.dot(h_l, v[k]) / (torch.norm(h_l) * torch.norm(v[k]) + 1e-6)
        S_noise = torch.randn(K) * 0.04
        w_l_t = torch.softmax((S + S_noise) / tau, dim=0)
        
        if l == L_frozen + 1:
            alpha_prev_depth = w_l_t.clone()
        
        beta_depth = 0.60
        mismatch = torch.norm(w_l_t - cmd_alpha[l], p=2)
        T_0 = 0.40
        lam = 2.0
        temp = T_0 * (1.0 + lam * mismatch.item())
        
        E_a = 0.80
        R = 1.0
        A_f = 12.0
        A_b = 6.0
        k_f = A_f * np.exp(-E_a / (R * temp))
        k_b = A_b * np.exp(-E_a / (R * temp))
        
        alpha_tau = cmd_alpha[l].clone()
        N_steps = 5
        dtau = 1.0 / N_steps
        for _ in range(N_steps):
            dalpha = dtau * (k_f * w_l_t * (1.0 - alpha_tau) - k_b * alpha_tau)
            alpha_tau = alpha_tau + dalpha
            
        alpha_tau = torch.clamp(alpha_tau, min=1e-6)
        alpha_tau = alpha_tau / (torch.sum(alpha_tau) + 1e-12)
        
        alpha_prev_temp = alpha_tau.clone()
        alpha = beta_depth * alpha_prev_depth + (1.0 - beta_depth) * alpha_prev_temp
        alpha_prev_depth = alpha.clone()
        cmd_alpha[l] = alpha.clone()
        
        h_l = h_l + g_scale * torch.matmul(alpha, v - h_l)
cm_time = (time.perf_counter() - start_time) / T * 1e6 # in microseconds

# 4. Benchmark 2D-STEM (Ours)
print("Profiling 2D-STEM (Ours)...")
stem_alpha = {l: torch.ones(K) / K for l in range(L_frozen, L + 1)}
e_prev = torch.zeros(K)
start_time = time.perf_counter()
for t in range(T):
    h_3 = h_inputs[t]
    e_t = torch.zeros(K)
    for k in range(K):
        e_t[k] = torch.max(torch.tensor(0.0), torch.dot(h_3, v[k]) / (torch.norm(h_3) * torch.norm(v[k]) + 1e-6))
        
    if t == 0:
        Sim_t = torch.tensor(1.0)
    else:
        Sim_t = torch.dot(e_t, e_prev) / (torch.norm(e_t) * torch.norm(e_prev) + 1e-6)
        Sim_t = torch.clamp(Sim_t, 0.0, 1.0)
    e_prev = e_t.clone()
    
    beta_depth = 0.40
    beta_temp_0 = 0.40
    beta_temp_t = beta_temp_0 * (Sim_t.item() ** 3) if t > 0 else 0.0
    
    h_l = h_3.clone()
    for l in range(L_frozen + 1, L + 1):
        S = torch.zeros(K)
        for k in range(K):
            S[k] = torch.dot(h_l, v[k]) / (torch.norm(h_l) * torch.norm(v[k]) + 1e-6)
        S_noise = torch.randn(K) * 0.04
        w_l_t = torch.softmax((S + S_noise) / tau, dim=0)
        
        if l == L_frozen + 1:
            alpha_prev_depth = e_t / (torch.sum(e_t) + 1e-9)
            
        alpha_prev_temp = stem_alpha[l]
        alpha = beta_depth * alpha_prev_depth + beta_temp_t * alpha_prev_temp + (1.0 - beta_depth - beta_temp_t) * w_l_t
        alpha_prev_depth = alpha.clone()
        stem_alpha[l] = alpha.clone()
        
        h_l = h_l + g_scale * torch.matmul(alpha, v - h_l)
stem_time = (time.perf_counter() - start_time) / T * 1e6 # in microseconds

# Results Summary
print("\n" + "="*40)
print(f"Latency Benchmark Results (per-step, CPU, over {T} steps):")
print(f"SABLE (Stateless):           {sable_time:.2f} microseconds")
print(f"PAC-Kinetics:                {pk_time:.2f} microseconds")
print(f"ChemMerge (Dynamic ODE):     {cm_time:.2f} microseconds")
print(f"2D-STEM (Ours):              {stem_time:.2f} microseconds")
print("="*40)

# Calculate relative improvements
print(f"2D-STEM latency relative to SABLE:       {stem_time/sable_time:.2f}x")
print(f"2D-STEM latency relative to ChemMerge:   {stem_time/cm_time:.2f}x (a {(cm_time-stem_time)/cm_time*100:.1f}% reduction!)")
print(f"PAC-Kinetics latency relative to SABLE:  {pk_time/sable_time:.2f}x")
print("="*40)
