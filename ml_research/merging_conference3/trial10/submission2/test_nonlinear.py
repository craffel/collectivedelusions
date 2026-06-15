import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from train_and_evaluate import (
    generate_signatures, generate_samples, extract_coordinates, compute_similarity,
    LDSKineticsRouter, compute_metrics, set_seed, K, D, L, gamma_V
)

def propagate_layers_nonlinear(h3, alpha_seq, signatures, layer_to_block_mapping):
    T = h3.shape[0]
    h = h3.clone()
    for l in range(4, L + 1):
        block_idx = layer_to_block_mapping[l - 4]
        alpha_layer = alpha_seq[block_idx]  # shape (T, K)
        
        # Expert update: (T, K, D) of v_k - h
        expert_diff = signatures.unsqueeze(0) - h.unsqueeze(1)  # (T, K, D)
        scaled_diff = expert_diff * alpha_layer.unsqueeze(-1)  # (T, K, D)
        update = torch.sum(scaled_diff, dim=1) * gamma_V  # (T, D)
        
        # Add update
        h = h + update
        
        # Non-linear activation: GELU
        h = F.gelu(h)
        
        # Layer Normalization along hidden dimension D
        mean = h.mean(dim=-1, keepdim=True)
        std = h.std(dim=-1, keepdim=True) + 1e-5
        h = (h - mean) / std
        
    return h

def train_router_nonlinear(config_type, M, layer_mapping, seed, regularized=True, lr=0.005, epochs=100, T_cal=32):
    set_seed(seed)
    signatures, indices = generate_signatures(config_type)
    
    samples_per_expert = T_cal // K
    y_cal = []
    for k in range(K):
        y_cal.extend([k] * samples_per_expert)
    y_cal = torch.tensor(y_cal)
    
    h3_cal = generate_samples(y_cal, signatures, indices)
    e_cal = extract_coordinates(h3_cal, indices)
    sim_cal = compute_similarity(e_cal)
    
    router = LDSKineticsRouter(M, K)
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    
    lambda_param = 0.5
    L_max = 5.0
    a_blocks = float(T_cal) / 4.0
    sigma_0_sq = 5.0
    
    u0 = torch.zeros(M, K)
    W0 = torch.stack([torch.eye(K) for _ in range(M)])
    w0 = torch.ones(M, K) * math.log(0.05)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alphas = router(e_cal, sim_cal)
        h_L = propagate_layers_nonlinear(h3_cal, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_cal, signatures)
        
        p = torch.softmax(logits, dim=-1)
        loss_ce = -torch.log(p[torch.arange(logits.shape[0]), y_cal] + 1e-8)
        truncated_loss = torch.clamp(loss_ce, max=L_max)
        R_hat = torch.mean(truncated_loss)
        
        if regularized:
            kl = (torch.sum((router.u - u0) ** 2) + 
                  torch.sum((router.W - W0) ** 2) + 
                  torch.sum((router.w - w0) ** 2)) / (2.0 * sigma_0_sq)
            loss = (lambda_param / L_max) * R_hat + (1.0 / (a_blocks * sigma_0_sq)) * kl
        else:
            loss = R_hat
            
        loss.backward()
        optimizer.step()
        
    return router, signatures, indices

def evaluate_model_nonlinear(config_type, model_type, router, signatures, indices, test_seed, stream_type='homogeneous'):
    set_seed(test_seed)
    T_test = 200
    if stream_type == 'homogeneous':
        y_test = []
        for k in range(K):
            y_test.extend([k] * 50)
    else:
        import random
        random.seed(test_seed)
        y_test = [random.randint(0, K - 1) for _ in range(T_test)]
        
    y_test = torch.tensor(y_test)
    h3_test = generate_samples(y_test, signatures, indices)
    e_test = extract_coordinates(h3_test, indices)
    sim_test = compute_similarity(e_test)
    
    if model_type == 'oracle':
        alphas = [torch.zeros(T_test, K) for _ in range(11)]
        for m in range(len(alphas)):
            for t in range(T_test):
                alphas[m][t, y_test[t]] = 1.0
        layer_mapping = list(range(11))
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
    elif model_type == 'uniform':
        alphas = [torch.ones(T_test, K) * 0.25]
        layer_mapping = [0] * 11
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
    elif model_type == 'sable_raw':
        tau = 0.05
        alpha_seq = torch.zeros(T_test, K)
        for t in range(T_test):
            max_e = torch.max(e_test[t] / tau)
            exp_e = torch.exp(e_test[t] / tau - max_e)
            alpha_seq[t] = exp_e / torch.sum(exp_e)
        alphas = [alpha_seq]
        layer_mapping = [0] * 11
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        for t in range(1, T_test):
            jitter += torch.sum(torch.abs(alpha_seq[t] - alpha_seq[t - 1])).item()
        jitter /= (T_test - 1)
    elif model_type == 'static_decay':
        tau = 0.05
        alpha_stateless = torch.zeros(T_test, K)
        for t in range(T_test):
            max_e = torch.max(e_test[t] / tau)
            exp_e = torch.exp(e_test[t] / tau - max_e)
            alpha_stateless[t] = exp_e / torch.sum(exp_e)
        alphas = []
        for l_idx in range(11):
            lambda_l = 1.0 - (l_idx / 10.0)
            alpha_l = lambda_l * alpha_stateless + (1.0 - lambda_l) * 0.25
            alphas.append(alpha_l)
        layer_mapping = list(range(11))
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        for l_idx in range(11):
            layer_jit = 0.0
            for t in range(1, T_test):
                layer_jit += torch.sum(torch.abs(alphas[l_idx][t] - alphas[l_idx][t - 1])).item()
            jitter += layer_jit / (T_test - 1)
        jitter /= 11.0
    elif model_type == 'static_block':
        tau = 0.05
        alpha_stateless = torch.zeros(T_test, K)
        for t in range(T_test):
            max_e = torch.max(e_test[t] / tau)
            exp_e = torch.exp(e_test[t] / tau - max_e)
            alpha_stateless[t] = exp_e / torch.sum(exp_e)
        alphas = []
        for l_idx in range(11):
            if l_idx < 4:
                lambda_l = 1.0
            elif l_idx < 8:
                lambda_l = 0.5
            else:
                lambda_l = 0.0
            alpha_l = lambda_l * alpha_stateless + (1.0 - lambda_l) * 0.25
            alphas.append(alpha_l)
        layer_mapping = list(range(11))
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        for l_idx in range(11):
            layer_jit = 0.0
            for t in range(1, T_test):
                layer_jit += torch.sum(torch.abs(alphas[l_idx][t] - alphas[l_idx][t - 1])).item()
            jitter += layer_jit / (T_test - 1)
        jitter /= 11.0
    elif model_type == 'pac_kinetics':
        with torch.no_grad():
            alphas = router(e_test, sim_test)
        layer_mapping = [0] * 11
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        for t in range(1, T_test):
            jitter += torch.sum(torch.abs(alphas[0][t] - alphas[0][t - 1])).item()
        jitter /= (T_test - 1)
    elif model_type == 'lds_m3':
        with torch.no_grad():
            alphas = router(e_test, sim_test)
        layer_mapping = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        for m in range(3):
            m_jitter = 0.0
            for t in range(1, T_test):
                m_jitter += torch.sum(torch.abs(alphas[m][t] - alphas[m][t - 1])).item()
            jitter += m_jitter / (T_test - 1)
        jitter /= 3.0
    elif model_type == 'lds_m11':
        with torch.no_grad():
            alphas = router(e_test, sim_test)
        layer_mapping = list(range(11))
        h_L = propagate_layers_nonlinear(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        for m in range(11):
            m_jitter = 0.0
            for t in range(1, T_test):
                m_jitter += torch.sum(torch.abs(alphas[m][t] - alphas[m][t - 1])).item()
            jitter += m_jitter / (T_test - 1)
        jitter /= 11.0
        
    p = torch.softmax(logits, dim=-1)
    preds = torch.argmax(p, dim=-1)
    acc = torch.mean((preds == y_test).float()).item() * 100.0
    return acc, jitter

if __name__ == "__main__":
    print("Evaluating Non-linear Sandbox (GELU + LayerNorm) across 5 seeds:")
    seeds = [101, 102, 103, 104, 105]
    for config in ['orthogonal', 'overlapping']:
        print(f"\nManifold Configuration: {config.upper()}")
        for stream in ['homogeneous', 'heterogeneous']:
            acc_oracle, acc_uniform, acc_sable = [], [], []
            acc_decay, acc_block = [], []
            acc_m1, jit_m1 = [], []
            acc_m3, jit_m3 = [], []
            acc_m11, jit_m11 = [], []
            
            for seed in seeds:
                signatures, indices = generate_signatures(config)
                
                # Oracle
                o_acc, _ = evaluate_model_nonlinear(config, 'oracle', None, signatures, indices, seed, stream)
                acc_oracle.append(o_acc)
                
                # Uniform
                u_acc, _ = evaluate_model_nonlinear(config, 'uniform', None, signatures, indices, seed, stream)
                acc_uniform.append(u_acc)

                # SABLE (Raw)
                s_acc, _ = evaluate_model_nonlinear(config, 'sable_raw', None, signatures, indices, seed, stream)
                acc_sable.append(s_acc)
                
                # Static Decay
                dec_acc, _ = evaluate_model_nonlinear(config, 'static_decay', None, signatures, indices, seed, stream)
                acc_decay.append(dec_acc)
                
                # Static Block
                blk_acc, _ = evaluate_model_nonlinear(config, 'static_block', None, signatures, indices, seed, stream)
                acc_block.append(blk_acc)
                
                # Global PAC-Kinetics M=1
                m1_mapping = [0] * 11
                router_m1, sigs, ind = train_router_nonlinear(config, 1, m1_mapping, seed, regularized=True)
                acc1, jit1 = evaluate_model_nonlinear(config, 'pac_kinetics', router_m1, sigs, ind, seed, stream)
                acc_m1.append(acc1)
                jit_m1.append(jit1)
                
                # LDS-Kinetics M=3 (Tri-Block)
                m3_mapping = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
                router_m3, sigs, ind = train_router_nonlinear(config, 3, m3_mapping, seed, regularized=True)
                acc3, jit3 = evaluate_model_nonlinear(config, 'lds_m3', router_m3, sigs, ind, seed, stream)
                acc_m3.append(acc3)
                jit_m3.append(jit3)
                
                # LDS-Kinetics M=11
                m11_mapping = list(range(11))
                router_m11, sigs, ind = train_router_nonlinear(config, 11, m11_mapping, seed, regularized=True)
                acc11, jit11 = evaluate_model_nonlinear(config, 'lds_m11', router_m11, sigs, ind, seed, stream)
                acc_m11.append(acc11)
                jit_m11.append(jit11)
                
            import numpy as np
            print(f"  Stream: {stream.capitalize()}")
            print(f"    Expert Oracle: {np.mean(acc_oracle):.2f}% ± {np.std(acc_oracle):.2f}%")
            print(f"    Uniform Merge: {np.mean(acc_uniform):.2f}% ± {np.std(acc_uniform):.2f}%")
            print(f"    SABLE (Raw):   {np.mean(acc_sable):.2f}% ± {np.std(acc_sable):.2f}%")
            print(f"    Static Decay:  {np.mean(acc_decay):.2f}% ± {np.std(acc_decay):.2f}%")
            print(f"    Static Block:  {np.mean(acc_block):.2f}% ± {np.std(acc_block):.2f}%")
            print(f"    Global PAC-Kinetics (M=1): {np.mean(acc_m1):.2f}% ± {np.std(acc_m1):.2f}% | Jitter: {np.mean(jit_m1):.4f}")
            print(f"    LDS-Kinetics (M=3):         {np.mean(acc_m3):.2f}% ± {np.std(acc_m3):.2f}% | Jitter: {np.mean(jit_m3):.4f}")
            print(f"    LDS-Kinetics (M=11):        {np.mean(acc_m11):.2f}% ± {np.std(acc_m11):.2f}% | Jitter: {np.mean(jit_m11):.4f}")
            
            import scipy.stats as stats
            t_stat3, p_val3 = stats.ttest_rel(acc_m3, acc_m1)
            t_stat11, p_val11 = stats.ttest_rel(acc_m11, acc_m1)
            print(f"    Paired t-test (M=3 vs M=1):  t = {t_stat3:.4f} | p-value = {p_val3:.6f}")
            print(f"    Paired t-test (M=11 vs M=1): t = {t_stat11:.4f} | p-value = {p_val11:.6f}")
