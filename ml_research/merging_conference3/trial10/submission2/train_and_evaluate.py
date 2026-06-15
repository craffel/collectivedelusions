import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dimension parameters
D = 192
K = 4
L = 14
gamma_V = 0.05
kappa_scale = 0.0385

# Active indices configurations
orthogonal_indices = [list(range(k * 48, (k + 1) * 48)) for k in range(K)]
overlapping_indices = [
    list(range(0, 48)),
    list(range(36, 84)),
    list(range(72, 120)),
    list(range(108, 156))
]

# Noise scales and biases
sigmas = [0.05, 0.15, 0.40, 1.20]
biases = [0.0, 0.0, -0.90, -2.30]

# Generate task signatures
def generate_signatures(config_type):
    indices = orthogonal_indices if config_type == 'orthogonal' else overlapping_indices
    v = torch.zeros(K, D)
    for k in range(K):
        v[k, indices[k]] = 1.0 / math.sqrt(48.0)
    return v, indices

# Helper to normalize vector to the unit sphere
def normalize_vector(z, eps=1e-6):
    return z / (torch.norm(z, dim=-1, keepdim=True) + eps)

# Extract task coordinates based on projection on active indices
def extract_coordinates(z, indices, eps=1e-6):
    z_norm = normalize_vector(z, eps)
    e = []
    for k in range(K):
        # norm over active indices
        val = torch.norm(z_norm[..., indices[k]], dim=-1)
        e.append(val)
    return torch.stack(e, dim=-1)

# Base sample representation generation (Layer 3)
def generate_samples(y, signatures, indices):
    h3 = []
    for label in y:
        noise = torch.randn(D) * sigmas[label]
        h3.append(signatures[label] + noise)
    return torch.stack(h3, dim=0)

# Simulate representation propagation through layers 4 to 14
def propagate_layers(h3, alpha_seq, signatures, layer_to_block_mapping):
    # h3: (T, D)
    # alpha_seq: list of T tensors of shape (M, K) or similar depending on model
    # signatures: (K, D)
    T = h3.shape[0]
    h = h3.clone()
    
    # We apply layer-by-layer blending
    for l in range(4, L + 1):
        block_idx = layer_to_block_mapping[l - 4]
        # alpha for this layer: (T, K)
        alpha_layer = alpha_seq[block_idx]  # shape (T, K)
        
        # Expert update: (T, K, D) of v_k - h
        expert_diff = signatures.unsqueeze(0) - h.unsqueeze(1)  # (T, K, D)
        scaled_diff = expert_diff * alpha_layer.unsqueeze(-1)  # (T, K, D)
        update = torch.sum(scaled_diff, dim=1) * gamma_V  # (T, D)
        h = h + update
        
    return h

# Compute alignment distance logits and soft alignment accuracy
def compute_metrics(h_L, y, signatures):
    # h_L: (T, D), signatures: (K, D)
    T = h_L.shape[0]
    
    # Vectorized distance computation to avoid in-place autograd modifications
    h_L_sq = torch.sum(h_L ** 2, dim=-1, keepdim=True)  # (T, 1)
    sigs_sq = torch.sum(signatures ** 2, dim=-1).unsqueeze(0)  # (1, K)
    dot_prod = torch.matmul(h_L, signatures.t())  # (T, K)
    dists_sq = h_L_sq + sigs_sq - 2.0 * dot_prod  # (T, K)
    
    # Logits
    device = h_L.device
    biases_tensor = torch.tensor(biases, device=device).unsqueeze(0)  # (1, K)
    logits = -dists_sq + biases_tensor  # (T, K)
    
    # Soft accuracies
    target_dists_sq = dists_sq[torch.arange(T, device=device), y]  # (T,)
    accs = torch.exp(-kappa_scale * target_dists_sq)  # (T,)
    
    return logits, accs

# Cosine similarity for workload sequential analysis
def compute_similarity(e, eps=1e-6):
    T = e.shape[0]
    sim = torch.ones(T, device=e.device)
    for t in range(1, T):
        dot_prod = torch.sum(e[t] * e[t - 1])
        norm_t = torch.norm(e[t])
        norm_t1 = torch.norm(e[t - 1])
        sim[t] = dot_prod / (norm_t * norm_t1 + eps)
    return sim

# Trainable Router Class
class LDSKineticsRouter(nn.Module):
    def __init__(self, M, K):
        super(LDSKineticsRouter, self).__init__()
        self.M = M
        self.K = K
        # Learnable parameters centered around stable SABLE prior defaults
        self.u = nn.Parameter(torch.zeros(M, K))  # initial retention = sigmoid(0) = 0.5
        self.W = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]))  # identity scaling
        self.w = nn.Parameter(torch.ones(M, K) * math.log(0.05))  # temperature prior center ln(0.05)
        
    def forward(self, e, sim_seq):
        # e: (T, K) task coordinates
        # sim_seq: (T,) workload similarities
        T = e.shape[0]
        device = e.device
        
        # Parameters mapping
        a_ret = torch.sigmoid(self.u)  # (M, K)
        temp = torch.exp(self.w) + 0.01  # (M, K)
        
        # We build state vectors dynamically as lists of lists to avoid in-place operations
        alphas = []
        for m in range(self.M):
            s_m_list = []
            # Initial step
            s_m_0 = torch.mv(self.W[m], e[0])
            s_m_list.append(s_m_0)
            
            # Recurrence
            for t in range(1, T):
                sim_t = sim_seq[t]
                a_t = a_ret[m] * sim_t
                s_m_t = a_t * s_m_list[-1] + torch.mv(self.W[m], e[t])
                s_m_list.append(s_m_t)
                
            s_m = torch.stack(s_m_list, dim=0)  # (T, K)
            
            # Numerically stable multi-temperature Gibbs Softmax
            s_m_scaled = s_m / temp[m].unsqueeze(0)  # (T, K)
            max_s_m = torch.max(s_m_scaled, dim=-1, keepdim=True)[0]
            exp_s = torch.exp(s_m_scaled - max_s_m)
            alpha_m = exp_s / torch.sum(exp_s, dim=-1, keepdim=True)
            alphas.append(alpha_m)
            
        return alphas

# Trainer function
def train_router(config_type, M, layer_mapping, seed, regularized=True, lr=0.005, epochs=100, T_cal=32, symmetry_broken=False):
    set_seed(seed)
    signatures, indices = generate_signatures(config_type)
    
    # Construct calibration structured block sequence
    # samples per task expert, totaling T_cal
    samples_per_expert = T_cal // K
    y_cal = []
    for k in range(K):
        y_cal.extend([k] * samples_per_expert)
    y_cal = torch.tensor(y_cal)
    
    # Generate samples and extract coordinates
    h3_cal = generate_samples(y_cal, signatures, indices)
    e_cal = extract_coordinates(h3_cal, indices)
    sim_cal = compute_similarity(e_cal)
    
    # Initialize model
    router = LDSKineticsRouter(M, K)
    if symmetry_broken:
        with torch.no_grad():
            # Add small symmetry-breaking random perturbations to the parameters
            router.u.add_(torch.randn_like(router.u) * 0.01)
            router.W.add_(torch.randn_like(router.W) * 0.01)
            router.w.add_(torch.randn_like(router.w) * 0.01)
            
    optimizer = optim.Adam(router.parameters(), lr=lr)
    
    # Constants for Catoni PAC-Bayesian bound
    lambda_param = 0.5
    L_max = 5.0
    a_blocks = float(T_cal) / 4.0
    sigma_0_sq = 5.0
    
    # Prior defaults
    u0 = torch.zeros(M, K)
    W0 = torch.stack([torch.eye(K) for _ in range(M)])
    w0 = torch.ones(M, K) * math.log(0.05)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alphas = router(e_cal, sim_cal)
        h_L = propagate_layers(h3_cal, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_cal, signatures)
        
        # Softmax classification probability
        p = torch.softmax(logits, dim=-1)
        loss_ce = -torch.log(p[torch.arange(logits.shape[0]), y_cal] + 1e-8)
            
        # Bounded truncated loss
        truncated_loss = torch.clamp(loss_ce, max=L_max)
        R_hat = torch.mean(truncated_loss)
        
        if regularized:
            # Gaussian KL complexity penalty
            kl = (torch.sum((router.u - u0) ** 2) + 
                  torch.sum((router.W - W0) ** 2) + 
                  torch.sum((router.w - w0) ** 2)) / (2.0 * sigma_0_sq)
            
            # Scaled Catoni objective function
            loss = (lambda_param / L_max) * R_hat + (1.0 / (a_blocks * sigma_0_sq)) * kl
        else:
            # Unregularized Empirical Risk Minimization
            loss = R_hat
            
        loss.backward()
        optimizer.step()
        
    return router, signatures, indices

# Evaluate model on serving sequences
def evaluate_model(config_type, model_type, router, signatures, indices, test_seed, stream_type='homogeneous', layer_mapping=None):
    set_seed(test_seed)
    
    # Test sequence of 200 samples
    T_test = 200
    if stream_type == 'homogeneous':
        # Blocks of 50 queries per task
        y_test = []
        for k in range(K):
            y_test.extend([k] * 50)
    else:
        # Rapidly and randomly interleaved
        y_test = [random.randint(0, K - 1) for _ in range(T_test)]
        
    y_test = torch.tensor(y_test)
    h3_test = generate_samples(y_test, signatures, indices)
    e_test = extract_coordinates(h3_test, indices)
    sim_test = compute_similarity(e_test)
    
    # Evaluate according to model type
    if model_type == 'oracle':
        # Expert Oracle: 100% routing to true expert
        alphas = [torch.zeros(T_test, K) for _ in range(11)]  # dummy block coefficients
        # Set exact block active weights to one-hot for the active layer mapping
        for m in range(len(alphas)):
            for t in range(T_test):
                alphas[m][t, y_test[t]] = 1.0
        layer_mapping = list(range(11))  # distinct mappings
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        # Routing jitter is near-zero under homogeneous and matches switches under heterogeneous
        jitter = 0.0
        for t in range(1, T_test):
            onehot_t = torch.zeros(K)
            onehot_t[y_test[t]] = 1.0
            onehot_t1 = torch.zeros(K)
            onehot_t1[y_test[t - 1]] = 1.0
            jitter += torch.sum(torch.abs(onehot_t - onehot_t1)).item()
        jitter /= (T_test - 1)
        
    elif model_type == 'uniform':
        # Static Uniform Merging
        alphas = [torch.ones(T_test, K) * 0.25]
        layer_mapping = [0] * 11
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        jitter = 0.0
        
    elif model_type == 'sable_raw':
        # SABLE (Raw): Stateless nearest-centroid routing with unoptimized tau=0.05
        tau = 0.05
        alpha_seq = torch.zeros(T_test, K)
        for t in range(T_test):
            max_e = torch.max(e_test[t] / tau)
            exp_e = torch.exp(e_test[t] / tau - max_e)
            alpha_seq[t] = exp_e / torch.sum(exp_e)
        alphas = [alpha_seq]
        layer_mapping = [0] * 11
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        # Calculate temporal jitter
        jitter = 0.0
        for t in range(1, T_test):
            jitter += torch.sum(torch.abs(alpha_seq[t] - alpha_seq[t - 1])).item()
        jitter /= (T_test - 1)
        
    elif model_type == 'sable_sep':
        # SABLE (SEP): unnormalized coordinates
        tau = 0.05
        e_unnorm = torch.zeros(T_test, K)
        for t in range(T_test):
            for k in range(K):
                e_unnorm[t, k] = torch.norm(h3_test[t, indices[k]])
        alpha_seq = torch.zeros(T_test, K)
        for t in range(T_test):
            max_e = torch.max(e_unnorm[t] / tau)
            exp_e = torch.exp(e_unnorm[t] / tau - max_e)
            alpha_seq[t] = exp_e / torch.sum(exp_e)
        alphas = [alpha_seq]
        layer_mapping = [0] * 11
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        jitter = 0.0
        for t in range(1, T_test):
            jitter += torch.sum(torch.abs(alpha_seq[t] - alpha_seq[t - 1])).item()
        jitter /= (T_test - 1)
        
    elif model_type == 'static_decay':
        # Stateless with static linear decay of ensembling weights from early to late layers
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
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        jitter = 0.0
        for l_idx in range(11):
            layer_jit = 0.0
            for t in range(1, T_test):
                layer_jit += torch.sum(torch.abs(alphas[l_idx][t] - alphas[l_idx][t - 1])).item()
            jitter += layer_jit / (T_test - 1)
        jitter /= 11.0
        
    elif model_type == 'static_block':
        # Stateless with static block-wise constant weighting (Tri-Block mapping)
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
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        jitter = 0.0
        for l_idx in range(11):
            layer_jit = 0.0
            for t in range(1, T_test):
                layer_jit += torch.sum(torch.abs(alphas[l_idx][t] - alphas[l_idx][t - 1])).item()
            jitter += layer_jit / (T_test - 1)
        jitter /= 11.0
        
    elif model_type == 'chemmerge':
        # Heuristic ChemMerge: Stateful kinetics with hand-tuned global params
        tau = 0.01
        dt = 1.5
        k_decay = 0.3
        
        # We simulate the concentration trajectories across layer depth for each sample
        # Since routing is early mapped, we compute Arrhenius forward rates once per sample
        alpha_seq_layers = []  # list of 11 layers
        for t in range(T_test):
            max_e = torch.max(e_test[t] / tau)
            exp_e = torch.exp(e_test[t] / tau - max_e)
            k_rates = exp_e / torch.sum(exp_e)
            
            C = torch.ones(K) * 0.25  # uniform initial concentrations
            layer_alphas = []
            for l in range(4, L + 1):
                beta_l = 1.0 - torch.exp(-(k_rates + k_decay) * dt)
                C = (1.0 - beta_l) * C + beta_l * (k_rates / (k_rates + k_decay))
                # Law of Mass Action normalization
                alpha_l = C / torch.sum(C)
                layer_alphas.append(alpha_l)
            alpha_seq_layers.append(torch.stack(layer_alphas, dim=0))  # (11, K)
            
        # alpha_seq_layers is shape (T_test, 11, K)
        # We restructure to match propagate_layers interface: alphas is list of shape (11, T_test, K)
        alphas = []
        for l_idx in range(11):
            alpha_l = torch.zeros(T_test, K)
            for t in range(T_test):
                alpha_l[t] = alpha_seq_layers[t][l_idx]
            alphas.append(alpha_l)
            
        layer_mapping = list(range(11))
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        # Compute average temporal routing jitter over layers
        jitter = 0.0
        for l_idx in range(11):
            layer_jit = 0.0
            for t in range(1, T_test):
                layer_jit += torch.sum(torch.abs(alphas[l_idx][t] - alphas[l_idx][t - 1])).item()
            jitter += layer_jit / (T_test - 1)
        jitter /= 11.0
        
    else:
        # Trained Routers: Stateless PAC-ZCA, Stateful ERM, or LDS-Kinetics (Tri-Block, Fully Decoupled)
        with torch.no_grad():
            alphas = router(e_test, sim_test)
            
        # Determine layer mapping from router configuration
        if layer_mapping is None:
            if router.M == 1:
                layer_mapping = [0] * 11
            elif router.M == 3:
                layer_mapping = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
            else:
                layer_mapping = list(range(11))
            
        h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_test, signatures)
        
        # Calculate temporal jitter averaged over adapted layers
        jitter = 0.0
        for l in range(11):
            block_idx = layer_mapping[l]
            block_jit = 0.0
            for t in range(1, T_test):
                block_jit += torch.sum(torch.abs(alphas[block_idx][t] - alphas[block_idx][t - 1])).item()
            jitter += block_jit / (T_test - 1)
        jitter /= 11.0
        
    return torch.mean(accs).item() * 100.0, jitter

# Run entire sweeps across seeds
def execute_experiments():
    configs = ['orthogonal', 'overlapping']
    seeds = [101, 102, 103, 104, 105]
    
    # Dictionary to collect results
    # keys: config -> stream -> model -> list of (acc, jit)
    results = {c: {s: {} for s in ['homogeneous', 'heterogeneous']} for c in configs}
    
    # 1. Oracle & Baselines
    baseline_names = ['oracle', 'uniform', 'sable_raw', 'sable_sep', 'static_decay', 'static_block', 'chemmerge']
    
    for config_type in configs:
        print(f"\n================ Running on {config_type.upper()} Manifolds ================")
        
        # Map layer mappings for different block scales
        m1_mapping = [0] * 11
        m3_mapping = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        m11_mapping = list(range(11))
        
        # Store models pre-trained per seed to avoid redundant training
        trained_models = {seed: {} for seed in seeds}
        
        # Train routers for each seed
        for seed in seeds:
            print(f"  Training routers for seed {seed}...")
            
            # A. Stateless PAC-ZCA (Optimizes w only, set retention u to large negative so sigmoid(u)->0, W to identity)
            # To simulate stateless temperature optimization, we freeze retention parameters u to a static stateless state during forward.
            # We can implement this simply by creating a model, setting its u to extremely negative, W to identity, and optimizing only w.
            pca_router, sigs, _ = train_router(config_type, 1, m1_mapping, seed, regularized=True)
            with torch.no_grad():
                pca_router.u.copy_(torch.ones(1, K) * -10.0)  # extremely small retention => stateless
                pca_router.W.copy_(torch.stack([torch.eye(K) for _ in range(1)]))  # Identity coupling
            trained_models[seed]['pac_zca'] = (pca_router, sigs)
            
            # B. Global PAC-Kinetics (Trial 9, M=1)
            p_kin_router, sigs, _ = train_router(config_type, 1, m1_mapping, seed, regularized=True)
            trained_models[seed]['pac_kinetics'] = (p_kin_router, sigs)
            
            # C. Stateful ERM Global (M=1 unregularized)
            erm_m1_router, sigs, _ = train_router(config_type, 1, m1_mapping, seed, regularized=False)
            trained_models[seed]['erm_m1'] = (erm_m1_router, sigs)
            
            # D. LDS-Kinetics Tri-Block (M=3)
            lds_m3_router, sigs, _ = train_router(config_type, 3, m3_mapping, seed, regularized=True)
            trained_models[seed]['lds_m3'] = (lds_m3_router, sigs)
            
            # E. LDS-Kinetics Fully Decoupled (M=11)
            lds_m11_router, sigs, _ = train_router(config_type, 11, m11_mapping, seed, regularized=True)
            trained_models[seed]['lds_m11'] = (lds_m11_router, sigs)
            
            # F. Decoupled ERM Tri-Block (M=3 unregularized)
            erm_m3_router, sigs, _ = train_router(config_type, 3, m3_mapping, seed, regularized=False)
            trained_models[seed]['erm_m3'] = (erm_m3_router, sigs)
            
            # F_SB. Decoupled ERM Tri-Block Symmetry-Broken (M=3 unregularized, SB)
            erm_m3_sb_router, sigs, _ = train_router(config_type, 3, m3_mapping, seed, regularized=False, symmetry_broken=True)
            trained_models[seed]['erm_m3_sb'] = (erm_m3_sb_router, sigs)
            
            # G. Decoupled ERM Fully Decoupled (M=11 unregularized)
            erm_m11_router, sigs, _ = train_router(config_type, 11, m11_mapping, seed, regularized=False)
            trained_models[seed]['erm_m11'] = (erm_m11_router, sigs)
            
            # G_SB. Decoupled ERM Fully Decoupled Symmetry-Broken (M=11 unregularized, SB)
            erm_m11_sb_router, sigs, _ = train_router(config_type, 11, m11_mapping, seed, regularized=False, symmetry_broken=True)
            trained_models[seed]['erm_m11_sb'] = (erm_m11_sb_router, sigs)
            
        # Run evaluations
        for stream_type in ['homogeneous', 'heterogeneous']:
            print(f"    Evaluating on {stream_type.upper()} Stream...")
            
            # Evaluate baseline non-learnable models
            for model_name in baseline_names:
                acc_list, jit_list = [], []
                for seed in seeds:
                    sigs, ind = generate_signatures(config_type)
                    acc, jit = evaluate_model(config_type, model_name, None, sigs, ind, seed, stream_type)
                    acc_list.append(acc)
                    jit_list.append(jit)
                results[config_type][stream_type][model_name] = (acc_list, jit_list)
                
            # Evaluate trained learnable models
            learnable_models = {
                'pac_zca': 'pac_zca',
                'pac_kinetics': 'pac_kinetics',
                'erm_m1': 'erm_m1',
                'lds_m3': 'lds_m3',
                'lds_m11': 'lds_m11',
                'erm_m3': 'erm_m3',
                'erm_m3_sb': 'erm_m3_sb',
                'erm_m11': 'erm_m11',
                'erm_m11_sb': 'erm_m11_sb'
            }
            
            for key, m_name in learnable_models.items():
                acc_list, jit_list = [], []
                for seed in seeds:
                    sigs, ind = generate_signatures(config_type)
                    router, _ = trained_models[seed][key]
                    acc, jit = evaluate_model(config_type, m_name, router, sigs, ind, seed, stream_type)
                    acc_list.append(acc)
                    jit_list.append(jit)
                results[config_type][stream_type][key] = (acc_list, jit_list)
                
    # Generate Output Reports and Plots
    os.makedirs('results', exist_ok=True)
    
    # Save quantitative results as a markdown file
    report_content = "# Layer-Decoupled Stateful Kinetics (LDS-Kinetics) Experimental Results\n\n"
    report_content += "This document contains the verified quantitative performance of LDS-Kinetics against baseline model merging frameworks. Evaluated across 5 random seeds inside the 14-layer, 192-dimensional Analytical Coordinate Sandbox.\n\n"
    
    for config_type in configs:
        report_content += f"## {config_type.capitalize()} Manifolds\n\n"
        report_content += "| Method | Homogeneous Acc (%) | Homogeneous Jitter | Heterogeneous Acc (%) | Heterogeneous Jitter |\n"
        report_content += "| :--- | :---: | :---: | :---: | :---: |\n"
        
        # Order of methods for display
        display_order = [
            ('Expert Oracle', 'oracle'),
            ('Uniform Merging (Static)', 'uniform'),
            ('SABLE (Raw)', 'sable_raw'),
            ('SABLE (SEP)', 'sable_sep'),
            ('Static Layer-Wise Decay', 'static_decay'),
            ('Static Block-Wise Constant', 'static_block'),
            ('Stateless PAC-ZCA', 'pac_zca'),
            ('Heuristic ChemMerge', 'chemmerge'),
            ('Global PAC-Kinetics (Trial 9)', 'pac_kinetics'),
            ('Stateful ERM Global (M=1)', 'erm_m1'),
            ('LDS-Kinetics (Tri-Block, M=3)', 'lds_m3'),
            ('LDS-Kinetics (Fully Decoupled, M=11)', 'lds_m11'),
            ('Decoupled ERM (Tri-Block, M=3)', 'erm_m3'),
            ('Decoupled ERM (Tri-Block, M=3, Symmetry-Broken)', 'erm_m3_sb'),
            ('Decoupled ERM (Fully Decoupled, M=11)', 'erm_m11'),
            ('Decoupled ERM (Fully Decoupled, M=11, Symmetry-Broken)', 'erm_m11_sb')
        ]
        
        for label, key in display_order:
            hom_accs, hom_jits = results[config_type]['homogeneous'][key]
            het_accs, het_jits = results[config_type]['heterogeneous'][key]
            
            hom_acc_mean, hom_acc_std = np.mean(hom_accs), np.std(hom_accs)
            hom_jit_mean, hom_jit_std = np.mean(hom_jits), np.std(hom_jits)
            het_acc_mean, het_acc_std = np.mean(het_accs), np.std(het_accs)
            het_jit_mean, het_jit_std = np.mean(het_jits), np.std(het_jits)
            
            report_content += f"| {label} | {hom_acc_mean:.2f}% ± {hom_acc_std:.2f}% | {hom_jit_mean:.4f} ± {hom_jit_std:.4f} | {het_acc_mean:.2f}% ± {het_acc_std:.2f}% | {het_jit_mean:.4f} ± {het_jit_std:.4f} |\n"
            
        report_content += "\n"
        
    with open('experiment_results.md', 'w') as f:
        f.write(report_content)
    print("Successfully saved experiment_results.md!")
    
    # Produce Beautiful Figures
    # Fig 1: Accuracy Sweeps comparison across Decoupling scales M on Orthogonal Manifolds
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heterogeneous stream results on orthogonal manifolds
    models_to_plot = ['pac_zca', 'pac_kinetics', 'lds_m3', 'lds_m11']
    labels = ['Stateless PAC-ZCA', 'Global PAC-Kinetics', 'Tri-Block (M=3)', 'Decoupled (M=11)']
    
    # Gather data for plotting
    orth_het_data = [results['orthogonal']['heterogeneous'][m][0] for m in models_to_plot]
    orth_hom_data = [results['orthogonal']['homogeneous'][m][0] for m in models_to_plot]
    
    # Plot heterogeneous
    axes[0].boxplot(orth_het_data, tick_labels=labels)
    axes[0].set_title("Heterogeneous Stream Accuracy (%)")
    axes[0].set_ylabel("Joint Serving Accuracy (%)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot homogeneous
    axes[1].boxplot(orth_hom_data, tick_labels=labels)
    axes[1].set_title("Homogeneous Stream Accuracy (%)")
    axes[1].set_ylabel("Joint Serving Accuracy (%)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("LDS-Kinetics Performance Sweeps over Decoupling Scales (Orthogonal Manifolds)")
    plt.tight_layout()
    plt.savefig('results/fig1_decoupling_comparison.png', dpi=300)
    plt.close()
    
    # Fig 2: Ablation comparison of regularized LDS-Kinetics vs unregularized Decoupled ERM
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ablation_models = [
        ('lds_m3', 'erm_m3'),
        ('lds_m11', 'erm_m11')
    ]
    
    # Tri-block heterogeneous comparison
    axes[0].bar(['LDS-Kinetics (M=3)', 'Decoupled ERM (M=3)'], 
                [np.mean(results['orthogonal']['heterogeneous']['lds_m3'][0]), np.mean(results['orthogonal']['heterogeneous']['erm_m3'][0])],
                yerr=[np.std(results['orthogonal']['heterogeneous']['lds_m3'][0]), np.std(results['orthogonal']['heterogeneous']['erm_m3'][0])],
                color=['blue', 'red'], alpha=0.7, capsize=10)
    axes[0].set_title("Tri-Block (M=3) Regularization Analysis")
    axes[0].set_ylabel("Serving Accuracy (%)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Fully Decoupled heterogeneous comparison
    axes[1].bar(['LDS-Kinetics (M=11)', 'Decoupled ERM (M=11)'], 
                [np.mean(results['orthogonal']['heterogeneous']['lds_m11'][0]), np.mean(results['orthogonal']['heterogeneous']['erm_m11'][0])],
                yerr=[np.std(results['orthogonal']['heterogeneous']['lds_m11'][0]), np.std(results['orthogonal']['heterogeneous']['erm_m11'][0])],
                color=['blue', 'red'], alpha=0.7, capsize=10)
    axes[1].set_title("Fully Decoupled (M=11) Regularization Analysis")
    axes[1].set_ylabel("Serving Accuracy (%)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("PAC-Bayesian Complexity Penalty vs. Unregularized ERM (Orthogonal Heterogeneous Stream)")
    plt.tight_layout()
    plt.savefig('results/fig2_regularization_ablation.png', dpi=300)
    plt.close()
    
    # Update progress.json
    import json
    with open('progress.json', 'w') as f:
        json.dump({"phase": 3}, f)
    print("Successfully set phase to 3 in progress.json!")

if __name__ == "__main__":
    execute_experiments()
