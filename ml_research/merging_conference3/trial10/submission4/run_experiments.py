import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)

# Hyperparameters
D = 192
K = 4
S = D // K  # 48
sigmas = [0.05, 0.15, 0.40, 1.20]
biases = [0.0, 0.0, -0.90, -2.30]
gamma_V = 0.05
L = 14

# Define active index intervals
# k-th task has active indices [k*S, (k+1)*S)
v = torch.zeros(K, D)
for k in range(K):
    v[k, k*S:(k+1)*S] = 1.0 / np.sqrt(S)

# Helper: Symmetric Quantization Operator
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def round_ste(x):
    return RoundSTE.apply(x)

def quantize_symmetric(x, bits=8, use_ste=False, per_sample=False):
    limit = 2 ** (bits - 1)
    if per_sample and x.ndim >= 2:
        max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
    else:
        max_val = torch.max(torch.abs(x))
    scale = max_val / (limit - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize
    if use_ste:
        q = torch.clamp(round_ste(x / scale), -limit, limit - 1)
    else:
        q = torch.clamp(torch.round(x / scale), -limit, limit - 1)
    # Dequantize
    x_tilde = q * scale
    return q, scale, x_tilde

def quantize_alpha(alpha, bits=4):
    levels = 2 ** bits - 1  # 15
    q = torch.clamp(torch.round(alpha * levels), 0, levels)
    alpha_tilde = q / levels
    return alpha_tilde

class QuantizeAlphaSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, bits=4):
        levels = 2 ** bits - 1
        q = torch.clamp(torch.round(alpha * levels), 0, levels)
        return q / levels
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def quantize_alpha_ste(alpha, bits=4):
    return QuantizeAlphaSTE.apply(alpha, bits)

def ProjectToSimplex(alpha):
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_alpha = torch.where(sum_alpha == 0.0, torch.ones_like(sum_alpha), sum_alpha)
    return alpha / sum_alpha

def discrete_simplex_projection(alpha, bits=4):
    levels = 2 ** bits - 1  # 15
    is_batched = (alpha.ndim == 2)
    if not is_batched:
        alpha = alpha.unsqueeze(0)
    B, K = alpha.shape
    alpha = torch.clamp(alpha, min=0.0)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_alpha = torch.where(sum_alpha == 0.0, torch.ones_like(sum_alpha), sum_alpha)
    alpha = alpha / sum_alpha
    
    scaled = alpha * levels
    q = torch.floor(scaled)
    remainders = scaled - q
    
    current_sums = torch.sum(q, dim=-1, keepdim=True)
    shortfalls = (levels - current_sums).long()
    
    _, indices = torch.sort(remainders, dim=-1, descending=True)
    
    rank = torch.zeros_like(remainders, dtype=torch.long)
    for b in range(B):
        rank[b, indices[b]] = torch.arange(K, device=alpha.device)
        
    increment = (rank < shortfalls).float()
    q = q + increment
    q = torch.clamp(q, min=0, max=levels)
    alpha_tilde = q / levels
    
    if not is_batched:
        alpha_tilde = alpha_tilde.squeeze(0)
        q = q.squeeze(0)
    return q, alpha_tilde

class DiscreteSimplexSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, bits=4):
        _, alpha_tilde = discrete_simplex_projection(alpha, bits)
        return alpha_tilde
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def discrete_simplex_ste(alpha, bits=4):
    return DiscreteSimplexSTE.apply(alpha, bits)

# Helper: Get Toeplitz Covariance Matrix and its square root
def get_covariance_half(rho, D):
    indices = torch.arange(D)
    Sigma = rho ** torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    Sigma_half = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    return Sigma_half

# Data Generation function
def generate_sandbox_data(num_samples_per_task, Sigma_half, sigmas, D):
    h3_all = []
    labels_all = []
    v_primes = []
    
    for k in range(4):
        v_k = torch.zeros(D)
        v_k[k * S : (k+1) * S] = 1.0 / np.sqrt(S)
        v_prime_k = Sigma_half @ v_k
        v_primes.append(v_prime_k)
        
        noise = torch.randn(num_samples_per_task, D) * sigmas[k]
        h3 = v_prime_k.unsqueeze(0) + noise
        
        h3_all.append(h3)
        labels_all.append(torch.full((num_samples_per_task,), k, dtype=torch.long))
        
    h3_all = torch.cat(h3_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    v_primes = torch.stack(v_primes, dim=0)
    
    return h3_all, labels_all, v_primes

# Parametric Router module
class ParametricRouter(nn.Module):
    def __init__(self, D=192, K=4, init_zero=True):
        super().__init__()
        self.W_g = nn.Parameter(torch.zeros(K, D))
        self.b_g = nn.Parameter(torch.zeros(K))
        if not init_zero:
            nn.init.normal_(self.W_g, std=0.1)
            nn.init.normal_(self.b_g, std=0.1)

    def forward(self, h3, quantize_mode=False, use_ste=False, per_sample=False):
        if quantize_mode:
            _, _, h3_tilde = quantize_symmetric(h3, bits=8, use_ste=use_ste, per_sample=per_sample)
            logits = h3_tilde @ self.W_g.t() + self.b_g
        else:
            logits = h3 @ self.W_g.t() + self.b_g
        alpha = torch.softmax(logits, dim=-1)
        return alpha

# Main evaluation loop for a given configuration
def evaluate_all_methods(N_cal, rho, evaluation_seeds=[2026]):
    print(f"\n--- Running evaluation with N_cal={N_cal}, rho={rho} ---")
    Sigma_half = get_covariance_half(rho, D)
    
    # 1. Calibration Data Generation
    h3_cal, labels_cal, v_primes = generate_sandbox_data(N_cal // K, Sigma_half, sigmas, D)
    
    # 2. Test Data Generation
    num_test = 1000 # 250 samples per task
    h3_test, labels_test, _ = generate_sandbox_data(num_test // K, Sigma_half, sigmas, D)
    
    # Compute centroids over calibration data (Float32 SABLE centroids)
    centroids_f32 = torch.zeros(K, D)
    for k in range(K):
        mask = (labels_cal == k)
        centroids_f32[k] = torch.mean(h3_cal[mask], dim=0)
        
    # Quantized Centroid Calibration (QCC)
    centroids_qcc = torch.zeros(K, D)
    for k in range(K):
        mask = (labels_cal == k)
        # Average first in Float32, then quantize the final centroid to get INT8 representation
        mean_h3 = torch.mean(h3_cal[mask], dim=0)
        q_centroid, _, _ = quantize_symmetric(mean_h3, bits=8)
        centroids_qcc[k] = q_centroid
        
    # Standard scale factor for QCC centroid evaluation
    # Since QCC computes centroids directly on INT8 space, we use them as quantized integers.
    
    results = {}
    
    # ------------------ EXPERT ORACLE (FLOAT32) ------------------
    accuracies = []
    for k in range(K):
        mask = (labels_test == k)
        h = h3_test[mask]
        v_prime_k = v_primes[k].unsqueeze(0)
        
        # Propagate as expert k
        for l in range(4, 15):
            h = h + gamma_V * (v_prime_k - h)
            
        logits = torch.zeros(h.shape[0], K)
        for j in range(K):
            dist = torch.sum((h - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
            logits[:, j] = -dist + biases[j]
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == k).float()).item()
        accuracies.append(acc)
    results['Expert Oracle'] = (np.mean(accuracies), 0.0) # Accuracy, Jitter

    # ------------------ UNIFORM MERGING (FLOAT32) ------------------
    h = h3_test.clone()
    alpha = torch.full((K,), 0.25)
    for l in range(4, 15):
        pull = torch.zeros_like(h)
        for j in range(K):
            pull += alpha[j] * gamma_V * (v_primes[j].unsqueeze(0) - h)
        h = h + pull
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        dist = torch.sum((h - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['Uniform Merging'] = (acc, 0.0)

    # ------------------ UNIFORM MERGING (QUANTIZED) ------------------
    # Activations and weights quantized
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8)
    alpha_q = ProjectToSimplex(quantize_alpha(torch.full((K,), 0.25), bits=4))
    for l in range(4, 15):
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
            pull += alpha_q[j] * gamma_V * (v_prime_q - h)
        _, _, h = quantize_symmetric(h + pull, bits=8)
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['Uniform Merging (Quantized)'] = (acc, 0.0)

    # ------------------ SABLE (FLOAT32, tau=0.05) ------------------
    h = h3_test.clone()
    tau_sable = 0.05
    jitter_accum = 0.0
    alpha_prev = None
    for l in range(4, 15):
        # Cosine similarity
        norm_h = torch.norm(h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_f32, dim=-1, keepdim=True).t()
        sims = (h @ centroids_f32.t()) / (norm_h @ norm_c + 1e-8)
        alpha = torch.softmax(sims / tau_sable, dim=-1)
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha - alpha_prev, dim=-1)).item()
        alpha_prev = alpha
        
        pull = torch.zeros_like(h)
        for j in range(K):
            pull += alpha[:, j:j+1] * gamma_V * (v_primes[j].unsqueeze(0) - h)
        h = h + pull
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        dist = torch.sum((h - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['SABLE'] = (acc, jitter_accum / 10.0)

    # ------------------ SABLE (QUANTIZED-NAIVE) ------------------
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8)
    tau_sable = 0.05
    jitter_accum = 0.0
    alpha_prev = None
    for l in range(4, 15):
        _, _, centroids_q = quantize_symmetric(centroids_f32, bits=8)
        norm_h = torch.norm(h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_q, dim=-1, keepdim=True).t()
        sims = (h @ centroids_q.t()) / (norm_h @ norm_c + 1e-8)
        alpha = torch.softmax(sims / tau_sable, dim=-1)
        alpha_q = ProjectToSimplex(quantize_alpha(alpha, bits=4))
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha_q - alpha_prev, dim=-1)).item()
        alpha_prev = alpha_q
        
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
            pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
        _, _, h = quantize_symmetric(h + pull, bits=8)
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['SABLE (Quantized-Naive)'] = (acc, jitter_accum / 10.0)

    # ------------------ SABLE (QA-MERGE: QCC + EF-SMOOTH, beta=0.8) ------------------
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8, per_sample=True)
    tau_sable = 0.05
    beta_ef = 0.8
    jitter_accum = 0.0
    alpha_prev = None
    error_ef = torch.zeros(h.shape[0], K, device=h.device) # error feedback vector shape: (B, K)
    error_act = torch.zeros_like(h) # Activation Error Feedback (AEF)
    for l in range(4, 15):
        # QCC Centroids evaluation directly in quantized INT8 space
        q_h, _, _ = quantize_symmetric(h, bits=8, per_sample=True)
        # Compute cosine similarity in integer space (scale-invariant)
        norm_h = torch.norm(q_h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_qcc, dim=-1, keepdim=True).t()
        sims = (q_h @ centroids_qcc.t()) / (norm_h @ norm_c + 1e-8)
        alpha = torch.softmax(sims / tau_sable, dim=-1)
        
        # EF-Smooth using discrete simplex projection
        alpha_corrected = alpha + beta_ef * error_ef
        _, alpha_q = discrete_simplex_projection(alpha_corrected, bits=4)
        error_ef = alpha_corrected - alpha_q
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha_q - alpha_prev, dim=-1)).item()
        alpha_prev = alpha_q
        
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
            pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
            
        # Apply Activation Error Feedback (AEF)
        h_next_unquantized = h + pull + error_act
        _, _, h_next_quantized = quantize_symmetric(h_next_unquantized, bits=8, per_sample=True)
        error_act = h_next_unquantized - h_next_quantized
        h = h_next_quantized
        
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['SABLE (QA-Merge)'] = (acc, jitter_accum / 10.0)

    # ------------------ CHEMMERGE (FLOAT32, dt=1.5, k_decay=0.3) ------------------
    h = h3_test.clone()
    tau_chem = 0.05
    dt = 1.5
    k_decay = 0.3
    C = torch.full((h.shape[0], K), 0.25) # Initial concentration
    jitter_accum = 0.0
    alpha_prev = None
    for l in range(4, 15):
        norm_h = torch.norm(h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_f32, dim=-1, keepdim=True).t()
        sims = (h @ centroids_f32.t()) / (norm_h @ norm_c + 1e-8)
        R = torch.softmax(sims / tau_chem, dim=-1)
        
        C = C + dt * (-k_decay * C + R)
        C = torch.clamp(C, min=0.0)
        alpha = C / torch.sum(C, dim=-1, keepdim=True)
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha - alpha_prev, dim=-1)).item()
        alpha_prev = alpha
        
        pull = torch.zeros_like(h)
        for j in range(K):
            pull += alpha[:, j:j+1] * gamma_V * (v_primes[j].unsqueeze(0) - h)
        h = h + pull
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        dist = torch.sum((h - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['ChemMerge'] = (acc, jitter_accum / 10.0)

    # ------------------ CHEMMERGE (QUANTIZED-NAIVE) ------------------
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8)
    tau_chem = 0.05
    dt = 1.5
    k_decay = 0.3
    C = torch.full((h.shape[0], K), 0.25)
    jitter_accum = 0.0
    alpha_prev = None
    for l in range(4, 15):
        _, _, centroids_q = quantize_symmetric(centroids_f32, bits=8)
        norm_h = torch.norm(h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_q, dim=-1, keepdim=True).t()
        sims = (h @ centroids_q.t()) / (norm_h @ norm_c + 1e-8)
        R = torch.softmax(sims / tau_chem, dim=-1)
        
        C = C + dt * (-k_decay * C + R)
        C = torch.clamp(C, min=0.0)
        alpha = C / torch.sum(C, dim=-1, keepdim=True)
        alpha_q = ProjectToSimplex(quantize_alpha(alpha, bits=4))
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha_q - alpha_prev, dim=-1)).item()
        alpha_prev = alpha_q
        
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
            pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
        _, _, h = quantize_symmetric(h + pull, bits=8)
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['ChemMerge (Quantized-Naive)'] = (acc, jitter_accum / 10.0)

    # ------------------ CHEMMERGE (QA-MERGE: QCC + EF-SMOOTH) ------------------
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8, per_sample=True)
    tau_chem = 0.05
    dt = 1.5
    k_decay = 0.3
    beta_ef = 0.8
    C = torch.full((h.shape[0], K), 0.25)
    jitter_accum = 0.0
    alpha_prev = None
    error_ef = torch.zeros(h.shape[0], K, device=h.device)
    error_act = torch.zeros_like(h) # Activation Error Feedback (AEF)
    for l in range(4, 15):
        # QCC
        q_h, _, _ = quantize_symmetric(h, bits=8, per_sample=True)
        # Compute cosine similarity in integer space (scale-invariant)
        norm_h = torch.norm(q_h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_qcc, dim=-1, keepdim=True).t()
        sims = (q_h @ centroids_qcc.t()) / (norm_h @ norm_c + 1e-8)
        R = torch.softmax(sims / tau_chem, dim=-1)
        
        C = C + dt * (-k_decay * C + R)
        C = torch.clamp(C, min=0.0)
        alpha = C / torch.sum(C, dim=-1, keepdim=True)
        
        # EF-Smooth using discrete simplex projection
        alpha_corrected = alpha + beta_ef * error_ef
        _, alpha_q = discrete_simplex_projection(alpha_corrected, bits=4)
        error_ef = alpha_corrected - alpha_q
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha_q - alpha_prev, dim=-1)).item()
        alpha_prev = alpha_q
        
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
            pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
            
        # Apply Activation Error Feedback (AEF)
        h_next_unquantized = h + pull + error_act
        _, _, h_next_quantized = quantize_symmetric(h_next_unquantized, bits=8, per_sample=True)
        error_act = h_next_unquantized - h_next_quantized
        h = h_next_quantized
        
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['ChemMerge (QA-Merge)'] = (acc, jitter_accum / 10.0)

    # ------------------ MOMENTUM-MERGE (FLOAT32, beta=0.6, tau=0.005) ------------------
    h = h3_test.clone()
    tau_mom = 0.005
    beta_mom = 0.60
    alpha = torch.full((h.shape[0], K), 0.25)
    jitter_accum = 0.0
    alpha_prev = None
    for l in range(4, 15):
        norm_h = torch.norm(h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_f32, dim=-1, keepdim=True).t()
        sims = (h @ centroids_f32.t()) / (norm_h @ norm_c + 1e-8)
        alpha_target = torch.softmax(sims / tau_mom, dim=-1)
        
        alpha = beta_mom * alpha + (1.0 - beta_mom) * alpha_target
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha - alpha_prev, dim=-1)).item()
        alpha_prev = alpha
        
        pull = torch.zeros_like(h)
        for j in range(K):
            pull += alpha[:, j:j+1] * gamma_V * (v_primes[j].unsqueeze(0) - h)
        h = h + pull
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        dist = torch.sum((h - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['Momentum-Merge'] = (acc, jitter_accum / 10.0)

    # ------------------ MOMENTUM-MERGE (QUANTIZED-NAIVE) ------------------
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8)
    tau_mom = 0.005
    beta_mom = 0.60
    alpha = torch.full((h.shape[0], K), 0.25)
    jitter_accum = 0.0
    alpha_prev = None
    for l in range(4, 15):
        _, _, centroids_q = quantize_symmetric(centroids_f32, bits=8)
        norm_h = torch.norm(h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_q, dim=-1, keepdim=True).t()
        sims = (h @ centroids_q.t()) / (norm_h @ norm_c + 1e-8)
        alpha_target = torch.softmax(sims / tau_mom, dim=-1)
        
        alpha = beta_mom * alpha + (1.0 - beta_mom) * alpha_target
        alpha_q = ProjectToSimplex(quantize_alpha(alpha, bits=4))
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha_q - alpha_prev, dim=-1)).item()
        alpha_prev = alpha_q
        
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
            pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
        _, _, h = quantize_symmetric(h + pull, bits=8)
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['Momentum-Merge (Quantized-Naive)'] = (acc, jitter_accum / 10.0)

    # ------------------ MOMENTUM-MERGE (QA-MERGE: QCC + EF-SMOOTH) ------------------
    _, _, h = quantize_symmetric(h3_test.clone(), bits=8, per_sample=True)
    tau_mom = 0.005
    beta_mom = 0.60
    beta_ef = 0.8
    alpha = torch.full((h.shape[0], K), 0.25)
    jitter_accum = 0.0
    alpha_prev = None
    error_ef = torch.zeros(h.shape[0], K, device=h.device)
    error_act = torch.zeros_like(h) # Activation Error Feedback (AEF)
    for l in range(4, 15):
        # QCC
        q_h, _, _ = quantize_symmetric(h, bits=8, per_sample=True)
        # Compute cosine similarity in integer space (scale-invariant)
        norm_h = torch.norm(q_h, dim=-1, keepdim=True)
        norm_c = torch.norm(centroids_qcc, dim=-1, keepdim=True).t()
        sims = (q_h @ centroids_qcc.t()) / (norm_h @ norm_c + 1e-8)
        alpha_target = torch.softmax(sims / tau_mom, dim=-1)
        
        alpha = beta_mom * alpha + (1.0 - beta_mom) * alpha_target
        
        # EF-Smooth using discrete simplex projection
        alpha_corrected = alpha + beta_ef * error_ef
        _, alpha_q = discrete_simplex_projection(alpha_corrected, bits=4)
        error_ef = alpha_corrected - alpha_q
        
        if alpha_prev is not None:
            jitter_accum += torch.mean(torch.norm(alpha_q - alpha_prev, dim=-1)).item()
        alpha_prev = alpha_q
        
        pull = torch.zeros_like(h)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
            pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
            
        # Apply Activation Error Feedback (AEF)
        h_next_unquantized = h + pull + error_act
        _, _, h_next_quantized = quantize_symmetric(h_next_unquantized, bits=8, per_sample=True)
        error_act = h_next_unquantized - h_next_quantized
        h = h_next_quantized
        
    logits = torch.zeros(h.shape[0], K)
    for j in range(K):
        _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
        dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
        logits[:, j] = -dist + biases[j]
    preds = torch.argmax(logits, dim=-1)
    acc = torch.mean((preds == labels_test).float()).item()
    results['Momentum-Merge (QA-Merge)'] = (acc, jitter_accum / 10.0)

    # ------------------ CLASSICAL PARAMETRIC ROUTER TRAINING & EVALUATION ------------------
    # Choose regularization lambda based on N_cal
    l2_lambda = 10**-2 if N_cal <= 64 else 10**-4
    
    # A. Float32 Router Training
    router_f32 = ParametricRouter(D, K, init_zero=True)
    optimizer = optim.Adam(router_f32.parameters(), lr=1e-3, weight_decay=l2_lambda)
    
    # Train
    for epoch in range(100):
        optimizer.zero_grad()
        alpha = router_f32(h3_cal)
        
        # Forward propagate representational flow
        h_flow = h3_cal.clone()
        for l in range(4, 15):
            pull = torch.zeros_like(h_flow)
            for j in range(K):
                pull += alpha[:, j:j+1] * gamma_V * (v_primes[j].unsqueeze(0) - h_flow)
            h_flow = h_flow + pull
            
        # Compute logits at layer 14
        logits = torch.zeros(h3_cal.shape[0], K)
        for j in range(K):
            dist = torch.sum((h_flow - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
            logits[:, j] = -dist + biases[j]
            
        loss = nn.CrossEntropyLoss()(logits, labels_cal)
        loss.backward()
        optimizer.step()
        
    # Evaluate Float32 Router
    with torch.no_grad():
        alpha_test = router_f32(h3_test)
        h = h3_test.clone()
        for l in range(4, 15):
            pull = torch.zeros_like(h)
            for j in range(K):
                pull += alpha_test[:, j:j+1] * gamma_V * (v_primes[j].unsqueeze(0) - h)
            h = h + pull
        logits = torch.zeros(h.shape[0], K)
        for j in range(K):
            dist = torch.sum((h - v_primes[j].unsqueeze(0)) ** 2, dim=-1)
            logits[:, j] = -dist + biases[j]
        preds = torch.argmax(logits, dim=-1)
        acc_f32 = torch.mean((preds == labels_test).float()).item()
    results['Parametric Router'] = (acc_f32, 0.0)

    # B. Quantized-Naive Router (Directly evaluate Float32 Router in Quantized setting)
    with torch.no_grad():
        alpha_test = router_f32(h3_test, quantize_mode=True)
        alpha_q = ProjectToSimplex(quantize_alpha(alpha_test, bits=4))
        
        _, _, h = quantize_symmetric(h3_test.clone(), bits=8)
        for l in range(4, 15):
            pull = torch.zeros_like(h)
            for j in range(K):
                _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
                pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
            _, _, h = quantize_symmetric(h + pull, bits=8)
        logits = torch.zeros(h.shape[0], K)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8)
            dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
            logits[:, j] = -dist + biases[j]
        preds = torch.argmax(logits, dim=-1)
        acc_qn = torch.mean((preds == labels_test).float()).item()
    results['Parametric Router (Quantized-Naive)'] = (acc_qn, 0.0)

    # C. QA-Merge Parametric Router (Train with QAT STE + Evaluate with EF-Smooth)
    router_qcc = ParametricRouter(D, K, init_zero=True)
    optimizer = optim.Adam(router_qcc.parameters(), lr=1e-3, weight_decay=l2_lambda)
    
    # Train with QAT (STE)
    for epoch in range(100):
        optimizer.zero_grad()
        # forward pass with STE and per-sample activation quantization!
        alpha = router_qcc(h3_cal, quantize_mode=True, use_ste=True, per_sample=True)
        
        # Quantize alpha in the forward pass using STE with discrete simplex projection!
        alpha_q = discrete_simplex_ste(alpha, bits=4)
        
        # Strict layer-to-layer activation propagation in quantized state with AEF!
        _, _, h_flow = quantize_symmetric(h3_cal.clone(), bits=8, use_ste=True, per_sample=True)
        error_act = torch.zeros_like(h_flow)
        for l in range(4, 15):
            pull = torch.zeros_like(h_flow)
            for j in range(K):
                _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, use_ste=True, per_sample=False)
                pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h_flow)
            
            h_next_unquantized = h_flow + pull + error_act
            _, _, h_next_quantized = quantize_symmetric(h_next_unquantized, bits=8, use_ste=True, per_sample=True)
            error_act = h_next_unquantized - h_next_quantized
            h_flow = h_next_quantized
            
        logit_list = []
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, use_ste=True, per_sample=False)
            dist = torch.sum((h_flow - v_prime_q) ** 2, dim=-1)
            logit_list.append(-dist + biases[j])
        logits = torch.stack(logit_list, dim=-1)
            
        loss = nn.CrossEntropyLoss()(logits, labels_cal)
        loss.backward()
        optimizer.step()
        
    # Evaluate QA-Merge Router (with EF-Smooth on test data)
    with torch.no_grad():
        alpha_test = router_qcc(h3_test, quantize_mode=True, per_sample=True)
        
        _, _, h = quantize_symmetric(h3_test.clone(), bits=8, per_sample=True)
        beta_ef = 0.8
        error_ef = torch.zeros(h.shape[0], K, device=h.device)
        error_act = torch.zeros_like(h)
        
        for l in range(4, 15):
            alpha_corrected = alpha_test + beta_ef * error_ef
            _, alpha_q = discrete_simplex_projection(alpha_corrected, bits=4)
            error_ef = alpha_corrected - alpha_q
            
            pull = torch.zeros_like(h)
            for j in range(K):
                _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
                pull += alpha_q[:, j:j+1] * gamma_V * (v_prime_q - h)
            
            h_next_unquantized = h + pull + error_act
            _, _, h_next_quantized = quantize_symmetric(h_next_unquantized, bits=8, per_sample=True)
            error_act = h_next_unquantized - h_next_quantized
            h = h_next_quantized
            
        logits = torch.zeros(h.shape[0], K)
        for j in range(K):
            _, _, v_prime_q = quantize_symmetric(v_primes[j].unsqueeze(0), bits=8, per_sample=False)
            dist = torch.sum((h - v_prime_q) ** 2, dim=-1)
            logits[:, j] = -dist + biases[j]
        preds = torch.argmax(logits, dim=-1)
        acc_qcc = torch.mean((preds == labels_test).float()).item()
    results['Parametric Router (QA-Merge)'] = (acc_qcc, 0.0)

    # Print nicely
    for method, (acc, jit) in results.items():
        print(f"  {method:<35} | Accuracy: {acc*100:6.2f}% | Jitter: {jit:.5f}")
        
    return results

# Now run sweeps and collect all metrics!
rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
small_results = {rho: {} for rho in rhos}
large_results = {rho: {} for rho in rhos}

print("=== STARTING EXPERIMENTAL EVALUATION OF QA-MERGE AND BASELINES ===")

for rho in rhos:
    # Small-sample regime
    small_results[rho] = evaluate_all_methods(64, rho)
    # Large-sample regime
    large_results[rho] = evaluate_all_methods(4000, rho)

print("\n=== SWEEPS COMPLETE. GENERATING CHARTS... ===")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Plot 1: Accuracy Sweep across entanglement levels (rho) for both regimes
methods_to_plot = [
    'Expert Oracle', 'Uniform Merging', 'SABLE', 'SABLE (Quantized-Naive)', 'SABLE (QA-Merge)',
    'ChemMerge', 'ChemMerge (Quantized-Naive)', 'ChemMerge (QA-Merge)',
    'Parametric Router', 'Parametric Router (Quantized-Naive)', 'Parametric Router (QA-Merge)'
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, regime_title, results_dict in zip(axes, ["Small-Sample Regime (N_cal = 64)", "Large-Sample Regime (N_cal = 4000)"], [small_results, large_results]):
    for method in ['Expert Oracle', 'Uniform Merging', 'SABLE', 'SABLE (Quantized-Naive)', 'SABLE (QA-Merge)', 'ChemMerge (QA-Merge)', 'Parametric Router (QA-Merge)']:
        y_vals = [results_dict[rho][method][0] * 100 for rho in rhos]
        marker = 'o' if 'QA-Merge' in method else 'x'
        linestyle = '-' if 'QA-Merge' in method or method in ['Expert Oracle', 'Uniform Merging'] else '--'
        ax.plot(rhos, y_vals, label=method, marker=marker, linestyle=linestyle, linewidth=2)
    ax.set_title(regime_title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Representation Entanglement ($\\rho$)", fontsize=10)
    ax.set_ylabel("Joint Mean Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(45, 85)

axes[0].legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.savefig("results/fig1.png", dpi=300)
plt.savefig("comparison_plot.png", dpi=300)
plt.close()

# Plot 2: Jitter comparison under small-sample regime at rho = 0.3
jitter_methods = ['SABLE', 'SABLE (Quantized-Naive)', 'SABLE (QA-Merge)', 'ChemMerge', 'ChemMerge (Quantized-Naive)', 'ChemMerge (QA-Merge)', 'Momentum-Merge', 'Momentum-Merge (Quantized-Naive)', 'Momentum-Merge (QA-Merge)']
jitter_vals = [small_results[0.3][m][1] for m in jitter_methods]

plt.figure(figsize=(10, 5))
colors = ['skyblue', 'salmon', 'lightgreen'] * 3
plt.bar(jitter_methods, jitter_vals, color=colors, edgecolor='grey')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title("Trajectory Jitter Comparison under Small-Sample Regime ($\\rho = 0.3$)", fontsize=12, fontweight='bold')
plt.ylabel("Routing Trajectory Jitter", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig("results/fig3.png", dpi=300)
plt.close()

# Plot 3: Sample Complexity Sweep at rho = 0.3
sample_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4000]
complexity_results = {size: {} for size in sample_sizes}

for size in sample_sizes:
    complexity_results[size] = evaluate_all_methods(size, 0.3)
    
plt.figure(figsize=(8, 6))
for method in ['SABLE (QA-Merge)', 'ChemMerge (QA-Merge)', 'Parametric Router (QA-Merge)', 'SABLE', 'Uniform Merging']:
    y_vals = [complexity_results[size][method][0] * 100 for size in sample_sizes]
    plt.plot(sample_sizes, y_vals, label=method, marker='o' if 'QA-Merge' in method else 'x', linewidth=2)
plt.xscale('log')
plt.xlabel("Total Calibration Sample Size ($N_{\\text{cal}}$)", fontsize=10)
plt.ylabel("Joint Mean Accuracy (%)", fontsize=10)
plt.title("Sample Complexity Sweep (Representation Entanglement $\\rho = 0.3$)", fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/fig2.png", dpi=300)
plt.close()

print("\n=== CHARTS SAVED SUCCESSFULLY ===")

# Save raw data tables for markdown conversion
import json
with open("small_results.json", "w") as f:
    json.dump(small_results, f, indent=2)
with open("large_results.json", "w") as f:
    json.dump(large_results, f, indent=2)
with open("complexity_results.json", "w") as f:
    json.dump(complexity_results, f, indent=2)
    
print("=== EXPERIMENTS AND ARTIFACTS COMPLETED ===")
