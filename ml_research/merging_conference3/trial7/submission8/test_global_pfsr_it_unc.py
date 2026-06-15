import torch
import torch.nn as nn
import numpy as np
from simulate import generate_expert_heads, set_seed

D = 192
K = 4
d = D // K
C = 10
SIGMAS = [0.05, 0.05, 0.35, 1.25]

def generate_data_unnormalized_inactive(W, num_samples_per_task):
    all_z = []
    all_tasks = []
    all_classes = []
    for t in range(K):
        for _ in range(num_samples_per_task):
            c_b = np.random.randint(0, C)
            
            # Active task block feature
            eps = torch.randn(d)
            z_t = W[t][c_b] + SIGMAS[t] * eps
            z_t = z_t / torch.norm(z_t) # UNC feature normalization for active task
            
            z_blocks = []
            for k in range(K):
                if k == t:
                    z_blocks.append(z_t)
                else:
                    # Inactive task block features: unnormalized Gaussian noise
                    eps_k = torch.randn(d) * 0.5 # with some variance
                    z_blocks.append(eps_k)
            
            z_b = torch.cat(z_blocks)
            all_z.append(z_b.unsqueeze(0))
            all_tasks.append(t)
            all_classes.append(c_b)
    return torch.cat(all_z), torch.tensor(all_tasks), torch.tensor(all_classes)

def compute_global_pfsr_coefficients(z_b, W, tau=0.001):
    B_sz = z_b.shape[0]
    u = torch.zeros(B_sz, K)
    for b in range(B_sz):
        z_norm = torch.norm(z_b[b])
        for k in range(K):
            z_kb = z_b[b, k*d : (k+1)*d]
            dot_products = torch.matmul(W[k], z_kb)
            cos_sims = dot_products / (1.0 * z_norm)
            u[b, k] = torch.max(cos_sims)
    calibration_factor = np.sqrt(2 * np.log(10) / 48)
    u_prime = u / calibration_factor
    alpha_pfsr = torch.softmax(u_prime / tau, dim=1)
    return alpha_pfsr, u_prime

def compute_global_pfsr_with_it_unc(z_b, W, tau=0.001):
    B_sz = z_b.shape[0]
    u = torch.zeros(B_sz, K)
    for b in range(B_sz):
        # IT-UNC block-wise normalization
        z_b_normalized_blocks = []
        for k in range(K):
            z_kb = z_b[b, k*d : (k+1)*d]
            z_kb_norm = z_kb / torch.norm(z_kb)
            z_b_normalized_blocks.append(z_kb_norm)
        z_b_hat = torch.cat(z_b_normalized_blocks)
        
        z_norm = torch.norm(z_b_hat) # Should be exactly sqrt(K) = 2.0
        for k in range(K):
            z_kb_hat = z_b_hat[k*d : (k+1)*d]
            dot_products = torch.matmul(W[k], z_kb_hat)
            cos_sims = dot_products / (1.0 * z_norm)
            u[b, k] = torch.max(cos_sims)
            
    calibration_factor = np.sqrt(2 * np.log(10) / 48)
    u_prime = u / calibration_factor
    alpha_pfsr = torch.softmax(u_prime / tau, dim=1)
    return alpha_pfsr, u_prime

def compute_local_pfsr_coefficients(z_b, W, tau=0.001):
    B_sz = z_b.shape[0]
    u = torch.zeros(B_sz, K)
    for b in range(B_sz):
        for k in range(K):
            z_kb = z_b[b, k*d : (k+1)*d]
            z_kb_norm = z_kb / torch.norm(z_kb)
            cos_sims = torch.matmul(W[k], z_kb_norm)
            u[b, k] = torch.max(cos_sims)
    calibration_factor = np.sqrt(2 * np.log(10) / 48)
    u_prime = u / calibration_factor
    alpha_pfsr = torch.softmax(u_prime / tau, dim=1)
    return alpha_pfsr, u_prime

set_seed(42)
W = generate_expert_heads()
test_z, test_tasks, test_classes = generate_data_unnormalized_inactive(W, 250)

# Evaluate Local PFSR
correct_local = 0
for b in range(len(test_z)):
    alpha, _ = compute_local_pfsr_coefficients(test_z[b:b+1], W)
    logits_c = torch.zeros(C)
    for k in range(K):
        z_kb = test_z[b, k*d : (k+1)*d]
        z_kb_norm = z_kb / torch.norm(z_kb)
        expert_logits = torch.matmul(W[k], z_kb_norm)
        logits_c += alpha[0, k] * expert_logits
    if torch.argmax(logits_c).item() == test_classes[b].item():
        correct_local += 1

print(f"Local PFSR Accuracy: {correct_local / len(test_z) * 100.0:.2f}%")

# Evaluate Standard Global PFSR
correct_global = 0
for b in range(len(test_z)):
    alpha, _ = compute_global_pfsr_coefficients(test_z[b:b+1], W)
    logits_c = torch.zeros(C)
    for k in range(K):
        z_kb = test_z[b, k*d : (k+1)*d]
        z_kb_norm = z_kb / torch.norm(z_kb)
        expert_logits = torch.matmul(W[k], z_kb_norm)
        logits_c += alpha[0, k] * expert_logits
    if torch.argmax(logits_c).item() == test_classes[b].item():
        correct_global += 1

print(f"Standard Global PFSR Accuracy: {correct_global / len(test_z) * 100.0:.2f}%")

# Evaluate Global PFSR with IT-UNC
correct_global_it_unc = 0
for b in range(len(test_z)):
    alpha, _ = compute_global_pfsr_with_it_unc(test_z[b:b+1], W)
    logits_c = torch.zeros(C)
    for k in range(K):
        z_kb = test_z[b, k*d : (k+1)*d]
        z_kb_norm = z_kb / torch.norm(z_kb)
        expert_logits = torch.matmul(W[k], z_kb_norm)
        logits_c += alpha[0, k] * expert_logits
    if torch.argmax(logits_c).item() == test_classes[b].item():
        correct_global_it_unc += 1

print(f"Global PFSR with IT-UNC Accuracy: {correct_global_it_unc / len(test_z) * 100.0:.2f}%")
