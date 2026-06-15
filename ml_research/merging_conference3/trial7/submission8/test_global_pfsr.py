import torch
import torch.nn as nn
import numpy as np
from simulate import generate_expert_heads, generate_data, set_seed

D = 192
K = 4
d = D // K
C = 10
SIGMAS = [0.05, 0.05, 0.35, 1.25]

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
test_z, test_tasks, test_classes = generate_data(W, 250, [42])

# Evaluate PFSR (Local Block-specific)
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

# Evaluate PFSR (Global Vector-specific)
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

print(f"Global PFSR Accuracy: {correct_global / len(test_z) * 100.0:.2f}%")
