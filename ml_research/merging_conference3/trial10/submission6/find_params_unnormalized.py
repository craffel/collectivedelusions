import torch
import numpy as np

D = 192
K = 4
L = 14
rho = 0.5
overlap_v = 12
gamma_V = 0.05

# Setup signatures (unnormalized)
v_k = torch.zeros((K, D))
S = D // K
for k in range(K):
    start_idx = k * S - k * overlap_v
    end_idx = start_idx + S
    v_k[k, start_idx:end_idx] = 1.0

# Covariance
idx = torch.arange(D)
Sigma = rho ** torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
eigenvalues = torch.clamp(eigenvalues, min=1e-6)
Sigma_half = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
v_prime = (Sigma_half @ v_k.T).T

sigmas_base = torch.tensor([0.05, 0.15, 0.40, 1.20])

def evaluate_oracle_and_uniform(sigmas_scale, kappa_scale):
    sigmas = sigmas_base * sigmas_scale
    scale_factor = (1 - gamma_V)**11
    
    oracle_accs = []
    uniform_accs = []
    
    torch.manual_seed(42)
    for y in range(K):
        epsilon = torch.randn(1000, D) * sigmas[y]
        
        # Oracle
        h_oracle_diff = scale_factor * epsilon
        dist_sq_oracle = torch.sum(h_oracle_diff**2, dim=1)
        acc_oracle = torch.exp(-kappa_scale * dist_sq_oracle)
        oracle_accs.append(torch.mean(acc_oracle).item())
        
        # Uniform
        # final representation: h_14 = scale_factor * (v_prime_y + epsilon) + (1-scale_factor) * bar_v
        # h_14 - v_prime_y = - (1-scale_factor) * (v_prime_y - bar_v) + scale_factor * epsilon
        bar_v = torch.mean(v_prime, dim=0)
        mismatch = (1.0 - scale_factor) * (v_prime[y] - bar_v)
        h_uniform_diff = -mismatch.unsqueeze(0) + scale_factor * epsilon
        dist_sq_uniform = torch.sum(h_uniform_diff**2, dim=1)
        acc_uniform = torch.exp(-kappa_scale * dist_sq_uniform)
        uniform_accs.append(torch.mean(acc_uniform).item())
        
    return np.mean(oracle_accs), np.mean(uniform_accs)

# Grid search for the parameters
best_sigmas_scale = 1.0
best_kappa_scale = 1.0
min_diff = 100.0

for sigmas_scale in np.linspace(0.01, 0.2, 30):
    for kappa_scale in np.linspace(0.001, 0.1, 50):
        oracle_val, uniform_val = evaluate_oracle_and_uniform(sigmas_scale, kappa_scale)
        diff = abs(oracle_val - 0.9504) + abs(uniform_val - 0.3268)
        if diff < min_diff:
            min_diff = diff
            best_sigmas_scale = sigmas_scale
            best_kappa_scale = kappa_scale
            print(f"sigmas_scale={sigmas_scale:.4f}, kappa_scale={kappa_scale:.4f} => Oracle={oracle_val*100:.2f}%, Uniform={uniform_val*100:.2f}% (diff={diff:.4f})")

print(f"\nBest parameters found: sigmas_scale={best_sigmas_scale:.4f}, kappa_scale={best_kappa_scale:.4f}")
