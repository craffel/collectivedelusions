# Empirical Validation of Post-Hoc ReLU Bias Correction Strategies
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

D = 192
L = 14
B = 100  # Batch size
sigma_e = 0.30  # Noise standard deviation

# Standard normal PDF and CDF in PyTorch
def normal_pdf(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def compute_expected_bias(y_target, sigma):
    """
    Computes the analytic expected bias introduced by passing 
    y_target + e (where e ~ N(0, sigma^2)) through ReLU.
    Expected ReLU output: E[ReLU(y + e)] = y * CDF(y/sigma) + sigma * PDF(y/sigma)
    Bias: B = E[ReLU(y + e)] - ReLU(y)
    """
    ratio = y_target / sigma
    e_relu = y_target * normal_cdf(ratio) + sigma * normal_pdf(ratio)
    true_relu = torch.clamp(y_target, min=0.0)
    bias = e_relu - true_relu
    return bias

# Simulation of representation propagation through deep ReLU layers
class DeepReLUBackbone:
    def __init__(self, num_layers=5, dim=192):
        self.L = num_layers
        self.D = dim
        # Generate random frozen target weights (orthogonal-like)
        self.W_layers = [torch.eye(dim) + torch.randn(dim, dim) * 0.05 for _ in range(num_layers)]
        
    def propagate(self, X, noise_std=0.0, correction_type="none", scale_shift_params=None):
        """
        Propagates inputs through the layers. At each layer, we apply:
        y_noisy = y_clean + e, where e ~ N(0, noise_std^2)
        Then ReLU(y_noisy) with optional correction.
        """
        h = X.clone()
        for l in range(self.L):
            # Compute clean pre-activation for reference
            W = self.W_layers[l]
            y_target = torch.matmul(h, W.t())
            
            # Add zero-mean noise simulating EHPB unbinding cross-talk
            noise = torch.randn_like(y_target) * noise_std
            y_noisy = y_target + noise
            
            # Apply ReLU with correction
            if correction_type == "none":
                h = torch.clamp(y_noisy, min=0.0)
            elif correction_type == "subtraction":
                # Compute analytic expected bias and subtract it post-activation
                bias = compute_expected_bias(y_target, noise_std)
                h_rect = torch.clamp(y_noisy, min=0.0)
                h = h_rect - bias
            elif correction_type == "learnable" and scale_shift_params is not None:
                h_rect = torch.clamp(y_noisy, min=0.0)
                gamma, beta = scale_shift_params[l]
                h = h_rect * gamma + beta
                
        return h

# Initialize backbone
backbone = DeepReLUBackbone(num_layers=5, dim=D)

# Generate synthetic pre-activation input features
X_clean = torch.randn(B, D)

# Compute target (noise-free) final layer activations
H_target = X_clean.clone()
for l in range(5):
    H_target = torch.clamp(torch.matmul(H_target, backbone.W_layers[l].t()), min=0.0)

# Evaluate Baseline Noisy Propagation (No Correction)
print("Evaluating Baseline Noisy Propagation...")
H_noisy_baseline = backbone.propagate(X_clean, noise_std=sigma_e, correction_type="none")
mse_baseline = torch.mean((H_noisy_baseline - H_target)**2).item()
cos_baseline = torch.mean(torch.sum(H_noisy_baseline * H_target, dim=-1) / 
                          (torch.norm(H_noisy_baseline, p=2, dim=-1) * torch.norm(H_target, p=2, dim=-1) + 1e-8)).item()
print(f"No Correction    | Final MSE: {mse_baseline:.6f} | Cosine Similarity: {cos_baseline:.4f}")

# Evaluate Running Noise Subtraction Correction
print("\nEvaluating Layer-wise Running Noise Subtraction...")
H_noisy_sub = backbone.propagate(X_clean, noise_std=sigma_e, correction_type="subtraction")
mse_sub = torch.mean((H_noisy_sub - H_target)**2).item()
cos_sub = torch.mean(torch.sum(H_noisy_sub * H_target, dim=-1) / 
                    (torch.norm(H_noisy_sub, p=2, dim=-1) * torch.norm(H_target, p=2, dim=-1) + 1e-8)).item()
print(f"Noise Subtraction | Final MSE: {mse_sub:.6f} | Cosine Similarity: {cos_sub:.4f}")

# Evaluate Learnable Scaling and Shift Correction
print("\nTraining Learnable Scaling and Shift Correction on calibration set...")
# Generate small calibration set (16 samples)
X_cal = torch.randn(16, D)
H_cal_target = X_cal.clone()
for l in range(5):
    H_cal_target = torch.clamp(torch.matmul(H_cal_target, backbone.W_layers[l].t()), min=0.0)

# Initialize scale and shift parameters per layer
scale_shift_params = {}
for l in range(5):
    gamma = nn.Parameter(torch.ones(D))
    beta = nn.Parameter(torch.zeros(D))
    scale_shift_params[l] = (gamma, beta)

# Fast optimization on calibration set (30 steps)
optimizer = optim.Adam([p for l in scale_shift_params for p in scale_shift_params[l]], lr=0.1)
criterion = nn.MSELoss()

for step in range(50):
    optimizer.zero_grad()
    # Propagate calibration set
    h = X_cal.clone()
    for l in range(5):
        y_target = torch.matmul(h, backbone.W_layers[l].t())
        # Simulate noisy propagation during calibration training
        noise = torch.randn_like(y_target) * sigma_e
        y_noisy = y_target + noise
        h_rect = torch.clamp(y_noisy, min=0.0)
        gamma, beta = scale_shift_params[l]
        h = h_rect * gamma + beta
        
    loss = criterion(h, H_cal_target)
    loss.backward()
    optimizer.step()

# Evaluate Learnable correction on test set
with torch.no_grad():
    H_noisy_learn = backbone.propagate(X_clean, noise_std=sigma_e, correction_type="learnable", scale_shift_params=scale_shift_params)
    mse_learn = torch.mean((H_noisy_learn - H_target)**2).item()
    cos_learn = torch.mean(torch.sum(H_noisy_learn * H_target, dim=-1) / 
                          (torch.norm(H_noisy_learn, p=2, dim=-1) * torch.norm(H_target, p=2, dim=-1) + 1e-8)).item()
print(f"Learnable Scale/Shift | Final MSE: {mse_learn:.6f} | Cosine Similarity: {cos_learn:.4f}")
