# Evaluate Robust CCNs under Subspace Drift
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

D = 192
num_samples = 1000

# True latent representations
Z_true = torch.randn(num_samples, D)
# EHPB Noise
E_noise = torch.randn(num_samples, D) * 0.30

# Naive pre-activations (with noise)
Y_naive = Z_true + E_noise

# Out-of-Distribution shift (drift) - adding systematic prototype shift/drift
Z_drift = Z_true + torch.randn(1, D) * 0.50
Y_drift = Z_drift + torch.randn(num_samples, D) * 0.30

# Define CCN Network
class CCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, D)
        )
    def forward(self, x):
        return self.net(x)

# 1. Train Standard CCN
ccn_std = CCN()
optimizer = optim.Adam(ccn_std.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = ccn_std(Y_naive)
    loss = criterion(outputs, Z_true)
    loss.backward()
    optimizer.step()

# 2. Train Robust CCN (with Coordinate-Robustness Data Augmentation)
# We inject extra noise scale variation and random coordinate coordinate drift during training
ccn_robust = CCN()
optimizer_rob = optim.Adam(ccn_robust.parameters(), lr=0.01)

for epoch in range(100):
    optimizer_rob.zero_grad()
    # Apply data augmentation
    noise_scale_aug = random.uniform(0.15, 0.45)
    E_noise_aug = torch.randn(num_samples, D) * noise_scale_aug
    drift_aug = torch.randn(num_samples, D) * 0.15
    Y_naive_aug = Z_true + E_noise_aug + drift_aug
    
    outputs = ccn_robust(Y_naive_aug)
    loss = criterion(outputs, Z_true)
    loss.backward()
    optimizer_rob.step()

# Evaluate Standard vs Robust CCN on In-Distribution (ID) and Out-of-Distribution (OOD)
with torch.no_grad():
    loss_id_naive = criterion(Y_naive, Z_true).item()
    loss_id_std = criterion(ccn_std(Y_naive), Z_true).item()
    loss_id_rob = criterion(ccn_robust(Y_naive), Z_true).item()
    
    loss_ood_naive = criterion(Y_drift, Z_drift).item()
    loss_ood_std = criterion(ccn_std(Y_drift), Z_drift).item()
    loss_ood_rob = criterion(ccn_robust(Y_drift), Z_drift).item()

print("--- CCN Evaluation on Reconstruction MSE ---")
print(f"ID Naive (Noisy) MSE:   {loss_id_naive:.6f}")
print(f"ID Standard CCN MSE:    {loss_id_std:.6f} ({loss_id_naive / loss_id_std:.2f}x reduction)")
print(f"ID Robust CCN MSE:      {loss_id_rob:.6f} ({loss_id_naive / loss_id_rob:.2f}x reduction)")
print()
print(f"OOD Naive (Drift) MSE:  {loss_ood_naive:.6f}")
print(f"OOD Standard CCN MSE:   {loss_ood_std:.6f}")
print(f"OOD Robust CCN MSE:     {loss_ood_rob:.6f} (Standard vs Robust improvement: {(loss_ood_std - loss_ood_rob) / loss_ood_std * 100:.1f}%)")
