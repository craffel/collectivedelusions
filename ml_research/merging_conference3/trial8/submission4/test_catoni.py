import torch
import torch.nn as nn
import numpy as np

# Let's test the Catoni PAC-Bayes bound optimization
N_opt = 32
K = 4
opt_calib_y = torch.tensor([0]*8 + [1]*8 + [2]*8 + [3]*8)
opt_block_norms = torch.randn(N_opt, K) + 2.0  # mock features
opt_block_norms[0:8, 0] += 5.0
opt_block_norms[8:16, 1] += 5.0
opt_block_norms[16:24, 2] += 5.0
opt_block_norms[24:32, 3] += 5.0

class PACRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(K))
    def forward(self, x):
        tau = torch.exp(self.log_tau)
        return x / tau

pac_router_block = PACRouter()
pac_opt_block = torch.optim.Adam(pac_router_block.parameters(), lr=0.05)
criterion_pac = nn.CrossEntropyLoss()
w_0 = np.log(0.05)
sigma_0_sq = 5.0

beta = 0.5  # Catoni's parameter
delta = 0.05

for epoch in range(100):
    pac_opt_block.zero_grad()
    logits = pac_router_block(opt_block_norms)
    risk = criterion_pac(logits, opt_calib_y)
    kl = ((pac_router_block.log_tau - w_0) ** 2).sum() / (2.0 * sigma_0_sq)
    
    # Catoni's Bound formula
    # B = (1.0 / (1.0 - exp(-beta))) * (1.0 - exp(-beta * risk - (kl + ln(1/delta)) / N))
    bound = (1.0 / (1.0 - np.exp(-beta))) * (1.0 - torch.exp(-beta * risk - (kl + np.log(1.0 / delta)) / N_opt))
    
    bound.backward()
    pac_opt_block.step()

print(f"Catoni optimization successful!")
print(f"Risk: {risk.item():.4f} | KL: {kl.item():.4f} | Bound: {bound.item():.4f}")
print(f"Learned temperatures: {torch.exp(pac_router_block.log_tau).detach().numpy()}")
