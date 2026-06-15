import torch
import torch.nn as nn
import numpy as np

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Experimental Parameters
D = 192
K = 4
L = 14
num_classes = 10
block_size = D // K  # 48
N_calib_per_task = 16
N_test_per_task = 250
N_calib = N_calib_per_task * K

# Generate class prototypes for each task
class_prototypes = {}
for k in range(K):
    U, S, V = torch.svd(torch.randn(block_size, num_classes))
    prototypes = torch.zeros(num_classes, D)
    prototypes[:, k*block_size : (k+1)*block_size] = U.t()[:num_classes]
    class_prototypes[k] = prototypes

# Classification heads
W_head = {}
for k in range(K):
    head = torch.zeros(D, num_classes)
    head[k*block_size : (k+1)*block_size, :] = class_prototypes[k][:, k*block_size : (k+1)*block_size].t()
    W_head[k] = head

# Shared base layers
W_base = {}
for l in range(1, 14):
    W_base[l] = 0.05 * torch.eye(D)

# Noise levels
noise_levels = [0.01, 0.05, 0.28, 1.35]

# Generate calibration dataset
calib_x = []
calib_y = []
for k in range(K):
    for i in range(N_calib_per_task):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        calib_x.append(x)
        calib_y.append(k)
calib_x = torch.stack(calib_x)
calib_y = torch.tensor(calib_y)

# Generate test dataset
test_x = []
test_y = []
for k in range(K):
    for i in range(N_test_per_task):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        test_x.append(x)
        test_y.append(k)
test_x = torch.stack(test_x)
test_y = torch.tensor(test_y)

# Extract Layer 3 features
h_calib = calib_x.clone()
for l in range(1, 4):
    h_calib = h_calib + torch.relu(h_calib @ W_base[l])
z_calib = h_calib.clone()

h_test = test_x.clone()
for l in range(1, 4):
    h_test = h_test + torch.relu(h_test @ W_base[l])
z_test = h_test.clone()

# Compute block-norm features
calib_block_norms = torch.zeros(z_calib.shape[0], K)
test_block_norms = torch.zeros(z_test.shape[0], K)
for b in range(K):
    calib_block_norms[:, b] = z_calib[:, b*block_size : (b+1)*block_size].norm(dim=1)
    test_block_norms[:, b] = z_test[:, b*block_size : (b+1)*block_size].norm(dim=1)

# Strictly Temperature-only Router trained via PAC-Bayes bound minimization
class TempOnlyPACRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize log_tau to 0.0
        self.log_tau = nn.Parameter(torch.zeros(K))
    def forward(self, x):
        tau = torch.exp(self.log_tau)
        return x / tau

model_pac = TempOnlyPACRouter()
optimizer = torch.optim.Adam(model_pac.parameters(), lr=0.05)

for epoch in range(500):
    optimizer.zero_grad()
    logits = model_pac(calib_block_norms)
    q = torch.softmax(logits, dim=1)
    
    # Empirical Risk
    risk = 1.0 - q[range(N_calib), calib_y].mean()
    
    # Shannon Entropy
    entropy = - (q * torch.log(q + 1e-8)).sum(dim=1).mean()
    
    # KL complexity and PAC Bound (aligned with parameter-space L2 norm)
    kl = (model_pac.log_tau ** 2).sum() / 2.0
    bound = risk + torch.sqrt((kl + np.log(2.0 * np.sqrt(N_calib) / 0.05)) / (2.0 * N_calib))
    
    bound.backward()
    optimizer.step()

with torch.no_grad():
    test_logits = model_pac(test_block_norms)
    preds = torch.argmax(test_logits, dim=1)
    acc = (preds == test_y).float().mean().item() * 100.0
    print(f"Strictly Temperature-only Router (PAC-Bayes Trained) Accuracy: {acc:.2f}%")
    print("Optimized Temperatures (tau):", torch.exp(model_pac.log_tau).detach().numpy())
