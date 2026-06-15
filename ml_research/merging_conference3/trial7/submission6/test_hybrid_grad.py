import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set seed
torch.manual_seed(42)

# Copy minimal environment setup from simulate_sr3
K = 4
L = 14
B_cal = 64
task_norms = torch.tensor([1.0, 4.0, 16.0, 64.0])

# Standalone expert ceilings
expert_ceilings = [95.0, 90.0, 82.0, 75.0]

# Entanglement
entanglement_matrix = torch.tensor([
    [0.2, 0.6, 0.2, 0.0],
    [0.0, 0.2, 0.6, 0.2],
    [0.1, 0.0, 0.2, 0.7],
    [0.6, 0.1, 0.0, 0.3]
])

def generate_unit_states(batch_size_per_task, noise_std=0.2):
    psi_list = []
    task_list = []
    for k in range(K):
        base = torch.zeros(batch_size_per_task, K)
        base[:, k] = 1.0
        noise = torch.randn(batch_size_per_task, K) * noise_std
        psi_t = base + noise
        psi_t = psi_t / (torch.norm(psi_t, dim=-1, keepdim=True) + 1e-8)
        psi_t = psi_t @ entanglement_matrix
        psi_t = psi_t / (torch.norm(psi_t, dim=-1, keepdim=True) + 1e-8)
        psi_list.append(psi_t)
        task_list.append(torch.full((batch_size_per_task,), k, dtype=torch.long))
    return torch.cat(psi_list, dim=0), torch.cat(task_list, dim=0)

psi_cal, task_cal = generate_unit_states(B_cal // K, noise_std=0.15)

class LinearRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(L, K, K))
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        logits = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)
        return torch.softmax(logits, dim=-1)

def compute_merged_distances(alpha):
    alpha_sq_weighted = (alpha ** 2) * task_norms.unsqueeze(0).unsqueeze(1)
    sum_all_weighted = torch.sum(alpha_sq_weighted, dim=-1)
    dist = sum_all_weighted.unsqueeze(-1) + task_norms.unsqueeze(0).unsqueeze(1) - 2 * alpha * task_norms.unsqueeze(0).unsqueeze(1)
    return torch.mean(dist, dim=1)

router = LinearRouter()
optimizer = optim.Adam(router.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

g = torch.zeros(K)
beta = 0.9
gamma = 1.0
lambda_reg = 5e-5

print("Starting training:")
for epoch in range(100):
    optimizer.zero_grad()
    alpha = router(psi_cal)
    distances = compute_merged_distances(alpha)
    logits = -distances / 4.0
    loss_ce = loss_fn(logits, task_cal)
    
    grad_W = torch.autograd.grad(loss_ce, router.W, retain_graph=True)[0].detach()
    grad_B = torch.autograd.grad(loss_ce, router.B, retain_graph=True)[0].detach()
    grad_norms = torch.sqrt(torch.sum(grad_W ** 2, dim=[0, 2]) + torch.sum(grad_B ** 2, dim=0))
    g = beta * g + (1.0 - beta) * grad_norms
    lambdas = lambda_reg * torch.exp(-gamma * g)
    
    loss_reg = 0.0 # simple test
    loss_total = loss_ce + loss_reg
    loss_total.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}:")
        print(f"  loss_ce: {loss_ce.item():.4f}")
        print(f"  grad_norms: {grad_norms.tolist()}")
        print(f"  g: {g.tolist()}")
        print(f"  lambdas: {lambdas.tolist()}")
