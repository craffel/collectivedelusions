import torch
import torch.nn as nn
import torch.nn.functional as F
from run_experiments import RepresentationSandbox

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)

X_train, Y_train, task_train = sandbox.generate_split(1000)
mask = (task_train == 0)
X_0 = X_train[mask]
Y_0 = Y_train[mask]

expert = nn.Linear(sandbox.D, sandbox.C, bias=True)
with torch.no_grad():
    for c in range(sandbox.C):
        expert.weight[c] = sandbox.prototypes[0, c] + torch.randn(sandbox.D) * 0.01
        expert.bias[c] = 0.0

# Train expert with much stronger regularization and lower learning rate
optimizer = torch.optim.AdamW(expert.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
for epoch in range(40):
    logits = expert(X_0)
    loss = F.cross_entropy(logits, Y_0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

trained_weight = expert.weight.data
print("Trained weight norm:", torch.norm(trained_weight).item())
print("Trained weight active energy in 0-48:", torch.sum(trained_weight[:, :48]**2).item() / torch.sum(trained_weight**2).item())
