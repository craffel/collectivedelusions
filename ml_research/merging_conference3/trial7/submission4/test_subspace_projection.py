import torch
from test_subspace_noise import RepresentationSandbox, PFSRRouter

sandbox = RepresentationSandbox(42)
X_test, Y_test, task_test = sandbox.generate_split(250)

# Evaluate expert ceilings
import torch.nn as nn
experts = []
for k in range(sandbox.K):
    expert = nn.Linear(sandbox.D, sandbox.C, bias=True)
    with torch.no_grad():
        for c in range(sandbox.C):
            expert.weight[c] = sandbox.prototypes[k, c]
            expert.bias[c] = 0.0
    experts.append(expert)

pfsr = PFSRRouter(experts)
tilde_X = X_test / torch.norm(X_test, dim=-1, keepdim=True)
u = torch.matmul(tilde_X, pfsr.v_bar.t())

print("First sample of task 0 tilde_z:", tilde_X[0])
print("First sample of task 0 projection u:", u[0])
print("Is tilde_X[0] zero outside 0-48?", torch.all(tilde_X[0][48:] == 0.0).item())
print("Is v_bar[1] zero outside 48-96?", torch.all(pfsr.v_bar[1][:48] == 0.0).item() and torch.all(pfsr.v_bar[1][96:] == 0.0).item())
print("Dot product of tilde_X[0] and v_bar[1]:", torch.dot(tilde_X[0], pfsr.v_bar[1]).item())
