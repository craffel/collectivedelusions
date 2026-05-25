import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from run_experiments import SimpleCNN, fuse_bn_stats, clone_model

device = torch.device("cpu")
exp0 = SimpleCNN(use_cosface=False).to(device)
exp1 = SimpleCNN(use_cosface=False).to(device)

merged = clone_model(exp0)

w_global = torch.tensor(0.0, requires_grad=True)
offsets = {name: torch.zeros(1, requires_grad=True) for name, _ in merged.named_parameters()}

# Reconstruct parameters dict
lam_global = torch.sigmoid(w_global)
state0 = exp0.state_dict()
state1 = exp1.state_dict()

params_dict = {}
for name, param in merged.named_parameters():
    lam_j = torch.sigmoid(w_global + offsets[name])
    params_dict[name] = (1.0 - lam_j) * state0[name] + lam_j * state1[name]

# Fuse BN stats in-place on merged
fuse_bn_stats(exp0, exp1, merged, lam_global.item())

# Forward pass
images = torch.randn(4, 1, 28, 28)
out = functional_call(merged, params_dict, (images,))
loss = torch.mean(out ** 2)
loss.backward()

print("w_global.grad =", w_global.grad)
print("Some offset grad =", list(offsets.values())[0].grad)
