import torch
import torch.nn as nn
from torch.func import functional_call

# Test if functional_call propagates gradients to lambdas
l = torch.tensor([0.5, 0.5], requires_grad=True)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 2.0]))
        
    def forward(self, x):
        return x * self.w

model = SimpleModel()

# Base and expert weights
base_weights = {'w': torch.tensor([1.0, 2.0])}
expert_weights = {'w': torch.tensor([5.0, 10.0])}

# Blend weights differentiably
v = expert_weights['w'] - base_weights['w']
blended_w = base_weights['w'] + l[0] * v + l[1] * v

# Form state dict for functional call
params = {'w': blended_w}

x = torch.tensor([1.0, 1.0])
out = functional_call(model, params, x)

loss = out.sum()

grads = torch.autograd.grad(loss, [l])
print("Grads:", grads)
