import torch
import torch.nn as nn

# Test if copy_ propagates gradients to lambdas
l = torch.tensor([0.5, 0.5], requires_grad=True)
p = nn.Parameter(torch.tensor([1.0, 2.0]))

loss = p[0]**2 + p[1]**2

# Let's do copy_
with torch.no_grad():
    p.copy_(l[0] * 5.0 + l[1] * 10.0)

# Compute loss
loss = (p - 1.0).sum()

try:
    grads = torch.autograd.grad(loss, [l])
    print("Grads:", grads)
except Exception as e:
    print("Error:", e)
