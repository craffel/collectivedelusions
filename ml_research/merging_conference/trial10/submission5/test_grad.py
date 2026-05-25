import torch
import torch.nn as nn
from run_ttmm import SimpleCNN, merge_models, get_layer_group, compute_entropy

device = torch.device("cpu")
expert_0 = SimpleCNN().to(device)
expert_1 = SimpleCNN().to(device)

delta = torch.zeros(4, requires_grad=True)
w_global = 2.0

lambdas = {}
for name, param in expert_0.named_parameters():
    group = get_layer_group(name)
    lambdas[name] = torch.sigmoid(w_global + delta[group])

merged_model = SimpleCNN().to(device)
merge_models(expert_0, expert_1, merged_model, lambdas)

x = torch.randn(2, 1, 28, 28)
logits, _ = merged_model(x)
loss = compute_entropy(logits)

loss.backward()
print("delta.grad:", delta.grad)
