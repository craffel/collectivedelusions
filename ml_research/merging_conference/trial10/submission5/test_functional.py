import torch
import torch.nn as nn
from run_ttmm import SimpleCNN, get_layer_group, compute_entropy
from torch.func import functional_call

device = torch.device("cpu")
expert_0 = SimpleCNN().to(device)
expert_1 = SimpleCNN().to(device)

delta = torch.zeros(4, requires_grad=True)
w_global = 2.0

# Compute differentiable merged state dict
state_dict_0 = expert_0.state_dict()
state_dict_1 = expert_1.state_dict()
merged_params_and_buffers = {}

for name, param in state_dict_0.items():
    if "running_mean" in name:
        var_name = name.replace("running_mean", "running_var")
        mu0 = state_dict_0[name]
        mu1 = state_dict_1[name]
        bn_key = name.split(".")[0] + ".weight"
        group = get_layer_group(bn_key)
        l = torch.sigmoid(w_global + delta[group]).detach() # detach here!
        merged_params_and_buffers[name] = l * mu0 + (1 - l) * mu1
    elif "running_var" in name:
        mean_name = name.replace("running_var", "running_mean")
        mu0 = state_dict_0[mean_name]
        mu1 = state_dict_1[mean_name]
        var0 = state_dict_0[name]
        var1 = state_dict_1[name]
        bn_key = name.split(".")[0] + ".weight"
        group = get_layer_group(bn_key)
        l = torch.sigmoid(w_global + delta[group]).detach() # detach here!
        merged_params_and_buffers[name] = l * var0 + (1 - l) * var1 + l * (1 - l) * (mu0 - mu1) ** 2
    elif "num_batches_tracked" in name:
        merged_params_and_buffers[name] = state_dict_0[name]
    else:
        group = get_layer_group(name)
        l = torch.sigmoid(w_global + delta[group]) # KEEP requires_grad=True for weights and biases!
        merged_params_and_buffers[name] = l * state_dict_0[name] + (1 - l) * state_dict_1[name]

# Standard template model for functional call
template_model = SimpleCNN().to(device)
template_model.eval()

x = torch.randn(2, 1, 28, 28)
# Run functional call
logits, _ = functional_call(template_model, merged_params_and_buffers, (x,))
loss = compute_entropy(logits)

loss.backward()
print("delta.grad:", delta.grad)
