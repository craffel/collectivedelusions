import torch
import torch.nn as nn

# Define a simple linear layer
linear0 = nn.Linear(5, 2, bias=False)
linear1 = nn.Linear(5, 2, bias=False)
linear2 = nn.Linear(5, 2, bias=False)

# Experts list
experts = [linear0, linear1, linear2]

# Target model
target_model = nn.Linear(5, 2, bias=False)

# Coefficients
lambdas = torch.tensor([0.2, 0.5, 0.3], requires_grad=True)

# 1. Native PyTorch Autograd
# We construct the merged weight explicitly as a function of lambdas
merged_weight = lambdas[0] * linear0.weight + lambdas[1] * linear1.weight + lambdas[2] * linear2.weight

inputs = torch.randn(4, 5)
outputs = torch.matmul(inputs, merged_weight.t())
loss = outputs.sum()

# Backward
loss.backward()
native_grad = lambdas.grad.clone()
print("Native Autograd lambdas.grad:", native_grad)

# 2. Manual Autograd via Chain Rule
# Reset
lambdas.grad = None
# Merge weights via copying
target_model.weight.data.copy_(merged_weight.data)
# Forward on target model
target_model.zero_grad()
out_target = target_model(inputs)
loss_target = out_target.sum()
loss_target.backward()

# Compute gradients with respect to lambdas manually
manual_grad = torch.zeros(3)
for k, expert in enumerate(experts):
    manual_grad[k] = torch.sum(target_model.weight.grad * expert.weight)

print("Manual Chain Rule lambdas.grad:", manual_grad)
print("Difference:", torch.abs(native_grad - manual_grad).max().item())
