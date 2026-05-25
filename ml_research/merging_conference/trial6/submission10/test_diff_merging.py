import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)
        self.fc = nn.Linear(4, 2)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=[2, 3])
        return self.fc(x)

# Create base, and two experts
base = ToyModel()
expert1 = ToyModel()
expert2 = ToyModel()

# Let's define the merging coefficients for each parameter tensor
# Suppose we have coefficients for each parameter tensor:
# conv.weight, conv.bias, fc.weight, fc.bias
param_names = [name for name, _ in base.named_parameters()]
print("Parameter names:", param_names)

# Initialize coefficients as a simplex (uniform)
lambdas = {}
for name in param_names:
    # 2 experts
    lambdas[name] = torch.tensor([0.5, 0.5], requires_grad=True)

# Define a function to reconstruct the merged model
def update_merged_model(merged_model, experts, lambdas):
    # For each parameter tensor, compute the weighted average
    for name in param_names:
        # Get reference to the parameter in experts
        expert_params = [dict(expert.named_parameters())[name] for expert in experts]
        # Compute merged weight: sum_k lambda_k * expert_k_weight
        coeff = lambdas[name]
        merged_weight = sum(coeff[k] * expert_params[k] for k in range(len(experts)))
        
        # We want to replace the parameter in merged_model with merged_weight while keeping grad history.
        # In PyTorch, we can do this by deleting the original parameter attribute and setting it as a regular tensor.
        # When PyTorch modules use this attribute in forward passes, they will use our merged_weight tensor
        # and backpropagate the gradients correctly!
        
        # Navigate to the submodule
        parts = name.split('.')
        submodule = merged_model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        
        # Delete parameter if it exists
        attr_name = parts[-1]
        if hasattr(submodule, attr_name):
            delattr(submodule, attr_name)
        
        # Set the attribute to our merged_weight tensor
        setattr(submodule, attr_name, merged_weight)

# Test forward and backward pass
merged = ToyModel()
experts = [expert1, expert2]

# Update the merged model weights
update_merged_model(merged, experts, lambdas)

# Create a random input
x = torch.randn(2, 1, 10, 10)
output = merged(x)

# Define some loss (e.g., sum of output)
loss = output.sum()

# Backward pass
loss.backward()

# Check if lambdas got gradients!
print("\nGradients:")
for name in param_names:
    print(f"{name} lambda grad: {lambdas[name].grad}")
