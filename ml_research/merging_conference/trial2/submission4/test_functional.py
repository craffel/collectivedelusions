import torch
import torch.nn as nn
from torch.func import functional_call

# Define a simple linear model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
x = torch.randn(3, 5)

# Extract original state
params = {k: v for k, v in model.named_parameters()}
buffers = {k: v for k, v in model.named_buffers()}

# Define a coefficient that requires grad
coeff = torch.tensor(0.5, requires_grad=True)

# Create a modified state dict where we scale the weight dynamically
modified_params = {}
for k, v in params.items():
    if k == 'fc.weight':
        modified_params[k] = v * coeff
    else:
        modified_params[k] = v

all_state = {**modified_params, **buffers}

# Run functional call
outputs = functional_call(model, all_state, (x,))
loss = outputs.sum()
loss.backward()

print("Gradient w.r.t coeff:", coeff.grad)
