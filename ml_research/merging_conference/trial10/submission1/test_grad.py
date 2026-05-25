import torch
import torch.nn as nn
from torch.func import functional_call

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, 2))
    def forward(self, x):
        return x @ self.w

model = SimpleModel()
model0_state = {"w": torch.ones(2, 2) * 2.0}
model1_state = {"w": torch.ones(2, 2) * 3.0}

global_w = torch.tensor(0.0, requires_grad=True)

# Parameter dict with gradients
lam = torch.sigmoid(global_w)
params = {
    "w": (1.0 - lam) * model0_state["w"] + lam * model1_state["w"]
}

out = functional_call(model, params, (torch.ones(1, 2),))
loss = torch.sum(out)
loss.backward()

print("Using torch.func.functional_call: global_w.grad =", global_w.grad)
