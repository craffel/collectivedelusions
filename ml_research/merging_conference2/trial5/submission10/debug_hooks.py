import torch
import torchvision
import torch.nn as nn

model = torchvision.models.resnet18()
print("Model modules:")
for name, m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        print("Found:", name)

# Let's register a forward hook on bn1 and run a forward pass
activated = []
def hook_fn(module, input, output):
    print("Hook activated for output shape:", output.shape)
    activated.append(output)

h = model.bn1.register_forward_hook(hook_fn)
x = torch.randn(2, 3, 32, 32)
model.eval()
_ = model(x)

print("Number of outputs saved in hook:", len(activated))
h.remove()
