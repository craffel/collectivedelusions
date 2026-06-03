import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create dummy model
model = torchvision.models.resnet18()
model.fc = nn.Linear(512, 10)

class CalibrationHookManager:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_names = []
        self.layer_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.layer_names.append(name)
                self.layer_dict[name] = module
        self.clear_hooks()
        
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.stats = {name: {"expert_mags": [], "expert_means": [], "expert_stds": []} for name in self.layer_names}
        
    def register_gather_hooks(self):
        self.clear_hooks()
        def make_hook(name):
            def hook(module, input, output):
                mag = torch.abs(torch.fft.fft2(output, dim=(-2, -1)))
                self.stats[name]["expert_mags"].append(mag.detach().cpu())
                print(f"Hook fired for {name}! Total captured: {len(self.stats[name]['expert_mags'])}")
            return hook
        for name in self.layer_names:
            h = self.layer_dict[name].register_forward_hook(make_hook(name))
            self.hooks.append(h)

mgr = CalibrationHookManager(model)
mgr.register_gather_hooks()

# Run a forward pass
inputs = torch.randn(4, 3, 32, 32)
model.eval()
with torch.no_grad():
    _ = model(inputs)

print("Layer names:", mgr.layer_names[:3])
print("Stats for bn1 captured count:", len(mgr.stats["bn1"]["expert_mags"]))
