import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models

# Set seed and disable cuDNN to avoid issues on this cluster
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

def get_modified_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class MergedModel(nn.Module):
    def __init__(self, base_model, experts, device):
        super().__init__()
        self.base_model = base_model
        self.experts = experts # list of 3 experts
        self.device = device
        
        # Logits of coefficients: shape (6, 3)
        self.logits = nn.Parameter(torch.zeros(6, 3, device=device))
        self.group_names = ["early", "layer1", "layer2", "layer3", "layer4", "fc"]
        
        # Pre-cache expert parameters to avoid state_dict lookups in forward pass
        self.expert_params = {}
        for name, param in self.base_model.named_parameters():
            self.expert_params[name] = [
                self.experts[0].state_dict()[name].to(device),
                self.experts[1].state_dict()[name].to(device),
                self.experts[2].state_dict()[name].to(device)
            ]
        
    def get_coefficients(self):
        return torch.softmax(self.logits, dim=1)
        
    def get_group_idx(self, param_name):
        if "conv1" in param_name or "bn1" in param_name:
            return 0
        elif "layer1" in param_name:
            return 1
        elif "layer2" in param_name:
            return 2
        elif "layer3" in param_name:
            return 3
        elif "layer4" in param_name:
            return 4
        elif "fc" in param_name:
            return 5
        return 0
        
    def forward(self, x):
        coeffs = self.get_coefficients()
        
        merged_params = {}
        # Differentiable weight merging using cached parameters
        for name in self.expert_params:
            g_idx = self.get_group_idx(name)
            w1, w2, w3 = self.expert_params[name]
            merged = coeffs[g_idx, 0] * w1 + coeffs[g_idx, 1] * w2 + coeffs[g_idx, 2] * w3
            merged_params[name] = merged
            
        for name, buf in self.base_model.named_buffers():
            merged_params[name] = buf
            
        return torch.func.functional_call(self.base_model, merged_params, x)

def profile_tent(device, num_steps=50):
    model = get_modified_resnet18(num_classes=10).to(device)
    
    # Configure TENT: adapt only BN affine parameters
    model.eval() # TENT configures model to eval mode except BN layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = True
            m.bias.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = False
                
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.01)
    
    # Dummy batch of size 32
    x = torch.randn(32, 1, 28, 28, device=device)
    
    # Warmup
    for _ in range(5):
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
    start_time = time.time()
    
    for _ in range(num_steps):
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB
    else:
        peak_mem = 0.0
        
    elapsed = (time.time() - start_time) / num_steps * 1000 # ms per step
    return elapsed, peak_mem

def profile_rgs_cop(device, num_steps=50):
    base_model = get_modified_resnet18(num_classes=10).to(device)
    experts = [get_modified_resnet18(num_classes=10).to(device) for _ in range(3)]
    for exp in experts:
        exp.eval()
        for p in exp.parameters():
            p.requires_grad = False
            
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
        
    merged_model = MergedModel(base_model, experts, device)
    merged_model.logits.requires_grad = True
    optimizer = optim.SGD([merged_model.logits], lr=0.1)
    
    # Dummy batch of size 32
    x = torch.randn(32, 1, 28, 28, device=device)
    
    # Warmup
    for _ in range(5):
        outputs = merged_model(x)
        probs = torch.softmax(outputs, dim=1)
        loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
    start_time = time.time()
    
    for _ in range(num_steps):
        outputs = merged_model(x)
        probs = torch.softmax(outputs, dim=1)
        loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB
    else:
        peak_mem = 0.0
        
    elapsed = (time.time() - start_time) / num_steps * 1000 # ms per step
    return elapsed, peak_mem

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    
    tent_time, tent_mem = profile_tent(device)
    print(f"TENT (BatchNorm TTA) - Step Latency: {tent_time:.2f} ms | Peak Memory: {tent_mem:.2f} MB")
    
    rgs_time, rgs_mem = profile_rgs_cop(device)
    print(f"RGS-COP (Model Merging) - Step Latency: {rgs_time:.2f} ms | Peak Memory: {rgs_mem:.2f} MB")
    
    results = {
        "tent": {
            "latency_ms": tent_time,
            "peak_memory_mb": tent_mem
        },
        "rgs_cop": {
            "latency_ms": rgs_time,
            "peak_memory_mb": rgs_mem
        }
    }
    
    with open("efficiency_profile.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Profiling complete. Saved results to efficiency_profile.json.")

if __name__ == "__main__":
    main()
