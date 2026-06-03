import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # Bypass cuDNN issues on this cluster

def get_datasets():
    # Transforms
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)
    
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color)
    
    return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (cifar_train, cifar_test)

def load_expert(name, device):
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(model.fc.in_features, 10)
    )
    model.load_state_dict(torch.load(f"expert_{name.lower()}.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

def load_base_model(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(model.fc.in_features, 10)
    )
    model.to(device)
    model.eval()
    return model

def merge_experts_wa(experts):
    merged = resnet18(weights=None)
    merged.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(merged.fc.in_features, 10)
    )
    
    merged_state_dict = {}
    keys = experts[0].state_dict().keys()
    for key in keys:
        tensors = [expert.state_dict()[key] for expert in experts]
        # Check if the tensor is integer/boolean and cast if necessary
        if tensors[0].is_floating_point():
            merged_state_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            # Just take the first expert's value (typically num_batches_tracked)
            merged_state_dict[key] = tensors[0]
            
    merged.load_state_dict(merged_state_dict)
    merged.eval()
    return merged

def merge_experts_ta(experts, base_model, lam=0.3):
    merged = resnet18(weights=None)
    merged.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(merged.fc.in_features, 10)
    )
    
    merged_state_dict = {}
    base_state = base_model.state_dict()
    keys = base_state.keys()
    for key in keys:
        tensors = [expert.state_dict()[key] for expert in experts]
        if 'fc' in key or not tensors[0].is_floating_point():
            # For classification head or integers, just average
            if tensors[0].is_floating_point():
                merged_state_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                merged_state_dict[key] = tensors[0]
        else:
            # Compute task vectors and apply scaling lam
            task_vectors = [t - base_state[key] for t in tensors]
            merged_state_dict[key] = base_state[key] + lam * torch.stack(task_vectors, dim=0).sum(dim=0)
            
    merged.load_state_dict(merged_state_dict)
    merged.eval()
    return merged

# Forward Hook Manager for Representation Alignment
class ActivationHookManager:
    def __init__(self, model, target_layers, mode="none"):
        self.model = model
        self.target_layers = target_layers # dict or list
        self.mode = mode
        self.captured = {}
        self.calibration_params = {}
        self.hooks = []
        self.current_task = 0 # Default task index for Task-Conditional NRA
        self._register_hooks()
        
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(self._make_hook(name)))
                
    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if self.mode == "capture":
                self.captured[name] = output.detach().clone()
            elif self.mode == "nra":
                # Neural Resonance Alignment online hook
                # output is [B, C, H, W]
                # Apply complex scaling in Fourier domain
                if name not in self.calibration_params:
                    return output
                g = self.calibration_params[name] # [C, H, W] complex spectrum scaling
                V_fft = torch.fft.fft2(output, dim=(-2, -1))
                V_cal_fft = V_fft * g.unsqueeze(0)
                V_cal = torch.fft.ifft2(V_cal_fft, dim=(-2, -1)).real
                return V_cal
            elif self.mode == "tcnra":
                # Task-Conditional Neural Resonance Alignment online hook
                if name not in self.calibration_params:
                    return output
                g_list = self.calibration_params[name] # list of [C, H, W] complex spectrum scaling factors
                task_idx = getattr(self, "current_task", 0)
                g = g_list[task_idx]
                V_fft = torch.fft.fft2(output, dim=(-2, -1))
                V_cal_fft = V_fft * g.unsqueeze(0)
                V_cal = torch.fft.ifft2(V_cal_fft, dim=(-2, -1)).real
                return V_cal
            elif self.mode == "wrsa":
                # Wiener-Regularized Spectral Alignment online hook
                if name not in self.calibration_params:
                    return output
                gamma = self.calibration_params[name] # [H, W] or [C, H, W] real scaling
                V_fft = torch.fft.fft2(output, dim=(-2, -1))
                V_mag = torch.abs(V_fft)
                V_phase = torch.angle(V_fft)
                # Scale magnitude
                if gamma.ndim == 2:
                    gamma_expanded = gamma.unsqueeze(0).unsqueeze(1) # [1, 1, H, W]
                else:
                    gamma_expanded = gamma.unsqueeze(0) # [1, C, H, W]
                V_mag_cal = V_mag * gamma_expanded
                # Reconstruct complex coefficients
                V_cal_fft = torch.polar(V_mag_cal, V_phase)
                V_cal = torch.fft.ifft2(V_cal_fft, dim=(-2, -1)).real
                return V_cal
            elif self.mode == "sptaac":
                # SP-TAAC online hook
                if name not in self.calibration_params:
                    return output
                scale = self.calibration_params[name] # [C]
                return output * scale.view(1, -1, 1, 1)
            elif self.mode == "repair":
                # REPAIR online hook
                if name not in self.calibration_params:
                    return output
                mean_merged, scale_repair, mean_target = self.calibration_params[name]
                # mean_merged, scale_repair, mean_target are [C]
                return mean_target.view(1, -1, 1, 1) + (output - mean_merged.view(1, -1, 1, 1)) * scale_repair.view(1, -1, 1, 1)
            return output
        return hook_fn
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def capture_activations(model, loader, layer_name, device):
    manager = ActivationHookManager(model, [layer_name], mode="capture")
    all_acts = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            all_acts.append(manager.captured[layer_name].cpu())
    manager.remove_hooks()
    return torch.cat(all_acts, dim=0) # [B_total, C, H, W]

def get_calibration_sets(train_datasets, size=128, seed=42):
    # Generates a joint calibration set of size*len(train_datasets)
    cal_subsets = []
    for i, train_dataset in enumerate(train_datasets):
        indices = list(range(len(train_dataset)))
        random.seed(seed + i)
        random.shuffle(indices)
        cal_subsets.append(Subset(train_dataset, indices[:size]))
    return cal_subsets

def run_evaluation(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total
