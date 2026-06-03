import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

# Disable cuDNN
torch.backends.cudnn.enabled = False

BATCH_SIZE = 128
EVAL_SAMPLES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(task):
    if task == 'mnist':
        base_transform = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
        norm_mean, norm_std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
    elif task == 'fashion':
        base_transform = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
        norm_mean, norm_std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
    elif task == 'cifar10':
        base_transform = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
        norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError(f"Unknown task: {task}")
        
    transform = transforms.Compose(base_transform + [transforms.Normalize(norm_mean, norm_std)])
    
    if task == 'mnist':
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif task == 'fashion':
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    elif task == 'cifar10':
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        
    indices = list(range(min(EVAL_SAMPLES, len(test_set))))
    test_subset = Subset(test_set, indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return test_loader

def quantize_weight(tensor, bits=8):
    if bits is None or bits >= 32:
        return tensor
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        return tensor
    scale = max_val / qmax
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q_tensor * scale

def load_models():
    progenitor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    progenitor.fc = nn.Linear(progenitor.fc.in_features, 10)
    
    mnist_expert = resnet18()
    mnist_expert.fc = nn.Linear(mnist_expert.fc.in_features, 10)
    mnist_expert.load_state_dict(torch.load("mnist_expert.pt", map_location='cpu'))
    
    fashion_expert = resnet18()
    fashion_expert.fc = nn.Linear(fashion_expert.fc.in_features, 10)
    fashion_expert.load_state_dict(torch.load("fashion_expert.pt", map_location='cpu'))
    
    cifar_expert = resnet18()
    cifar_expert.fc = nn.Linear(cifar_expert.fc.in_features, 10)
    cifar_expert.load_state_dict(torch.load("cifar10_expert.pt", map_location='cpu'))
    
    return progenitor, {'mnist': mnist_expert, 'fashion': fashion_expert, 'cifar10': cifar_expert}

def merge_models_base(progenitor, experts, method='ta', lam=0.5):
    merged = copy.deepcopy(progenitor)
    merged_state = merged.state_dict()
    prog_state = progenitor.state_dict()
    expert_states = {t: experts[t].state_dict() for t in experts}
    keys = [k for k in prog_state.keys() if 'fc' not in k]
    
    for key in keys:
        tvs = [expert_states[t][key].float() - prog_state[key].float() for t in experts]
        merged_state[key] = prog_state[key].float() + lam * torch.sum(torch.stack(tvs), dim=0)
            
    merged.load_state_dict(merged_state)
    return merged

def apply_calibration_gamma(progenitor, experts, merged_model, gamma=1.5, bits=8):
    calibrated = copy.deepcopy(merged_model)
    cal_state = calibrated.state_dict()
    prog_state = progenitor.state_dict()
    expert_states = {t: experts[t].state_dict() for t in experts}
    
    keys = [k for k in prog_state.keys() if 'fc' not in k]
    
    for key in keys:
        if not prog_state[key].is_floating_point():
            continue
            
        w_init = prog_state[key].float()
        w_merged = cal_state[key].float()
        t_merged = w_merged - w_init
        t_experts = [expert_states[t][key].float() - w_init for t in experts]
        
        if t_merged.dim() >= 2:
            scales = []
            for c in range(t_merged.shape[0]):
                norm_merged_c = torch.norm(t_merged[c])
                norm_experts_c = torch.mean(torch.stack([torch.norm(t[c]) for t in t_experts]))
                scale_c = norm_experts_c / (norm_merged_c + 1e-8)
                scales.append(scale_c.item())
            
            scales = np.array(scales)
            median = np.median(scales)
            mad = np.median(np.abs(scales - median))
            if mad == 0:
                mad = 1e-4
                
            lower_bound = max(0.1, median - gamma * mad)
            upper_bound = min(4.0, median + gamma * mad)
            
            t_corrected = torch.zeros_like(t_merged)
            for c in range(t_merged.shape[0]):
                scale_c = scales[c]
                scale_c_clamped = np.clip(scale_c, lower_bound, upper_bound)
                t_corrected[c] = scale_c_clamped * t_merged[c]
            w_corrected = w_init + t_corrected
        else:
            norm_merged = torch.norm(t_merged)
            norm_experts = torch.mean(torch.stack([torch.norm(t) for t in t_experts]))
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=3.0)
            w_corrected = w_init + scale * t_merged
            
        cal_state[key] = quantize_weight(w_corrected, bits)
            
    calibrated.load_state_dict(cal_state)
    return calibrated

def evaluate_model(model, experts, task):
    task_model = copy.deepcopy(model)
    task_model_state = task_model.state_dict()
    expert_state = experts[task].state_dict()
    for k in expert_state.keys():
        if 'fc' in k:
            task_model_state[k] = expert_state[k]
    task_model.load_state_dict(task_model_state)
    task_model = task_model.to(DEVICE)
    task_model.eval()
    
    loader = get_dataloaders(task)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = task_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

if __name__ == "__main__":
    print("Loading models...")
    progenitor, experts = load_models()
    print("Models loaded successfully.")
    
    base_merged = merge_models_base(progenitor, experts, 'ta', 0.5)
    
    gammas = [1.0, 1.5, 2.0, 3.0]
    results = {}
    
    for g in gammas:
        print(f"Evaluating gamma = {g}...")
        cal_model = apply_calibration_gamma(progenitor, experts, base_merged, gamma=g, bits=8)
        accs = {}
        for task in ['mnist', 'fashion', 'cifar10']:
            acc = evaluate_model(cal_model, experts, task)
            accs[task] = acc
        results[str(g)] = {
            'mnist': accs['mnist'],
            'fashion': accs['fashion'],
            'cifar10': accs['cifar10'],
            'avg': np.mean(list(accs.values()))
        }
        print(f"Gamma {g}: MNIST={accs['mnist']:.2f}%, Fashion={accs['fashion']:.2f}%, CIFAR={accs['cifar10']:.2f}%, Avg={results[str(g)]['avg']:.2f}%")
        
    with open("gamma_sweep.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Gamma sweep completed and saved to gamma_sweep.json!")
