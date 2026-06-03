import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Disable cuDNN to bypass driver compatibility issues
torch.backends.cudnn.enabled = False

BATCH_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---
transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

transform_fmnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
])

transform_cifar10 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("Loading datasets...")
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)
test_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)

test_loaders = {
    'mnist': DataLoader(test_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'fmnist': DataLoader(test_fmnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'cifar10': DataLoader(test_cifar10, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

# --- Model Definitions ---
class MergedModel(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        
    def forward(self, x, task_name=None):
        features = self.backbone(x)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

def get_base_resnet():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

# --- Expert Loading ---
print("Setting up experts and progenitor...")
progenitor_backbone = get_base_resnet()
progenitor_state = torch.load("progenitor_backbone.pt", map_location=DEVICE)
progenitor_backbone.load_state_dict(progenitor_state)

expert_backbones = {}
expert_heads = {}

for task in ['mnist', 'fmnist', 'cifar10']:
    checkpoint_path = f"expert_{task}.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained expert checkpoint for {task}...")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        backbone_dict = {}
        head_dict = {}
        for k, v in state_dict.items():
            if k.startswith('0.'):
                backbone_dict[k[2:]] = v
            elif k.startswith('1.'):
                head_dict[k[2:]] = v
        expert_backbones[task] = backbone_dict
        expert_heads[task] = head_dict
    else:
        print(f"Error: checkpoint {checkpoint_path} not found!")

heads_modules = {task: nn.Linear(512, 10).to(DEVICE) for task in ['mnist', 'fmnist', 'cifar10']}
for task in ['mnist', 'fmnist', 'cifar10']:
    heads_modules[task].load_state_dict(expert_heads[task])
    
merged_model = MergedModel(get_base_resnet().to(DEVICE), heads_modules).to(DEVICE)

# --- Merging & Calibration Functions ---
def apply_task_arithmetic(backbone_dest, progenitor_state, expert_states, scaling=0.4):
    device = next(backbone_dest.parameters()).device
    merged_state_dict = {}
    keys = progenitor_state.keys()
    for key in keys:
        p_val = progenitor_state[key].to(device)
        if p_val.dtype.is_floating_point:
            task_vectors = [expert[key].to(device) - p_val for expert in expert_states]
            sum_task_vectors = torch.stack(task_vectors).sum(dim=0)
            merged_state_dict[key] = p_val + scaling * sum_task_vectors
        else:
            merged_state_dict[key] = p_val.clone()
    backbone_dest.load_state_dict(merged_state_dict)

def save_bn_stats(model):
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            stats[name] = (m.running_mean.clone(), m.running_var.clone())
    return stats

def load_bn_stats(model, stats):
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.running_mean.copy_(stats[name][0])
            m.running_var.copy_(stats[name][1])

def calibrate_bn_static(model, loader, N=32):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.momentum = 1.0
            m.reset_running_stats()
    with torch.no_grad():
        inputs, _ = next(iter(loader))
        inputs = inputs[:N].to(DEVICE)
        _ = model.backbone(inputs)
    model.eval()

# --- Quantization Helper ---
def quantize_model_weights(model, num_bits=4, per_channel=True):
    quant_state_dict = {}
    qmax = 2**(num_bits - 1) - 1
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2 and param.dtype.is_floating_point:
            W = param.detach()
            if per_channel:
                num_channels = W.shape[0]
                W_q = torch.zeros_like(W)
                for c in range(num_channels):
                    max_val = torch.max(torch.abs(W[c]))
                    if max_val > 1e-8:
                        delta = max_val / qmax
                        W_q[c] = torch.clamp(torch.round(W[c] / delta), -qmax, qmax) * delta
                    else:
                        W_q[c] = 0.0
            else:
                max_val = torch.max(torch.abs(W))
                if max_val > 1e-8:
                    delta = max_val / qmax
                    W_q = torch.clamp(torch.round(W / delta), -qmax, qmax) * delta
                else:
                    W_q = torch.zeros_like(W)
            quant_state_dict[name] = W_q
        else:
            quant_state_dict[name] = param.detach().clone()
    return quant_state_dict

def apply_quant_weights(model, quant_state_dict):
    for name, param in model.named_parameters():
        if name in quant_state_dict:
            param.data.copy_(quant_state_dict[name])

# --- Evaluation Function ---
def evaluate_model(model, task_name, mode='static', test_batch_size=256, initial_bn_stats=None):
    task_dataset = test_loaders[task_name].dataset
    loader = DataLoader(task_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    model.eval()
    if mode == 'itsc_pw_ams':
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.momentum = 0.1
                if initial_bn_stats is not None:
                    load_bn_stats(model, initial_bn_stats)
                else:
                    m.reset_running_stats()
                
    correct = 0
    total = 0
    batch_idx = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if mode == 'itsc_pw_ams':
                batch_idx += 1
                current_m = max(0.1, 1.0 / batch_idx)
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.momentum = current_m
            
            if mode == 'itsc_pw_ams':
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.train()
                _ = model(inputs, task_name)
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()
            
            outputs = model(inputs, task_name)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    model.eval()
    return (correct / total) * 100

def run_full_suite(model, mode='static', test_batch_size=256, initial_bn_stats=None):
    results = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        results[task] = evaluate_model(model, task, mode, test_batch_size, initial_bn_stats)
    results['avg'] = np.mean(list(results.values()))
    return results

# --- SWEEP RUNNER ---
scalings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
de_bn_accs = []
itsc_pw_ams_accs = []

print("\n=== STARTING SCALE SWEEP (INT4) ===")

for lmbda in scalings:
    print(f"Evaluating merging scale lambda = {lmbda:.1f}...")
    
    # 1. Static DE-BN Evaluation
    apply_task_arithmetic(merged_model.backbone, progenitor_state, list(expert_backbones.values()), scaling=lmbda)
    quant_dict = quantize_model_weights(merged_model, num_bits=4)
    apply_quant_weights(merged_model, quant_dict)
    
    real_bn_stats = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        calibrate_bn_static(merged_model, test_loaders[task], N=32)
        real_bn_stats[task] = save_bn_stats(merged_model)
        
    de_bn_res = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        load_bn_stats(merged_model, real_bn_stats[task])
        de_bn_res[task] = evaluate_model(merged_model, task, mode='static')
    de_bn_res['avg'] = np.mean(list(de_bn_res.values()))
    de_bn_accs.append(de_bn_res['avg'])
    print(f"  [Static DE-BN] Avg: {de_bn_res['avg']:.2f}%")
    
    # 2. Ours ITSC-PW-AMS Evaluation
    apply_task_arithmetic(merged_model.backbone, progenitor_state, list(expert_backbones.values()), scaling=lmbda)
    quant_dict = quantize_model_weights(merged_model, num_bits=4)
    apply_quant_weights(merged_model, quant_dict)
    initial_bn_stats = save_bn_stats(merged_model)
    
    itsc_res = run_full_suite(merged_model, mode='itsc_pw_ams', initial_bn_stats=initial_bn_stats)
    itsc_pw_ams_accs.append(itsc_res['avg'])
    print(f"  [ITSC-PW-AMS] Avg: {itsc_res['avg']:.2f}%")

print(f"Scalings: {scalings}")
print(f"DE-BN Sweep: {de_bn_accs}")
print(f"ITSC-PW-AMS Sweep: {itsc_pw_ams_accs}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(scalings, de_bn_accs, 'o-', label='Static DE-BN (Offline, N=32)', color='#e056fd')
plt.plot(scalings, itsc_pw_ams_accs, 'd-', label='ITSC-PW-AMS (Ours, Online)', color='#6ab04c')
plt.xlabel('Merging Coefficient (Scaling Lambda)')
plt.ylabel('Average Multi-Task Accuracy (INT4, %)')
plt.title('Performance vs. Merging Coefficient (INT4 Quantization)')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig('scaling_sweep_int4.png')
print("Saved plot to 'scaling_sweep_int4.png'")

with open("scaling_sweep_results.txt", "w") as f:
    f.write("=== SCALING COEFFICIENT SWEEP (INT4) ===\n")
    f.write(f"Scalings: {scalings}\n")
    f.write(f"DE-BN Accs: {de_bn_accs}\n")
    f.write(f"ITSC-PW-AMS Accs: {itsc_pw_ams_accs}\n")
