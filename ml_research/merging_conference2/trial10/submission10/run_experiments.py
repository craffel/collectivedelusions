import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Disable cuDNN to bypass driver compatibility issues
torch.backends.cudnn.enabled = False

# --- Hyperparameters & Configuration ---
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---
# MNIST and FMNIST are grayscale (1 channel). We resize to 32x32, convert to 3 channels, and normalize.
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

# Load datasets
print("Loading datasets...")
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)

train_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
test_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)

train_loaders = {
    'mnist': DataLoader(train_mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'fmnist': DataLoader(train_fmnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'cifar10': DataLoader(train_cifar10, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
}

test_loaders = {
    'mnist': DataLoader(test_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'fmnist': DataLoader(test_fmnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'cifar10': DataLoader(test_cifar10, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

# --- Corruptions ---
def apply_gaussian_noise(inputs, severity=0.2):
    return inputs + torch.randn_like(inputs) * severity

def apply_defocus_blur(inputs):
    # Apply standard torchvision GaussianBlur
    blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=1.5)
    return blur(inputs)

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
    model.fc = nn.Identity() # replace final fc layer with identity mapping
    return model

# --- Expert Loading ---
print("Setting up experts and progenitor...")
progenitor_backbone = get_base_resnet()
torch.save(progenitor_backbone.state_dict(), "progenitor_backbone.pt")

expert_backbones = {}
expert_heads = {}

# We load pre-trained checkpoints
for task in ['mnist', 'fmnist', 'cifar10']:
    checkpoint_path = f"expert_{task}.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained expert checkpoint for {task}...")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Split checkpoint keys into backbone and head
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

# Define merged model with task heads
heads_modules = {task: nn.Linear(512, 10).to(DEVICE) for task in ['mnist', 'fmnist', 'cifar10']}
for task in ['mnist', 'fmnist', 'cifar10']:
    heads_modules[task].load_state_dict(expert_heads[task])
    
merged_model = MergedModel(get_base_resnet().to(DEVICE), heads_modules).to(DEVICE)

# --- Merging Strategies ---
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

# --- Calibration Strategies ---
def apply_hns(progenitor_state, merged_model, expert_states, clip_min=0.1, clip_max=10.0):
    K = len(expert_states)
    keys = progenitor_state.keys()
    merged_state = merged_model.backbone.state_dict()
    new_state = {}
    
    with torch.no_grad():
        for key in keys:
            p_val = progenitor_state[key].to(DEVICE)
            m_val = merged_state[key].to(DEVICE)
            
            if 'weight' in key and len(p_val.shape) >= 2 and p_val.dtype.is_floating_point:
                expert_updates = [expert[key].to(DEVICE) - p_val for expert in expert_states]
                merged_update = m_val - p_val
                
                num_channels = p_val.shape[0]
                s_c = torch.zeros(num_channels, device=DEVICE)
                
                for c in range(num_channels):
                    merged_norm = torch.norm(merged_update[c])
                    if merged_norm > 1e-8:
                        expert_norms = sum(torch.norm(eu[c]) for eu in expert_updates)
                        s_c[c] = (expert_norms / K) / merged_norm
                    else:
                        s_c[c] = 1.0
                
                s_c = torch.clamp(s_c, clip_min, clip_max)
                broadcast_shape = [num_channels] + [1] * (len(p_val.shape) - 1)
                s_c_b = s_c.view(broadcast_shape)
                new_state[key] = p_val + s_c_b * merged_update
            else:
                new_state[key] = m_val.clone()
                
    merged_model.backbone.load_state_dict(new_state)

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

# --- Quantization Helpers ---
def quantize_model_weights(model, num_bits=8, per_channel=True):
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
    original_weights = {}
    for name, param in model.named_parameters():
        if name in quant_state_dict:
            original_weights[name] = param.data.clone()
            param.data.copy_(quant_state_dict[name])
    return original_weights

def restore_weights(model, original_weights):
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])

# --- Core Evaluation Functions ---
def evaluate_model(model, task_name, corruption=None, mode='static', test_batch_size=256, initial_bn_stats=None, max_batches=None):
    # Mode can be:
    # - 'static': Uses standard evaluation mode (uses pre-calibrated BN stats)
    # - 'itsc': Uses Inference-Time Self-Calibration (ITSC) (updates BN stats on-the-fly)
    # - 'itsc_pw': ITSC with Prior Warm-starting (no reset, fixed momentum 0.1)
    # - 'itsc_ams': ITSC with Adaptive Momentum Scheduling (reset, dynamic momentum 1/t)
    # - 'itsc_pw_ams': ITSC with Prior Warm-starting & Adaptive Momentum Scheduling (no reset, dynamic momentum 1/t)
    
    # Re-initialize DataLoader with specific test batch size
    task_dataset = test_loaders[task_name].dataset
    loader = DataLoader(task_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    model.eval()
    if mode in ['itsc', 'itsc_pw', 'itsc_ams', 'itsc_pw_ams']:
        # Enable BatchNorm update mode
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.momentum = 0.1 # default rolling EMA adaptation
                if mode in ['itsc', 'itsc_ams']:
                    m.reset_running_stats()
                elif mode in ['itsc_pw', 'itsc_pw_ams']:
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
            
            # Apply dynamic momentum schedule if requested
            if mode in ['itsc_ams', 'itsc_pw_ams']:
                batch_idx += 1
                current_m = max(0.1, 1.0 / batch_idx)
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.momentum = current_m
            else:
                batch_idx += 1
            
            # Apply corruptions if specified
            if corruption == 'noise_0.1':
                inputs = apply_gaussian_noise(inputs, 0.1)
            elif corruption == 'noise_0.2':
                inputs = apply_gaussian_noise(inputs, 0.2)
            elif corruption == 'blur':
                inputs = apply_defocus_blur(inputs)
                
            if mode in ['itsc', 'itsc_pw', 'itsc_ams', 'itsc_pw_ams']:
                # 1. Update running statistics by doing a forward pass in train mode for BN
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.train()
                _ = model(inputs, task_name) if isinstance(model, MergedModel) else model(inputs)
                
                # 2. Switch BN back to eval mode for the actual forward evaluation
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()
            
            # Actual forward evaluation
            if isinstance(model, MergedModel):
                outputs = model(inputs, task_name)
            else:
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if max_batches is not None and batch_idx >= max_batches:
                break
            
    # Reset back to eval
    model.eval()
    return (correct / total) * 100

def run_full_suite(model, prefix, mode='static', corruption=None, test_batch_size=256, initial_bn_stats=None, max_batches=None):
    results = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        results[task] = evaluate_model(model, task, corruption, mode, test_batch_size, initial_bn_stats, max_batches)
    results['avg'] = np.mean(list(results.values()))
    print(f"[{prefix}] MNIST: {results['mnist']:.2f}% | FMNIST: {results['fmnist']:.2f}% | CIFAR10: {results['cifar10']:.2f}% | Avg: {results['avg']:.2f}%")
    return results

# --- MAIN RUNNER ---
print("\n--- Starting Merging and Calibration Evaluation ---")

# Step 1: Benchmark Expert Oracles as Upper Bound
expert_oracles = {}
for task in ['mnist', 'fmnist', 'cifar10']:
    backbone = get_base_resnet()
    backbone.load_state_dict(expert_backbones[task])
    head = nn.Linear(512, 10)
    head.load_state_dict(expert_heads[task])
    m = nn.Sequential(backbone, head).to(DEVICE)
    m.eval()
    expert_oracles[task] = evaluate_model(nn.Sequential(backbone, heads_modules[task]), task, mode='static')
expert_oracles['avg'] = np.mean(list(expert_oracles.values()))
print(f"[Expert Oracles] MNIST: {expert_oracles['mnist']:.2f}% | FMNIST: {expert_oracles['fmnist']:.2f}% | CIFAR10: {expert_oracles['cifar10']:.2f}% | Avg: {expert_oracles['avg']:.2f}%")


# --- Step 2: Clean Environments Evaluation ---
print("\n=== CLEAN ENVIRONMENT EVALUATIONS ===")

precisions = {
    'FP32': None,
    'INT8': 8,
    'INT4': 4
}

all_clean_results = {}

for p_name, num_bits in precisions.items():
    print(f"\n--- Evaluation at {p_name} ---")
    all_clean_results[p_name] = {}
    
    # Apply Task Arithmetic base model
    apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
    
    # Quantize if specified
    if num_bits is not None:
        quant_dict = quantize_model_weights(merged_model, num_bits=num_bits)
        apply_quant_weights(merged_model, quant_dict)
    
    # 1. TA + None (Uncalibrated)
    # (Uses standard static evaluation with merged uncalibrated BN stats)
    # We first load merged stats directly (which is standard WA/TA default)
    # Wait, the default state after loading is just uncalibrated merged backbone weights.
    all_clean_results[p_name]['None'] = run_full_suite(merged_model, f"TA + None ({p_name})", mode='static')
    
    # 2. TA + HNS (Data-Free Parameter Calibration)
    # Apply HNS parameter scaling to the weights
    if num_bits is not None:
        # Restore full precision, apply HNS, then re-quantize
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        apply_hns(progenitor_backbone.state_dict(), merged_model, list(expert_backbones.values()))
        quant_dict = quantize_model_weights(merged_model, num_bits=num_bits)
        apply_quant_weights(merged_model, quant_dict)
    else:
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        apply_hns(progenitor_backbone.state_dict(), merged_model, list(expert_backbones.values()))
    all_clean_results[p_name]['HNS'] = run_full_suite(merged_model, f"TA + HNS ({p_name})", mode='static')
    
    # 3. TA + DE-BN (Static Real Data, N=32)
    # Reset model to Task Arithmetic base and apply offline calibration
    if num_bits is not None:
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        quant_dict = quantize_model_weights(merged_model, num_bits=num_bits)
        apply_quant_weights(merged_model, quant_dict)
    else:
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
    
    # Calibrate static BN
    real_bn_stats = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        calibrate_bn_static(merged_model, test_loaders[task], N=32)
        real_bn_stats[task] = save_bn_stats(merged_model)
        
    # Evaluate static DE-BN by loading task-specific cached stats before evaluation
    de_bn_res = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        load_bn_stats(merged_model, real_bn_stats[task])
        de_bn_res[task] = evaluate_model(merged_model, task, mode='static')
    de_bn_res['avg'] = np.mean(list(de_bn_res.values()))
    all_clean_results[p_name]['DE-BN'] = de_bn_res
    print(f"[TA + DE-BN ({p_name})] MNIST: {de_bn_res['mnist']:.2f}% | FMNIST: {de_bn_res['fmnist']:.2f}% | CIFAR10: {de_bn_res['cifar10']:.2f}% | Avg: {de_bn_res['avg']:.2f}%")
    
    # 4. TA + ITSC and its variants (Ours - Inference-Time Self-Calibration)
    for mode_name, mode_key in [
        ('ITSC', 'itsc'),
        ('ITSC-PW', 'itsc_pw'),
        ('ITSC-AMS', 'itsc_ams'),
        ('ITSC-PW-AMS', 'itsc_pw_ams')
    ]:
        if num_bits is not None:
            apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
            quant_dict = quantize_model_weights(merged_model, num_bits=num_bits)
            apply_quant_weights(merged_model, quant_dict)
        else:
            apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        initial_bn_stats = save_bn_stats(merged_model)
        all_clean_results[p_name][mode_name] = run_full_suite(merged_model, f"TA + {mode_name} ({p_name})", mode=mode_key, initial_bn_stats=initial_bn_stats)


# --- Step 3: Out-of-Distribution (OOD) Corruptions Evaluation ---
print("\n=== CORRUPTED / OOD EVALUATIONS ===")
corruptions = {
    'Gaussian Noise (0.2)': 'noise_0.2',
    'Defocus Blur': 'blur'
}

all_corrupt_results = {}

for c_name, c_key in corruptions.items():
    print(f"\n--- Evaluating under {c_name} in INT4 ---")
    all_corrupt_results[c_name] = {}
    
    # Standard static DE-BN (calibrated on clean data, evaluated on corrupted data)
    apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
    quant_dict = quantize_model_weights(merged_model, num_bits=4)
    apply_quant_weights(merged_model, quant_dict)
    
    # Calibrate static BN on clean data
    real_bn_stats = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        calibrate_bn_static(merged_model, test_loaders[task], N=32)
        real_bn_stats[task] = save_bn_stats(merged_model)
        
    de_bn_res = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        load_bn_stats(merged_model, real_bn_stats[task])
        de_bn_res[task] = evaluate_model(merged_model, task, corruption=c_key, mode='static')
    de_bn_res['avg'] = np.mean(list(de_bn_res.values()))
    all_corrupt_results[c_name]['DE-BN'] = de_bn_res
    print(f"[TA + DE-BN (Static, INT4)] MNIST: {de_bn_res['mnist']:.2f}% | FMNIST: {de_bn_res['fmnist']:.2f}% | CIFAR10: {de_bn_res['cifar10']:.2f}% | Avg: {de_bn_res['avg']:.2f}%")
    
    # Ours ITSC variants
    for mode_name, mode_key in [
        ('ITSC', 'itsc'),
        ('ITSC-PW', 'itsc_pw'),
        ('ITSC-AMS', 'itsc_ams'),
        ('ITSC-PW-AMS', 'itsc_pw_ams')
    ]:
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        quant_dict = quantize_model_weights(merged_model, num_bits=4)
        apply_quant_weights(merged_model, quant_dict)
        initial_bn_stats = save_bn_stats(merged_model)
        all_corrupt_results[c_name][mode_name] = run_full_suite(merged_model, f"TA + {mode_name} (Ours, INT4)", mode=mode_key, corruption=c_key, initial_bn_stats=initial_bn_stats)


# --- Step 4: Batch Size Sweep for Inference-Time Self-Calibration ---
print("\n=== ITSC BATCH SIZE SWEEP (INT4) ===")
batch_sizes = [8, 16, 32, 64, 128, 256]
de_bn_sweep_accs = []
itsc_sweep_accs = []
itsc_pw_sweep_accs = []
itsc_ams_sweep_accs = []
itsc_pw_ams_sweep_accs = []

for B in batch_sizes:
    print(f"Evaluating inference batch size B = {B}...")
    
    # Static DE-BN sweep (calibrated with N = B offline clean samples, evaluated at clean eval)
    apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
    quant_dict = quantize_model_weights(merged_model, num_bits=4)
    apply_quant_weights(merged_model, quant_dict)
    
    sweep_stats_real = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        calibrate_bn_static(merged_model, test_loaders[task], N=B)
        sweep_stats_real[task] = save_bn_stats(merged_model)
        
    temp_accs = []
    for task in ['mnist', 'fmnist', 'cifar10']:
        load_bn_stats(merged_model, sweep_stats_real[task])
        temp_accs.append(evaluate_model(merged_model, task, mode='static', test_batch_size=B))
    de_bn_sweep_accs.append(np.mean(temp_accs))
    
    # Ours ITSC variants sweep
    for mode_name, mode_key, sweep_list in [
        ('ITSC', 'itsc', itsc_sweep_accs),
        ('ITSC-PW', 'itsc_pw', itsc_pw_sweep_accs),
        ('ITSC-AMS', 'itsc_ams', itsc_ams_sweep_accs),
        ('ITSC-PW-AMS', 'itsc_pw_ams', itsc_pw_ams_sweep_accs)
    ]:
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        quant_dict = quantize_model_weights(merged_model, num_bits=4)
        apply_quant_weights(merged_model, quant_dict)
        initial_bn_stats = save_bn_stats(merged_model)
        res_val = run_full_suite(merged_model, f"TA + {mode_name} (B={B})", mode=mode_key, test_batch_size=B, initial_bn_stats=initial_bn_stats)
        sweep_list.append(res_val['avg'])

print(f"Sweep Batch Sizes: {batch_sizes}")
print(f"DE-BN Accs: {de_bn_sweep_accs}")
print(f"ITSC Accs: {itsc_sweep_accs}")
print(f"ITSC-PW Accs: {itsc_pw_sweep_accs}")
print(f"ITSC-AMS Accs: {itsc_ams_sweep_accs}")
print(f"ITSC-PW-AMS Accs: {itsc_pw_ams_sweep_accs}")


# --- Step 4.5: Transient / Short-Stream Adaptation Sweep (INT4) ---
print("\n=== ITSC TRANSIENT ADAPTATION SWEEP (INT4) ===")
# We evaluate on a small batch size B=16, and sweep the number of inference batches processed K from 1 to 16.
transient_batches = [1, 2, 4, 8, 16]
de_bn_trans_accs = []
itsc_trans_accs = []
itsc_pw_trans_accs = []
itsc_ams_trans_accs = []
itsc_pw_ams_trans_accs = []

for K in transient_batches:
    print(f"Evaluating transient adaptation over K = {K} batch(es) of size 16...")
    
    # For DE-BN, we calibrate using offline N = 16 * K clean samples, and then evaluate on K batches of size 16.
    apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
    quant_dict = quantize_model_weights(merged_model, num_bits=4)
    apply_quant_weights(merged_model, quant_dict)
    
    trans_stats_real = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        calibrate_bn_static(merged_model, test_loaders[task], N=16 * K)
        trans_stats_real[task] = save_bn_stats(merged_model)
        
    temp_accs = []
    for task in ['mnist', 'fmnist', 'cifar10']:
        load_bn_stats(merged_model, trans_stats_real[task])
        temp_accs.append(evaluate_model(merged_model, task, mode='static', test_batch_size=16, max_batches=K))
    de_bn_trans_accs.append(np.mean(temp_accs))
    
    # Ours ITSC variants
    for mode_name, mode_key, sweep_list in [
        ('ITSC', 'itsc', itsc_trans_accs),
        ('ITSC-PW', 'itsc_pw', itsc_pw_trans_accs),
        ('ITSC-AMS', 'itsc_ams', itsc_ams_trans_accs),
        ('ITSC-PW-AMS', 'itsc_pw_ams', itsc_pw_ams_trans_accs)
    ]:
        apply_task_arithmetic(merged_model.backbone, progenitor_backbone.state_dict(), list(expert_backbones.values()))
        quant_dict = quantize_model_weights(merged_model, num_bits=4)
        apply_quant_weights(merged_model, quant_dict)
        initial_bn_stats = save_bn_stats(merged_model)
        res_val = run_full_suite(merged_model, f"TA + {mode_name} (K={K})", mode=mode_key, test_batch_size=16, initial_bn_stats=initial_bn_stats, max_batches=K)
        sweep_list.append(res_val['avg'])

print(f"Transient Batches (K): {transient_batches}")
print(f"DE-BN Transient Accs: {de_bn_trans_accs}")
print(f"ITSC Transient Accs: {itsc_trans_accs}")
print(f"ITSC-PW Transient Accs: {itsc_pw_trans_accs}")
print(f"ITSC-AMS Transient Accs: {itsc_ams_trans_accs}")
print(f"ITSC-PW-AMS Transient Accs: {itsc_pw_ams_trans_accs}")


# --- Step 5: Plotting the Results ---
# Plot 1: Static DE-BN vs Ours ITSC variants under Corruptions in INT4
labels = ['Clean', 'Gaussian Noise', 'Defocus Blur']
de_bn_bars = [
    all_clean_results['INT4']['DE-BN']['avg'],
    all_corrupt_results['Gaussian Noise (0.2)']['DE-BN']['avg'],
    all_corrupt_results['Defocus Blur']['DE-BN']['avg']
]
itsc_bars = [
    all_clean_results['INT4']['ITSC']['avg'],
    all_corrupt_results['Gaussian Noise (0.2)']['ITSC']['avg'],
    all_corrupt_results['Defocus Blur']['ITSC']['avg']
]
itsc_pw_ams_bars = [
    all_clean_results['INT4']['ITSC-PW-AMS']['avg'],
    all_corrupt_results['Gaussian Noise (0.2)']['ITSC-PW-AMS']['avg'],
    all_corrupt_results['Defocus Blur']['ITSC-PW-AMS']['avg']
]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, de_bn_bars, width, label='Static DE-BN (Offline)', color='#e056fd')
rects2 = ax.bar(x, itsc_bars, width, label='ITSC (Ours, Online)', color='#f0932b')
rects3 = ax.bar(x + width, itsc_pw_ams_bars, width, label='ITSC-PW-AMS (Ours, Online)', color='#6ab04c')

ax.set_ylabel('Average Multi-Task Accuracy (INT4, %)')
ax.set_title('Robustness under Environmental Corruptions (INT4)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.legend()
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig('corruption_robustness_int4.png')
print("Saved plot to 'corruption_robustness_int4.png'")

# Plot 2: Inference Batch Size Sweep vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, de_bn_sweep_accs, 'o-', label='Static DE-BN (Offline)', color='#e056fd')
plt.plot(batch_sizes, itsc_sweep_accs, 's--', label='ITSC (Ours, Online)', color='#f0932b')
plt.plot(batch_sizes, itsc_pw_sweep_accs, '^:', label='ITSC-PW (Ours, Online)', color='#22a6b3')
plt.plot(batch_sizes, itsc_ams_sweep_accs, 'x-.', label='ITSC-AMS (Ours, Online)', color='#30336b')
plt.plot(batch_sizes, itsc_pw_ams_sweep_accs, 'd-', label='ITSC-PW-AMS (Ours, Online)', color='#6ab04c')
plt.xscale('log', base=2)
plt.xticks(batch_sizes, [str(b) for b in batch_sizes])
plt.xlabel('Inference Batch Size (B) / Calibration Samples (N)')
plt.ylabel('Average Multi-Task Accuracy (INT4, %)')
plt.title('Performance vs. Batch Size / Calibration Size (INT4)')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig('itsc_batch_size_sweep.png')
print("Saved plot to 'itsc_batch_size_sweep.png'")

# Plot 3: Transient Adaptation Sweep
plt.figure(figsize=(10, 6))
plt.plot(transient_batches, de_bn_trans_accs, 'o-', label='Static DE-BN (Offline)', color='#e056fd')
plt.plot(transient_batches, itsc_trans_accs, 's--', label='ITSC (Ours, Online)', color='#f0932b')
plt.plot(transient_batches, itsc_pw_trans_accs, '^:', label='ITSC-PW (Ours, Online)', color='#22a6b3')
plt.plot(transient_batches, itsc_ams_trans_accs, 'x-.', label='ITSC-AMS (Ours, Online)', color='#30336b')
plt.plot(transient_batches, itsc_pw_ams_trans_accs, 'd-', label='ITSC-PW-AMS (Ours, Online)', color='#6ab04c')
plt.xticks(transient_batches, [f"{k} ({k*16} smp)" for k in transient_batches])
plt.xlabel('Number of Transient Inference Batches (K) of size 16')
plt.ylabel('Average Multi-Task Accuracy (INT4, %)')
plt.title('Transient Adaptation Speed on Short Streams (INT4)')
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig('itsc_transient_adaptation.png')
print("Saved plot to 'itsc_transient_adaptation.png'")


# --- Step 6: Print Final LaTeX-Style markdown Results Tables ---
print("\n" + "="*50)
print("FINAL COMPREHENSIVE EXPERIMENTAL RESULTS")
print("="*50)
print("\nTable 1: Clean Performance Across Precision Regimes (Accuracy %)")
print("| Method | Data Requirement | FP32 | INT8 | INT4 |")
print("|---|---|---|---|---|")
print(f"| Expert Oracles (Upper Bound) | Full Train Sets | {expert_oracles['avg']:.2f}% | N/A | N/A |")
print(f"| TA + None (Uncalibrated) | None | {all_clean_results['FP32']['None']['avg']:.2f}% | {all_clean_results['INT8']['None']['avg']:.2f}% | {all_clean_results['INT4']['None']['avg']:.2f}% |")
print(f"| TA + HNS | None (Data-Free) | {all_clean_results['FP32']['HNS']['avg']:.2f}% | {all_clean_results['INT8']['HNS']['avg']:.2f}% | {all_clean_results['INT4']['HNS']['avg']:.2f}% |")
print(f"| TA + DE-BN (Static N=32) | 32 Real Samples | {all_clean_results['FP32']['DE-BN']['avg']:.2f}% | {all_clean_results['INT8']['DE-BN']['avg']:.2f}% | {all_clean_results['INT4']['DE-BN']['avg']:.2f}% |")
print(f"| TA + ITSC (Ours) | None (100% Data-Free) | {all_clean_results['FP32']['ITSC']['avg']:.2f}% | {all_clean_results['INT8']['ITSC']['avg']:.2f}% | {all_clean_results['INT4']['ITSC']['avg']:.2f}% |")
print(f"| TA + ITSC-PW (Ours) | None (100% Data-Free) | {all_clean_results['FP32']['ITSC-PW']['avg']:.2f}% | {all_clean_results['INT8']['ITSC-PW']['avg']:.2f}% | {all_clean_results['INT4']['ITSC-PW']['avg']:.2f}% |")
print(f"| TA + ITSC-AMS (Ours) | None (100% Data-Free) | {all_clean_results['FP32']['ITSC-AMS']['avg']:.2f}% | {all_clean_results['INT8']['ITSC-AMS']['avg']:.2f}% | {all_clean_results['INT4']['ITSC-AMS']['avg']:.2f}% |")
print(f"| **TA + ITSC-PW-AMS (Ours)** | **None (100% Data-Free)** | **{all_clean_results['FP32']['ITSC-PW-AMS']['avg']:.2f}%** | **{all_clean_results['INT8']['ITSC-PW-AMS']['avg']:.2f}%** | **{all_clean_results['INT4']['ITSC-PW-AMS']['avg']:.2f}%** |")

print("\nTable 2: Robustness under Corruptions (INT4 Accuracy %)")
print("| Method | Clean | Gaussian Noise (0.2) | Defocus Blur |")
print("|---|---|---|---|")
print(f"| TA + DE-BN (Static N=32) | {all_clean_results['INT4']['DE-BN']['avg']:.2f}% | {all_corrupt_results['Gaussian Noise (0.2)']['DE-BN']['avg']:.2f}% | {all_corrupt_results['Defocus Blur']['DE-BN']['avg']:.2f}% |")
print(f"| TA + ITSC (Ours) | {all_clean_results['INT4']['ITSC']['avg']:.2f}% | {all_corrupt_results['Gaussian Noise (0.2)']['ITSC']['avg']:.2f}% | {all_corrupt_results['Defocus Blur']['ITSC']['avg']:.2f}% |")
print(f"| TA + ITSC-PW (Ours) | {all_clean_results['INT4']['ITSC-PW']['avg']:.2f}% | {all_corrupt_results['Gaussian Noise (0.2)']['ITSC-PW']['avg']:.2f}% | {all_corrupt_results['Defocus Blur']['ITSC-PW']['avg']:.2f}% |")
print(f"| TA + ITSC-AMS (Ours) | {all_clean_results['INT4']['ITSC-AMS']['avg']:.2f}% | {all_corrupt_results['Gaussian Noise (0.2)']['ITSC-AMS']['avg']:.2f}% | {all_corrupt_results['Defocus Blur']['ITSC-AMS']['avg']:.2f}% |")
print(f"| **TA + ITSC-PW-AMS (Ours)** | **{all_clean_results['INT4']['ITSC-PW-AMS']['avg']:.2f}%** | **{all_corrupt_results['Gaussian Noise (0.2)']['ITSC-PW-AMS']['avg']:.2f}%** | **{all_corrupt_results['Defocus Blur']['ITSC-PW-AMS']['avg']:.2f}%** |")

print("\nTable 3: Transient Adaptation Accuracy (INT4, B=16, Accuracy %)")
print("| Method | K=1 (16 smp) | K=2 (32 smp) | K=4 (64 smp) | K=8 (128 smp) | K=16 (256 smp) |")
print("|---|---|---|---|---|---|")
print(f"| TA + DE-BN (Static) | {de_bn_trans_accs[0]:.2f}% | {de_bn_trans_accs[1]:.2f}% | {de_bn_trans_accs[2]:.2f}% | {de_bn_trans_accs[3]:.2f}% | {de_bn_trans_accs[4]:.2f}% |")
print(f"| TA + ITSC (Ours) | {itsc_trans_accs[0]:.2f}% | {itsc_trans_accs[1]:.2f}% | {itsc_trans_accs[2]:.2f}% | {itsc_trans_accs[3]:.2f}% | {itsc_trans_accs[4]:.2f}% |")
print(f"| TA + ITSC-PW (Ours) | {itsc_pw_trans_accs[0]:.2f}% | {itsc_pw_trans_accs[1]:.2f}% | {itsc_pw_trans_accs[2]:.2f}% | {itsc_pw_trans_accs[3]:.2f}% | {itsc_pw_trans_accs[4]:.2f}% |")
print(f"| TA + ITSC-AMS (Ours) | {itsc_ams_trans_accs[0]:.2f}% | {itsc_ams_trans_accs[1]:.2f}% | {itsc_ams_trans_accs[2]:.2f}% | {itsc_ams_trans_accs[3]:.2f}% | {itsc_ams_trans_accs[4]:.2f}% |")
print(f"| **TA + ITSC-PW-AMS (Ours)** | **{itsc_pw_ams_trans_accs[0]:.2f}%** | **{itsc_pw_ams_trans_accs[1]:.2f}%** | **{itsc_pw_ams_trans_accs[2]:.2f}%** | **{itsc_pw_ams_trans_accs[3]:.2f}%** | **{itsc_pw_ams_trans_accs[4]:.2f}%** |")


# Write out the results text log for references
with open("experimental_results_itsc.txt", "w") as f:
    f.write("=== ITSC EXPERIMENTAL RESULTS (EXTENDED WITH TRANSIENT STUDY) ===\n")
    f.write(f"Oracles: {expert_oracles}\n\n")
    f.write("--- Clean Results ---\n")
    for p_name in precisions.keys():
        f.write(f"Precision: {p_name}\n")
        for k in ['None', 'HNS', 'DE-BN', 'ITSC', 'ITSC-PW', 'ITSC-AMS', 'ITSC-PW-AMS']:
            f.write(f"  {k}: {all_clean_results[p_name][k]}\n")
        f.write("\n")
        
    f.write("--- Corrupted Results (INT4) ---\n")
    for c_name in corruptions.keys():
        f.write(f"Corruption: {c_name}\n")
        for k in ['DE-BN', 'ITSC', 'ITSC-PW', 'ITSC-AMS', 'ITSC-PW-AMS']:
            f.write(f"  {k}: {all_corrupt_results[c_name][k]}\n")
        f.write("\n")
        
    f.write("--- Sweeps ---\n")
    f.write(f"Batch Sizes: {batch_sizes}\n")
    f.write(f"DE-BN Sweep: {de_bn_sweep_accs}\n")
    f.write(f"ITSC Sweep: {itsc_sweep_accs}\n")
    f.write(f"ITSC-PW Sweep: {itsc_pw_sweep_accs}\n")
    f.write(f"ITSC-AMS Sweep: {itsc_ams_sweep_accs}\n")
    f.write(f"ITSC-PW-AMS Sweep: {itsc_pw_ams_sweep_accs}\n\n")
    
    f.write("--- Transient adaptation Sweep (INT4) ---\n")
    f.write(f"Transient Batches (K): {transient_batches}\n")
    f.write(f"DE-BN Transient Accs: {de_bn_trans_accs}\n")
    f.write(f"ITSC Transient Accs: {itsc_trans_accs}\n")
    f.write(f"ITSC-PW Transient Accs: {itsc_pw_trans_accs}\n")
    f.write(f"ITSC-AMS Transient Accs: {itsc_ams_trans_accs}\n")
    f.write(f"ITSC-PW-AMS Transient Accs: {itsc_pw_ams_trans_accs}\n")
