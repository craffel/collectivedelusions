import os
import copy
import hashlib
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

# 2. Data Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 3. Load Test Datasets
print("Loading test datasets...")
test_sets = {
    'mnist': datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
}

test_loaders = {
    name: DataLoader(test_sets[name], batch_size=256, shuffle=False, num_workers=4)
    for name in test_sets
}

# Create calibration loaders (tiny subsets for DE-BN and DE-QC)
# We use N = 32 samples per task
print("Creating calibration subsets (N=32)...")
cal_loaders = {}
for name in test_sets:
    cal_subset = Subset(test_sets[name], range(32))
    cal_loaders[name] = DataLoader(cal_subset, batch_size=32, shuffle=False)

# 4. Progenitor Model Definition
def get_progenitor():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

# 5. Load Trained Checkpoints
print("Loading checkpoints...")
progenitor_state = torch.load("checkpoint_progenitor.pth", map_location=device)

expert_states = {
    'mnist': torch.load("checkpoint_mnist.pth", map_location=device),
    'fmnist': torch.load("checkpoint_fmnist.pth", map_location=device),
    'cifar10': torch.load("checkpoint_cifar10.pth", map_location=device)
}

# Extract backbone and head weights for each expert
expert_backbones = {}
expert_heads = {}
tasks = ['mnist', 'fmnist', 'cifar10']

for t in tasks:
    # Separate backbone and head from nn.Sequential wrapper saved state_dict
    backbone_sd = {k[2:]: v for k, v in expert_states[t].items() if k.startswith("0.")}
    head_sd = {k[2:]: v for k, v in expert_states[t].items() if k.startswith("1.")}
    expert_backbones[t] = backbone_sd
    expert_heads[t] = head_sd

# 6. Uniform PTQ Implementation
def quantize_tensor(tensor, num_bits=8, per_channel=False):
    if num_bits is None:
        return tensor
    qmax = 2**(num_bits - 1) - 1
    
    if per_channel:
        orig_shape = tensor.shape
        flat_tensor = tensor.view(orig_shape[0], -1)
        max_vals = flat_tensor.abs().max(dim=1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-8)
        delta = max_vals / qmax
        
        q_tensor = torch.round(tensor / delta.view(orig_shape[0], *([1] * (len(orig_shape) - 1))))
        q_tensor = torch.clamp(q_tensor, -qmax, qmax)
        dq_tensor = q_tensor * delta.view(orig_shape[0], *([1] * (len(orig_shape) - 1)))
        return dq_tensor
    else:
        max_val = tensor.abs().max()
        if max_val < 1e-8:
            return tensor
        delta = max_val / qmax
        q_tensor = torch.round(tensor / delta)
        q_tensor = torch.clamp(q_tensor, -qmax, qmax)
        dq_tensor = q_tensor * delta
        return dq_tensor

def apply_ptq_to_state_dict(state_dict, num_bits=8, per_channel=False):
    if num_bits is None:
        return state_dict
    quant_sd = {}
    for name, param in state_dict.items():
        if "weight" in name and ("conv" in name or "fc" in name or "linear" in name or "classifier" in name):
            quant_sd[name] = quantize_tensor(param.clone(), num_bits, per_channel)
        else:
            quant_sd[name] = param.clone()
    return quant_sd

# 7. Holographic Phase Key Generator (Reproducible, Seeded local torch.Generator)
def get_phase_key(param_name, param_shape, task_idx, device):
    seed_str = f"task_{task_idx}_{param_name}"
    seed_hash = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest()[:8], 16)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_hash)
    
    signs = torch.randint(0, 2, size=param_shape, generator=gen, device=device) * 2 - 1
    return signs.float()

# 8. Evaluation Helper
def evaluate_model(backbone_sd, head_sd, task_name, num_bits=None, per_channel=False, corruption=None):
    # Quantize the parameters if requested
    quant_backbone = apply_ptq_to_state_dict(backbone_sd, num_bits, per_channel)
    quant_head = apply_ptq_to_state_dict(head_sd, num_bits, per_channel)
    
    # Instantiate models
    backbone = get_progenitor()
    backbone.load_state_dict(quant_backbone)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(quant_head)
    
    model = nn.Sequential(backbone, head).to(device)
    model.eval()
    
    # Apply corruption transform if requested
    loader = test_loaders[task_name]
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply environmental corruptions on inputs
            if corruption == 'noise':
                # Additive zero-mean Gaussian Noise with sigma = 0.1
                inputs = inputs + 0.1 * torch.randn_like(inputs)
            elif corruption == 'blur':
                # Gaussian Blur (kernel size (3,3), sigma 1.0)
                blur_transform = transforms.GaussianBlur(kernel_size=(3, 3), sigma=1.0)
                inputs = blur_transform(inputs)
                
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return correct / total * 100.0

# 9. Calibration Helper (DE-BN)
def calibrate_model_bn(backbone_sd, head_sd, task_name, num_bits=None, per_channel=False):
    # Quantize beforehand
    quant_backbone = apply_ptq_to_state_dict(backbone_sd, num_bits, per_channel)
    quant_head = apply_ptq_to_state_dict(head_sd, num_bits, per_channel)
    
    backbone = get_progenitor()
    backbone.load_state_dict(quant_backbone)
    
    head = nn.Linear(512, 10)
    head.load_state_dict(quant_head)
    
    model = nn.Sequential(backbone, head).to(device)
    
    # Set momentum of all BN layers to 1.0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = 1.0
            m.reset_running_stats()
            m.train()
            
    # Run calibration batch of N = 32 samples
    with torch.no_grad():
        for inputs, _ in cal_loaders[task_name]:
            inputs = inputs.to(device)
            _ = model(inputs)
            break
            
    # Put model back into eval mode
    model.eval()
    
    # Return calibrated backbone state dict
    return backbone.state_dict()

# 10. Core Experiments Definition
results = {}

# --- EXPERIMENT 0: Individual Expert Oracles (Upper Bound) ---
print("\n=== Running Experiment: Expert Oracles ===")
oracle_accs = {}
for t in tasks:
    acc = evaluate_model(expert_backbones[t], expert_heads[t], t)
    oracle_accs[t] = acc
    print(f"Expert Oracle {t.upper()} Accuracy: {acc:.2f}%")
results['oracle'] = oracle_accs

# --- EXPERIMENT 1: Standard Weight Averaging (WA) ---
print("\n=== Running Experiment: Weight Averaging (WA) ===")
# Perform linear average of weights and stats
wa_backbone_sd = {}
for key in progenitor_state.keys():
    if progenitor_state[key].is_floating_point():
        wa_backbone_sd[key] = torch.stack([expert_backbones[t][key] for t in tasks]).mean(dim=0)
    else:
        wa_backbone_sd[key] = progenitor_state[key].clone()

wa_accs = {}
for t in tasks:
    acc = evaluate_model(wa_backbone_sd, expert_heads[t], t)
    wa_accs[t] = acc
    print(f"WA {t.upper()} Accuracy: {acc:.2f}%")
results['wa'] = wa_accs

# --- EXPERIMENT 2: Task Arithmetic (TA) ---
print("\n=== Running Experiment: Task Arithmetic (TA) ===")
# Sweep lambda in [0.1, 0.3, 0.5, 0.7, 1.0]
ta_results = {}
best_lambda = 0.3
best_ta_backbone = None
best_ta_acc = 0

for lam in [0.1, 0.3, 0.5, 0.7, 1.0]:
    ta_backbone_sd = {}
    for key in progenitor_state.keys():
        if progenitor_state[key].is_floating_point():
            task_vectors = [expert_backbones[t][key] - progenitor_state[key] for t in tasks]
            ta_backbone_sd[key] = progenitor_state[key] + lam * sum(task_vectors)
        else:
            ta_backbone_sd[key] = progenitor_state[key].clone()
            
    ta_accs = {}
    for t in tasks:
        acc = evaluate_model(ta_backbone_sd, expert_heads[t], t)
        ta_accs[t] = acc
    avg_acc = sum(ta_accs.values()) / len(tasks)
    print(f"TA (lambda={lam}) Average Accuracy: {avg_acc:.2f}% (MNIST: {ta_accs['mnist']:.2f}%, FMNIST: {ta_accs['fmnist']:.2f}%, CIFAR10: {ta_accs['cifar10']:.2f}%)")
    ta_results[lam] = ta_accs
    if avg_acc > best_ta_acc:
        best_ta_acc = avg_acc
        best_lambda = lam
        best_ta_backbone = ta_backbone_sd
        
results['ta'] = ta_results
print(f"Best TA Scaling lambda: {best_lambda}")

# --- EXPERIMENT 3: Task-Specific DE-BN (Baseline) ---
print("\n=== Running Experiment: Task-Specific DE-BN ===")
# We calibrate best_ta_backbone using 32 samples from each task
de_bn_accs = {}
for t in tasks:
    cal_backbone = calibrate_model_bn(best_ta_backbone, expert_heads[t], t)
    acc = evaluate_model(cal_backbone, expert_heads[t], t)
    de_bn_accs[t] = acc
    print(f"TA + DE-BN {t.upper()} Accuracy: {acc:.2f}%")
results['ta_de_bn'] = de_bn_accs

# --- EXPERIMENT 4: QCOT (Quantization-Constrained Optimal Transport) ---
print("\n=== Running Experiment: QCOT ===")
# C = 0.5 is identified as the optimal threshold in Paper 9
qcot_backbone_sd = {}
for key in progenitor_state.keys():
    if progenitor_state[key].is_floating_point() and "weight" in key:
        orig_shape = progenitor_state[key].shape
        flat_init = progenitor_state[key].view(-1)
        flat_updates = [(expert_backbones[t][key] - progenitor_state[key]).view(-1) for t in tasks]
        
        # 1D OT: Sort each update
        sorted_updates = [u.sort()[0] for u in flat_updates]
        y = torch.stack(sorted_updates, dim=0).mean(dim=0)
        
        # Clip to -0.5, 0.5
        clipped_y = torch.clamp(y, -0.5, 0.5)
        
        # Sort indices of merged update to project back
        merged_update = torch.stack(flat_updates, dim=0).mean(dim=0)
        _, sort_indices = merged_update.sort()
        
        qcot_update = torch.zeros_like(merged_update)
        qcot_update[sort_indices] = clipped_y
        
        qcot_backbone_sd[key] = progenitor_state[key] + qcot_update.view(orig_shape)
    else:
        qcot_backbone_sd[key] = progenitor_state[key].clone()

qcot_accs = {}
for t in tasks:
    acc = evaluate_model(qcot_backbone_sd, expert_heads[t], t)
    qcot_accs[t] = acc
    print(f"QCOT {t.upper()} Accuracy: {acc:.2f}%")
results['qcot'] = qcot_accs

# --- EXPERIMENT 5: Proposed Holographic Synaptic Alignment (HSA) ---
print("\n=== Running Experiment: Holographic Synaptic Alignment (HSA) ===")
# Build the holographic parameter update
# tau_HSA = 1/sqrt(K) * sum(tau_t * P_t)
K_sqrt = len(tasks) ** 0.5
hsa_backbone_sd = {}
for key in progenitor_state.keys():
    if progenitor_state[key].is_floating_point() and "weight" in key:
        hsa_update = torch.zeros_like(progenitor_state[key])
        for idx, t in enumerate(tasks):
            tau = expert_backbones[t][key] - progenitor_state[key]
            P = get_phase_key(key, progenitor_state[key].shape, idx, device)
            hsa_update += tau * P
        hsa_backbone_sd[key] = progenitor_state[key] + (hsa_update / K_sqrt)
    else:
        hsa_backbone_sd[key] = progenitor_state[key].clone()

# Evaluate HSA on each task after unbinding and DE-BN calibration
hsa_accs = {}
for idx, t in enumerate(tasks):
    # Unbind: W_retrieved = W_init + sqrt(K) * (W_HSA - W_init) * P_t
    retrieved_backbone_sd = {}
    for key in progenitor_state.keys():
        if progenitor_state[key].is_floating_point() and "weight" in key:
            hsa_update_tensor = hsa_backbone_sd[key] - progenitor_state[key]
            P = get_phase_key(key, progenitor_state[key].shape, idx, device)
            retrieved_backbone_sd[key] = progenitor_state[key] + (K_sqrt * hsa_update_tensor * P)
        else:
            retrieved_backbone_sd[key] = hsa_backbone_sd[key].clone()
            
    # Apply DE-BN calibration to clean up crosstalk noise
    cal_backbone = calibrate_model_bn(retrieved_backbone_sd, expert_heads[t], t)
    acc = evaluate_model(cal_backbone, expert_heads[t], t)
    hsa_accs[t] = acc
    print(f"HSA + DE-BN {t.upper()} Accuracy: {acc:.2f}%")
results['hsa'] = hsa_accs


# --- SYSTEMATIC EVALUATION UNDER PTQ QUANTIZATION (8-bit and 4-bit) ---
print("\n=== Running PTQ Quantization Sweeps (FP32, 8-bit, 4-bit, Per-Channel) ===")
ptq_results = {}

for num_bits in [None, 8, 4]:
    bit_name = 'FP32' if num_bits is None else f'INT{num_bits}'
    print(f"\n--- Bitwidth: {bit_name} ---")
    ptq_results[bit_name] = {}
    
    # Evaluate TA (best lambda)
    ta_accs = {}
    for t in tasks:
        ta_accs[t] = evaluate_model(best_ta_backbone, expert_heads[t], t, num_bits, per_channel=True)
    avg_ta = sum(ta_accs.values()) / len(tasks)
    print(f"TA + PTQ: {avg_ta:.2f}%")
    ptq_results[bit_name]['ta'] = ta_accs
    
    # Evaluate TA + DE-BN
    ta_de_bn_accs = {}
    for t in tasks:
        # Calibrate quantized model
        cal_backbone = calibrate_model_bn(best_ta_backbone, expert_heads[t], t, num_bits, per_channel=True)
        ta_de_bn_accs[t] = evaluate_model(cal_backbone, expert_heads[t], t, num_bits, per_channel=True)
    avg_ta_de_bn = sum(ta_de_bn_accs.values()) / len(tasks)
    print(f"TA + DE-BN + PTQ: {avg_ta_de_bn:.2f}%")
    ptq_results[bit_name]['ta_de_bn'] = ta_de_bn_accs
    
    # Evaluate QCOT
    qcot_accs = {}
    for t in tasks:
        qcot_accs[t] = evaluate_model(qcot_backbone_sd, expert_heads[t], t, num_bits, per_channel=True)
    avg_qcot = sum(qcot_accs.values()) / len(tasks)
    print(f"QCOT + PTQ: {avg_qcot:.2f}%")
    ptq_results[bit_name]['qcot'] = qcot_accs
    
    # Evaluate HSA + DE-BN
    hsa_accs = {}
    for idx, t in enumerate(tasks):
        # Retrieved
        retrieved_backbone_sd = {}
        for key in progenitor_state.keys():
            if progenitor_state[key].is_floating_point() and "weight" in key:
                hsa_update_tensor = hsa_backbone_sd[key] - progenitor_state[key]
                P = get_phase_key(key, progenitor_state[key].shape, idx, device)
                retrieved_backbone_sd[key] = progenitor_state[key] + (K_sqrt * hsa_update_tensor * P)
            else:
                retrieved_backbone_sd[key] = hsa_backbone_sd[key].clone()
        # Calibrate quantized
        cal_backbone = calibrate_model_bn(retrieved_backbone_sd, expert_heads[t], t, num_bits, per_channel=True)
        hsa_accs[t] = evaluate_model(cal_backbone, expert_heads[t], t, num_bits, per_channel=True)
    avg_hsa = sum(hsa_accs.values()) / len(tasks)
    print(f"HSA + DE-BN + PTQ: {avg_hsa:.2f}%")
    ptq_results[bit_name]['hsa'] = hsa_accs

results['ptq_sweeps'] = ptq_results


# --- EVALUATION UNDER ENVIRONMENTAL CORRUPTIONS (NOISE & BLUR) ---
print("\n=== Running Robustness Sweeps (Noise and Blur) ===")
robustness_results = {}

for corr in ['noise', 'blur']:
    print(f"\n--- Corruption: {corr.upper()} ---")
    robustness_results[corr] = {}
    
    # Evaluate TA (best lambda)
    ta_accs = {}
    for t in tasks:
        ta_accs[t] = evaluate_model(best_ta_backbone, expert_heads[t], t, corruption=corr)
    avg_ta = sum(ta_accs.values()) / len(tasks)
    print(f"TA: {avg_ta:.2f}%")
    robustness_results[corr]['ta'] = ta_accs
    
    # Evaluate TA + DE-BN
    ta_de_bn_accs = {}
    for t in tasks:
        cal_backbone = calibrate_model_bn(best_ta_backbone, expert_heads[t], t)
        ta_de_bn_accs[t] = evaluate_model(cal_backbone, expert_heads[t], t, corruption=corr)
    avg_ta_de_bn = sum(ta_de_bn_accs.values()) / len(tasks)
    print(f"TA + DE-BN: {avg_ta_de_bn:.2f}%")
    robustness_results[corr]['ta_de_bn'] = ta_de_bn_accs
    
    # Evaluate QCOT
    qcot_accs = {}
    for t in tasks:
        qcot_accs[t] = evaluate_model(qcot_backbone_sd, expert_heads[t], t, corruption=corr)
    avg_qcot = sum(qcot_accs.values()) / len(tasks)
    print(f"QCOT: {avg_qcot:.2f}%")
    robustness_results[corr]['qcot'] = qcot_accs
    
    # Evaluate HSA + DE-BN
    hsa_accs = {}
    for idx, t in enumerate(tasks):
        # Retrieved
        retrieved_backbone_sd = {}
        for key in progenitor_state.keys():
            if progenitor_state[key].is_floating_point() and "weight" in key:
                hsa_update_tensor = hsa_backbone_sd[key] - progenitor_state[key]
                P = get_phase_key(key, progenitor_state[key].shape, idx, device)
                retrieved_backbone_sd[key] = progenitor_state[key] + (K_sqrt * hsa_update_tensor * P)
            else:
                retrieved_backbone_sd[key] = hsa_backbone_sd[key].clone()
        cal_backbone = calibrate_model_bn(retrieved_backbone_sd, expert_heads[t], t)
        hsa_accs[t] = evaluate_model(cal_backbone, expert_heads[t], t, corruption=corr)
    avg_hsa = sum(hsa_accs.values()) / len(tasks)
    print(f"HSA + DE-BN: {avg_hsa:.2f}%")
    robustness_results[corr]['hsa'] = hsa_accs

results['robustness_sweeps'] = robustness_results

# 11. Save Results JSON
with open("experimental_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nSaved experimental results to experimental_results.json")
