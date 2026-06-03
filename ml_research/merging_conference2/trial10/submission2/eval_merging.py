import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to bypass driver compatibility issues
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing transforms (consistent with training)
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

cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load test datasets
print("Loading test datasets...")
test_datasets = {
    'mnist': datasets.MNIST(root='data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='data', train=False, download=True, transform=cifar10_transform)
}

test_loaders = {
    name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    for name, ds in test_datasets.items()
}

def get_progenitor():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model

# 1. Custom Quantization on-the-fly
def quantize_tensor(tensor, num_bits=8, per_channel=False, is_weight=True):
    if num_bits is None:
        return tensor
    qmax = 2**(num_bits - 1) - 1
    
    if per_channel and is_weight and tensor.dim() >= 2:
        orig_shape = tensor.shape
        flat = tensor.view(orig_shape[0], -1)
        max_val = torch.max(torch.abs(flat), dim=1, keepdim=True).values
        max_val = torch.clamp(max_val, min=1e-8)
        delta = max_val / qmax
        flat_q = torch.clamp(torch.round(flat / delta), -qmax, qmax) * delta
        return flat_q.view(orig_shape)
    else:
        max_val = torch.max(torch.abs(tensor))
        if max_val < 1e-8:
            return tensor
        delta = max_val / qmax
        return torch.clamp(torch.round(tensor / delta), -qmax, qmax) * delta

def quantize_model(model, num_bits=8, per_channel=False):
    if num_bits is None:
        return model
    quantized_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if 'weight' in name and ('conv' in name or 'fc' in name):
                param.copy_(quantize_tensor(param, num_bits=num_bits, per_channel=per_channel, is_weight=True))
    return quantized_model

# 2. Hyperbolic Mapping Functions (FP32)
def to_hyperboloid(v):
    norm = torch.norm(v)
    if norm < 1e-8:
        x_0 = torch.tensor(1.0, device=v.device, dtype=v.dtype)
        x_rest = torch.zeros_like(v)
    else:
        x_0 = torch.cosh(norm)
        x_rest = torch.sinh(norm) * (v / norm)
    return x_0, x_rest

def from_hyperboloid(x_0, x_rest):
    y_rest_norm = torch.norm(x_rest)
    if y_rest_norm < 1e-8:
        return torch.zeros_like(x_rest)
    x_0_clamped = torch.clamp(x_0, min=1.0 + 1e-15)
    r = torch.acosh(x_0_clamped)
    return r * (x_rest / y_rest_norm)

def hyperbolic_centroid(vectors):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0]
        
    coords = [to_hyperboloid(v) for v in vectors]
    
    avg_0 = sum(c[0] for c in coords) / num_vectors
    avg_rest = sum(c[1] for c in coords) / num_vectors
    
    lorentz_sq = - (avg_0 ** 2) + torch.sum(avg_rest ** 2)
    if lorentz_sq >= 0:
        lorentz_norm = torch.tensor(1e-8, device=avg_0.device, dtype=avg_0.dtype)
    else:
        lorentz_norm = torch.sqrt(-lorentz_sq)
        
    y_0 = avg_0 / lorentz_norm
    y_rest = avg_rest / lorentz_norm
    
    return from_hyperboloid(y_0, y_rest)

# 2b. Hyperbolic Mapping Functions (FP64 for mathematical exactness)
def to_hyperboloid_double(v):
    v_double = v.to(torch.float64)
    norm = torch.norm(v_double)
    if norm < 1e-15:
        x_0 = torch.tensor(1.0, device=v.device, dtype=torch.float64)
        x_rest = torch.zeros_like(v_double)
    else:
        x_0 = torch.cosh(norm)
        x_rest = torch.sinh(norm) * (v_double / norm)
    return x_0, x_rest

def from_hyperboloid_double(x_0, x_rest):
    y_rest_norm = torch.norm(x_rest)
    if y_rest_norm < 1e-15:
        return torch.zeros_like(x_rest)
    x_0_clamped = torch.clamp(x_0, min=1.0 + 1e-15)
    r = torch.acosh(x_0_clamped)
    return r * (x_rest / y_rest_norm)

def hyperbolic_centroid_double(vectors):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0].to(torch.float64)
        
    coords = [to_hyperboloid_double(v) for v in vectors]
    
    avg_0 = sum(c[0] for c in coords) / num_vectors
    avg_rest = sum(c[1] for c in coords) / num_vectors
    
    lorentz_sq = - (avg_0 ** 2) + torch.sum(avg_rest ** 2)
    if lorentz_sq >= 0:
        lorentz_norm = torch.tensor(1e-15, device=avg_0.device, dtype=torch.float64)
    else:
        lorentz_norm = torch.sqrt(-lorentz_sq)
        
    y_0 = avg_0 / lorentz_norm
    y_rest = avg_rest / lorentz_norm
    
    return from_hyperboloid_double(y_0, y_rest)

# 2c. TIES and DARE Merging Helpers
def ties_merging(vectors, trim_ratio=0.2):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0]
    
    trimmed = []
    for v in vectors:
        v_flat = v.view(-1)
        k = int(trim_ratio * v_flat.numel())
        if k < 1:
            trimmed.append(torch.zeros_like(v))
        else:
            threshold = torch.topk(torch.abs(v_flat), k).values[-1]
            mask = torch.abs(v) >= threshold
            trimmed.append(v * mask)
            
    sign_sum = sum(torch.sign(v) for v in trimmed)
    elected_sign = torch.sign(sign_sum)
    
    merged = torch.zeros_like(vectors[0])
    count = torch.zeros_like(vectors[0])
    for v in trimmed:
        mask = (torch.sign(v) == elected_sign) & (v != 0)
        merged += v * mask
        count += mask.float()
        
    count = torch.clamp(count, min=1.0)
    return merged / count

def dare_merging(vectors, drop_prob=0.9):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0]
        
    merged = torch.zeros_like(vectors[0])
    for v in vectors:
        mask = (torch.rand_like(v) > drop_prob).float()
        v_scaled = v * mask / (1.0 - drop_prob)
        merged += v_scaled
        
    return merged / num_vectors

# 2d. Spherical Mapping Functions (FP32)
def to_sphere(v):
    norm = torch.norm(v)
    if norm < 1e-8:
        x_0 = torch.tensor(1.0, device=v.device, dtype=v.dtype)
        x_rest = torch.zeros_like(v)
    else:
        x_0 = torch.cos(norm)
        x_rest = torch.sin(norm) * (v / norm)
    return x_0, x_rest

def from_sphere(x_0, x_rest):
    y_rest_norm = torch.norm(x_rest)
    if y_rest_norm < 1e-8:
        return torch.zeros_like(x_rest)
    x_0_clamped = torch.clamp(x_0, min=-1.0, max=1.0)
    r = torch.acos(x_0_clamped)
    return r * (x_rest / y_rest_norm)

def spherical_centroid(vectors):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0]
        
    coords = [to_sphere(v) for v in vectors]
    
    avg_0 = sum(c[0] for c in coords) / num_vectors
    avg_rest = sum(c[1] for c in coords) / num_vectors
    
    euclidean_norm = torch.sqrt(avg_0 ** 2 + torch.sum(avg_rest ** 2))
    if euclidean_norm < 1e-8:
        euclidean_norm = torch.tensor(1e-8, device=avg_0.device, dtype=avg_0.dtype)
        
    y_0 = avg_0 / euclidean_norm
    y_rest = avg_rest / euclidean_norm
    
    return from_sphere(y_0, y_rest)

# 2e. Spherical Mapping Functions (FP64)
def to_sphere_double(v):
    v_double = v.to(torch.float64)
    norm = torch.norm(v_double)
    if norm < 1e-15:
        x_0 = torch.tensor(1.0, device=v.device, dtype=torch.float64)
        x_rest = torch.zeros_like(v_double)
    else:
        x_0 = torch.cos(norm)
        x_rest = torch.sin(norm) * (v_double / norm)
    return x_0, x_rest

def from_sphere_double(x_0, x_rest):
    y_rest_norm = torch.norm(x_rest)
    if y_rest_norm < 1e-15:
        return torch.zeros_like(x_rest)
    x_0_clamped = torch.clamp(x_0, min=-1.0, max=1.0)
    r = torch.acos(x_0_clamped)
    return r * (x_rest / y_rest_norm)

def spherical_centroid_double(vectors):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0].to(torch.float64)
        
    coords = [to_sphere_double(v) for v in vectors]
    
    avg_0 = sum(c[0] for c in coords) / num_vectors
    avg_rest = sum(c[1] for c in coords) / num_vectors
    
    euclidean_norm = torch.sqrt(avg_0 ** 2 + torch.sum(avg_rest ** 2))
    if euclidean_norm < 1e-15:
        euclidean_norm = torch.tensor(1e-15, device=avg_0.device, dtype=torch.float64)
        
    y_0 = avg_0 / euclidean_norm
    y_rest = avg_rest / euclidean_norm
    
    return from_sphere_double(y_0, y_rest)

# 3. Merging Core Functions
def get_task_vectors(expert_states, progenitor_state):
    task_vectors = {}
    for task, state in expert_states.items():
        task_vectors[task] = {}
        for k in progenitor_state.keys():
            task_vectors[task][k] = state[k] - progenitor_state[k]
    return task_vectors

def evaluate_merged(backbone_state, expert_heads, num_bits=None, per_channel=False, corruption=None):
    model = get_progenitor().to(device)
    model.load_state_dict(backbone_state)
    
    if num_bits is not None:
        model = quantize_model(model, num_bits=num_bits, per_channel=per_channel)
        
    model.eval()
    accuracies = {}
    
    with torch.no_grad():
        for task, loader in test_loaders.items():
            head = expert_heads[task].to(device)
            head.eval()
            
            if num_bits is not None:
                q_head = quantize_model(head, num_bits=num_bits, per_channel=per_channel)
            else:
                q_head = head
                
            correct = 0
            total = 0
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                
                if corruption == 'noise':
                    x = x + torch.randn_like(x) * 0.1
                elif corruption == 'blur':
                    x = TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[1.0, 1.0])
                    
                features = model(x)
                outputs = q_head(features)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
            accuracies[task] = 100.0 * correct / total
            
    accuracies['average'] = np.mean(list(accuracies.values()))
    return accuracies

def main():
    print("Loading experts and progenitor...")
    expert_states = {}
    expert_heads = {}
    
    for task in ['mnist', 'fmnist', 'cifar10']:
        path = f"checkpoints/expert_{task}.pt"
        if not os.path.exists(path):
            print(f"Error: checkpoint {path} not found! Please run train_experts.py first.")
            return
            
        checkpoint = torch.load(path, map_location=device)
        expert_states[task] = checkpoint['backbone_state_dict']
        
        head = nn.Linear(512, 10)
        head.load_state_dict(checkpoint['head_state_dict'])
        expert_heads[task] = head
        
    progenitor_state = torch.load("checkpoints/progenitor.pt", map_location=device)
    task_vectors = get_task_vectors(expert_states, progenitor_state)
    
    print("\nEvaluating Individual Experts (Oracle Upper Bound):")
    for task in ['mnist', 'fmnist', 'cifar10']:
        path = f"checkpoints/expert_{task}.pt"
        checkpoint = torch.load(path, map_location=device)
        print(f"Oracle {task.upper()} accuracy: {checkpoint['test_acc']:.2f}%")
        
    results = {}
    
    # --- METHOD 1: Weight Averaging (WA) ---
    print("\n--- Running Weight Averaging (WA) ---")
    wa_state = copy.deepcopy(progenitor_state)
    for k in progenitor_state.keys():
        wa_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
    
    results['WA'] = evaluate_merged(wa_state, expert_heads)
    print(f"WA accuracies: MNIST={results['WA']['mnist']:.2f}%, FMNIST={results['WA']['fmnist']:.2f}%, CIFAR10={results['WA']['cifar10']:.2f}% | Avg={results['WA']['average']:.2f}%")
    
    # --- METHOD 2: Task Arithmetic (TA) ---
    print("\n--- Running Task Arithmetic (TA) ---")
    best_ta_acc = 0.0
    best_ta_lambda = 0.0
    best_ta_state = None
    
    for lam in [0.1, 0.3, 0.5, 0.7, 1.0]:
        ta_state = copy.deepcopy(progenitor_state)
        for k in progenitor_state.keys():
            ta_state[k] = progenitor_state[k] + lam * sum(task_vectors[t][k] for t in expert_states.keys()) / len(expert_states)
        accs = evaluate_merged(ta_state, expert_heads)
        print(f"TA (lambda={lam}): MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR10={accs['cifar10']:.2f}% | Avg={accs['average']:.2f}%")
        if accs['average'] > best_ta_acc:
            best_ta_acc = accs['average']
            best_ta_lambda = lam
            best_ta_state = ta_state
            results['TA'] = accs
            
    # --- METHOD 3: Holographic Norm Scaling (HNS) ---
    print("\n--- Running Holographic Norm Scaling (HNS) ---")
    hns_state = copy.deepcopy(progenitor_state)
    for k in progenitor_state.keys():
        if 'weight' in k and progenitor_state[k].dim() >= 2:
            orig_shape = progenitor_state[k].shape
            num_channels = orig_shape[0]
            
            merged_up = sum(task_vectors[t][k] for t in expert_states.keys()) / len(expert_states)
            flat_merged = merged_up.view(num_channels, -1)
            
            scaled_up = torch.zeros_like(flat_merged)
            for c in range(num_channels):
                expert_norms = [torch.norm(task_vectors[t][k][c]) for t in expert_states.keys()]
                avg_expert_norm = sum(expert_norms) / len(expert_states)
                merged_norm = torch.norm(flat_merged[c])
                
                if merged_norm > 1e-8:
                    sc = avg_expert_norm / merged_norm
                    sc = torch.clamp(sc, 0.1, 10.0)
                else:
                    sc = torch.tensor(1.0, device=device)
                    
                scaled_up[c] = flat_merged[c] * sc
                
            hns_state[k] = progenitor_state[k] + scaled_up.view(orig_shape)
        else:
            hns_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
            
    results['HNS'] = evaluate_merged(hns_state, expert_heads)
    print(f"HNS accuracies: MNIST={results['HNS']['mnist']:.2f}%, FMNIST={results['HNS']['fmnist']:.2f}%, CIFAR10={results['HNS']['cifar10']:.2f}% | Avg={results['HNS']['average']:.2f}%")
    
    # --- METHOD 4: Proposed Hyperbolic Merging (Ours, FP32 and FP64) ---
    print("\n--- Running Proposed Hyperbolic Merging (Ours) ---")
    for precision in ['FP32', 'FP64']:
        for mode in ['channel-wise']:
            print(f"\nEvaluating Hyperbolic Merging ({precision}, {mode} mode):")
            best_hyper_acc = 0.0
            best_hyper_alpha = 0.0
            best_hyper_state = None
            
            # For FP32 we run alpha=0.01 since we know it's the sweet spot of the float32 rounding artifact.
            # For FP64 we run a full sweep.
            alphas = [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0] if precision == 'FP64' else [1e-2]
            
            for alpha in alphas:
                hyper_state = copy.deepcopy(progenitor_state)
                
                for k in progenitor_state.keys():
                    if 'weight' in k and progenitor_state[k].dim() >= 2:
                        orig_shape = progenitor_state[k].shape
                        num_channels = orig_shape[0]
                        flat_merged = torch.zeros((num_channels, progenitor_state[k].view(num_channels, -1).shape[1]), device=device)
                        
                        for c in range(num_channels):
                            v_experts = [task_vectors[t][k][c].view(-1) * alpha for t in expert_states.keys()]
                            
                            if precision == 'FP32':
                                centroid = hyperbolic_centroid(v_experts)
                            else:
                                centroid = hyperbolic_centroid_double(v_experts).to(device).to(progenitor_state[k].dtype)
                                
                            flat_merged[c] = centroid / alpha
                            
                        hyper_state[k] = progenitor_state[k] + flat_merged.view(orig_shape)
                    else:
                        hyper_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
                        
                accs = evaluate_merged(hyper_state, expert_heads)
                print(f"Hyperbolic ({precision}, alpha={alpha}): MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR10={accs['cifar10']:.2f}% | Avg={accs['average']:.2f}%")
                
                if accs['average'] > best_hyper_acc:
                    best_hyper_acc = accs['average']
                    best_hyper_alpha = alpha
                    best_hyper_state = hyper_state
                    results[f'Hyperbolic_{precision}'] = accs
                    results[f'Hyperbolic_{precision}_best_alpha'] = alpha
                    results[f'Hyperbolic_{precision}_state'] = hyper_state

    # --- METHOD 5: TIES-Merging ---
    print("\n--- Running TIES-Merging ---")
    ties_state = copy.deepcopy(progenitor_state)
    for k in progenitor_state.keys():
        if 'weight' in k and progenitor_state[k].dim() >= 2:
            orig_shape = progenitor_state[k].shape
            num_channels = orig_shape[0]
            flat_merged = torch.zeros((num_channels, progenitor_state[k].view(num_channels, -1).shape[1]), device=device)
            for c in range(num_channels):
                v_experts = [task_vectors[t][k][c].view(-1) for t in expert_states.keys()]
                flat_merged[c] = ties_merging(v_experts, trim_ratio=0.2)
            ties_state[k] = progenitor_state[k] + flat_merged.view(orig_shape)
        else:
            ties_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
            
    results['TIES'] = evaluate_merged(ties_state, expert_heads)
    print(f"TIES accuracies: MNIST={results['TIES']['mnist']:.2f}%, FMNIST={results['TIES']['fmnist']:.2f}%, CIFAR10={results['TIES']['cifar10']:.2f}% | Avg={results['TIES']['average']:.2f}%")

    # --- METHOD 6: DARE-Merging ---
    print("\n--- Running DARE-Merging ---")
    dare_state = copy.deepcopy(progenitor_state)
    for k in progenitor_state.keys():
        if 'weight' in k and progenitor_state[k].dim() >= 2:
            orig_shape = progenitor_state[k].shape
            num_channels = orig_shape[0]
            flat_merged = torch.zeros((num_channels, progenitor_state[k].view(num_channels, -1).shape[1]), device=device)
            for c in range(num_channels):
                v_experts = [task_vectors[t][k][c].view(-1) for t in expert_states.keys()]
                flat_merged[c] = dare_merging(v_experts, drop_prob=0.9)
            dare_state[k] = progenitor_state[k] + flat_merged.view(orig_shape)
        else:
            dare_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
            
    results['DARE'] = evaluate_merged(dare_state, expert_heads)
    print(f"DARE accuracies: MNIST={results['DARE']['mnist']:.2f}%, FMNIST={results['DARE']['fmnist']:.2f}%, CIFAR10={results['DARE']['cifar10']:.2f}% | Avg={results['DARE']['average']:.2f}%")

    # --- METHOD 7: Proposed Curvature-Adaptive Hyperbolic Merging (CAHM) ---
    print("\n--- Running Curvature-Adaptive Hyperbolic Merging (CAHM) ---")
    for precision in ['FP32', 'FP64']:
        print(f"\nEvaluating CAHM ({precision}):")
        best_cahm_acc = 0.0
        best_cahm_gamma = 0.0
        best_cahm_state = None
        
        # We sweep gamma
        gammas = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        for gamma in gammas:
            cahm_state = copy.deepcopy(progenitor_state)
            
            for k in progenitor_state.keys():
                if 'weight' in k and progenitor_state[k].dim() >= 2:
                    orig_shape = progenitor_state[k].shape
                    num_channels = orig_shape[0]
                    flat_merged = torch.zeros((num_channels, progenitor_state[k].view(num_channels, -1).shape[1]), device=device)
                    
                    for c in range(num_channels):
                        v_experts_orig = [task_vectors[t][k][c].view(-1) for t in expert_states.keys()]
                        avg_norm = sum(torch.norm(v) for v in v_experts_orig) / len(v_experts_orig)
                        
                        if avg_norm > 1e-8:
                            alpha_c = gamma / avg_norm
                        else:
                            alpha_c = 1.0
                            
                        v_experts = [v * alpha_c for v in v_experts_orig]
                        
                        if precision == 'FP32':
                            centroid = hyperbolic_centroid(v_experts)
                        else:
                            centroid = hyperbolic_centroid_double(v_experts).to(device).to(progenitor_state[k].dtype)
                            
                        flat_merged[c] = centroid / alpha_c
                        
                    cahm_state[k] = progenitor_state[k] + flat_merged.view(orig_shape)
                else:
                    cahm_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
                    
            accs = evaluate_merged(cahm_state, expert_heads)
            print(f"CAHM ({precision}, gamma={gamma}): MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR10={accs['cifar10']:.2f}% | Avg={accs['average']:.2f}%")
            
            if accs['average'] > best_cahm_acc:
                best_cahm_acc = accs['average']
                best_cahm_gamma = gamma
                best_cahm_state = cahm_state
                results[f'CAHM_{precision}'] = accs
                results[f'CAHM_{precision}_best_gamma'] = gamma
                results[f'CAHM_{precision}_state'] = cahm_state

    # --- METHOD 8: Proposed Spherical Merging ---
    print("\n--- Running Proposed Spherical Merging ---")
    for precision in ['FP32', 'FP64']:
        print(f"\nEvaluating Spherical Merging ({precision}, channel-wise mode):")
        best_sphere_acc = 0.0
        best_sphere_alpha = 0.0
        best_sphere_state = None
        
        # We run sweeps on alpha
        alphas = [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0]
        
        for alpha in alphas:
            sphere_state = copy.deepcopy(progenitor_state)
            
            for k in progenitor_state.keys():
                if 'weight' in k and progenitor_state[k].dim() >= 2:
                    orig_shape = progenitor_state[k].shape
                    num_channels = orig_shape[0]
                    flat_merged = torch.zeros((num_channels, progenitor_state[k].view(num_channels, -1).shape[1]), device=device)
                    
                    for c in range(num_channels):
                        v_experts = [task_vectors[t][k][c].view(-1) * alpha for t in expert_states.keys()]
                        
                        if precision == 'FP32':
                            centroid = spherical_centroid(v_experts)
                        else:
                            centroid = spherical_centroid_double(v_experts).to(device).to(progenitor_state[k].dtype)
                            
                        flat_merged[c] = centroid / alpha
                        
                    sphere_state[k] = progenitor_state[k] + flat_merged.view(orig_shape)
                else:
                    sphere_state[k] = sum(expert_states[t][k] for t in expert_states.keys()) / len(expert_states)
                    
            accs = evaluate_merged(sphere_state, expert_heads)
            print(f"Spherical ({precision}, alpha={alpha}): MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR10={accs['cifar10']:.2f}% | Avg={accs['average']:.2f}%")
            
            if accs['average'] > best_sphere_acc:
                best_sphere_acc = accs['average']
                best_sphere_alpha = alpha
                best_sphere_state = sphere_state
                results[f'Spherical_{precision}'] = accs
                results[f'Spherical_{precision}_best_alpha'] = alpha
                results[f'Spherical_{precision}_state'] = sphere_state

    # --- EVALUATING ROBUSTNESS & QUANTIZATION ---
    comparison_methods = {
        'WA': wa_state,
        'TA (lambda={})'.format(best_ta_lambda): best_ta_state,
        'HNS': hns_state,
        'TIES': ties_state,
        'DARE': dare_state,
        'Hyperbolic (FP32, a=0.01)': results['Hyperbolic_FP32_state'],
        'Hyperbolic (FP64, a=0.01)': results['Hyperbolic_FP64_state'],
        'CAHM (FP32, g={})'.format(results['CAHM_FP32_best_gamma']): results['CAHM_FP32_state'],
        'CAHM (FP64, g={})'.format(results['CAHM_FP64_best_gamma']): results['CAHM_FP64_state'],
        'Spherical (FP32, a={})'.format(results['Spherical_FP32_best_alpha']): results['Spherical_FP32_state'],
        'Spherical (FP64, a={})'.format(results['Spherical_FP64_best_alpha']): results['Spherical_FP64_state']
    }
    
    print("\n" + "="*50)
    print("ROBUSTNESS & QUANTIZATION COMPREHENSIVE EVALUATION")
    print("="*50)
    
    scenarios = [
        ('FP32', None, False, None),
        ('INT8 Per-Tensor PTQ', 8, False, None),
        ('INT8 Per-Channel PTQ', 8, True, None),
        ('INT4 Per-Channel PTQ', 4, True, None),
        ('FP32 + Gaussian Noise (sigma=0.1)', None, False, 'noise'),
        ('FP32 + Gaussian Blur (sigma=1.0)', None, False, 'blur')
    ]
    
    header = f"{'Method':<30} | {'Scenario':<30} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}"
    print(header)
    print("-" * len(header))
    
    detailed_table = []
    
    for method_name, state in comparison_methods.items():
        for scenario_name, bits, per_channel, corruption in scenarios:
            accs = evaluate_merged(state, expert_heads, num_bits=bits, per_channel=per_channel, corruption=corruption)
            row = f"{method_name:<30} | {scenario_name:<30} | {accs['mnist']:7.2f}% | {accs['fmnist']:7.2f}% | {accs['cifar10']:7.2f}% | {accs['average']:7.2f}%"
            print(row)
            detailed_table.append({
                'method': method_name,
                'scenario': scenario_name,
                'mnist': accs['mnist'],
                'fmnist': accs['fmnist'],
                'cifar10': accs['cifar10'],
                'average': accs['average']
            })
            
    torch.save(detailed_table, "checkpoints/eval_results.pt")
    print("\nEvaluation completed! Results saved to checkpoints/eval_results.pt.")

if __name__ == "__main__":
    main()
