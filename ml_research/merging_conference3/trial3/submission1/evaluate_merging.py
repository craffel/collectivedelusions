import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
from torch.func import functional_call
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("results", exist_ok=True)

# ImageNet normalization transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_grayscale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
def get_dataset(name, train=True):
    if name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform_grayscale)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform_grayscale)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform_rgb)
    elif name == "svhn":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root="./data", split=split, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Map parameter names to L=14 layer groups
def get_layer_group(name):
    if "patch_embed" in name:
        return 0
    elif "blocks" in name:
        parts = name.split(".")
        block_idx = int(parts[1])
        return block_idx + 1
    elif "norm" in name:
        return 13
    else:
        return 0

# --- Quantization Operators ---

def quantize_sym(W, num_bits, per_channel=False):
    if num_bits >= 16:
        return W
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    if per_channel and W.dim() > 1:
        max_val = W.abs().flatten(1).max(dim=1)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        scale = max_val / qmax
        scale = scale.view(-1, *([1] * (W.dim() - 1)))
    else:
        max_val = W.abs().max()
        scale = max_val / qmax
        scale = torch.clamp(scale, min=1e-8)
        
    W_scaled = W / scale
    W_quant = W_scaled + (torch.clamp(torch.round(W_scaled), qmin, qmax) - W_scaled).detach()
    return W_quant * scale

def quantize_asym(W, num_bits, per_channel=False):
    if num_bits >= 16:
        return W
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    if per_channel and W.dim() > 1:
        min_val = W.flatten(1).min(dim=1)[0]
        max_val = W.flatten(1).max(dim=1)[0]
        scale = (max_val - min_val) / (2**num_bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        
        zp = torch.round(-min_val / scale) - (2**(num_bits - 1))
        zp = torch.clamp(zp, qmin, qmax)
        
        scale = scale.view(-1, *([1] * (W.dim() - 1)))
        zp = zp.view(-1, *([1] * (W.dim() - 1)))
    else:
        min_val = W.min()
        max_val = W.max()
        scale = (max_val - min_val) / (2**num_bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        zp = torch.round(-min_val / scale) - (2**(num_bits - 1))
        zp = torch.clamp(zp, qmin, qmax)
        
    W_scaled = W / scale + zp
    W_quant = W_scaled + (torch.clamp(torch.round(W_scaled), qmin, qmax) - W_scaled).detach()
    return (W_quant - zp) * scale

def quantize_double(W, num_bits):
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    if W.dim() > 1:
        max_val = W.abs().flatten(1).max(dim=1)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        scale = max_val / qmax
        
        # Second-stage 8-bit quantization on scales
        scale_qmax = 127
        scale_max = scale.max()
        scale_scale = scale_max / scale_qmax
        scale_scale = torch.clamp(scale_scale, min=1e-8)
        
        scale_scaled = scale / scale_scale
        scale_quant = scale_scaled + (torch.clamp(torch.round(scale_scaled), -128, 127) - scale_scaled).detach()
        scale_quantized = scale_quant * scale_scale
        
        scale_reshaped = scale_quantized.view(-1, *([1] * (W.dim() - 1)))
    else:
        max_val = W.abs().max()
        scale_reshaped = torch.clamp(max_val / qmax, min=1e-8)
        
    W_scaled = W / scale_reshaped
    W_quant = W_scaled + (torch.clamp(torch.round(W_scaled), qmin, qmax) - W_scaled).detach()
    return W_quant * scale_reshaped

def quantize_tensor(W, schema, num_bits):
    if schema == 'none' or num_bits >= 16:
        return W
    elif schema == 'sym_tensor':
        return quantize_sym(W, num_bits, per_channel=False)
    elif schema == 'sym_channel':
        return quantize_sym(W, num_bits, per_channel=True)
    elif schema == 'asym_tensor':
        return quantize_asym(W, num_bits, per_channel=False)
    elif schema == 'asym_channel':
        return quantize_asym(W, num_bits, per_channel=True)
    elif schema == 'double_quant':
        return quantize_double(W, num_bits)
    else:
        raise ValueError(f"Unknown schema: {schema}")

# --- Merging Logic ---

def get_merged_params(Lambda, pre_dict, expert_dicts, K, L, schema='none', num_bits=16):
    merged = {}
    for name in pre_dict.keys():
        l = get_layer_group(name)
        delta = torch.stack([expert_dicts[k][name] - pre_dict[name] for k in range(K)], dim=0)
        coeff = Lambda[:, l].view(K, *([1] * (delta.dim() - 1)))
        W_merged = pre_dict[name] + (delta * coeff).sum(dim=0)
        
        # Apply quantization to merged parameters
        merged[name] = quantize_tensor(W_merged, schema, num_bits)
    return merged

def forward_pass(backbone_model, backbone_params, head_params, images):
    params = {**backbone_params, "head.weight": head_params["weight"], "head.bias": head_params["bias"]}
    return functional_call(backbone_model, params, images)

# --- Calibration Dataloader Construction ---

def build_calibration_loader(datasets_list, N, batch_size=32, corrupt=False, skew=False):
    loaders = {}
    for name, full_dataset in datasets_list.items():
        # Handle label skew / class imbalance
        if skew:
            # Pick a dominant class (e.g. class 0) which gets 80% of samples
            targets = np.array(full_dataset.targets if hasattr(full_dataset, 'targets') else full_dataset.labels)
            dom_class_indices = np.where(targets == 0)[0]
            other_indices = np.where(targets != 0)[0]
            
            n_dom = int(N * 0.8)
            n_oth = N - n_dom
            
            # Draw indices
            selected_dom = np.random.choice(dom_class_indices, n_dom, replace=True).tolist()
            selected_oth = np.random.choice(other_indices, n_oth, replace=True).tolist()
            indices = selected_dom + selected_oth
        else:
            indices = torch.randperm(len(full_dataset))[:N].tolist()
            
        subset = Subset(full_dataset, indices)
        
        # Optionally inject OOD corruptions (Gaussian noise)
        if corrupt:
            # We can define a wrapper dataset or transform
            class CorruptedDataset(torch.utils.data.Dataset):
                def __init__(self, base_subset):
                    self.base_subset = base_subset
                def __len__(self):
                    return len(self.base_subset)
                def __getitem__(self, idx):
                    img, label = self.base_subset[idx]
                    # Add heavy Gaussian noise
                    noise = torch.randn_like(img) * 0.4
                    return torch.clamp(img + noise, -2.5, 2.5), label
            subset = CorruptedDataset(subset)
            
        loaders[name] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loaders

# --- True Evaluation Function ---

def evaluate_accuracy(backbone_model, Lambda, pre_dict, expert_dicts, heads, test_loaders, schema, num_bits, K, L):
    # Assemble the merged, quantized parameters once for evaluation
    with torch.no_grad():
        merged_params = get_merged_params(Lambda, pre_dict, expert_dicts, K, L, schema, num_bits)
    
    accuracies = {}
    for name, loader in test_loaders.items():
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = forward_pass(backbone_model, merged_params, heads[name], images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracies[name] = 100.0 * correct / total
    accuracies["avg"] = sum(accuracies.values()) / len(accuracies)
    return accuracies

# --- Optimization Routines ---

def run_ste_optimization(pre_dict, expert_dicts, heads, cal_loaders, test_loaders, backbone_model, K, L, 
                         schema='sym_channel', num_bits=4, steps=40, lr=0.01, 
                         reg_type='none', alpha=0.1):
    # Initialize Lambda as parameter
    Lambda = nn.Parameter(torch.full((K, L), 0.3, device=device))
    optimizer = optim.Adam([Lambda], lr=lr)
    
    # Track metrics
    losses = []
    
    for step in range(steps):
        # We process a single batch per task jointly
        optimizer.zero_grad()
        
        # Assemble merged and quantized parameters
        merged_params = get_merged_params(Lambda, pre_dict, expert_dicts, K, L, schema, num_bits)
        
        total_entropy_loss = 0.0
        
        # Joint multi-task entropy minimization
        for name, loader in cal_loaders.items():
            try:
                images, _ = next(iter(loader))
            except StopIteration:
                continue
            images = images.to(device)
            
            outputs = forward_pass(backbone_model, merged_params, heads[name], images)
            probs = torch.softmax(outputs, dim=-1)
            entropy = - torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            total_entropy_loss += entropy
            
        loss = total_entropy_loss
        
        # Apply spatial/structural regularization
        if reg_type == 'tv' or reg_type == 'esr':
            # Total Variation (smooth adjacent layers)
            tv_reg = 0.0
            for k in range(K):
                for l in range(L - 1):
                    tv_reg += (Lambda[k, l+1] - Lambda[k, l]).pow(2)
            loss += alpha * tv_reg
        elif reg_type == 'l2':
            # Regularize to start value of 0.3
            l2_reg = (Lambda - 0.3).pow(2).sum()
            loss += alpha * l2_reg
            
        loss.backward()
        optimizer.step()
        
        # Clamp Lambda to valid [0, 1] range
        with torch.no_grad():
            Lambda.clamp_(0.0, 1.0)
            
        losses.append(loss.item())
        
    return Lambda.detach(), losses

def run_es_optimization(pre_dict, expert_dicts, heads, cal_loaders, test_loaders, backbone_model, K, L,
                        schema='sym_channel', num_bits=4, steps=40, sigma=0.02):
    # Derivative-free 1+1 Evolution Strategy
    # Parent candidate
    Lambda = torch.full((K, L), 0.3, device=device)
    
    def evaluate_loss(cand):
        total_entropy_loss = 0.0
        with torch.no_grad():
            merged_params = get_merged_params(cand, pre_dict, expert_dicts, K, L, schema, num_bits)
            for name, loader in cal_loaders.items():
                try:
                    images, _ = next(iter(loader))
                except StopIteration:
                    continue
                images = images.to(device)
                outputs = forward_pass(backbone_model, merged_params, heads[name], images)
                probs = torch.softmax(outputs, dim=-1)
                entropy = - torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                total_entropy_loss += entropy.item()
        return total_entropy_loss
        
    parent_loss = evaluate_loss(Lambda)
    
    for step in range(steps):
        # Mutate
        noise = torch.randn_like(Lambda) * sigma
        mutant = torch.clamp(Lambda + noise, 0.0, 1.0)
        
        mutant_loss = evaluate_loss(mutant)
        if mutant_loss < parent_loss:
            Lambda = mutant
            parent_loss = mutant_loss
            
    return Lambda, [parent_loss]

# --- Main Audit Execution ---

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    K = 4
    L = 14
    task_names = ["mnist", "fmnist", "cifar10", "svhn"]
    
    # 1. Load pre-trained backbone and experts
    print("\nLoading models and checkpoints...")
    pre_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10).to(device)
    pre_dict = {k: v.to(device) for k, v in pre_model.state_dict().items() if not k.startswith("head")}
    
    expert_dicts = {}
    heads = {}
    for name in task_names:
        m = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
        m.load_state_dict(torch.load(f"checkpoints/expert_{name}.pth", map_location=device))
        m_dict = m.state_dict()
        expert_dicts[name] = {k: v.to(device) for k, v in m_dict.items() if not k.startswith("head")}
        heads[name] = {
            "weight": m_dict["head.weight"].to(device),
            "bias": m_dict["head.bias"].to(device)
        }
        
    # Convert expert_dicts dict-of-dicts to indexed-by-idx list
    expert_dicts_list = [expert_dicts[name] for name in task_names]
    
    # 2. Setup datasets
    print("Preparing train-calibration and test datasets...")
    train_datasets = {name: get_dataset(name, train=True) for name in task_names}
    test_datasets = {name: get_dataset(name, train=False) for name in task_names}
    
    # Build fast test loaders
    test_loaders = {}
    for name, test_ds in test_datasets.items():
        indices = torch.randperm(len(test_ds))[:200].tolist()  # 200 samples per task
        subset = Subset(test_ds, indices)
        test_loaders[name] = DataLoader(subset, batch_size=64, shuffle=False)
        
    # Result container to save to metrics.json
    results_db = {}
    
    # --- AXIS 1: Calibration Sweep & Baselines ---
    print("\n--- AXIS 1: Calibration Sweep and Baselines (4-bit, Symmetric Per-Channel) ---")
    results_db["axis1"] = {}
    
    # Baselines
    # FP16 Task Arithmetic
    uniform_Lambda = torch.full((K, L), 0.3, device=device)
    acc_fp16_ta = evaluate_accuracy(pre_model, uniform_Lambda, pre_dict, expert_dicts_list, heads, test_loaders, 'none', 16, K, L)
    print(f"FP16 Task Arithmetic Accuracy: {acc_fp16_ta['avg']:.2f}% (MNIST={acc_fp16_ta['mnist']:.1f}, FMNIST={acc_fp16_ta['fmnist']:.1f}, CIFAR10={acc_fp16_ta['cifar10']:.1f}, SVHN={acc_fp16_ta['svhn']:.1f})")
    results_db["axis1"]["fp16_task_arithmetic"] = acc_fp16_ta
    
    # Naive Merge-then-Quantize (M-then-Q 4-bit)
    acc_m_then_q_4bit = evaluate_accuracy(pre_model, uniform_Lambda, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    print(f"Naive 4-bit M-then-Q Accuracy: {acc_m_then_q_4bit['avg']:.2f}% (MNIST={acc_m_then_q_4bit['mnist']:.1f}, FMNIST={acc_m_then_q_4bit['fmnist']:.1f}, CIFAR10={acc_m_then_q_4bit['cifar10']:.1f}, SVHN={acc_m_then_q_4bit['svhn']:.1f})")
    results_db["axis1"]["m_then_q_4bit"] = acc_m_then_q_4bit
    
    # Quantized AdaMerging (N=16, FP16 optimized, then 4-bit Sym Channel evaluated)
    print("Optimizing unquantized AdaMerging baseline (N=16, FP16)...")
    cal_loaders_n16 = build_calibration_loader(train_datasets, N=16, batch_size=32)
    Lambda_adamerge, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, cal_loaders_n16, test_loaders, pre_model, K, L,
                                              schema='none', num_bits=16, steps=40, lr=0.01)
    acc_quant_adamerge = evaluate_accuracy(pre_model, Lambda_adamerge, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    print(f"Quantized AdaMerging Accuracy: {acc_quant_adamerge['avg']:.2f}% (MNIST={acc_quant_adamerge['mnist']:.1f}, FMNIST={acc_quant_adamerge['fmnist']:.1f}, CIFAR10={acc_quant_adamerge['cifar10']:.1f}, SVHN={acc_quant_adamerge['svhn']:.1f})")
    results_db["axis1"]["quant_adamerge"] = acc_quant_adamerge

    # Sweep learning rates for Q-Merge under N=16 to see if smaller lr stabilizes STE
    results_db["axis1"]["lr_sweep"] = {}
    for lr_val in [0.001, 0.0001]:
        print(f"Running Q-Merge sweep with lr={lr_val}...")
        Lambda_lr, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, cal_loaders_n16, test_loaders, pre_model, K, L,
                                             schema='sym_channel', num_bits=4, steps=100, lr=lr_val)
        acc_lr = evaluate_accuracy(pre_model, Lambda_lr, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
        print(f"Q-Merge with lr={lr_val} (100 steps): {acc_lr['avg']:.2f}% (MNIST={acc_lr['mnist']:.1f}, FMNIST={acc_lr['fmnist']:.1f}, CIFAR10={acc_lr['cifar10']:.1f}, SVHN={acc_lr['svhn']:.1f})")
        results_db["axis1"]["lr_sweep"][f"lr_{lr_val}"] = acc_lr
    
    # Sweep N
    results_db["axis1"]["qmerge_sweep"] = {}
    for N in [1, 4, 16, 64]:
        cal_loaders = build_calibration_loader(train_datasets, N=N, batch_size=32)
        
        # Optimize with Q-Merge (STE under sym_channel 4-bit)
        Lambda_opt, losses = run_ste_optimization(pre_dict, expert_dicts_list, heads, cal_loaders, test_loaders, pre_model, K, L,
                                                  schema='sym_channel', num_bits=4, steps=40, lr=0.01)
        acc_qmerge = evaluate_accuracy(pre_model, Lambda_opt, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
        print(f"Q-Merge 4-bit (N={N}) Accuracy: {acc_qmerge['avg']:.2f}% (MNIST={acc_qmerge['mnist']:.1f}, FMNIST={acc_qmerge['fmnist']:.1f}, CIFAR10={acc_qmerge['cifar10']:.1f}, SVHN={acc_qmerge['svhn']:.1f})")
        results_db["axis1"]["qmerge_sweep"][f"N_{N}"] = acc_qmerge
        
    # --- AXIS 2: Cross-Schema Generalization Audit ---
    print("\n--- AXIS 2: Cross-Schema Generalization Audit (N=16, 4-bit) ---")
    results_db["axis2"] = {}
    
    schemas = ['sym_tensor', 'sym_channel', 'asym_tensor', 'asym_channel', 'double_quant']
    fixed_cal_loaders = build_calibration_loader(train_datasets, N=16, batch_size=32)
    
    # We will build a 5x5 grid of Source Opt Schema -> Target Eval Schema
    for src_schema in ['sym_tensor', 'sym_channel', 'asym_tensor', 'asym_channel']:
        print(f"Optimizing coefficients under source schema: {src_schema}")
        # Optimize
        Lambda_opt, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, fixed_cal_loaders, test_loaders, pre_model, K, L,
                                              schema=src_schema, num_bits=4, steps=40, lr=0.01)
        
        results_db["axis2"][src_schema] = {}
        # Evaluate under all target schemas
        for tgt_schema in schemas:
            acc_eval = evaluate_accuracy(pre_model, Lambda_opt, pre_dict, expert_dicts_list, heads, test_loaders, tgt_schema, 4, K, L)
            print(f"  -> Evaluated under target {tgt_schema}: {acc_eval['avg']:.2f}%")
            results_db["axis2"][src_schema][tgt_schema] = acc_eval
            
    # --- AXIS 3: Regularization Analysis & Black-Box ES ---
    print("\n--- AXIS 3: Regularization Analysis (N=16, Source=sym_channel, Target=sym_tensor) ---")
    results_db["axis3"] = {}
    
    # 1. Unregularized STE (we already did this, but let's run it specifically)
    Lambda_unreg, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, fixed_cal_loaders, test_loaders, pre_model, K, L,
                                           schema='sym_channel', num_bits=4, steps=40, lr=0.01, reg_type='none')
    acc_unreg_src = evaluate_accuracy(pre_model, Lambda_unreg, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    acc_unreg_tgt = evaluate_accuracy(pre_model, Lambda_unreg, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_tensor', 4, K, L)
    print(f"Unregularized STE: Source (sym_channel) = {acc_unreg_src['avg']:.2f}% | Target (sym_tensor) = {acc_unreg_tgt['avg']:.2f}%")
    results_db["axis3"]["unregularized"] = {"source": acc_unreg_src, "target": acc_unreg_tgt}
    
    # 2. Spatial Regularized STE (TV regularization)
    Lambda_tv, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, fixed_cal_loaders, test_loaders, pre_model, K, L,
                                        schema='sym_channel', num_bits=4, steps=40, lr=0.01, reg_type='tv', alpha=0.5)
    acc_tv_src = evaluate_accuracy(pre_model, Lambda_tv, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    acc_tv_tgt = evaluate_accuracy(pre_model, Lambda_tv, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_tensor', 4, K, L)
    print(f"TV Regularized STE: Source (sym_channel) = {acc_tv_src['avg']:.2f}% | Target (sym_tensor) = {acc_tv_tgt['avg']:.2f}%")
    results_db["axis3"]["spatial_regularized"] = {"source": acc_tv_src, "target": acc_tv_tgt}
    
    # 3. Black-Box 1+1 ES
    Lambda_es, _ = run_es_optimization(pre_dict, expert_dicts_list, heads, fixed_cal_loaders, test_loaders, pre_model, K, L,
                                       schema='sym_channel', num_bits=4, steps=40, sigma=0.02)
    acc_es_src = evaluate_accuracy(pre_model, Lambda_es, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    acc_es_tgt = evaluate_accuracy(pre_model, Lambda_es, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_tensor', 4, K, L)
    print(f"Derivative-Free 1+1 ES: Source (sym_channel) = {acc_es_src['avg']:.2f}% | Target (sym_tensor) = {acc_es_tgt['avg']:.2f}%")
    results_db["axis3"]["blackbox_es"] = {"source": acc_es_src, "target": acc_es_tgt}
    
    # --- AXIS 4: Data Corruption and Skew Robustness ---
    print("\n--- AXIS 4: Stream Corruption and Class Imbalance Sweep (N=16) ---")
    results_db["axis4"] = {}
    
    # 1. Standard (No shift, baseline for comparison)
    results_db["axis4"]["clean"] = results_db["axis1"]["qmerge_sweep"]["N_16"]
    
    # 2. Corrupted stream (Gaussian noise)
    corrupted_loaders = build_calibration_loader(train_datasets, N=16, batch_size=32, corrupt=True)
    Lambda_corr, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, corrupted_loaders, test_loaders, pre_model, K, L,
                                          schema='sym_channel', num_bits=4, steps=40, lr=0.01)
    acc_corr = evaluate_accuracy(pre_model, Lambda_corr, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    print(f"Q-Merge with Noise Corrupted Stream: {acc_corr['avg']:.2f}% (MNIST={acc_corr['mnist']:.1f}, FMNIST={acc_corr['fmnist']:.1f}, CIFAR10={acc_corr['cifar10']:.1f}, SVHN={acc_corr['svhn']:.1f})")
    results_db["axis4"]["corrupted"] = acc_corr
    
    # 3. Highly Skewed Stream (Class imbalance)
    skewed_loaders = build_calibration_loader(train_datasets, N=16, batch_size=32, skew=True)
    Lambda_skew, _ = run_ste_optimization(pre_dict, expert_dicts_list, heads, skewed_loaders, test_loaders, pre_model, K, L,
                                          schema='sym_channel', num_bits=4, steps=40, lr=0.01)
    acc_skew = evaluate_accuracy(pre_model, Lambda_skew, pre_dict, expert_dicts_list, heads, test_loaders, 'sym_channel', 4, K, L)
    print(f"Q-Merge with Highly Skewed Stream: {acc_skew['avg']:.2f}% (MNIST={acc_skew['mnist']:.1f}, FMNIST={acc_skew['fmnist']:.1f}, CIFAR10={acc_skew['cifar10']:.1f}, SVHN={acc_skew['svhn']:.1f})")
    results_db["axis4"]["skewed"] = acc_skew
    
    # 4. Save results to results/metrics.json
    with open("results/metrics.json", "w") as f:
        json.dump(results_db, f, indent=2)
    print("\nMetrics saved to results/metrics.json")
    
    # 5. Generate and save Plots
    # Plot 1: Calibration Sweep N (Axis 1)
    plt.figure(figsize=(8, 5))
    ns = [1, 4, 16, 64]
    accs = [results_db["axis1"]["qmerge_sweep"][f"N_{n}"]["avg"] for n in ns]
    plt.plot(ns, accs, marker='o', linewidth=2, color='blue', label='Q-Merge (4-bit)')
    plt.axhline(y=results_db["axis1"]["fp16_task_arithmetic"]["avg"], color='green', linestyle='--', label='FP16 Task Arithmetic')
    plt.axhline(y=results_db["axis1"]["m_then_q_4bit"]["avg"], color='red', linestyle=':', label='Naive 4-bit M-then-Q')
    plt.xlabel('Calibration Stream Size (N)')
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.xscale('log')
    plt.xticks(ns, ns)
    plt.title('Impact of Calibration Stream Size on Quantized Model Merging')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig1_calibration_sweep.png')
    plt.close()
    
    # Plot 2: Cross-Schema Generalization Audit (Axis 2 Matrix)
    plt.figure(figsize=(10, 8))
    # We build a matrix
    matrix = []
    src_list = ['sym_tensor', 'sym_channel', 'asym_tensor', 'asym_channel']
    tgt_list = ['sym_tensor', 'sym_channel', 'asym_tensor', 'asym_channel', 'double_quant']
    for src in src_list:
        row = []
        for tgt in tgt_list:
            row.append(results_db["axis2"][src][tgt]["avg"])
        matrix.append(row)
        
    matrix = np.array(matrix)
    im = plt.imshow(matrix, cmap='YlGnBu', interpolation='nearest', vmin=matrix.min()-1, vmax=matrix.max()+1)
    plt.colorbar(im, label='Average Accuracy (%)')
    nice_names = {
        'sym_tensor': 'Symmetric\nPer-Tensor',
        'sym_channel': 'Symmetric\nPer-Channel',
        'asym_tensor': 'Asymmetric\nPer-Tensor',
        'asym_channel': 'Asymmetric\nPer-Channel',
        'double_quant': 'Double\nQuantization'
    }
    plt.xticks(range(len(tgt_list)), [nice_names[t] for t in tgt_list])
    plt.yticks(range(len(src_list)), [nice_names[s].replace('\n', ' ') for s in src_list])
    plt.xlabel('Target Deployment Schema ($Q_{\text{eval}}$)')
    plt.ylabel('Source Optimization Schema ($Q_{\text{opt}}$)')
    plt.title('Cross-Schema Generalization Matrix of Quantization-Aware Merging')
    for i in range(len(src_list)):
        for j in range(len(tgt_list)):
            plt.text(j, i, f"{matrix[i, j]:.1f}%", ha='center', va='center', color='black', weight='bold')
    plt.tight_layout()
    plt.savefig('results/fig2_cross_schema_matrix.png')
    plt.close()
    
    # Plot 3: Regularization Comparison (Axis 3)
    plt.figure(figsize=(8, 5))
    categories = ['Source\n(sym_channel)', 'Target\n(sym_tensor)']
    unreg_accs = [results_db["axis3"]["unregularized"]["source"]["avg"], results_db["axis3"]["unregularized"]["target"]["avg"]]
    reg_accs = [results_db["axis3"]["spatial_regularized"]["source"]["avg"], results_db["axis3"]["spatial_regularized"]["target"]["avg"]]
    es_accs = [results_db["axis3"]["blackbox_es"]["source"]["avg"], results_db["axis3"]["blackbox_es"]["target"]["avg"]]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.bar(x - width, unreg_accs, width, label='Unregularized STE (Q-Merge)', color='coral')
    plt.bar(x, reg_accs, width, label='TV Regularized STE', color='mediumaquamarine')
    plt.bar(x + width, es_accs, width, label='Derivative-Free (1+1 ES)', color='royalblue')
    
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Mitigating Cross-Operator Overfitting via Spatial Regularization')
    plt.xticks(x, categories)
    plt.ylim(0, 100)
    plt.grid(True, axis='y', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig3_regularization_comparison.png')
    plt.close()
    
    print("Plots generated successfully in results/ directory!")
