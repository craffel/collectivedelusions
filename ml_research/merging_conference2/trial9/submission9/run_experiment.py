import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import copy
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Disable cuDNN to bypass driver compatibility issues on the GPU cluster
    torch.backends.cudnn.enabled = False

class ResNet18Expert(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Expert, self).__init__()
        # Load ResNet18 with ImageNet weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def get_transforms(task_name):
    # Standard transformation as described in the literature
    if task_name in ['mnist', 'fashion_mnist']:
        # Resize to 32x32, replicate to 3 channels, then normalize
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:  # cifar10
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    return transform

def load_data(task_name, batch_size, is_test_run=False, eval_samples=None):
    transform = get_transforms(task_name)
    
    if task_name == 'mnist':
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif task_name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif task_name == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task_name}")
        
    if is_test_run:
        # Mini subset for fast syntax checking
        train_set = Subset(train_set, range(100))
        test_set = Subset(test_set, range(100))
    elif eval_samples is not None:
        # Limit evaluation set size if requested
        test_set = Subset(test_set, range(min(eval_samples, len(test_set))))
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_expert(task_name, epochs, batch_size, lr, device, is_test_run=False):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    train_loader, test_loader = load_data(task_name, batch_size, is_test_run=is_test_run)
    
    model = ResNet18Expert(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
    # Evaluate expert
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = correct / total * 100.0
    print(f"Expert {task_name.upper()} Test Accuracy: {test_acc:.2f}%")
    return model, test_acc

# Quantization helper function (Symmetric Uniform Quantization)
def quantize_tensor(tensor, num_bits=8, per_channel=False):
    if num_bits is None:
        return tensor
    
    qmax = 2**(num_bits - 1) - 1
    
    if not per_channel or tensor.dim() < 2:
        # Per-tensor quantization
        max_val = torch.max(torch.abs(tensor)).item()
        if max_val == 0:
            return tensor
        scale = max_val / qmax
        quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale
        return quantized
    else:
        # Per-channel quantization (along dim 0)
        quantized = torch.zeros_like(tensor)
        for i in range(tensor.size(0)):
            max_val = torch.max(torch.abs(tensor[i])).item()
            if max_val == 0:
                quantized[i] = tensor[i]
                continue
            scale = max_val / qmax
            quantized[i] = torch.clamp(torch.round(tensor[i] / scale), -qmax, qmax) * scale
        return quantized

def apply_ptq_to_model(model, num_bits=8, per_channel=False):
    quantized_model = copy.deepcopy(model)
    state_dict = quantized_model.state_dict()
    for name, param in state_dict.items():
        if 'weight' in name and param.dim() >= 2: # Only quantize weight tensors
            state_dict[name] = quantize_tensor(param, num_bits=num_bits, per_channel=per_channel)
    quantized_model.load_state_dict(state_dict)
    return quantized_model

# Environmental corruption helpers
def add_gaussian_noise(imgs, sigma=0.1):
    noise = torch.randn_like(imgs) * sigma
    return torch.clamp(imgs + noise, -1.0, 1.0)

def apply_gaussian_blur(imgs, kernel_size=3):
    # Simple uniform box blur as a proxy for blur corruption
    pad = kernel_size // 2
    kh, kw = kernel_size, kernel_size
    channels = imgs.shape[1]
    kernel = torch.ones(channels, 1, kh, kw, device=imgs.device) / (kh * kw)
    # Perform depthwise separable convolution
    blurred = torch.nn.functional.conv2d(imgs, kernel, padding=pad, groups=channels)
    return blurred

def evaluate_model(model, task_loaders, device, quant_bits=None, per_channel=False, corruption=None, sigma=0.1):
    model.eval()
    
    # Apply PTQ if requested
    if quant_bits is not None:
        eval_model = apply_ptq_to_model(model, num_bits=quant_bits, per_channel=per_channel)
    else:
        eval_model = model
        
    eval_model.eval()
    results = {}
    
    with torch.no_grad():
        for task_name, loader in task_loaders.items():
            # Standard head attaching
            # For evaluating task t, we use the specific classification head of expert t
            # The classification head of ResNet-18 is the 'resnet.fc' layer
            correct = 0
            total = 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                # Apply environmental corruptions
                if corruption == 'noise':
                    imgs = add_gaussian_noise(imgs, sigma=sigma)
                elif corruption == 'blur':
                    imgs = apply_gaussian_blur(imgs, kernel_size=3)
                    
                outputs = eval_model(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            acc = correct / total * 100.0
            results[task_name] = acc
            
    results['average'] = np.mean([results[t] for t in task_loaders.keys()])
    return results

# Model Merging Implementations
def get_task_vectors(experts, progenitor):
    # Compute task vectors: T_k = W_k - W_init
    task_vectors = []
    p_sd = progenitor.state_dict()
    for exp in experts:
        e_sd = exp.state_dict()
        tv = {}
        for name in p_sd.keys():
            if 'fc' not in name: # Exclude classification head
                tv[name] = e_sd[name] - p_sd[name]
        task_vectors.append(tv)
    return task_vectors

def merge_weight_averaging(experts, progenitor):
    # Simple Weight Averaging: W_merged = average of experts
    merged = copy.deepcopy(progenitor)
    merged_sd = merged.state_dict()
    expert_sds = [exp.state_dict() for exp in experts]
    K = len(experts)
    
    for name in merged_sd.keys():
        if 'fc' not in name:
            sum_w = torch.zeros_like(merged_sd[name], dtype=torch.float32)
            for sd in expert_sds:
                sum_w += sd[name].float()
            merged_sd[name] = (sum_w / K).to(merged_sd[name].dtype)
    merged.load_state_dict(merged_sd)
    return merged

def merge_task_arithmetic(experts, progenitor, lam=0.5):
    # W_merged = W_init + lam * sum(T_k)
    merged = copy.deepcopy(progenitor)
    merged_sd = merged.state_dict()
    task_vectors = get_task_vectors(experts, progenitor)
    
    for name in merged_sd.keys():
        if 'fc' not in name:
            sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
            for tv in task_vectors:
                sum_t += tv[name].float()
            merged_sd[name] = (merged_sd[name].float() + lam * sum_t).to(merged_sd[name].dtype)
    merged.load_state_dict(merged_sd)
    return merged

def merge_u_ipr(experts, progenitor):
    # Update-level Isotropic Parameter Resonance (U-IPR)
    # W_cal = W_init + s * T_merged
    # s = (1/K * sum(||T_k||_F)) / ||T_merged||_F
    merged = copy.deepcopy(progenitor)
    merged_sd = merged.state_dict()
    task_vectors = get_task_vectors(experts, progenitor)
    K = len(experts)
    
    for name in merged_sd.keys():
        if 'fc' not in name:
            sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
            sum_norm = 0.0
            for tv in task_vectors:
                sum_t += tv[name].float()
                sum_norm += torch.norm(tv[name].float()).item()
            
            t_merged = sum_t / K
            norm_merged = torch.norm(t_merged).item()
            
            if norm_merged > 1e-8:
                s = (sum_norm / K) / norm_merged
                # Heuristic clamp as described in literature [0.1, 10.0]
                s = max(0.1, min(10.0, s))
                merged_sd[name] = (merged_sd[name].float() + s * t_merged).to(merged_sd[name].dtype)
            else:
                merged_sd[name] = (merged_sd[name].float() + t_merged).to(merged_sd[name].dtype)
    merged.load_state_dict(merged_sd)
    return merged

def merge_qr_ipr(experts, progenitor, gamma=2.0):
    # Quantization-Robust Parameter Resonance (QR-IPR) from Submission 5
    # Clamps channel-wise scale factors using Median and MAD
    merged = copy.deepcopy(progenitor)
    merged_sd = merged.state_dict()
    task_vectors = get_task_vectors(experts, progenitor)
    K = len(experts)
    
    for name in merged_sd.keys():
        if 'fc' not in name:
            param_dim = merged_sd[name].dim()
            if param_dim >= 2: # Only for conv/linear layers
                Cout = merged_sd[name].size(0)
                sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
                for tv in task_vectors:
                    sum_t += tv[name].float()
                t_merged = sum_t / K
                
                scales = []
                for c in range(Cout):
                    norms_expert_c = [torch.norm(tv[name][c].float()).item() for tv in task_vectors]
                    norm_merged_c = torch.norm(t_merged[c]).item()
                    avg_expert_norm_c = np.mean(norms_expert_c)
                    if norm_merged_c > 1e-8:
                        sc = avg_expert_norm_c / norm_merged_c
                    else:
                        sc = 1.0
                    scales.append(sc)
                    
                scales = np.array(scales)
                # Compute robust statistics
                med = np.median(scales)
                mad = np.median(np.abs(scales - med))
                mad = max(mad, 1e-4)
                
                L = max(0.1, med - gamma * mad)
                U = min(4.0, med + gamma * mad)
                
                clamped_scales = np.clip(scales, L, U)
                
                # Apply channel-wise clamped scale
                t_cal = torch.zeros_like(t_merged)
                for c in range(Cout):
                    t_cal[c] = clamped_scales[c] * t_merged[c]
                    
                merged_sd[name] = (merged_sd[name].float() + t_cal).to(merged_sd[name].dtype)
            else:
                # 1D parameters fallback to tighter layer-wise robust scaling
                sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
                sum_norm = 0.0
                for tv in task_vectors:
                    sum_t += tv[name].float()
                    sum_norm += torch.norm(tv[name].float()).item()
                t_merged = sum_t / K
                norm_merged = torch.norm(t_merged).item()
                if norm_merged > 1e-8:
                    s = (sum_norm / K) / norm_merged
                    s = max(0.1, min(3.0, s))
                    merged_sd[name] = (merged_sd[name].float() + s * t_merged).to(merged_sd[name].dtype)
                else:
                    merged_sd[name] = (merged_sd[name].float() + t_merged).to(merged_sd[name].dtype)
                    
    merged.load_state_dict(merged_sd)
    return merged

def merge_wcpr(experts, progenitor):
    # Wasserstein-Calibrated Parameter Resonance (WCPR) from Submission 9
    # Grounded in 1D Optimal Transport, maps merged parameters to the Wasserstein barycenter channel-by-channel
    merged = copy.deepcopy(progenitor)
    merged_sd = merged.state_dict()
    task_vectors = get_task_vectors(experts, progenitor)
    K = len(experts)
    
    for name in merged_sd.items():
        name = name[0]
        if 'fc' not in name:
            param_dim = merged_sd[name].dim()
            if param_dim >= 2:
                Cout = merged_sd[name].size(0)
                sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
                for tv in task_vectors:
                    sum_t += tv[name].float()
                t_merged = sum_t / K
                
                t_cal = torch.zeros_like(t_merged)
                for c in range(Cout):
                    mc = t_merged[c].flatten()
                    Ic = torch.argsort(mc)
                    
                    # Sort experts channel c
                    s_experts = []
                    for tv in task_vectors:
                        s_experts.append(torch.sort(tv[name][c].flatten())[0])
                    
                    # Target is the element-wise average of sorted experts (exact Wasserstein-2 barycenter)
                    s_target = torch.stack(s_experts).mean(dim=0)
                    
                    # Map back based on sorted rank of merged update
                    cflat = torch.zeros_like(mc)
                    cflat[Ic] = s_target
                    t_cal[c] = cflat.view_as(t_merged[c])
                    
                merged_sd[name] = (merged_sd[name].float() + t_cal).to(merged_sd[name].dtype)
            else:
                # 1D fallback to layer-wise isotropic scale (U-IPR)
                sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
                sum_norm = 0.0
                for tv in task_vectors:
                    sum_t += tv[name].float()
                    sum_norm += torch.norm(tv[name].float()).item()
                t_merged = sum_t / K
                norm_merged = torch.norm(t_merged).item()
                if norm_merged > 1e-8:
                    s = (sum_norm / K) / norm_merged
                    s = max(0.1, min(10.0, s))
                    merged_sd[name] = (merged_sd[name].float() + s * t_merged).to(merged_sd[name].dtype)
                else:
                    merged_sd[name] = (merged_sd[name].float() + t_merged).to(merged_sd[name].dtype)
                    
    merged.load_state_dict(merged_sd)
    return merged

def merge_qcot(experts, progenitor, C=2.0):
    # Proposed Quantization-Constrained Optimal Transport (QCOT) Calibration
    # Formulated as constrained OT to prevent dynamic range inflation while preserving distribution shape.
    # We clip the target Wasserstein barycenter to [-C, C] to strictly bound the infinity norm of task updates.
    merged = copy.deepcopy(progenitor)
    merged_sd = merged.state_dict()
    task_vectors = get_task_vectors(experts, progenitor)
    K = len(experts)
    
    for name in merged_sd.items():
        name = name[0]
        if 'fc' not in name:
            param_dim = merged_sd[name].dim()
            if param_dim >= 2:
                Cout = merged_sd[name].size(0)
                sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
                for tv in task_vectors:
                    sum_t += tv[name].float()
                t_merged = sum_t / K
                
                t_cal = torch.zeros_like(t_merged)
                for c in range(Cout):
                    mc = t_merged[c].flatten()
                    Ic = torch.argsort(mc)
                    
                    # Sort experts channel c
                    s_experts = []
                    for tv in task_vectors:
                        s_experts.append(torch.sort(tv[name][c].flatten())[0])
                    
                    # Target is the element-wise average of sorted experts (exact Wasserstein-2 barycenter)
                    s_target = torch.stack(s_experts).mean(dim=0)
                    
                    # Constraint step: Clip the target barycenter to prevent dynamic range inflation
                    s_target_constrained = torch.clamp(s_target, -C, C)
                    
                    # Map back based on sorted rank of merged update
                    cflat = torch.zeros_like(mc)
                    cflat[Ic] = s_target_constrained
                    t_cal[c] = cflat.view_as(t_merged[c])
                    
                merged_sd[name] = (merged_sd[name].float() + t_cal).to(merged_sd[name].dtype)
            else:
                # 1D fallback to layer-wise isotropic scale (U-IPR) clipped tightly
                sum_t = torch.zeros_like(merged_sd[name], dtype=torch.float32)
                sum_norm = 0.0
                for tv in task_vectors:
                    sum_t += tv[name].float()
                    sum_norm += torch.norm(tv[name].float()).item()
                t_merged = sum_t / K
                norm_merged = torch.norm(t_merged).item()
                if norm_merged > 1e-8:
                    s = (sum_norm / K) / norm_merged
                    s = max(0.1, min(C, s)) # Limit isotropic scale factor as well
                    merged_sd[name] = (merged_sd[name].float() + s * t_merged).to(merged_sd[name].dtype)
                else:
                    merged_sd[name] = (merged_sd[name].float() + t_merged).to(merged_sd[name].dtype)
                    
    merged.load_state_dict(merged_sd)
    return merged

def main():
    parser = argparse.ArgumentParser(description="Model Merging & Calibration Experiment")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per expert")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test", action="store_true", help="Quick syntax test run on CPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_samples", type=int, default=None, help="Limit test samples evaluated")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.test else "cpu")
    print(f"Running on device: {device}")
    
    # Progenitor initialization
    progenitor = ResNet18Expert(num_classes=10).to(device)
    # Freeze progenitor weights to make sure it acts as reference
    progenitor.eval()
    
    # 1. Train or load task experts
    tasks = ['mnist', 'fashion_mnist', 'cifar10']
    experts = []
    expert_accs = {}
    
    if args.test:
        print("Running test sanity check on CPU...")
        args.epochs = 1
        args.eval_samples = 100
        
    os.makedirs("./checkpoints", exist_ok=True)
    
    for task in tasks:
        ckpt_path = f"./checkpoints/expert_{task}.pt"
        if os.path.exists(ckpt_path) and not args.test:
            print(f"Loading pre-trained expert for {task} from {ckpt_path}")
            exp = ResNet18Expert(num_classes=10).to(device)
            exp.load_state_dict(torch.load(ckpt_path, map_location=device))
            experts.append(exp)
            # Evaluate expert accuracy
            _, test_loader = load_data(task, args.batch_size, is_test_run=args.test, eval_samples=args.eval_samples)
            acc = evaluate_model(exp, {task: test_loader}, device)[task]
            expert_accs[task] = acc
            print(f"Loaded Expert {task} Test Acc: {acc:.2f}%")
        else:
            exp, acc = train_expert(task, args.epochs, args.batch_size, args.lr, device, is_test_run=args.test)
            if not args.test:
                torch.save(exp.state_dict(), ckpt_path)
            experts.append(exp)
            expert_accs[task] = acc
            
    # Load all test data loaders for evaluation
    task_loaders = {}
    for task in tasks:
        _, loader = load_data(task, args.batch_size, is_test_run=args.test, eval_samples=args.eval_samples)
        task_loaders[task] = loader
        
    # Standard head attaching helper
    # Since each task expert has its own task-specific classification head,
    # during evaluation we attach the classification head of the respective expert.
    def attach_head(merged_model, expert_model):
        merged_sd = merged_model.state_dict()
        expert_sd = expert_model.state_dict()
        # Copy the fc classification head from expert to merged
        for name in expert_sd.keys():
            if 'fc' in name:
                merged_sd[name] = expert_sd[name]
        merged_model.load_state_dict(merged_sd)
        
    # Helper to evaluate all experts heads on a merged model
    def evaluate_merged_backbone(backbone_model):
        results = {}
        for i, task in enumerate(tasks):
            model_eval = copy.deepcopy(backbone_model)
            attach_head(model_eval, experts[i])
            acc = evaluate_model(model_eval, {task: task_loaders[task]}, device)[task]
            results[task] = acc
        results['average'] = np.mean([results[t] for t in tasks])
        return results
        
    def evaluate_merged_backbone_quantized(backbone_model, num_bits=8, per_channel=False):
        results = {}
        for i, task in enumerate(tasks):
            model_eval = copy.deepcopy(backbone_model)
            attach_head(model_eval, experts[i])
            acc = evaluate_model(model_eval, {task: task_loaders[task]}, device, quant_bits=num_bits, per_channel=per_channel)[task]
            results[task] = acc
        results['average'] = np.mean([results[t] for t in tasks])
        return results

    def evaluate_merged_backbone_corrupted(backbone_model, corruption='noise', sigma=0.1):
        results = {}
        for i, task in enumerate(tasks):
            model_eval = copy.deepcopy(backbone_model)
            attach_head(model_eval, experts[i])
            acc = evaluate_model(model_eval, {task: task_loaders[task]}, device, corruption=corruption, sigma=sigma)[task]
            results[task] = acc
        results['average'] = np.mean([results[t] for t in tasks])
        return results

    print("\n=== Evaluating Merging Strategies ===")
    
    methods = {}
    
    # 1. Weight Averaging
    print("Merging: Weight Averaging...")
    wa_model = merge_weight_averaging(experts, progenitor)
    methods["WA"] = wa_model
    
    # 2. Task Arithmetic (tuned lambda)
    for lam in [0.1, 0.3, 0.5, 0.7, 1.0]:
        print(f"Merging: Task Arithmetic (lambda={lam})...")
        ta_model = merge_task_arithmetic(experts, progenitor, lam=lam)
        methods[f"TA_lam_{lam}"] = ta_model
        
    # 3. Isotropic Parameter Resonance (U-IPR)
    print("Merging: Update Isotropic Parameter Resonance (U-IPR)...")
    u_ipr_model = merge_u_ipr(experts, progenitor)
    methods["U-IPR"] = u_ipr_model
    
    # 4. Quantization-Robust Parameter Resonance (QR-IPR)
    print("Merging: Quantization-Robust Parameter Resonance (QR-IPR)...")
    qr_ipr_model = merge_qr_ipr(experts, progenitor, gamma=2.0)
    methods["QR-IPR"] = qr_ipr_model
    
    # 5. Wasserstein-Calibrated Parameter Resonance (WCPR)
    print("Merging: Wasserstein-Calibrated Parameter Resonance (WCPR)...")
    wcpr_model = merge_wcpr(experts, progenitor)
    methods["WCPR"] = wcpr_model
    
    # 6. Our Proposed QCOT Calibration (sweep clipping values C)
    for C in [0.1, 0.2, 0.5, 1.0, 2.0]:
        print(f"Merging: Proposed QCOT Calibration (C={C})...")
        qcot_model = merge_qcot(experts, progenitor, C=C)
        methods[f"QCOT_C_{C}"] = qcot_model
        
    # Run evaluation suite across all methods
    all_results = {}
    
    for name, model in methods.items():
        print(f"\nEvaluating: {name}")
        res_fp32 = evaluate_merged_backbone(model)
        print(f"  FP32: Avg Acc = {res_fp32['average']:.2f}% (MNIST={res_fp32['mnist']:.2f}%, F-MNIST={res_fp32['fashion_mnist']:.2f}%, CIFAR10={res_fp32['cifar10']:.2f}%)")
        
        # INT8 Per-Tensor PTQ
        res_tensor_int8 = evaluate_merged_backbone_quantized(model, num_bits=8, per_channel=False)
        print(f"  INT8 Per-Tensor PTQ: Avg Acc = {res_tensor_int8['average']:.2f}%")
        
        # INT8 Per-Channel PTQ
        res_channel_int8 = evaluate_merged_backbone_quantized(model, num_bits=8, per_channel=True)
        print(f"  INT8 Per-Channel PTQ: Avg Acc = {res_channel_int8['average']:.2f}%")
        
        # Environmental Corruption: Gaussian Noise (sigma=0.1)
        res_noise = evaluate_merged_backbone_corrupted(model, corruption='noise', sigma=0.1)
        print(f"  Gaussian Noise (sigma=0.1): Avg Acc = {res_noise['average']:.2f}%")

        # Environmental Corruption: Gaussian Blur (3x3 kernel)
        res_blur = evaluate_merged_backbone_corrupted(model, corruption='blur')
        print(f"  Gaussian Blur (3x3): Avg Acc = {res_blur['average']:.2f}%")
        
        all_results[name] = {
            "FP32": res_fp32,
            "INT8_PerTensor": res_tensor_int8,
            "INT8_PerChannel": res_channel_int8,
            "GaussianNoise": res_noise,
            "GaussianBlur": res_blur
        }
        
    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nResults saved to results.json")
    
    # Generate some beautiful, theoretically revealing plots
    generate_plots(all_results)

def generate_plots(all_results):
    # Plot 1: FP32 vs. INT8 Per-Tensor Accuracy across methods
    fig, ax = plt.subplots(figsize=(10, 6))
    
    selected_methods = ["WA", "TA_lam_0.5", "U-IPR", "QR-IPR", "WCPR", "QCOT_C_0.1", "QCOT_C_0.2", "QCOT_C_0.5", "QCOT_C_1.0", "QCOT_C_2.0"]
    methods_disp = ["WA", "TA (0.5)", "U-IPR", "QR-IPR", "WCPR", "QCOT (0.1)", "QCOT (0.2)", "QCOT (0.5)", "QCOT (1.0)", "QCOT (2.0)"]
    
    fp32_accs = [all_results[m]["FP32"]["average"] for m in selected_methods if m in all_results]
    int8_accs = [all_results[m]["INT8_PerTensor"]["average"] for m in selected_methods if m in all_results]
    actual_methods = [methods_disp[i] for i, m in enumerate(selected_methods) if m in all_results]
    
    x = np.arange(len(actual_methods))
    width = 0.35
    
    ax.bar(x - width/2, fp32_accs, width, label='FP32', color='#1f77b4')
    ax.bar(x + width/2, int8_accs, width, label='INT8 Per-Tensor', color='#ff7f0e')
    
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Model Merging Performance: FP32 vs. INT8 Per-Tensor PTQ')
    ax.set_xticks(x)
    ax.set_xticklabels(actual_methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("plots_comparison.png", dpi=300)
    plt.close()
    
    # Plot 2: Pareto Frontier of QCOT (Accuracy vs Clipping C)
    qcot_C = [0.1, 0.2, 0.5, 1.0, 2.0]
    qcot_keys = [f"QCOT_C_{c}" for c in qcot_C]
    
    if all(k in all_results for k in qcot_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        fp32_qcot = [all_results[k]["FP32"]["average"] for k in qcot_keys]
        int8_qcot = [all_results[k]["INT8_PerTensor"]["average"] for k in qcot_keys]
        
        ax.plot(qcot_C, fp32_qcot, marker='o', linestyle='-', label='FP32 Accuracy', color='#2ca02c')
        ax.plot(qcot_C, int8_qcot, marker='s', linestyle='--', label='INT8 Per-Tensor Accuracy', color='#d62728')
        
        ax.set_xlabel('Clipping Constraint $C$')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title('QCOT Performance vs. Clipping Constraint $C$')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig("plots_qcot_sweep.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
