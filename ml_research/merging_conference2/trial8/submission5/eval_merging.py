import os
import copy
import json
import torch
import torch.nn as nn

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on the cluster
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

# Settings
BATCH_SIZE = 128
EVAL_SAMPLES = 1000  # Evaluate on 1000 samples per task for stable and fast grid sweep
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(task, corruption_type=None, corruption_severity=0.0):
    # Base transforms
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
        
    # Apply corruption before normalization
    corrupt_transforms = []
    if corruption_type == 'noise' and corruption_severity > 0:
        corrupt_transforms.append(transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * corruption_severity, 0.0, 1.0)))
    elif corruption_type == 'blur' and corruption_severity > 0:
        kernel_size = int(corruption_severity)
        if kernel_size % 2 == 0:
            kernel_size += 1
        corrupt_transforms.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=1.0))
    elif corruption_type == 'contrast' and corruption_severity > 0:
        corrupt_transforms.append(transforms.ColorJitter(contrast=(corruption_severity, corruption_severity)))
        
    # Complete transform pipeline
    transform = transforms.Compose(
        base_transform + corrupt_transforms + [transforms.Normalize(norm_mean, norm_std)]
    )
    
    # Load dataset
    if task == 'mnist':
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif task == 'fashion':
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    elif task == 'cifar10':
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        
    # Subset for faster evaluation
    indices = list(range(min(EVAL_SAMPLES, len(test_set))))
    test_subset = Subset(test_set, indices)
    
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return test_loader

# Quantization helper supporting per-tensor and per-channel modes
def quantize_weight(tensor, bits=8, qmode='per_tensor'):
    if bits is None or bits >= 32:
        return tensor
    
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    
    if qmode == 'per_tensor' or tensor.dim() < 2:
        # Symmetric uniform per-tensor quantization
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            return tensor
            
        scale = max_val / qmax
        q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return q_tensor * scale
    elif qmode == 'per_channel':
        # Symmetric uniform per-channel quantization (along dim 0)
        q_tensor = tensor.clone()
        for c in range(tensor.shape[0]):
            channel_tensor = tensor[c]
            max_val = torch.max(torch.abs(channel_tensor))
            if max_val == 0:
                continue
            scale = max_val / qmax
            q_tensor[c] = torch.clamp(torch.round(channel_tensor / scale), qmin, qmax) * scale
        return q_tensor
    else:
        raise ValueError(f"Unknown qmode: {qmode}")

def load_models():
    # Load progenitor
    progenitor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    progenitor.fc = nn.Linear(progenitor.fc.in_features, 10)
    
    # Load experts
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

def merge_models_base(progenitor, experts, method='wa', lam=0.5):
    merged = copy.deepcopy(progenitor)
    merged_state = merged.state_dict()
    prog_state = progenitor.state_dict()
    
    expert_states = {t: experts[t].state_dict() for t in experts}
    keys = [k for k in prog_state.keys() if 'fc' not in k]  # Only merge backbone
    
    if method == 'wa':
        for key in keys:
            weights = [expert_states[t][key].float() for t in experts]
            merged_state[key] = torch.mean(torch.stack(weights), dim=0)
    elif method == 'ta':
        for key in keys:
            # Task vectors
            tvs = [expert_states[t][key].float() - prog_state[key].float() for t in experts]
            merged_state[key] = prog_state[key].float() + lam * torch.sum(torch.stack(tvs), dim=0)
    elif method == 'ties':
        # TIES-Merging
        for key in keys:
            tvs = [expert_states[t][key].float() - prog_state[key].float() for t in experts]
            # Prune bottom 20% by absolute value
            pruned_tvs = []
            for tv in tvs:
                if tv.numel() == 0:
                    pruned_tvs.append(tv)
                    continue
                threshold = torch.quantile(torch.abs(tv), 0.2)
                mask = torch.abs(tv) >= threshold
                pruned_tvs.append(tv * mask)
            
            # Sign election
            stacked = torch.stack(pruned_tvs)
            signs = torch.sign(stacked)
            sum_signs = torch.sum(signs, dim=0)
            consensus_sign = torch.sign(sum_signs)
            
            # Disjoint merge: average updates that match consensus sign
            mask_consensus = (signs == consensus_sign.unsqueeze(0)) & (stacked != 0)
            matching_vals = stacked * mask_consensus
            counts = torch.sum(mask_consensus, dim=0)
            avg_update = torch.zeros_like(consensus_sign)
            valid_mask = counts > 0
            avg_update[valid_mask] = torch.sum(matching_vals, dim=0)[valid_mask] / counts[valid_mask]
            
            merged_state[key] = prog_state[key].float() + lam * avg_update
    elif method == 'dare':
        # DARE-Merging with drop probability p = 0.2
        drop_prob = 0.2
        for key in keys:
            tvs = [expert_states[t][key].float() - prog_state[key].float() for t in experts]
            dare_tvs = []
            for tv in tvs:
                if drop_prob > 0 and tv.numel() > 0:
                    mask = (torch.rand_like(tv) >= drop_prob).float()
                    dare_tv = tv * mask / (1.0 - drop_prob)
                else:
                    dare_tv = tv
                dare_tvs.append(dare_tv)
            
            merged_update = torch.mean(torch.stack(dare_tvs), dim=0)
            merged_state[key] = prog_state[key].float() + lam * merged_update
            
    merged.load_state_dict(merged_state)
    return merged

def apply_calibration(progenitor, experts, merged_model, cal_type='none', bits=None, qmode='per_tensor'):
    calibrated = copy.deepcopy(merged_model)
    cal_state = calibrated.state_dict()
    prog_state = progenitor.state_dict()
    expert_states = {t: experts[t].state_dict() for t in experts}
    
    keys = [k for k in prog_state.keys() if 'fc' not in k]
    K = len(experts)
    
    for key in keys:
        # Check if the parameter is float/weight (non-integer buffers)
        if not prog_state[key].is_floating_point():
            continue
            
        w_init = prog_state[key].float()
        w_merged = cal_state[key].float()
        t_merged = w_merged - w_init
        
        # Expert task updates
        t_experts = [expert_states[t][key].float() - w_init for t in experts]
        
        if cal_type == 'none':
            # No calibration, but we must still quantize if bits is specified
            cal_state[key] = quantize_weight(w_merged, bits, qmode=qmode)
            
        elif cal_type == 'u-ipr':
            # Isotropic Parameter Resonance (layer-wise)
            norm_merged = torch.norm(t_merged)
            norm_experts = torch.mean(torch.stack([torch.norm(t) for t in t_experts]))
            
            scale = norm_experts / (norm_merged + 1e-8)
            # Standard clamp
            scale = torch.clamp(scale, min=0.1, max=10.0)
            
            t_corrected = scale * t_merged
            w_corrected = w_init + t_corrected
            
            # Apply quantization to the calibrated full weights
            cal_state[key] = quantize_weight(w_corrected, bits, qmode=qmode)
            
        elif cal_type == 'hns':
            # Holographic Norm Scaling (channel-wise)
            # Check tensor dimensions (channel-wise is applied on dimension 0)
            if t_merged.dim() >= 2:
                t_corrected = torch.zeros_like(t_merged)
                for c in range(t_merged.shape[0]):
                    norm_merged_c = torch.norm(t_merged[c])
                    # Average original expert norm for this channel
                    norm_experts_c = torch.mean(torch.stack([torch.norm(t[c]) for t in t_experts]))
                    
                    scale_c = norm_experts_c / (norm_merged_c + 1e-8)
                    scale_c = torch.clamp(scale_c, min=0.1, max=10.0)
                    t_corrected[c] = scale_c * t_merged[c]
                w_corrected = w_init + t_corrected
            else:
                # Fallback to layer-wise for 1D biases/buffers
                norm_merged = torch.norm(t_merged)
                norm_experts = torch.mean(torch.stack([torch.norm(t) for t in t_experts]))
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                w_corrected = w_init + scale * t_merged
                
            cal_state[key] = quantize_weight(w_corrected, bits, qmode=qmode)
            
        elif cal_type == 'qr-ipr':
            # Our Proposed Quantization-Robust Parameter Resonance (Adaptive clamping with outlier mitigation)
            if t_merged.dim() >= 2:
                scales = []
                # First, calculate raw channel scale factors
                for c in range(t_merged.shape[0]):
                    norm_merged_c = torch.norm(t_merged[c])
                    norm_experts_c = torch.mean(torch.stack([torch.norm(t[c]) for t in t_experts]))
                    scale_c = norm_experts_c / (norm_merged_c + 1e-8)
                    scales.append(scale_c.item())
                
                scales = np.array(scales)
                # Compute robust statistics (Median and Median Absolute Deviation)
                median = np.median(scales)
                mad = np.median(np.abs(scales - median))
                if mad == 0:
                    mad = 1e-4
                    
                # Robust dynamic clamping threshold (prevents outliers from blowing up the quantization dynamic range)
                lower_bound = max(0.1, median - 1.5 * mad)
                upper_bound = min(4.0, median + 1.5 * mad)
                
                t_corrected = torch.zeros_like(t_merged)
                for c in range(t_merged.shape[0]):
                    scale_c = scales[c]
                    scale_c_clamped = np.clip(scale_c, lower_bound, upper_bound)
                    t_corrected[c] = scale_c_clamped * t_merged[c]
                w_corrected = w_init + t_corrected
            else:
                # Layer-wise robust scaling
                norm_merged = torch.norm(t_merged)
                norm_experts = torch.mean(torch.stack([torch.norm(t) for t in t_experts]))
                scale = norm_experts / (norm_merged + 1e-8)
                # Tighter clamp for 1D parameters under quantization
                scale = torch.clamp(scale, min=0.1, max=3.0)
                w_corrected = w_init + scale * t_merged
                
            cal_state[key] = quantize_weight(w_corrected, bits, qmode=qmode)

        elif cal_type == 'sc-qr-ipr':
            # Our Proposed Sparsity-Compensated Quantization-Robust Parameter Resonance (SC-QR-IPR)
            if t_merged.dim() >= 2:
                scales = []
                # First, calculate raw channel scale factors with pruning-aware sparsity compensation
                for c in range(t_merged.shape[0]):
                    norm_merged_c = torch.norm(t_merged[c])
                    norm_experts_c = torch.mean(torch.stack([torch.norm(t[c]) for t in t_experts]))
                    scale_c = norm_experts_c / (norm_merged_c + 1e-8)
                    
                    # Compute sparsity (active ratio) of merged update and expert updates
                    active_ratio_merged_c = (t_merged[c] != 0).float().mean().item()
                    active_ratio_experts_c = torch.mean(torch.stack([(t[c] != 0).float().mean() for t in t_experts])).item()
                    
                    # Only apply compensation if the merged update was explicitly pruned compared to the experts
                    if active_ratio_merged_c < active_ratio_experts_c - 0.05:
                        compensation_factor = np.sqrt(active_ratio_merged_c / (active_ratio_experts_c + 1e-8))
                        scale_c = scale_c * compensation_factor
                        
                    scales.append(scale_c.item() if isinstance(scale_c, torch.Tensor) else scale_c)
                
                scales = np.array(scales)
                # Compute robust statistics (Median and Median Absolute Deviation)
                median = np.median(scales)
                mad = np.median(np.abs(scales - median))
                if mad == 0:
                    mad = 1e-4
                    
                # Robust dynamic clamping threshold (prevents outliers from blowing up the quantization dynamic range)
                lower_bound = max(0.1, median - 1.5 * mad)
                upper_bound = min(4.0, median + 1.5 * mad)
                
                t_corrected = torch.zeros_like(t_merged)
                for c in range(t_merged.shape[0]):
                    scale_c = scales[c]
                    scale_c_clamped = np.clip(scale_c, lower_bound, upper_bound)
                    t_corrected[c] = scale_c_clamped * t_merged[c]
                w_corrected = w_init + t_corrected
            else:
                # Layer-wise robust scaling with pruning-aware sparsity compensation
                norm_merged = torch.norm(t_merged)
                norm_experts = torch.mean(torch.stack([torch.norm(t) for t in t_experts]))
                scale = norm_experts / (norm_merged + 1e-8)
                
                active_ratio_merged = (t_merged != 0).float().mean().item()
                active_ratio_experts = torch.mean(torch.stack([(t != 0).float().mean() for t in t_experts])).item()
                
                if active_ratio_merged < active_ratio_experts - 0.05:
                    compensation_factor = np.sqrt(active_ratio_merged / (active_ratio_experts + 1e-8))
                    scale = scale * compensation_factor
                    
                # Tighter clamp for 1D parameters under quantization
                scale = torch.clamp(scale, min=0.1, max=3.0)
                w_corrected = w_init + scale * t_merged
                
            cal_state[key] = quantize_weight(w_corrected, bits, qmode=qmode)
            
    calibrated.load_state_dict(cal_state)
    return calibrated

def evaluate_model(model, experts, task, corruption_type=None, corruption_severity=0.0):
    # Construct task model by combining merged backbone + task-specific head
    task_model = copy.deepcopy(model)
    task_model_state = task_model.state_dict()
    expert_state = experts[task].state_dict()
    
    # Load task classification head (fc) from the expert
    for k in expert_state.keys():
        if 'fc' in k:
            task_model_state[k] = expert_state[k]
            
    task_model.load_state_dict(task_model_state)
    task_model = task_model.to(DEVICE)
    task_model.eval()
    
    loader = get_dataloaders(task, corruption_type, corruption_severity)
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

def run_grid_sweep(progenitor, experts):
    results = []
    
    # Sweeps configuration
    merge_paradigms = [
        ('wa', 0.5), # Weight Averaging
        ('ta', 0.5), # Task Arithmetic lambda = 0.5
        ('ta', 0.7), # Task Arithmetic lambda = 0.7
        ('ties', 0.5), # TIES-Merging
        ('dare', 0.5), # DARE-Merging
    ]
    
    cal_methods = ['none', 'u-ipr', 'hns', 'qr-ipr', 'sc-qr-ipr']
    
    # Quantization configurations: (bits, qmode)
    quant_settings = [
        (None, 'per_tensor'),    # FP32
        (8, 'per_tensor'),       # INT8 per-tensor
        (8, 'per_channel'),      # INT8 per-channel
        (4, 'per_tensor'),       # INT4 per-tensor
        (4, 'per_channel'),      # INT4 per-channel
    ]
    
    corruptions = [
        ('clean', 0.0),
        ('noise', 0.1), # Gaussian Noise std=0.1
        ('blur', 3.0),   # Gaussian Blur kernel=3
    ]
    
    total_runs = len(merge_paradigms) * len(cal_methods) * len(quant_settings) * len(corruptions)
    print(f"\nStarting comprehensive sweep over {total_runs} configurations...")
    
    run_idx = 1
    for m_type, lam in merge_paradigms:
        # Create base merge model
        base_merged = merge_models_base(progenitor, experts, m_type, lam)
        
        for cal in cal_methods:
            for q_bit, qmode in quant_settings:
                # Apply calibration and simulated quantization
                cal_quant_model = apply_calibration(progenitor, experts, base_merged, cal, q_bit, qmode)
                
                for corr_type, corr_sev in corruptions:
                    accs = {}
                    for task in ['mnist', 'fashion', 'cifar10']:
                        acc = evaluate_model(cal_quant_model, experts, task, corr_type, corr_sev)
                        accs[task] = acc
                        
                    avg_acc = np.mean(list(accs.values()))
                    
                    config = {
                        'merge_type': m_type,
                        'lambda': lam,
                        'calibration': cal,
                        'quantization_bits': q_bit if q_bit else 'FP32',
                        'quantization_mode': qmode,
                        'corruption': corr_type,
                        'corruption_severity': corr_sev,
                        'mnist_acc': accs['mnist'],
                        'fashion_acc': accs['fashion'],
                        'cifar10_acc': accs['cifar10'],
                        'avg_acc': avg_acc
                    }
                    results.append(config)
                    
                    if run_idx % 20 == 0 or run_idx == total_runs:
                        print(f"Sweep Progress: {run_idx}/{total_runs} - Merge: {m_type}({lam}), Cal: {cal}, Q: {q_bit}({qmode}), Corr: {corr_type} -> Avg Acc: {avg_acc:.2f}%")
                    run_idx += 1
                    
    # Save results to JSON
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Sweep completed and results saved to sweep_results.json!")
    return results

if __name__ == "__main__":
    if not (os.path.exists("mnist_expert.pt") and os.path.exists("fashion_expert.pt") and os.path.exists("cifar10_expert.pt")):
        print("Expert model checkpoints not found! Please run train_experts.py first.")
    else:
        print("Loading models...")
        progenitor, experts = load_models()
        print("Models loaded successfully.")
        
        # Run grid sweep
        results = run_grid_sweep(progenitor, experts)
        
        # Print summary table for Clean FP32, INT8, and INT4 (for Task Arithmetic lambda=0.5)
        print("\n=== SUMMARY OF CLEAN ACCURACY BY QUANTIZATION LEVEL (Task Arithmetic lambda = 0.5, Per-Tensor) ===")
        print(f"{'Calibration':<15} | {'FP32':<10} | {'INT8 (8-bit)':<12} | {'INT4 (4-bit)':<12}")
        print("-" * 60)
        for cal in ['none', 'u-ipr', 'hns', 'qr-ipr', 'sc-qr-ipr']:
            row = [cal]
            for q in ['FP32', 8, 4]:
                acc = [r['avg_acc'] for r in results if r['merge_type'] == 'ta' and r['lambda'] == 0.5 and r['calibration'] == cal and r['quantization_bits'] == q and r['quantization_mode'] == 'per_tensor' and r['corruption'] == 'clean']
                val = f"{acc[0]:.2f}%" if acc else "N/A"
                row.append(val)
            print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<12} | {row[3]:<12}")
            
        print("\n=== SUMMARY OF CLEAN ACCURACY BY QUANTIZATION LEVEL (Task Arithmetic lambda = 0.5, Per-Channel) ===")
        print(f"{'Calibration':<15} | {'FP32':<10} | {'INT8 (8-bit)':<12} | {'INT4 (4-bit)':<12}")
        print("-" * 60)
        for cal in ['none', 'u-ipr', 'hns', 'qr-ipr', 'sc-qr-ipr']:
            row = [cal]
            for q in ['FP32', 8, 4]:
                # For FP32, per-channel or per-tensor are identical in evaluation, so we grab from FP32 configuration
                q_val = 'FP32' if q == 'FP32' else q
                mode_val = 'per_tensor' if q == 'FP32' else 'per_channel'
                acc = [r['avg_acc'] for r in results if r['merge_type'] == 'ta' and r['lambda'] == 0.5 and r['calibration'] == cal and r['quantization_bits'] == q_val and r['quantization_mode'] == mode_val and r['corruption'] == 'clean']
                val = f"{acc[0]:.2f}%" if acc else "N/A"
                row.append(val)
            print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<12} | {row[3]:<12}")
            
        print("\n=== SUMMARY OF ROBUSTNESS UNDER NOISE (Task Arithmetic lambda = 0.5, FP32) ===")
        print(f"{'Calibration':<15} | {'Clean':<10} | {'Gaussian Noise (std=0.1)':<24} | {'Gaussian Blur (k=3)':<20}")
        print("-" * 80)
        for cal in ['none', 'u-ipr', 'hns', 'qr-ipr', 'sc-qr-ipr']:
            row = [cal]
            # Clean
            acc_clean = [r['avg_acc'] for r in results if r['merge_type'] == 'ta' and r['lambda'] == 0.5 and r['calibration'] == cal and r['quantization_bits'] == 'FP32' and r['corruption'] == 'clean']
            row.append(f"{acc_clean[0]:.2f}%" if acc_clean else "N/A")
            # Noise
            acc_noise = [r['avg_acc'] for r in results if r['merge_type'] == 'ta' and r['lambda'] == 0.5 and r['calibration'] == cal and r['quantization_bits'] == 'FP32' and r['corruption'] == 'noise']
            row.append(f"{acc_noise[0]:.2f}%" if acc_noise else "N/A")
            # Blur
            acc_blur = [r['avg_acc'] for r in results if r['merge_type'] == 'ta' and r['lambda'] == 0.5 and r['calibration'] == cal and r['quantization_bits'] == 'FP32' and r['corruption'] == 'blur']
            row.append(f"{acc_blur[0]:.2f}%" if acc_blur else "N/A")
            print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<24} | {row[3]:<20}")
