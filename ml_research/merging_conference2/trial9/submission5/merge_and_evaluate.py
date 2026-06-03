import torch
import torch.nn as nn
import argparse
import os
import json
import numpy as np
from models import ResNet18CIFAR, MLPCIFAR
from data_utils import get_dataloader, get_calibration_subset
import torchvision.transforms.functional as F_vision

def get_model(model_type, num_classes=10):
    if model_type == "resnet18":
        return ResNet18CIFAR(num_classes)
    else:
        return MLPCIFAR(num_classes)

def apply_corruption(images, corruption_type, severity=1.0):
    if corruption_type == "none" or severity == 0.0:
        return images
    elif corruption_type == "gaussian_noise":
        # Add zero-mean Gaussian noise scaled by severity
        noise_std = 0.15 * severity
        noise = torch.randn_like(images) * noise_std
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == "gaussian_blur":
        # kernel_size must be odd, sigma is scaled by severity
        kernel_size = 3
        if severity > 1.5:
            kernel_size = 5
        sigma = 0.5 * severity
        return F_vision.gaussian_blur(images, [kernel_size, kernel_size], [sigma, sigma])
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

def quantize_weight(W, num_bits=8):
    if num_bits == 0:
        return W
    qmax = 2**(num_bits - 1) - 1
    max_val = torch.max(torch.abs(W))
    if max_val == 0:
        return W
    delta = max_val / qmax
    W_quant = torch.clamp(torch.round(W / delta), -qmax, qmax) * delta
    return W_quant

def quantize_weight_channelwise(W, num_bits=8):
    if num_bits == 0:
        return W
    if W.dim() < 2:
        return quantize_weight(W, num_bits)
    qmax = 2**(num_bits - 1) - 1
    
    # Vectorized channel-wise quantization
    shape = W.shape
    W_flat = W.view(shape[0], -1)
    max_val = torch.max(torch.abs(W_flat), dim=1, keepdim=True)[0]
    
    # Avoid division by zero
    delta = max_val / qmax
    delta = torch.where(max_val == 0, torch.ones_like(delta), delta)
    
    W_quant_flat = torch.clamp(torch.round(W_flat / delta), -qmax, qmax) * delta
    return W_quant_flat.view(shape)

def quantize_model_weights_(model, num_bits=8, mode="channelwise"):
    """
    In-place weight-only post-training quantization for Conv2d and Linear layers.
    """
    if num_bits == 0:
        return model
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if mode == "channelwise":
                    m.weight.copy_(quantize_weight_channelwise(m.weight, num_bits))
                else:
                    m.weight.copy_(quantize_weight(m.weight, num_bits))
    return model

def calibrate_bn(model, dataset_name, num_samples, device, seed=42):
    """
    Data-efficient BatchNorm stats calibration using a small calibration set.
    """
    if num_samples == 0:
        return model
    
    # Check if there are BN layers to calibrate
    has_bn = any(isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) for m in model.modules())
    if not has_bn:
        return model

    # Put in train mode to estimate batch statistics
    model.train()
    for param in model.parameters():
        param.requires_grad = False
        
    # Reset stats and set momentum to None to calculate standard average
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None
            
    calib_loader = get_calibration_subset(dataset_name, num_samples=num_samples, seed=seed)
    
    with torch.no_grad():
        for images, _ in calib_loader:
            images = images.to(device)
            model(images)
            
    # Set back to eval mode
    model.eval()
    return model

def evaluate_multi_task(model, datasets, device, num_bits=0, corruption="none", severity=0.0, bn_calib_samples=0, seed=42, batch_size=256, num_workers=0):
    """
    Evaluate the merged model across multiple datasets.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    results = {}
    total_acc = 0.0
    
    # We copy the model to avoid corrupting the original weights with quantization or calibration in successive runs
    import copy
    
    for dataset_name in datasets:
        # Clone model for task-specific evaluation (important if we do task-specific BN calibration)
        eval_model = copy.deepcopy(model).to(device)
        
        # 1. Calibrate BatchNorm if requested (DE-BN is task-specific since it uses task-specific activations!)
        if bn_calib_samples > 0:
            eval_model = calibrate_bn(eval_model, dataset_name, bn_calib_samples, device, seed=seed)
            
        # 2. Quantize model if requested
        if num_bits > 0:
            eval_model = quantize_model_weights_(eval_model, num_bits, mode="channelwise")
            
        eval_model.eval()
        
        test_loader = get_dataloader(dataset_name, batch_size=batch_size, train=False, num_workers=num_workers)
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                # Apply environmental corruptions to the test images
                images = apply_corruption(images, corruption, severity)
                images, labels = images.to(device), labels.to(device)
                
                outputs = eval_model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = correct / total
        loss_val = running_loss / total
        results[dataset_name] = {"acc": acc, "loss": loss_val}
        total_acc += acc
        
    results["average_acc"] = total_acc / len(datasets)
    return results

# Merging functions
def merge_weight_averaging(progenitor, experts):
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            stacked = torch.stack([state[key] for state in expert_states])
            merged_state[key].copy_(torch.mean(stacked, dim=0))
        else:
            # Copy first expert's non-floating parameters (e.g. tracking steps)
            merged_state[key].copy_(expert_states[0][key])
    return merged

def merge_task_arithmetic(progenitor, experts, lambd=0.3):
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            task_updates = [state[key] - progenitor_state[key] for state in expert_states]
            summed_updates = torch.sum(torch.stack(task_updates), dim=0)
            merged_state[key].copy_(progenitor_state[key] + lambd * summed_updates)
        else:
            # Handle non-float params by averaging or copying
            merged_state[key].copy_(expert_states[0][key])
    return merged

def merge_qr_ipr(progenitor, experts, lambd=0.3, gamma=1.5):
    """
    QR-IPR (from submission 5): Clamps channel-wise scaling factors using Median and MAD.
    """
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            # Only apply scaling resonance to Conv and Linear weights (dim >= 2)
            if merged_state[key].dim() >= 2:
                T_merged = lambd * torch.sum(torch.stack([state[key] - progenitor_state[key] for state in expert_states]), dim=0)
                expert_updates = [state[key] - progenitor_state[key] for state in expert_states]
                
                # Compute channel-wise scale factors
                shape = T_merged.shape
                sc_list = []
                for c in range(shape[0]):
                    nm = torch.norm(T_merged[c], p=2)
                    ne = torch.mean(torch.stack([torch.norm(exp_up[c], p=2) for exp_up in expert_updates]))
                    sc = ne / (nm + 1e-8)
                    sc_list.append(sc)
                
                sc_tensor = torch.stack(sc_list)
                median_s = torch.median(sc_tensor)
                mad_s = torch.median(torch.abs(sc_tensor - median_s))
                
                L = torch.clamp(median_s - gamma * mad_s, min=0.1)
                U = torch.clamp(median_s + gamma * mad_s, max=4.0)
                
                # Clamp and scale channel-wise
                T_cal = torch.zeros_like(T_merged)
                for c in range(shape[0]):
                    s_clamped = torch.clamp(sc_tensor[c], L, U)
                    T_cal[c] = T_merged[c] * s_clamped
                merged_state[key].copy_(progenitor_state[key] + T_cal)
            else:
                # For 1D tensors, use standard task arithmetic
                task_updates = [state[key] - progenitor_state[key] for state in expert_states]
                summed_updates = torch.sum(torch.stack(task_updates), dim=0)
                merged_state[key].copy_(progenitor_state[key] + lambd * summed_updates)
        else:
            merged_state[key].copy_(expert_states[0][key])
    return merged

def merge_ties(progenitor, experts, lambd=0.3, fraction=0.2):
    """
    TIES-Merging: Trims task updates to the top `fraction` magnitude parameters,
    elects consensus signs, and averages updates that match the consensus.
    """
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            task_updates = [state[key] - progenitor_state[key] for state in expert_states]
            
            # 1. Trim step: Keep only top fraction by magnitude
            trimmed_updates = []
            for V in task_updates:
                if V.numel() == 0:
                    trimmed_updates.append(V)
                    continue
                abs_V = torch.abs(V)
                flat_abs_V = abs_V.view(-1)
                
                # Using torch.quantile is extremely robust and fast
                threshold = torch.quantile(flat_abs_V, 1.0 - fraction)
                
                mask = abs_V >= threshold
                trimmed_updates.append(V * mask)
                
            stacked_trimmed = torch.stack(trimmed_updates)
            
            # 2. Elect Sign step
            signs = torch.sign(stacked_trimmed)
            sum_signs = torch.sum(signs, dim=0)
            consensus_sign = torch.sign(sum_signs)
            
            # 3. Disjoint Merge (Aggregate) step
            agreed_mask = (signs == consensus_sign) & (signs != 0)
            summed_updates = torch.sum(stacked_trimmed * agreed_mask, dim=0)
            counts = torch.sum(agreed_mask, dim=0)
            
            average_update = summed_updates / torch.clamp(counts, min=1.0)
            merged_state[key].copy_(progenitor_state[key] + lambd * average_update)
        else:
            merged_state[key].copy_(expert_states[0][key])
    return merged

def merge_dare(progenitor, experts, lambd=0.3, p=0.2):
    """
    DARE (Drop and Rescale): Randomly drops a fraction `p` of task updates,
    rescales the remaining updates by 1/(1-p), and sums them up.
    """
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            task_updates = [state[key] - progenitor_state[key] for state in expert_states]
            
            dare_updates = []
            for V in task_updates:
                if V.numel() == 0:
                    dare_updates.append(V)
                    continue
                # Create drop mask (1 with probability 1-p, 0 with probability p)
                mask = (torch.rand_like(V) >= p).to(V.dtype)
                # Rescale remaining elements
                V_dare = V * mask / (1.0 - p)
                dare_updates.append(V_dare)
                
            summed_updates = torch.sum(torch.stack(dare_updates), dim=0)
            merged_state[key].copy_(progenitor_state[key] + lambd * summed_updates)
        else:
            merged_state[key].copy_(expert_states[0][key])
    return merged

def ot_1d_wcpr_channelwise(x, y_list):
    """
    Vectorized 1D Optimal Transport map that aligns the channels of x to the Wasserstein barycenter of y_list.
    x and y in y_list have shape [C, D].
    """
    # Sort each expert's channel-wise updates along the feature dimension (dim=1)
    y_sorted_list = [torch.sort(y, dim=1)[0] for y in y_list]
    y_target_sorted = torch.mean(torch.stack(y_sorted_list), dim=0)

    # Get the argsort indices of the merged update x
    indices = torch.argsort(x, dim=1)
    
    # Scatter the sorted target values into the positions of indices
    x_cal = torch.zeros_like(x)
    x_cal.scatter_(1, indices, y_target_sorted)
    return x_cal

def ot_1d_wcpr(x, y_list):
    """
    1D Optimal Transport map that aligns the 1D tensor x to the Wasserstein barycenter of y_list.
    """
    y_sorted_list = [torch.sort(y)[0] for y in y_list]
    y_target_sorted = torch.mean(torch.stack(y_sorted_list), dim=0)
    
    indices = torch.argsort(x)
    x_cal = torch.zeros_like(x)
    x_cal[indices] = y_target_sorted
    return x_cal

def merge_wcpr(progenitor, experts, lambd=0.3):
    """
    WCPR (from submission 9): Non-parametric Wasserstein calibration channel-by-channel.
    """
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            if merged_state[key].dim() >= 2:
                # For multi-dimensional weights, do channel-by-channel 1D OT (fully vectorized!)
                shape = merged_state[key].shape
                T_merged = lambd * torch.sum(torch.stack([state[key] - progenitor_state[key] for state in expert_states]), dim=0)
                expert_updates = [state[key] - progenitor_state[key] for state in expert_states]
                
                x = T_merged.view(shape[0], -1)
                y_list = [exp_up.view(shape[0], -1) for exp_up in expert_updates]
                
                x_cal = ot_1d_wcpr_channelwise(x, y_list)
                
                T_cal = x_cal.view(shape)
                merged_state[key].copy_(progenitor_state[key] + T_cal)
            else:
                # 1D weights
                T_merged = lambd * torch.sum(torch.stack([state[key] - progenitor_state[key] for state in expert_states]), dim=0)
                expert_updates = [state[key] - progenitor_state[key] for state in expert_states]
                x = T_merged.view(1, -1)
                y_list = [exp_up.view(1, -1) for exp_up in expert_updates]
                x_cal = ot_1d_wcpr_channelwise(x, y_list)
                merged_state[key].copy_(progenitor_state[key] + x_cal.view(merged_state[key].shape))
        else:
            merged_state[key].copy_(expert_states[0][key])
            
    return merged

def merge_qr_wcpr(progenitor, experts, lambd=0.3, gamma=1.5):
    """
    Our proposed QR-WCPR: Quantization-Robust Wasserstein-Calibrated Parameter Resonance.
    Applies WCPR first, then clamps each channel's dynamic range based on a robust multiplier
    of the experts' maximum absolute task update values.
    """
    merged = copy_deepcopy_model(progenitor)
    merged_state = merged.state_dict()
    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in merged_state.keys():
        if merged_state[key].is_floating_point():
            if merged_state[key].dim() >= 2:
                shape = merged_state[key].shape
                T_merged = lambd * torch.sum(torch.stack([state[key] - progenitor_state[key] for state in expert_states]), dim=0)
                expert_updates = [state[key] - progenitor_state[key] for state in expert_states]
                
                x = T_merged.view(shape[0], -1)
                y_list = [exp_up.view(shape[0], -1) for exp_up in expert_updates]
                
                # 1. 1D OT alignment (WCPR) - fully vectorized!
                x_cal = ot_1d_wcpr_channelwise(x, y_list)
                
                # 2. Robust outlier scaling to protect dynamic range - fully vectorized!
                expert_maxes = torch.stack([torch.max(torch.abs(y), dim=1)[0] for y in y_list])
                median_max = torch.median(expert_maxes, dim=0)[0]
                mad_max = torch.median(torch.abs(expert_maxes - median_max), dim=0)[0]
                
                max_allowed = median_max + gamma * mad_max
                max_allowed = torch.max(max_allowed, median_max) # Ensure it doesn't go below median
                max_allowed = torch.clamp(max_allowed, min=1e-5) # Prevent divide-by-zero
                
                cal_max = torch.max(torch.abs(x_cal), dim=1)[0]
                
                # Scale factor calculation and broadcasting
                scale_factor = torch.where(cal_max > max_allowed, max_allowed / cal_max, torch.ones_like(cal_max))
                x_cal = x_cal * scale_factor.unsqueeze(1)
                
                T_cal = x_cal.view(shape)
                merged_state[key].copy_(progenitor_state[key] + T_cal)
            else:
                # 1D weights: simple WCPR
                T_merged = lambd * torch.sum(torch.stack([state[key] - progenitor_state[key] for state in expert_states]), dim=0)
                expert_updates = [state[key] - progenitor_state[key] for state in expert_states]
                x = T_merged.view(1, -1)
                y_list = [exp_up.view(1, -1) for exp_up in expert_updates]
                x_cal = ot_1d_wcpr_channelwise(x, y_list)
                merged_state[key].copy_(progenitor_state[key] + x_cal.view(merged_state[key].shape))
        else:
            merged_state[key].copy_(expert_states[0][key])
            
    return merged

def copy_deepcopy_model(model):
    import copy
    return copy.deepcopy(model)

def run_experiment_suite(model_type, device, methods_to_run=None):
    print(f"\n==========================================")
    print(f"Running Experiment Suite for {model_type.upper()}")
    print(f"==========================================")
    
    # 1. Load Progenitor and Experts
    progenitor = get_model(model_type)
    progenitor.load_state_dict(torch.load(f"checkpoints/progenitor_{model_type}.pt", map_location="cpu"))
    
    experts = []
    for ds in ["mnist", "fmnist", "cifar10"]:
        model = get_model(model_type)
        model.load_state_dict(torch.load(f"checkpoints/expert_{ds}_{model_type}.pt", map_location="cpu"))
        experts.append(model)
        
    datasets = ["mnist", "fmnist", "cifar10"]
    
    # Evaluate experts individually to print oracle results
    print("\n--- Expert Oracles (FP32 Clean) ---")
    for idx, ds in enumerate(datasets):
        res = evaluate_multi_task(experts[idx], [ds], device)
        print(f"Expert on {ds.upper()}: {res[ds]['acc']*100:.2f}% accuracy")
        
    # Pre-merge models once to avoid redundant slow merging operations
    print("\nPre-merging models...")
    merging_methods = {
        "WA": lambda: merge_weight_averaging(progenitor, experts),
        "TA (l=0.3)": lambda: merge_task_arithmetic(progenitor, experts, lambd=0.3),
        "TA (l=0.5)": lambda: merge_task_arithmetic(progenitor, experts, lambd=0.5),
        "QR-IPR": lambda: merge_qr_ipr(progenitor, experts, lambd=0.5, gamma=1.5),
        "TIES (l=0.5)": lambda: merge_ties(progenitor, experts, lambd=0.5, fraction=0.2),
        "DARE (l=0.5)": lambda: merge_dare(progenitor, experts, lambd=0.5, p=0.2),
        "WCPR": lambda: merge_wcpr(progenitor, experts, lambd=0.5),
        "QR-WCPR": lambda: merge_qr_wcpr(progenitor, experts, lambd=0.5, gamma=1.5)
    }
    
    if methods_to_run is not None:
        merging_methods = {k: v for k, v in merging_methods.items() if k in methods_to_run}
    
    merged_models = {}
    for name, merge_fn in merging_methods.items():
        print(f"  Merging {name}...")
        merged_models[name] = merge_fn().to(device)
    print("Pre-merging complete!\n")
    
    scenarios = [
        {"name": "FP32 Clean", "bits": 0, "corr": "none", "sev": 0.0, "de_bn": 0},
        {"name": "FP32 Clean + DE-BN (16)", "bits": 0, "corr": "none", "sev": 0.0, "de_bn": 16},
        {"name": "INT8 Quantized", "bits": 8, "corr": "none", "sev": 0.0, "de_bn": 0},
        {"name": "INT8 + DE-BN (16)", "bits": 8, "corr": "none", "sev": 0.0, "de_bn": 16},
        {"name": "INT4 Quantized", "bits": 4, "corr": "none", "sev": 0.0, "de_bn": 0},
        {"name": "INT4 + DE-BN (16)", "bits": 4, "corr": "none", "sev": 0.0, "de_bn": 16},
        {"name": "Noise (sev 1.5)", "bits": 0, "corr": "gaussian_noise", "sev": 1.5, "de_bn": 0},
        {"name": "Noise + DE-BN (16)", "bits": 0, "corr": "gaussian_noise", "sev": 1.5, "de_bn": 16},
        {"name": "Blur (sev 1.5)", "bits": 0, "corr": "gaussian_blur", "sev": 1.5, "de_bn": 0},
        {"name": "Blur + DE-BN (16)", "bits": 0, "corr": "gaussian_blur", "sev": 1.5, "de_bn": 16}
    ]
    
    # Run evaluations across 3 seeds for comprehensive evaluation
    seeds = [42, 43, 44]
    all_results = {}
    for sc in scenarios:
        # Skip DE-BN scenarios for MLP to save time
        if model_type == "mlp" and sc["de_bn"] > 0:
            continue
            
        print(f"\nEvaluating Scenario: {sc['name']}")
        all_results[sc["name"]] = {}
        
        # Check if the scenario is deterministic (no BN calib and no noise)
        is_deterministic = (sc["de_bn"] == 0 and sc["corr"] != "gaussian_noise")
        
        for name, merged_model in merged_models.items():
            # Evaluate across multiple seeds
            seed_results = []
            
            if is_deterministic:
                # Run only once on seed 42 for deterministic scenarios
                res = evaluate_multi_task(
                    merged_model, 
                    datasets, 
                    device, 
                    num_bits=sc["bits"], 
                    corruption=sc["corr"], 
                    severity=sc["sev"], 
                    bn_calib_samples=sc["de_bn"],
                    seed=42,
                    batch_size=1024,
                    num_workers=4
                )
                seed_results = [res, res, res]
            else:
                # Run across all seeds for stochastic scenarios
                for seed in seeds:
                    res = evaluate_multi_task(
                        merged_model, 
                        datasets, 
                        device, 
                        num_bits=sc["bits"], 
                        corruption=sc["corr"], 
                        severity=sc["sev"], 
                        bn_calib_samples=sc["de_bn"],
                        seed=seed,
                        batch_size=1024,
                        num_workers=4
                    )
                    seed_results.append(res)
                
            # Aggregate results across seeds
            avg_accs = [r["average_acc"] for r in seed_results]
            mean_avg_acc = float(np.mean(avg_accs))
            std_avg_acc = float(np.std(avg_accs)) if not is_deterministic else 0.0
            
            aggregated_res = {
                "average_acc": mean_avg_acc,
                "average_acc_std": std_avg_acc,
                "raw_average_accs": avg_accs
            }
            
            for ds in datasets:
                ds_accs = [r[ds]["acc"] for r in seed_results]
                ds_losses = [r[ds]["loss"] for r in seed_results]
                
                aggregated_res[ds] = {
                    "acc": float(np.mean(ds_accs)),
                    "acc_std": float(np.std(ds_accs)) if not is_deterministic else 0.0,
                    "raw_accs": ds_accs,
                    "loss": float(np.mean(ds_losses)),
                    "loss_std": float(np.std(ds_losses)) if not is_deterministic else 0.0
                }
                
            all_results[sc["name"]][name] = aggregated_res
            print(f"  [{name}]: Mean Avg Acc = {mean_avg_acc*100:.2f}% (std={std_avg_acc*100:.2f}%) | MNIST: {aggregated_res['mnist']['acc']*100:.1f}% | FMNIST: {aggregated_res['fmnist']['acc']*100:.1f}% | CIFAR10: {aggregated_res['cifar10']['acc']*100:.1f}%")
            
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model merging techniques")
    parser.add_argument("--model_type", type=str, default="resnet18", choices=["resnet18", "mlp", "both"])
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated list of methods to evaluate")
    parser.add_argument("--patch", action="store_true", help="Patch results into existing results.json instead of overwriting")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid initialization errors
    print(f"Evaluation running on device: {device}")
    
    methods_to_run = None
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(",")]
        
    results = {}
    if args.patch and os.path.exists("results.json"):
        import json
        with open("results.json", "r") as f:
            results = json.load(f)
            
    if args.model_type in ["resnet18", "both"]:
        r18_res = run_experiment_suite("resnet18", device, methods_to_run)
        if args.patch and "resnet18" in results:
            for sc_name, sc_results in r18_res.items():
                if sc_name not in results["resnet18"]:
                    results["resnet18"][sc_name] = {}
                for m_name, m_res in sc_results.items():
                    results["resnet18"][sc_name][m_name] = m_res
        else:
            results["resnet18"] = r18_res
            
    if args.model_type in ["mlp", "both"]:
        mlp_res = run_experiment_suite("mlp", device, methods_to_run)
        if args.patch and "mlp" in results:
            for sc_name, sc_results in mlp_res.items():
                if sc_name not in results["mlp"]:
                    results["mlp"][sc_name] = {}
                for m_name, m_res in sc_results.items():
                    results["mlp"][sc_name][m_name] = m_res
        else:
            results["mlp"] = mlp_res
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSuccessfully saved all experimental results to results.json!")

if __name__ == "__main__":
    main()
