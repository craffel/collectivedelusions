import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Workaround for cuDNN initialization errors on some GPU nodes
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("cuDNN disabled as a workaround for cluster compatibility.")

# Directories
os.makedirs("results", exist_ok=True)

# Datasets & Transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            "mnist": nn.Linear(512, 10),
            "fmnist": nn.Linear(512, 10),
            "cifar10": nn.Linear(512, 10)
        })
        
    def forward(self, x, task_name):
        feats = self.backbone(x)
        return self.heads[task_name](feats)

def get_dataloaders(batch_size=256):
    mnist_test = datasets.MNIST(root="data", train=False, download=False, transform=transform_gray)
    fmnist_test = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform_gray)
    cifar_test = datasets.CIFAR10(root="data", train=False, download=False, transform=transform_color)
    
    return {
        "mnist": DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        "fmnist": DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        "cifar10": DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

# Global caches to avoid re-loading datasets and re-sampling batches
DATASET_CACHE = {}
CALIBRATION_BATCH_CACHE = {}

def get_calibration_batches(task_name, n_samples, count, seed):
    """
    Get 'count' independent batches of size 'n_samples' for calibration from the training set.
    """
    cache_key = (task_name, n_samples, count, seed)
    if cache_key in CALIBRATION_BATCH_CACHE:
        return CALIBRATION_BATCH_CACHE[cache_key]
        
    global DATASET_CACHE
    if task_name not in DATASET_CACHE:
        print(f"Loading {task_name.upper()} train set into memory...")
        if task_name == "mnist":
            DATASET_CACHE[task_name] = datasets.MNIST(root="data", train=True, download=False, transform=transform_gray)
        elif task_name == "fmnist":
            DATASET_CACHE[task_name] = datasets.FashionMNIST(root="data", train=True, download=False, transform=transform_gray)
        elif task_name == "cifar10":
            DATASET_CACHE[task_name] = datasets.CIFAR10(root="data", train=True, download=False, transform=transform_color)
            
    ds = DATASET_CACHE[task_name]
    g = torch.Generator()
    g.manual_seed(seed)
    # Using num_workers=0 to avoid process spawn and serialization overhead
    loader = DataLoader(ds, batch_size=n_samples, shuffle=True, generator=g, num_workers=0)
    
    batches = []
    for x, y in loader:
        batches.append(x)
        if len(batches) == count:
            break
            
    CALIBRATION_BATCH_CACHE[cache_key] = batches
    return batches

# Merging functions
def merge_models(progenitor_sd, expert_sds, method="TA", lambda_val=0.4):
    merged_sd = {}
    all_keys = list(progenitor_sd.keys())
    
    for key in all_keys:
        if key.startswith("heads."):
            task_name = key.split(".")[1]
            merged_sd[key] = expert_sds[task_name][key].clone()
        else:
            if method == "WA":
                merged_sd[key] = sum(expert_sds[task][key] for task in expert_sds) / len(expert_sds)
            elif method == "TA":
                p_val = progenitor_sd[key]
                if p_val.is_floating_point():
                    task_vectors_sum = sum(expert_sds[task][key] - p_val for task in expert_sds)
                    merged_sd[key] = p_val + lambda_val * task_vectors_sum
                else:
                    merged_sd[key] = p_val.clone()
    return merged_sd

# PTQ Quantization Functions
def quantize_weight(tensor, bits, mode='per-channel'):
    if bits is None:
        return tensor
    qmax = 2**(bits - 1) - 1
    if mode == 'per-tensor':
        scale = tensor.abs().max().item() / qmax
        if scale == 0:
            return tensor
        tensor_quant = (tensor / scale).round().clamp(-qmax, qmax) * scale
    elif mode == 'per-channel':
        flat = tensor.view(tensor.size(0), -1)
        scale = flat.abs().max(dim=1)[0] / qmax
        scale = scale.view(-1, *([1] * (tensor.dim() - 1)))
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        tensor_quant = (tensor / scale).round().clamp(-qmax, qmax) * scale
    return tensor_quant

def quantize_model_weights(model, bits, mode='per-channel'):
    if bits is None:
        return
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = quantize_weight(module.weight.data, bits, mode)

# Calibration Protocols
def calibrate_de_bn(model, calib_x, task_name):
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    original_momenta = []
    for m in bn_modules:
        original_momenta.append(m.momentum)
        m.momentum = 1.0
        
    model.train()
    with torch.no_grad():
        _ = model(calib_x.to(device), task_name)
        
    for m, orig in zip(bn_modules, original_momenta):
        m.momentum = orig
    model.eval()

def calibrate_rms_bc(model, calib_xs, task_name):
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    S = len(calib_xs)
    
    sum_running_mean = [torch.zeros_like(m.running_mean) for m in bn_modules]
    sum_running_var = [torch.zeros_like(m.running_var) for m in bn_modules]
    
    for s in range(S):
        calib_x = calib_xs[s]
        
        # Temp save & set momentum 1.0
        original_momenta = []
        for m in bn_modules:
            original_momenta.append(m.momentum)
            m.momentum = 1.0
            
        model.train()
        with torch.no_grad():
            _ = model(calib_x.to(device), task_name)
            
        # Accumulate stats
        for idx, m in enumerate(bn_modules):
            sum_running_mean[idx] += m.running_mean
            sum_running_var[idx] += m.running_var
            
        # Restore momenta
        for m, orig in zip(bn_modules, original_momenta):
            m.momentum = orig
            
    # Compute and set average
    for idx, m in enumerate(bn_modules):
        m.running_mean.copy_(sum_running_mean[idx] / S)
        m.running_var.copy_(sum_running_var[idx] / S)
        
    model.eval()

def calibrate_dem_bc(model, calib_xs, task_name, beta=0.1):
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    
    original_momenta = []
    for m in bn_modules:
        original_momenta.append(m.momentum)
        m.momentum = beta
        
    model.train()
    with torch.no_grad():
        for calib_x in calib_xs:
            _ = model(calib_x.to(device), task_name)
            
    for m, orig in zip(bn_modules, original_momenta):
        m.momentum = orig
        
    model.eval()

# Global cache for pre-loaded test tensors in GPU memory
TEST_BATCH_CACHE = {}

def get_preloaded_test_data(loaders):
    """
    Pre-loads and caches the entire test loaders into GPU memory as list of batches.
    """
    global TEST_BATCH_CACHE
    if len(TEST_BATCH_CACHE) > 0:
        return TEST_BATCH_CACHE
        
    print("Pre-loading and caching test datasets into GPU memory...")
    for task_name, loader in loaders.items():
        batches = []
        for x, y in loader:
            batches.append((x.to(device), y.to(device)))
        TEST_BATCH_CACHE[task_name] = batches
    print("Test datasets successfully cached!")
    return TEST_BATCH_CACHE

# Evaluator
def evaluate_model(model, loaders, task_name, noise_severity=0.0):
    model.eval()
    
    # Use pre-loaded and cached GPU batches
    cached_batches = get_preloaded_test_data(loaders)[task_name]
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in cached_batches:
            if noise_severity > 0.0:
                # Add noise to a clone to avoid mutating cached test images in-place
                x = x + torch.randn_like(x) * noise_severity
            logits = model(x, task_name)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total

def main_benchmark():
    # Load loaders
    loaders = get_dataloaders()
    
    # Check if expert checkpoints are available
    checkpoints = ["mnist", "fmnist", "cifar10"]
    expert_sds = {}
    
    print("Loading checkpoints...")
    try:
        progenitor_sd = torch.load("checkpoints/progenitor.pt", map_location="cpu")
        for task in checkpoints:
            expert_sds[task] = torch.load(f"checkpoints/{task}_expert.pt", map_location="cpu")
    except FileNotFoundError as e:
        print(f"Error loading checkpoints: {e}")
        print("Please train experts first.")
        return
        
    # Evaluate Experts (Oracles)
    print("\n--- Evaluating Experts (Oracles) ---")
    oracle_accs = {}
    for task in checkpoints:
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(expert_sds[task])
        acc = evaluate_model(model, loaders, task)
        oracle_accs[task] = acc
        print(f"Oracle {task.upper()} Accuracy: {acc*100:.2f}%")
    oracle_avg = sum(oracle_accs.values()) / 3
    print(f"Oracle Average: {oracle_avg*100:.2f}%")
    
    # Define Sweep configurations
    merging_methods = ["WA", "TA"]
    precision_regimes = [
        {"name": "FP32", "bits": None, "mode": None},
        {"name": "INT8-Tensor", "bits": 8, "mode": "per-tensor"},
        {"name": "INT8-Channel", "bits": 8, "mode": "per-channel"},
        {"name": "INT4-Channel", "bits": 4, "mode": "per-channel"},
        {"name": "INT3-Channel", "bits": 3, "mode": "per-channel"},
        {"name": "INT2-Channel", "bits": 2, "mode": "per-channel"}
    ]
    
    # Number of random seeds to evaluate calibration variance
    num_seeds = 5
    seeds = [100 + i for i in range(num_seeds)]
    
    results = {
        "oracle": oracle_accs,
        "oracle_avg": oracle_avg,
        "sweeps": []
    }
    
    # Run Sweeps
    for m_method in merging_methods:
        # For TA, we use lambda = 0.4 based on literature
        lambda_val = 0.4 if m_method == "TA" else 0.0
        print(f"\n==========================================")
        print(f"Merging Method: {m_method} (lambda={lambda_val})")
        print(f"==========================================")
        
        # Merge weights in FP32
        base_merged_sd = merge_models(progenitor_sd, expert_sds, method=m_method, lambda_val=lambda_val)
        
        for prec in precision_regimes:
            p_name = prec["name"]
            bits = prec["bits"]
            q_mode = prec["mode"]
            print(f"\n--- Precision: {p_name} ---")
            
            # Calibration configurations
            # Each config is (calib_name, N, S, beta)
            calib_configs = [
                ("No-Cal", 0, 0, 0.0),
                # Standard DE-BN (S=1, momentum=1.0)
                ("DE-BN-16", 16, 1, 1.0),
                ("DE-BN-32", 32, 1, 1.0),
                ("DE-BN-64", 64, 1, 1.0),
                # Proposed Robust Multi-Seed Calibration (RMS-BC)
                ("RMS-BC-16x4", 16, 4, 1.0),
                ("RMS-BC-32x4", 32, 4, 1.0),
                ("RMS-BC-16x8", 16, 8, 1.0),
                ("RMS-BC-32x8", 32, 8, 1.0),
                # Proposed Data-Efficient Momentum Calibration (DEM-BC)
                ("DEM-BC-16x4-b0.2", 16, 4, 0.2),
                ("DEM-BC-32x4-b0.2", 32, 4, 0.2),
                ("DEM-BC-16x8-b0.2", 16, 8, 0.2),
                ("DEM-BC-32x8-b0.2", 32, 8, 0.2),
                ("DEM-BC-32x8-b0.1", 32, 8, 0.1),
                ("DEM-BC-32x8-b0.3", 32, 8, 0.3)
            ]
            
            for calib_name, N, S, beta in calib_configs:
                # To check variance, we run across multiple seeds
                run_accuracies = []
                
                # If No-Cal, we only need to run once since it has no stochasticity
                eval_seeds = [100] if calib_name == "No-Cal" else seeds
                
                for seed in eval_seeds:
                    # Initialize model from merged weights
                    model = MultiTaskResNet18().to(device)
                    model.load_state_dict(base_merged_sd)
                    
                    # Apply weight quantization
                    quantize_model_weights(model, bits, q_mode)
                    
                    # Apply Calibration
                    if calib_name != "No-Cal":
                        # Calibrate each task independently
                        for task in checkpoints:
                            calib_xs = get_calibration_batches(task, N, S, seed)
                            if "RMS-BC" in calib_name:
                                calibrate_rms_bc(model, calib_xs, task)
                            elif "DEM-BC" in calib_name:
                                calibrate_dem_bc(model, calib_xs, task, beta=beta)
                            else: # DE-BN
                                calibrate_de_bn(model, calib_xs[0], task)
                                
                    # Evaluate on all tasks
                    task_accs = {}
                    for task in checkpoints:
                        task_accs[task] = evaluate_model(model, loaders, task)
                    avg_acc = sum(task_accs.values()) / 3
                    run_accuracies.append(avg_acc)
                    
                # Compute statistics
                mean_acc = np.mean(run_accuracies)
                std_acc = np.std(run_accuracies) if len(run_accuracies) > 1 else 0.0
                min_acc = np.min(run_accuracies)
                max_acc = np.max(run_accuracies)
                
                print(f"[{calib_name}] Mean: {mean_acc*100:.2f}% | Std: {std_acc*100:.4f}% | Min: {min_acc*100:.2f}% | Max: {max_acc*100:.2f}%")
                
                results["sweeps"].append({
                    "merging": m_method,
                    "precision": p_name,
                    "calibration": calib_name,
                    "samples_N": N,
                    "batches_S": S,
                    "beta": beta,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "min_acc": min_acc,
                    "max_acc": max_acc,
                    "all_accs": run_accuracies
                })
                
    # Run Environmental Robustness Sweep under Task Arithmetic INT4-Channel
    print("\n==========================================")
    print("Running Environmental Robustness Sweep (TA, INT4-Channel)")
    print("==========================================")
    
    robustness_results = []
    base_merged_sd_ta = merge_models(progenitor_sd, expert_sds, method="TA", lambda_val=0.4)
    
    rob_calib_configs = [
        ("No-Cal", 0, 0, 0.0),
        ("DE-BN-32", 32, 1, 1.0),
        ("RMS-BC-32x8", 32, 8, 1.0),
        ("DEM-BC-32x8-b0.1", 32, 8, 0.1)
    ]
    noise_levels = [0.0, 0.05, 0.1]
    
    for calib_name, N, S, beta in rob_calib_configs:
        for noise in noise_levels:
            run_accuracies = []
            eval_seeds = [100] if calib_name == "No-Cal" else seeds
            
            for seed in eval_seeds:
                model = MultiTaskResNet18().to(device)
                model.load_state_dict(base_merged_sd_ta)
                quantize_model_weights(model, 4, "per-channel")
                
                if calib_name != "No-Cal":
                    for task in checkpoints:
                        calib_xs = get_calibration_batches(task, N, S, seed)
                        if "RMS-BC" in calib_name:
                            calibrate_rms_bc(model, calib_xs, task)
                        elif "DEM-BC" in calib_name:
                            calibrate_dem_bc(model, calib_xs, task, beta=beta)
                        else:
                            calibrate_de_bn(model, calib_xs[0], task)
                            
                task_accs = {}
                for task in checkpoints:
                    task_accs[task] = evaluate_model(model, loaders, task, noise_severity=noise)
                avg_acc = sum(task_accs.values()) / 3
                run_accuracies.append(avg_acc)
                
            mean_acc = np.mean(run_accuracies)
            std_acc = np.std(run_accuracies) if len(run_accuracies) > 1 else 0.0
            
            print(f"[Robustness - {calib_name} - Noise {noise}] Mean: {mean_acc*100:.2f}% | Std: {std_acc*100:.4f}%")
            
            robustness_results.append({
                "calibration": calib_name,
                "noise_severity": noise,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "all_accs": run_accuracies
            })
            
    results["robustness"] = robustness_results
                
    # Save results to JSON
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nBenchmark completed. Results saved to results/results.json")

if __name__ == "__main__":
    main_benchmark()
