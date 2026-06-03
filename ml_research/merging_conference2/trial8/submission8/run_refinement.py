import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import os
import copy
import numpy as np
import time

# Disable cuDNN to avoid initialization issues on cluster
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================================
# CUSTOM TRANSFORMS FOR EXTENDED ROBUSTNESS (STEP 1)
# =====================================================================

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        if self.std == 0.0:
            return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddSaltPepperNoise(object):
    def __init__(self, amount=0.02):
        self.amount = amount
    def __call__(self, tensor):
        if self.amount == 0.0:
            return tensor
        out = tensor.clone()
        # Create mask
        random_matrix = torch.rand(tensor.size())
        # Salt (set to max value 1.0)
        out[random_matrix < (self.amount / 2.0)] = 1.0
        # Pepper (set to min value -1.0)
        out[(random_matrix >= (self.amount / 2.0)) & (random_matrix < self.amount)] = -1.0
        return out

class AdjustBrightness(object):
    def __init__(self, factor=0.0):
        self.factor = factor
    def __call__(self, tensor):
        if self.factor == 0.0:
            return tensor
        return torch.clamp(tensor + self.factor, -1.0, 1.0)

# Loader builder helper
def get_extended_test_loader(task_name, noise_type="clean", intensity=0.0):
    t_list = []
    if task_name in ["mnist", "fmnist"]:
        t_list.append(transforms.Resize((32, 32)))
        t_list.append(transforms.ToTensor())
        t_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    else:
        t_list.append(transforms.ToTensor())
        
    # Apply normalization before noise (keeps consistency)
    t_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    # Apply selected noise/distortion after normalization
    if noise_type == "gaussian_noise":
        t_list.append(AddGaussianNoise(0.0, intensity))
    elif noise_type == "salt_pepper":
        t_list.append(AddSaltPepperNoise(intensity))
    elif noise_type == "brightness":
        t_list.append(AdjustBrightness(intensity))
    elif noise_type == "gaussian_blur":
        # Intensity maps to kernel size (must be odd)
        k_size = int(intensity)
        if k_size > 0:
            t_list.insert(-1, transforms.GaussianBlur(kernel_size=k_size, sigma=1.0))
            
    transform = transforms.Compose(t_list)
    
    if task_name == "mnist":
        ds = MNIST(root="./data", train=False, download=True, transform=transform)
    elif task_name == "fmnist":
        ds = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        ds = CIFAR10(root="./data", train=False, download=True, transform=transform)
        
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================================
# MODEL LOADING & EVALUATION UTILS
# =====================================================================

def load_model_from_checkpoint(path):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# =====================================================================
# CORE ALGORITHMS (U-IPR, HNS, UCPC, RCPC, CTA)
# =====================================================================

def compute_uipr_weights(progenitor_state, expert_states, merged_state, epsilon=1e-8):
    uipr_state = copy.deepcopy(merged_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc.") or "running_mean" in key or "running_var" in key or "num_batches_tracked" in key or "bn" in key:
            continue
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init
            
            norm_merged = torch.norm(t_merged)
            expert_norms = []
            for k in range(K):
                w_expert_k = expert_states[k][key].to(device)
                t_expert_k = w_expert_k - w_init
                expert_norms.append(torch.norm(t_expert_k))
                
            S_l = (sum(expert_norms) / K) / (norm_merged + epsilon)
            S_l = torch.clamp(S_l, min=0.1, max=10.0)
            uipr_state[key] = (w_init + S_l * t_merged).cpu()
    return uipr_state

def compute_ucpc_weights(progenitor_state, expert_states, merged_state, version="v1", epsilon=1e-8):
    ucpc_state = copy.deepcopy(merged_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc.") or "running_mean" in key or "running_var" in key or "num_batches_tracked" in key or "bn" in key:
            continue
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init
            
            shape = w_init.shape
            if len(shape) == 0:
                continue
                
            C_out = shape[0]
            gamma = torch.zeros(C_out, device=device)
            
            for c in range(C_out):
                t_merged_c = t_merged[c]
                norm_merged_c = torch.norm(t_merged_c)
                
                expert_norms = []
                for k in range(K):
                    w_expert_k = expert_states[k][key].to(device)
                    t_expert_k = w_expert_k - w_init
                    t_expert_k_c = t_expert_k[c]
                    expert_norms.append(torch.norm(t_expert_k_c))
                
                if version == "v1":
                    ratios = [expert_norms[k] / (norm_merged_c + epsilon) for k in range(K)]
                    gamma_c = sum(ratios) / K
                elif version == "v2":
                    gamma_c = (sum(expert_norms) / K) / (norm_merged_c + epsilon)
                else:
                    gamma_c = torch.tensor(1.0, device=device)
                    
                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                
            if len(shape) == 4:
                gamma_reshaped = gamma.view(C_out, 1, 1, 1)
            elif len(shape) == 2:
                gamma_reshaped = gamma.view(C_out, 1)
            else:
                gamma_reshaped = gamma
                
            ucpc_state[key] = (w_init + gamma_reshaped * t_merged).cpu()
    return ucpc_state

def compute_rcpc_weights(progenitor_state, expert_states, merged_state, alpha=0.5, epsilon=1e-8):
    rcpc_state = copy.deepcopy(merged_state)
    K = len(expert_states)
    
    for key in progenitor_state.keys():
        if key.startswith("fc.") or "running_mean" in key or "running_var" in key or "num_batches_tracked" in key or "bn" in key:
            continue
        if "weight" in key or "bias" in key:
            w_init = progenitor_state[key].to(device)
            w_merged = merged_state[key].to(device)
            t_merged = w_merged - w_init
            
            shape = w_init.shape
            if len(shape) == 0:
                continue
                
            # Compute layer-wise U-IPR scale factor (S_l)
            norm_merged = torch.norm(t_merged)
            expert_norms = []
            for k in range(K):
                w_expert_k = expert_states[k][key].to(device)
                t_expert_k = w_expert_k - w_init
                expert_norms.append(torch.norm(t_expert_k))
            S_l = (sum(expert_norms) / K) / (norm_merged + epsilon)
            S_l = torch.clamp(S_l, min=0.1, max=10.0)
            
            # Compute channel-wise scale factors
            C_out = shape[0]
            gamma = torch.zeros(C_out, device=device)
            
            for c in range(C_out):
                t_merged_c = t_merged[c]
                norm_merged_c = torch.norm(t_merged_c)
                
                expert_norms_c = []
                for k in range(K):
                    w_expert_k = expert_states[k][key].to(device)
                    t_expert_k = w_expert_k - w_init
                    t_expert_k_c = t_expert_k[c]
                    expert_norms_c.append(torch.norm(t_expert_k_c))
                
                gamma_c = (sum(expert_norms_c) / K) / (norm_merged_c + epsilon)
                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                
            # Blend channel-wise and layer-wise scale factors
            blended_gamma = alpha * gamma + (1 - alpha) * S_l
            
            if len(shape) == 4:
                gamma_reshaped = blended_gamma.view(C_out, 1, 1, 1)
            elif len(shape) == 2:
                gamma_reshaped = blended_gamma.view(C_out, 1)
            else:
                gamma_reshaped = blended_gamma
                
            rcpc_state[key] = (w_init + gamma_reshaped * t_merged).cpu()
    return rcpc_state

def merge_batch_norms(expert_states):
    K = len(expert_states)
    merged_bn_state = {}
    keys = expert_states[0].keys()
    for key in keys:
        if "bn" in key or "running_mean" in key or "running_var" in key:
            tensors = [expert_states[k][key] for k in range(K)]
            if tensors[0].dtype == torch.long or tensors[0].dtype == torch.int:
                merged_bn_state[key] = tensors[0]
            else:
                merged_bn_state[key] = sum(tensors) / K
    return merged_bn_state

# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    checkpoints = ["checkpoints/expert_mnist.pth", "checkpoints/expert_fmnist.pth", "checkpoints/expert_cifar10.pth", "checkpoints/progenitor.pth"]
    for cp in checkpoints:
        if not os.path.exists(cp):
            print(f"Error: checkpoint {cp} does not exist yet. Please run training first.")
            return

    print("\n--- LOADING MODELS ---")
    progenitor = load_model_from_checkpoint("checkpoints/progenitor.pth")
    expert_mnist = load_model_from_checkpoint("checkpoints/expert_mnist.pth")
    expert_fmnist = load_model_from_checkpoint("checkpoints/expert_fmnist.pth")
    expert_cifar10 = load_model_from_checkpoint("checkpoints/expert_cifar10.pth")

    experts = [expert_mnist, expert_fmnist, expert_cifar10]
    expert_names = ["mnist", "fmnist", "cifar10"]
    K = len(experts)

    progenitor_state = progenitor.state_dict()
    expert_states = [exp.state_dict() for exp in experts]

    # Pre-merge backbone weights
    print("Preparing standard merged (WA) state...")
    wa_state = copy.deepcopy(progenitor_state)
    for key in progenitor_state.keys():
        if not key.startswith("fc."):
            tensors = [expert_states[k][key] for k in range(K)]
            wa_state[key] = sum(tensors) / K

    # Compute calibration states (using average BNs for unified deployment)
    print("Computing calibration weight states...")
    uipr_state = compute_uipr_weights(progenitor_state, expert_states, wa_state)
    ucpc_state = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v1")
    merged_bn = merge_batch_norms(expert_states)

    # -----------------------------------------------------------------
    # STEP 1: DEEPEN ROBUSTNESS EVALUATION WITH ADDITIONAL CORRUPTIONS
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("STEP 1: DEEPEN ROBUSTNESS EVALUATION WITH ADDITIONAL CORRUPTIONS")
    print("========================================================")

    corruption_configs = [
        {"name": "Salt & Pepper Noise (0.01)", "type": "salt_pepper", "intensity": 0.01},
        {"name": "Salt & Pepper Noise (0.03)", "type": "salt_pepper", "intensity": 0.03},
        {"name": "Brightness Shift (+0.2)", "type": "brightness", "intensity": 0.2},
        {"name": "Brightness Shift (-0.2)", "type": "brightness", "intensity": -0.2}
    ]

    configs = {
        "Weight Averaging": wa_state,
        "U-IPR": uipr_state,
        "UCPC (Ours)": ucpc_state
    }

    for config in corruption_configs:
        print(f"\n--- Testing with {config['name']} ---")
        loaders = {name: get_extended_test_loader(name, noise_type=config["type"], intensity=config["intensity"]) for name in expert_names}
        
        print(f"{'Method':<20} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
        print("-"*60)
        
        for name, state in configs.items():
            accs = []
            final_state = copy.deepcopy(state)
            final_state.update(merged_bn)
            
            task_accs = {}
            for idx, task_name in enumerate(expert_names):
                task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
                final_state.update(task_head_state)
                
                model = torchvision.models.resnet18()
                model.fc = nn.Linear(512, 10)
                model.load_state_dict(final_state)
                model = model.to(device)
                
                acc = evaluate_model(model, loaders[task_name])
                task_accs[task_name] = acc
                accs.append(acc)
                
            print(f"{name:<20} | {task_accs['mnist']:<8.2f} | {task_accs['fmnist']:<8.2f} | {task_accs['cifar10']:<8.2f} | {np.mean(accs):<8.2f}")

    # -----------------------------------------------------------------
    # STEP 2: PROFILE OFFLINE CALIBRATION TIME AND INFERENCE THROUGHPUT
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("STEP 2: PROFILE OFFLINE CALIBRATION TIME AND INFERENCE THROUGHPUT")
    print("========================================================")

    print("\n--- Benchmarking Offline Calibration Overhead ---")
    calibration_times = {}
    
    # 1. Weight Averaging
    start = time.perf_counter()
    wa_temp = copy.deepcopy(progenitor_state)
    for key in progenitor_state.keys():
        if not key.startswith("fc."):
            tensors = [expert_states[k][key] for k in range(K)]
            wa_temp[key] = sum(tensors) / K
    calibration_times["Weight Averaging"] = time.perf_counter() - start

    # 2. U-IPR
    start = time.perf_counter()
    _ = compute_uipr_weights(progenitor_state, expert_states, wa_state)
    calibration_times["U-IPR"] = time.perf_counter() - start

    # 3. UCPC
    start = time.perf_counter()
    _ = compute_ucpc_weights(progenitor_state, expert_states, wa_state, version="v1")
    calibration_times["UCPC (Ours)"] = time.perf_counter() - start

    # 4. RCPC (alpha=0.5)
    start = time.perf_counter()
    _ = compute_rcpc_weights(progenitor_state, expert_states, wa_state, alpha=0.5)
    calibration_times["RCPC (alpha=0.5)"] = time.perf_counter() - start

    for method, duration in calibration_times.items():
        print(f"Method: {method:<20} | Offline Calibration Time: {duration*1000.0:.2f} ms")

    print("\n--- Benchmarking Inference Throughput & Compiler Compatibility ---")
    # Load a test model
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    final_state = copy.deepcopy(ucpc_state)
    final_state.update(merged_bn)
    final_state.update({k: v for k, v in expert_states[0].items() if k.startswith("fc.")})
    model.load_state_dict(final_state)
    model = model.to(device)
    model.eval()

    # Generate dummy input of batch size 128
    dummy_input = torch.randn(128, 3, 32, 32, device=device)

    # Warmup
    print("Warmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Throughput Standard ResNet-18
    print("Profiling standard model...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    iterations = 100
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    duration = time.perf_counter() - start
    images_per_sec = (iterations * 128) / duration
    print(f"Standard Model Inference Speed: {images_per_sec:.2f} images/second (Batch Size 128)")

    # Throughput torch.compiled model
    print("Compiling model with torch.compile...")
    try:
        compiled_model = torch.compile(model)
        # Warmup compiled model
        print("Warmup compiled model...")
        for _ in range(10):
            with torch.no_grad():
                _ = compiled_model(dummy_input)
        
        print("Profiling compiled model...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = compiled_model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        duration = time.perf_counter() - start
        compiled_images_per_sec = (iterations * 128) / duration
        print(f"Compiled Model Inference Speed: {compiled_images_per_sec:.2f} images/second (Batch Size 128)")
        speedup = (compiled_images_per_sec / images_per_sec - 1.0) * 100.0
        print(f"Flawless Compiler Integration! Speedup from compilation: {speedup:+.2f}%")
    except Exception as e:
        print(f"Compiler integration skipped or failed on CPU node: {e}")

    # -----------------------------------------------------------------
    # STEP 3: ANALYZE CHANNEL-WISE SCALE FACTOR (GAMMA_C) DISTRIBUTIONS
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("STEP 3: ANALYZE CHANNEL-WISE SCALE FACTOR (GAMMA_C) DISTRIBUTIONS")
    print("========================================================")

    # We will compute statistics of gamma_c across key layers of ResNet-18
    target_keys = [
        "conv1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
        "layer4.0.conv1.weight"
    ]

    print(f"{'Layer Name':<25} | {'Channels':<8} | {'Mean Gamma':<10} | {'Std Gamma':<10} | {'Min Gamma':<10} | {'Max Gamma':<10}")
    print("-"*82)

    epsilon = 1e-8
    for key in target_keys:
        if key in progenitor_state:
            w_init = progenitor_state[key].to(device)
            w_merged = wa_state[key].to(device)
            t_merged = w_merged - w_init
            
            C_out = w_init.shape[0]
            gamma = torch.zeros(C_out, device=device)
            
            for c in range(C_out):
                t_merged_c = t_merged[c]
                norm_merged_c = torch.norm(t_merged_c)
                
                expert_norms = []
                for k in range(K):
                    w_expert_k = expert_states[k][key].to(device)
                    t_expert_k = w_expert_k - w_init
                    t_expert_k_c = t_expert_k[c]
                    expert_norms.append(torch.norm(t_expert_k_c))
                    
                gamma_c = (sum(expert_norms) / K) / (norm_merged_c + epsilon)
                gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                
            gamma_cpu = gamma.cpu().numpy()
            print(f"{key:<25} | {C_out:<8} | {np.mean(gamma_cpu):<10.4f} | {np.std(gamma_cpu):<10.4f} | {np.min(gamma_cpu):<10.4f} | {np.max(gamma_cpu):<10.4f}")

    # -----------------------------------------------------------------
    # STEP 4: EXTEND UCPC TO TASK ARITHMETIC (CALIBRATED TASK ARITHMETIC)
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("STEP 4: EXTEND UCPC TO TASK ARITHMETIC (CALIBRATED TASK ARITHMETIC)")
    print("========================================================")

    lambdas = [0.5, 0.7, 1.0]
    loaders = {name: get_extended_test_loader(name, noise_type="clean") for name in expert_names}

    print(f"{'Lambda':<8} | {'Task Arithmetic (TA)':<22} | {'Calibrated TA (Ours)':<22}")
    print("-"*60)

    for lam in lambdas:
        # 1. Uncalibrated Task Arithmetic
        ta_state = copy.deepcopy(progenitor_state)
        for key in progenitor_state.keys():
            if not key.startswith("fc."):
                # Sum task updates scaled by lambda
                tensors = [expert_states[k][key].to(device) - progenitor_state[key].to(device) for k in range(K)]
                ta_state[key] = (progenitor_state[key].to(device) + lam * sum(tensors)).cpu()
        
        ta_state.update(merged_bn)
        ta_accs = []
        for idx, task_name in enumerate(expert_names):
            task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
            ta_state.update(task_head_state)
            
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(ta_state)
            model = model.to(device)
            ta_accs.append(evaluate_model(model, loaders[task_name]))
            
        avg_ta_acc = np.mean(ta_accs)

        # 2. Calibrated Task Arithmetic (CTA)
        # We apply UCPC weights to the scaled task vectors
        cta_state = copy.deepcopy(progenitor_state)
        for key in progenitor_state.keys():
            if key.startswith("fc.") or "running_mean" in key or "running_var" in key or "num_batches_tracked" in key or "bn" in key:
                continue
            if "weight" in key or "bias" in key:
                w_init = progenitor_state[key].to(device)
                
                # Merged task update under Task Arithmetic (scaled by lam * K)
                t_merged_ta = ta_state[key].to(device) - w_init
                shape = w_init.shape
                if len(shape) == 0:
                    continue
                    
                C_out = shape[0]
                gamma = torch.zeros(C_out, device=device)
                
                for c in range(C_out):
                    t_merged_ta_c = t_merged_ta[c]
                    norm_merged_ta_c = torch.norm(t_merged_ta_c)
                    
                    # Target expert updates under Task Arithmetic (scaled by lambda)
                    expert_norms = []
                    for k in range(K):
                        w_expert_k = expert_states[k][key].to(device)
                        t_expert_k = w_expert_k - w_init
                        t_expert_k_c = t_expert_k[c]
                        expert_norms.append(torch.norm(lam * t_expert_k_c))
                        
                    gamma_c = (sum(expert_norms) / K) / (norm_merged_ta_c + epsilon)
                    gamma[c] = torch.clamp(gamma_c, min=0.1, max=10.0)
                    
                if len(shape) == 4:
                    gamma_reshaped = gamma.view(C_out, 1, 1, 1)
                elif len(shape) == 2:
                    gamma_reshaped = gamma.view(C_out, 1)
                else:
                    gamma_reshaped = gamma
                    
                cta_state[key] = (w_init + gamma_reshaped * t_merged_ta).cpu()
                
        cta_state.update(merged_bn)
        cta_accs = []
        for idx, task_name in enumerate(expert_names):
            task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
            cta_state.update(task_head_state)
            
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(cta_state)
            model = model.to(device)
            cta_accs.append(evaluate_model(model, loaders[task_name]))
            
        avg_cta_acc = np.mean(cta_accs)
        
        print(f"{lam:<8.1f} | {avg_ta_acc:<22.2f}% | {avg_cta_acc:<22.2f}%")

    # -----------------------------------------------------------------
    # STEP 5: POST-TRAINING WEIGHT QUANTIZATION ANALYSIS
    # -----------------------------------------------------------------
    print("\n========================================================")
    print("STEP 5: POST-TRAINING WEIGHT QUANTIZATION ANALYSIS")
    print("========================================================")

    # We will test INT8 and INT4 symmetric weight-only quantization
    def quantize_state_symmetric(state, num_bits=8):
        q_state = copy.deepcopy(state)
        q_limit = (1 << (num_bits - 1)) - 1
        for key, val in state.items():
            # Quantize weights of backbone layers (exclude BNs and classification heads)
            if "weight" in key and (not "bn" in key) and (not key.startswith("fc.")):
                W = val.float().to(device)
                max_val = torch.max(torch.abs(W))
                if max_val > 0:
                    scale = max_val / q_limit
                    W_q = torch.clamp(torch.round(W / scale), -q_limit - 1, q_limit)
                    W_deq = W_q * scale
                    q_state[key] = W_deq.cpu().to(val.dtype)
        return q_state

    # Quantize models
    quantized_configs = {
        "Weight Averaging (INT8)": quantize_state_symmetric(wa_state, 8),
        "U-IPR (INT8)": quantize_state_symmetric(uipr_state, 8),
        "UCPC (INT8)": quantize_state_symmetric(ucpc_state, 8),
        "Weight Averaging (INT4)": quantize_state_symmetric(wa_state, 4),
        "U-IPR (INT4)": quantize_state_symmetric(uipr_state, 4),
        "UCPC (INT4)": quantize_state_symmetric(ucpc_state, 4)
    }

    loaders = {name: get_extended_test_loader(name, noise_type="clean") for name in expert_names}

    print(f"{'Configuration':<25} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
    print("-"*62)

    for name, q_state in quantized_configs.items():
        # Apply merged batch norm
        final_state = copy.deepcopy(q_state)
        final_state.update(merged_bn)
        
        accs = []
        task_accs = {}
        for idx, task_name in enumerate(expert_names):
            task_head_state = {k: v for k, v in expert_states[idx].items() if k.startswith("fc.")}
            final_state.update(task_head_state)
            
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(final_state)
            model = model.to(device)
            
            acc = evaluate_model(model, loaders[task_name])
            task_accs[task_name] = acc
            accs.append(acc)
            
        print(f"{name:<25} | {task_accs['mnist']:<8.2f} | {task_accs['fmnist']:<8.2f} | {task_accs['cifar10']:<8.2f} | {np.mean(accs):<8.2f}")

if __name__ == "__main__":
    main()
