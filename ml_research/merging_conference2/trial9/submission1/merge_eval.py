import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms.functional import gaussian_blur
from torchvision.models import resnet18

class ExpertModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18()
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def get_dataloader(dataset_name, batch_size=256, is_train=True, num_samples=1000):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.MNIST(root="./data", train=is_train, download=True, transform=transform)
    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.FashionMNIST(root="./data", train=is_train, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.CIFAR10(root="./data", train=is_train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    if not is_train:
        # Standard subset of 1000 samples for fast and stable evaluation
        dataset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    else:
        # Limit the calibration sample size if requested
        if num_samples is not None:
            dataset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
            
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)
    return loader

def get_bn_name(conv_name):
    if "downsample.0" in conv_name:
        return conv_name.replace("downsample.0", "downsample.1")
    elif "conv1" in conv_name:
        return conv_name.replace("conv1", "bn1")
    elif "conv2" in conv_name:
        return conv_name.replace("conv2", "bn2")
    return None

def quantize_weight(w, num_bits=8, per_channel=False):
    if num_bits is None:
        return w
    q_max = 2**(num_bits - 1) - 1
    if per_channel and w.dim() >= 2:
        w_quant = w.clone()
        for c in range(w.shape[0]):
            max_val = torch.max(torch.abs(w[c]))
            if max_val > 0:
                delta = max_val / q_max
                w_quant[c] = torch.clamp(torch.round(w[c] / delta), -q_max, q_max) * delta
        return w_quant
    else:
        max_val = torch.max(torch.abs(w))
        if max_val == 0:
            return w
        delta = max_val / q_max
        return torch.clamp(torch.round(w / delta), -q_max, q_max) * delta

def evaluate_model(model, task_heads, num_bits=None, per_channel=False, corruption=None, de_bn_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Apply post-training quantization to the model weights
    eval_model = copy.deepcopy(model)
    if num_bits is not None:
        with torch.no_grad():
            for name, param in eval_model.backbone.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    param.copy_(quantize_weight(param.data, num_bits=num_bits, per_channel=per_channel))
                    
    tasks = ["mnist", "fmnist", "cifar10"]
    accuracies = {}
    
    for task in tasks:
        # Load correct evaluation dataset
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        
        # Determine the model to use for this task (with task-specific DE-BN if active)
        if de_bn_samples is not None:
            task_model = copy.deepcopy(eval_model)
            task_model.train()
            for m in task_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.momentum = None
            task_model.fc = copy.deepcopy(task_heads[task]).to(device)
            cal_loader = get_dataloader(task, batch_size=de_bn_samples, is_train=True, num_samples=de_bn_samples)
            with torch.no_grad():
                for inputs, _ in cal_loader:
                    inputs = inputs.to(device)
                    _ = task_model(inputs)
                    break
            task_model.eval()
        else:
            task_model = eval_model
            # Attach the expert head
            task_model.fc = copy.deepcopy(task_heads[task]).to(device)
            task_model.fc.eval()
            
        # Apply head quantization if PTQ is active
        if num_bits is not None:
            with torch.no_grad():
                task_model.fc.weight.copy_(quantize_weight(task_model.fc.weight.data, num_bits=num_bits, per_channel=per_channel))
                
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply corruptions if specified
                if corruption == "noise":
                    inputs = inputs + 0.1 * torch.randn_like(inputs)
                elif corruption == "blur":
                    inputs = gaussian_blur(inputs, kernel_size=[3, 3])
                    
                outputs = task_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracies[task] = 100. * correct / total
        
    accuracies["average"] = sum(accuracies[task] for task in tasks) / len(tasks)
    return accuracies

def run_de_bn(model, task_heads, num_samples=16):
    """
    Data-Efficient BatchNorm calibration.
    Resets running stats and re-estimates them using standard forward pass on unlabeled samples.
    """
    print(f"Running DE-BN calibration with N={num_samples}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal_model = copy.deepcopy(model).to(device)
    
    # Put model in training mode, reset running stats
    cal_model.train()
    for m in cal_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None # Use simple average over the calibration set
            
    tasks = ["mnist", "fmnist", "cifar10"]
    # We do forward passes for each task with their respective heads to align representations
    for task in tasks:
        cal_loader = get_dataloader(task, batch_size=num_samples, is_train=True, num_samples=num_samples)
        cal_model.fc = copy.deepcopy(task_heads[task]).to(device)
        
        # Forward pass on a single batch of size N
        with torch.no_grad():
            for inputs, _ in cal_loader:
                inputs = inputs.to(device)
                _ = cal_model(inputs)
                break # Only use one batch of size N
                
    cal_model.eval()
    return cal_model

def get_model_parameters(model):
    return {name: param.clone() for name, param in model.backbone.named_parameters()}

def merge_models(progenitor_state, expert_states, alg="wa", lam=0.5, drop_rate=0.2):
    """
    Merge the expert backbones.
    """
    merged_state = copy.deepcopy(progenitor_state)
    K = len(expert_states)
    
    if alg == "wa":
        for key in merged_state.keys():
            if "backbone" in key:
                stacked = torch.stack([expert_states[t][key] for t in range(K)])
                if stacked.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    merged_state[key] = torch.mean(stacked, dim=0)
                else:
                    merged_state[key] = expert_states[0][key]
    elif alg == "ta":
        for key in merged_state.keys():
            if "backbone" in key:
                if progenitor_state[key].dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    task_updates = [expert_states[t][key] - progenitor_state[key] for t in range(K)]
                    merged_state[key] = progenitor_state[key] + lam * sum(task_updates)
                else:
                    merged_state[key] = expert_states[0][key]
    elif alg == "ties":
        # Keep top 20% updates, resolve sign conflicts, average sign-consistent updates
        for key in merged_state.keys():
            if "backbone" in key and "weight" in key and progenitor_state[key].dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                updates = [expert_states[t][key] - progenitor_state[key] for t in range(K)]
                # 1. Trimming
                trimmed_updates = []
                for up in updates:
                    flat_up = up.flatten()
                    k_val = int(0.20 * flat_up.numel())
                    if k_val > 0:
                        threshold = torch.topk(torch.abs(flat_up), k_val).values[-1]
                        mask = torch.abs(up) >= threshold
                        trimmed_updates.append(up * mask)
                    else:
                        trimmed_updates.append(torch.zeros_like(up))
                # 2. Sign election
                stacked_trimmed = torch.stack(trimmed_updates)
                signs = torch.sign(stacked_trimmed)
                sum_signs = torch.sum(signs, dim=0)
                elected_sign = torch.sign(sum_signs)
                # 3. Disjoint merge
                mask_sign = signs == elected_sign.unsqueeze(0)
                sum_values = torch.sum(stacked_trimmed * mask_sign, dim=0)
                active_counts = torch.sum(mask_sign.float(), dim=0)
                active_counts[active_counts == 0] = 1.0
                merged_update = sum_values / active_counts
                merged_state[key] = progenitor_state[key] + merged_update
            elif "backbone" in key:
                # For non-weight or non-conv params, just average
                stacked = torch.stack([expert_states[t][key] for t in range(K)])
                if stacked.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    merged_state[key] = torch.mean(stacked, dim=0)
                else:
                    merged_state[key] = expert_states[0][key]
    elif alg == "dare":
        # Random drop with rate drop_rate, rescale, average
        for key in merged_state.keys():
            if "backbone" in key and "weight" in key and progenitor_state[key].dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                updates = [expert_states[t][key] - progenitor_state[key] for t in range(K)]
                rescaled_updates = []
                for up in updates:
                    mask = (torch.rand_like(up) >= drop_rate).float()
                    rescaled_updates.append(up * mask / (1.0 - drop_rate))
                merged_state[key] = progenitor_state[key] + torch.mean(torch.stack(rescaled_updates), dim=0)
            elif "backbone" in key:
                stacked = torch.stack([expert_states[t][key] for t in range(K)])
                if stacked.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    merged_state[key] = torch.mean(stacked, dim=0)
                else:
                    merged_state[key] = expert_states[0][key]
                
    return merged_state

def apply_calibration(model, progenitor_state, expert_states, cal_method="none", dynamic_clip=True, sparsity_compensated=False):
    """
    Apply parameter calibration on the merged model backbone.
    If cal_method is 'cbvc' (our proposed method), we DO NOT change the weights,
    but we scale the BatchNorm running variance!
    """
    K = len(expert_states)
    cal_model = copy.deepcopy(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal_model = cal_model.to(device)
    
    # Standard state dict of the merged model
    merged_state = cal_model.state_dict()
    
    # 1. Compute scaling factors layer-wise or channel-wise
    scale_factors = {}
    sparsity_ratios = {}
    
    with torch.no_grad():
        for name, module in cal_model.backbone.named_modules():
            # We compute calibration scale factors for conv layers with dim >= 2
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_key = f"backbone.{name}.weight"
                if weight_key in merged_state:
                    w_init = progenitor_state[weight_key].to(device)
                    w_merged = merged_state[weight_key].to(device)
                    w_merged_update = w_merged - w_init
                    w_expert_updates = [expert_states[t][weight_key].to(device) - w_init for t in range(K)]
                    
                    if isinstance(module, nn.Conv2d):
                        C_out = module.out_channels
                        s_c = torch.zeros(C_out, device=device)
                        p_c = torch.ones(C_out, device=device)
                        
                        for c in range(C_out):
                            # norm_merged
                            norm_m = torch.norm(w_merged_update[c], p=2)
                            # norm_experts
                            norm_exp = sum(torch.norm(exp_up[c], p=2) for exp_up in w_expert_updates) / K
                            # compute raw scale factor s_c
                            s_c[c] = norm_exp / (norm_m + 1e-8)
                            
                            # Sparsity compensation active parameter ratio
                            if sparsity_compensated:
                                active_params = torch.sum((torch.abs(w_merged_update[c]) > 1e-8).float())
                                total_params = w_merged_update[c].numel()
                                p_c[c] = active_params / total_params if total_params > 0 else 1.0
                                # scale factor compensated by sqrt(p_c)
                                s_c[c] = s_c[c] * torch.sqrt(p_c[c])
                                
                        scale_factors[name] = s_c
                        sparsity_ratios[name] = p_c
                    else:
                        # Linear layer fallback to layer-wise U-IPR
                        norm_m = torch.norm(w_merged_update, p="fro")
                        norm_exp = sum(torch.norm(exp_up, p="fro") for exp_up in w_expert_updates) / K
                        s_l = norm_exp / (norm_m + 1e-8)
                        scale_factors[name] = s_l
                        
    # 2. Apply calibration to the model
    if cal_method == "none":
        return cal_model
        
    elif cal_method in ["hns", "qr-ipr", "u-ipr"]:
        # Weight-scaling calibrations (HNS, QR-IPR, U-IPR)
        print(f"Applying weight-scaling calibration ({cal_method.upper()})...")
        with torch.no_grad():
            for name, module in cal_model.backbone.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    weight_key = f"backbone.{name}.weight"
                    if weight_key in merged_state:
                        s = scale_factors[name]
                        
                        # Clip scaling factors
                        if cal_method == "hns":
                            s_clamped = torch.clamp(s, 0.1, 10.0)
                        elif cal_method == "qr-ipr":
                            # Dynamic clamping using Median and MAD
                            if s.dim() > 0:
                                median_s = torch.median(s)
                                mad_s = torch.median(torch.abs(s - median_s))
                                mad_s = torch.max(mad_s, torch.tensor(1e-4, device=device))
                                L = torch.max(torch.tensor(0.1, device=device), median_s - 2.0 * mad_s)
                                U = torch.min(torch.tensor(4.0, device=device), median_s + 2.0 * mad_s)
                                s_clamped = torch.clamp(s, L, U)
                            else:
                                s_clamped = torch.clamp(s, 0.1, 3.0)
                        elif cal_method == "u-ipr":
                            s_clamped = torch.clamp(s, 0.1, 10.0)
                            
                        # Apply to weights
                        w_init = progenitor_state[weight_key].to(device)
                        w_merged = merged_state[weight_key].to(device)
                        w_update = w_merged - w_init
                        
                        if isinstance(module, nn.Conv2d):
                            # Scale output channels independently
                            for c in range(module.out_channels):
                                cal_model.state_dict()[weight_key][c] = w_init[c] + s_clamped[c] * w_update[c]
                        else:
                            # Scale globally
                            cal_model.state_dict()[weight_key] = w_init + s_clamped * w_update
                            
    elif cal_method == "cbvc":
        # Proposed: Channel-wise BatchNorm Variance Calibration (CBVC)
        # Keeps weights as standard merged, but rescales BN running variance by 1 / s^2!
        print(f"Applying CBVC (Channel-wise BatchNorm Variance Calibration)...")
        with torch.no_grad():
            for name, module in cal_model.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    bn_name = get_bn_name(name)
                    if bn_name is not None:
                        # Find the corresponding BN module
                        bn_module = None
                        for b_name, b_mod in cal_model.backbone.named_modules():
                            if b_name == bn_name:
                                bn_module = b_mod
                                break
                                
                        if bn_module is not None:
                            s = scale_factors[name]
                            
                            # Clip scale factor to prevent division by zero or extreme scaling
                            if dynamic_clip:
                                # Clip s using MAD
                                median_s = torch.median(s)
                                mad_s = torch.median(torch.abs(s - median_s))
                                mad_s = torch.max(mad_s, torch.tensor(1e-4, device=device))
                                L = torch.max(torch.tensor(0.1, device=device), median_s - 2.0 * mad_s)
                                U = torch.min(torch.tensor(4.0, device=device), median_s + 2.0 * mad_s)
                                s_clamped = torch.clamp(s, L, U)
                            else:
                                s_clamped = torch.clamp(s, 0.1, 10.0)
                                
                            # Re-estimate calibrated running mean and variance to prevent mean shift and scale collapse
                            bn_module.running_mean.copy_(bn_module.running_mean / s_clamped)
                            bn_module.running_var.copy_(bn_module.running_var / (s_clamped**2 + 1e-8))
                            
    return cal_model

def run_experiment_suite():
    print("="*50)
    print("RUNNING COMPLETE EXPERIMENT SUITE")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Load model checkpoints
    print("Loading progenitor and expert models...")
    progenitor = ExpertModel()
    progenitor.load_state_dict(torch.load("checkpoints/progenitor.pt", map_location="cpu"))
    progenitor_state = progenitor.state_dict()
    
    expert_states = []
    task_heads = {}
    tasks = ["mnist", "fmnist", "cifar10"]
    
    for task in tasks:
        expert = ExpertModel()
        expert.load_state_dict(torch.load(f"checkpoints/{task}_expert.pt", map_location="cpu"))
        expert_states.append(expert.state_dict())
        
        # Pull task head
        task_heads[task] = copy.deepcopy(expert.fc)
        
    print("Models loaded successfully.")
    
    # We will record results for LaTeX and progress log
    results = {}
    
    # 1. EVALUATE INDIVIDUAL EXPERTS (SANITY CHECK / UPPER BOUNDS)
    print("\nEvaluating individual expert oracles (Upper Bounds)...")
    expert_accs = {}
    for task in tasks:
        expert = ExpertModel()
        expert.load_state_dict(torch.load(f"checkpoints/{task}_expert.pt", map_location="cpu"))
        accs = evaluate_model(expert, task_heads)
        expert_accs[task] = accs[task]
    expert_accs["average"] = sum(expert_accs[t] for t in tasks) / len(tasks)
    print(f"Expert Oracles Accs: {expert_accs}")
    results["Oracles"] = expert_accs
    
    # 2. WEIGHT AVERAGING (WA) - BASELINES & CALIBRATIONS
    # We test WA backbone with None, U-IPR, HNS, QR-IPR, DE-BN, and CBVC (Ours)
    wa_backbone_state = merge_models(progenitor_state, expert_states, alg="wa")
    wa_model = ExpertModel()
    wa_model.load_state_dict(wa_backbone_state)
    
    cal_methods = [
        ("None", "none", False, False),
        ("U-IPR", "u-ipr", False, False),
        ("HNS", "hns", False, False),
        ("QR-IPR", "qr-ipr", True, False),
        ("CBVC (Ours, standard clip)", "cbvc", False, False),
        ("CBVC (Ours, dynamic MAD clip)", "cbvc", True, False)
    ]
    
    for label, method_name, use_mad, use_sparsity in cal_methods:
        print(f"\nEvaluating WA + {label}...")
        cal_model = apply_calibration(wa_model, progenitor_state, expert_states, cal_method=method_name, dynamic_clip=use_mad, sparsity_compensated=use_sparsity)
        
        # Evaluate under multiple conditions:
        # FP32
        fp32_acc = evaluate_model(cal_model, task_heads)
        # INT8 Per-Tensor PTQ
        int8_tensor_acc = evaluate_model(cal_model, task_heads, num_bits=8, per_channel=False)
        # INT8 Per-Channel PTQ
        int8_channel_acc = evaluate_model(cal_model, task_heads, num_bits=8, per_channel=True)
        # INT4 Per-Channel PTQ
        int4_channel_acc = evaluate_model(cal_model, task_heads, num_bits=4, per_channel=True)
        # Robustness to Noise
        noise_acc = evaluate_model(cal_model, task_heads, corruption="noise")
        # Robustness to Blur
        blur_acc = evaluate_model(cal_model, task_heads, corruption="blur")
        
        results[f"WA + {label}"] = {
            "FP32": fp32_acc,
            "INT8_Tensor": int8_tensor_acc,
            "INT8_Channel": int8_channel_acc,
            "INT4_Channel": int4_channel_acc,
            "Noise": noise_acc,
            "Blur": blur_acc
        }
        print(f"WA + {label} | FP32 Average Acc: {fp32_acc['average']:.2f}% | INT8 Per-Tensor Acc: {int8_tensor_acc['average']:.2f}% | Noise: {noise_acc['average']:.2f}%")
        
    # 3. DATA-EFFICIENT BATCHNORM CALIBRATION (DE-BN) - COMPARISON
    # Re-estimate BatchNorm running stats using 8, 16, and 64 samples of real data task-specifically
    for n_samples in [8, 16, 64]:
        print(f"\nEvaluating WA + DE-BN (N={n_samples})...")
        fp32_acc = evaluate_model(wa_model, task_heads, de_bn_samples=n_samples)
        int8_tensor_acc = evaluate_model(wa_model, task_heads, num_bits=8, per_channel=False, de_bn_samples=n_samples)
        int8_channel_acc = evaluate_model(wa_model, task_heads, num_bits=8, per_channel=True, de_bn_samples=n_samples)
        noise_acc = evaluate_model(wa_model, task_heads, corruption="noise", de_bn_samples=n_samples)
        
        results[f"WA + DE-BN (N={n_samples})"] = {
            "FP32": fp32_acc,
            "INT8_Tensor": int8_tensor_acc,
            "INT8_Channel": int8_channel_acc,
            "Noise": noise_acc
        }
        print(f"WA + DE-BN (N={n_samples}) | FP32 Average Acc: {fp32_acc['average']:.2f}% | INT8 Per-Tensor Acc: {int8_tensor_acc['average']:.2f}% | Noise: {noise_acc['average']:.2f}%")
        
    # 4. TUNED TASK ARITHMETIC (TA)
    # Let's search over lambda
    for lam in [0.1, 0.5, 0.7]:
        print(f"\nEvaluating Tuned TA (lambda={lam})...")
        ta_backbone_state = merge_models(progenitor_state, expert_states, alg="ta", lam=lam)
        ta_model = ExpertModel()
        ta_model.load_state_dict(ta_backbone_state)
        
        # Test TA + None and TA + CBVC (Ours)
        for label, method_name, use_mad in [("None", "none", False), ("CBVC (Ours)", "cbvc", True)]:
            print(f"Applying {label} on TA...")
            cal_model = apply_calibration(ta_model, progenitor_state, expert_states, cal_method=method_name, dynamic_clip=use_mad)
            fp32_acc = evaluate_model(cal_model, task_heads)
            int8_tensor_acc = evaluate_model(cal_model, task_heads, num_bits=8, per_channel=False)
            results[f"TA (lambda={lam}) + {label}"] = {
                "FP32": fp32_acc,
                "INT8_Tensor": int8_tensor_acc
            }
            print(f"TA (lambda={lam}) + {label} | FP32 Average: {fp32_acc['average']:.2f}% | INT8 Tensor: {int8_tensor_acc['average']:.2f}%")

    # 5. SPARSIFIED MERGING PARADIGMS: TIES AND DARE
    # We evaluate TIES and DARE with None, standard CBVC, and Sparsity-Compensated SC-CBVC (Ours)
    for alg_name, func_name, kwargs in [("TIES", "ties", {}), ("DARE", "dare", {"drop_rate": 0.2})]:
        print(f"\nEvaluating {alg_name} Merging...")
        sparse_backbone_state = merge_models(progenitor_state, expert_states, alg=func_name, **kwargs)
        sparse_model = ExpertModel()
        sparse_model.load_state_dict(sparse_backbone_state)
        
        for label, method_name, use_mad, use_sparsity in [
            ("None", "none", False, False),
            ("CBVC (Ours)", "cbvc", True, False),
            ("SC-CBVC (Ours)", "cbvc", True, True)
        ]:
            cal_model = apply_calibration(sparse_model, progenitor_state, expert_states, cal_method=method_name, dynamic_clip=use_mad, sparsity_compensated=use_sparsity)
            fp32_acc = evaluate_model(cal_model, task_heads)
            int8_tensor_acc = evaluate_model(cal_model, task_heads, num_bits=8, per_channel=False)
            results[f"{alg_name} + {label}"] = {
                "FP32": fp32_acc,
                "INT8_Tensor": int8_tensor_acc
            }
            print(f"{alg_name} + {label} | FP32 Average: {fp32_acc['average']:.2f}% | INT8 Tensor: {int8_tensor_acc['average']:.2f}%")

    # Save results to a file
    import json
    with open("checkpoints/results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAll experiments finished and results saved to checkpoints/results.json!")
    
    # Print a beautiful summarized markdown table
    print("\n" + "="*40 + " SUMMARY OF EXPERIMENTAL RESULTS " + "="*40)
    print("| Method | FP32 Acc | INT8 Per-Tensor | INT8 Per-Channel | INT4 Per-Channel | Noise Acc | Blur Acc |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(results.keys()):
        if key == "Oracles":
            print(f"| Expert Oracles (Upper Bounds) | {results[key]['average']:.2f}% | N/A | N/A | N/A | N/A | N/A |")
            continue
        res = results[key]
        fp32_val = f"{res.get('FP32', {}).get('average', 0.0):.2f}%"
        i8_tensor = f"{res.get('INT8_Tensor', {}).get('average', 0.0):.2f}%" if "INT8_Tensor" in res else "N/A"
        i8_channel = f"{res.get('INT8_Channel', {}).get('average', 0.0):.2f}%" if "INT8_Channel" in res else "N/A"
        i4_channel = f"{res.get('INT4_Channel', {}).get('average', 0.0):.2f}%" if "INT4_Channel" in res else "N/A"
        noise_val = f"{res.get('Noise', {}).get('average', 0.0):.2f}%" if "Noise" in res else "N/A"
        blur_val = f"{res.get('Blur', {}).get('average', 0.0):.2f}%" if "Blur" in res else "N/A"
        print(f"| {key} | {fp32_val} | {i8_tensor} | {i8_channel} | {i4_channel} | {noise_val} | {blur_val} |")
    print("="*113)

if __name__ == "__main__":
    # Disable cuDNN to bypass driver issues as per submission5 instructions
    torch.backends.cudnn.enabled = False
    run_experiment_suite()
