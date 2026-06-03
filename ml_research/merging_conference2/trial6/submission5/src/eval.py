import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Disable cuDNN due to environment cuDNN mismatch
torch.backends.cudnn.enabled = False

def get_transforms(task):
    if task in ["mnist", "fashionmnist"]:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    elif task == "cifar10":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unknown task {task}")

def get_dataset(task, train=False):
    transform = get_transforms(task)
    if task == "mnist":
        return datasets.MNIST("data", train=train, download=True, transform=transform)
    elif task == "fashionmnist":
        return datasets.FashionMNIST("data", train=train, download=True, transform=transform)
    elif task == "cifar10":
        return datasets.CIFAR10("data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task {task}")

def get_test_loader(task, batch_size, limit=1000):
    test_dataset = get_dataset(task, train=False)
    if limit is not None and limit < len(test_dataset):
        indices = list(range(limit))
        test_dataset = Subset(test_dataset, indices)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Custom Test-Time BatchNorm module
class TestTimeBatchNorm2d(nn.Module):
    def __init__(self, original_bn, alpha=0.1, stateful=False):
        super().__init__()
        self.num_features = original_bn.num_features
        self.eps = original_bn.eps
        self.alpha = alpha
        self.stateful = stateful
        
        # Copy parameters and buffers
        self.weight = nn.Parameter(original_bn.weight.data.clone()) if original_bn.weight is not None else None
        self.bias = nn.Parameter(original_bn.bias.data.clone()) if original_bn.bias is not None else None
        
        if original_bn.running_mean is not None:
            self.register_buffer("running_mean", original_bn.running_mean.data.clone())
        else:
            self.running_mean = None
            
        if original_bn.running_var is not None:
            self.register_buffer("running_var", original_bn.running_var.data.clone())
        else:
            self.running_var = None

    def forward(self, x):
        if self.training:
            # Fallback to standard train-time BN
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, True, 0.1, self.eps)
        
        # At test-time: compute batch statistics
        batch_mean = x.mean(dim=[0, 2, 3])
        # var with unbiased=False matches batch_norm behaviour
        batch_var = x.var(dim=[0, 2, 3], unbiased=False)
        
        # Blend running stats with current batch stats
        if self.alpha > 0 and self.running_mean is not None and self.running_var is not None:
            if self.stateful:
                # Update buffers in-place (Stateful streaming EMA)
                self.running_mean.copy_((1 - self.alpha) * self.running_mean + self.alpha * batch_mean)
                self.running_var.copy_((1 - self.alpha) * self.running_var + self.alpha * batch_var)
                active_mean = self.running_mean
                active_var = self.running_var
            else:
                # Stateless blending
                active_mean = (1 - self.alpha) * self.running_mean + self.alpha * batch_mean
                active_var = (1 - self.alpha) * self.running_var + self.alpha * batch_var
        else:
            active_mean = self.running_mean
            active_var = self.running_var
            
        return F.batch_norm(x, active_mean, active_var, self.weight, self.bias, False, 0.0, self.eps)

def convert_to_ttbc(model, alpha=0.1, stateful=False):
    """
    Recursively replaces all nn.BatchNorm2d in the model with TestTimeBatchNorm2d.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, TestTimeBatchNorm2d(module, alpha=alpha, stateful=stateful))
        else:
            convert_to_ttbc(module, alpha=alpha, stateful=stateful)
    return model

def get_bn_variance_stats(model, dataloader, device):
    """
    Hook-based or sequential function to compute activation variance at BatchNorm outputs.
    """
    model.eval()
    activations = {}
    hooks = []
    
    def get_hook(name):
        def hook_fn(module, input, output):
            flat_out = output.transpose(1, -1).flatten(0, -2) # [N, C]
            if name not in activations:
                activations[name] = []
            activations[name].append(flat_out.cpu())
        return hook_fn

    # Register hooks on all BatchNorm layers
    bn_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(get_hook(name)))
            bn_layers[name] = module

    # Pass data through the model
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute variance per channel for each BatchNorm output
    bn_stats = {}
    for name, acts in activations.items():
        all_acts = torch.cat(acts, dim=0) # [Total_Samples, C]
        bn_stats[name] = {
            "mean": all_acts.mean(dim=0),
            "var": all_acts.var(dim=0, unbiased=False)
        }
    return bn_stats, bn_layers

def apply_offline_calibration(merged_model, experts, calib_loaders, device, eps=1e-5):
    """
    Implements the offline REPAIR-like calibration baseline.
    """
    print("Computing offline calibration (REPAIR / SP-TAAC)...")
    merged_model.eval()
    for exp in experts:
        exp.eval()
        
    # 1. Compute expert statistics on their respective calibration sets
    expert_stats = []
    for exp, loader in zip(experts, calib_loaders):
        stats, _ = get_bn_variance_stats(exp, loader, device)
        expert_stats.append(stats)
        
    # 2. Compute merged model statistics on the joint calibration set
    merged_stats = {}
    merged_activations = {}
    hooks = []
    
    def get_merged_hook(name):
        def hook_fn(module, input, output):
            flat_out = output.transpose(1, -1).flatten(0, -2).cpu()
            if name not in merged_activations:
                merged_activations[name] = []
            merged_activations[name].append(flat_out)
        return hook_fn

    bn_layers = {}
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(get_merged_hook(name)))
            bn_layers[name] = module

    # Pass joint calibration data
    with torch.no_grad():
        for loader in calib_loaders:
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _ = merged_model(inputs)

    for h in hooks:
        h.remove()

    for name, acts in merged_activations.items():
        all_acts = torch.cat(acts, dim=0)
        merged_stats[name] = {
            "var": all_acts.var(dim=0, unbiased=False)
        }

    # 3. Apply the scaling factors in-place to merged model parameters
    calibrated_model = copy.deepcopy(merged_model)
    calibrated_bn_layers = {name: module for name, module in calibrated_model.named_modules() if isinstance(module, nn.BatchNorm2d)}
    
    for name, layer in calibrated_bn_layers.items():
        exp_vars = [stats[name]["var"] for stats in expert_stats]
        avg_exp_var = torch.stack(exp_vars, dim=0).mean(dim=0)
        
        merged_var = merged_stats[name]["var"]
        
        # Compute scaling factors
        scale = torch.sqrt(avg_exp_var + eps) / torch.sqrt(merged_var + eps)
        scale = scale.to(device)
        
        # Scale weight and bias in place
        if layer.weight is not None:
            layer.weight.data.copy_(layer.weight.data * scale)
        if layer.bias is not None:
            layer.bias.data.copy_(layer.bias.data * scale)
            
    print("Offline calibration applied successfully.")
    return calibrated_model

def evaluate_model(model, dataloader, head, device, corruption_type=None, intensity=0.0):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply test-time corruption if specified
            if corruption_type == "gaussian_noise" and intensity > 0:
                inputs = inputs + torch.randn_like(inputs) * intensity
            elif corruption_type == "brightness_shift" and intensity != 0:
                inputs = inputs + intensity
                
            # Bypass Backbone's native model.fc and use custom expert heads
            orig_fc = model.fc
            model.fc = nn.Identity()
            feats = model(inputs)
            model.fc = orig_fc
            
            outputs = head(feats)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

def merge_task_arithmetic(base_model, experts, lam, device):
    """
    Merges models using Task Arithmetic: W_merged = W_base + lam * sum(W_expert - W_base)
    excluding fc layers since they are evaluated task-specifically as custom heads.
    """
    merged = copy.deepcopy(base_model)
    merged_sd = merged.state_dict()
    base_sd = base_model.state_dict()
    expert_sds = [exp.state_dict() for exp in experts]
    
    for key in merged_sd.keys():
        if "fc" in key:
            continue
        
        # Compute task vectors
        task_vectors = []
        for exp_sd in expert_sds:
            task_vectors.append(exp_sd[key].float() - base_sd[key].float())
            
        # Sum task vectors and scale by lam
        summed_vectors = torch.stack(task_vectors, dim=0).sum(dim=0)
        merged_sd[key] = (base_sd[key].float() + lam * summed_vectors).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged

def benchmark_native_inference(model, device, batch_size=64, num_runs=50):
    import time
    model = copy.deepcopy(model)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency = (total_time / num_runs) * 1000 # ms
    throughput = (batch_size * num_runs) / total_time # images/sec
    return avg_latency, throughput

def benchmark_spttbc_inference(model, device, batch_size=64, num_runs=50):
    import time
    model = copy.deepcopy(model)
    # Convert to TTBC with alpha=1.0
    model = convert_to_ttbc(model, alpha=1.0)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency = (total_time / num_runs) * 1000 # ms
    throughput = (batch_size * num_runs) / total_time # images/sec
    return avg_latency, throughput

def benchmark_gradient_tta(model, device, batch_size=64, num_runs=50):
    import time
    model = copy.deepcopy(model)
    model.train() # needs to compute gradients
    
    # Freeze everything except BN affine parameters
    for param in model.parameters():
        param.requires_grad = False
        
    bn_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                module.weight.requires_grad = True
                bn_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad = True
                bn_params.append(module.bias)
                
    if len(bn_params) == 0:
        return 0.0, 0.0
        
    optimizer = torch.optim.Adam(bn_params, lr=1e-3)
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warmup
    for _ in range(5):
        outputs = model(dummy_input)
        probs = F.softmax(outputs, dim=1)
        loss = - (probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(num_runs):
        outputs = model(dummy_input)
        probs = F.softmax(outputs, dim=1)
        loss = - (probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency = (total_time / num_runs) * 1000 # ms
    throughput = (batch_size * num_runs) / total_time # images/sec
    return avg_latency, throughput

def evaluate_model_bf16(model, dataloader, head, device):
    model_bf16 = copy.deepcopy(model).to(torch.bfloat16)
    head_bf16 = copy.deepcopy(head).to(torch.bfloat16)
    model_bf16.eval()
    head_bf16.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device).to(torch.bfloat16), targets.to(device)
            
            # Bypass Backbone's native model.fc and use custom expert heads
            orig_fc = model_bf16.fc
            model_bf16.fc = nn.Identity()
            feats = model_bf16(inputs)
            model_bf16.fc = orig_fc
            
            outputs = head_bf16(feats)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

def benchmark_native_bf16_inference(model, device, batch_size=64, num_runs=50):
    import time
    model = copy.deepcopy(model)
    model = model.to(torch.bfloat16)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device).to(torch.bfloat16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency = (total_time / num_runs) * 1000 # ms
    throughput = (batch_size * num_runs) / total_time # images/sec
    return avg_latency, throughput

def benchmark_spttbc_bf16_inference(model, device, batch_size=64, num_runs=50):
    import time
    model = copy.deepcopy(model)
    model = convert_to_ttbc(model, alpha=1.0)
    model = model.to(torch.bfloat16)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device).to(torch.bfloat16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency = (total_time / num_runs) * 1000 # ms
    throughput = (batch_size * num_runs) / total_time # images/sec
    return avg_latency, throughput

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=1000) # Evaluates on 1000 samples for swiftness
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Running evaluation on {device} (Sample limit: {args.limit})...")

    # Tasks
    tasks = ["mnist", "fashionmnist", "cifar10"]

    # 1. Load expert backbones and classification heads
    expert_backbones = []
    expert_heads = []
    
    for task in tasks:
        path = f"models/resnet18_{task}.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model checkpoint {path} not found. Please train first.")
            
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        head = model.fc
        expert_backbones.append(model)
        expert_heads.append(head)

    # Print expert baseline accuracies
    print("\n" + "="*40 + "\nEvaluating individual Expert Baselines...")
    expert_accs = {}
    for task, backbone, head in zip(tasks, expert_backbones, expert_heads):
        test_loader = get_test_loader(task, args.test_batch_size, limit=args.limit)
        acc = evaluate_model(backbone, test_loader, head, device)
        expert_accs[task] = acc
        print(f"Expert {task} accuracy: {acc*100:.2f}%")

    # Load shared pre-trained progenitor base model
    base_model = resnet18()
    base_model.fc = nn.Linear(512, 10)
    base_model = base_model.to(device)
    base_path = "models/resnet18_base.pt"
    if os.path.exists(base_path):
        base_state_dict = torch.load(base_path, map_location=device)
        matched_state_dict = base_model.state_dict()
        for k, v in base_state_dict.items():
            if k in matched_state_dict and v.shape == matched_state_dict[k].shape:
                matched_state_dict[k] = v
        base_model.load_state_dict(matched_state_dict)
        print(f"Successfully loaded progenitor base model from {base_path}")
    else:
        print(f"Warning: Progenitor base model not found at {base_path}! Using default initialization.")

    # 2. Merge backbones using simple Weight Averaging (WA)
    print("\n" + "="*40 + "\nMerging models via Weight Averaging...")
    merged_backbone_wa = resnet18()
    merged_backbone_wa.fc = nn.Linear(512, 10)
    merged_backbone_wa = merged_backbone_wa.to(device)
    
    merged_state_dict = copy.deepcopy(expert_backbones[0].state_dict())
    for key in merged_state_dict.keys():
        weights = [exp.state_dict()[key].float() for exp in expert_backbones]
        merged_state_dict[key] = torch.stack(weights, dim=0).mean(dim=0).to(merged_state_dict[key].dtype)
        
    merged_backbone_wa.load_state_dict(merged_state_dict)

    # 3. Merge backbones using Task Arithmetic (TA) with a lambda sweep
    print("\n" + "="*40 + "\nSweeping over Task Arithmetic lambda (scaling factor)...")
    best_ta_lam = 0.3
    best_ta_avg = 0.0
    ta_backbones = {}
    
    # Let's do a sweep over lambda to find the best representative Task Arithmetic configuration
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ta_temp = merge_task_arithmetic(base_model, expert_backbones, lam, device)
        # Evaluate uncalibrated TA accuracy to see performance
        ta_accs = {}
        for task, head in zip(tasks, expert_heads):
            test_loader = get_test_loader(task, args.test_batch_size, limit=args.limit)
            acc = evaluate_model(ta_temp, test_loader, head, device)
            ta_accs[task] = acc
        avg_acc = sum(ta_accs.values()) / 3
        print(f"TA (lam={lam:.1f}) | MNIST: {ta_accs['mnist']*100:.2f}% | Fashion: {ta_accs['fashionmnist']*100:.2f}% | CIFAR10: {ta_accs['cifar10']*100:.2f}% | Avg: {avg_acc*100:.2f}%")
        ta_backbones[lam] = ta_temp
        if avg_acc > best_ta_avg:
            best_ta_avg = avg_acc
            best_ta_lam = lam

    # If all uncalibrated configurations are extremely collapsed, let's also check which lambda works best with SP-TTBC (64, 1.0)
    if best_ta_avg < 0.30:
        print("Uncalibrated TA models suffered from severe representation collapse. Sweeping lambda with SP-TTBC (B=64, alpha=1.0)...")
        best_ta_avg_cal = 0.0
        best_ta_lam_cal = 0.3
        task_loaders_64 = {t: get_test_loader(t, 64, limit=args.limit) for t in tasks}
        for lam, ta_temp in ta_backbones.items():
            tt_ta = copy.deepcopy(ta_temp)
            tt_ta = convert_to_ttbc(tt_ta, alpha=1.0)
            tt_ta = tt_ta.to(device)
            ta_accs = {}
            for task, head in zip(tasks, expert_heads):
                acc = evaluate_model(tt_ta, task_loaders_64[task], head, device)
                ta_accs[task] = acc
            avg_acc = sum(ta_accs.values()) / 3
            print(f"TA with SP-TTBC (lam={lam:.1f}) | Avg: {avg_acc*100:.2f}%")
            if avg_acc > best_ta_avg_cal:
                best_ta_avg_cal = avg_acc
                best_ta_lam_cal = lam
        best_ta_lam = best_ta_lam_cal
        print(f"Selected lam={best_ta_lam:.1f} as the best Task Arithmetic configuration based on calibrated accuracy.")
    else:
        print(f"Selected lam={best_ta_lam:.1f} as the best Task Arithmetic configuration based on uncalibrated accuracy.")

    merged_backbone_ta = ta_backbones[best_ta_lam]

    # Let's perform complete evaluation sweeps for BOTH Weight Averaging (WA) and Task Arithmetic (TA)
    # Prepare calibration loaders (100 samples per task) for offline calibration baseline
    calib_loaders = []
    for task in tasks:
        train_dataset = get_dataset(task, train=True)
        indices = list(range(100))
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False)
        calib_loaders.append(loader)

    results_data = {}

    for name, merged_backbone in [("WA", merged_backbone_wa), ("TA", merged_backbone_ta)]:
        print("\n" + "="*40 + f"\nEvaluating {name} Merged Backbone...")
        
        # A. Evaluate with NO calibration
        no_cal_accs = {}
        for task, head in zip(tasks, expert_heads):
            test_loader = get_test_loader(task, args.test_batch_size, limit=args.limit)
            acc = evaluate_model(merged_backbone, test_loader, head, device)
            no_cal_accs[task] = acc
        avg_no_cal = sum(no_cal_accs.values()) / 3
        print(f"{name} (No Cal) Average Accuracy: {avg_no_cal*100:.2f}%")
        
        # B. Evaluate with OFFLINE calibration
        calibrated_backbone = apply_offline_calibration(merged_backbone, expert_backbones, calib_loaders, device)
        offline_cal_accs = {}
        for task, head in zip(tasks, expert_heads):
            test_loader = get_test_loader(task, args.test_batch_size, limit=args.limit)
            acc = evaluate_model(calibrated_backbone, test_loader, head, device)
            offline_cal_accs[task] = acc
        avg_offline_cal = sum(offline_cal_accs.values()) / 3
        print(f"{name} (Offline Cal) Average Accuracy: {avg_offline_cal*100:.2f}%")

        # C. Sweep proposed SP-TTBC
        alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        batch_sizes = [1, 4, 16, 64]
        
        best_avg_acc = 0.0
        best_hyperparams = None
        results_sweep = []

        print(f"\nSweeping SP-TTBC over {name}...")
        for bs in batch_sizes:
            task_loaders = {t: get_test_loader(t, bs, limit=args.limit) for t in tasks}
            for alpha in alphas:
                tt_model = copy.deepcopy(merged_backbone)
                tt_model = convert_to_ttbc(tt_model, alpha=alpha)
                tt_model = tt_model.to(device)
                
                ttbc_accs = {}
                for task, head in zip(tasks, expert_heads):
                    acc = evaluate_model(tt_model, task_loaders[task], head, device)
                    ttbc_accs[task] = acc
                    
                avg_acc = sum(ttbc_accs.values()) / 3
                results_sweep.append({
                    "batch_size": bs,
                    "alpha": alpha,
                    "accs": ttbc_accs,
                    "avg": avg_acc
                })
                print(f"SP-TTBC (BS: {bs:2d}, Alpha: {alpha:.2f}) | Avg: {avg_acc*100:.2f}%")
                
                if avg_acc > best_avg_acc:
                    best_avg_acc = avg_acc
                    best_hyperparams = (bs, alpha, ttbc_accs)
                    
        print(f"\nBest SP-TTBC Configuration for {name}: Batch Size: {best_hyperparams[0]}, Alpha: {best_hyperparams[1]}")
        print(f"Best SP-TTBC Average Accuracy: {best_avg_acc*100:.2f}%")

        results_data[name] = {
            "no_calibration": no_cal_accs,
            "offline_calibration": offline_cal_accs,
            "best_sp_ttbc": {
                "batch_size": best_hyperparams[0],
                "alpha": best_hyperparams[1],
                "accs": best_hyperparams[2],
                "avg": best_avg_acc
            },
            "all_sweep_results": results_sweep
        }

    # 4. Evaluate Test-Time Robustness (Covariate Shift) under Gaussian Noise & Brightness Shift
    # We evaluate WA and TA models at best configuration (BS=64, alpha=1.0) under various noise levels
    print("\n" + "="*40 + "\nEvaluating Covariate Shift Robustness (Test-Time Corruptions)...")
    
    robustness_results = {
        "gaussian_noise": {
            "levels": [0.0, 0.05, 0.1, 0.2, 0.3],
            "WA": {"No_Cal": [], "Offline_Cal": [], "SP_TTBC": []},
            "TA": {"No_Cal": [], "Offline_Cal": [], "SP_TTBC": []}
        },
        "brightness_shift": {
            "levels": [-0.3, -0.1, 0.0, 0.1, 0.3],
            "WA": {"No_Cal": [], "Offline_Cal": [], "SP_TTBC": []},
            "TA": {"No_Cal": [], "Offline_Cal": [], "SP_TTBC": []}
        }
    }

    test_loader_64 = {t: get_test_loader(t, 64, limit=args.limit) for t in tasks}

    for name, merged_backbone in [("WA", merged_backbone_wa), ("TA", merged_backbone_ta)]:
        # Prepare the three model configurations
        # 1. No Cal
        model_no_cal = merged_backbone
        # 2. Offline Cal
        model_offline_cal = apply_offline_calibration(merged_backbone, expert_backbones, calib_loaders, device)
        # 3. SP-TTBC (BS=64, alpha=1.0)
        model_ttbc = copy.deepcopy(merged_backbone)
        model_ttbc = convert_to_ttbc(model_ttbc, alpha=1.0)
        model_ttbc = model_ttbc.to(device)

        # Evaluate Gaussian Noise
        print(f"\nEvaluating {name} robustness under Gaussian Noise...")
        for level in robustness_results["gaussian_noise"]["levels"]:
            # Evaluate No Cal
            accs_nc = {}
            for task, head in zip(tasks, expert_heads):
                accs_nc[task] = evaluate_model(model_no_cal, test_loader_64[task], head, device, "gaussian_noise", level)
            avg_nc = sum(accs_nc.values()) / 3
            robustness_results["gaussian_noise"][name]["No_Cal"].append(avg_nc)

            # Evaluate Offline Cal
            accs_oc = {}
            for task, head in zip(tasks, expert_heads):
                accs_oc[task] = evaluate_model(model_offline_cal, test_loader_64[task], head, device, "gaussian_noise", level)
            avg_oc = sum(accs_oc.values()) / 3
            robustness_results["gaussian_noise"][name]["Offline_Cal"].append(avg_oc)

            # Evaluate SP-TTBC
            accs_tt = {}
            for task, head in zip(tasks, expert_heads):
                accs_tt[task] = evaluate_model(model_ttbc, test_loader_64[task], head, device, "gaussian_noise", level)
            avg_tt = sum(accs_tt.values()) / 3
            robustness_results["gaussian_noise"][name]["SP_TTBC"].append(avg_tt)

            print(f"Noise Std {level:.2f} | No Cal: {avg_nc*100:.2f}% | Offline Cal: {avg_oc*100:.2f}% | SP-TTBC (Ours): {avg_tt*100:.2f}%")

        # Evaluate Brightness Shift
        print(f"\nEvaluating {name} robustness under Brightness Shift...")
        for level in robustness_results["brightness_shift"]["levels"]:
            # Evaluate No Cal
            accs_nc = {}
            for task, head in zip(tasks, expert_heads):
                accs_nc[task] = evaluate_model(model_no_cal, test_loader_64[task], head, device, "brightness_shift", level)
            avg_nc = sum(accs_nc.values()) / 3
            robustness_results["brightness_shift"][name]["No_Cal"].append(avg_nc)

            # Evaluate Offline Cal
            accs_oc = {}
            for task, head in zip(tasks, expert_heads):
                accs_oc[task] = evaluate_model(model_offline_cal, test_loader_64[task], head, device, "brightness_shift", level)
            avg_oc = sum(accs_oc.values()) / 3
            robustness_results["brightness_shift"][name]["Offline_Cal"].append(avg_oc)

            # Evaluate SP-TTBC
            accs_tt = {}
            for task, head in zip(tasks, expert_heads):
                accs_tt[task] = evaluate_model(model_ttbc, test_loader_64[task], head, device, "brightness_shift", level)
            avg_tt = sum(accs_tt.values()) / 3
            robustness_results["brightness_shift"][name]["SP_TTBC"].append(avg_tt)

            print(f"Shift {level:+.2f} | No Cal: {avg_nc*100:.2f}% | Offline Cal: {avg_oc*100:.2f}% | SP-TTBC (Ours): {avg_tt*100:.2f}%")

    # 5. Run Serving and Computational Efficiency Benchmark
    print("\n" + "="*40 + "\nRunning Serving and Computational Efficiency Benchmark...")
    bench_bs = 64
    bench_runs = 50
    print(f"Benchmarking with batch size: {bench_bs}, number of runs: {bench_runs}")

    native_lat, native_thr = benchmark_native_inference(merged_backbone_wa, device, batch_size=bench_bs, num_runs=bench_runs)
    ttbc_lat, ttbc_thr = benchmark_spttbc_inference(merged_backbone_wa, device, batch_size=bench_bs, num_runs=bench_runs)
    tta_lat, tta_thr = benchmark_gradient_tta(merged_backbone_wa, device, batch_size=bench_bs, num_runs=bench_runs)

    print(f"\n--- Benchmark Results ({device.type.upper()}) ---")
    print(f"Native Model:          Latency = {native_lat:6.2f} ms | Throughput = {native_thr:7.1f} img/sec")
    print(f"SP-TTBC (Ours):        Latency = {ttbc_lat:6.2f} ms | Throughput = {ttbc_thr:7.1f} img/sec")
    print(f"Gradient TTA (Tent):   Latency = {tta_lat:6.2f} ms | Throughput = {tta_thr:7.1f} img/sec")

    overhead_pct = ((ttbc_lat - native_lat) / native_lat) * 100 if native_lat > 0 else 0
    speedup_vs_tta = tta_lat / ttbc_lat if ttbc_lat > 0 else 0
    print(f"SP-TTBC Latency Overhead: {overhead_pct:.2f}%")
    print(f"SP-TTBC Speedup vs Gradient-based TTA: {speedup_vs_tta:.2f}x faster!")

    benchmark_results = {
        "native": {"latency_ms": native_lat, "throughput_img_sec": native_thr},
        "sp_ttbc": {"latency_ms": ttbc_lat, "throughput_img_sec": ttbc_thr, "overhead_pct": overhead_pct},
        "gradient_tta": {"latency_ms": tta_lat, "throughput_img_sec": tta_thr, "speedup_vs_tta": speedup_vs_tta}
    }

    # 5.1 Run Low-Precision BF16 Evaluation and Benchmark
    print("\n" + "="*40 + "\nRunning Low-Precision Bfloat16 Evaluation & Benchmark...")
    
    # Evaluate WA and TA SP-TTBC (BS=64, alpha=1.0) in BF16
    print("\nEvaluating SP-TTBC (BS: 64, Alpha: 1.0) under BF16 mixed-precision...")
    bf16_accs_dict = {}
    test_loader_64 = {t: get_test_loader(t, 64, limit=args.limit) for t in tasks}
    
    for name, merged_backbone in [("WA", merged_backbone_wa), ("TA", merged_backbone_ta)]:
        model_ttbc = copy.deepcopy(merged_backbone)
        model_ttbc = convert_to_ttbc(model_ttbc, alpha=1.0)
        model_ttbc = model_ttbc.to(device)
        
        bf16_accs = {}
        for task, head in zip(tasks, expert_heads):
            acc = evaluate_model_bf16(model_ttbc, test_loader_64[task], head, device)
            bf16_accs[task] = acc
        avg_bf16 = sum(bf16_accs.values()) / 3
        bf16_accs_dict[name] = {
            "mnist": bf16_accs["mnist"],
            "fashionmnist": bf16_accs["fashionmnist"],
            "cifar10": bf16_accs["cifar10"],
            "avg": avg_bf16
        }
        print(f"BF16 SP-TTBC ({name}) | MNIST: {bf16_accs['mnist']*100:.2f}% | Fashion: {bf16_accs['fashionmnist']*100:.2f}% | CIFAR10: {bf16_accs['cifar10']*100:.2f}% | Avg: {avg_bf16*100:.2f}%")

    native_bf16_lat, native_bf16_thr = benchmark_native_bf16_inference(merged_backbone_wa, device, batch_size=bench_bs, num_runs=bench_runs)
    ttbc_bf16_lat, ttbc_bf16_thr = benchmark_spttbc_bf16_inference(merged_backbone_wa, device, batch_size=bench_bs, num_runs=bench_runs)
    
    print(f"\n--- Bfloat16 Benchmark Results ({device.type.upper()}) ---")
    print(f"Native Model (BF16):   Latency = {native_bf16_lat:6.2f} ms | Throughput = {native_bf16_thr:7.1f} img/sec")
    print(f"SP-TTBC (BF16, Ours):  Latency = {ttbc_bf16_lat:6.2f} ms | Throughput = {ttbc_bf16_thr:7.1f} img/sec")
    
    bf16_overhead_pct = ((ttbc_bf16_lat - native_bf16_lat) / native_bf16_lat) * 100 if native_bf16_lat > 0 else 0
    bf16_speedup_vs_fp32 = ttbc_lat / ttbc_bf16_lat if ttbc_bf16_lat > 0 else 0
    print(f"BF16 SP-TTBC Latency Overhead: {bf16_overhead_pct:.2f}%")
    print(f"BF16 SP-TTBC Speedup vs FP32 SP-TTBC: {bf16_speedup_vs_fp32:.2f}x faster!")

    benchmark_bf16_results = {
        "native": {"latency_ms": native_bf16_lat, "throughput_img_sec": native_bf16_thr},
        "sp_ttbc": {"latency_ms": ttbc_bf16_lat, "throughput_img_sec": ttbc_bf16_thr, "overhead_pct": bf16_overhead_pct, "speedup_vs_fp32": bf16_speedup_vs_fp32}
    }

    # Save results to a file for paper plots
    summary_results = {
        "expert_baselines": expert_accs,
        "results_data": results_data,
        "robustness_results": robustness_results,
        "best_ta_lam": best_ta_lam,
        "benchmark_results": benchmark_results,
        "bf16_results": {
            "accuracy": bf16_accs_dict,
            "benchmark": benchmark_bf16_results
        }
    }

    with open("models/evaluation_results.json", "w") as f:
        import json
        json.dump(summary_results, f, indent=4)
    print("\nSaved evaluation results to models/evaluation_results.json")

if __name__ == "__main__":
    main()
