import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import torch.multiprocessing as mp
import shutil
import re

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on some cluster nodes
torch.backends.cudnn.enabled = False

def get_safe_filename(name):
    safe = name.lower()
    safe = re.sub(r'[^a-z0-9]', '_', safe)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe + ".pt"

# Function to evaluate a model with possible corruptions
def evaluate_model(model, test_loader, head, corruption=None, noise_std=0.1):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    
    dev = next(model.parameters()).device
    
    if corruption == "blur":
        blur_transform = transforms.GaussianBlur(kernel_size=3, sigma=1.0)
        
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(dev), targets.to(dev)
            
            if corruption == "noise":
                inputs = inputs + noise_std * torch.randn_like(inputs)
            elif corruption == "blur":
                inputs = blur_transform(inputs)
                
            features = model(inputs)
            outputs = head(features)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100.0 * correct / total

# DE-BN Calibration function
def calibrate_bn(model, calib_loader, num_samples=32):
    model.eval()
    bn_layers = []
    original_momentums = []
    
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(m)
            original_momentums.append(m.momentum)
            m.momentum = 1.0
            m.train()
            m.reset_running_stats()
            
    inputs_accum = []
    samples_collected = 0
    for inputs, _ in calib_loader:
        inputs_accum.append(inputs)
        samples_collected += inputs.size(0)
        if samples_collected >= num_samples:
            break
            
    if len(inputs_accum) == 0:
        return
        
    dev = next(model.parameters()).device
    inputs = torch.cat(inputs_accum, dim=0)[:num_samples].to(dev)
    
    with torch.no_grad():
        _ = model(inputs)
        
    for m, orig_mom in zip(bn_layers, original_momentums):
        m.momentum = orig_mom
        m.eval()

# Post-Training Quantization (PTQ) helper
def quantize_symmetric(weight, bits=8, per_channel=False):
    if bits is None:
        return weight.clone()
    q_max = 2**(bits - 1) - 1
    
    if per_channel and weight.dim() >= 2:
        scale = weight.abs().flatten(1).max(dim=1)[0]
        scale = scale.clamp(min=1e-8) / q_max
        scale_reshaped = scale.view(-1, *([1] * (weight.dim() - 1)))
        weight_q = torch.clamp(torch.round(weight / scale_reshaped), -q_max, q_max) * scale_reshaped
        return weight_q
    else:
        scale = weight.abs().max().item()
        scale = max(scale, 1e-8) / q_max
        weight_q = torch.clamp(torch.round(weight / scale), -q_max, q_max) * scale
        return weight_q

# Apply PTQ to the entire backbone model
def quantize_model(model, bits=8, per_channel=False):
    dev = next(model.parameters()).device
    quantized_model = resnet18()
    quantized_model.fc = nn.Identity()
    quantized_model.load_state_dict(model.state_dict())
    quantized_model = quantized_model.to(dev)
    
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if "weight" in name and ("conv" in name or "fc" in name or "downsample" in name):
                param.copy_(quantize_symmetric(param.data, bits=bits, per_channel=per_channel))
    return quantized_model

# QCOT (Quantization-Constrained Optimal Transport) implementation
def apply_qcot(w_init, w_experts, w_merged, C):
    w_calibrated = {}
    K = len(w_experts)
    
    for name in w_init.keys():
        p_init = w_init[name]
        p_merged = w_merged[name]
        p_experts = [we[name] for we in w_experts]
        
        if "weight" in name and ("conv" in name or "fc" in name or "downsample" in name):
            t_merged = p_merged - p_init
            t_experts = [pe - p_init for pe in p_experts]
            t_calibrated = torch.zeros_like(t_merged)
            
            if t_merged.dim() >= 2:
                for c in range(t_merged.size(0)):
                    x = t_merged[c].flatten()
                    s_k_sorted = [torch.sort(t_experts[k][c].flatten())[0] for k in range(K)]
                    y = torch.mean(torch.stack(s_k_sorted), dim=0)
                    y_clipped = torch.clamp(y, -C, C)
                    
                    _, perm = torch.sort(x)
                    x_calibrated = torch.zeros_like(x)
                    x_calibrated[perm] = y_clipped
                    t_calibrated[c] = x_calibrated.view(t_merged[c].shape)
            else:
                x = t_merged.flatten()
                s_k_sorted = [torch.sort(t_experts[k].flatten())[0] for k in range(K)]
                y = torch.mean(torch.stack(s_k_sorted), dim=0)
                y_clipped = torch.clamp(y, -C, C)
                
                _, perm = torch.sort(x)
                x_calibrated = torch.zeros_like(x)
                x_calibrated[perm] = y_clipped
                t_calibrated = x_calibrated.view(t_merged.shape)
                
            w_calibrated[name] = p_init + t_calibrated
        else:
            w_calibrated[name] = p_merged.clone()
            
    return w_calibrated

# QWC (Quantile-Based Weight Clipping) implementation
def apply_qwc(w_init, w_merged, q):
    w_calibrated = {}
    for name in w_init.keys():
        p_init = w_init[name]
        p_merged = w_merged[name]
        
        if "weight" in name and ("conv" in name or "fc" in name or "downsample" in name):
            t_merged = p_merged - p_init
            t_calibrated = torch.zeros_like(t_merged)
            
            if t_merged.dim() >= 2:
                for c in range(t_merged.size(0)):
                    x = t_merged[c]
                    threshold = torch.quantile(x.abs(), q)
                    t_calibrated[c] = torch.clamp(x, -threshold, threshold)
            else:
                threshold = torch.quantile(t_merged.abs(), q)
                t_calibrated = torch.clamp(t_merged, -threshold, threshold)
                
            w_calibrated[name] = p_init + t_calibrated
        else:
            w_calibrated[name] = p_merged.clone()
            
    return w_calibrated

# EMQC (Error-Minimizing Quantile Clipping) implementation
def apply_emqc(w_init, w_merged, q_candidates=[0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0], bits=4, per_channel=True):
    w_calibrated = {}
    for name in w_init.keys():
        p_init = w_init[name]
        p_merged = w_merged[name]
        
        if "weight" in name and ("conv" in name or "fc" in name or "downsample" in name):
            t_merged = p_merged - p_init
            t_calibrated = torch.zeros_like(t_merged)
            
            if t_merged.dim() >= 2:
                for c in range(t_merged.size(0)):
                    x = t_merged[c]
                    best_err = float('inf')
                    best_x_clipped = x
                    
                    for q in q_candidates:
                        if q < 1.0:
                            threshold = torch.quantile(x.abs(), q)
                            x_clipped = torch.clamp(x, -threshold, threshold)
                        else:
                            x_clipped = x.clone()
                            
                        w_c = p_init[c] + x_clipped
                        w_c_q = quantize_symmetric(w_c, bits=bits, per_channel=False)
                        err = torch.sum((w_c - w_c_q) ** 2).item()
                        
                        if err < best_err:
                            best_err = err
                            best_x_clipped = x_clipped
                            
                    t_calibrated[c] = best_x_clipped
            else:
                best_err = float('inf')
                best_t_clipped = t_merged
                
                for q in q_candidates:
                    if q < 1.0:
                        threshold = torch.quantile(t_merged.abs(), q)
                        t_clipped = torch.clamp(t_merged, -threshold, threshold)
                    else:
                        t_clipped = t_merged.clone()
                        
                    w_c = p_init + t_clipped
                    w_c_q = quantize_symmetric(w_c, bits=bits, per_channel=False)
                    err = torch.sum((w_c - w_c_q) ** 2).item()
                    
                    if err < best_err:
                        best_err = err
                        best_t_clipped = t_clipped
                        
                t_calibrated = best_t_clipped
                
            w_calibrated[name] = p_init + t_calibrated
        else:
            w_calibrated[name] = p_merged.clone()
            
    return w_calibrated


# CWSS (Channel-wise Weight Standardization & Scaling) implementation
def apply_cwss(w_init, w_experts, w_merged):
    w_calibrated = {}
    K = len(w_experts)
    for name in w_init.keys():
        p_init = w_init[name]
        p_merged = w_merged[name]
        p_experts = [we[name] for we in w_experts]
        
        if "weight" in name and ("conv" in name or "fc" in name or "downsample" in name):
            t_merged = p_merged - p_init
            t_experts = [pe - p_init for pe in p_experts]
            t_calibrated = torch.zeros_like(t_merged)
            
            if t_merged.dim() >= 2:
                for c in range(t_merged.size(0)):
                    x = t_merged[c]
                    expert_stds = torch.stack([te[c].std() for te in t_experts])
                    expert_means = torch.stack([te[c].mean() for te in t_experts])
                    
                    target_std = torch.mean(expert_stds)
                    target_mean = torch.mean(expert_means)
                    
                    x_std = x.std()
                    x_mean = x.mean()
                    
                    # Normalize and scale
                    x_norm = (x - x_mean) / (x_std + 1e-8)
                    t_calibrated[c] = x_norm * target_std + target_mean
            else:
                x = t_merged
                expert_stds = torch.stack([te.std() for te in t_experts])
                expert_means = torch.stack([te.mean() for te in t_experts])
                
                target_std = torch.mean(expert_stds)
                target_mean = torch.mean(expert_means)
                
                x_std = x.std()
                x_mean = x.mean()
                
                x_norm = (x - x_mean) / (x_std + 1e-8)
                t_calibrated = x_norm * target_std + target_mean
                
            w_calibrated[name] = p_init + t_calibrated
        else:
            w_calibrated[name] = p_merged.clone()
            
    return w_calibrated


# CWSS-QC (Channel-wise Weight Standardization & Scaling with Quantile Clipping) implementation
def apply_cwss_qc(w_init, w_experts, w_merged, q=0.9999):
    w_cwss = apply_cwss(w_init, w_experts, w_merged)
    w_cwss_qc = apply_qwc(w_init, w_cwss, q)
    return w_cwss_qc


# Worker process function for GPU parallelization (100% deadlock-free, static partitioned, no nested processes)
def evaluate_config_worker(gpu_id, tasks, temp_results_dir):
    device = torch.device(f"cuda:{gpu_id}")
    torch.backends.cudnn.enabled = False
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # No nested multiprocessing (num_workers=0) - 100% safe!
    test_loaders = {
        "mnist": DataLoader(datasets.MNIST(root="data", train=False, download=False, transform=transform_gray), batch_size=256, shuffle=False, num_workers=0),
        "fmnist": DataLoader(datasets.FashionMNIST(root="data", train=False, download=False, transform=transform_gray), batch_size=256, shuffle=False, num_workers=0),
        "cifar10": DataLoader(datasets.CIFAR10(root="data", train=False, download=False, transform=transform_color), batch_size=256, shuffle=False, num_workers=0)
    }
    
    calib_loaders = {
        "mnist": DataLoader(datasets.MNIST(root="data", train=True, download=False, transform=transform_gray), batch_size=64, shuffle=True, num_workers=0),
        "fmnist": DataLoader(datasets.FashionMNIST(root="data", train=True, download=False, transform=transform_gray), batch_size=64, shuffle=True, num_workers=0),
        "cifar10": DataLoader(datasets.CIFAR10(root="data", train=True, download=False, transform=transform_color), batch_size=64, shuffle=True, num_workers=0)
    }
    
    tasks_names = ["mnist", "fmnist", "cifar10"]
    heads = {task_n: nn.Linear(512, 10).to(device) for task_n in tasks_names}
    for task_n in tasks_names:
        heads[task_n].load_state_dict(torch.load(f"checkpoints/{task_n}_head.pt", map_location=device))
        
    for task_idx, task in enumerate(tasks):
        method_name = task["method_name"]
        prec_name = task["precision"]
        bits = task["bits"]
        per_channel = task["per_channel"]
        calib_size = task["bn_calib"]
        corr_name = task["corruption"]
        corr_type = task["corruption_type"]
        model_path = task["model_path"]
        
        # Load state dict directly from disk onto the designated GPU
        w_state = torch.load(model_path, map_location=device)
        
        merged_backbone = resnet18()
        merged_backbone.fc = nn.Identity()
        merged_backbone.load_state_dict(w_state)
        merged_backbone = merged_backbone.to(device)
        
        if bits is not None:
            eval_model = quantize_model(merged_backbone, bits=bits, per_channel=per_channel)
        else:
            eval_model = resnet18()
            eval_model.fc = nn.Identity()
            eval_model.load_state_dict(merged_backbone.state_dict())
            eval_model = eval_model.to(device)
            
        orig_stats = {}
        for name, buf in eval_model.named_buffers():
            if "running_mean" in name or "running_var" in name:
                orig_stats[name] = buf.clone()
                
        task_accs = {}
        for task_t in tasks_names:
            for name, buf in eval_model.named_buffers():
                if name in orig_stats:
                    buf.copy_(orig_stats[name])
                    
            if calib_size > 0:
                calibrate_bn(eval_model, calib_loaders[task_t], num_samples=calib_size)
                
            acc = evaluate_model(eval_model, test_loaders[task_t], heads[task_t], corruption=corr_type)
            task_accs[task_t] = acc
            
        avg_acc = sum(task_accs.values()) / 3
        
        # Save individual task result to disk
        run_info = {
            "method": method_name,
            "precision": prec_name,
            "bn_calib": calib_size,
            "corruption": corr_name,
            "mnist": task_accs["mnist"],
            "fmnist": task_accs["fmnist"],
            "cifar10": task_accs["cifar10"],
            "average": avg_acc
        }
        
        unique_id = f"worker_{gpu_id}_task_{task_idx}"
        res_file = os.path.join(temp_results_dir, f"{unique_id}.json")
        with open(res_file, "w") as f:
            json.dump(run_info, f)


def main():
    mp.set_start_method("spawn", force=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main process device: {device}")
    
    # Create local directories
    temp_dir = "temp_models"
    temp_results_dir = "temp_results"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_results_dir, exist_ok=True)
    
    print("Loading checkpoints...")
    w_init = torch.load("checkpoints/progenitor_backbone.pt", map_location="cpu")
    
    tasks = ["mnist", "fmnist", "cifar10"]
    w_experts = [torch.load(f"checkpoints/{task}_backbone.pt", map_location="cpu") for task in tasks]
    
    heads = {task: nn.Linear(512, 10).to(device) for task in tasks}
    for task in tasks:
        heads[task].load_state_dict(torch.load(f"checkpoints/{task}_head.pt", map_location=device))
        
    print("\nEvaluating Individual Experts (Oracle)...")
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_loaders = {
        "mnist": DataLoader(datasets.MNIST(root="data", train=False, download=True, transform=transform_gray), batch_size=256, shuffle=False, num_workers=4, pin_memory=True),
        "fmnist": DataLoader(datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_gray), batch_size=256, shuffle=False, num_workers=4, pin_memory=True),
        "cifar10": DataLoader(datasets.CIFAR10(root="data", train=False, download=True, transform=transform_color), batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    oracle_accs = {}
    for task in tasks:
        backbone = resnet18()
        backbone.fc = nn.Identity()
        backbone.load_state_dict(torch.load(f"checkpoints/{task}_backbone.pt", map_location=device))
        backbone = backbone.to(device)
        acc = evaluate_model(backbone, test_loaders[task], heads[task])
        oracle_accs[task] = acc
        print(f"Expert {task.upper()} Accuracy: {acc:.2f}%")
    oracle_avg = sum(oracle_accs.values()) / 3
    print(f"Average Oracle Accuracy: {oracle_avg:.2f}%")
    
    results = {
        "oracle": oracle_accs,
        "oracle_avg": oracle_avg,
        "runs": []
    }
    
    print("\n--- Running Weight Averaging (WA) ---")
    w_wa = {}
    for name in w_init.keys():
        if w_init[name].is_floating_point():
            w_wa[name] = torch.mean(torch.stack([we[name] for we in w_experts]), dim=0)
        else:
            w_wa[name] = w_init[name].clone()
    wa_path = os.path.join(temp_dir, get_safe_filename("WA"))
    torch.save(w_wa, wa_path)
        
    print("\n--- Running Task Arithmetic (TA) ---")
    lambda_val = 0.4
    w_ta = {}
    for name in w_init.keys():
        if w_init[name].is_floating_point():
            t_merged = lambda_val * torch.sum(torch.stack([we[name] - w_init[name] for we in w_experts]), dim=0)
            w_ta[name] = w_init[name] + t_merged
        else:
            w_ta[name] = w_init[name].clone()
    ta_path = os.path.join(temp_dir, get_safe_filename("TA (lambda=0.4)"))
    torch.save(w_ta, ta_path)
        
    qcot_configs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    qwc_configs = [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    
    methods_meta = {
        "WA": wa_path,
        "TA (lambda=0.4)": ta_path
    }
    
    print("\nGenerating QCOT models...")
    for C in qcot_configs:
        start_t = time.time()
        w_qcot = apply_qcot(w_init, w_experts, w_wa, C)
        elapsed = time.time() - start_t
        print(f"QCOT (C={C}) generated in {elapsed:.3f} seconds.")
        path = os.path.join(temp_dir, get_safe_filename(f"QCOT (C={C})"))
        torch.save(w_qcot, path)
        methods_meta[f"QCOT (C={C})"] = path
        
    print("\nGenerating QWC models...")
    for q in qwc_configs:
        start_t = time.time()
        w_qwc = apply_qwc(w_init, w_wa, q)
        elapsed = time.time() - start_t
        print(f"QWC (q={q}) generated in {elapsed:.3f} seconds.")
        path = os.path.join(temp_dir, get_safe_filename(f"QWC (q={q})"))
        torch.save(w_qwc, path)
        methods_meta[f"QWC (q={q})"] = path
        
    print("\nGenerating EMQC models...")
    start_t = time.time()
    w_init_gpu = {k: v.to(device) for k, v in w_init.items()}
    w_wa_gpu = {k: v.to(device) for k, v in w_wa.items()}
    w_emqc = apply_emqc(w_init_gpu, w_wa_gpu, q_candidates=[0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0], bits=4, per_channel=True)
    w_emqc_cpu = {k: v.to("cpu") for k, v in w_emqc.items()}
    elapsed = time.time() - start_t
    print(f"EMQC generated in {elapsed:.3f} seconds.")
    path = os.path.join(temp_dir, get_safe_filename("EMQC"))
    torch.save(w_emqc_cpu, path)
    methods_meta["EMQC"] = path
    
    print("\nGenerating CWSS models...")
    start_t = time.time()
    w_cwss = apply_cwss(w_init, w_experts, w_wa)
    elapsed = time.time() - start_t
    print(f"CWSS generated in {elapsed:.3f} seconds.")
    path = os.path.join(temp_dir, get_safe_filename("CWSS"))
    torch.save(w_cwss, path)
    methods_meta["CWSS"] = path

    print("\nGenerating CWSS-QC models...")
    start_t = time.time()
    w_cwss_qc = apply_cwss_qc(w_init, w_experts, w_wa, q=0.9999)
    elapsed = time.time() - start_t
    print(f"CWSS-QC generated in {elapsed:.3f} seconds.")
    path = os.path.join(temp_dir, get_safe_filename("CWSS-QC (q=0.9999)"))
    torch.save(w_cwss_qc, path)
    methods_meta["CWSS-QC (q=0.9999)"] = path
    
    # Define tasks list
    precisions = [
        ("FP32", None, False),
        ("INT8_Tensor", 8, False),
        ("INT8_Channel", 8, True),
        ("INT4_Channel", 4, True)
    ]
    corruptions = [
        ("clean", None),
        ("noise", "noise"),
        ("blur", "blur")
    ]
    bn_calib_sizes = [0, 32]
    
    tasks_list = []
    for method_name, model_path in methods_meta.items():
        for prec_name, bits, per_channel in precisions:
            for calib_size in bn_calib_sizes:
                for corr_name, corr_type in corruptions:
                    tasks_list.append({
                        "method_name": method_name,
                        "model_path": model_path,
                        "precision": prec_name,
                        "bits": bits,
                        "per_channel": per_channel,
                        "bn_calib": calib_size,
                        "corruption": corr_name,
                        "corruption_type": corr_type
                    })
                    
    total_evals = len(tasks_list)
    print(f"\nCreated {total_evals} evaluation tasks.")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")
    
    if num_gpus > 0:
        # Static task partitioning
        chunks = [[] for _ in range(num_gpus)]
        for idx, t in enumerate(tasks_list):
            chunks[idx % num_gpus].append(t)
            
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=evaluate_config_worker, args=(i, chunks[i], temp_results_dir))
            p.start()
            processes.append(p)
            
        print(f"Spawned {num_gpus} GPU workers. Monitoring files in {temp_results_dir}...")
        
        last_count = -1
        while any(p.is_alive() for p in processes):
            num_files = len(os.listdir(temp_results_dir))
            if num_files != last_count:
                print(f"Progress: {num_files}/{total_evals} tasks completed on disk.")
                last_count = num_files
            time.sleep(5)
            
        for p in processes:
            p.join()
            
        print("All processes joined. Gathering results...")
        
        # Read and aggregate results
        for f_name in os.listdir(temp_results_dir):
            if f_name.endswith(".json"):
                with open(os.path.join(temp_results_dir, f_name), "r") as f:
                    results["runs"].append(json.load(f))
    else:
        print("Warning: No GPUs found. Running sequentially on CPU.")
        pass
        
    # Cleanup local directories
    try:
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_results_dir)
        print("Cleaned up temporary directories.")
    except Exception as e:
        print(f"Error cleaning up: {e}")
        
    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAll evaluations complete! Results saved to results.json.")


if __name__ == "__main__":
    main()
