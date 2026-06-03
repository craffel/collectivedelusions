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

# Disable cuDNN
torch.backends.cudnn.enabled = False

# Import methods from evaluate_merging
from evaluate_merging import (
    apply_qcot, apply_qwc, apply_emqc, apply_cwss, apply_cwss_qc,
    get_safe_filename, evaluate_model, calibrate_bn, quantize_model,
    evaluate_config_worker
)

# Custom worker for calibration size ablation
def evaluate_ablation_worker(gpu_id, tasks, temp_results_dir):
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

    # Load test sets and training sets (for calibration data)
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
        model_path = task["model_path"]
        
        # Load state dict
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
                
            acc = evaluate_model(eval_model, test_loaders[task_t], heads[task_t], corruption=None)
            task_accs[task_t] = acc
            
        avg_acc = sum(task_accs.values()) / 3
        
        run_info = {
            "method": method_name,
            "precision": prec_name,
            "bn_calib": calib_size,
            "mnist": task_accs["mnist"],
            "fmnist": task_accs["fmnist"],
            "cifar10": task_accs["cifar10"],
            "average": avg_acc
        }
        
        unique_id = f"worker_{gpu_id}_ablation_{task_idx}"
        res_file = os.path.join(temp_results_dir, f"{unique_id}.json")
        with open(res_file, "w") as f:
            json.dump(run_info, f)

def main():
    mp.set_start_method("spawn", force=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main process device: {device}")
    
    # Create local directories
    temp_dir = "temp_models_ablation"
    temp_results_dir = "temp_results_ablation"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_results_dir, exist_ok=True)
    
    print("Loading checkpoints...")
    w_init = torch.load("checkpoints/progenitor_backbone.pt", map_location="cpu")
    tasks = ["mnist", "fmnist", "cifar10"]
    w_experts = [torch.load(f"checkpoints/{task}_backbone.pt", map_location="cpu") for task in tasks]
    
    print("\nGenerating model checkpoints for ablation...")
    
    # 1. WA
    w_wa = {}
    for name in w_init.keys():
        if w_init[name].is_floating_point():
            w_wa[name] = torch.mean(torch.stack([we[name] for we in w_experts]), dim=0)
        else:
            w_wa[name] = w_init[name].clone()
    wa_path = os.path.join(temp_dir, "wa.pt")
    torch.save(w_wa, wa_path)
    
    # 2. TA (lambda=0.4)
    lambda_val = 0.4
    w_ta = {}
    for name in w_init.keys():
        if w_init[name].is_floating_point():
            t_merged = lambda_val * torch.sum(torch.stack([we[name] - w_init[name] for we in w_experts]), dim=0)
            w_ta[name] = w_init[name] + t_merged
        else:
            w_ta[name] = w_init[name].clone()
    ta_path = os.path.join(temp_dir, "ta.pt")
    torch.save(w_ta, ta_path)
    
    # 3. QCOT (C=0.05)
    w_qcot = apply_qcot(w_init, w_experts, w_wa, C=0.05)
    qcot_path = os.path.join(temp_dir, "qcot.pt")
    torch.save(w_qcot, qcot_path)
    
    # 4. QWC (q=0.999)
    w_qwc = apply_qwc(w_init, w_wa, q=0.999)
    qwc_path = os.path.join(temp_dir, "qwc.pt")
    torch.save(w_qwc, qwc_path)
    
    # 5. EMQC
    w_init_gpu = {k: v.to(device) for k, v in w_init.items()}
    w_wa_gpu = {k: v.to(device) for k, v in w_wa.items()}
    w_emqc = apply_emqc(w_init_gpu, w_wa_gpu, q_candidates=[0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0], bits=4, per_channel=True)
    w_emqc_cpu = {k: v.to("cpu") for k, v in w_emqc.items()}
    emqc_path = os.path.join(temp_dir, "emqc.pt")
    torch.save(w_emqc_cpu, emqc_path)
    
    # 6. CWSS
    w_cwss = apply_cwss(w_init, w_experts, w_wa)
    cwss_path = os.path.join(temp_dir, "cwss.pt")
    torch.save(w_cwss, cwss_path)
    
    # 7. CWSS-QC (q=0.9999)
    w_cwss_qc = apply_cwss_qc(w_init, w_experts, w_wa, q=0.9999)
    cwss_qc_path = os.path.join(temp_dir, "cwss_qc.pt")
    torch.save(w_cwss_qc, cwss_qc_path)
    
    methods_meta = {
        "WA": wa_path,
        "TA": ta_path,
        "QCOT": qcot_path,
        "QWC": qwc_path,
        "EMQC": emqc_path,
        "CWSS": cwss_path,
        "CWSS-QC": cwss_qc_path
    }
    
    precisions = [
        ("FP32", None, False),
        ("INT8_Channel", 8, True),
        ("INT4_Channel", 4, True)
    ]
    
    # Calibration sample sizes to ablate
    bn_calib_sizes = [0, 4, 8, 16, 32, 64]
    
    tasks_list = []
    for method_name, model_path in methods_meta.items():
        for prec_name, bits, per_channel in precisions:
            for calib_size in bn_calib_sizes:
                tasks_list.append({
                    "method_name": method_name,
                    "model_path": model_path,
                    "precision": prec_name,
                    "bits": bits,
                    "per_channel": per_channel,
                    "bn_calib": calib_size
                })
                
    total_evals = len(tasks_list)
    print(f"\nCreated {total_evals} ablation evaluation tasks.")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")
    
    results = {"runs": []}
    
    if num_gpus > 0:
        # Static task partitioning across available GPUs
        chunks = [[] for _ in range(num_gpus)]
        for idx, t in enumerate(tasks_list):
            chunks[idx % num_gpus].append(t)
            
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=evaluate_ablation_worker, args=(i, chunks[i], temp_results_dir))
            p.start()
            processes.append(p)
            
        print(f"Spawned {num_gpus} GPU workers. Monitoring files in {temp_results_dir}...")
        
        last_count = -1
        while any(p.is_alive() for p in processes):
            num_files = len(os.listdir(temp_results_dir))
            if num_files != last_count:
                print(f"Ablation Progress: {num_files}/{total_evals} tasks completed on disk.")
                last_count = num_files
            time.sleep(2)
            
        for p in processes:
            p.join()
            
        print("All processes joined. Gathering results...")
        
        for f_name in os.listdir(temp_results_dir):
            if f_name.endswith(".json"):
                with open(os.path.join(temp_results_dir, f_name), "r") as f:
                    results["runs"].append(json.load(f))
    else:
        print("Warning: No GPUs found. Ablation study requires GPUs to complete in time.")
        return
        
    # Cleanup temporary directories
    try:
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_results_dir)
        print("Cleaned up temporary ablation directories.")
    except Exception as e:
        print(f"Error cleaning up: {e}")
        
    # Save ablation results
    with open("ablation_bn_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nBatchNorm Calibration Ablation complete! Results saved to ablation_bn_results.json.")

if __name__ == "__main__":
    main()
