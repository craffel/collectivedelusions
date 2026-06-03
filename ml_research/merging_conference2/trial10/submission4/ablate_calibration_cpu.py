import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Disable cuDNN
torch.backends.cudnn.enabled = False

# Import methods from evaluate_merging
from evaluate_merging import (
    apply_qcot, apply_qwc, apply_cwss, apply_cwss_qc,
    evaluate_model, calibrate_bn, quantize_model
)

def main():
    print("Initializing CPU BN Calibration size ablation...")
    device = torch.device("cpu")
    
    print("Loading checkpoints...")
    w_init = torch.load("checkpoints/progenitor_backbone.pt", map_location="cpu")
    tasks = ["mnist", "fmnist", "cifar10"]
    w_experts = [torch.load(f"checkpoints/{task}_backbone.pt", map_location="cpu") for task in tasks]
    
    # Let's focus on CIFAR-10 task since it is the most representative and has the largest space for calibration healing
    print("Loading CIFAR-10 test set and training set (for calibration)...")
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Batch size 128 for faster CPU evaluation
    test_loader = DataLoader(datasets.CIFAR10(root="data", train=False, download=False, transform=transform_color), batch_size=128, shuffle=False, num_workers=0)
    calib_loader = DataLoader(datasets.CIFAR10(root="data", train=True, download=False, transform=transform_color), batch_size=32, shuffle=True, num_workers=0)
    
    head = nn.Linear(512, 10).to(device)
    head.load_state_dict(torch.load("checkpoints/cifar10_head.pt", map_location="cpu"))
    
    # Generate models
    print("\nGenerating model checkpoints...")
    
    # 1. WA
    w_wa = {}
    for name in w_init.keys():
        if w_init[name].is_floating_point():
            w_wa[name] = torch.mean(torch.stack([we[name] for we in w_experts]), dim=0)
        else:
            w_wa[name] = w_init[name].clone()
            
    # 2. TA (lambda=0.4)
    lambda_val = 0.4
    w_ta = {}
    for name in w_init.keys():
        if w_init[name].is_floating_point():
            t_merged = lambda_val * torch.sum(torch.stack([we[name] - w_init[name] for we in w_experts]), dim=0)
            w_ta[name] = w_init[name] + t_merged
        else:
            w_ta[name] = w_init[name].clone()
            
    # 3. QCOT (C=0.05)
    print("Generating QCOT...")
    w_qcot = apply_qcot(w_init, w_experts, w_wa, C=0.05)
    
    # 4. QWC (q=0.999)
    print("Generating QWC...")
    w_qwc = apply_qwc(w_init, w_wa, q=0.999)
    
    # 5. CWSS
    print("Generating CWSS...")
    w_cwss = apply_cwss(w_init, w_experts, w_wa)
    
    # 6. CWSS-QC (q=0.9999)
    print("Generating CWSS-QC...")
    w_cwss_qc = apply_cwss_qc(w_init, w_experts, w_wa, q=0.9999)
    
    methods = {
        "WA": w_wa,
        "TA": w_ta,
        "QCOT (C=0.05)": w_qcot,
        "QWC (q=0.999)": w_qwc,
        "CWSS": w_cwss,
        "CWSS-QC (q=0.9999)": w_cwss_qc
    }
    
    bn_calib_sizes = [0, 4, 8, 16, 32, 64]
    results = []
    
    print("\nEvaluating configurations sequentially on CPU...")
    for method_name, w_state in methods.items():
        print(f"--- Method: {method_name} ---")
        
        # Load backbone
        merged_backbone = resnet18()
        merged_backbone.fc = nn.Identity()
        merged_backbone.load_state_dict(w_state)
        merged_backbone = merged_backbone.to(device)
        
        # Quantize to INT4-Channel
        eval_model = quantize_model(merged_backbone, bits=4, per_channel=True)
        
        # Save original BN running statistics
        orig_stats = {}
        for name, buf in eval_model.named_buffers():
            if "running_mean" in name or "running_var" in name:
                orig_stats[name] = buf.clone()
                
        for calib_size in bn_calib_sizes:
            # Restore original statistics before calibration
            for name, buf in eval_model.named_buffers():
                if name in orig_stats:
                    buf.copy_(orig_stats[name])
                    
            if calib_size > 0:
                calibrate_bn(eval_model, calib_loader, num_samples=calib_size)
                
            start_t = time.time()
            acc = evaluate_model(eval_model, test_loader, head, corruption=None)
            elapsed = time.time() - start_t
            
            print(f"  BN={calib_size:2d} samples: Accuracy = {acc:.2f}% ({elapsed:.2f}s)")
            results.append({
                "method": method_name,
                "bn_calib": calib_size,
                "cifar10_acc": acc
            })
            
    # Save results
    with open("ablation_bn_results_cpu.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nCPU BN Calibration size ablation complete! Saved to ablation_bn_results_cpu.json.")

if __name__ == "__main__":
    main()
