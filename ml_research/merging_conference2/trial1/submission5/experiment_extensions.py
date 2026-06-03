import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import helpers from the existing experiment script
from experiment import (
    device,
    get_dataset,
    get_dataloader,
    get_resnet18_backbone_and_head,
    MultiTaskModel,
    evaluate_model
)

# Define flexible BN calibration function supporting selective layers
def calibrate_backbone_bn_flexible(merged_backbone, calib_loader, device, expert_backbone=None, use_expert_affine=False, calibrate_layers=None):
    calibrated_backbone = copy.deepcopy(merged_backbone).to(device)
    
    # If using expert affine parameters (gamma, beta), copy them for selected layers
    if use_expert_affine and expert_backbone is not None:
        expert_state = expert_backbone.state_dict()
        calibrated_state = calibrated_backbone.state_dict()
        for key in calibrated_state.keys():
            is_bn = "bn" in key and ("weight" in key or "bias" in key)
            if is_bn:
                # Check if this layer matches any specified calibrate layers
                if calibrate_layers is None or any(layer_name in key for layer_name in calibrate_layers):
                    calibrated_state[key] = expert_state[key].clone()
        calibrated_backbone.load_state_dict(calibrated_state)
        
    # Reset running stats of selected BN layers and configure them to track stats
    for name, m in calibrated_backbone.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if calibrate_layers is None or any(layer_name in name for layer_name in calibrate_layers):
                m.reset_running_stats()
                m.momentum = None  # Cumulative average
                m.track_running_stats = True
                m.training = True
            else:
                m.training = False
                m.track_running_stats = False
                
    calibrated_backbone.eval()
    
    # Force selected BN layers to remain in training mode during forward passes
    for name, m in calibrated_backbone.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if calibrate_layers is None or any(layer_name in name for layer_name in calibrate_layers):
                m.training = True
            else:
                m.training = False
                
    with torch.no_grad():
        for inputs, _ in calib_loader:
            inputs = inputs.to(device)
            _ = calibrated_backbone(inputs)
            
    # Revert all BN layers back to standard evaluation mode
    for m in calibrated_backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = False
            
    return calibrated_backbone

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Load expert models
    experts = {}
    for task in tasks:
        backbone_path = f"models/backbone_{task}.pt"
        head_path = f"models/head_{task}.pt"
        
        backbone, head = get_resnet18_backbone_and_head()
        backbone = backbone.to(device)
        head = head.to(device)
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        head.load_state_dict(torch.load(head_path, map_location=device))
        experts[task] = (backbone, head)
        
    # Get base backbone
    base_backbone, _ = get_resnet18_backbone_and_head()
    base_backbone = base_backbone.to(device)
    base_state = base_backbone.state_dict()
    
    # Test loaders for evaluation
    test_loaders = {task: get_dataloader(task, batch_size=256, is_train=False) for task in tasks}
    
    # -------------------------------------------------------------
    # EXTENSION 1: Calibration Dataset Size (N) Sweep
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("EXTENSION 1: Calibration Dataset Size (N) Sweep")
    print("=======================================================")
    
    n_values = [16, 32, 64, 128, 256, 512]
    n_results = {n: {} for n in n_values}
    
    # We will use Task Arithmetic with lambda = 0.4 as our target merged backbone
    merged_backbone_ta, _ = get_resnet18_backbone_and_head()
    merged_backbone_ta = merged_backbone_ta.to(device)
    ta_state = {}
    lam_val = 0.4
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            task_vectors = [experts[t][0].state_dict()[key] - base_state[key] for t in tasks]
            ta_state[key] = base_state[key] + lam_val * torch.stack(task_vectors).sum(dim=0)
        else:
            ta_state[key] = base_state[key].clone()
    merged_backbone_ta.load_state_dict(ta_state)
    
    for n in n_values:
        print(f"Evaluating Calibration Size N = {n}...")
        task_accs = {}
        for task in tasks:
            calib_dataset = get_dataset(task, is_train=True)
            calib_subset = Subset(calib_dataset, list(range(n)))
            calib_loader = DataLoader(calib_subset, batch_size=min(64, n), shuffle=False)
            
            # Calibrate using Task-Specific Affine
            calibrated_backbone = calibrate_backbone_bn_flexible(
                merged_backbone_ta, calib_loader, device, 
                expert_backbone=experts[task][0], use_expert_affine=True
            )
            acc = evaluate_model(calibrated_backbone, experts[task][1], test_loaders[task])
            task_accs[task] = acc
        
        avg_acc = sum(task_accs.values()) / len(tasks)
        n_results[n] = {
            "tasks": task_accs,
            "average": avg_acc
        }
        print(f"N = {n} -> Average Accuracy: {avg_acc:.2f}% (MNIST: {task_accs['mnist']:.2f}%, Fashion: {task_accs['fashion']:.2f}%, CIFAR: {task_accs['cifar10']:.2f}%)")
        
    # -------------------------------------------------------------
    # EXTENSION 2: Task Arithmetic Lambda (Scale) Sweep
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("EXTENSION 2: Task Arithmetic Lambda (Scale) Sweep")
    print("=======================================================")
    
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambda_results = {
        "baseline": [],
        "tcac_shared": [],
        "tcac_expert": []
    }
    
    # We calibrate with N = 128
    calib_loaders = {}
    for task in tasks:
        calib_dataset = get_dataset(task, is_train=True)
        calib_subset = Subset(calib_dataset, list(range(128)))
        calib_loaders[task] = DataLoader(calib_subset, batch_size=64, shuffle=False)
        
    for lam in lambda_values:
        print(f"Evaluating Lambda = {lam:.1f}...")
        
        # Merge backbone with current lambda
        current_ta, _ = get_resnet18_backbone_and_head()
        current_ta = current_ta.to(device)
        curr_state = {}
        for key in base_state.keys():
            if base_state[key].is_floating_point():
                task_vectors = [experts[t][0].state_dict()[key] - base_state[key] for t in tasks]
                curr_state[key] = base_state[key] + lam * torch.stack(task_vectors).sum(dim=0)
            else:
                curr_state[key] = base_state[key].clone()
        current_ta.load_state_dict(curr_state)
        
        # 1. Baseline
        base_accs = []
        for task in tasks:
            acc = evaluate_model(current_ta, experts[task][1], test_loaders[task])
            base_accs.append(acc)
        avg_base = sum(base_accs) / len(tasks)
        lambda_results["baseline"].append(avg_base)
        
        # 2. TCAC Shared
        shared_accs = []
        for task in tasks:
            calibrated = calibrate_backbone_bn_flexible(
                current_ta, calib_loaders[task], device, use_expert_affine=False
            )
            acc = evaluate_model(calibrated, experts[task][1], test_loaders[task])
            shared_accs.append(acc)
        avg_shared = sum(shared_accs) / len(tasks)
        lambda_results["tcac_shared"].append(avg_shared)
        
        # 3. TCAC Task-Specific
        expert_accs_list = []
        for task in tasks:
            calibrated = calibrate_backbone_bn_flexible(
                current_ta, calib_loaders[task], device,
                expert_backbone=experts[task][0], use_expert_affine=True
            )
            acc = evaluate_model(calibrated, experts[task][1], test_loaders[task])
            expert_accs_list.append(acc)
        avg_expert = sum(expert_accs_list) / len(tasks)
        lambda_results["tcac_expert"].append(avg_expert)
        
        print(f"  Lambda = {lam:.1f} -> Baseline: {avg_base:.2f}%, TCAC Shared: {avg_shared:.2f}%, TCAC Task-Specific: {avg_expert:.2f}%")

    # -------------------------------------------------------------
    # EXTENSION 3: Depth Ablation (Which layers to calibrate)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("EXTENSION 3: Depth Ablation Study")
    print("=======================================================")
    
    # Let's consider different depths of calibration for WA and TA (lambda = 0.4)
    # Layer settings:
    # A. None (Uncalibrated Baseline)
    # B. Initial Only (conv1/bn1)
    # C. Late Only (layer4)
    # D. Late and Mid (layer3, layer4)
    # E. All (bn1, layer1, layer2, layer3, layer4)
    
    schemes = {
        "Uncalibrated Baseline": [], # special case
        "Initial BN Only (bn1)": ["bn1"],
        "Late-Block Only (layer4)": ["layer4"],
        "Mid & Late Blocks (layer3, layer4)": ["layer3", "layer4"],
        "All Layers (Full TCAC)": None
    }
    
    depth_results = {
        "wa": {},
        "ta": {}
    }
    
    # We use Weight Averaging (WA) and Task Arithmetic (TA) as targets
    # Weight Averaging backbone
    merged_backbone_wa, _ = get_resnet18_backbone_and_head()
    merged_backbone_wa = merged_backbone_wa.to(device)
    wa_state = {}
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            wa_state[key] = torch.stack([experts[t][0].state_dict()[key] for t in tasks]).mean(dim=0)
        else:
            wa_state[key] = experts[tasks[0]][0].state_dict()[key].clone()
    merged_backbone_wa.load_state_dict(wa_state)
    
    for scheme_name, layers in schemes.items():
        print(f"Evaluating Depth Scheme: {scheme_name}...")
        
        # 1. Evaluate for WA
        wa_accs = {}
        for task in tasks:
            if scheme_name == "Uncalibrated Baseline":
                acc = evaluate_model(merged_backbone_wa, experts[task][1], test_loaders[task])
            else:
                calibrated = calibrate_backbone_bn_flexible(
                    merged_backbone_wa, calib_loaders[task], device,
                    expert_backbone=experts[task][0], use_expert_affine=True,
                    calibrate_layers=layers
                )
                acc = evaluate_model(calibrated, experts[task][1], test_loaders[task])
            wa_accs[task] = acc
        avg_wa = sum(wa_accs.values()) / len(tasks)
        depth_results["wa"][scheme_name] = {
            "tasks": wa_accs,
            "average": avg_wa
        }
        
        # 2. Evaluate for TA
        ta_accs = {}
        for task in tasks:
            if scheme_name == "Uncalibrated Baseline":
                acc = evaluate_model(merged_backbone_ta, experts[task][1], test_loaders[task])
            else:
                calibrated = calibrate_backbone_bn_flexible(
                    merged_backbone_ta, calib_loaders[task], device,
                    expert_backbone=experts[task][0], use_expert_affine=True,
                    calibrate_layers=layers
                )
                acc = evaluate_model(calibrated, experts[task][1], test_loaders[task])
            ta_accs[task] = acc
        avg_ta = sum(ta_accs.values()) / len(tasks)
        depth_results["ta"][scheme_name] = {
            "tasks": ta_accs,
            "average": avg_ta
        }
        
        print(f"  WA average: {avg_wa:.2f}%, TA average: {avg_ta:.2f}%")

    # -------------------------------------------------------------
    # Save Results and Generate Plots
    # -------------------------------------------------------------
    output_data = {
        "n_sweep": n_results,
        "lambda_sweep": {
            "lambda_values": lambda_values,
            "baseline": lambda_results["baseline"],
            "tcac_shared": lambda_results["tcac_shared"],
            "tcac_expert": lambda_results["tcac_expert"]
        },
        "depth_ablation": depth_results
    }
    
    with open("results/extension_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("\nExtension results successfully saved to results/extension_results.json!")
    
    # Plot 1: Calibration Size (N) Sweep
    plt.figure(figsize=(7, 5))
    n_avg_accs = [n_results[n]["average"] for n in n_values]
    plt.plot(n_values, n_avg_accs, marker='o', color='darkorange', linewidth=2.5, label='TCAC (Task-Specific Affine)')
    
    # Draw reference line for uncalibrated TA (which is around 9.93%)
    plt.axhline(y=9.93, color='gray', linestyle='--', label='Uncalibrated Baseline')
    
    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.xlabel('Calibration Dataset Size ($N$ per task, log scale)', fontsize=11)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=11)
    plt.title('TCAC Efficiency vs. Calibration Set Size ($N$)', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/calibration_size_sweep.png", dpi=300)
    print("Plot saved to plots/calibration_size_sweep.png!")
    
    # Plot 2: Lambda Sweep
    plt.figure(figsize=(7, 5))
    plt.plot(lambda_values, lambda_results["baseline"], marker='x', color='crimson', linestyle=':', label='Uncalibrated Baseline')
    plt.plot(lambda_values, lambda_results["tcac_shared"], marker='s', color='dodgerblue', linestyle='--', label='TCAC (Shared Affine)')
    plt.plot(lambda_values, lambda_results["tcac_expert"], marker='o', color='forestgreen', linestyle='-', linewidth=2, label='TCAC (Task-Specific Affine)')
    
    plt.xlabel('Task Arithmetic Coefficient ($\lambda$)', fontsize=11)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=11)
    plt.title('Robustness of TCAC Across Task Vector Scaling Factors', fontsize=12, fontweight='bold')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lambda_sweep.png", dpi=300)
    print("Plot saved to plots/lambda_sweep.png!")

if __name__ == "__main__":
    main()
