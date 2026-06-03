import os
import json
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import helpers from the existing experiment scripts
from experiment import (
    device,
    get_dataset,
    get_dataloader,
    get_resnet18_backbone_and_head,
    MultiTaskModel,
    calibrate_backbone_bn
)

def evaluate_model_with_noise(backbone, head, dataloader, noise_type=None, noise_level=0.0):
    model = MultiTaskModel(backbone, head).to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply input corruption on-the-fly
            if noise_type == "gaussian_noise" and noise_level > 0.0:
                # inputs are already normalized to approx [-1.0, 1.0] by transforms.Normalize((0.5,), (0.5,))
                # so we can add gaussian noise directly
                noise = torch.randn_like(inputs) * noise_level
                inputs = torch.clamp(inputs + noise, -1.0, 1.0)
            elif noise_type == "gaussian_blur" and noise_level > 0.0:
                # TF.gaussian_blur expects kernel_size and sigma
                # standard kernel size is determined from sigma: odd kernel size roughly 2 * ceil(2 * sigma) + 1
                kernel_size = int(2 * np.ceil(2 * noise_level) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                # Clip kernel size to reasonable range and ensure odd
                kernel_size = max(3, min(21, kernel_size))
                inputs = TF.gaussian_blur(inputs, [kernel_size, kernel_size], [noise_level, noise_level])
                
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return (correct / total) * 100.0

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    print(f"Loading expert models on device: {device}...")
    
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

    # 1. Reconstruct baseline pretrained model (base)
    base_backbone, _ = get_resnet18_backbone_and_head()
    base_backbone = base_backbone.to(device)
    
    # 2. Perform Task Arithmetic merging with lambda = 0.4
    print("Performing Task Arithmetic model merging...")
    lam = 0.4
    merged_backbone_state = copy.deepcopy(base_backbone.state_dict())
    
    task_vectors = {}
    for task in tasks:
        expert_state = experts[task][0].state_dict()
        task_vectors[task] = {}
        for key in expert_state.keys():
            task_vectors[task][key] = expert_state[key] - merged_backbone_state[key]
            
    # Aggregate task vectors
    for key in merged_backbone_state.keys():
        # Do not modify keys that are not part of weights (e.g. tracking buffers if any, though standard resnet18 has BN running buffers)
        # We only interpolate float/double tensors
        if merged_backbone_state[key].is_floating_point():
            sum_vector = sum(task_vectors[task][key] for task in tasks)
            merged_backbone_state[key] += lam * sum_vector
            
    merged_backbone = copy.deepcopy(base_backbone)
    merged_backbone.load_state_dict(merged_backbone_state)
    merged_backbone = merged_backbone.to(device)
    
    # 3. Get loaders
    test_loaders = {task: get_dataloader(task, is_train=False) for task in tasks}
    calib_loaders = {task: get_dataloader(task, is_train=True) for task in tasks}
    
    # Pre-build clean calibration subsets
    # We calibrate on clean datasets (size N = 128)
    N = 128
    calib_subsets = {}
    for task in tasks:
        dataset = calib_loaders[task].dataset
        # Get subset of size N
        indices = list(range(N))
        subset = torch.utils.data.Subset(dataset, indices)
        calib_subsets[task] = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

    # 4. Calibration for TCAC (Task-Specific Affine)
    print("Calibrating TCAC with clean calibration samples...")
    tcac_backbones = {}
    for task in tasks:
        tcac_backbones[task] = calibrate_backbone_bn(
            merged_backbone=merged_backbone,
            calib_loader=calib_subsets[task],
            device=device,
            expert_backbone=experts[task][0],
            use_expert_affine=True
        )

    # SWEEPS
    noise_sweep_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    blur_sweep_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    results = {
        "gaussian_noise": {level: {"baseline": {}, "tcac": {}} for level in noise_sweep_levels},
        "gaussian_blur": {level: {"baseline": {}, "tcac": {}} for level in blur_sweep_levels}
    }
    
    # Evaluates Gaussian Noise
    print("\n--- Sweeping Input Gaussian Noise ---")
    for level in noise_sweep_levels:
        print(f"Noise level std={level:.2f}:")
        for task in tasks:
            # Baseline evaluation
            baseline_acc = evaluate_model_with_noise(
                backbone=merged_backbone,
                head=experts[task][1],
                dataloader=test_loaders[task],
                noise_type="gaussian_noise",
                noise_level=level
            )
            # TCAC evaluation
            tcac_acc = evaluate_model_with_noise(
                backbone=tcac_backbones[task],
                head=experts[task][1],
                dataloader=test_loaders[task],
                noise_type="gaussian_noise",
                noise_level=level
            )
            results["gaussian_noise"][level]["baseline"][task] = baseline_acc
            results["gaussian_noise"][level]["tcac"][task] = tcac_acc
            print(f"  Task: {task:10} | Baseline: {baseline_acc:6.2f}% | TCAC: {tcac_acc:6.2f}%")
            
    # Evaluates Gaussian Blur
    print("\n--- Sweeping Input Gaussian Blur ---")
    for level in blur_sweep_levels:
        print(f"Blur level sigma={level:.2f}:")
        for task in tasks:
            # Baseline evaluation
            baseline_acc = evaluate_model_with_noise(
                backbone=merged_backbone,
                head=experts[task][1],
                dataloader=test_loaders[task],
                noise_type="gaussian_blur",
                noise_level=level
            )
            # TCAC evaluation
            tcac_acc = evaluate_model_with_noise(
                backbone=tcac_backbones[task],
                head=experts[task][1],
                dataloader=test_loaders[task],
                noise_type="gaussian_blur",
                noise_level=level
            )
            results["gaussian_blur"][level]["baseline"][task] = baseline_acc
            results["gaussian_blur"][level]["tcac"][task] = tcac_acc
            print(f"  Task: {task:10} | Baseline: {baseline_acc:6.2f}% | TCAC: {tcac_acc:6.2f}%")

    # Save results
    results_serialized = {
        "gaussian_noise": {str(k): v for k, v in results["gaussian_noise"].items()},
        "gaussian_blur": {str(k): v for k, v in results["gaussian_blur"].items()}
    }
    with open("results/noise_robustness_results.json", "w") as f:
        json.dump(results_serialized, f, indent=4)
    print("\nSaved noise robustness results to results/noise_robustness_results.json")

    # Generate Plots
    # 1. Noise Plot
    plt.figure(figsize=(7, 5))
    for task in tasks:
        base_ys = [results["gaussian_noise"][lvl]["baseline"][task] for lvl in noise_sweep_levels]
        tcac_ys = [results["gaussian_noise"][lvl]["tcac"][task] for lvl in noise_sweep_levels]
        line = plt.plot(noise_sweep_levels, tcac_ys, marker='o', label=f"TCAC - {task.upper()}")
        plt.plot(noise_sweep_levels, base_ys, linestyle='--', marker='x', color=line[0].get_color(), label=f"Baseline - {task.upper()}")
    
    plt.title("Robustness to Input Gaussian Noise")
    plt.xlabel("Gaussian Noise Std")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/noise_robustness.png", dpi=150)
    plt.close()
    
    # 2. Blur Plot
    plt.figure(figsize=(7, 5))
    for task in tasks:
        base_ys = [results["gaussian_blur"][lvl]["baseline"][task] for lvl in blur_sweep_levels]
        tcac_ys = [results["gaussian_blur"][lvl]["tcac"][task] for lvl in blur_sweep_levels]
        line = plt.plot(blur_sweep_levels, tcac_ys, marker='o', label=f"TCAC - {task.upper()}")
        plt.plot(blur_sweep_levels, base_ys, linestyle='--', marker='x', color=line[0].get_color(), label=f"Baseline - {task.upper()}")
    
    plt.title("Robustness to Input Gaussian Blur")
    plt.xlabel("Blur Sigma")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/blur_robustness.png", dpi=150)
    plt.close()
    
    print("Saved plots/noise_robustness.png and plots/blur_robustness.png")

if __name__ == "__main__":
    main()
