import os
import time
import json
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

# Import helper functions from run_experiments
from run_experiments import (
    get_dataloaders,
    check_and_train_experts,
    merge_models,
    calibrate_model_offline,
    calibrate_model_offline_tspc,
    calibrate_model_offline_real_data,
    evaluate_backbone
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for sweep: {device}")
    
    loaders = get_dataloaders()
    expert_states = check_and_train_experts(loaders, device)
    
    progenitor = resnet18(weights=ResNet18_Weights.DEFAULT)
    progenitor_state = progenitor.state_dict()
    
    lambda_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    calib_methods = ["none", "sp_ttbc", "geometry", "tspc", "oracle"]
    
    # Initialize dictionary to hold results
    sweep_results = {cm: [] for cm in calib_methods}
    sweep_results["lambdas"] = lambda_vals
    
    # Setup evaluation loaders for batch size 64
    eval_loaders_64 = {
        "mnist": DataLoader(torchvision.datasets.MNIST(root="./data", train=False, download=False, 
                            transform=loaders["mnist"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4),
        "fmnist": DataLoader(torchvision.datasets.FashionMNIST(root="./data", train=False, download=False,
                            transform=loaders["fmnist"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4),
        "cifar10": DataLoader(torchvision.datasets.CIFAR10(root="./data", train=False, download=False,
                            transform=loaders["cifar10"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4)
    }
    
    for lambda_val in lambda_vals:
        print(f"\n==========================================")
        print(f"RUNNING SWEEP FOR LAMBDA = {lambda_val}")
        print(f"==========================================")
        
        # Merge backbone with the specific lambda
        merged_backbone = merge_models(progenitor_state, expert_states, merge_method="ta", lambda_val=lambda_val)
        
        # Run offline calibrations for this merged model
        print("Calibrating PGAC (Geometry)...")
        pgac_backbone = calibrate_model_offline(merged_backbone, "geometry", device, num_samples=2560)
        
        print("Calibrating TSPC...")
        tspc_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            tspc_backbones[task] = calibrate_model_offline_tspc(merged_backbone, task, device, num_samples=2560)
            
        print("Calibrating Oracle...")
        oracle_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            oracle_backbones[task] = calibrate_model_offline_real_data(merged_backbone, loaders[task][0], device, num_samples=2560)
            
        # Evaluate each calibration method
        for cm in calib_methods:
            print(f"Evaluating {cm.upper()}...")
            
            if cm == "none":
                task_res = evaluate_backbone(merged_backbone, expert_states, eval_loaders_64, device, calib_method="none")
            elif cm == "sp_ttbc":
                task_res = evaluate_backbone(merged_backbone, expert_states, eval_loaders_64, device, calib_method="sp_ttbc")
            elif cm == "geometry":
                task_res = evaluate_backbone(pgac_backbone, expert_states, eval_loaders_64, device, calib_method="none")
            elif cm == "tspc":
                task_res = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(tspc_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
                    task_res[task] = res[task]
            elif cm == "oracle":
                task_res = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(oracle_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
                    task_res[task] = res[task]
                    
            avg_acc = np.mean(list(task_res.values()))
            sweep_results[cm].append(avg_acc)
            print(f"Lambda {lambda_val} | {cm.upper()} Average Accuracy: {avg_acc:.2f}%")
            
    # Save sweep results to json
    with open("lambda_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
    print("\nSaved lambda sweep results to lambda_sweep_results.json")
    
    # Plot results
    plt.figure(figsize=(8, 6))
    
    # Custom styling
    method_styles = {
        "none": {"label": "Uncalibrated", "color": "#7f7f7f", "marker": "o", "linestyle": "--"},
        "sp_ttbc": {"label": "SP-TTBC (Online)", "color": "#d62728", "marker": "^", "linestyle": "-."},
        "geometry": {"label": "PGAC (Ours, Data-Free)", "color": "#9467bd", "marker": "s", "linestyle": "-"},
        "tspc": {"label": "TSPC (Ours, Data-Free)", "color": "#e377c2", "marker": "d", "linestyle": "-"},
        "oracle": {"label": "Oracle (Real Data)", "color": "#bcbd22", "marker": "*", "linestyle": "-"}
    }
    
    for cm in calib_methods:
        style = method_styles[cm]
        plt.plot(lambda_vals, sweep_results[cm], label=style["label"], color=style["color"],
                 marker=style["marker"], linestyle=style["linestyle"], linewidth=2, markersize=8)
        
    plt.xlabel(r"Task Arithmetic scaling coefficient ($\lambda$)", fontsize=12)
    plt.ylabel("Multi-Task Average Accuracy (%)", fontsize=12)
    plt.title("Calibration Robustness under Varying Scaling Coefficients", fontsize=14)
    plt.xticks(lambda_vals)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(10, 85)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    plt.savefig("lambda_sweep_robustness.png", dpi=300)
    plt.close()
    print("Generated lambda sweep robustness plot saved to lambda_sweep_robustness.png")

if __name__ == "__main__":
    main()
