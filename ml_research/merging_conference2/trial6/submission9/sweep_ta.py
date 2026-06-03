import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Import components from eval.py
from eval import (
    device, resnet18, ResNet18_Weights,
    get_dataloader, get_mpcs_dataloader,
    calibrate_bn, apply_slr_wbc_single_task,
    get_mspr_prototypes, evaluate_mspr_routed, evaluate_model
)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("\nLoading trained expert weights for Task Arithmetic Lambda sweep...")
    experts = {}
    expert_heads = {}
    for task in ["mnist", "fashion", "cifar"]:
        ckpt_path = f"checkpoints/{task}_expert.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model_state = model.state_dict()
        model_state.update(checkpoint['backbone_state_dict'])
        for k, v in checkpoint['fc_state_dict'].items():
            model_state[f"fc.{k}"] = v
            
        model.load_state_dict(model_state)
        model = model.to(device)
        experts[task] = model
        
        head = nn.Linear(512, 10)
        head.load_state_dict(checkpoint['fc_state_dict'])
        head = head.to(device)
        expert_heads[task] = head

    progenitor = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    progenitor_state = {k: v.to(device) for k, v in progenitor.state_dict().items() if not k.startswith('fc.')}

    # Calibration datasets
    calib_loaders_clean = {task: get_dataloader(task, is_train=True, batch_size=128, subset_size=128) for task in ["mnist", "fashion", "cifar"]}
    calib_loaders_mpcs = {task: get_mpcs_dataloader(task, batch_size=64, subset_size=128) for task in ["mnist", "fashion", "cifar"]}

    # Test loaders (1000 samples)
    test_loaders = {task: get_dataloader(task, is_train=False, batch_size=256, subset_size=1000) for task in ["mnist", "fashion", "cifar"]}

    deep_layer_names = ["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2"]
    early_layer_name = "layer2.1.conv2"

    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    strategies = ["None", "BN-Calib", "SLR-WBC", "REC-Routing"]

    print("Starting Lambda Sweep for Task Arithmetic...")
    results = {lam: {cs: {"clean": 0.0, "corrupted": 0.0} for cs in strategies} for lam in lambdas}

    for lam in lambdas:
        print(f"\n==========================================")
        print(f" Evaluating Task Arithmetic Lambda = {lam:.2f}")
        print(f"==========================================")
        
        tau_mnist = {k: experts["mnist"].state_dict()[k] - progenitor_state[k] for k in progenitor_state.keys()}
        tau_fashion = {k: experts["fashion"].state_dict()[k] - progenitor_state[k] for k in progenitor_state.keys()}
        tau_cifar = {k: experts["cifar"].state_dict()[k] - progenitor_state[k] for k in progenitor_state.keys()}
        
        merged_state = {}
        for k in progenitor_state.keys():
            merged_state[k] = progenitor_state[k] + lam * (tau_mnist[k] + tau_fashion[k] + tau_cifar[k])
            
        def get_merged_backbone():
            mb = resnet18()
            mb.fc = nn.Identity()
            mb_state = mb.state_dict()
            mb_state.update(merged_state)
            mb.load_state_dict(mb_state)
            return mb.to(device)
            
        for cs in strategies:
            backbones = {}
            routing_b = None
            
            if cs == "None":
                shared_b = get_merged_backbone()
                backbones = {task: shared_b for task in ["mnist", "fashion", "cifar"]}
            elif cs == "BN-Calib":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_clean[task], device=device)
                    backbones[task] = mb
            elif cs == "SLR-WBC":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_clean[task], device=device)
                    apply_slr_wbc_single_task(mb, experts[task], calib_loaders_clean[task], deep_layer_names, device, rank=8, is_robust=False)
                    backbones[task] = mb
            elif cs == "REC-Routing":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_mpcs[task], device=device)
                    backbones[task] = mb
                progenitor_dict = {t: progenitor for t in ["mnist", "fashion", "cifar"]}
                prototypes = get_mspr_prototypes(progenitor_dict, calib_loaders_mpcs, early_layer_name, device)
                routing_b = progenitor
                
            # 1. Evaluate Clean Performance
            if cs == "REC-Routing":
                acc = evaluate_mspr_routed(backbones, test_loaders, expert_heads, prototypes, early_layer_name, device, routing_backbone=routing_b)
                results[lam][cs]["clean"] = acc
            else:
                accs = []
                for task in ["mnist", "fashion", "cifar"]:
                    acc = evaluate_model(backbones[task], test_loaders[task], expert_heads[task], device)
                    accs.append(acc)
                results[lam][cs]["clean"] = np.mean(accs)
                
            # 2. Evaluate Corrupted OOD Performance
            selected_corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "pixelation", "jpeg_compression"]
            selected_severities = [1, 3, 5]
            corr_accs = []
            for corr in selected_corruptions:
                for sev in selected_severities:
                    if cs == "REC-Routing":
                        acc = evaluate_mspr_routed(backbones, test_loaders, expert_heads, prototypes, early_layer_name, device, corruption_name=corr, severity=sev, routing_backbone=routing_b)
                        corr_accs.append(acc)
                    else:
                        task_accs = []
                        for task in ["mnist", "fashion", "cifar"]:
                            acc = evaluate_model(backbones[task], test_loaders[task], expert_heads[task], device, corruption_name=corr, severity=sev)
                            task_accs.append(acc)
                        corr_accs.append(np.mean(task_accs))
            results[lam][cs]["corrupted"] = np.mean(corr_accs)
            
            print(f"Strategy: {cs:<12} | Clean Avg: {results[lam][cs]['clean']:6.2f}% | OOD Avg: {results[lam][cs]['corrupted']:6.2f}%")

    # Display final Comparative Table
    print("\n" + "="*80)
    print(" SWEEP RESULTS: TASK ARITHMETIC LAMBDA COEFF SWEEP")
    print("="*80)
    print(f"{'Lambda':<8} | {'Strategy':<15} | {'Clean Avg':<11} | {'OOD Avg':<11}")
    print("-"*55)
    for lam in lambdas:
        for cs in strategies:
            print(f"{lam:<8.2f} | {cs:<15} | {results[lam][cs]['clean']:9.2f}% | {results[lam][cs]['corrupted']:9.2f}%")
    print("="*85)

if __name__ == "__main__":
    main()
