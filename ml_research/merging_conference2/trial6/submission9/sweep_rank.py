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
    print("\nLoading trained expert weights for SLR-WBC Rank sweep...")
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

    # Calibration datasets (clean)
    calib_loaders_clean = {task: get_dataloader(task, is_train=True, batch_size=128, subset_size=128) for task in ["mnist", "fashion", "cifar"]}

    # Test loaders (1000 samples)
    test_loaders = {task: get_dataloader(task, is_train=False, batch_size=256, subset_size=1000) for task in ["mnist", "fashion", "cifar"]}

    deep_layer_names = ["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2"]

    ranks = [2, 4, 8, 16, 32]
    
    # We do the sweep under Weight Averaging (WA)
    print("Starting Rank Sweep for SLR-WBC under Weight Averaging...")
    results = {r: {"clean": 0.0, "corrupted": 0.0} for r in ranks}

    for r in ranks:
        print(f"\nEvaluating SLR-WBC with Rank = {r}")
        
        # Merge backbones with Weight Averaging
        merged_state = {}
        for k in progenitor_state.keys():
            merged_state[k] = (experts["mnist"].state_dict()[k] + 
                               experts["fashion"].state_dict()[k] + 
                               experts["cifar"].state_dict()[k]) / 3.0
            
        def get_merged_backbone():
            mb = resnet18()
            mb.fc = nn.Identity()
            mb_state = mb.state_dict()
            mb_state.update(merged_state)
            mb.load_state_dict(mb_state)
            return mb.to(device)
            
        backbones = {}
        for task in ["mnist", "fashion", "cifar"]:
            mb = get_merged_backbone()
            calibrate_bn(mb, calib_loaders_clean[task], device=device)
            apply_slr_wbc_single_task(mb, experts[task], calib_loaders_clean[task], deep_layer_names, device, rank=r, is_robust=False)
            backbones[task] = mb
            
        # 1. Evaluate Clean Performance
        accs = []
        for task in ["mnist", "fashion", "cifar"]:
            acc = evaluate_model(backbones[task], test_loaders[task], expert_heads[task], device)
            accs.append(acc)
        results[r]["clean"] = np.mean(accs)
            
        # 2. Evaluate Corrupted OOD Performance
        selected_corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "pixelation", "jpeg_compression"]
        selected_severities = [1, 3, 5]
        corr_accs = []
        for corr in selected_corruptions:
            for sev in selected_severities:
                task_accs = []
                for task in ["mnist", "fashion", "cifar"]:
                    acc = evaluate_model(backbones[task], test_loaders[task], expert_heads[task], device, corruption_name=corr, severity=sev)
                    task_accs.append(acc)
                corr_accs.append(np.mean(task_accs))
        results[r]["corrupted"] = np.mean(corr_accs)
        
        print(f"Rank: {r:<4} | Clean Avg: {results[r]['clean']:6.2f}% | OOD Avg: {results[r]['corrupted']:6.2f}%")

    # Display final Comparative Table
    print("\n" + "="*50)
    print(" SWEEP RESULTS: SLR-WBC SVD TRUNCATION RANK SWEEP")
    print("="*50)
    print(f"{'Rank':<8} | {'Clean Avg':<11} | {'OOD Avg':<11}")
    print("-"*35)
    for r in ranks:
        print(f"{r:<8} | {results[r]['clean']:9.2f}% | {results[r]['corrupted']:9.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
