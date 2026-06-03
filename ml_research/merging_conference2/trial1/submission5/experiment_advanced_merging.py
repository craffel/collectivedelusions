import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# Import helper functions and setup from the existing experiment script
from experiment import (
    device,
    get_dataset,
    get_dataloader,
    get_resnet18_backbone_and_head,
    MultiTaskModel,
    evaluate_model,
    calibrate_backbone_bn
)

def run_ties_merging(base_state, experts, tasks, p_trim=0.2, lam=0.4):
    print(f"\nMerging via TIES-Merging (p_trim={p_trim}, lam={lam})...")
    merged_state = {}
    
    for key in base_state.keys():
        if not base_state[key].is_floating_point():
            merged_state[key] = base_state[key].clone()
            continue
            
        # Compute task vectors
        task_vectors = []
        for t in tasks:
            vec = experts[t][0].state_dict()[key] - base_state[key]
            task_vectors.append(vec)
            
        # Step 1: Trim top p_trim absolute values
        trimmed_vectors = []
        for vec in task_vectors:
            flat_vec = vec.flatten()
            k_val = int(p_trim * len(flat_vec))
            if k_val > 0:
                threshold = torch.topk(torch.abs(flat_vec), k_val).values[-1]
                mask = torch.abs(vec) >= threshold
                trimmed_vec = vec * mask
            else:
                trimmed_vec = torch.zeros_like(vec)
            trimmed_vectors.append(trimmed_vec)
            
        # Step 2: Elect Sign
        signs = torch.stack([torch.sign(v) for v in trimmed_vectors])
        sign_sum = signs.sum(dim=0)
        majority_sign = torch.sign(sign_sum)
        
        # Step 3: Disagreement Resolution and Merging
        agreements = torch.stack([(torch.sign(v) == majority_sign) & (v != 0) for v in trimmed_vectors])
        sum_agreements = agreements.sum(dim=0)
        
        sum_values = torch.zeros_like(base_state[key])
        for i, v in enumerate(trimmed_vectors):
            sum_values += v * agreements[i]
            
        mask_agreed = sum_agreements > 0
        merged_vector = torch.zeros_like(base_state[key])
        merged_vector[mask_agreed] = sum_values[mask_agreed] / sum_agreements[mask_agreed]
        
        # Apply scaling and merge
        merged_state[key] = base_state[key] + lam * merged_vector
        
    return merged_state

def run_dare_merging(base_state, experts, tasks, p_drop=0.5, lam=0.4):
    print(f"\nMerging via DARE (p_drop={p_drop}, lam={lam})...")
    merged_state = {}
    
    for key in base_state.keys():
        if not base_state[key].is_floating_point():
            merged_state[key] = base_state[key].clone()
            continue
            
        # Compute task vectors
        task_vectors = []
        for t in tasks:
            vec = experts[t][0].state_dict()[key] - base_state[key]
            task_vectors.append(vec)
            
        # Step 1: Drop and Rescale
        dare_vectors = []
        for vec in task_vectors:
            # Mask out values with probability p_drop
            mask = (torch.rand_like(vec) > p_drop).float()
            rescaled_vec = (vec * mask) / (1.0 - p_drop)
            dare_vectors.append(rescaled_vec)
            
        # Step 2: Average and scale
        merged_vector = torch.stack(dare_vectors).mean(dim=0)
        merged_state[key] = base_state[key] + lam * merged_vector
        
    return merged_state

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Load experts
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
    # 1. TIES-Merging
    # -------------------------------------------------------------
    ties_state = run_ties_merging(base_state, experts, tasks, p_trim=0.2, lam=0.4)
    ties_backbone, _ = get_resnet18_backbone_and_head()
    ties_backbone = ties_backbone.to(device)
    ties_backbone.load_state_dict(ties_state)
    
    # 2. DARE-Merging
    dare_state = run_dare_merging(base_state, experts, tasks, p_drop=0.5, lam=0.4)
    dare_backbone, _ = get_resnet18_backbone_and_head()
    dare_backbone = dare_backbone.to(device)
    dare_backbone.load_state_dict(dare_state)
    
    # Calibration loaders (N=128)
    calib_loaders = {}
    for task in tasks:
        calib_dataset = get_dataset(task, is_train=True)
        calib_subset = Subset(calib_dataset, list(range(128)))
        calib_loaders[task] = DataLoader(calib_subset, batch_size=64, shuffle=False)
        
    results = {
        "ties": {
            "baseline": {},
            "tcac_shared": {},
            "tcac_expert": {}
        },
        "dare": {
            "baseline": {},
            "tcac_shared": {},
            "tcac_expert": {}
        }
    }
    
    # Evaluate TIES
    print("\nEvaluating TIES-Merging...")
    ties_baselines = []
    ties_shareds = []
    ties_experts = []
    for task in tasks:
        # 1. Baseline
        base_acc = evaluate_model(ties_backbone, experts[task][1], test_loaders[task])
        results["ties"]["baseline"][task] = base_acc
        ties_baselines.append(base_acc)
        
        # 2. TCAC Shared
        calibrated_shared = calibrate_backbone_bn(
            ties_backbone, calib_loaders[task], device, use_expert_affine=False
        )
        shared_acc = evaluate_model(calibrated_shared, experts[task][1], test_loaders[task])
        results["ties"]["tcac_shared"][task] = shared_acc
        ties_shareds.append(shared_acc)
        
        # 3. TCAC Task-Specific
        calibrated_expert = calibrate_backbone_bn(
            ties_backbone, calib_loaders[task], device,
            expert_backbone=experts[task][0], use_expert_affine=True
        )
        expert_acc = evaluate_model(calibrated_expert, experts[task][1], test_loaders[task])
        results["ties"]["tcac_expert"][task] = expert_acc
        ties_experts.append(expert_acc)
        
        print(f"  {task.upper()} -> Baseline: {base_acc:.2f}%, TCAC Shared: {shared_acc:.2f}%, TCAC Task-Specific: {expert_acc:.2f}%")
        
    avg_ties_base = sum(ties_baselines) / len(tasks)
    avg_ties_shared = sum(ties_shareds) / len(tasks)
    avg_ties_expert = sum(ties_experts) / len(tasks)
    print(f"TIES-Merging Averages -> Baseline: {avg_ties_base:.2f}%, TCAC Shared: {avg_ties_shared:.2f}%, TCAC Task-Specific: {avg_ties_expert:.2f}%")
    results["ties"]["average_baseline"] = avg_ties_base
    results["ties"]["average_shared"] = avg_ties_shared
    results["ties"]["average_expert"] = avg_ties_expert
    
    # Evaluate DARE
    print("\nEvaluating DARE...")
    dare_baselines = []
    dare_shareds = []
    dare_experts = []
    for task in tasks:
        # 1. Baseline
        base_acc = evaluate_model(dare_backbone, experts[task][1], test_loaders[task])
        results["dare"]["baseline"][task] = base_acc
        dare_baselines.append(base_acc)
        
        # 2. TCAC Shared
        calibrated_shared = calibrate_backbone_bn(
            dare_backbone, calib_loaders[task], device, use_expert_affine=False
        )
        shared_acc = evaluate_model(calibrated_shared, experts[task][1], test_loaders[task])
        results["dare"]["tcac_shared"][task] = shared_acc
        dare_shareds.append(shared_acc)
        
        # 3. TCAC Task-Specific
        calibrated_expert = calibrate_backbone_bn(
            dare_backbone, calib_loaders[task], device,
            expert_backbone=experts[task][0], use_expert_affine=True
        )
        expert_acc = evaluate_model(calibrated_expert, experts[task][1], test_loaders[task])
        results["dare"]["tcac_expert"][task] = expert_acc
        dare_experts.append(expert_acc)
        
        print(f"  {task.upper()} -> Baseline: {base_acc:.2f}%, TCAC Shared: {shared_acc:.2f}%, TCAC Task-Specific: {expert_acc:.2f}%")
        
    avg_dare_base = sum(dare_baselines) / len(tasks)
    avg_dare_shared = sum(dare_shareds) / len(tasks)
    avg_dare_expert = sum(dare_experts) / len(tasks)
    print(f"DARE Averages -> Baseline: {avg_dare_base:.2f}%, TCAC Shared: {avg_dare_shared:.2f}%, TCAC Task-Specific: {avg_dare_expert:.2f}%")
    results["dare"]["average_baseline"] = avg_dare_base
    results["dare"]["average_shared"] = avg_dare_shared
    results["dare"]["average_expert"] = avg_dare_expert
    
    # Save the results to a json file
    with open("results/advanced_merging_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAdvanced merging results saved to results/advanced_merging_results.json!")

if __name__ == "__main__":
    main()
