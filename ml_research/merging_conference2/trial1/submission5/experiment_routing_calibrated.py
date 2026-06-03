import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# Import helpers from the existing experiment script
from experiment import (
    device,
    get_dataset,
    get_dataloader,
    get_resnet18_backbone_and_head,
    MultiTaskModel,
    calibrate_backbone_bn
)

def compute_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy

def compute_energy(logits, T=1.0):
    # energy = -T * logsumexp(logits / T)
    # we want to route to the task with HIGHER negative energy (i.e., logsumexp)
    return T * torch.logsumexp(logits / T, dim=-1)

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    print(f"Running calibrated zero-shot task routing benchmark on device: {device}")
    
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
    
    # Perform Task Arithmetic merging with lambda = 0.4
    print("Performing Task Arithmetic merging...")
    lam = 0.4
    merged_backbone_state = copy.deepcopy(base_backbone.state_dict())
    task_vectors = {}
    for task in tasks:
        expert_state = experts[task][0].state_dict()
        task_vectors[task] = {}
        for key in expert_state.keys():
            task_vectors[task][key] = expert_state[key] - merged_backbone_state[key]
            
    for key in merged_backbone_state.keys():
        if merged_backbone_state[key].is_floating_point():
            sum_vector = sum(task_vectors[task][key] for task in tasks)
            merged_backbone_state[key] += lam * sum_vector
            
    merged_backbone = copy.deepcopy(base_backbone)
    merged_backbone.load_state_dict(merged_backbone_state)
    merged_backbone = merged_backbone.to(device)
    
    # Setup calibration loaders (N=128) and compute normalization statistics
    print("Calibrating TCAC backbones and extracting routing calibration profiles...")
    tcac_backbones = {}
    calib_stats = {
        "mean_max_prob": {},
        "std_max_prob": {},
        "mean_entropy": {},
        "std_entropy": {},
        "mean_energy": {},
        "std_energy": {}
    }
    
    for task in tasks:
        calib_dataset = get_dataset(task, is_train=True)
        calib_subset = Subset(calib_dataset, list(range(128)))
        calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
        
        # TCAC Task-Specific Affine
        tcac_backbones[task] = calibrate_backbone_bn(
            merged_backbone, calib_loader, device,
            expert_backbone=experts[task][0], use_expert_affine=True
        ).eval()
        
        # Compute baseline calibration profile of the correct head on its own calibration set
        head = experts[task][1]
        probs_list = []
        entropy_list = []
        energy_list = []
        
        with torch.no_grad():
            for inputs, _ in calib_loader:
                inputs = inputs.to(device)
                features = tcac_backbones[task](inputs)
                logits = head(features)
                
                probs = torch.softmax(logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                entropy = compute_entropy(logits)
                energy = compute_energy(logits)
                
                probs_list.append(max_probs)
                entropy_list.append(entropy)
                energy_list.append(energy)
                
        all_probs = torch.cat(probs_list)
        all_entropies = torch.cat(entropy_list)
        all_energies = torch.cat(energy_list)
        
        calib_stats["mean_max_prob"][task] = all_probs.mean().item()
        calib_stats["std_max_prob"][task] = all_probs.std().item() + 1e-8
        calib_stats["mean_entropy"][task] = all_entropies.mean().item()
        calib_stats["std_entropy"][task] = all_entropies.std().item() + 1e-8
        calib_stats["mean_energy"][task] = all_energies.mean().item()
        calib_stats["std_energy"][task] = all_energies.std().item() + 1e-8
        
        print(f"  {task.upper()} calib stats -> "
              f"Max Prob: {calib_stats['mean_max_prob'][task]:.4f} (std={calib_stats['std_max_prob'][task]:.4f}) | "
              f"Entropy: {calib_stats['mean_entropy'][task]:.4f} (std={calib_stats['std_entropy'][task]:.4f}) | "
              f"Energy: {calib_stats['mean_energy'][task]:.4f} (std={calib_stats['std_energy'][task]:.4f})")
        
    # Evaluate routing metrics on 500 test samples per task
    num_samples_per_task = 500
    print(f"\nEvaluating task routing on {num_samples_per_task} samples per task...")
    
    metrics = [
        "max_prob", "min_entropy", "calibrated_max_prob", "calibrated_min_entropy", "energy",
        "z_score_max_prob", "z_score_min_entropy", "z_score_energy"
    ]
    
    routing_correct = {metric: 0 for metric in metrics}
    classification_correct_with_routing = {metric: 0 for metric in metrics}
    total_routed = 0
    classification_correct_oracle = 0
    
    confusion_matrix = {
        metric: {true_task: {pred_task: 0 for pred_task in tasks} for true_task in tasks}
        for metric in metrics
    }
    
    for true_task_idx, true_task in enumerate(tasks):
        dataset = get_dataset(true_task, is_train=False)
        subset_indices = list(range(min(num_samples_per_task, len(dataset))))
        loader = DataLoader(Subset(dataset, subset_indices), batch_size=64, shuffle=False)
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_routed += batch_size
            
            # For each input, we evaluate predictions under all 3 calibrated backbones and heads
            scores = {metric: torch.zeros(batch_size, len(tasks), device=device) for metric in metrics}
            task_predictions = {}
            
            for t_idx, t in enumerate(tasks):
                backbone = tcac_backbones[t]
                head = experts[t][1]
                
                with torch.no_grad():
                    features = backbone(inputs)
                    logits = head(features)
                    
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    entropy = compute_entropy(logits)
                    energy = compute_energy(logits)
                    
                    task_predictions[t] = preds
                    
                    # 1. Standard Max Prob
                    scores["max_prob"][:, t_idx] = max_probs
                    # 2. Standard Min Entropy (negative entropy, higher is better)
                    scores["min_entropy"][:, t_idx] = -entropy
                    # 3. Calibrated Max Prob
                    scores["calibrated_max_prob"][:, t_idx] = max_probs / calib_stats["mean_max_prob"][t]
                    # 4. Calibrated Min Entropy (ratio of entropy, lower is better. Let's use negative ratio of entropy)
                    scores["calibrated_min_entropy"][:, t_idx] = -(entropy / calib_stats["mean_entropy"][t])
                    # 5. Energy-based (higher is better)
                    scores["energy"][:, t_idx] = energy / calib_stats["mean_energy"][t]
                    # 6. Z-Score Max Prob (higher is better)
                    scores["z_score_max_prob"][:, t_idx] = (max_probs - calib_stats["mean_max_prob"][t]) / calib_stats["std_max_prob"][t]
                    # 7. Z-Score Min Entropy (since lower entropy is better, negative z-score is higher-is-better)
                    scores["z_score_min_entropy"][:, t_idx] = -((entropy - calib_stats["mean_entropy"][t]) / calib_stats["std_entropy"][t])
                    # 8. Z-Score Energy (higher is better)
                    scores["z_score_energy"][:, t_idx] = (energy - calib_stats["mean_energy"][t]) / calib_stats["std_energy"][t]
            
            # Perform routing for each metric
            for metric in metrics:
                pred_task_indices = torch.argmax(scores[metric], dim=-1)
                
                for i in range(batch_size):
                    pred_task_idx = pred_task_indices[i].item()
                    pred_task = tasks[pred_task_idx]
                    
                    confusion_matrix[metric][true_task][pred_task] += 1
                    
                    if pred_task_idx == true_task_idx:
                        routing_correct[metric] += 1
                        
                    final_pred = task_predictions[pred_task][i]
                    if final_pred == targets[i]:
                        classification_correct_with_routing[metric] += 1
                        
            # Oracle classification (correct task label is always known)
            with torch.no_grad():
                features = tcac_backbones[true_task](inputs)
                logits = experts[true_task][1](features)
                _, preds = torch.max(logits, dim=-1)
                classification_correct_oracle += preds.eq(targets).sum().item()
                
    # Compile metrics
    results = {}
    for metric in metrics:
        routing_acc = (routing_correct[metric] / total_routed) * 100.0
        class_acc = (classification_correct_with_routing[metric] / total_routed) * 100.0
        
        results[metric] = {
            "routing_accuracy": routing_acc,
            "classification_accuracy_with_routing": class_acc,
            "confusion_matrix": confusion_matrix[metric]
        }
        
    oracle_class_acc = (classification_correct_oracle / total_routed) * 100.0
    results["oracle_classification_accuracy"] = oracle_class_acc
    results["calib_profiles"] = calib_stats
    
    print("\n=======================================================")
    print("RESULTS: Calibrated Zero-Shot Task Routing Benchmark")
    print("=======================================================")
    print(f"Oracle Classification Accuracy: {oracle_class_acc:.2f}%")
    
    for metric in metrics:
        print(f"\nMetric: {metric.upper()}")
        print(f"  Routing Accuracy: {results[metric]['routing_accuracy']:.2f}%")
        print(f"  Classification Accuracy with Routing: {results[metric]['classification_accuracy_with_routing']:.2f}%")
        print("  Confusion Matrix (True \\ Pred):")
        for true_t in tasks:
            row = [f"{pred_t}: {results[metric]['confusion_matrix'][true_t][pred_t]}" for pred_t in tasks]
            print(f"    {true_t:10} | " + " | ".join(row))
            
    # Save results to file
    with open("results/routing_results_calibrated.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nRouting results saved to results/routing_results_calibrated.json!")

if __name__ == "__main__":
    main()