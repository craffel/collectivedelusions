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
    # logits shape: [B, num_classes]
    probs = torch.softmax(logits, dim=-1)
    # prevent log(0) with eps
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy

def main():
    tasks = ["mnist", "fashion", "cifar10"]
    print(f"Running zero-shot task routing benchmark on device: {device}")
    
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
    
    # Setup calibration loaders (N=128)
    print("Calibrating TCAC backbones...")
    tcac_backbones = {}
    for task in tasks:
        calib_dataset = get_dataset(task, is_train=True)
        calib_subset = Subset(calib_dataset, list(range(128)))
        calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
        
        # TCAC Task-Specific Affine
        tcac_backbones[task] = calibrate_backbone_bn(
            merged_backbone, calib_loader, device,
            expert_backbone=experts[task][0], use_expert_affine=True
        ).eval()
        
    # Test routing on 500 samples per task (for speed on CPU, or standard on GPU)
    num_samples_per_task = 500
    print(f"Evaluating task routing on {num_samples_per_task} samples per task...")
    
    # Store results
    routing_correct = {metric: 0 for metric in ["max_prob", "min_entropy"]}
    total_routed = 0
    
    classification_correct_with_routing = {metric: 0 for metric in ["max_prob", "min_entropy"]}
    classification_correct_oracle = 0
    
    confusion_matrix = {
        metric: {true_task: {pred_task: 0 for pred_task in tasks} for true_task in tasks}
        for metric in ["max_prob", "min_entropy"]
    }
    
    for true_task_idx, true_task in enumerate(tasks):
        dataset = get_dataset(true_task, is_train=False)
        subset_indices = list(range(min(num_samples_per_task, len(dataset))))
        loader = DataLoader(Subset(dataset, subset_indices), batch_size=64, shuffle=False)
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_routed += batch_size
            
            # For each input, we evaluate the predictions under all 3 calibrated backbones and heads
            # Shape of outputs for each metric: [batch_size, num_tasks]
            task_confidences = {
                "max_prob": torch.zeros(batch_size, len(tasks), device=device),
                "min_entropy": torch.zeros(batch_size, len(tasks), device=device) # stored as negative entropy
            }
            
            task_predictions = {}
            
            for t_idx, t in enumerate(tasks):
                backbone = tcac_backbones[t]
                head = experts[t][1]
                
                with torch.no_grad():
                    features = backbone(inputs)
                    logits = head(features)
                    
                    # Compute prediction
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    entropy = compute_entropy(logits)
                    
                    task_confidences["max_prob"][:, t_idx] = max_probs
                    task_confidences["min_entropy"][:, t_idx] = -entropy # higher is lower entropy
                    task_predictions[t] = preds
            
            # Perform routing
            for metric in ["max_prob", "min_entropy"]:
                # Select the task index with the highest score
                pred_task_indices = torch.argmax(task_confidences[metric], dim=-1)
                
                for i in range(batch_size):
                    pred_task_idx = pred_task_indices[i].item()
                    pred_task = tasks[pred_task_idx]
                    
                    # Log to confusion matrix
                    confusion_matrix[metric][true_task][pred_task] += 1
                    
                    # Check if routing is correct
                    is_correct_routing = (pred_task_idx == true_task_idx)
                    if is_correct_routing:
                        routing_correct[metric] += 1
                        
                    # Check if final classification is correct
                    final_pred = task_predictions[pred_task][i]
                    if final_pred == targets[i]:
                        classification_correct_with_routing[metric] += 1
                        
            # Oracle classification (where routing is always correct)
            with torch.no_grad():
                features = tcac_backbones[true_task](inputs)
                logits = experts[true_task][1](features)
                _, preds = torch.max(logits, dim=-1)
                classification_correct_oracle += preds.eq(targets).sum().item()
                
    # Compile metrics
    results = {}
    for metric in ["max_prob", "min_entropy"]:
        routing_acc = (routing_correct[metric] / total_routed) * 100.0
        class_acc = (classification_correct_with_routing[metric] / total_routed) * 100.0
        
        results[metric] = {
            "routing_accuracy": routing_acc,
            "classification_accuracy_with_routing": class_acc,
            "confusion_matrix": confusion_matrix[metric]
        }
        
    oracle_class_acc = (classification_correct_oracle / total_routed) * 100.0
    results["oracle_classification_accuracy"] = oracle_class_acc
    
    print("\n=======================================================")
    print("RESULTS: Zero-Shot Task Routing Benchmark")
    print("=======================================================")
    print(f"Oracle Classification Accuracy: {oracle_class_acc:.2f}%")
    for metric in ["max_prob", "min_entropy"]:
        print(f"\nMetric: {metric.upper()}")
        print(f"  Routing Accuracy: {results[metric]['routing_accuracy']:.2f}%")
        print(f"  Classification Accuracy with Routing: {results[metric]['classification_accuracy_with_routing']:.2f}%")
        print("  Confusion Matrix (True \\ Pred):")
        for true_t in tasks:
            row = [f"{pred_t}: {results[metric]['confusion_matrix'][true_t][pred_t]}" for pred_t in tasks]
            print(f"    {true_t:10} | " + " | ".join(row))
            
    # Save results to file
    with open("results/routing_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nRouting results saved to results/routing_results.json!")

if __name__ == "__main__":
    main()