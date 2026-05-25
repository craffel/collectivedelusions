import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from run_experiments import (
    set_seed,
    get_modified_resnet18,
    get_datasets,
    compute_layer_wise_fisher,
    compute_prototypes,
    build_test_streams,
    apply_corruption,
    detect_active_task,
    MergedModel
)

def run_evaluation_continuous(base_model, experts, stream_batches, prototypes, group_fishers, corruption, method, device):
    adapted_base = get_modified_resnet18(num_classes=10).to(device)
    adapted_base.load_state_dict(base_model.state_dict())
    
    merged_model = MergedModel(adapted_base, experts, device)
    
    lr = 0.05
    if "ours" in method:
        lr = 0.1
        
    coeff_history = []
    correct = 0
    total = 0
    
    active_task_predictions = []
    actual_tasks = []
    
    optimizer = optim.SGD([merged_model.logits], lr=lr)
    
    prev_pred_task = None
    
    for b_idx, (batch_subset, actual_task) in enumerate(stream_batches):
        loader = DataLoader(batch_subset, batch_size=32, shuffle=False)
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs_corrupted = apply_corruption(inputs, corruption)
        
        pred_task = detect_active_task(inputs_corrupted, experts, prototypes, device)
        active_task_predictions.append(pred_task)
        actual_tasks.append(actual_task)
        
        # Reset strategy
        if method in ["cpa_reset", "ours_reset"]:
            # Standard: Reset on every batch
            with torch.no_grad():
                merged_model.logits.copy_(torch.full_like(merged_model.logits, -10.0))
                merged_model.logits[:, pred_task].copy_(torch.full_like(merged_model.logits[:, pred_task], 10.0))
        elif method in ["cpa_continuous", "ours_continuous"]:
            # Task-boundary-aware: Only reset when task changes
            if b_idx == 0 or pred_task != prev_pred_task:
                with torch.no_grad():
                    merged_model.logits.copy_(torch.full_like(merged_model.logits, -10.0))
                    merged_model.logits[:, pred_task].copy_(torch.full_like(merged_model.logits[:, pred_task], 10.0))
        elif method == "adamerge":
            # Baseline: Never reset
            pass
            
        coeffs = merged_model.get_coefficients().detach().cpu().numpy().copy()
        coeff_history.append(coeffs)
        
        merged_model.logits.requires_grad = True
        
        if method == "adamerge":
            outputs = merged_model(inputs_corrupted)
        else:
            outputs, feats = merged_model(inputs_corrupted, return_features=True)
            
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        if method == "static":
            continue
            
        optimizer.zero_grad()
        
        if method == "adamerge":
            probs = torch.softmax(outputs, dim=1)
            loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
            loss.backward()
            
            with torch.no_grad():
                if merged_model.logits.grad is not None:
                    grad = merged_model.logits.grad.clone()
                    merged_model.logits.data -= lr * grad
        else:
            # Differentiable InfoNCE contrastive alignment loss
            protos_tensor = torch.stack([prototypes[pred_task][c] for c in range(10)]) # (10, 512)
            
            # Apply Isotropic Feature Centering (IFC)
            feats_mean = feats.mean(dim=0, keepdim=True)
            feats_centered = feats - feats_mean
            feats_norm = nn.functional.normalize(feats_centered, p=2, dim=1) # (batch_size, 512)
            
            protos_mean = protos_tensor.mean(dim=0, keepdim=True)
            protos_centered = protos_tensor - protos_mean
            protos_norm = nn.functional.normalize(protos_centered, p=2, dim=1) # (10, 512)
            
            sim_matrix = torch.matmul(feats_norm, protos_norm.T) / 0.1
            
            pseudo_labels = outputs.argmax(dim=1)
            criterion_contrastive = nn.CrossEntropyLoss(reduction='none')
            
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = probs.max(dim=1)
            confidence_mask = (max_probs >= 0.7)
            
            if confidence_mask.sum() > 0:
                loss = criterion_contrastive(sim_matrix[confidence_mask], pseudo_labels[confidence_mask]).mean()
            else:
                loss = None
                
            if loss is not None:
                loss.backward()
                
                with torch.no_grad():
                    if merged_model.logits.grad is not None:
                        grad = merged_model.logits.grad.clone()
                        
                        if "ours" in method:
                            for g_idx, g_name in enumerate(merged_model.group_names):
                                f_active = group_fishers[pred_task][g_name]
                                f_inactive = sum([group_fishers[k][g_name] for k in range(3) if k != pred_task]) / 2.0
                                
                                cop_factor = f_active / (f_inactive + 1e-3)
                                # Let's use the optimal wide constraint bounds we discovered!
                                cop_factor = np.clip(cop_factor, 0.01, 10.0)
                                grad[g_idx] *= cop_factor
                                
                        merged_model.logits.data -= lr * grad
                        
        merged_model.logits.grad = None
        prev_pred_task = pred_task
        
    accuracy = 100.0 * correct / total
    routing_accuracy = 100.0 * np.mean(np.array(active_task_predictions) == np.array(actual_tasks))
    
    return accuracy, routing_accuracy, np.array(coeff_history)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load datasets
    (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test) = get_datasets()
    
    # Load experts
    experts = []
    expert_paths = ["expert_mnist.pt", "expert_fmnist.pt", "expert_kmnist.pt"]
    expert_datasets = [mnist_train, fmnist_train, kmnist_train]
    expert_names = ["MNIST", "FashionMNIST", "KMNIST"]
    
    for path, dataset, name in zip(expert_paths, expert_datasets, expert_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model path {path} not found! Please run run_experiments.py first.")
        model = get_modified_resnet18(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        experts.append(model)
        
    # Base model
    base_model = get_modified_resnet18(num_classes=10).to(device)
    
    # Pre-compute Fisher and Prototypes
    print("--- Pre-computing Fisher and Prototypes ---")
    group_fishers = []
    prototypes = []
    for k in range(3):
        fish = compute_layer_wise_fisher(experts[k], expert_datasets[k], device, num_samples=500)
        group_fishers.append(fish)
        
        protos = compute_prototypes(experts[k], expert_datasets[k], device, num_samples=1000)
        prototypes.append(protos)
        
    # Build sequential stream
    _, seq_batches = build_test_streams(mnist_test, fmnist_test, kmnist_test)
    
    corruptions = ["clean", "gaussian_noise", "contrast_shift"]
    methods = ["adamerge", "cpa_reset", "ours_reset", "cpa_continuous", "ours_continuous"]
    
    results = {corr: {} for corr in corruptions}
    trajectories = {}
    
    for corr in corruptions:
        print(f"\nEvaluating Sequential Stream under corruption: {corr}...")
        for method in methods:
            acc, r_acc, coeff_hist = run_evaluation_continuous(
                base_model, experts, seq_batches, prototypes, group_fishers, corr, method, device
            )
            print(f"  Method: {method:<15} | Acc: {acc:.2f}% | Routing Acc: {r_acc:.2f}%")
            results[corr][method] = {
                "accuracy": acc,
                "routing_accuracy": r_acc
            }
            if corr == "gaussian_noise" and method in ["cpa_continuous", "ours_continuous", "adamerge"]:
                trajectories[method] = coeff_hist
                
    # Save results to JSON
    with open("continuous_adaptation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nContinuous adaptation study complete! Saved results to continuous_adaptation_results.json")
    
    # Plot trajectories for continuous evaluation under Gaussian Noise
    if "ours_continuous" in trajectories and "cpa_continuous" in trajectories:
        print("\n--- Generating Continuous Trajectory Plots ---")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, method in enumerate(["cpa_continuous", "ours_continuous"]):
            global_coeffs = trajectories[method].mean(axis=1) # shape (150, 3)
            
            axes[idx].plot(global_coeffs[:, 0], label="$\lambda_1$ (MNIST)", color="red", linewidth=2)
            axes[idx].plot(global_coeffs[:, 1], label="$\lambda_2$ (FashionMNIST)", color="green", linewidth=2)
            axes[idx].plot(global_coeffs[:, 2], label="$\lambda_3$ (KMNIST)", color="blue", linewidth=2)
            
            axes[idx].axvline(x=50, color="gray", linestyle="--")
            axes[idx].axvline(x=100, color="gray", linestyle="--")
            
            axes[idx].set_title(f"Continuous Trajectory - {'CPA-Continuous' if method == 'cpa_continuous' else 'RGS-COP-Continuous (Ours)'}", fontsize=14)
            axes[idx].set_xlabel("Adaptation Steps (Batches)", fontsize=12)
            axes[idx].set_ylabel("Merging Coefficient Value", fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=11)
            axes[idx].set_ylim(-0.05, 1.05)
            
        plt.tight_layout()
        plt.savefig("continuous_coefficient_trajectories.png", dpi=300)
        print("Successfully saved trajectory plot to continuous_coefficient_trajectories.png!")

if __name__ == "__main__":
    main()
