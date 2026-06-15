import torch
import torch.nn as nn
import numpy as np
from run_physical_validation import SimpleCNN, load_experts, build_data_splits, TASKS, blend_parameters_functional
from test_improved_router import ImprovedPhysicalRoutingHead, extract_improved_features, train_improved_router, evaluate_improved_routing

def test_seeds():
    experts = load_experts()
    SEEDS = [42, 43, 44]
    
    k_sweeps_results = {k: {task: [] for task in TASKS + ['Joint Mean']} for k in [0, 1, 2, 3, 4]}
    
    for seed in SEEDS:
        print(f"\n[Seed {seed}] Training improved routing head...")
        cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=seed)
        head, mean, std = train_improved_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=300)
        
        for k in [0, 1, 2, 3, 4]:
            results = {}
            total_correct = 0
            total_samples = 0
            base_model = SimpleCNN()
            uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
            
            with torch.no_grad():
                for task_id, task in enumerate(TASKS):
                    imgs = test_data[task]
                    lbls = test_labels[task]
                    
                    task_h_features = extract_improved_features(base_model, uniform_params, imgs)
                    task_norm_features = (task_h_features - mean) / std
                    
                    logits = head(task_norm_features)
                    alphas = torch.softmax(logits / 0.1, dim=-1)
                    mean_alphas = alphas.mean(dim=0)
                    
                    task_params = blend_parameters_functional(experts, mean_alphas, k=k)
                    out = torch.func.functional_call(base_model, task_params, imgs)
                    _, predicted = out.max(1)
                    correct = predicted.eq(lbls).sum().item()
                    acc = 100.0 * correct / len(lbls)
                    
                    results[task] = acc
                    total_correct += correct
                    total_samples += len(lbls)
                    
                joint_mean = 100.0 * total_correct / total_samples
                k_sweeps_results[k]['Joint Mean'].append(joint_mean)
                for task in TASKS:
                    k_sweeps_results[k][task].append(results[task])
                    
    print("\n" + "-"*80)
    print("--- SUMMARY OF PHYSICAL SWEEP OVER k WITH IMPROVED ROUTER ---")
    print("-"*80)
    for k in [0, 1, 2, 3, 4]:
        res = k_sweeps_results[k]
        print(f"k={k} | MNIST: {np.mean(res['MNIST']):.2f}% ± {np.std(res['MNIST']):.2f}% | "
              f"FMNIST: {np.mean(res['FashionMNIST']):.2f}% ± {np.std(res['FashionMNIST']):.2f}% | "
              f"CIFAR10: {np.mean(res['CIFAR10']):.2f}% ± {np.std(res['CIFAR10']):.2f}% | "
              f"SVHN: {np.mean(res['SVHN']):.2f}% ± {np.std(res['SVHN']):.2f}% | "
              f"Joint Mean: {np.mean(res['Joint Mean']):.2f}% ± {np.std(res['Joint Mean']):.2f}%")

if __name__ == '__main__':
    test_seeds()
