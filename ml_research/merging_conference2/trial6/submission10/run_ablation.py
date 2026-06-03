import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import copy
from merge_and_evaluate import (
    get_dataloaders,
    load_expert,
    reset_bn_stats,
    calibrate_model,
    optimize_synthetic_data,
    evaluate_model,
    merge_weights_ta
)

def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading datasets...")
    loaders = get_dataloaders()
    
    print("Loading experts...")
    expert_mnist = load_expert("mnist")
    expert_fmnist = load_expert("fmnist")
    expert_cifar = load_expert("cifar10")
    
    experts = [expert_mnist, expert_fmnist, expert_cifar]
    task_names = ["mnist", "fmnist", "cifar10"]
    
    # Pre-load base progenitor
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model = base_model.to(device)
    
    # Merge weights using Task Arithmetic lambda=0.5
    merged_model_raw = merge_weights_ta(experts, base_model, lam=0.5)
    
    # Epoch counts to evaluate
    epoch_options = [10, 25, 50, 100, 150]
    
    print("\n=== STARTING SYNTHESIS EPOCH ABLATION ===")
    
    # Clear previous results if any
    with open("ablation_results.txt", "w") as f:
        f.write("=== Synthesis Epoch Ablation Results ===\n")
    
    for ep in epoch_options:
        print(f"\n--- Evaluation with {ep} Optimization Epochs ---")
        
        # Optimize synthetic loaders
        gen_loaders = {
            "mnist": optimize_synthetic_data(expert_mnist, size=256, batch_size=64, epochs=ep, lr=0.1),
            "fmnist": optimize_synthetic_data(expert_fmnist, size=256, batch_size=64, epochs=ep, lr=0.1),
            "cifar10": optimize_synthetic_data(expert_cifar, size=256, batch_size=64, epochs=ep, lr=0.1)
        }
        
        # Evaluate
        task_accs = {}
        for eval_task in task_names:
            model_to_eval = copy.deepcopy(merged_model_raw)
            target_expert = expert_mnist if eval_task == "mnist" else (expert_fmnist if eval_task == "fmnist" else expert_cifar)
            model_to_eval.fc.load_state_dict(target_expert.fc.state_dict())
            
            reset_bn_stats(model_to_eval)
            calibrate_model(model_to_eval, gen_loaders[eval_task])
            
            acc = evaluate_model(model_to_eval, loaders[eval_task]["test"])
            task_accs[eval_task] = acc
            print(f"[{ep} epochs] Task {eval_task.upper()} Accuracy: {acc:.2f}%")
            
        avg_acc = np.mean(list(task_accs.values()))
        print(f"[{ep} epochs] Average Multi-Task Accuracy: {avg_acc:.2f}%")
        
        # Append to ablation_results.txt
        with open("ablation_results.txt", "a") as f:
            f.write(f"Epochs: {ep} | MNIST: {task_accs['mnist']:.2f}% | FMNIST: {task_accs['fmnist']:.2f}% | CIFAR10: {task_accs['cifar10']:.2f}% | Avg: {avg_acc:.2f}%\n")

if __name__ == "__main__":
    run_ablation()
