import os
import sys
import json
import torch

from experiment import (
    set_seed,
    get_dataloaders,
    get_base_model_and_params,
    extract_lora_and_head,
    run_tta,
    evaluate_merged_model
)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    _, cifar_test_loader, _, svhn_test_loader = get_dataloaders(
        subset_size=10000, batch_size=64, seed=42
    )
    
    if not os.path.exists("cifar10_best_model.pt") or not os.path.exists("svhn_best_model.pt"):
        print("Error: Trained expert models not found!")
        return
        
    cifar_state = torch.load("cifar10_best_model.pt", map_location="cpu")
    svhn_state = torch.load("svhn_best_model.pt", map_location="cpu")
    
    lora1, head1 = extract_lora_and_head(cifar_state)
    lora2, head2 = extract_lora_and_head(svhn_state)
    
    base_model, base_params, base_buffers = get_base_model_and_params(device=device)
    
    results = []
    steps_tta = 100
    
    print("\n================== SWEEPING TTA LEARNING RATE ==================")
    lr_list = [0.005, 0.01, 0.02, 0.05, 0.10]
    for lr in lr_list:
        print(f"\nEvaluating lr = {lr}...")
        coeff, h1, h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=steps_tta, batch_size=64, lr=lr,
            rho=0.05, use_sam=True, use_sosr=False, device=device
        )
        c10_acc = evaluate_merged_model(base_model, base_params, base_buffers, lora1, h1, lora2, h2, coeff, cifar_test_loader, task_idx=1, device=device)
        svhn_acc = evaluate_merged_model(base_model, base_params, base_buffers, lora1, h1, lora2, h2, coeff, svhn_test_loader, task_idx=2, device=device)
        avg_acc = (c10_acc + svhn_acc) / 2.0
        
        results.append({
            "lr": lr,
            "cifar10": c10_acc,
            "svhn": svhn_acc,
            "avg": avg_acc
        })
        print(f"Results for lr = {lr} | CIFAR-10: {c10_acc:.2f}% | SVHN: {svhn_acc:.2f}% | Avg: {avg_acc:.2f}%")
        
    with open("sweep_results_lr.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSweep lr completed! Results saved to sweep_results_lr.json.")

if __name__ == "__main__":
    main()
