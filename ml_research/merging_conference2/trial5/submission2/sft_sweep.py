import torch
import json
import os
from run_experiments import (
    device, subsets, test_loaders, load_expert_model, merge_wa, sft_heads, evaluate
)

def run_sft_sweep():
    print(f"Using device: {device}")
    
    # Load experts
    print("Loading expert models...")
    experts = [load_expert_model(name) for name in ["mnist", "fmnist", "cifar10"]]
    
    # Merge WA
    print("Merging models via WA...")
    wa_backbone = merge_wa(experts)
    
    # Extract original heads dict
    heads_dict = {}
    for name, exp in zip(["mnist", "fmnist", "cifar10"], experts):
        heads_dict[name] = {k.replace("fc.", ""): v.cpu().clone() for k, v in exp.fc.state_dict().items()}
        
    # Get calibration subsets
    calib_sets = {name: subsets[name]["calib"] for name in subsets}
    
    # Sweeps
    epochs_list = [5, 10, 20]
    lr_list = [1e-4, 5e-4, 1e-3, 5e-3]
    
    sweep_results = {}
    
    for epochs in epochs_list:
        sweep_results[epochs] = {}
        for lr in lr_list:
            print(f"\n--- Running SFT with epochs={epochs}, lr={lr} ---")
            adapted_heads = sft_heads(
                backbone_weights=wa_backbone,
                original_heads_dict=heads_dict,
                calib_subsets=calib_sets,
                num_samples=64,
                epochs=epochs,
                lr=lr
            )
            eval_res = evaluate(wa_backbone, adapted_heads, test_loaders)
            sweep_results[epochs][str(lr)] = eval_res
            print(f"Result (epochs={epochs}, lr={lr}): MNIST={eval_res['mnist']:.2f}%, F-MNIST={eval_res['fmnist']:.2f}%, CIFAR-10={eval_res['cifar10']:.2f}%, Avg={eval_res['average']:.2f}%")
            
    # Save sweep results to results/sft_sensitivity_results.json
    os.makedirs("results", exist_ok=True)
    with open("results/sft_sensitivity_results.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
    print("\nSweep completed! Saved results to results/sft_sensitivity_results.json")

if __name__ == "__main__":
    run_sft_sweep()
