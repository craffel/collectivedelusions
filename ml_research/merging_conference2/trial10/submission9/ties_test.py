import os
import json
import copy
import torch
import torch.nn as nn
from main import (
    MultiTaskResNet, 
    get_datasets, 
    quantize_weights, 
    run_task_specific_de_bn, 
    run_naive_mixed_calibration, 
    run_centroid_aligned_unified_calibration, 
    evaluate_model, 
    ties_merge_models, 
    set_seed
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for TIES-Merging evaluation")
    set_seed(42)

    # 1. Load Datasets
    train_loaders, test_loaders = get_datasets(batch_size=256, dry_run=False)

    # 2. Re-create model structure
    heads = {
        "MNIST": nn.Linear(512, 10),
        "FMNIST": nn.Linear(512, 10),
        "CIFAR10": nn.Linear(512, 10)
    }
    
    # 3. Load pre-trained models from checkpoints
    checkpoint_dir = "./checkpoints"
    print("Loading pre-trained progenitor and experts...")
    
    progenitor = MultiTaskResNet(heads)
    progenitor.load_state_dict(torch.load(f"{checkpoint_dir}/progenitor.pt", map_location=device))
    
    experts = {}
    for task_name in heads.keys():
        expert_model = copy.deepcopy(progenitor)
        expert_model.load_state_dict(torch.load(f"{checkpoint_dir}/{task_name}_expert.pt", map_location=device))
        experts[task_name] = expert_model

    # 4. Merge experts using TIES-Merging
    merged_model = ties_merge_models(progenitor, experts, lam=0.4, keep_fraction=0.2)

    # 5. Run Calibration Algorithms
    print("\n--- Running Calibration Algorithms on TIES-Merged Model ---")
    
    # A. Task-Specific DE-BN (Oracle/Routing stats)
    task_specific_stats = {}
    for task_name in train_loaders.keys():
        captured = run_task_specific_de_bn(merged_model, task_name, train_loaders[task_name], N=64, device=device)
        task_specific_stats[task_name] = captured
        
    # B. Naive Mixed Calibration (Mixed task batch)
    naive_cal_model = run_naive_mixed_calibration(merged_model, train_loaders, N=64, device=device)
    
    # C. Proposed Centroid-Aligned Unified Calibration (CA-UC)
    ca_uc_model = run_centroid_aligned_unified_calibration(merged_model, train_loaders, N=64, device=device)

    # 6. Evaluation on diverse regimes
    print("\n--- Starting Comprehensive Evaluation for TIES-Merging ---")
    
    regimes = [
        {"name": "FP32", "bits": None, "per_channel": True, "noise": 0.0},
        {"name": "PC-INT8", "bits": 8, "per_channel": True, "noise": 0.0},
        {"name": "PT-INT8", "bits": 8, "per_channel": False, "noise": 0.0},
        {"name": "PC-INT4", "bits": 4, "per_channel": True, "noise": 0.0},
        {"name": "PT-INT4", "bits": 4, "per_channel": False, "noise": 0.0},
        {"name": "Noisy FP32", "bits": None, "per_channel": True, "noise": 0.1}
    ]
    
    evaluation_results = []
    
    for regime in regimes:
        print(f"\nEvaluating Regime: {regime['name']}")
        
        # Apply quantization to weights if specified
        base_merged = quantize_weights(merged_model, bits=regime["bits"], per_channel=regime["per_channel"])
        naive_model = quantize_weights(naive_cal_model, bits=regime["bits"], per_channel=regime["per_channel"])
        ca_model = quantize_weights(ca_uc_model, bits=regime["bits"], per_channel=regime["per_channel"])
        
        # Setup 1: No Calibration
        res_none = evaluate_model(base_merged, test_loaders, calibration_type="None", noise_std=regime["noise"], device=device)
        
        # Setup 2: Task-Specific DE-BN (Oracle)
        res_debn = evaluate_model(base_merged, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats, noise_std=regime["noise"], device=device)
        
        # Setup 3: Naive Mixed Calibration
        res_naive = evaluate_model(naive_model, test_loaders, calibration_type="None", noise_std=regime["noise"], device=device)
        
        # Setup 4: Proposed Centroid-Aligned Unified Calibration
        res_cauc = evaluate_model(ca_model, test_loaders, calibration_type="None", noise_std=regime["noise"], device=device)
        
        evaluation_results.append({
            "regime": regime["name"],
            "methods": {
                "Uncalibrated": res_none,
                "DE-BN (Oracle, routed)": res_debn,
                "Naive Mixed Cal": res_naive,
                "Proposed CA-UC (Task-Agnostic)": res_cauc
            }
        })

    # Print out results as markdown-style table
    print("\n================== TIES-MERGING EVALUATION RESULTS SUMMARY ==================")
    for res in evaluation_results:
        print(f"\nRegime: {res['regime']}")
        print(f"| Method | MNIST | FMNIST | CIFAR-10 | Average |")
        print(f"|---|---|---|---|---|")
        for method_name, task_res in res["methods"].items():
            print(f"| {method_name} | {task_res['MNIST']:.2f}% | {task_res['FMNIST']:.2f}% | {task_res['CIFAR10']:.2f}% | {task_res['Average']:.2f}% |")

    # Save results to JSON file
    output_data = {
        "evaluation_results": evaluation_results
    }
    with open("ties_results.json", "w") as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
