import time
import torch
import torch.nn as nn
from torch.func import functional_call
import copy
from evaluate_tta import (
    load_experts, get_test_datasets, construct_test_streams,
    get_merged_params, ExpertModel, device,
    translate_augmentation, run_evaluation
)
from torchvision.models import resnet18, ResNet18_Weights

# Ensure CPU/GPU device setup is correct
print(f"Running on device: {device}")

base_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
base_backbone.fc = nn.Identity()
base_backbone.to(device).eval()

if __name__ == "__main__":
    print("Loading datasets...")
    mnist_test, fashion_test, kmnist_test = get_test_datasets()
    print("Constructing full streams...")
    seq_stream, _ = construct_test_streams(mnist_test, fashion_test, kmnist_test) # 150 batches
    
    print("Loading experts...")
    experts = load_experts()
    
    mc_values = [5, 3, 1]
    results = {}
    
    print("\nStarting evaluation of MC-VTI with different MC passes (M):")
    for M in mc_values:
        print(f"Evaluating M = {M}...")
        start_time = time.perf_counter()
        
        # Use optimal hyperparameters from sweep
        acc, _, _ = run_evaluation(
            "mc_vti", seq_stream, experts, base_backbone,
            lr_lambda=0.5, lr_head=1e-4, gamma_reg=100.0, num_mc_passes=M
        )
        
        elapsed = time.perf_counter() - start_time
        avg_latency_ms = (elapsed / len(seq_stream)) * 1000
        results[M] = {
            "accuracy": acc,
            "latency": avg_latency_ms
        }
        print(f"M = {M}: Accuracy = {acc:.2f}%, Latency = {avg_latency_ms:.2f} ms/batch")
        
    print("\n--- MC Passes Ablation and Latency Trade-off ---")
    print("| Number of Passes (M) | Sequential Accuracy (%) | Latency per Batch (ms) | Speedup vs. M=5 |")
    print("|---|---|---|---|")
    lat5 = results[5]["latency"]
    for M in mc_values:
        acc = results[M]["accuracy"]
        lat = results[M]["latency"]
        speedup = lat5 / lat
        print(f"| {M} | {acc:.2f}% | {lat:.2f} | {speedup:.2f}x |")
