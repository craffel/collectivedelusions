import torch
import numpy as np
import json
import time
import os
from run_ttmm import (
    device,
    get_modified_resnet18,
    construct_test_stream,
    evaluate_method
)

# Load experts
expert_mnist = torch.load("expert_mnist.pth", map_location=device)
expert_kmnist = torch.load("expert_kmnist.pth", map_location=device)
expert_fashionmnist = torch.load("expert_fashionmnist.pth", map_location=device)
expert_state_dicts = [expert_mnist, expert_kmnist, expert_fashionmnist]

batch_sizes = [16, 32, 64, 128]
results = {}

for B in batch_sizes:
    num_batches = 1920 // B
    print(f"\n==========================================")
    print(f"Running KT-Fisher with Batch Size B = {B} (num_batches = {num_batches})")
    print(f"==========================================")
    
    # Construct stream for this specific batch size
    stream_batches, task_labels = construct_test_stream(batch_size=B, num_batches_per_task=num_batches)
    
    t0 = time.time()
    res = evaluate_method(
        "kt_fisher", 
        stream_batches, 
        task_labels, 
        expert_state_dicts
    )
    res["total_run_time_sec"] = time.time() - t0
    results[str(B)] = res

with open("ablation_batch_size_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nBatch Size Ablation Summary:")
print("Batch Size | MNIST (%) | KMNIST (%) | F-MNIST (%) | Overall (%) | Latency (ms)")
print("-" * 75)
for B in batch_sizes:
    res = results[str(B)]
    print(f"{B:<10} | {res['mnist']:<9.2f} | {res['kmnist']:<10.2f} | {res['fashionmnist']:<11.2f} | {res['overall']:<11.2f} | {res['time']:<12.2f}")
