import torch
import numpy as np
import json
import time
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

stream_batches, task_labels = construct_test_stream(batch_size=64, num_batches_per_task=30)

intervals = [1, 5, 10, 15, 30]
results = {}

for K in intervals:
    print(f"\n==========================================")
    print(f"Running KT-Fisher with periodic interval K = {K}")
    print(f"==========================================")
    # Measure time specifically for this run
    t0 = time.time()
    res = evaluate_method(
        "kt_fisher", 
        stream_batches, 
        task_labels, 
        expert_state_dicts, 
        precondition_interval=K
    )
    res["total_run_time_sec"] = time.time() - t0
    results[str(K)] = res

with open("ablation_periodic_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nPeriodic Preconditioning Ablation Summary:")
print("Interval K | MNIST (%) | KMNIST (%) | F-MNIST (%) | Overall (%) | Latency (ms)")
print("-" * 75)
for K in intervals:
    res = results[str(K)]
    print(f"{K:<10} | {res['mnist']:<9.2f} | {res['kmnist']:<10.2f} | {res['fashionmnist']:<11.2f} | {res['overall']:<11.2f} | {res['time']:<12.2f}")
