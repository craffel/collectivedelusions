import torch
import numpy as np
import json
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

thresholds = [0.45, 0.50, 0.55, 0.58, 0.60, 0.65]
results = {}

for thresh in thresholds:
    print(f"\n==========================================")
    print(f"Running KT-Fisher with novelty threshold = {thresh}")
    print(f"==========================================")
    res = evaluate_method("kt_fisher", stream_batches, task_labels, expert_state_dicts, threshold_N=thresh)
    results[str(thresh)] = res

with open("ablation_threshold_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nAblation Results Summary (Novelty Threshold):")
print("Thresh | MNIST (%) | KMNIST (%) | F-MNIST (%) | NDR (%) | FPR (%) | Overall (%)")
print("-" * 75)
for thresh in thresholds:
    res = results[str(thresh)]
    print(f"{thresh:<6} | {res['mnist']:<9.2f} | {res['kmnist']:<10.2f} | {res['fashionmnist']:<11.2f} | {res['ndr']:<7.2f} | {res['fpr']:<7.2f} | {res['overall']:<11.2f}")
