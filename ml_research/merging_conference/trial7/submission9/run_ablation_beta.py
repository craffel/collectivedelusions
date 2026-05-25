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

betas = [0.0, 0.25, 0.5, 0.75, 1.0]
results = {}

for beta in betas:
    print(f"\n==========================================")
    print(f"Running KT-Fisher with damping beta = {beta}")
    print(f"==========================================")
    res = evaluate_method("kt_fisher", stream_batches, task_labels, expert_state_dicts, damping_factor=beta)
    results[str(beta)] = res

with open("ablation_beta_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nAblation Results Summary:")
print("Beta | MNIST (%) | KMNIST (%) | F-MNIST (%) | Overall (%)")
print("-" * 55)
for beta in betas:
    res = results[str(beta)]
    print(f"{beta:<4} | {res['mnist']:<9.2f} | {res['kmnist']:<10.2f} | {res['fashionmnist']:<11.2f} | {res['overall']:<11.2f}")
