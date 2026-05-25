import torch
import torch.nn as nn
from diagnose import base_backbone, base_backbone_params, expert_backbones, expert_heads, task_vectors, fisher_priors
from tta import get_dataset, run_evaluation, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

mnist_test = get_dataset('mnist', train=False)
fashion_test = get_dataset('fashion', train=False)
kmnist_test = get_dataset('kmnist', train=False)

mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
fashion_loader = DataLoader(fashion_test, batch_size=32, shuffle=False)
kmnist_loader = DataLoader(kmnist_test, batch_size=32, shuffle=False)

mnist_batches = []
for i, batch in enumerate(mnist_loader):
    if i >= 50: break
    mnist_batches.append((batch, 0))
    
fashion_batches = []
for i, batch in enumerate(fashion_loader):
    if i >= 50: break
    fashion_batches.append((batch, 1))
    
kmnist_batches = []
for i, batch in enumerate(kmnist_loader):
    if i >= 50: break
    kmnist_batches.append((batch, 2))

sequential_batches = mnist_batches + fashion_batches + kmnist_batches
alternating_batches = []
for i in range(50):
    alternating_batches.append(mnist_batches[i])
    alternating_batches.append(fashion_batches[i])
    alternating_batches.append(kmnist_batches[i])

print("\n--- Sweeping beta and gamma on Sequential and Alternating streams ---")
for beta in [0.0, 0.01, 0.05, 0.1]:
    for gamma in [1.0, 10.0, 100.0]:
        seq_acc, _ = run_evaluation(
            'Sequential', sequential_batches, 'tf_ewc_dts',
            base_backbone, base_backbone_params, expert_backbones, expert_heads,
            task_vectors, fisher_priors, 1e-4, 0.5, gamma, 0.9, beta, device
        )
        alt_acc, _ = run_evaluation(
            'Alternating', alternating_batches, 'tf_ewc_dts',
            base_backbone, base_backbone_params, expert_backbones, expert_heads,
            task_vectors, fisher_priors, 1e-4, 0.5, gamma, 0.9, beta, device
        )
        print(f"beta={beta:<5} | gamma={gamma:<5} | Seq Acc: {seq_acc:.2f}% | Alt Acc: {alt_acc:.2f}%")
