import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim

# Set seed
torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
num_classes = 10
subspace_dim = 96  # Make each task occupy 96 dimensions (overlap!)

# Generate overlapping class prototypes for each task
prototypes = {}
for k in range(K):
    # Overlapping subspaces
    subspace_start = k * 32  # Task 0: 0-96, Task 1: 32-128, Task 2: 64-160, Task 3: 96-192
    subspace_end = subspace_start + subspace_dim
    
    random_matrix = torch.randn(subspace_dim, num_classes)
    q, _ = torch.linalg.qr(random_matrix)
    q = q * 3.5
    
    centers = torch.zeros(num_classes, D)
    centers[:, subspace_start:subspace_end] = q.t()
    prototypes[k] = centers

noise_stds = [0.15, 0.25, 0.50, 1.80]

def generate_dataset(num_samples_per_task, noise_stds):
    data_x = []
    data_y = []
    data_task = []
    for k in range(K):
        centers = prototypes[k]
        std = noise_stds[k]
        samples_per_class = num_samples_per_task // num_classes
        for c in range(num_classes):
            center = centers[c]
            noise = torch.randn(samples_per_class, D) * std
            samples = center.unsqueeze(0) + noise
            data_x.append(samples)
            data_y.append(torch.full((samples_per_class,), c, dtype=torch.long))
            data_task.append(torch.full((samples_per_class,), k, dtype=torch.long))
            
    return (torch.cat(data_x, dim=0), 
            torch.cat(data_y, dim=0), 
            torch.cat(data_task, dim=0))

train_x, train_y, train_task = generate_dataset(1000, noise_stds)
calib_x, calib_y, calib_task = generate_dataset(64, noise_stds)
test_x, test_y, test_task = generate_dataset(250, noise_stds)

# Copy other model code from run_experiments
from run_experiments import LoRAAdapter, SandboxBlock, SandboxViT, device

model = SandboxViT(D).to(device)

# Train the experts on this overlapping dataset
print("Training experts on overlapping dataset...")
for k in range(K):
    mask = (train_task == k)
    task_x = train_x[mask]
    task_y = train_y[mask]
    
    block_adapters_params = []
    for block in model.blocks:
        if block.has_adapters:
            block_adapters_params.extend(list(block.adapters[k].parameters()))
    head_params = list(model.heads[k].parameters())
    
    optimizer = optim.AdamW(block_adapters_params + head_params, lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 64
    num_epochs = 8
    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(task_x.size(0))
        for i in range(0, task_x.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            bx, by = task_x[indices], task_y[indices]
            optimizer.zero_grad()
            outputs = model(bx, task_idx=k, active_expert_idx=k)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()

# Evaluate unmerged expert accuracies
model.eval()
with torch.no_grad():
    for k in range(K):
        test_mask = (test_task == k)
        task_test_x = test_x[test_mask]
        task_test_y = test_y[test_mask]
        outputs = model(task_test_x, task_idx=k, active_expert_idx=k)
        preds = outputs.argmax(dim=1)
        acc = (preds == task_test_y).float().mean().item() * 100.0
        print(f"Task {k} Expert Accuracy: {acc:.2f}%")

# Compute centroids
centroids_layer3 = {}
with torch.no_grad():
    for k in range(K):
        mask = (calib_task == k)
        task_cal_x = calib_x[mask]
        h = task_cal_x
        for block in model.blocks[:3]:
            h = block(h)
        centroids_layer3[k] = h.mean(dim=0)

# True PMQ weight-space merging
def true_pmq_eval():
    model.eval()
    correct = 0
    total = 0
    B_size = 256
    with torch.no_grad():
        num_batches = int(np.ceil(test_x.shape[0] / B_size))
        for b in range(num_batches):
            start = b * B_size
            end = min((b + 1) * B_size, test_x.shape[0])
            bx = test_x[start:end]
            by = test_y[start:end]
            btask = test_task[start:end]
            
            # Simulate forward pass with true weight-space merging
            h = bx
            for l_idx, block in enumerate(model.blocks, 1):
                if block.has_adapters:
                    W_merged = block.W_base.clone()
                    for k in range(K):
                        W_merged = W_merged + (1.0 / K) * (block.adapters[k].A @ block.adapters[k].B)
                    
                    # 4-bit Quantization
                    max_val = torch.max(torch.abs(W_merged), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W_merged / S, -7, 7))
                    W_merged_dequant = Q * S
                    h = h @ W_merged_dequant
                else:
                    W = block.W_base
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -7, 7))
                    W_dequant = Q * S
                    h = h @ W_dequant
                    
            logits = torch.zeros(bx.shape[0], num_classes)
            for k in range(K):
                mask = (btask == k)
                if mask.any():
                    logits[mask] = model.heads[k](h[mask])
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.shape[0]
    return correct / total * 100.0

pmq_acc = true_pmq_eval()
print(f"True PMQ 4-bit Joint Accuracy under Overlapping Subspaces: {pmq_acc:.2f}%")
