import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Setup & Reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

D = 192
K = 4
subspace_dim = 48
num_classes = 10

# Generate orthogonal class prototypes for each task
prototypes = {}
for k in range(K):
    subspace_start = k * subspace_dim
    subspace_end = (k + 1) * subspace_dim
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

class LoRAAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.1)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
    def forward(self, x):
        return x @ self.A @ self.B

class SandboxBlock(nn.Module):
    def __init__(self, dim, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        base_w = torch.eye(dim) + torch.randn(dim, dim) * 0.01
        self.W_base = nn.Parameter(base_w)
        self.has_adapters = (layer_idx >= 4)
        if self.has_adapters:
            self.adapters = nn.ModuleList([LoRAAdapter(dim, dim, 8) for _ in range(K)])
            
    def forward(self, x, active_expert_idx=None, alpha=None, scale_factors=None):
        base_out = x @ self.W_base
        if not self.has_adapters:
            return F.gelu(base_out)
        if active_expert_idx is not None:
            adapter_out = self.adapters[active_expert_idx](x)
            return F.gelu(base_out + adapter_out)
        if alpha is not None:
            blend_out = torch.zeros_like(base_out)
            for k in range(K):
                coeff = alpha[:, k].unsqueeze(1)
                adapter_out = self.adapters[k](x)
                if scale_factors is not None:
                    adapter_out = adapter_out * scale_factors[k]
                blend_out += coeff * adapter_out
            return F.gelu(base_out + blend_out)
        uniform_out = torch.zeros_like(base_out)
        for k in range(K):
            uniform_out += self.adapters[k](x) / K
        return F.gelu(base_out + uniform_out)

class SandboxViT(nn.Module):
    def __init__(self, dim, num_layers=12):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([SandboxBlock(dim, l) for l in range(1, num_layers + 1)])
        self.heads = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(K)])
        
    def forward(self, x, task_idx, active_expert_idx=None, alpha=None, scale_alignment=None, fake_quant_base_bit=None, use_weight_merge=False):
        h = x
        for l_idx, block in enumerate(self.blocks, 1):
            if fake_quant_base_bit is not None:
                if use_weight_merge and block.has_adapters:
                    W_merged = block.W_base.clone()
                    for k in range(K):
                        if alpha is not None:
                            coeff = alpha[0, k].item()
                        else:
                            coeff = 1.0 / K
                        adapter = block.adapters[k]
                        W_merged = W_merged + coeff * (adapter.A @ adapter.B)
                    
                    if fake_quant_base_bit == 4:
                        max_val = torch.max(torch.abs(W_merged), dim=1, keepdim=True)[0]
                        S = max_val / 7.0
                        S = torch.clamp(S, min=1e-8)
                        Q = torch.round(torch.clamp(W_merged / S, -7, 7))
                        W_merged_dequant = Q * S
                    elif fake_quant_base_bit == 8:
                        max_val = torch.max(torch.abs(W_merged), dim=1, keepdim=True)[0]
                        S = max_val / 127.0
                        S = torch.clamp(S, min=1e-8)
                        Q = torch.round(torch.clamp(W_merged / S, -127, 127))
                        W_merged_dequant = Q * S
                    h = F.gelu(h @ W_merged_dequant)
                    continue

                W = block.W_base
                if fake_quant_base_bit == 4:
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -7, 7))
                    W_dequant = Q * S
                elif fake_quant_base_bit == 8:
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 127.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -127, 127))
                    W_dequant = Q * S
                
                base_out = h @ W_dequant
                if not block.has_adapters:
                    h = F.gelu(base_out)
                    continue
                    
                if active_expert_idx is not None:
                    adapter = block.adapters[active_expert_idx]
                    A, B = adapter.A, adapter.B
                    max_A = torch.max(torch.abs(A))
                    S_A = max_A / 127.0
                    S_A = torch.clamp(S_A, min=1e-8)
                    Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                    A_dequant = Q_A * S_A
                    
                    max_B = torch.max(torch.abs(B))
                    S_B = max_B / 127.0
                    S_B = torch.clamp(S_B, min=1e-8)
                    Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                    B_dequant = Q_B * S_B
                    
                    adapter_out = h @ A_dequant @ B_dequant
                    h = F.gelu(base_out + adapter_out)
                else:
                    blend_out = torch.zeros_like(base_out)
                    for k in range(K):
                        if alpha is not None:
                            coeff = alpha[:, k].unsqueeze(1)
                        else:
                            coeff = 1.0 / K
                        adapter = block.adapters[k]
                        A, B = adapter.A, adapter.B
                        max_A = torch.max(torch.abs(A))
                        S_A = max_A / 127.0
                        S_A = torch.clamp(S_A, min=1e-8)
                        Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                        A_dequant = Q_A * S_A
                        
                        max_B = torch.max(torch.abs(B))
                        S_B = max_B / 127.0
                        S_B = torch.clamp(S_B, min=1e-8)
                        Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                        B_dequant = Q_B * S_B
                        
                        adapter_out = h @ A_dequant @ B_dequant
                        if scale_alignment is not None:
                            adapter_out = adapter_out * scale_alignment[l_idx][k]
                        blend_out += coeff * adapter_out
                    h = F.gelu(base_out + blend_out)
            else:
                h = block(h, active_expert_idx=active_expert_idx, alpha=alpha, scale_factors=scale_alignment[l_idx] if scale_alignment else None)
        
        if isinstance(task_idx, torch.Tensor) and task_idx.ndim > 0:
            logits = torch.zeros(x.shape[0], num_classes)
            for k in range(K):
                mask = (task_idx == k)
                if mask.any():
                    logits[mask] = self.heads[k](h[mask])
            return logits
        else:
            return self.heads[task_idx](h)

model = SandboxViT(D).to(device)

print("Training experts with GELU non-linearities...")
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

# Evaluate expert ceilings
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
                    h = F.gelu(h @ W_merged_dequant)
                else:
                    W = block.W_base
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -7, 7))
                    W_dequant = Q * S
                    h = F.gelu(h @ W_dequant)
                    
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
print(f"True PMQ 4-bit Joint Accuracy under Non-Linear Sandbox: {pmq_acc:.2f}%")
