import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

D = 192
K = 4
num_classes = 10
r = 8
noise_stds = [0.15, 0.25, 0.50, 0.80]

# --- Model Definitions ---
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
            self.adapters = nn.ModuleList([LoRAAdapter(dim, dim, r) for _ in range(K)])

    def forward(self, x, active_expert_idx=None, alpha=None, scale_factors=None):
        base_out = x @ self.W_base

        if not self.has_adapters:
            return torch.nn.functional.gelu(base_out)

        if active_expert_idx is not None:
            adapter_out = self.adapters[active_expert_idx](x)
            return torch.nn.functional.gelu(base_out + adapter_out)

        if alpha is not None:
            blend_out = torch.zeros_like(base_out)
            for k in range(K):
                coeff = alpha[:, k].unsqueeze(1)
                adapter_out = self.adapters[k](x)
                if scale_factors is not None:
                    adapter_out = adapter_out * scale_factors[k]
                blend_out += coeff * adapter_out
            return torch.nn.functional.gelu(base_out + blend_out)

        # Uniform fallback
        uniform_out = torch.zeros_like(base_out)
        for k in range(K):
            uniform_out += self.adapters[k](x) / K
        return torch.nn.functional.gelu(base_out + uniform_out)

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
                    
                    h = torch.nn.functional.gelu(h @ W_merged_dequant)
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
                    h = torch.nn.functional.gelu(base_out)
                    continue
                    
                if active_expert_idx is not None:
                    adapter = block.adapters[active_expert_idx]
                    A = adapter.A
                    B = adapter.B
                    
                    max_A = torch.max(torch.abs(A))
                    S_A = max_A / 127.0
                    S_A = torch.clamp(S_A, min=1e-8)
                    Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                    A_deq = Q_A * S_A
                    
                    max_B = torch.max(torch.abs(B))
                    S_B = max_B / 127.0
                    S_B = torch.clamp(S_B, min=1e-8)
                    Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                    B_deq = Q_B * S_B
                    
                    h = torch.nn.functional.gelu(base_out + h @ A_deq @ B_deq)
                elif alpha is not None:
                    blend_out = torch.zeros_like(base_out)
                    for k in range(K):
                        coeff = alpha[:, k].unsqueeze(1)
                        adapter = block.adapters[k]
                        A = adapter.A
                        B = adapter.B
                        
                        max_A = torch.max(torch.abs(A))
                        S_A = max_A / 127.0
                        S_A = torch.clamp(S_A, min=1e-8)
                        Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                        A_deq = Q_A * S_A
                        
                        max_B = torch.max(torch.abs(B))
                        S_B = max_B / 127.0
                        S_B = torch.clamp(S_B, min=1e-8)
                        Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                        B_deq = Q_B * S_B
                        
                        adapter_quant_out = h @ A_deq @ B_deq
                        if scale_alignment is not None:
                            adapter_quant_out = adapter_quant_out * scale_alignment[l_idx][k]
                        blend_out += coeff * adapter_quant_out
                    h = torch.nn.functional.gelu(base_out + blend_out)
                else:
                    # Uniform fallback quantized
                    uniform_out = torch.zeros_like(base_out)
                    for k in range(K):
                        adapter = block.adapters[k]
                        A = adapter.A
                        B = adapter.B
                        
                        max_A = torch.max(torch.abs(A))
                        S_A = max_A / 127.0
                        S_A = torch.clamp(S_A, min=1e-8)
                        Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                        A_deq = Q_A * S_A
                        
                        max_B = torch.max(torch.abs(B))
                        S_B = max_B / 127.0
                        S_B = torch.clamp(S_B, min=1e-8)
                        Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                        B_deq = Q_B * S_B
                        
                        uniform_out += (h @ A_deq @ B_deq) / K
                    h = torch.nn.functional.gelu(base_out + uniform_out)
            else:
                h = block(h, active_expert_idx=active_expert_idx, alpha=alpha, scale_factors=scale_alignment[l_idx] if scale_alignment else None)
                
        # Return logits from the specified task's classification head
        if isinstance(task_idx, torch.Tensor) and task_idx.ndim > 0:
            logits = torch.zeros(x.shape[0], num_classes).to(device)
            for k in range(K):
                mask = (task_idx == k)
                if mask.any():
                    logits[mask] = self.heads[k](h[mask])
            return logits
        else:
            return self.heads[task_idx](h)

# --- Helper Functions ---
def generate_prototypes(subspace_dim):
    prototypes = {}
    for k in range(K):
        # Determine subspace start and end to control overlap
        subspace_start = int(k * (D - subspace_dim) / (K - 1)) if K > 1 else 0
        subspace_end = subspace_start + subspace_dim
        
        random_matrix = torch.randn(subspace_dim, num_classes)
        q, _ = torch.linalg.qr(random_matrix)
        q = q * 3.5
        
        centers = torch.zeros(num_classes, D)
        centers[:, subspace_start:subspace_end] = q.t()
        prototypes[k] = centers
    return prototypes

def generate_dataset(prototypes, num_samples_per_task):
    data_x, data_y, data_task = [], [], []
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

def get_fp_zca_coefficients(h_b3, centroids, tau=0.05):
    similarities = []
    for k in range(K):
        mu = centroids[k]
        dot_product = torch.sum(h_b3 * mu, dim=-1)
        norm_h = torch.norm(h_b3, p=2, dim=-1)
        norm_mu = torch.norm(mu, p=2)
        sim = dot_product / (norm_h * norm_mu + 1e-8)
        similarities.append(sim)
    similarities = torch.stack(similarities, dim=1)
    return torch.softmax(similarities / tau, dim=1)

def get_quantized_zca_coefficients(h_b3, centroids, tau=0.05):
    max_h = torch.max(torch.abs(h_b3), dim=-1, keepdim=True)[0]
    S_h = max_h / 127.0
    S_h = torch.clamp(S_h, min=1e-8)
    Q_h = torch.round(torch.clamp(h_b3 / S_h, -127, 127))
    h_q = Q_h * S_h

    similarities = []
    for k in range(K):
        mu = centroids[k]
        max_mu = torch.max(torch.abs(mu))
        S_mu = max_mu / 127.0
        S_mu = torch.clamp(S_mu, min=1e-8)
        Q_mu = torch.round(torch.clamp(mu / S_mu, -127, 127))
        mu_q = Q_mu * S_mu

        dot_product = torch.sum(h_q * mu_q, dim=-1)
        norm_h = torch.norm(h_q, p=2, dim=-1)
        norm_mu = torch.norm(mu_q, p=2)
        sim = dot_product / (norm_h * norm_mu + 1e-8)
        similarities.append(sim)

    similarities = torch.stack(similarities, dim=1)
    return torch.softmax(similarities / tau, dim=1)

# --- Sweep Loop ---
subspace_dims = [48, 80, 112, 144, 176, 192]
results = []

for s_dim in subspace_dims:
    overlap_factor = (s_dim - 48) / 144.0
    print(f"\n===========================================")
    print(f"Sweeping Subspace Dimension: {s_dim} (Overlap Factor: {overlap_factor:.2f})")
    print(f"===========================================")
    
    # Generate prototypes once per step
    prototypes = generate_prototypes(s_dim)
    # Generate datasets
    train_x, train_y, train_task = generate_dataset(prototypes, 1000)
    calib_x, calib_y, calib_task = generate_dataset(prototypes, 64)
    test_x, test_y, test_task = generate_dataset(prototypes, 250)
    
    train_x, train_y, train_task = train_x.to(device), train_y.to(device), train_task.to(device)
    calib_x, calib_y, calib_task = calib_x.to(device), calib_y.to(device), calib_task.to(device)
    test_x, test_y, test_task = test_x.to(device), test_y.to(device), test_task.to(device)
    
    # Instantiate and train model experts
    model = SandboxViT(D).to(device)
    
    # Train experts
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
                
    model.eval()
    
    # Offline Calibration: Centroids & Scale Alignment Factors
    centroids_l3 = {}
    with torch.no_grad():
        for k in range(K):
            mask = (calib_task == k)
            task_cal_x = calib_x[mask]
            h = task_cal_x
            for block in model.blocks[:3]:
                h = block(h)
            centroids_l3[k] = h.mean(dim=0)
            
    scale_align = {l: [1.0]*K for l in range(1, 13)}
    with torch.no_grad():
        for k in range(K):
            mask = (calib_task == k)
            task_cal_x = calib_x[mask]
            
            h_base = task_cal_x
            for l_idx, block in enumerate(model.blocks, 1):
                if l_idx < 4:
                    h_base = block(h_base)
                else:
                    base_out = h_base @ block.W_base
                    adapter_out = block.adapters[k](h_base)
                    norm_adapter = torch.norm(adapter_out, p=2, dim=-1).mean().item()
                    
                    A = block.adapters[k].A
                    B = block.adapters[k].B
                    
                    max_A = torch.max(torch.abs(A))
                    S_A = max_A / 127.0
                    S_A = torch.clamp(S_A, min=1e-8)
                    Q_A = torch.round(torch.clamp(A / S_A, -127, 127))
                    A_deq = Q_A * S_A
                    
                    max_B = torch.max(torch.abs(B))
                    S_B = max_B / 127.0
                    S_B = torch.clamp(S_B, min=1e-8)
                    Q_B = torch.round(torch.clamp(B / S_B, -127, 127))
                    B_deq = Q_B * S_B
                    
                    adapter_quant_out = h_base @ A_deq @ B_deq
                    norm_adapter_quant = torch.norm(adapter_quant_out, p=2, dim=-1).mean().item()
                    
                    beta = norm_adapter / max(norm_adapter_quant, 1e-8)
                    scale_align[l_idx][k] = beta
                    h_base = base_out + adapter_out

    # --- Evaluation ---
    B_size = 256
    
    # 1. Expert Ceiling
    correct_ceil, total_ceil = 0, 0
    with torch.no_grad():
        for k in range(K):
            mask = (test_task == k)
            task_test_x = test_x[mask]
            task_test_y = test_y[mask]
            
            num_batches = int(np.ceil(task_test_x.shape[0] / B_size))
            for b in range(num_batches):
                start = b * B_size
                end = min((b + 1) * B_size, task_test_x.shape[0])
                bx, by = task_test_x[start:end], task_test_y[start:end]
                logits = model(bx, task_idx=k, active_expert_idx=k)
                preds = logits.argmax(dim=1)
                correct_ceil += (preds == by).sum().item()
                total_ceil += bx.shape[0]
    acc_ceil = (correct_ceil / total_ceil) * 100.0
    
    # 2. PMQ 4-bit Static
    correct_pmq, total_pmq = 0, 0
    with torch.no_grad():
        for k in range(K):
            mask = (test_task == k)
            task_test_x = test_x[mask]
            task_test_y = test_y[mask]
            
            num_batches = int(np.ceil(task_test_x.shape[0] / B_size))
            for b in range(num_batches):
                start = b * B_size
                end = min((b + 1) * B_size, task_test_x.shape[0])
                bx, by = task_test_x[start:end], task_test_y[start:end]
                logits = model(bx, task_idx=k, fake_quant_base_bit=4, use_weight_merge=True)
                preds = logits.argmax(dim=1)
                correct_pmq += (preds == by).sum().item()
                total_pmq += bx.shape[0]
    acc_pmq = (correct_pmq / total_pmq) * 100.0

    # 3. SPS-ZCA (Ours, FP16)
    correct_sps, total_sps = 0, 0
    with torch.no_grad():
        for k in range(K):
            mask = (test_task == k)
            task_test_x = test_x[mask]
            task_test_y = test_y[mask]
            
            num_batches = int(np.ceil(task_test_x.shape[0] / B_size))
            for b in range(num_batches):
                start = b * B_size
                end = min((b + 1) * B_size, task_test_x.shape[0])
                bx, by = task_test_x[start:end], task_test_y[start:end]
                
                h = bx
                for block in model.blocks[:3]:
                    h = block(h)
                alpha = get_fp_zca_coefficients(h, centroids_l3, tau=0.001)
                logits = model(bx, task_idx=k, alpha=alpha)
                preds = logits.argmax(dim=1)
                correct_sps += (preds == by).sum().item()
                total_sps += bx.shape[0]
    acc_sps = (correct_sps / total_sps) * 100.0

    # 4. SA-QAB (Ours, Quantized)
    correct_saqab, total_saqab = 0, 0
    with torch.no_grad():
        for k in range(K):
            mask = (test_task == k)
            task_test_x = test_x[mask]
            task_test_y = test_y[mask]
            
            num_batches = int(np.ceil(task_test_x.shape[0] / B_size))
            for b in range(num_batches):
                start = b * B_size
                end = min((b + 1) * B_size, task_test_x.shape[0])
                bx, by = task_test_x[start:end], task_test_y[start:end]
                
                h = bx
                for block in model.blocks[:3]:
                    W = block.W_base
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -7, 7))
                    W_dequant = Q * S
                    h = h @ W_dequant
                    
                alpha = get_quantized_zca_coefficients(h, centroids_l3, tau=0.001)
                logits = model(bx, task_idx=k, alpha=alpha, scale_alignment=scale_align, fake_quant_base_bit=4)
                preds = logits.argmax(dim=1)
                correct_saqab += (preds == by).sum().item()
                total_saqab += bx.shape[0]
    acc_saqab = (correct_saqab / total_saqab) * 100.0

    # 5. Q-ZCA Routing Accuracy
    correct_route, total_route = 0, 0
    with torch.no_grad():
        for k in range(K):
            mask = (test_task == k)
            task_test_x = test_x[mask]
            
            num_batches = int(np.ceil(task_test_x.shape[0] / B_size))
            for b in range(num_batches):
                start = b * B_size
                end = min((b + 1) * B_size, task_test_x.shape[0])
                bx = task_test_x[start:end]
                
                h = bx
                for block in model.blocks[:3]:
                    W = block.W_base
                    max_val = torch.max(torch.abs(W), dim=1, keepdim=True)[0]
                    S = max_val / 7.0
                    S = torch.clamp(S, min=1e-8)
                    Q = torch.round(torch.clamp(W / S, -7, 7))
                    W_dequant = Q * S
                    h = h @ W_dequant
                    
                alpha = get_quantized_zca_coefficients(h, centroids_l3, tau=0.001)
                preds = alpha.argmax(dim=1)
                correct_route += (preds == k).sum().item()
                total_route += bx.shape[0]
    acc_route = (correct_route / total_route) * 100.0

    print(f"Results for overlap factor {overlap_factor:.2f}:")
    print(f"  Expert Ceiling (FP16): {acc_ceil:.2f}%")
    print(f"  PMQ (4-bit Static):   {acc_pmq:.2f}%")
    print(f"  SPS-ZCA (FP16 Ours):  {acc_sps:.2f}%")
    print(f"  SA-QAB (Ours, Quant): {acc_saqab:.2f}%")
    print(f"  Q-ZCA Routing Acc:    {acc_route:.2f}%")
    
    results.append({
        'overlap_factor': overlap_factor,
        'acc_ceil': acc_ceil,
        'acc_pmq': acc_pmq,
        'acc_sps': acc_sps,
        'acc_saqab': acc_saqab,
        'acc_route': acc_route
    })

# --- Plot and Save ---
overlap_factors = [r['overlap_factor'] for r in results]
acc_ceils = [r['acc_ceil'] for r in results]
acc_pmqs = [r['acc_pmq'] for r in results]
acc_spss = [r['acc_sps'] for r in results]
acc_saqabs = [r['acc_saqab'] for r in results]
acc_routes = [r['acc_route'] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(overlap_factors, acc_ceils, 'k--', marker='o', label='Expert Ceiling (FP16)')
plt.plot(overlap_factors, acc_spss, 'g-', marker='s', label='SPS-ZCA (Ours, FP16)')
plt.plot(overlap_factors, acc_saqabs, 'b-', marker='D', label='SA-QAB (Ours, Quantized)')
plt.plot(overlap_factors, acc_pmqs, 'r-', marker='^', label='PMQ (Static, 4-bit)')
plt.plot(overlap_factors, acc_routes, 'm:', marker='x', label='Q-ZCA Routing Accuracy')

plt.title('Performance and Routing Accuracy vs Task Overlap Factor')
plt.xlabel('Task Overlap Factor ($\Omega$)')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/task_overlap_sweep.png', dpi=300)
print("\nSaved plot to 'results/task_overlap_sweep.png'")

# Generate Markdown table
print("\n### Task Overlap Factor Sweep Results:")
print("| Overlap Factor $\\Omega$ | Expert Ceiling (FP16 %) | SPS-ZCA (FP16 Ours %) | SA-QAB (Quantized Ours %) | PMQ (Static 4-bit %) | Q-ZCA Routing Accuracy (%) |")
print("| :---: | :---: | :---: | :---: | :---: | :---: |")
for r in results:
    print(f"| {r['overlap_factor']:.2f} | {r['acc_ceil']:.2f}% | {r['acc_sps']:.2f}% | {r['acc_saqab']:.2f}% | {r['acc_pmq']:.2f}% | {r['acc_route']:.2f}% |")
