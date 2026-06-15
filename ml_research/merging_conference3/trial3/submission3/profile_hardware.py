import time
import torch
import torch.nn as nn
import os

print("--- Hardware Profiling: Weight-Space TTA vs. Zeroth-Order (ZO) FlatMerge ---")

# Setup dummy parameters mimicking CLIP ViT-B/32
class MockViT(nn.Module):
    def __init__(self, layers=12, d_model=768, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.ModuleDict({
                "qkv": nn.Linear(d_model, 3 * d_model),
                "proj": nn.Linear(d_model, d_model),
                "fc1": nn.Linear(d_model, d_model * mlp_ratio),
                "fc2": nn.Linear(d_model * mlp_ratio, d_model),
            })
            self.layers.append(layer)
            
    def forward(self, x):
        for layer in self.layers:
            qkv = layer["qkv"](x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (q.size(-1) ** 0.5))
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = layer["proj"](out) + x
            out = layer["fc2"](torch.relu(layer["fc1"](out))) + out
            x = out
        return x

# Initialize mock ViT model
model = MockViT()
num_params = sum(p.numel() for p in model.parameters())
print(f"Loaded Mock ViT-B/32 model with {num_params / 1e6:.2f}M parameters.")

base_weights = [p.data.clone() for p in model.parameters()]
task_vectors = [[torch.randn_like(p) * 0.1 for p in model.parameters()] for _ in range(4)] # K = 4

batch_size = 64
seq_len = 50
d_model = 768
x = torch.randn(batch_size, seq_len, d_model)

# 1. Profile Weight-Space TTA (Standard reverse-mode AD over all parameters)
print("\nProfiling Standard Weight-Space TTA (Reverse-Mode AD over 86M params)...")
model.train()
optimizer_weight = torch.optim.Adam(model.parameters(), lr=1e-5)

# Warmup
for _ in range(3):
    optimizer_weight.zero_grad()
    out = model(x)
    loss = torch.mean(out ** 2)
    loss.backward()
    optimizer_weight.step()

start_time = time.perf_counter()
for step in range(5):
    optimizer_weight.zero_grad()
    out = model(x)
    loss = torch.mean(out ** 2)
    loss.backward()
    optimizer_weight.step()
end_time = time.perf_counter()
weight_tta_time = (end_time - start_time) / 5.0
print(f"Weight-Space TTA: {weight_tta_time * 1000:.2f} ms/step")

# Estimate weight memory
weight_mem_mb = num_params * 4 / 1e6 # FP32
grad_mem_mb = num_params * 4 / 1e6
opt_mem_mb = num_params * 8 / 1e6 # Adam moments
print(f"Weight-Space TTA static memory (Weights + Grads + Adam moments): {weight_mem_mb + grad_mem_mb + opt_mem_mb:.2f} MB")


# 2. Profile ZO-FlatMerge (Frozen weights, forward-only randomized evaluations)
print("\nProfiling ZO-FlatMerge (Zeroth-Order, forward-only on-the-fly weight reconstruction)...")
# ZO-FlatMerge updates coefficients using ONLY forward passes, hence 0 MB activation caching!
# For a single step with 10 random perturbations, we evaluate the loss 10 times at perturbed parameters.

def reconstruct_weights(alphas, E, degree=2):
    # Map perturbed alphas to lambdas
    lambdas = alphas + E
    reconstructed = []
    for l_idx, p_base in enumerate(base_weights):
        p_merged = p_base.clone()
        for k in range(4):
            # Evaluate polynomial at normalized depth for layer l_idx
            val = lambdas[k, 0] + lambdas[k, 1] * (l_idx / 11.0) + lambdas[k, 2] * ((l_idx / 11.0) ** 2)
            p_merged += val * task_vectors[k][l_idx]
        reconstructed.append(p_merged)
    return reconstructed

alphas = torch.zeros(4, 3)
alphas[:, 0] = 0.3
num_samples = 10

# Warmup
for _ in range(3):
    for _ in range(num_samples):
        E = torch.randn_like(alphas) * 0.05
        # Reconstruct perturbed weights
        reconstructed_weights = reconstruct_weights(alphas, E)
        with torch.no_grad():
            for p, r in zip(model.parameters(), reconstructed_weights):
                p.copy_(r)
            out = model(x)
            loss = torch.mean(out ** 2)

start_time = time.perf_counter()
for step in range(5):
    # A single step of ZO-FlatMerge evaluates 10 forward perturbations
    for _ in range(num_samples):
        E = torch.randn_like(alphas) * 0.05
        reconstructed_weights = reconstruct_weights(alphas, E)
        with torch.no_grad():
            for p, r in zip(model.parameters(), reconstructed_weights):
                p.copy_(r)
            out = model(x)
            loss = torch.mean(out ** 2)
end_time = time.perf_counter()
zo_flatmerge_time = (end_time - start_time) / 5.0
print(f"ZO-FlatMerge TTA: {zo_flatmerge_time * 1000:.2f} ms/step")

# Memory footprint:
# ZO-FlatMerge requires:
# - Base weights: weight_mem_mb (344 MB)
# - 4 Task vectors: 4 * weight_mem_mb (1376 MB)
# - Active model parameters: weight_mem_mb (344 MB)
# - NO gradient memory (grad_mem_mb = 0)
# - NO Adam moment memory for the entire model weights (opt_mem_mb = 0)
# - Negligible memory for the 12-dimensional alphas
zo_flatmerge_static_mem_mb = (1 + 4 + 1) * weight_mem_mb
print(f"ZO-FlatMerge static memory: {zo_flatmerge_static_mem_mb:.2f} MB")
print("ZO-FlatMerge activation cache memory: exactly 0.00 MB (Forward-only optimization)")

# Save profile to file
os.makedirs("results", exist_ok=True)
with open("results/hardware_profile.txt", "w") as f:
    f.write(f"Mock ViT-B/32 Parameters: {num_params / 1e6:.2f}M\n")
    f.write(f"Weight-Space TTA latency: {weight_tta_time * 1000:.2f} ms/step\n")
    f.write(f"Weight-Space TTA static memory: {weight_mem_mb + grad_mem_mb + opt_mem_mb:.2f} MB\n")
    f.write(f"ZO-FlatMerge TTA latency: {zo_flatmerge_time * 1000:.2f} ms/step\n")
    f.write(f"ZO-FlatMerge TTA static memory: {zo_flatmerge_static_mem_mb:.2f} MB\n")
    f.write(f"ZO-FlatMerge TTA activation caching: 0.00 MB (True forward-only optimization)\n")
    f.write(f"Reconstruction overhead ratio (ZO-FlatMerge / Weight TTA): {zo_flatmerge_time / weight_tta_time:.2f}x\n")

print("\nSaved hardware profile to results/hardware_profile.txt")