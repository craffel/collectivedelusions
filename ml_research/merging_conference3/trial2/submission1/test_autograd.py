import torch
import time

# Create dummy parameters of size similar to ViT-B-32 (e.g. 150 tensors of size 512x512)
num_params = 150
param_shape = (512, 512)

# Case 1: Components require grad (original)
pretrained_1 = [torch.randn(param_shape, requires_grad=True) for _ in range(num_params)]
task_vec_1 = [torch.randn(param_shape, requires_grad=True) for _ in range(num_params)]

lambdas_raw_1 = torch.nn.Parameter(torch.ones(num_params, 1) * 0.3)

t0 = time.time()
# forward pass
alph_1 = torch.cat((torch.ones(num_params, 1), lambdas_raw_1), 1)
loss_1 = 0.0
for j in range(num_params):
    merged_p = pretrained_1[j] * alph_1[j, 0] + task_vec_1[j] * alph_1[j, 1]
    loss_1 += merged_p.sum()

# backward pass
loss_1.backward()
t1 = time.time()
grad_1 = lambdas_raw_1.grad.clone()
time_original = t1 - t0
print(f"Original (requires_grad=True): Time={time_original:.4f}s, Grad Norm={grad_1.norm().item():.4f}")

# Case 2: Components DO NOT require grad (optimized)
pretrained_2 = [torch.randn(param_shape, requires_grad=False) for _ in range(num_params)]
task_vec_2 = [torch.randn(param_shape, requires_grad=False) for _ in range(num_params)]

lambdas_raw_2 = torch.nn.Parameter(torch.ones(num_params, 1) * 0.3)

t0 = time.time()
# forward pass
alph_2 = torch.cat((torch.ones(num_params, 1), lambdas_raw_2), 1)
loss_2 = 0.0
for j in range(num_params):
    merged_p = pretrained_2[j] * alph_2[j, 0] + task_vec_2[j] * alph_2[j, 1]
    loss_2 += merged_p.sum()

# backward pass
loss_2.backward()
t1 = time.time()
grad_2 = lambdas_raw_2.grad.clone()
time_optimized = t1 - t0
print(f"Optimized (requires_grad=False): Time={time_optimized:.4f}s, Grad Norm={grad_2.norm().item():.4f}")

# Verify correctness
print("Gradients are mathematically identical:", torch.allclose(grad_1, grad_2))
print(f"Speedup: {time_original / time_optimized:.2f}x")
