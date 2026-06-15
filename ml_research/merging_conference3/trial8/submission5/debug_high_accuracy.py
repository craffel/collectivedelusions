import torch
import numpy as np
from test_high_accuracy import generate_sandbox_data_high_accuracy

prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data_high_accuracy(10)
X_cal, Y_cal_task, Y_cal_class = cal_data
X_test, Y_test_task, Y_test_class = test_data

# Train expert heads
K = 4
D = 192
C = 10
subspace_dim = 48
expert_heads_W = torch.zeros(K, C, D)
expert_heads_B = torch.zeros(K, C)
for k in range(K):
    expert_heads_W[k, :, k*subspace_dim : (k+1)*subspace_dim] = prototypes[k]

# Compute dispersion
pear_dispersion = torch.zeros(K)
for k in range(K):
    task_mask = (Y_cal_task == k)
    X_task_cal = X_cal[task_mask]
    Y_class_cal = Y_cal_class[task_mask]
    sims = []
    for s_idx, s in enumerate(X_task_cal):
        c_label = Y_class_cal[s_idx].item()
        w_c = expert_heads_W[k, c_label]
        cos_sim = torch.dot(s, w_c) / (s.norm() * w_c.norm() + 1e-8)
        sims.append(cos_sim.item())
    pear_dispersion[k] = np.mean(sims)

print("Dispersion factors:", pear_dispersion)

# Let's check some cosine similarities
for k in range(K):
    test_task_mask = (Y_test_task == k)
    xb = X_test[test_task_mask][:3]
    print(f"\nSamples from Task {k}:")
    for x in xb:
        # compute max cosine similarities
        cos_sims = torch.zeros(K)
        for j in range(K):
            W_j = expert_heads_W[j]
            W_j_norms = W_j.norm(dim=-1, keepdim=True) + 1e-8
            x_norm = x.norm() + 1e-8
            dot_prods = W_j @ x
            cos_class = dot_prods / (W_j_norms.squeeze() * x_norm)
            cos_sims[j] = cos_class.max()
            
        calibrated_sims = cos_sims / (pear_dispersion + 1e-8)
        route_coeffs = torch.softmax(calibrated_sims / 0.001, dim=-1)
        print(f"Max Cosine Sims: {['%.4f' % s.item() for s in cos_sims]}")
        print(f"Calibrated Sims: {['%.4f' % s.item() for s in calibrated_sims]}")
        print(f"Routing Coeffs:  {['%.4f' % s.item() for s in route_coeffs]}")
