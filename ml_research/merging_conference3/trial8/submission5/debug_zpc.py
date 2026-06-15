import torch
import numpy as np
from run_experiments import generate_sandbox_data

prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(10)
X_cal, Y_cal_task, Y_cal_class = cal_data
X_test, Y_test_task, Y_test_class = test_data

K = 4
D = 192

mu = torch.zeros(K, D)
for k in range(K):
    task_mask = (Y_cal_task == k)
    mu[k] = X_cal[task_mask].mean(dim=0)

# Let's check some cosine similarities
print("Centroid Norms:", [mu[k].norm().item() for k in range(K)])
for k in range(K):
    test_task_mask = (Y_test_task == k)
    xb = X_test[test_task_mask][:3]
    print(f"\nSamples from Task {k}:")
    for x in xb:
        sims = [torch.dot(x, mu[j]).item() / (x.norm() * mu[j].norm() + 1e-8) for j in range(K)]
        print(f"Cosine Sims: {['%.4f' % s for s in sims]}")
