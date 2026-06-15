import torch
import torch.nn as nn
import numpy as np
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc(x))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K = 3
scales = [0.1, 0.5, 2.0]
seeds = [101, 102, 103]

layers = ['conv1.weight', 'conv2.weight', 'fc.weight']
layer_stats = {l: {'alpha': [], 'lambda': []} for l in layers}

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # We will simulate task vectors for SimpleCNN layers
    for layer_name in layers:
        if 'conv1' in layer_name:
            shape = (16, 1, 3, 3)
        elif 'conv2' in layer_name:
            shape = (32, 16, 3, 3)
        elif 'fc' in layer_name:
            shape = (128, 32 * 7 * 7)
            
        d1 = shape[0]
        d2 = np.prod(shape[1:])
        
        task_vectors = []
        for k in range(K):
            raw_update = torch.randn(*shape, device=device)
            rms_raw = torch.sqrt((raw_update**2).mean() + 1e-8)
            tau = scales[k] * (raw_update / rms_raw)
            task_vectors.append(tau)
            
        rms_vals = [torch.sqrt((t**2).mean() + 1e-8) for t in task_vectors]
        norm_vectors = [t / r for t, r in zip(task_vectors, rms_vals)]
        avg_norm = sum(norm_vectors) / len(norm_vectors)
        alpha = torch.sqrt((avg_norm**2).mean() + 1e-8).item()
        
        layer_stats[layer_name]['alpha'].append(alpha)
        layer_stats[layer_name]['lambda'].append(1.0 / alpha)

print("--- SimpleCNN Layer-wise scale stats ---")
for layer_name in layers:
    alpha_mean = np.mean(layer_stats[layer_name]['alpha'])
    lambda_mean = np.mean(layer_stats[layer_name]['lambda'])
    print(f"Layer: {layer_name}")
    print(f"  alpha  = {alpha_mean:.4f}")
    print(f"  lambda = {lambda_mean:.4f}")
