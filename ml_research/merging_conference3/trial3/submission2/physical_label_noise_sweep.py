import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import random
import json

# Define Deeper CNN Architecture (5 Layers)
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Layer 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Layer 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Layer 3
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                    # Layer 4
        self.fc2 = nn.Linear(128, 10)                             # Layer 5
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # pool 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x))) # pool 14x14 -> 7x7
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

PARAM_LAYERS = {
    "conv1.weight": 0, "conv1.bias": 0,
    "conv2.weight": 1, "conv2.bias": 1,
    "conv3.weight": 2, "conv3.bias": 2,
    "fc1.weight": 3, "fc1.bias": 3,
    "fc2.weight": 4, "fc2.bias": 4
}
L = 5  # Total number of layers

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fmnist_test_full = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_size = 2000
test_size = 1000

mnist_train = Subset(mnist_train_full, list(range(train_size)))
mnist_test = Subset(mnist_test_full, list(range(test_size)))
fmnist_train = Subset(fmnist_train_full, list(range(train_size)))
fmnist_test = Subset(fmnist_test_full, list(range(test_size)))

mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
fmnist_train_loader = DataLoader(fmnist_train, batch_size=64, shuffle=True)
fmnist_test_loader = DataLoader(fmnist_test, batch_size=64, shuffle=False)

def eval_model_functional(model, params, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = functional_call(model, params, x)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def add_label_noise(y, noise_rate, seed):
    if noise_rate == 0.0:
        return y.clone()
    np.random.seed(seed)
    y_noisy = y.clone()
    n_to_flip = int(len(y) * noise_rate)
    indices = np.random.choice(len(y), n_to_flip, replace=False)
    for idx in indices:
        current_label = y[idx].item()
        possible_labels = [l for l in range(10) if l != current_label]
        new_label = np.random.choice(possible_labels)
        y_noisy[idx] = int(new_label)
    return y_noisy

# We will run across 5 random seeds
seeds = [42, 43, 44, 45, 46]
noise_levels = [0.0, 0.15, 0.30]

all_results = {}

for noise_rate in noise_levels:
    all_results[noise_rate] = {
        "uniform_ta": [],
        "ofs_gt": [],
        "ofs_poly": [],
        "head_val": [],
        "ft_val": []
    }

print("Running physical CNN validation sweep across seeds and label noise levels...")

criterion = nn.CrossEntropyLoss()

for seed in seeds:
    print(f"\n=================== SEED {seed} ===================")
    # Set seed for training and data splits
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize base and experts
    base_model = DeepCNN()
    model_A = DeepCNN()
    model_A.load_state_dict(base_model.state_dict())
    
    # Train Expert A on MNIST
    optimizer_A = optim.Adam(model_A.parameters(), lr=0.001)
    model_A.train()
    for epoch in range(3):
        for x, y in mnist_train_loader:
            optimizer_A.zero_grad()
            loss = criterion(model_A(x), y)
            loss.backward()
            optimizer_A.step()
            
    # Train Expert B on FMNIST
    model_B = DeepCNN()
    model_B.load_state_dict(base_model.state_dict())
    optimizer_B = optim.Adam(model_B.parameters(), lr=0.001)
    model_B.train()
    for epoch in range(3):
        for x, y in fmnist_train_loader:
            optimizer_B.zero_grad()
            loss = criterion(model_B(x), y)
            loss.backward()
            optimizer_B.step()
            
    base_params = {k: v.clone().detach() for k, v in base_model.state_dict().items()}
    V_A = {k: model_A.state_dict()[k] - base_params[k] for k in base_params.keys()}
    V_B = {k: model_B.state_dict()[k] - base_params[k] for k in base_params.keys()}
    
    # Evaluation values
    uniform_params = {k: base_params[k] + 0.5 * V_A[k] + 0.5 * V_B[k] for k in base_params.keys()}
    acc_uni_mnist = eval_model_functional(base_model, uniform_params, mnist_test_loader)
    acc_uni_fmnist = eval_model_functional(base_model, uniform_params, fmnist_test_loader)
    acc_uni = (acc_uni_mnist + acc_uni_fmnist) / 2
    
    # Get validation set for this seed
    # To make it seed-dependent, we shift the validation subset range
    M = 10
    start_idx = train_size + (seed - 42) * M
    mnist_val = Subset(mnist_train_full, list(range(start_idx, start_idx + M)))
    fmnist_val = Subset(fmnist_train_full, list(range(start_idx, start_idx + M)))
    
    mnist_val_loader = DataLoader(mnist_val, batch_size=M, shuffle=False)
    fmnist_val_loader = DataLoader(fmnist_val, batch_size=M, shuffle=False)
    
    for x_mv, y_mv in mnist_val_loader:
        mnist_val_x, mnist_val_y_clean = x_mv, y_mv
    for x_fv, y_fv in fmnist_val_loader:
        fmnist_val_x, fmnist_val_y_clean = x_fv, y_fv
        
    for noise_rate in noise_levels:
        print(f"\n--- Seed {seed} | Noise Rate {noise_rate} ---")
        val_y_mnist = add_label_noise(mnist_val_y_clean, noise_rate, seed + 100)
        val_y_fmnist = add_label_noise(fmnist_val_y_clean, noise_rate, seed + 200)
        
        # 1. Uniform
        all_results[noise_rate]["uniform_ta"].append(acc_uni)
        
        # 2. OFS-Tune GT-Merge
        alpha_A = torch.tensor(0.5, requires_grad=True)
        alpha_B = torch.tensor(0.5, requires_grad=True)
        optimizer_gt = optim.Adam([alpha_A, alpha_B], lr=0.1)
        for step in range(50):
            optimizer_gt.zero_grad()
            m_params = {k: base_params[k] + alpha_A * V_A[k] + alpha_B * V_B[k] for k in base_params.keys()}
            loss = criterion(functional_call(base_model, m_params, mnist_val_x), val_y_mnist) + \
                   criterion(functional_call(base_model, m_params, fmnist_val_x), val_y_fmnist)
            loss.backward()
            optimizer_gt.step()
            
        final_gt_params = {k: base_params[k] + alpha_A.item() * V_A[k] + alpha_B.item() * V_B[k] for k in base_params.keys()}
        acc_gt = (eval_model_functional(base_model, final_gt_params, mnist_test_loader) + eval_model_functional(base_model, final_gt_params, fmnist_test_loader)) / 2
        all_results[noise_rate]["ofs_gt"].append(acc_gt)
        print(f"OFS-Tune GT: {acc_gt:.4f}")
        
        # 3. OFS-Tune Poly-Val (d=1)
        c_A0 = torch.tensor(0.5, requires_grad=True)
        c_A1 = torch.tensor(0.0, requires_grad=True)
        c_B0 = torch.tensor(0.5, requires_grad=True)
        c_B1 = torch.tensor(0.0, requires_grad=True)
        optimizer_poly = optim.Adam([c_A0, c_A1, c_B0, c_B1], lr=0.1)
        for step in range(50):
            optimizer_poly.zero_grad()
            m_params = {}
            for k in base_params.keys():
                layer_idx = PARAM_LAYERS[k]
                norm_depth = layer_idx / (L - 1)
                a_A = c_A0 + c_A1 * norm_depth
                a_B = c_B0 + c_B1 * norm_depth
                m_params[k] = base_params[k] + a_A * V_A[k] + a_B * V_B[k]
            loss = criterion(functional_call(base_model, m_params, mnist_val_x), val_y_mnist) + \
                   criterion(functional_call(base_model, m_params, fmnist_val_x), val_y_fmnist)
            loss.backward()
            optimizer_poly.step()
            
        final_poly_params = {}
        for k in base_params.keys():
            layer_idx = PARAM_LAYERS[k]
            norm_depth = layer_idx / (L - 1)
            a_A = c_A0.item() + c_A1.item() * norm_depth
            a_B = c_B0.item() + c_B1.item() * norm_depth
            final_poly_params[k] = base_params[k] + a_A * V_A[k] + a_B * V_B[k]
        acc_poly = (eval_model_functional(base_model, final_poly_params, mnist_test_loader) + eval_model_functional(base_model, final_poly_params, fmnist_test_loader)) / 2
        all_results[noise_rate]["ofs_poly"].append(acc_poly)
        print(f"OFS-Tune Poly: {acc_poly:.4f}")
        
        # 4. Few-Shot Head-Only Tuning
        head_params = {k: v.clone().requires_grad_(PARAM_LAYERS[k] == 4) for k, v in uniform_params.items()}
        opt_params = [v for k, v in head_params.items() if PARAM_LAYERS[k] == 4]
        optimizer_head = optim.Adam(opt_params, lr=0.01)
        for step in range(20):
            optimizer_head.zero_grad()
            loss = criterion(functional_call(base_model, head_params, mnist_val_x), val_y_mnist) + \
                   criterion(functional_call(base_model, head_params, fmnist_val_x), val_y_fmnist)
            loss.backward()
            optimizer_head.step()
            
        acc_head = (eval_model_functional(base_model, head_params, mnist_test_loader) + eval_model_functional(base_model, head_params, fmnist_test_loader)) / 2
        all_results[noise_rate]["head_val"].append(acc_head)
        print(f"Head Tuning: {acc_head:.4f}")
        
        # 5. Few-Shot Joint FT
        ft_params = {k: v.clone().requires_grad_(True) for k, v in uniform_params.items()}
        optimizer_ft = optim.Adam(ft_params.values(), lr=0.001)
        for step in range(20):
            optimizer_ft.zero_grad()
            loss = criterion(functional_call(base_model, ft_params, mnist_val_x), val_y_mnist) + \
                   criterion(functional_call(base_model, ft_params, fmnist_val_x), val_y_fmnist)
            loss.backward()
            optimizer_ft.step()
            
        acc_ft = (eval_model_functional(base_model, ft_params, mnist_test_loader) + eval_model_functional(base_model, ft_params, fmnist_test_loader)) / 2
        all_results[noise_rate]["ft_val"].append(acc_ft)
        print(f"Joint FT: {acc_ft:.4f}")

# Summarize statistics
summary = {}
for noise_rate in noise_levels:
    summary[noise_rate] = {}
    print(f"\n=================== SUMMARY FOR NOISE {noise_rate*100:.0f}% ===================")
    for method in ["uniform_ta", "ofs_gt", "ofs_poly", "head_val", "ft_val"]:
        vals = all_results[noise_rate][method]
        mean = np.mean(vals)
        std = np.std(vals)
        summary[noise_rate][method] = {"mean": mean, "std": std}
        print(f"{method}: {mean:.4f} +- {std:.4f}")

with open("physical_noise_sweep_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSweep results saved to 'physical_noise_sweep_results.json'")
