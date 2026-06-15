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

# Set random seeds for absolute reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print("Starting physical CNN sensitivity and ablation script...")

# 1. Define Deeper CNN Architecture (5 Layers)
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
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
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
L = 5

# 2. Prepare datasets
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

M = 10
mnist_val = Subset(mnist_train_full, list(range(train_size, train_size + M)))
fmnist_val = Subset(fmnist_train_full, list(range(train_size, train_size + M)))

mnist_val_loader = DataLoader(mnist_val, batch_size=M, shuffle=False)
fmnist_val_loader = DataLoader(fmnist_val, batch_size=M, shuffle=False)

for x_mv, y_mv in mnist_val_loader:
    mnist_val_x, mnist_val_y = x_mv, y_mv
for x_fv, y_fv in fmnist_val_loader:
    fmnist_val_x, fmnist_val_y = x_fv, y_fv

# 3. Train base and expert models
def train_model(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

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

base_model = DeepCNN()
model_A = DeepCNN()
model_A.load_state_dict(base_model.state_dict())
train_model(model_A, mnist_train_loader)

model_B = DeepCNN()
model_B.load_state_dict(base_model.state_dict())
train_model(model_B, fmnist_train_loader)

base_params = {k: v.clone().detach() for k, v in base_model.state_dict().items()}
params_A = {k: v.clone().detach() for k, v in model_A.state_dict().items()}
params_B = {k: v.clone().detach() for k, v in model_B.state_dict().items()}

V_A = {k: params_A[k] - base_params[k] for k in base_params.keys()}
V_B = {k: params_B[k] - base_params[k] for k in base_params.keys()}

criterion = nn.CrossEntropyLoss()

# 4. Evaluation 1: Initialization Sensitivity Sweep
# We run OFS-Tune GT-Merge starting from 10 different random weight initializations
print("\nRunning Initialization Sensitivity Sweep (10 distinct configurations)...")
init_configs = [
    (0.5, 0.5),    # Standard uniform
    (0.0, 0.0),    # Zeros
    (1.0, 1.0),    # Large positive
    (-0.5, -0.5),  # Negative
    (0.1, 0.9),    # Highly skewed A-dominant
    (0.9, 0.1),    # Highly skewed B-dominant
    (-0.2, 0.8),   # Mixed sign
    (0.8, -0.2),   # Mixed sign
    (0.3, 0.3),    # Low uniform
    (1.2, 1.2)     # Extrapolated
]

init_results = []

for i, (alpha_A_start, alpha_B_start) in enumerate(init_configs):
    a_A = torch.tensor(alpha_A_start, requires_grad=True)
    a_B = torch.tensor(alpha_B_start, requires_grad=True)
    opt = optim.Adam([a_A, a_B], lr=0.1)
    
    # Run optimization
    for step in range(50):
        opt.zero_grad()
        merged_params = {
            k: base_params[k] + a_A * V_A[k] + a_B * V_B[k] for k in base_params.keys()
        }
        out_mnist = functional_call(base_model, merged_params, mnist_val_x)
        loss_mnist = criterion(out_mnist, mnist_val_y)
        
        out_fmnist = functional_call(base_model, merged_params, fmnist_val_x)
        loss_fmnist = criterion(out_fmnist, fmnist_val_y)
        
        loss = loss_mnist + loss_fmnist
        loss.backward()
        opt.step()
        
    final_A, final_B = a_A.item(), a_B.item()
    
    # Evaluate
    test_params = {
        k: base_params[k] + final_A * V_A[k] + final_B * V_B[k] for k in base_params.keys()
    }
    acc_m = eval_model_functional(base_model, test_params, mnist_test_loader)
    acc_f = eval_model_functional(base_model, test_params, fmnist_test_loader)
    acc_avg = (acc_m + acc_f) / 2
    
    print(f"Init {i+1} ({alpha_A_start:.1f}, {alpha_B_start:.1f}) -> Optimized: ({final_A:.4f}, {final_B:.4f}) -> Test Acc: {acc_avg:.4f}")
    
    init_results.append({
        "init": (alpha_A_start, alpha_B_start),
        "optimized": (final_A, final_B),
        "test_acc": acc_avg
    })

# Compute statistics
accs = [r["test_acc"] for r in init_results]
alphas_A_opt = [r["optimized"][0] for r in init_results]
alphas_B_opt = [r["optimized"][1] for r in init_results]

print(f"\nInitialization Sensitivity Stats:")
print(f"Optimized Accuracy Mean: {np.mean(accs):.4f}, Std: {np.std(accs):.4f}")
print(f"Optimized alpha_A Mean: {np.mean(alphas_A_opt):.4f}, Std: {np.std(alphas_A_opt):.4f}")
print(f"Optimized alpha_B Mean: {np.mean(alphas_B_opt):.4f}, Std: {np.std(alphas_B_opt):.4f}")


# 5. Evaluation 2: Optimization Budget Convergence Tracking
print("\nRunning Optimization Budget Convergence Sweep...")
a_A = torch.tensor(0.5, requires_grad=True)
a_B = torch.tensor(0.5, requires_grad=True)
opt = optim.Adam([a_A, a_B], lr=0.1)

budget_results = []

# Step 0 (Before optimization, i.e., Uniform TA)
with torch.no_grad():
    m_params = {k: base_params[k] + 0.5 * V_A[k] + 0.5 * V_B[k] for k in base_params.keys()}
    val_out_m = functional_call(base_model, m_params, mnist_val_x)
    val_out_f = functional_call(base_model, m_params, fmnist_val_x)
    val_loss = criterion(val_out_m, mnist_val_y).item() + criterion(val_out_f, fmnist_val_y).item()
acc_m = eval_model_functional(base_model, m_params, mnist_test_loader)
acc_f = eval_model_functional(base_model, m_params, fmnist_test_loader)
acc_avg = (acc_m + acc_f) / 2

budget_results.append({
    "step": 0,
    "val_loss": val_loss,
    "test_acc": acc_avg,
    "alpha_A": 0.5,
    "alpha_B": 0.5
})
print(f"Step 0 (Uniform) -> Val Loss: {val_loss:.4f}, Test Acc: {acc_avg:.4f}")

for step in range(1, 51):
    opt.zero_grad()
    merged_params = {
        k: base_params[k] + a_A * V_A[k] + a_B * V_B[k] for k in base_params.keys()
    }
    out_mnist = functional_call(base_model, merged_params, mnist_val_x)
    loss_mnist = criterion(out_mnist, mnist_val_y)
    
    out_fmnist = functional_call(base_model, merged_params, fmnist_val_x)
    loss_fmnist = criterion(out_fmnist, fmnist_val_y)
    
    loss = loss_mnist + loss_fmnist
    loss.backward()
    opt.step()
    
    # Record metrics at select steps
    if step in [1, 2, 5, 10, 15, 20, 30, 40, 50]:
        curr_A, curr_B = a_A.item(), a_B.item()
        test_params = {
            k: base_params[k] + curr_A * V_A[k] + curr_B * V_B[k] for k in base_params.keys()
        }
        acc_m = eval_model_functional(base_model, test_params, mnist_test_loader)
        acc_f = eval_model_functional(base_model, test_params, fmnist_test_loader)
        acc_avg = (acc_m + acc_f) / 2
        print(f"Step {step} -> Val Loss: {loss.item():.4f}, Test Acc: {acc_avg:.4f}, Coeffs: ({curr_A:.4f}, {curr_B:.4f})")
        
        budget_results.append({
            "step": step,
            "val_loss": loss.item(),
            "test_acc": acc_avg,
            "alpha_A": curr_A,
            "alpha_B": curr_B
        })

# Save results
sensitivity_data = {
    "init_sensitivity": init_results,
    "init_stats": {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "alpha_A_mean": np.mean(alphas_A_opt), "alpha_A_std": np.std(alphas_A_opt),
        "alpha_B_mean": np.mean(alphas_B_opt), "alpha_B_std": np.std(alphas_B_opt)
    },
    "budget_convergence": budget_results
}

with open("sensitivity_results.json", "w") as f:
    json.dump(sensitivity_data, f, indent=2)
print("\nSensitivity results saved to 'sensitivity_results.json'.")
