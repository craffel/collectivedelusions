import os
import sys
sys.path.insert(0, os.path.abspath("local_packages"))
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import torch.func as tf
from scipy.optimize import minimize

# =====================================================================
# 1. Device and Path Setup
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================================
# 2. Dataset and Transform Definition
# =====================================================================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading datasets...")
train_datasets = {
    "MNIST": datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    "FashionMNIST": datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True),
    "CIFAR-10": datasets.CIFAR10(root="./data", train=True, transform=transform, download=True),
    "SVHN": datasets.SVHN(root="./data", split="train", transform=transform, download=True)
}

raw_test_datasets = {
    "MNIST": datasets.MNIST(root="./data", train=False, transform=transform, download=True),
    "FashionMNIST": datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True),
    "CIFAR-10": datasets.CIFAR10(root="./data", train=False, transform=transform, download=True),
    "SVHN": datasets.SVHN(root="./data", split="test", transform=transform, download=True)
}

# Subsample test datasets to 2000 samples per task deterministically
print("Subsampling test sets to 2000 samples per task...")
test_datasets = {}
g_test = torch.Generator()
g_test.manual_seed(42)
for name, ds in raw_test_datasets.items():
    indices = torch.randperm(len(ds), generator=g_test)[:2000].tolist()
    test_datasets[name] = Subset(ds, indices)

# Dynamic Validation (Calibration) split generator (we use seed 101 for comparison)
def get_validation_data(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    
    validation_data = {}
    for name, ds in train_datasets.items():
        indices = torch.randperm(len(ds), generator=g)[:16].tolist()
        sub_ds = Subset(ds, indices)
        loader = DataLoader(sub_ds, batch_size=16, shuffle=False)
        inputs, targets = next(iter(loader))
        validation_data[name] = (inputs.to(device), targets.to(device))
    return validation_data

# =====================================================================
# 3. Loading Experts
# =====================================================================
tasks = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
expert_models = {}
for task in tasks:
    path = f"experts/{task.lower()}_expert.pt"
    if os.path.exists(path):
        print(f"Loading expert checkpoint for {task}...")
        model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
        model.head = nn.Linear(192, 10)
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        expert_models[task] = model
    else:
        raise ValueError(f"Expert checkpoint {path} not found!")

# =====================================================================
# 4. Extract Base Parameters, Expert Parameters and Heads
# =====================================================================
base_model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True).to(device)
base_params = {n: p.clone().detach() for n, p in base_model.named_parameters()}

task_heads = {}
expert_params = {}
for task in tasks:
    state_dict = expert_models[task].state_dict()
    task_heads[task] = {
        "head.weight": state_dict["head.weight"].clone().detach(),
        "head.bias": state_dict["head.bias"].clone().detach()
    }
    expert_params[task] = {
        n: p.clone().detach() for n, p in expert_models[task].named_parameters() if "head" not in n
    }

target_layers = []
for name, p in base_model.named_parameters():
    if "blocks" in name and "weight" in name and len(p.shape) == 2:
        target_layers.append(name)

print(f"Identified {len(target_layers)} target layers.")

# Extract Task Vectors
task_vectors = {task: {} for task in tasks}
for name in target_layers:
    for task in tasks:
        task_vectors[task][name] = expert_params[task][name] - base_params[name]

# =====================================================================
# 5. Precompute default Left SVD Projection Operators (gamma = 0.3)
# =====================================================================
gamma = 0.3
projected_task_vectors = {task: {} for task in tasks}

for name in target_layers:
    expert_updates = [task_vectors[task][name] for task in tasks]
    M = torch.cat(expert_updates, dim=1)
    U, S, V = torch.linalg.svd(M.double(), full_matrices=False)
    U = U.to(device)
    
    d_out, d_in = base_params[name].shape
    r = max(1, int(gamma * d_out))
    U_r = U[:, :r]
    P = torch.mm(U_r, U_r.t()).float()
    
    for task in tasks:
        projected_task_vectors[task][name] = torch.mm(P, task_vectors[task][name])

# =====================================================================
# 6. Evaluation Helpers
# =====================================================================
def evaluate_merged_model(params_dict, base_model, task_heads):
    base_model.eval()
    accuracies = {}
    
    with torch.no_grad():
        for task in task_heads.keys():
            correct = 0
            total = 0
            
            task_params = params_dict.copy()
            for name in base_params.keys():
                if name not in target_layers and "head" not in name:
                    task_params[name] = expert_params[task][name]
            for hn, hp in task_heads[task].items():
                task_params[hn] = hp
                
            test_ds = test_datasets[task]
            loader = DataLoader(test_ds, batch_size=256, shuffle=False)
            
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tf.functional_call(base_model, task_params, (inputs,))
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            accuracies[task] = 100. * correct / total
            
    accuracies["Joint Mean"] = np.mean([accuracies[t] for t in task_heads.keys()])
    return accuracies

# =====================================================================
# 7. Benchmarking Core
# =====================================================================
seed = 101
val_data = get_validation_data(seed)
criterion = nn.CrossEntropyLoss()

# Function to compute validation loss given alpha coefficients
def compute_loss_from_alpha(alpha_np):
    # Reshape the 1D flat array back to (num_layers, num_tasks)
    alpha_tensor = torch.from_numpy(alpha_np.reshape((len(target_layers), len(tasks)))).float().to(device)
    
    merged_params = base_params.copy()
    for l_idx, name in enumerate(target_layers):
        sum_v = torch.zeros_like(base_params[name])
        for t_idx, task in enumerate(tasks):
            sum_v += alpha_tensor[l_idx, t_idx] * projected_task_vectors[task][name]
        merged_params[name] = base_params[name] + sum_v
        
    loss = 0.0
    for t_idx, task in enumerate(tasks):
        inputs, targets = val_data[task]
        task_params = merged_params.copy()
        for name in base_params.keys():
            if name not in target_layers and "head" not in name:
                task_params[name] = expert_params[task][name]
        for hn, hp in task_heads[task].items():
            task_params[hn] = hp
        outputs = tf.functional_call(base_model, task_params, (inputs,))
        loss += criterion(outputs, targets).item()
        
    return loss / len(tasks)

# Benchmark 1: First-Order Adam (Gradient-Based)
print("\n--- Running Adam Optimizer (Gradient-Based) ---")
alpha_adam = nn.Parameter(torch.full((len(target_layers), len(tasks)), 0.25, device=device))
opt_adam = optim.Adam([alpha_adam], lr=1e-2, weight_decay=1e-4)

t0 = time.time()
for step in range(100):
    opt_adam.zero_grad()
    loss = 0.0
    merged_params = base_params.copy()
    for l_idx, name in enumerate(target_layers):
        sum_v = torch.zeros_like(base_params[name])
        for t_idx, task in enumerate(tasks):
            sum_v += alpha_adam[l_idx, t_idx] * projected_task_vectors[task][name]
        merged_params[name] = base_params[name] + sum_v
        
    for t_idx, task in enumerate(tasks):
        inputs, targets = val_data[task]
        task_params = merged_params.copy()
        for name in base_params.keys():
            if name not in target_layers and "head" not in name:
                task_params[name] = expert_params[task][name]
        for hn, hp in task_heads[task].items():
            task_params[hn] = hp
        outputs = tf.functional_call(base_model, task_params, (inputs,))
        loss += criterion(outputs, targets)
    loss = loss / len(tasks)
    loss.backward()
    opt_adam.step()
adam_time = time.time() - t0
final_adam_loss = loss.item()

# Evaluate Adam parameters
final_adam_params = base_params.copy()
with torch.no_grad():
    for l_idx, name in enumerate(target_layers):
        sum_v = torch.zeros_like(base_params[name])
        for t_idx, task in enumerate(tasks):
            sum_v += alpha_adam[l_idx, t_idx] * projected_task_vectors[task][name]
        final_adam_params[name] = base_params[name] + sum_v
adam_accs = evaluate_merged_model(final_adam_params, base_model, task_heads)
print(f"Adam Completed in {adam_time:.2f}s | Val Loss: {final_adam_loss:.4f} | Test Joint Mean Acc: {adam_accs['Joint Mean']:.2f}%")

# Benchmark 2: Derivative-Free Powell Search
print("\n--- Running Powell Optimizer (Derivative-Free) ---")
# Initialize at uniform 0.25
init_alpha = np.full((len(target_layers) * len(tasks)), 0.25)

t0 = time.time()
res_powell = minimize(
    compute_loss_from_alpha, 
    init_alpha, 
    method='Powell', 
    options={'maxfev': 300, 'disp': True}
)
powell_time = time.time() - t0
final_powell_loss = res_powell.fun

# Evaluate Powell parameters
powell_alpha_tensor = torch.from_numpy(res_powell.x.reshape((len(target_layers), len(tasks)))).float().to(device)
final_powell_params = base_params.copy()
with torch.no_grad():
    for l_idx, name in enumerate(target_layers):
        sum_v = torch.zeros_like(base_params[name])
        for t_idx, task in enumerate(tasks):
            sum_v += powell_alpha_tensor[l_idx, t_idx] * projected_task_vectors[task][name]
        final_powell_params[name] = base_params[name] + sum_v
powell_accs = evaluate_merged_model(final_powell_params, base_model, task_heads)
print(f"Powell Completed in {powell_time:.2f}s | Val Loss: {final_powell_loss:.4f} | Test Joint Mean Acc: {powell_accs['Joint Mean']:.2f}%")

# Benchmark 3: Derivative-Free Nelder-Mead Search
print("\n--- Running Nelder-Mead Optimizer (Derivative-Free) ---")
t0 = time.time()
res_nm = minimize(
    compute_loss_from_alpha, 
    init_alpha, 
    method='Nelder-Mead', 
    options={'maxfev': 300, 'disp': True}
)
nm_time = time.time() - t0
final_nm_loss = res_nm.fun

# Evaluate Nelder-Mead parameters
nm_alpha_tensor = torch.from_numpy(res_nm.x.reshape((len(target_layers), len(tasks)))).float().to(device)
final_nm_params = base_params.copy()
with torch.no_grad():
    for l_idx, name in enumerate(target_layers):
        sum_v = torch.zeros_like(base_params[name])
        for t_idx, task in enumerate(tasks):
            sum_v += nm_alpha_tensor[l_idx, t_idx] * projected_task_vectors[task][name]
        final_nm_params[name] = base_params[name] + sum_v
nm_accs = evaluate_merged_model(final_nm_params, base_model, task_heads)
print(f"Nelder-Mead Completed in {nm_time:.2f}s | Val Loss: {final_nm_loss:.4f} | Test Joint Mean Acc: {nm_accs['Joint Mean']:.2f}%")

# =====================================================================
# 8. Save Results
# =====================================================================
results = {
    "Adam": {
        "time_s": adam_time,
        "val_loss": final_adam_loss,
        "joint_mean_acc": adam_accs["Joint Mean"],
        "mnist_acc": adam_accs["MNIST"],
        "fmnist_acc": adam_accs["FashionMNIST"],
        "cifar_acc": adam_accs["CIFAR-10"],
        "svhn_acc": adam_accs["SVHN"]
    },
    "Powell": {
        "time_s": powell_time,
        "val_loss": final_powell_loss,
        "joint_mean_acc": powell_accs["Joint Mean"],
        "mnist_acc": powell_accs["MNIST"],
        "fmnist_acc": powell_accs["FashionMNIST"],
        "cifar_acc": powell_accs["CIFAR-10"],
        "svhn_acc": powell_accs["SVHN"]
    },
    "Nelder-Mead": {
        "time_s": nm_time,
        "val_loss": final_nm_loss,
        "joint_mean_acc": nm_accs["Joint Mean"],
        "mnist_acc": nm_accs["MNIST"],
        "fmnist_acc": nm_accs["FashionMNIST"],
        "cifar_acc": nm_accs["CIFAR-10"],
        "svhn_acc": nm_accs["SVHN"]
    }
}

os.makedirs("results", exist_ok=True)
output_path = "results/derivative_free_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nSaved profiling results to {output_path}")
