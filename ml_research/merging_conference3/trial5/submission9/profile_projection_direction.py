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
print("Subsampling test sets to 2000 samples per task deterministically...")
test_datasets = {}
g_test = torch.Generator()
g_test.manual_seed(42)
for name, ds in raw_test_datasets.items():
    indices = torch.randperm(len(ds), generator=g_test)[:2000].tolist()
    test_datasets[name] = Subset(ds, indices)

# Dynamic Validation (Calibration) split generator (we use seed 101 for the pilot study)
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
        print(f"Found saved expert checkpoint for {task} at {path}. Loading...")
        model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
        model.head = nn.Linear(192, 10)
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        expert_models[task] = model
    else:
        raise ValueError(f"Expert model checkpoint {path} not found!")

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

print(f"\nIdentified {len(target_layers)} target layers for merging.")

# Extract Task Vectors
task_vectors = {task: {} for task in tasks}
for name in target_layers:
    for task in tasks:
        task_vectors[task][name] = expert_params[task][name] - base_params[name]

# =====================================================================
# 5. Evaluation Helpers
# =====================================================================
def evaluate_merged_model(params_dict, base_model, task_heads):
    base_model.eval()
    accuracies = {}
    
    with torch.no_grad():
        for task in task_heads.keys():
            correct = 0
            total = 0
            
            task_params = params_dict.copy()
            # Task-conditional swapping of non-target parameters
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
# 6. Precompute SVD Projection Operators (Left and Right)
# =====================================================================
projection_left = {}
projection_right = {}

for name in target_layers:
    expert_updates = [task_vectors[task][name] for task in tasks]
    
    # Left SVD Projection: M_left shape [d_out, K * d_in]
    M_left = torch.cat(expert_updates, dim=1)
    U_l, S_l, V_l = torch.linalg.svd(M_left.double(), full_matrices=False)
    projection_left[name] = U_l.to(device)
    
    # Right SVD Projection: M_right shape [K * d_out, d_in]
    M_right = torch.cat(expert_updates, dim=0)
    U_r, S_r, V_r = torch.linalg.svd(M_right.double(), full_matrices=False)
    # Right-singular vectors from svd(M_right) are columns of V_r (since M_right = U S V_r^T)
    # V_r has shape [d_in, d_in]
    projection_right[name] = V_r.to(device)

# =====================================================================
# 7. Evaluate Projection Directions for Seed 101
# =====================================================================
seed = 101
val_data = get_validation_data(seed)
criterion = nn.CrossEntropyLoss()

gammas = [0.1, 0.3, 0.5]
results = {
    "Left (Output-Space)": {},
    "Right (Input-Space)": {},
    "Bilateral": {}
}

for gamma in gammas:
    print(f"\n--- Training GSC-Merge with gamma={gamma} ---")
    
    # A. Left Projection (GSC-Merge Default)
    projected_left_updates = {task: {} for task in tasks}
    for name in target_layers:
        d_out, d_in = base_params[name].shape
        r_l = max(1, int(gamma * d_out))
        U_l = projection_left[name][:, :r_l]
        P_l = torch.mm(U_l, U_l.t()).float()
        for task in tasks:
            v_k = task_vectors[task][name]
            projected_left_updates[task][name] = torch.mm(P_l, v_k)
            
    alpha_l = nn.Parameter(torch.full((len(target_layers), len(tasks)), 0.25, device=device))
    opt_l = optim.Adam([alpha_l], lr=1e-2, weight_decay=1e-4)
    for step in range(100):
        opt_l.zero_grad()
        loss = 0.0
        merged_params = base_params.copy()
        for l_idx, name in enumerate(target_layers):
            sum_v = torch.zeros_like(base_params[name])
            for t_idx, task in enumerate(tasks):
                sum_v += alpha_l[l_idx, t_idx] * projected_left_updates[task][name]
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
        opt_l.step()
        
    final_l_params = base_params.copy()
    with torch.no_grad():
        for l_idx, name in enumerate(target_layers):
            sum_v = torch.zeros_like(base_params[name])
            for t_idx, task in enumerate(tasks):
                sum_v += alpha_l[l_idx, t_idx] * projected_left_updates[task][name]
            final_l_params[name] = base_params[name] + sum_v
    accs_l = evaluate_merged_model(final_l_params, base_model, task_heads)
    results["Left (Output-Space)"][gamma] = accs_l
    print(f"Left Projection Joint Mean: {accs_l['Joint Mean']:.2f}%")
    
    # B. Right Projection (Input-Space)
    projected_right_updates = {task: {} for task in tasks}
    for name in target_layers:
        d_out, d_in = base_params[name].shape
        r_r = max(1, int(gamma * d_in))
        # V_r is of shape [d_in, d_in], we take first r_r rows because we transposed in SVD
        # V_r is obtained from svd(M_right) which returns V_r as [d_in, d_in].
        # The first r_r columns of V_r span the shared input subspace.
        V_r = projection_right[name][:, :r_r]
        P_r = torch.mm(V_r, V_r.t()).float()
        for task in tasks:
            v_k = task_vectors[task][name]
            # Projection in input space is: v_k * P_r
            projected_right_updates[task][name] = torch.mm(v_k, P_r)
            
    alpha_r = nn.Parameter(torch.full((len(target_layers), len(tasks)), 0.25, device=device))
    opt_r = optim.Adam([alpha_r], lr=1e-2, weight_decay=1e-4)
    for step in range(100):
        opt_r.zero_grad()
        loss = 0.0
        merged_params = base_params.copy()
        for l_idx, name in enumerate(target_layers):
            sum_v = torch.zeros_like(base_params[name])
            for t_idx, task in enumerate(tasks):
                sum_v += alpha_r[l_idx, t_idx] * projected_right_updates[task][name]
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
        opt_r.step()
        
    final_r_params = base_params.copy()
    with torch.no_grad():
        for l_idx, name in enumerate(target_layers):
            sum_v = torch.zeros_like(base_params[name])
            for t_idx, task in enumerate(tasks):
                sum_v += alpha_r[l_idx, t_idx] * projected_right_updates[task][name]
            final_r_params[name] = base_params[name] + sum_v
    accs_r = evaluate_merged_model(final_r_params, base_model, task_heads)
    results["Right (Input-Space)"][gamma] = accs_r
    print(f"Right Projection Joint Mean: {accs_r['Joint Mean']:.2f}%")
    
    # C. Bilateral Projection
    projected_bilateral_updates = {task: {} for task in tasks}
    for name in target_layers:
        d_out, d_in = base_params[name].shape
        r_l = max(1, int(gamma * d_out))
        U_l = projection_left[name][:, :r_l]
        P_l = torch.mm(U_l, U_l.t()).float()
        
        r_r = max(1, int(gamma * d_in))
        V_r = projection_right[name][:, :r_r]
        P_r = torch.mm(V_r, V_r.t()).float()
        
        for task in tasks:
            v_k = task_vectors[task][name]
            # Bilateral projection: P_l * v_k * P_r
            projected_bilateral_updates[task][name] = torch.mm(P_l, torch.mm(v_k, P_r))
            
    alpha_b = nn.Parameter(torch.full((len(target_layers), len(tasks)), 0.25, device=device))
    opt_b = optim.Adam([alpha_b], lr=1e-2, weight_decay=1e-4)
    for step in range(100):
        opt_b.zero_grad()
        loss = 0.0
        merged_params = base_params.copy()
        for l_idx, name in enumerate(target_layers):
            sum_v = torch.zeros_like(base_params[name])
            for t_idx, task in enumerate(tasks):
                sum_v += alpha_b[l_idx, t_idx] * projected_bilateral_updates[task][name]
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
        opt_b.step()
        
    final_b_params = base_params.copy()
    with torch.no_grad():
        for l_idx, name in enumerate(target_layers):
            sum_v = torch.zeros_like(base_params[name])
            for t_idx, task in enumerate(tasks):
                sum_v += alpha_b[l_idx, t_idx] * projected_bilateral_updates[task][name]
            final_b_params[name] = base_params[name] + sum_v
    accs_b = evaluate_merged_model(final_b_params, base_model, task_heads)
    results["Bilateral"][gamma] = accs_b
    print(f"Bilateral Projection Joint Mean: {accs_b['Joint Mean']:.2f}%")

# Save results
output_path = "results/projection_direction_results.json"
with open(output_path, "w") as f:
    # Convert gamma keys to strings so json supports them
    serializable_results = {}
    for key, val in results.items():
        serializable_results[key] = {str(g): v for g, v in val.items()}
    json.dump(serializable_results, f, indent=4)

print(f"\nSaved projection direction pilot results to {output_path}!")
