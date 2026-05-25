import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

# Define the shared MLP encoder architecture
class MLPEncoder(nn.Module):
    def __init__(self):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# Task-specific classification head
class TaskHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super(TaskHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

# Full model combining shared encoder and task-specific head
class MultiTaskModel(nn.Module):
    def __init__(self, encoder, head):
        super(MultiTaskModel, self).__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

def get_dataloader(dataset_name, train=True, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "KMNIST":
        dataset = torchvision.datasets.KMNIST(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)

def train_expert(dataset_name, base_encoder, device, epochs=3, lr=1e-3):
    print(f"\n--- Training Expert for {dataset_name} ---")
    encoder = copy.deepcopy(base_encoder).to(device)
    head = TaskHead().to(device)
    model = MultiTaskModel(encoder, head).to(device)
    
    train_loader = get_dataloader(dataset_name, train=True, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    return model.encoder.cpu(), model.head.cpu()

def compute_fisher(dataset_name, encoder, head, device, num_samples=2000):
    print(f"Computing Fisher Information for {dataset_name}...")
    encoder = copy.deepcopy(encoder).to(device)
    head = copy.deepcopy(head).to(device)
    model = MultiTaskModel(encoder, head).to(device)
    model.eval()
    
    loader = get_dataloader(dataset_name, train=True, batch_size=1)
    fisher = {}
    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            
    count = 0
    for images, labels in loader:
        if count >= num_samples:
            break
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        for name, param in model.encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2
                
        count += 1
        
    for name in fisher:
        fisher[name] /= count
        fisher[name] = torch.clamp(fisher[name], min=1e-8)
        
    print(f"Fisher computed. Sample parameter mean: {next(iter(fisher.values())).mean().item():.6f}")
    return {name: f.cpu() for name, f in fisher.items()}

def evaluate_model(encoder, head, dataset_name, device, corruption=None):
    encoder = copy.deepcopy(encoder).to(device)
    head = copy.deepcopy(head).to(device)
    model = MultiTaskModel(encoder, head).to(device)
    model.eval()
    
    test_loader = get_dataloader(dataset_name, train=False, batch_size=256)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Apply corruption if requested
            if corruption is not None:
                if corruption == "noise":
                    images = images + 0.2 * torch.randn_like(images)
                elif corruption == "blur":
                    images = F.avg_pool2d(images, kernel_size=3, stride=1, padding=1)
                elif corruption == "contrast":
                    images = images * 0.25
                elif corruption == "rotation":
                    images = torch.rot90(images, 1, [2, 3])
                    
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100.0 * correct / total
    return acc

# Optimization loop to find the orthogonal Procrustes rotation
def optimize_procrustes(W_0, W_k, F_k, steps=150, lr=0.05, weighted=True, device='cpu'):
    orig_shape = W_0.shape
    out_dim = orig_shape[0]
    
    W_0_2d = W_0.reshape(out_dim, -1).to(device)
    W_k_2d = W_k.reshape(out_dim, -1).to(device)
    if F_k is not None:
        F_k_2d = F_k.reshape(out_dim, -1).to(device)
        F_k_2d = torch.log1p(F_k_2d / (F_k_2d.mean() + 1e-8))
    else:
        F_k_2d = torch.ones_like(W_k_2d).to(device)
        
    B = nn.Parameter(torch.zeros((out_dim, out_dim), device=device))
    optimizer = torch.optim.Adam([B], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        A = B - B.t()
        R = torch.matrix_exp(A)
        W_rot = R @ W_0_2d
        
        if weighted:
            loss = (F_k_2d * (W_k_2d - W_rot) ** 2).sum()
        else:
            loss = ((W_k_2d - W_rot) ** 2).sum()
            
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        A = B - B.t()
        R = torch.matrix_exp(A)
        W_rot = R @ W_0_2d
        E_2d = W_k_2d - W_rot
        E = E_2d.reshape(orig_shape).cpu()
        A_cpu = A.cpu()
        R_cpu = R.cpu()
        
    return A_cpu, R_cpu, E

def merge_models_task_arithmetic(base_encoder, experts, lambda_val):
    print(f"Merging models with Task Arithmetic (lambda = {lambda_val})...")
    merged_encoder = copy.deepcopy(base_encoder)
    merged_state = merged_encoder.state_dict()
    
    expert_states = [expert.state_dict() for expert in experts]
    base_state = base_encoder.state_dict()
    
    for name in merged_state:
        task_vectors_sum = torch.zeros_like(merged_state[name], dtype=torch.float32)
        for estate in expert_states:
            task_vectors_sum += estate[name].float() - base_state[name].float()
            
        merged_state[name] = base_state[name] + lambda_val * task_vectors_sum
        
    merged_encoder.load_state_dict(merged_state)
    return merged_encoder

def merge_models_dare(base_encoder, experts, lambda_val, drop_rate=0.2):
    print(f"Merging models with DARE (lambda = {lambda_val}, drop_rate = {drop_rate})...")
    merged_encoder = copy.deepcopy(base_encoder)
    merged_state = merged_encoder.state_dict()
    
    expert_states = [expert.state_dict() for expert in experts]
    base_state = base_encoder.state_dict()
    
    for name in merged_state:
        task_vectors_sum = torch.zeros_like(merged_state[name], dtype=torch.float32)
        for estate in expert_states:
            v = estate[name].float() - base_state[name].float()
            mask = (torch.rand_like(v) > drop_rate).float()
            v_dare = v * mask / (1.0 - drop_rate + 1e-8)
            task_vectors_sum += v_dare
            
        merged_state[name] = base_state[name] + lambda_val * task_vectors_sum
        
    merged_encoder.load_state_dict(merged_state)
    return merged_encoder

def merge_models_ties(base_encoder, experts, lambda_val, keep_rate=0.2):
    print(f"Merging models with TIES-Merging (lambda = {lambda_val}, keep_rate = {keep_rate})...")
    merged_encoder = copy.deepcopy(base_encoder)
    merged_state = merged_encoder.state_dict()
    
    expert_states = [expert.state_dict() for expert in experts]
    base_state = base_encoder.state_dict()
    
    for name in merged_state:
        trimmed_task_vectors = []
        for estate in expert_states:
            v = estate[name].float() - base_state[name].float()
            flat_v = v.flatten()
            k = int(keep_rate * flat_v.numel())
            if k > 0:
                threshold = torch.topk(flat_v.abs(), k).values[-1]
                mask = (v.abs() >= threshold).float()
                trimmed_v = v * mask
            else:
                trimmed_v = torch.zeros_like(v)
            trimmed_task_vectors.append(trimmed_v)
            
        stacked = torch.stack(trimmed_task_vectors, dim=0)
        signs = torch.sign(stacked)
        sum_signs = signs.sum(dim=0)
        majority_sign = torch.sign(sum_signs)
        
        matching_mask = (signs == majority_sign.unsqueeze(0)) & (majority_sign.unsqueeze(0) != 0)
        filtered_values = stacked * matching_mask.float()
        sum_filtered = filtered_values.sum(dim=0)
        num_matching = matching_mask.float().sum(dim=0)
        
        merged_task_vector = torch.where(num_matching > 0, sum_filtered / num_matching, torch.zeros_like(sum_filtered))
        merged_state[name] = base_state[name] + lambda_val * merged_task_vector
        
    merged_encoder.load_state_dict(merged_state)
    return merged_encoder

def merge_models_procrustes(base_encoder, experts, fishers, device, weighted=True, steps=150, lr=0.05, lambda_resid=None):
    method_name = "Fisher-Weighted Procrustes Alignment (FWPA)" if weighted else "Unweighted OrthoMerge"
    if lambda_resid is not None:
        print(f"Merging models with {method_name} (lambda_resid = {lambda_resid})...")
    else:
        print(f"Merging models with {method_name}...")
    
    merged_encoder = copy.deepcopy(base_encoder)
    merged_state = merged_encoder.state_dict()
    base_state = base_encoder.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    A_all = {name: [] for name in base_state}
    E_all = {name: [] for name in base_state}
    
    num_experts = len(experts)
    actual_lambda_resid = lambda_resid if lambda_resid is not None else (1.0 / num_experts)
    
    for name in base_state:
        is_weight = "weight" in name
        is_multidim = len(base_state[name].shape) >= 2
        
        if is_weight and is_multidim:
            print(f"  Processing layer rotation: {name}")
            for k in range(num_experts):
                W_0 = base_state[name].float()
                W_k = expert_states[k][name].float()
                F_k = fishers[k][name].float() if (weighted and fishers is not None) else None
                
                A_k, R_k, E_k = optimize_procrustes(W_0, W_k, F_k, steps=steps, lr=lr, weighted=weighted, device=device)
                A_all[name].append(A_k)
                E_all[name].append(E_k)
        else:
            print(f"  Processing layer linear-interpolation: {name}")
            for k in range(num_experts):
                A_all[name].append(None)
                E_all[name].append(expert_states[k][name].float() - base_state[name].float())
                
    for name in base_state:
        if A_all[name][0] is not None:
            A_merged = torch.zeros_like(A_all[name][0])
            for A_k in A_all[name]:
                A_merged += A_k
            A_merged /= num_experts
            
            E_merged = torch.zeros_like(E_all[name][0])
            for E_k in E_all[name]:
                E_merged += E_k
            E_merged *= actual_lambda_resid
            
            with torch.no_grad():
                R_merged = torch.matrix_exp(A_merged.to(device))
                W_0_2d = base_state[name].float().reshape(base_state[name].shape[0], -1).to(device)
                W_rot_merged = R_merged @ W_0_2d
                W_merged_2d = W_rot_merged.cpu() + E_merged.reshape(base_state[name].shape[0], -1)
                merged_state[name] = W_merged_2d.reshape(base_state[name].shape)
        else:
            E_merged = torch.zeros_like(E_all[name][0])
            for E_k in E_all[name]:
                E_merged += E_k
            E_merged *= actual_lambda_resid
            merged_state[name] = base_state[name] + E_merged
            
    merged_encoder.load_state_dict(merged_state)
    return merged_encoder

def merge_models_procrustes_weighted_fusion(base_encoder, experts, fishers, device, weighted=True, steps=150, lr=0.05, lambda_val=0.5, scale_rotation=True):
    method_name = "FWPA with Fisher-Weighted Fusion (FWPA-FWF)" if weighted else "OrthoMerge with Fisher-Weighted Fusion"
    print(f"Merging models with {method_name} (lambda = {lambda_val}, scale_rotation = {scale_rotation})...")
    
    merged_encoder = copy.deepcopy(base_encoder)
    merged_state = merged_encoder.state_dict()
    base_state = base_encoder.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    A_all = {name: [] for name in base_state}
    E_all = {name: [] for name in base_state}
    
    num_experts = len(experts)
    
    for name in base_state:
        is_weight = "weight" in name
        is_multidim = len(base_state[name].shape) >= 2
        
        if is_weight and is_multidim:
            print(f"  Processing layer rotation: {name}")
            for k in range(num_experts):
                W_0 = base_state[name].float()
                W_k = expert_states[k][name].float()
                F_k = fishers[k][name].float() if (weighted and fishers is not None) else None
                
                A_k, R_k, E_k = optimize_procrustes(W_0, W_k, F_k, steps=steps, lr=lr, weighted=weighted, device=device)
                A_all[name].append(A_k)
                E_all[name].append(E_k)
        else:
            print(f"  Processing layer linear-interpolation: {name}")
            for k in range(num_experts):
                A_all[name].append(None)
                E_all[name].append(expert_states[k][name].float() - base_state[name].float())
                
    for name in base_state:
        if weighted and fishers is not None:
            layer_fishers = [fishers[k][name].float() for k in range(num_experts)]
            layer_means = torch.tensor([lf.mean().item() for lf in layer_fishers])
            sum_means = layer_means.sum() + 1e-8
            layer_weights = layer_means / sum_means
            
            stacked_fishers = torch.stack(layer_fishers, dim=0)
            sum_fishers = stacked_fishers.sum(dim=0, keepdim=True) + 1e-8
            param_weights = stacked_fishers / sum_fishers
        else:
            layer_weights = torch.ones(num_experts) / num_experts
            param_weights = torch.ones((num_experts,) + base_state[name].shape) / num_experts
            
        if A_all[name][0] is not None:
            A_merged = torch.zeros_like(A_all[name][0])
            for k in range(num_experts):
                w_k = layer_weights[k] * num_experts
                A_merged += w_k * A_all[name][k]
                
            if scale_rotation:
                A_merged *= lambda_val
            else:
                A_merged /= num_experts
            
            E_merged = torch.zeros_like(E_all[name][0])
            for k in range(num_experts):
                w_k_p = param_weights[k] * num_experts
                E_merged += w_k_p * E_all[name][k]
            E_merged *= lambda_val
            
            with torch.no_grad():
                R_merged = torch.matrix_exp(A_merged.to(device))
                W_0_2d = base_state[name].float().reshape(base_state[name].shape[0], -1).to(device)
                W_rot_merged = R_merged @ W_0_2d
                W_merged_2d = W_rot_merged.cpu() + E_merged.reshape(base_state[name].shape[0], -1)
                merged_state[name] = W_merged_2d.reshape(base_state[name].shape)
        else:
            E_merged = torch.zeros_like(E_all[name][0])
            for k in range(num_experts):
                w_k_p = param_weights[k] * num_experts
                E_merged += w_k_p * E_all[name][k]
            E_merged *= lambda_val
            merged_state[name] = base_state[name] + E_merged
            
    merged_encoder.load_state_dict(merged_state)
    return merged_encoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize base MLP encoder
    base_encoder = MLPEncoder()
    
    # Train experts
    datasets = ["MNIST", "FashionMNIST", "KMNIST"]
    experts_encoders = []
    experts_heads = []
    
    for dname in datasets:
        enc, hd = train_expert(dname, base_encoder, device, epochs=3, lr=1e-3)
        experts_encoders.append(enc)
        experts_heads.append(hd)
        
    # Compute Fishers for each expert
    fishers = []
    for k, dname in enumerate(datasets):
        f = compute_fisher(dname, experts_encoders[k], experts_heads[k], device, num_samples=2000)
        fishers.append(f)
        
    # Evaluate experts
    print("\n=== Evaluating Experts (Individual Baseline) ===")
    for k, dname in enumerate(datasets):
        acc = evaluate_model(experts_encoders[k], experts_heads[k], dname, device)
        print(f"Expert {dname} on its own dataset: {acc:.2f}%")
        
    # Environments
    environments = [
        {"name": "Clean", "corruption": None},
        {"name": "Noise", "corruption": "noise"},
        {"name": "Blur", "corruption": "blur"},
        {"name": "Contrast", "corruption": "contrast"},
        {"name": "Rotation", "corruption": "rotation"}
    ]
    
    merging_strategies = {}
    
    for lamb in [0.33, 0.4, 0.5]:
        ta_encoder = merge_models_task_arithmetic(base_encoder, experts_encoders, lamb)
        merging_strategies[f"Task Arithmetic (λ={lamb})"] = ta_encoder
        
    for lamb in [0.33, 0.4, 0.5]:
        dare_encoder = merge_models_dare(base_encoder, experts_encoders, lamb, drop_rate=0.2)
        merging_strategies[f"DARE (λ={lamb})"] = dare_encoder

    for lamb in [0.33, 0.4, 0.5]:
        ties_encoder = merge_models_ties(base_encoder, experts_encoders, lamb, keep_rate=0.2)
        merging_strategies[f"TIES-Merging (λ={lamb})"] = ties_encoder

    for lamb in [0.33, 0.4, 0.5]:
        ortho_encoder = merge_models_procrustes(base_encoder, experts_encoders, fishers=None, device=device, weighted=False, lambda_resid=lamb)
        merging_strategies[f"Unweighted OrthoMerge (λ={lamb})"] = ortho_encoder
    
    for lamb in [0.33, 0.4, 0.5]:
        fw_ortho_encoder = merge_models_procrustes(base_encoder, experts_encoders, fishers=fishers, device=device, weighted=True, lambda_resid=lamb)
        merging_strategies[f"FWPA (Ours) (λ={lamb})"] = fw_ortho_encoder
        
    for lamb in [0.33, 0.4, 0.5]:
        fw_fwf_encoder = merge_models_procrustes_weighted_fusion(base_encoder, experts_encoders, fishers=fishers, device=device, weighted=True, lambda_val=lamb, scale_rotation=True)
        merging_strategies[f"FWPA-FWF (Ours) (λ={lamb})"] = fw_fwf_encoder

    # Results
    results = {}
    
    print("\n=== Beginning Multi-Task Evaluation ===")
    for strategy_name, encoder in merging_strategies.items():
        results[strategy_name] = {}
        print(f"\nEvaluating strategy: {strategy_name}")
        for env in environments:
            env_name = env["name"]
            results[strategy_name][env_name] = {}
            total_acc = 0.0
            for k, dname in enumerate(datasets):
                acc = evaluate_model(encoder, experts_heads[k], dname, device, corruption=env["corruption"])
                results[strategy_name][env_name][dname] = acc
                total_acc += acc
            avg_acc = total_acc / len(datasets)
            results[strategy_name][env_name]["Average"] = avg_acc
            print(f"  Env: {env_name:8s} | Average Multi-Task Accuracy: {avg_acc:.2f}%")
            
    # Save results to json
    with open("metrics_mlp.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults successfully saved to metrics_mlp.json.")
    
    generate_plot(results, environments)

def generate_plot(results, environments):
    env_names = [env["name"] for env in environments]
    strategies = list(results.keys())
    
    plt.figure(figsize=(12, 7))
    for strat in strategies:
        avg_accs = [results[strat][env]["Average"] for env in env_names]
        plt.plot(env_names, avg_accs, marker='o', linewidth=2.5, label=strat)
        
    plt.title("Multi-Task Merged MLP Model Accuracy Across Environments", fontsize=14, fontweight='bold')
    plt.xlabel("Environment (Clean & Corrupted)", fontsize=12)
    plt.ylabel("Average Multi-Task Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig("performance_comparison_mlp.png", dpi=300)
    print("Performance comparison plot saved as performance_comparison_mlp.png.")

if __name__ == "__main__":
    main()
