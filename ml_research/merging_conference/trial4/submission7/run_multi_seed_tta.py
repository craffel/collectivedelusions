import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import copy
import json

# Same CNN Architecture as in train_experts.py
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(3136, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class TaskExpert(nn.Module):
    def __init__(self, encoder):
        super(TaskExpert, self).__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features)
        return out

# Differentiable Merged Model
class DifferentiableMergedModel(nn.Module):
    def __init__(self, experts, fims=None, alpha=1.0, use_fim=True, epsilon=1e-6):
        super(DifferentiableMergedModel, self).__init__()
        self.experts = experts  # List of 3 TaskExpert models
        self.fims = fims        # List of 3 FIM dictionaries
        self.alpha = alpha      # Exponent/temperature for Fisher weighting
        self.use_fim = use_fim  # Whether to use FiT-Merge (True) or standard TTA (False)
        self.epsilon = epsilon  # Smoothing epsilon to prevent log(0) or extreme log values
        
        self.param_names = [
            "encoder.conv1.weight", "encoder.conv1.bias",
            "encoder.conv2.weight", "encoder.conv2.bias",
            "encoder.conv3.weight", "encoder.conv3.bias",
            "encoder.fc.weight", "encoder.fc.bias"
        ]
        
        # Trainable merging logits: [8 layers, 3 experts]
        self.logits = nn.Parameter(torch.zeros(len(self.param_names), 3))
        
        # Task classification heads (loaded from experts)
        self.heads = nn.ModuleList([nn.Linear(128, 10) for _ in range(3)])
        for k in range(3):
            self.heads[k].load_state_dict(experts[k].head.state_dict())
            
    def get_merged_param(self, param_idx, name, device):
        logits_i = self.logits[param_idx] # shape [3]
        
        if self.use_fim and self.fims is not None and name in self.fims[0]:
            log_fims = []
            for k in range(3):
                fim_k = self.fims[k][name].to(device)
                fim_k = fim_k / (torch.max(fim_k) + 1e-8)
                log_fim_k = torch.log(fim_k + self.epsilon)
                log_fims.append(log_fim_k)
            log_fims = torch.stack(log_fims, dim=0) # shape [3, ...]
            
            unsqueeze_dims = [1] * (log_fims.ndim - 1)
            logits_i_expanded = logits_i.view(3, *unsqueeze_dims)
            combined_logits = logits_i_expanded + self.alpha * log_fims
            merging_weights = torch.softmax(combined_logits, dim=0) # shape [3, ...]
            
            expert_params_stacked = torch.stack([
                dict(self.experts[k].named_parameters())[name].to(device) for k in range(3)
            ], dim=0)
            merged_param = torch.sum(merging_weights * expert_params_stacked, dim=0)
        else:
            merging_weights = torch.softmax(logits_i, dim=0) # shape [3]
            expert_params_stacked = torch.stack([
                dict(self.experts[k].named_parameters())[name].to(device) for k in range(3)
            ], dim=0)
            unsqueeze_dims = [1] * (expert_params_stacked.ndim - 1)
            merging_weights_expanded = merging_weights.view(3, *unsqueeze_dims)
            merged_param = torch.sum(merging_weights_expanded * expert_params_stacked, dim=0)
            
        return merged_param

    def forward(self, x, task_idx, device):
        w1 = self.get_merged_param(0, "encoder.conv1.weight", device)
        b1 = self.get_merged_param(1, "encoder.conv1.bias", device)
        w2 = self.get_merged_param(2, "encoder.conv2.weight", device)
        b2 = self.get_merged_param(3, "encoder.conv2.bias", device)
        w3 = self.get_merged_param(4, "encoder.conv3.weight", device)
        b3 = self.get_merged_param(5, "encoder.conv3.bias", device)
        w_fc = self.get_merged_param(6, "encoder.fc.weight", device)
        b_fc = self.get_merged_param(7, "encoder.fc.bias", device)
        
        x = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), kernel_size=2, stride=2)
        x = F.relu(F.conv2d(x, w2, b2, padding=1))
        x = F.max_pool2d(F.relu(F.conv2d(x, w3, b3, padding=1)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        features = F.relu(F.linear(x, w_fc, b_fc))
        out = self.heads[task_idx](features)
        return out

def get_raw_test_dataset(task_name):
    transform = transforms.ToTensor()
    if task_name == "MNIST":
        return datasets.MNIST("./data", train=False, download=True, transform=transform)
    elif task_name == "FashionMNIST":
        return datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    elif task_name == "KMNIST":
        return datasets.KMNIST("./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown task")

def corrupt_and_normalize(x, corruption_type, device):
    x = x.to(device)
    if corruption_type == "Clean":
        pass
    elif corruption_type == "Gaussian Noise":
        noise = torch.randn_like(x) * 0.4
        x = torch.clamp(x + noise, 0.0, 1.0)
    elif corruption_type == "Gaussian Blur":
        x = TF.gaussian_blur(x, [5, 5], [2.0, 2.0])
    elif corruption_type == "Contrast":
        x = torch.clamp(0.5 + 0.15 * (x - 0.5), 0.0, 1.0)
    
    return (x - 0.5) / 0.5

def build_test_stream(stream_type, batch_size=64, num_batches_per_task=50, seed=42):
    # Set seed specifically for stream generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    datasets_dict = {task: get_raw_test_dataset(task) for task in tasks}
    
    # Enable shuffle to get different batches for different seeds
    loaders = {
        task: DataLoader(datasets_dict[task], batch_size=batch_size, shuffle=True, drop_last=True)
        for task in tasks
    }
    iters = {task: iter(loaders[task]) for task in tasks}
    
    batches = []
    
    if stream_type == "Alternating":
        for b in range(num_batches_per_task):
            for task_idx, task in enumerate(tasks):
                try:
                    x, y = next(iters[task])
                    batches.append((x, y, task_idx))
                except StopIteration:
                    iters[task] = iter(loaders[task])
                    x, y = next(iters[task])
                    batches.append((x, y, task_idx))
    elif stream_type == "Sequential":
        for task_idx, task in enumerate(tasks):
            for b in range(num_batches_per_task):
                try:
                    x, y = next(iters[task])
                    batches.append((x, y, task_idx))
                except StopIteration:
                    iters[task] = iter(loaders[task])
                    x, y = next(iters[task])
                    batches.append((x, y, task_idx))
                    
    return batches

def run_evaluation(model_type, stream, environment, experts, fims, device, loss_mode="teacher-supervised", alpha=0.1, epsilon=1e-6):
    use_fim = (model_type in ['FiT-Merge (Ours)', 'Static Fisher'])
    eval_alpha = 1.0 if model_type == 'Static Fisher' else alpha
    model = DifferentiableMergedModel(experts, fims=fims, alpha=eval_alpha, use_fim=use_fim, epsilon=epsilon).to(device)
    
    if model_type not in ['Static', 'Static Fisher']:
        if loss_mode == "teacher-free":
            optimizer = optim.Adam([
                {'params': [model.logits], 'lr': 0.005}
            ])
            for p in model.heads.parameters():
                p.requires_grad = False
        else:
            optimizer = optim.Adam([
                {'params': [model.logits], 'lr': 0.005},
                {'params': model.heads.parameters(), 'lr': 0.05}
            ])
            for p in model.heads.parameters():
                p.requires_grad = True
    else:
        optimizer = None
        
    correct_predictions = 0
    total_samples = 0
    
    for step, (x_raw, y, task_idx) in enumerate(stream):
        x_raw, y = x_raw.to(device), y.to(device)
        x = corrupt_and_normalize(x_raw, environment, device)
        
        if model_type not in ['Static', 'Static Fisher']:
            model.train()
            optimizer.zero_grad()
            
            if loss_mode == "teacher-supervised":
                with torch.no_grad():
                    teacher_out = experts[task_idx](x)
                    teacher_probs = F.softmax(teacher_out / 1.0, dim=1)
                merged_out = model(x, task_idx, device)
                loss = F.kl_div(F.log_softmax(merged_out / 1.0, dim=1), teacher_probs, reduction='batchmean')
            elif loss_mode == "teacher-free":
                merged_out = model(x, task_idx, device)
                probs = F.softmax(merged_out, dim=1)
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))
                
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            output = model(x, task_idx, device)
            _, predicted = output.max(1)
            correct = predicted.eq(y).sum().item()
            correct_predictions += correct
            total_samples += y.size(0)
            
    avg_accuracy = 100.0 * correct_predictions / total_samples
    return avg_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running multi-seed TTA evaluation on device: {device}")
    
    # 1. Load experts
    print("Loading experts...")
    experts = []
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    for task in tasks:
        encoder = CNNEncoder()
        expert = TaskExpert(encoder).to(device)
        checkpoint_path = f"./checkpoints/expert_{task}.pt"
        expert.load_state_dict(torch.load(checkpoint_path, map_location=device))
        expert.eval()
        experts.append(expert)
        
    # 2. Load FIMs
    print("Loading FIMs...")
    fims = []
    for task in tasks:
        fim_path = f"./checkpoints/fim_{task}.pt"
        fim = torch.load(fim_path, map_location=device)
        fims.append(fim)
        
    environments = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    streams = ["Alternating", "Sequential"]
    methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
    loss_modes = ["teacher-supervised", "teacher-free"]
    
    seeds = [42, 43, 44, 45, 46]
    
    # Structure: results[loss_mode][stream_type][environment][method] -> list of accuracies
    raw_results = {lm: {st: {env: {m: [] for m in methods} for env in environments} for st in streams} for lm in loss_modes}
    
    for seed in seeds:
        print(f"\n=================== EVALUATING SEED {seed} ===================")
        for loss_mode in loss_modes:
            for stream_type in streams:
                # Build test stream for this seed
                stream = build_test_stream(stream_type, batch_size=64, num_batches_per_task=50, seed=seed)
                
                for env in environments:
                    for method in methods:
                        # Static methods only depend on the test data sequence, not optimization.
                        # However, we still evaluate them per seed because the stream inputs are shuffled.
                        acc = run_evaluation(method, stream, env, experts, fims, device, loss_mode)
                        raw_results[loss_mode][stream_type][env][method].append(acc)
                        print(f"Seed {seed} | {loss_mode:18s} | {stream_type:11s} | {env:14s} | {method:20s}: {acc:.2f}%")
                        
    # Compute mean and standard deviation
    aggregated_results = {}
    for loss_mode in loss_modes:
        aggregated_results[loss_mode] = {}
        for stream_type in streams:
            aggregated_results[loss_mode][stream_type] = {}
            for env in environments:
                aggregated_results[loss_mode][stream_type][env] = {}
                for method in methods:
                    accs = raw_results[loss_mode][stream_type][env][method]
                    mean = np.mean(accs)
                    std = np.std(accs)
                    aggregated_results[loss_mode][stream_type][env][method] = {
                        "mean": mean,
                        "std": std,
                        "raw": accs
                    }
                    
    # Save results to JSON
    with open("multi_seed_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=4)
    print("\nMulti-seed evaluation complete! Saved aggregated results to multi_seed_results.json")

if __name__ == "__main__":
    main()
