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

# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

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
        
        # Order of parameters in our custom functional forward pass
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
            # FiT-Merge (Ours): parameter-wise Fisher-weighted merging
            log_fims = []
            for k in range(3):
                fim_k = self.fims[k][name].to(device)
                # Normalize FIM to have maximum value of 1.0
                fim_k = fim_k / (torch.max(fim_k) + 1e-8)
                # Ensure no log(0) by adding a small epsilon
                log_fim_k = torch.log(fim_k + self.epsilon)
                log_fims.append(log_fim_k)
            log_fims = torch.stack(log_fims, dim=0) # shape [3, ...]
            
            # Reshape logits_i to expand across other dimensions of the parameter tensor
            unsqueeze_dims = [1] * (log_fims.ndim - 1)
            logits_i_expanded = logits_i.view(3, *unsqueeze_dims)
            
            # Combine logits and Fisher importances
            combined_logits = logits_i_expanded + self.alpha * log_fims
            
            # Softmax over expert dimension
            merging_weights = torch.softmax(combined_logits, dim=0) # shape [3, ...]
            
            # Average expert parameters
            expert_params_stacked = torch.stack([
                dict(self.experts[k].named_parameters())[name].to(device) for k in range(3)
            ], dim=0)
            
            merged_param = torch.sum(merging_weights * expert_params_stacked, dim=0)
        else:
            # Standard dynamic TTA or Static merged: uniform / simple softmax over experts
            merging_weights = torch.softmax(logits_i, dim=0) # shape [3]
            expert_params_stacked = torch.stack([
                dict(self.experts[k].named_parameters())[name].to(device) for k in range(3)
            ], dim=0)
            
            # Reshape merging_weights for broadcasting
            unsqueeze_dims = [1] * (expert_params_stacked.ndim - 1)
            merging_weights_expanded = merging_weights.view(3, *unsqueeze_dims)
            merged_param = torch.sum(merging_weights_expanded * expert_params_stacked, dim=0)
            
        return merged_param

    def forward(self, x, task_idx, device):
        # Dynamically compute merged weights for the encoder layers
        w1 = self.get_merged_param(0, "encoder.conv1.weight", device)
        b1 = self.get_merged_param(1, "encoder.conv1.bias", device)
        w2 = self.get_merged_param(2, "encoder.conv2.weight", device)
        b2 = self.get_merged_param(3, "encoder.conv2.bias", device)
        w3 = self.get_merged_param(4, "encoder.conv3.weight", device)
        b3 = self.get_merged_param(5, "encoder.conv3.bias", device)
        w_fc = self.get_merged_param(6, "encoder.fc.weight", device)
        b_fc = self.get_merged_param(7, "encoder.fc.bias", device)
        
        # Forward pass using functional operations
        x = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), kernel_size=2, stride=2)
        x = F.relu(F.conv2d(x, w2, b2, padding=1))
        x = F.max_pool2d(F.relu(F.conv2d(x, w3, b3, padding=1)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        features = F.relu(F.linear(x, w_fc, b_fc))
        
        # Pass features through task-specific classification head
        out = self.heads[task_idx](features)
        return out

def get_raw_test_dataset(task_name):
    # Load dataset without normalization to allow corruptions first
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
    
    # Normalize with mean=0.5, std=0.5
    return (x - 0.5) / 0.5

def build_test_stream(stream_type, batch_size=64, num_batches_per_task=50):
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    datasets_dict = {task: get_raw_test_dataset(task) for task in tasks}
    
    # Simple iterators
    loaders = {
        task: DataLoader(datasets_dict[task], batch_size=batch_size, shuffle=False, drop_last=True)
        for task in tasks
    }
    iters = {task: iter(loaders[task]) for task in tasks}
    
    batches = []
    
    if stream_type == "Alternating":
        # Rapid switching: Task 0, Task 1, Task 2, Task 0...
        for b in range(num_batches_per_task):
            for task_idx, task in enumerate(tasks):
                try:
                    x, y = next(iters[task])
                    batches.append((x, y, task_idx))
                except StopIteration:
                    # Reset iterator if we run out of data
                    iters[task] = iter(loaders[task])
                    x, y = next(iters[task])
                    batches.append((x, y, task_idx))
    elif stream_type == "Sequential":
        # Block shift: all Task 0, then all Task 1, then all Task 2
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
    """
    model_type: 'Static', 'Static Fisher', 'Standard TTA', 'FiT-Merge (Ours)'
    stream: list of (x_raw, y, task_idx)
    environment: 'Clean', 'Gaussian Noise', 'Gaussian Blur', 'Contrast'
    loss_mode: 'teacher-supervised' (KL loss) or 'teacher-free' (Entropy loss)
    """
    # Create the model with soft Fisher exponent alpha
    use_fim = (model_type in ['FiT-Merge (Ours)', 'Static Fisher'])
    eval_alpha = 1.0 if model_type == 'Static Fisher' else alpha
    model = DifferentiableMergedModel(experts, fims=fims, alpha=eval_alpha, use_fim=use_fim, epsilon=epsilon).to(device)
    
    # Set up optimizer
    if model_type not in ['Static', 'Static Fisher']:
        # Optimize merging coefficients (logits) and classification heads
        # In teacher-free mode, freeze task classification heads to prevent decision boundary collapse (as in S2C-Merge)
        if loss_mode == "teacher-free":
            optimizer = optim.Adam([
                {'params': [model.logits], 'lr': 0.005}
            ])
            # Freeze task classification heads
            for p in model.heads.parameters():
                p.requires_grad = False
        else:
            optimizer = optim.Adam([
                {'params': [model.logits], 'lr': 0.005},
                {'params': model.heads.parameters(), 'lr': 0.05}
            ])
            # Ensure task classification heads require gradients
            for p in model.heads.parameters():
                p.requires_grad = True
    else:
        optimizer = None
        
    correct_predictions = 0
    total_samples = 0
    
    # For recording batch-by-batch accuracy for plotting
    batch_accuracies = []
    
    for step, (x_raw, y, task_idx) in enumerate(stream):
        x_raw, y = x_raw.to(device), y.to(device)
        
        # Apply corruption and normalization
        x = corrupt_and_normalize(x_raw, environment, device)
        
        # Test-Time Adaptation Step
        if model_type not in ['Static', 'Static Fisher']:
            model.train() # Enable gradients
            optimizer.zero_grad()
            
            if loss_mode == "teacher-supervised":
                # Compute soft labels using original expert (teacher)
                with torch.no_grad():
                    teacher_out = experts[task_idx](x)
                    teacher_probs = F.softmax(teacher_out / 1.0, dim=1)
                
                # Forward pass on merged model
                merged_out = model(x, task_idx, device)
                loss = F.kl_div(F.log_softmax(merged_out / 1.0, dim=1), teacher_probs, reduction='batchmean')
            elif loss_mode == "teacher-free":
                # Predict entropy minimization
                merged_out = model(x, task_idx, device)
                probs = F.softmax(merged_out, dim=1)
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))
            else:
                raise ValueError("Unknown loss mode")
                
            loss.backward()
            optimizer.step()
            
        # Inference and Evaluation on the same batch (active/on-the-fly TTA)
        model.eval()
        with torch.no_grad():
            output = model(x, task_idx, device)
            _, predicted = output.max(1)
            correct = predicted.eq(y).sum().item()
            acc = 100.0 * correct / y.size(0)
            batch_accuracies.append(acc)
            
            correct_predictions += correct
            total_samples += y.size(0)
            
    avg_accuracy = 100.0 * correct_predictions / total_samples
    return avg_accuracy, batch_accuracies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TTA evaluation on device: {device}")
    
    # 1. Load trained expert models
    print("Loading experts...")
    experts = []
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    for task in tasks:
        # Load encoder
        encoder = CNNEncoder()
        expert = TaskExpert(encoder).to(device)
        checkpoint_path = f"./checkpoints/expert_{task}.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expert model not found at {checkpoint_path}. Please run train_experts.py first.")
        expert.load_state_dict(torch.load(checkpoint_path, map_location=device))
        expert.eval()
        experts.append(expert)
        
    # 2. Load Fisher Information Matrices
    print("Loading FIMs...")
    fims = []
    for task in tasks:
        fim_path = f"./checkpoints/fim_{task}.pt"
        if not os.path.exists(fim_path):
            raise FileNotFoundError(f"FIM not found at {fim_path}. Please run train_experts.py first.")
        fim = torch.load(fim_path, map_location=device)
        fims.append(fim)
        
    environments = ["Clean", "Gaussian Noise", "Gaussian Blur", "Contrast"]
    streams = ["Alternating", "Sequential"]
    methods = ["Static", "Static Fisher", "Standard TTA", "FiT-Merge (Ours)"]
    
    # Run evaluation under both teacher-supervised and teacher-free loss modes
    loss_modes = ["teacher-supervised", "teacher-free"]
    
    results = {}
    
    for loss_mode in loss_modes:
        print(f"\n=================== LOSS MODE: {loss_mode.upper()} ===================")
        results[loss_mode] = {}
        for stream_type in streams:
            print(f"\n--- Stream Type: {stream_type} ---")
            results[loss_mode][stream_type] = {}
            
            # Build the stream
            print("Building test stream...")
            stream = build_test_stream(stream_type, batch_size=64, num_batches_per_task=50)
            print(f"Stream built: total of {len(stream)} batches.")
            
            for env in environments:
                print(f"\nDomain Shift / Corruption: {env}")
                results[loss_mode][stream_type][env] = {}
                for method in methods:
                    # For Static methods, loss_mode is irrelevant, so we only run it once
                    if method in ["Static", "Static Fisher"] and loss_mode == "teacher-free":
                        results[loss_mode][stream_type][env][method] = results["teacher-supervised"][stream_type][env][method]
                        continue
                        
                    acc, batch_accs = run_evaluation(method, stream, env, experts, fims, device, loss_mode)
                    results[loss_mode][stream_type][env][method] = {
                        "accuracy": acc,
                        "batch_accuracies": batch_accs
                    }
                    print(f"  {method:20s}: {acc:.2f}%")
                    
    # Save the evaluation results
    import json
    # Convert numpy types to native types for JSON serialization
    def convert_to_serializable(d):
        if isinstance(d, dict):
            return {k: convert_to_serializable(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_to_serializable(v) for v in d]
        elif isinstance(d, np.float32) or isinstance(d, np.float64):
            return float(d)
        else:
            return d
            
    with open("tta_results.json", "w") as f:
        json.dump(convert_to_serializable(results), f, indent=4)
    print("\nAll evaluations complete! Results saved to tta_results.json")

if __name__ == "__main__":
    main()
