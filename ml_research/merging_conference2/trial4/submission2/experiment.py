import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import copy

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on some cluster nodes
    torch.backends.cudnn.enabled = False

# Dataset utility
def get_datasets(data_dir="./data"):
    # ResNet-18 expects 3-channel images, so we resize and replicate channels for MNIST/Fashion-MNIST
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_fashion = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_mnist = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_mnist)

    train_fashion = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_fashion)
    test_fashion = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_fashion)

    train_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_cifar)
    test_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_cifar)

    return {
        "mnist": (train_mnist, test_mnist),
        "fashion": (train_fashion, test_fashion),
        "cifar": (train_cifar, test_cifar)
    }

# Helper to get calibration subset
def get_calibration_subset(dataset, N, seed=42):
    set_seed(seed)
    indices = list(range(len(dataset)))
    selected_indices = random.sample(indices, N)
    return Subset(dataset, selected_indices)

# Helper to add noise/corruption to a dataset
class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, corruption_type="gaussian", severity=0.0):
        self.dataset = dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.severity > 0.0:
            if self.corruption_type == "gaussian":
                noise = torch.randn_like(image) * self.severity
                image = image + noise
            elif self.corruption_type == "salt_pepper":
                image = image.clone()
                mask = torch.rand_like(image)
                min_val = image.min()
                max_val = image.max()
                image[mask < self.severity / 2.0] = min_val
                image[mask > 1.0 - self.severity / 2.0] = max_val
            elif self.corruption_type == "blur":
                from torchvision.transforms.functional import gaussian_blur
                # severity maps to different kernel sizes & sigmas
                if self.severity <= 0.15:
                    image = gaussian_blur(image, [3, 3], 0.5)
                elif self.severity <= 0.25:
                    image = gaussian_blur(image, [5, 5], 1.0)
                else:
                    image = gaussian_blur(image, [7, 7], 1.5)
        return image, label

    def __len__(self):
        return len(self.dataset)

# Expert training function
def train_experts(data_dir="./data", save_dir="./weights", epochs=5):
    os.makedirs(save_dir, exist_ok=True)
    datasets = get_datasets(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Save pretrained base model
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    torch.save(base_model.state_dict(), os.path.join(save_dir, "resnet18_pretrained.pt"))
    print("Pretrained base model saved.")

    for task, (train_set, test_set) in datasets.items():
        print(f"\n--- Training Expert for task: {task} ---")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 10)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            model.train()
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

            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * correct / total
        print(f"Test Accuracy for {task}: {test_acc:.2f}%")

        # Save expert
        torch.save(model.state_dict(), os.path.join(save_dir, f"resnet18_{task}.pt"))
        print(f"Expert model for {task} saved.")

# Model Merging Function
def merge_models(save_dir="./weights", method="wa", lambda_val=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained base W0
    base_state = torch.load(os.path.join(save_dir, "resnet18_pretrained.pt"), map_location=device)
    
    tasks = ["mnist", "fashion", "cifar"]
    expert_states = {}
    for task in tasks:
        expert_states[task] = torch.load(os.path.join(save_dir, f"resnet18_{task}.pt"), map_location=device)

    # Initialize merged model with backbone from merged and heads from original experts
    # Heads are not merged, they remain task-specific
    merged_model = resnet18()
    merged_model.fc = nn.Linear(512, 10) # Placeholder
    merged_state = copy.deepcopy(base_state) # We will replace backbone weights

    # Identify backbone keys (everything except fc)
    backbone_keys = [k for k in base_state.keys() if not k.startswith("fc.")]

    if method == "wa":
        print("Merging via Weight Averaging (WA)...")
        for key in backbone_keys:
            # WA = sum(expert_keys) / 3
            merged_state[key] = (expert_states["mnist"][key] + 
                                 expert_states["fashion"][key] + 
                                 expert_states["cifar"][key]) / 3.0
    elif method == "ta":
        print(f"Merging via Task Arithmetic (TA) with lambda={lambda_val}...")
        for key in backbone_keys:
            # TA = W0 + lambda * sum(expert - W0)
            task_vector_sum = torch.zeros_like(base_state[key])
            for task in tasks:
                task_vector_sum += (expert_states[task][key] - base_state[key])
            merged_state[key] = base_state[key] + lambda_val * task_vector_sum

    # Load merged backbone into merged model
    # Note: fc remains a placeholder; when evaluating task k, we will swap in expert k's fc layer!
    if "fc.weight" in merged_state:
        del merged_state["fc.weight"]
    if "fc.bias" in merged_state:
        del merged_state["fc.bias"]
    merged_model.load_state_dict(merged_state, strict=False)
    
    # Create task-specific heads dictionary
    heads = {}
    for task in tasks:
        # Extract fc.weight and fc.bias from expert state
        fc_weight = expert_states[task]["fc.weight"]
        fc_bias = expert_states[task]["fc.bias"]
        heads[task] = (fc_weight, fc_bias)

    return merged_model, heads

# Get all BatchNorm2d layers of the model in feedforward order
def get_bn_layers(model):
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers

# Dynamic Calibration Classes & Functions

class CalibrationHook:
    def __init__(self, module):
        self.module = module
        self.scale = None  # torch.Tensor (C,)
        self.bias = None   # torch.Tensor (C,)
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if self.scale is not None and self.bias is not None:
            s = self.scale.view(1, -1, 1, 1).to(output.device)
            b = self.bias.view(1, -1, 1, 1).to(output.device)
            return s * output + b
        return output

    def remove(self):
        self.hook_handle.remove()

class SPTAACHook:
    def __init__(self, module):
        self.module = module
        self.gamma = None  # Scalar
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if self.gamma is not None:
            return self.gamma * output
        return output

    def remove(self):
        self.hook_handle.remove()

# Helper to pass data and capture output of a single BN layer
def capture_bn_activation(model, dataloader, layer_module, device):
    activations = []
    def cap_hook(module, input, output):
        activations.append(output.detach().cpu())

    handle = layer_module.register_forward_hook(cap_hook)
    
    # Forward pass on a few batches
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            model(images)
            break # Just 1 batch is enough for the calibration set
            
    handle.remove()
    return torch.cat(activations, dim=0)

# Sequential Calibration Collection (TCAC)
def collect_tcac_sequential(merged_model, heads, experts_dir, datasets, N, device, eps=1e-5):
    print(f"Collecting TCAC Sequential Calibration statistics (N={N})...")
    tasks = ["mnist", "fashion", "cifar"]
    
    # Store calibration parameters per task: {task: {layer_name: (scale, bias)}}
    tcac_params = {task: {} for task in tasks}
    
    # For each task, perform sequential collection
    for task in tasks:
        # Load full expert model
        expert = resnet18()
        expert.fc = nn.Linear(512, 10)
        expert.load_state_dict(torch.load(os.path.join(experts_dir, f"resnet18_{task}.pt"), map_location=device))
        expert = expert.to(device)
        expert.eval()

        # Get calibration subset for this task
        cal_subset = get_calibration_subset(datasets[task][0], N)
        cal_loader = DataLoader(cal_subset, batch_size=N, shuffle=False)

        # Clone merged model and set up task head
        task_merged = copy.deepcopy(merged_model).to(device)
        task_merged.fc.weight.data.copy_(heads[task][0])
        task_merged.fc.bias.data.copy_(heads[task][1])
        task_merged.eval()

        # Get BN layers for both models
        expert_bn = get_bn_layers(expert)
        merged_bn = get_bn_layers(task_merged)
        
        # Instantiate calibration hooks on task_merged
        merged_hooks = {name: CalibrationHook(module) for name, module in merged_bn}

        # Sequence layer-by-layer
        for idx, (name, module) in enumerate(merged_bn):
            expert_module = expert_bn[idx][1]

            # 1. Capture expert activation at this layer
            expert_act = capture_bn_activation(expert, cal_loader, expert_module, device)
            mu_target = expert_act.mean(dim=(0, 2, 3))
            std_target = expert_act.std(dim=(0, 2, 3))

            # 2. Capture merged activation at this layer (all preceding calibrations are active!)
            merged_act = capture_bn_activation(task_merged, cal_loader, module, device)
            mu_merged = merged_act.mean(dim=(0, 2, 3))
            std_merged = merged_act.std(dim=(0, 2, 3))

            # 3. Compute scaling and bias
            scale = std_target / (std_merged + eps)
            bias = mu_target - scale * mu_merged

            # 4. Set parameters in active hook so subsequent layers receive calibrated input
            merged_hooks[name].scale = scale
            merged_hooks[name].bias = bias

            # Store parameters
            tcac_params[task][name] = (scale, bias)

        # Cleanup hooks
        for hook in merged_hooks.values():
            hook.remove()

    return tcac_params

# SP-TAAC Calibration Collection
def collect_sp_taac(merged_model, heads, experts_dir, datasets, N, device, eps=1e-5):
    print(f"Collecting SP-TAAC Calibration statistics (N={N})...")
    tasks = ["mnist", "fashion", "cifar"]
    
    # 1. Get calibration subsets and loader for joint calibration
    joint_cal_samples = []
    for task in tasks:
        joint_cal_samples.append(get_calibration_subset(datasets[task][0], N))
    
    # Create joint dataset
    from torch.utils.data import ConcatDataset
    joint_dataset = ConcatDataset(joint_cal_samples)
    joint_loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)

    # 2. Load experts to compute target global layer standard deviation
    experts = {}
    for task in tasks:
        exp = resnet18()
        exp.fc = nn.Linear(512, 10)
        exp.load_state_dict(torch.load(os.path.join(experts_dir, f"resnet18_{task}.pt"), map_location=device))
        exp = exp.to(device).eval()
        experts[task] = exp

    # Get BN layers
    merged_bn = get_bn_layers(merged_model)
    
    sp_taac_params = {}

    for idx, (name, module) in enumerate(merged_bn):
        # Compute average expert standard deviation globally
        std_expert_list = []
        for task in tasks:
            exp_module = get_bn_layers(experts[task])[idx][1]
            # Use task-specific loader for expert
            task_loader = DataLoader(joint_cal_samples[tasks.index(task)], batch_size=N, shuffle=False)
            exp_act = capture_bn_activation(experts[task], task_loader, exp_module, device)
            # Global standard deviation across all batch, channel, spatial dimensions
            global_std = torch.sqrt(exp_act.var() + eps)
            std_expert_list.append(global_std.item())
        
        std_target = sum(std_expert_list) / len(tasks)

        # Compute merged model standard deviation on joint loader (with correct head, doesn't matter for BN activations)
        # Setup merged model task-specific head (using MNIST fc as default since we only capture backbone activations)
        task_merged = copy.deepcopy(merged_model).to(device)
        task_merged.fc.weight.data.copy_(heads["mnist"][0])
        task_merged.fc.bias.data.copy_(heads["mnist"][1])
        task_merged.eval()
        
        merged_module = get_bn_layers(task_merged)[idx][1]
        merged_act = capture_bn_activation(task_merged, joint_loader, merged_module, device)
        std_merged = torch.sqrt(merged_act.var() + eps)

        # Compute gamma
        gamma = std_target / std_merged.item()
        sp_taac_params[name] = gamma

    return sp_taac_params

# BatchNorm Adaptation Functions

def run_bn_adaptation(model, calibration_loader, device, momentum=1.0, epochs=1):
    # Set model to train mode, freeze weights, and run passes to update running stats
    model.train()
    # Freeze all weights
    for param in model.parameters():
        param.requires_grad = False
        
    # Set momentum for BN layers
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum
            module.reset_running_stats() # Clear current stats to re-estimate completely

    with torch.no_grad():
        for ep in range(epochs):
            for images, _ in calibration_loader:
                images = images.to(device)
                model(images)
                if momentum == 1.0 and epochs == 1:
                    break # A single batch is enough for momentum=1.0

    model.eval()

# Main Evaluation Loop
def evaluate_model(model, task, test_dataset, corruption_type, severity, device):
    model.eval()
    # Use clean test_dataset directly to optimize DataLoader speed (running on CPU)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Apply corruption on GPU batch on the fly
            if severity > 0.0:
                if corruption_type == "gaussian":
                    noise = torch.randn_like(images) * severity
                    images = images + noise
                elif corruption_type == "salt_pepper":
                    images = images.clone()
                    mask = torch.rand_like(images)
                    min_val = images.min()
                    max_val = images.max()
                    images[mask < severity / 2.0] = min_val
                    images[mask > 1.0 - severity / 2.0] = max_val
                elif corruption_type == "blur":
                    from torchvision.transforms.functional import gaussian_blur
                    if severity <= 0.15:
                        images = gaussian_blur(images, [3, 3], 0.5)
                    elif severity <= 0.25:
                        images = gaussian_blur(images, [5, 5], 1.0)
                    else:
                        images = gaussian_blur(images, [7, 7], 1.5)
                        
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

# Master Experiment Suite
def run_experiments(data_dir="./data", weights_dir="./weights"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on: {device}")
    
    datasets = get_datasets(data_dir)
    tasks = ["mnist", "fashion", "cifar"]
    
    results = []
    if os.path.exists("merging_robustness_results.csv"):
        try:
            import pandas as pd
            df_existing = pd.read_csv("merging_robustness_results.csv")
            results = df_existing.to_dict("records")
            print(f"Loaded {len(results)} existing result rows from merging_robustness_results.csv")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")

    # Helper to check if configuration already done
    def is_done(merge_method, N, corruption, severity):
        if not results:
            return False
        tasks_found = [r for r in results if r["Merge"] == merge_method and r["N"] == N and r["Corruption"] == corruption and abs(r["Severity"] - severity) < 1e-4]
        return len(tasks_found) >= 3  # All 3 tasks done

    # Let's run for both WA and TA merging
    for merge_method in ["wa", "ta"]:
        print(f"\n=======================================================")
        print(f"=== Merging Method: {merge_method.upper()} ===")
        print(f"=======================================================")
        
        # Load merged model base
        base_merged_model, heads = merge_models(weights_dir, method=merge_method, lambda_val=0.3)
        
        # Sweep calibration subset size N
        for N in [16, 64, 128]:
            print(f"\n--- Running sweeps for Calibration Budget N = {N} ---")
            
            # 1. Collect Calibration Parameters
            tcac_params = collect_tcac_sequential(base_merged_model, heads, weights_dir, datasets, N, device)
            sp_taac_params = collect_sp_taac(base_merged_model, heads, weights_dir, datasets, N, device)
            
            # Prepare Joint and Task-specific Calibration Loaders for BN Adaptation
            joint_cal_samples = [get_calibration_subset(datasets[t][0], N) for t in tasks]
            from torch.utils.data import ConcatDataset
            joint_dataset = ConcatDataset(joint_cal_samples)
            joint_loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)
            
            task_loaders = {}
            for t in tasks:
                t_subset = get_calibration_subset(datasets[t][0], N)
                task_loaders[t] = DataLoader(t_subset, batch_size=N, shuffle=False)

            # Sweep Out-of-Distribution Corruptions
            for corruption in ["gaussian", "salt_pepper", "blur"]:
                for severity in [0.0, 0.1, 0.2, 0.3]:
                    # Skip redundant evaluation at severity 0.0
                    if severity == 0.0 and corruption != "gaussian":
                        continue
                        
                    if is_done(merge_method, N, corruption, severity):
                        print(f"  Skipping already completed configuration: {merge_method.upper()} | N={N} | {corruption} | Severity: {severity:.1f}")
                        continue
                        
                    print(f"  Evaluating at Corruption: {corruption} | Severity: {severity:.1f}")
                    
                    # --- Method 1: Uncalibrated Baseline ---
                    uncal_accs = {}
                    for task in tasks:
                        model = copy.deepcopy(base_merged_model).to(device)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        uncal_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)
                    
                    # --- Method 2: TCAC Sequential Pre-ReLU (Task-Conditional) ---
                    tcac_accs = {}
                    for task in tasks:
                        model = copy.deepcopy(base_merged_model).to(device)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        
                        # Register and load calibration parameters
                        bn_layers = get_bn_layers(model)
                        hooks = []
                        for name, module in bn_layers:
                            hook = CalibrationHook(module)
                            hook.scale = tcac_params[task][name][0]
                            hook.bias = tcac_params[task][name][1]
                            hooks.append(hook)
                        
                        tcac_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)
                        
                        # Cleanup
                        for hook in hooks:
                            hook.remove()

                    # --- Method 3: SP-TAAC (Task-Agnostic) ---
                    sp_taac_accs = {}
                    for task in tasks:
                        model = copy.deepcopy(base_merged_model).to(device)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        
                        # Register and load SP-TAAC parameters
                        bn_layers = get_bn_layers(model)
                        hooks = []
                        for name, module in bn_layers:
                            hook = SPTAACHook(module)
                            hook.gamma = sp_taac_params[name]
                            hooks.append(hook)
                            
                        sp_taac_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)
                        
                        # Cleanup
                        for hook in hooks:
                            hook.remove()

                    # --- Method 4: Joint BN Adaptation (Task-Agnostic) ---
                    joint_bn_accs = {}
                    # Update BN on joint loader
                    model_joint_bn = copy.deepcopy(base_merged_model).to(device)
                    run_bn_adaptation(model_joint_bn, joint_loader, device, momentum=1.0, epochs=1)
                    for task in tasks:
                        model = copy.deepcopy(model_joint_bn)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        joint_bn_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)

                    # --- Method 5: Task-Conditional BN Adaptation (TC-BN-Adapt) ---
                    tc_bn_accs = {}
                    for task in tasks:
                        model = copy.deepcopy(base_merged_model).to(device)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        # Update BN on task-specific calibration set
                        run_bn_adaptation(model, task_loaders[task], device, momentum=1.0, epochs=1)
                        tc_bn_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)

                    # --- Method 6: Joint BN Adaptation Iterative (Task-Agnostic, momentum=0.1, epochs=5) ---
                    joint_bn_iter_accs = {}
                    model_joint_bn_iter = copy.deepcopy(base_merged_model).to(device)
                    # We need a shuffled loader for multi-pass iterative adaptation
                    joint_loader_shuffled = DataLoader(joint_dataset, batch_size=32, shuffle=True)
                    run_bn_adaptation(model_joint_bn_iter, joint_loader_shuffled, device, momentum=0.1, epochs=5)
                    for task in tasks:
                        model = copy.deepcopy(model_joint_bn_iter)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        joint_bn_iter_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)

                    # --- Method 7: Task-Conditional BN Adaptation Iterative (TC-BN-Adapt-Iter, momentum=0.1, epochs=5) ---
                    tc_bn_iter_accs = {}
                    for task in tasks:
                        model = copy.deepcopy(base_merged_model).to(device)
                        model.fc.weight.data.copy_(heads[task][0])
                        model.fc.bias.data.copy_(heads[task][1])
                        t_subset = get_calibration_subset(datasets[task][0], N)
                        tc_loader_shuffled = DataLoader(t_subset, batch_size=16, shuffle=True)
                        run_bn_adaptation(model, tc_loader_shuffled, device, momentum=0.1, epochs=5)
                        tc_bn_iter_accs[task] = evaluate_model(model, task, datasets[task][1], corruption, severity, device)

                    # Store result row
                    for task in tasks:
                        results.append({
                            "Merge": merge_method,
                            "N": N,
                            "Corruption": corruption,
                            "Severity": severity,
                            "Task": task,
                            "Uncalibrated": uncal_accs[task],
                            "TCAC": tcac_accs[task],
                            "SP-TAAC": sp_taac_accs[task],
                            "Joint-BN-Adapt": joint_bn_accs[task],
                            "TC-BN-Adapt": tc_bn_accs[task],
                            "Joint-BN-Adapt-Iter": joint_bn_iter_accs[task],
                            "TC-BN-Adapt-Iter": tc_bn_iter_accs[task]
                        })
                        
                    # Save checkpoint results on the fly
                    import pandas as pd
                    pd.DataFrame(results).to_csv("merging_robustness_results.csv", index=False)
                        
                    # Print current block
                    print(f"    Task | Uncal | TCAC | SP-TAAC | Joint-BN | TC-BN | Joint-BN-Iter | TC-BN-Iter")
                    for task in tasks:
                        print(f"    {task:5s} | {uncal_accs[task]:5.2f}% | {tcac_accs[task]:5.2f}% | {sp_taac_accs[task]:5.2f}% | {joint_bn_accs[task]:5.2f}% | {tc_bn_accs[task]:5.2f}% | {joint_bn_iter_accs[task]:5.2f}% | {tc_bn_iter_accs[task]:5.2f}%")

    # Compile results into markdown table or save to a file
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("merging_robustness_results.csv", index=False)
    print("\nExperiments complete. Results saved to merging_robustness_results.csv")
    
    # Generate a nice summary
    summary_df = df.groupby(["Merge", "N", "Corruption", "Severity"])[["Uncalibrated", "TCAC", "SP-TAAC", "Joint-BN-Adapt", "TC-BN-Adapt", "Joint-BN-Adapt-Iter", "TC-BN-Adapt-Iter"]].mean().reset_index()
    print("\n--- AVERAGE OVER ALL TASKS ---")
    print(summary_df.to_string(index=False))
    
    # Save a nice markdown summary of results to be included in the paper
    with open("results_summary.md", "w") as f:
        f.write("# Robustness & Generalization Experimental Results\n\n")
        f.write("## Average Accuracy Across All Tasks (MNIST, Fashion-MNIST, CIFAR-10)\n\n")
        f.write(summary_df.to_markdown(index=False))

# Main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness and Generalization of Activation Calibration in Model Merging")
    parser.add_argument("--action", type=str, default="run_all", choices=["train", "eval", "run_all"], help="Action to run")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train experts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    if args.action == "train":
        train_experts(epochs=args.epochs)
    elif args.action == "eval":
        run_experiments()
    elif args.action == "run_all":
        train_experts(epochs=args.epochs)
        run_experiments()
