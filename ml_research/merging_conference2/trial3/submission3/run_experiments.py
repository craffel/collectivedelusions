import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configurations
BATCH_SIZE = 128
EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Dataset Loading & Preparation
def get_datasets():
    # Transforms for MNIST (Grayscale 1-channel -> 3-channel, 32x32)
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Transforms for Fashion-MNIST (Grayscale 1-channel -> 3-channel, 32x32)
    fashion_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Transforms for CIFAR-10 (3-channel, 32x32)
    cifar_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load datasets
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=fashion_transform)
    test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=fashion_transform)

    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

    return {
        'mnist': (train_mnist, test_mnist),
        'fashion': (train_fashion, test_fashion),
        'cifar': (train_cifar, test_cifar)
    }

# 2. Model definition helper
def get_resnet18_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # The default resnet18 has 512 out features in fc, we will replace fc with task-specific head
    model.fc = nn.Linear(512, 10)
    return model

# Train an expert model on a specific task
def train_expert(task_name, train_dataset, test_dataset):
    checkpoint_path = f"checkpoints/{task_name}.pt"
    model = get_resnet18_model().to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for {task_name}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        test_acc = evaluate_model(model, test_dataset)
        print(f"{task_name} loaded, Test Acc: {test_acc:.2f}%")
        return model, test_acc

    print(f"Training expert for {task_name}...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

    test_acc = evaluate_model(model, test_dataset)
    print(f"Finished training {task_name}, Test Acc: {test_acc:.2f}%")
    torch.save(model.state_dict(), checkpoint_path)
    return model, test_acc

def evaluate_model(model, test_dataset, batch_size=BATCH_SIZE):
    model.to(DEVICE)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# 3. Model Merging Helpers
def get_base_model():
    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

def merge_models_wa(experts_state_dicts, base_state_dict):
    # Parameter weight averaging (WA)
    merged_backbone = get_base_model()
    merged_state_dict = copy.deepcopy(merged_backbone.state_dict())
    
    # We only merge the backbone parameters (not fc)
    backbone_keys = [k for k in base_state_dict.keys() if not k.startswith('fc')]
    
    for key in backbone_keys:
        stacked = torch.stack([experts_state_dicts[task][key].float() for task in experts_state_dicts])
        merged_state_dict[key] = torch.mean(stacked, dim=0).to(experts_state_dicts[list(experts_state_dicts.keys())[0]][key].dtype)
        
    merged_backbone.load_state_dict(merged_state_dict)
    return merged_backbone

def merge_models_ta(experts_state_dicts, base_state_dict, lam=0.3):
    # Task Arithmetic (TA)
    merged_backbone = get_base_model()
    merged_state_dict = copy.deepcopy(merged_backbone.state_dict())
    
    backbone_keys = [k for k in base_state_dict.keys() if not k.startswith('fc')]
    
    for key in backbone_keys:
        # compute task vectors
        task_vectors = []
        for task in experts_state_dicts:
            tv = experts_state_dicts[task][key].float() - base_state_dict[key].float()
            task_vectors.append(tv)
        # sum task vectors and scale, then add to base
        sum_tv = torch.stack(task_vectors).sum(dim=0)
        merged_state_dict[key] = (base_state_dict[key].float() + lam * sum_tv).to(base_state_dict[key].dtype)
        
    merged_backbone.load_state_dict(merged_state_dict)
    return merged_backbone

# 4. CKA (Centered Kernel Alignment) Calculation
def linear_cka(X, Y):
    # Flatten features if they are spatial (e.g. [B, C, H, W] -> [B, C*H*W])
    if len(X.shape) > 2:
        X = X.view(X.size(0), -1)
    if len(Y.shape) > 2:
        Y = Y.view(Y.size(0), -1)
        
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    X_XT = torch.matmul(X, X.t())
    Y_YT = torch.matmul(Y, Y.t())
    
    hsic = torch.sum(X_XT * Y_YT)
    var_x = torch.sum(X_XT * X_XT)
    var_y = torch.sum(Y_YT * Y_YT)
    
    return (hsic / (torch.sqrt(var_x * var_y) + 1e-8)).item()

def compute_layer_cka(expert_models, merged_model, test_datasets):
    # We will compute layer-wise CKA on a subset of the test sets
    cka_results = {}
    subset_size = 256
    
    # Identify layers to hook
    hook_layers = {
        'layer1': merged_model.layer1,
        'layer2': merged_model.layer2,
        'layer3': merged_model.layer3,
        'layer4': merged_model.layer4
    }
    
    for task in expert_models:
        expert = expert_models[task].to(DEVICE)
        expert.eval()
        merged_model = merged_model.to(DEVICE)
        merged_model.eval()
        
        # Load a batch of test data
        test_sub = Subset(test_datasets[task], range(subset_size))
        loader = DataLoader(test_sub, batch_size=subset_size, shuffle=False)
        inputs, _ = next(iter(loader))
        inputs = inputs.to(DEVICE)
        
        # We will extract activations from the end of each layer block
        expert_acts = {}
        merged_acts = {}
        
        # Register hooks
        expert_hooks = []
        merged_hooks = []
        
        def get_hook(acts, name):
            def hook(module, input, output):
                acts[name] = output.detach().cpu()
            return hook
            
        for name, layer in hook_layers.items():
            # register on expert
            exp_layer = getattr(expert, name)
            expert_hooks.append(exp_layer.register_forward_hook(get_hook(expert_acts, name)))
            # register on merged
            mrg_layer = getattr(merged_model, name)
            merged_hooks.append(mrg_layer.register_forward_hook(get_hook(merged_acts, name)))
            
        # Forward pass
        with torch.no_grad():
            expert(inputs)
            merged_model(inputs)
            
        # Remove hooks
        for h in expert_hooks: h.remove()
        for h in merged_hooks: h.remove()
        
        # Compute CKA
        cka_results[task] = {}
        for name in hook_layers:
            cka_val = linear_cka(expert_acts[name], merged_acts[name])
            cka_results[task][name] = cka_val
            
    return cka_results

# 5. Calibration Mechanics
class CalibrationHook:
    def __init__(self, mode='TCAC', placement='Pre-ReLU', epsilon=1e-5, target_task=0, alpha=0.2):
        self.mode = mode  # 'TCAC', 'LSC', 'Targeted'
        self.placement = placement # 'Pre-ReLU', 'Post-ReLU'
        self.epsilon = epsilon
        self.target_task = target_task
        self.alpha = alpha
        
        # Store statistics: dict mapping module -> {statistics}
        self.stats = {}
        
    def get_calibration_hook(self, module, name, is_expert=True, task_idx=0):
        # This hook collects original activation statistics on the calibration set
        def hook(mod, input, output):
            # output is [B, C, H, W]
            act = output.detach()
            if self.placement == 'Post-ReLU' and is_expert:
                # If we are post-ReLU, output has already passed through ReLU in the module?
                # Actually, BatchNorm is usually followed by ReLU.
                # So we can apply ReLU in the hook to simulate post-ReLU
                act = torch.relu(act)
                
            # Compute statistics
            key = (name, task_idx)
            if self.mode == 'TCAC':
                # Channel-wise mean and std (over batch B, height H, width W)
                mean = act.mean(dim=(0, 2, 3), keepdim=True)
                std = act.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
                if is_expert:
                    if key not in self.stats:
                        self.stats[key] = {}
                    self.stats[key]['orig_mean'] = mean
                    self.stats[key]['orig_std'] = std
                else:
                    if key not in self.stats:
                        self.stats[key] = {}
                    self.stats[key]['merged_mean'] = mean
                    self.stats[key]['merged_std'] = std
            elif self.mode == 'S-TCAC':
                # Channel-wise mean and std (over batch B, height H, width W)
                c_mean = act.mean(dim=(0, 2, 3), keepdim=True)
                c_std = act.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
                # Layer-wise global mean and std
                l_mean = act.mean()
                l_std = act.std(unbiased=False)
                # Perform shrinkage using self.alpha
                mean = (1 - self.alpha) * c_mean + self.alpha * l_mean
                std = (1 - self.alpha) * c_std + self.alpha * l_std
                if is_expert:
                    if key not in self.stats:
                        self.stats[key] = {}
                    self.stats[key]['orig_mean'] = mean
                    self.stats[key]['orig_std'] = std
                else:
                    if key not in self.stats:
                        self.stats[key] = {}
                    self.stats[key]['merged_mean'] = mean
                    self.stats[key]['merged_std'] = std
            elif self.mode == 'LSC':
                # Layer-wise global mean and std (over all dimensions except channel? No, over all dimensions including channel)
                mean = act.mean()
                std = act.std(unbiased=False)
                if is_expert:
                    if key not in self.stats:
                        self.stats[key] = {}
                    self.stats[key]['orig_mean'] = mean
                    self.stats[key]['orig_std'] = std
                else:
                    if key not in self.stats:
                        self.stats[key] = {}
                    self.stats[key]['merged_mean'] = mean
                    self.stats[key]['merged_std'] = std
        return hook

    def get_inference_hook(self, name):
        # This hook applies the calibration during evaluation
        def hook(mod, input, output):
            # output: [B, C, H, W]
            key = (name, self.target_task)
            if key not in self.stats or 'merged_std' not in self.stats[key]:
                return output # Skip if stats not available
                
            stats = self.stats[key]
            
            if self.mode in ['TCAC', 'S-TCAC']:
                # Channel-wise normalization & restoration
                orig_mean = stats['orig_mean'].to(output.device)
                orig_std = stats['orig_std'].to(output.device)
                merged_mean = stats['merged_mean'].to(output.device)
                merged_std = stats['merged_std'].to(output.device)
                
                # Check for "Sparsity Trap" (channels with very low std)
                # To simulate the original TCAC behavior: divide by merged_std + epsilon
                normalized = (output - merged_mean) / (merged_std + self.epsilon)
                calibrated = normalized * orig_std + orig_mean
                return calibrated
                
            elif self.mode == 'LSC':
                # Layer-wise global scaling (no mean shift, only scaling)
                orig_std = stats['orig_std'].to(output.device)
                merged_std = stats['merged_std'].to(output.device)
                
                scaling_factor = orig_std / (merged_std + self.epsilon)
                return output * scaling_factor
                
            return output
        return hook

# Run Task-Agnostic Activation Calibration (N-TAAC)
def run_ntaac(merged_model, joint_calibration_loader):
    # Native TAAC: Put in training mode, freeze params, set momentum to 1.0, run forward pass
    calibrated_model = copy.deepcopy(merged_model)
    calibrated_model.to(DEVICE)
    calibrated_model.train()
    
    # Freeze all parameters
    for param in calibrated_model.parameters():
        param.requires_grad = False
        
    # Set momentum of all BatchNorm layers to 1.0
    for module in calibrated_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 1.0
            
    # Run a single forward pass of the joint calibration set
    # Wait, the classification head in merged_model is not used during the backbone forward,
    # but the ResNet forward needs to pass through fc. Since fc is just linear, we can pass it through.
    # Note: the joint dataset contains samples from all tasks, so we can just forward them.
    with torch.no_grad():
        for inputs, _ in joint_calibration_loader:
            inputs = inputs.to(DEVICE)
            calibrated_model(inputs)
            
    # Put back to eval mode
    calibrated_model.eval()
    return calibrated_model

# Core Evaluation Pipeline
def run_evaluation_suite(expert_models, base_model, datasets, N_budget=128):
    results = {}
    task_names = list(expert_models.keys())
    
    # Create Calibration sets
    calib_subsets = {}
    for task in task_names:
        # Pick first N_budget samples of train set as calibration set
        sub = Subset(datasets[task][0], range(N_budget))
        calib_subsets[task] = sub
        
    # Create Joint Calibration Set for TAAC
    joint_dataset = ConcatDataset([calib_subsets[t] for t in task_names])
    joint_loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)
    
    # Evaluate Experts Upper Bound
    expert_accs = {}
    for task in task_names:
        expert_accs[task] = evaluate_model(expert_models[task], datasets[task][1])
    results['Expert_Oracle'] = expert_accs
    results['Expert_Oracle_Avg'] = np.mean(list(expert_accs.values()))
    
    print("\n--- Model Merging Baselines ---")
    experts_state_dicts = {t: expert_models[t].state_dict() for t in task_names}
    base_state_dict = base_model.state_dict()
    
    # 1. Weight Averaging (WA)
    merged_wa = merge_models_wa(experts_state_dicts, base_state_dict)
    acc_wa = {}
    for i, task in enumerate(task_names):
        # Apply task specific classification head to merged backbone
        merged_wa.fc = expert_models[task].fc
        acc_wa[task] = evaluate_model(merged_wa, datasets[task][1])
    results['WA'] = acc_wa
    results['WA_Avg'] = np.mean(list(acc_wa.values()))
    print(f"WA (No Calib) Accuracies: {acc_wa}, Avg: {results['WA_Avg']:.2f}%")
    
    # Compute CKA for WA
    cka_wa = compute_layer_cka(expert_models, merged_wa, {t: datasets[t][1] for t in task_names})
    results['CKA_WA'] = cka_wa
    print(f"Layer-wise CKA (WA): {cka_wa}")
    
    # 2. Task Arithmetic (TA)
    merged_ta = merge_models_ta(experts_state_dicts, base_state_dict, lam=0.3)
    acc_ta = {}
    for i, task in enumerate(task_names):
        merged_ta.fc = expert_models[task].fc
        acc_ta[task] = evaluate_model(merged_ta, datasets[task][1])
    results['TA'] = acc_ta
    results['TA_Avg'] = np.mean(list(acc_ta.values()))
    print(f"TA (No Calib) Accuracies: {acc_ta}, Avg: {results['TA_Avg']:.2f}%")
    
    # 3. Native TAAC (N-TAAC)
    merged_wa_taac = run_ntaac(merged_wa, joint_loader)
    acc_taac = {}
    for i, task in enumerate(task_names):
        merged_wa_taac.fc = expert_models[task].fc
        acc_taac[task] = evaluate_model(merged_wa_taac, datasets[task][1])
    results['N_TAAC'] = acc_taac
    results['N_TAAC_Avg'] = np.mean(list(acc_taac.values()))
    print(f"N-TAAC Accuracies: {acc_taac}, Avg: {results['N_TAAC_Avg']:.2f}%")

    # 4. Activation Calibration (TCAC vs. LSC vs. Targeted)
    # We will test Pre-ReLU vs. Post-ReLU, and Full vs. Targeted (Layer 4 only), and Parallel vs. Sequential collection.
    calibration_scenarios = [
        # (mode, placement, target_layers, collection)
        ('TCAC', 'Pre-ReLU', 'all', 'parallel'),
        ('TCAC', 'Post-ReLU', 'all', 'parallel'),
        ('LSC', 'Pre-ReLU', 'all', 'parallel'),
        ('LSC', 'Post-ReLU', 'all', 'parallel'),
        ('LSC', 'Pre-ReLU', 'layer4_only', 'parallel'),
        ('TCAC', 'Pre-ReLU', 'layer4_only', 'parallel'),
        
        # New Sequential Collection Scenarios
        ('TCAC', 'Pre-ReLU', 'all', 'sequential'),
        ('TCAC', 'Post-ReLU', 'all', 'sequential'),
        ('LSC', 'Pre-ReLU', 'all', 'sequential'),
        ('LSC', 'Post-ReLU', 'all', 'sequential'),

        # NEW ADVANCED SCENARIOS (Methodological Refinement Cycles 7 & 8)
        ('TCAC_TA', 'Pre-ReLU', 'all', 'sequential'),
        ('TCAC_Joint', 'Pre-ReLU', 'all', 'sequential'),
        ('S-TCAC_Joint', 'Pre-ReLU', 'all', 'sequential'),
    ]
    
    for mode, placement, target, collection in calibration_scenarios:
        scenario_name = f"{mode}_{placement}_{target}_{collection}"
        print(f"\nRunning Calibration Scenario: {scenario_name}")
        
        # Determine actual hook mode and base merged model
        hook_mode = mode
        base_merged = merged_wa
        if mode == 'TCAC_TA':
            hook_mode = 'TCAC'
            base_merged = merged_ta
        elif mode == 'TCAC_Joint':
            hook_mode = 'TCAC'
            base_merged = merged_wa
        elif mode == 'S-TCAC_Joint':
            hook_mode = 'S-TCAC'
            base_merged = merged_wa

        # Initialize helper
        calib_hook = CalibrationHook(mode=hook_mode, placement=placement, epsilon=1e-5, alpha=0.2)
        
        # Collect statistics from Experts
        for task_idx, task in enumerate(task_names):
            expert = expert_models[task].to(DEVICE)
            expert.eval()
            
            # Register hooks on all BatchNorm layers of expert
            expert_hooks = []
            for name, module in expert.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Filter for layer4 only if specified
                    if target == 'layer4_only' and 'layer4' not in name:
                        continue
                    hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=True, task_idx=task_idx)
                    expert_hooks.append(module.register_forward_hook(hook_fn))
                    
            # Forward pass on task-specific calibration set
            loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
            inputs, _ = next(iter(loader))
            inputs = inputs.to(DEVICE)
            with torch.no_grad():
                expert(inputs)
                
            # Remove hooks
            for h in expert_hooks: h.remove()

        if 'Joint' in mode:
            # Average original expert stats across tasks and save under 'joint'
            for name, module in base_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    if target == 'layer4_only' and 'layer4' not in name:
                        continue
                    means = [calib_hook.stats[(name, t_idx)]['orig_mean'] for t_idx in range(len(task_names))]
                    stds = [calib_hook.stats[(name, t_idx)]['orig_std'] for t_idx in range(len(task_names))]
                    calib_hook.stats[(name, 'joint')] = {
                        'orig_mean': torch.stack(means).mean(dim=0),
                        'orig_std': torch.stack(stds).mean(dim=0)
                    }
            
        # Collect statistics from Merged Model
        if collection == 'parallel':
            merged_model = copy.deepcopy(base_merged).to(DEVICE)
            merged_model.eval()
            for task_idx, task in enumerate(task_names):
                merged_model.fc = expert_models[task].fc.to(DEVICE)
                
                # Register hooks
                merged_hooks = []
                for name, module in merged_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        if target == 'layer4_only' and 'layer4' not in name:
                            continue
                        hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=False, task_idx=task_idx)
                        merged_hooks.append(module.register_forward_hook(hook_fn))
                        
                # Forward pass on calibration set
                loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                inputs, _ = next(iter(loader))
                inputs = inputs.to(DEVICE)
                with torch.no_grad():
                    merged_model(inputs)
                    
                # Remove hooks
                for h in merged_hooks: h.remove()
                
        elif collection == 'sequential':
            # Collect layer-by-layer
            merged_model = copy.deepcopy(base_merged).to(DEVICE)
            bn_names_and_modules = []
            for name, module in merged_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    if target == 'layer4_only' and 'layer4' not in name:
                        continue
                    bn_names_and_modules.append((name, module))
                    
            if 'Joint' in mode:
                # Joint sequential collection on the joint dataset
                loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)
                inputs, _ = next(iter(loader))
                inputs = inputs.to(DEVICE)
                
                calibrated_so_far = []
                for current_name, current_module in bn_names_and_modules:
                    curr_merged_model = copy.deepcopy(base_merged).to(DEVICE)
                    curr_merged_model.eval()
                    curr_merged_model.fc = expert_models[task_names[0]].fc.to(DEVICE)
                    
                    calib_hook.target_task = 'joint'
                    
                    active_hooks = []
                    for prev_name in calibrated_so_far:
                        prev_module = dict(curr_merged_model.named_modules())[prev_name]
                        hook_fn = calib_hook.get_inference_hook(prev_name)
                        active_hooks.append(prev_module.register_forward_hook(hook_fn))
                        
                    curr_module = dict(curr_merged_model.named_modules())[current_name]
                    hook_fn = calib_hook.get_calibration_hook(curr_module, current_name, is_expert=False, task_idx='joint')
                    active_hooks.append(curr_module.register_forward_hook(hook_fn))
                    
                    with torch.no_grad():
                        curr_merged_model(inputs)
                    for h in active_hooks: h.remove()
                    calibrated_so_far.append(current_name)
            else:
                # Standard task-conditional sequential collection
                for task_idx, task in enumerate(task_names):
                    loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                    inputs, _ = next(iter(loader))
                    inputs = inputs.to(DEVICE)
                    
                    calibrated_so_far = []
                    for current_name, current_module in bn_names_and_modules:
                        curr_merged_model = copy.deepcopy(base_merged).to(DEVICE)
                        curr_merged_model.eval()
                        curr_merged_model.fc = expert_models[task].fc.to(DEVICE)
                        
                        calib_hook.target_task = task_idx
                        
                        active_hooks = []
                        # Register inference hooks on all previous layers
                        for prev_name in calibrated_so_far:
                            prev_module = dict(curr_merged_model.named_modules())[prev_name]
                            hook_fn = calib_hook.get_inference_hook(prev_name)
                            active_hooks.append(prev_module.register_forward_hook(hook_fn))
                            
                        # Register calibration hook on current layer
                        curr_module = dict(curr_merged_model.named_modules())[current_name]
                        hook_fn = calib_hook.get_calibration_hook(curr_module, current_name, is_expert=False, task_idx=task_idx)
                        active_hooks.append(curr_module.register_forward_hook(hook_fn))
                        
                        # Forward pass
                        with torch.no_grad():
                            curr_merged_model(inputs)
                            
                        # Remove all hooks
                        for h in active_hooks: h.remove()
                        
                        # Add current name to calibrated list
                        calibrated_so_far.append(current_name)
            
        # Perform Inference with calibration hooks registered
        scenario_accs = {}
        for task_idx, task in enumerate(task_names):
            merged_model = copy.deepcopy(base_merged).to(DEVICE)
            merged_model.eval()
            merged_model.fc = expert_models[task].fc.to(DEVICE)
            
            # Set target task in calibration hook
            if 'Joint' in mode:
                calib_hook.target_task = 'joint'
            else:
                calib_hook.target_task = task_idx
            
            # Register inference hooks
            inf_hooks = []
            for name, module in merged_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    if target == 'layer4_only' and 'layer4' not in name:
                        continue
                    hook_fn = calib_hook.get_inference_hook(name)
                    inf_hooks.append(module.register_forward_hook(hook_fn))
                    
            # Evaluate
            scenario_accs[task] = evaluate_model(merged_model, datasets[task][1])
            
            # Remove hooks
            for h in inf_hooks: h.remove()
            
        results[scenario_name] = scenario_accs
        results[f"{scenario_name}_Avg"] = np.mean(list(scenario_accs.values()))
        print(f"Result {scenario_name} Accuracies: {scenario_accs}, Avg: {results[f'{scenario_name}_Avg']:.2f}%")
        
    return results

# 5.5. Sample Complexity Sweep
def run_sample_complexity_sweep(expert_models, base_model, datasets):
    print("\n================== RUNNING SAMPLE COMPLEXITY SWEEP ==================")
    task_names = list(expert_models.keys())
    budgets = [16, 32, 64, 128, 256]
    
    scenarios = [
        ('TCAC', 'Pre-ReLU', 'all', 'parallel'),
        ('TCAC', 'Pre-ReLU', 'all', 'sequential'),
        ('S-TCAC', 'Pre-ReLU', 'all', 'sequential'),
        ('LSC', 'Pre-ReLU', 'all', 'parallel'),
        ('LSC', 'Pre-ReLU', 'all', 'sequential'),
    ]
    
    sweep_results = {f"{mode}_{placement}_{target}_{collection}": [] for mode, placement, target, collection in scenarios}
    
    experts_state_dicts = {t: expert_models[t].state_dict() for t in task_names}
    base_state_dict = base_model.state_dict()
    merged_wa = merge_models_wa(experts_state_dicts, base_state_dict)
    
    for N_budget in budgets:
        print(f"\n--- Evaluation with N = {N_budget} ---")
        calib_subsets = {}
        for task in task_names:
            sub = Subset(datasets[task][0], range(N_budget))
            calib_subsets[task] = sub
            
        for mode, placement, target, collection in scenarios:
            scenario_key = f"{mode}_{placement}_{target}_{collection}"
            calib_hook = CalibrationHook(mode=mode, placement=placement, epsilon=1e-5)
            
            # Collect original expert statistics
            for task_idx, task in enumerate(task_names):
                expert = expert_models[task].to(DEVICE)
                expert.eval()
                expert_hooks = []
                for name, module in expert.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=True, task_idx=task_idx)
                        expert_hooks.append(module.register_forward_hook(hook_fn))
                
                loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                inputs, _ = next(iter(loader))
                inputs = inputs.to(DEVICE)
                with torch.no_grad():
                    expert(inputs)
                for h in expert_hooks: h.remove()
                
            # Collect merged statistics
            if collection == 'parallel':
                merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                merged_model.eval()
                for task_idx, task in enumerate(task_names):
                    merged_model.fc = expert_models[task].fc.to(DEVICE)
                    merged_hooks = []
                    for name, module in merged_model.named_modules():
                        if isinstance(module, nn.BatchNorm2d):
                            hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=False, task_idx=task_idx)
                            merged_hooks.append(module.register_forward_hook(hook_fn))
                    loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                    inputs, _ = next(iter(loader))
                    inputs = inputs.to(DEVICE)
                    with torch.no_grad():
                        merged_model(inputs)
                    for h in merged_hooks: h.remove()
                    
            elif collection == 'sequential':
                merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                bn_names_and_modules = []
                for name, module in merged_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn_names_and_modules.append((name, module))
                        
                for task_idx, task in enumerate(task_names):
                    loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                    inputs, _ = next(iter(loader))
                    inputs = inputs.to(DEVICE)
                    
                    calibrated_so_far = []
                    for current_name, current_module in bn_names_and_modules:
                        curr_merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                        curr_merged_model.eval()
                        curr_merged_model.fc = expert_models[task].fc.to(DEVICE)
                        calib_hook.target_task = task_idx
                        
                        active_hooks = []
                        for prev_name in calibrated_so_far:
                            prev_module = dict(curr_merged_model.named_modules())[prev_name]
                            hook_fn = calib_hook.get_inference_hook(prev_name)
                            active_hooks.append(prev_module.register_forward_hook(hook_fn))
                            
                        curr_module = dict(curr_merged_model.named_modules())[current_name]
                        hook_fn = calib_hook.get_calibration_hook(curr_module, current_name, is_expert=False, task_idx=task_idx)
                        active_hooks.append(curr_module.register_forward_hook(hook_fn))
                        
                        with torch.no_grad():
                            curr_merged_model(inputs)
                        for h in active_hooks: h.remove()
                        calibrated_so_far.append(current_name)
                        
            # Evaluate
            scenario_accs = {}
            for task_idx, task in enumerate(task_names):
                merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                merged_model.eval()
                merged_model.fc = expert_models[task].fc.to(DEVICE)
                calib_hook.target_task = task_idx
                
                inf_hooks = []
                for name, module in merged_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        hook_fn = calib_hook.get_inference_hook(name)
                        inf_hooks.append(module.register_forward_hook(hook_fn))
                        
                scenario_accs[task] = evaluate_model(merged_model, datasets[task][1])
                for h in inf_hooks: h.remove()
                
            avg_acc = np.mean(list(scenario_accs.values()))
            sweep_results[scenario_key].append(avg_acc)
            print(f"[{scenario_key}, N={N_budget}] Avg Acc: {avg_acc:.2f}%")
            
    return sweep_results

# 5.6. Shrinkage Factor Alpha Ablation Sweep
def run_alpha_ablation_sweep(expert_models, base_model, datasets, N_budget=16):
    print("\n================== RUNNING ALPHA ABLATION SWEEP ==================")
    task_names = list(expert_models.keys())
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    experts_state_dicts = {t: expert_models[t].state_dict() for t in task_names}
    base_state_dict = base_model.state_dict()
    merged_wa = merge_models_wa(experts_state_dicts, base_state_dict)
    
    calib_subsets = {}
    for task in task_names:
        sub = Subset(datasets[task][0], range(N_budget))
        calib_subsets[task] = sub
        
    ablation_results = {}
    
    for alpha in alphas:
        print(f"Evaluating S-TCAC with alpha = {alpha} (N = {N_budget})...")
        calib_hook = CalibrationHook(mode='S-TCAC', placement='Pre-ReLU', epsilon=1e-5, alpha=alpha)
        
        # Collect original expert statistics
        for task_idx, task in enumerate(task_names):
            expert = expert_models[task].to(DEVICE)
            expert.eval()
            expert_hooks = []
            for name, module in expert.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=True, task_idx=task_idx)
                    expert_hooks.append(module.register_forward_hook(hook_fn))
            
            loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
            inputs, _ = next(iter(loader))
            inputs = inputs.to(DEVICE)
            with torch.no_grad():
                expert(inputs)
            for h in expert_hooks: h.remove()
            
        # Collect merged statistics (sequential)
        merged_model = copy.deepcopy(merged_wa).to(DEVICE)
        bn_names_and_modules = []
        for name, module in merged_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_names_and_modules.append((name, module))
                
        for task_idx, task in enumerate(task_names):
            loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
            inputs, _ = next(iter(loader))
            inputs = inputs.to(DEVICE)
            
            calibrated_so_far = []
            for current_name, current_module in bn_names_and_modules:
                curr_merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                curr_merged_model.eval()
                curr_merged_model.fc = expert_models[task].fc.to(DEVICE)
                calib_hook.target_task = task_idx
                
                active_hooks = []
                for prev_name in calibrated_so_far:
                    prev_module = dict(curr_merged_model.named_modules())[prev_name]
                    hook_fn = calib_hook.get_inference_hook(prev_name)
                    active_hooks.append(prev_module.register_forward_hook(hook_fn))
                    
                curr_module = dict(curr_merged_model.named_modules())[current_name]
                hook_fn = calib_hook.get_calibration_hook(curr_module, current_name, is_expert=False, task_idx=task_idx)
                active_hooks.append(curr_module.register_forward_hook(hook_fn))
                
                with torch.no_grad():
                    curr_merged_model(inputs)
                for h in active_hooks: h.remove()
                calibrated_so_far.append(current_name)
                
        # Evaluate
        scenario_accs = {}
        for task_idx, task in enumerate(task_names):
            merged_model = copy.deepcopy(merged_wa).to(DEVICE)
            merged_model.eval()
            merged_model.fc = expert_models[task].fc.to(DEVICE)
            calib_hook.target_task = task_idx
            
            inf_hooks = []
            for name, module in merged_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    hook_fn = calib_hook.get_inference_hook(name)
                    inf_hooks.append(module.register_forward_hook(hook_fn))
                    
            scenario_accs[task] = evaluate_model(merged_model, datasets[task][1])
            for h in inf_hooks: h.remove()
            
        avg_acc = np.mean(list(scenario_accs.values()))
        ablation_results[str(alpha)] = {
            'mnist': scenario_accs['mnist'],
            'fashion': scenario_accs['fashion'],
            'cifar': scenario_accs['cifar'],
            'average': avg_acc
        }
        print(f"alpha = {alpha}: Avg Acc = {avg_acc:.2f}%")
        
    return ablation_results

# 5.7. Composition Bias Sweep
def run_composition_bias_sweep(expert_models, base_model, datasets):
    print("\n================== RUNNING COMPOSITION BIAS SWEEP ==================")
    task_names = list(expert_models.keys()) # ['mnist', 'fashion', 'cifar']
    
    compositions = {
        'balanced': {'mnist': 128, 'fashion': 128, 'cifar': 128},
        'mnist_heavy': {'mnist': 300, 'fashion': 42, 'cifar': 42},
        'fashion_heavy': {'mnist': 42, 'fashion': 300, 'cifar': 42},
        'cifar_heavy': {'mnist': 42, 'fashion': 42, 'cifar': 300}
    }
    
    experts_state_dicts = {t: expert_models[t].state_dict() for t in task_names}
    base_state_dict = base_model.state_dict()
    merged_wa = merge_models_wa(experts_state_dicts, base_state_dict)
    
    bias_results = {}
    
    for comp_name, counts in compositions.items():
        print(f"\nEvaluating with Composition: {comp_name} {counts}")
        
        # Build task-specific calibration subsets based on counts
        calib_subsets = {}
        for task in task_names:
            sub = Subset(datasets[task][0], range(counts[task]))
            calib_subsets[task] = sub
            
        joint_dataset = ConcatDataset([calib_subsets[t] for t in task_names])
        
        scenarios = [
            ('TCAC_Joint', 'TCAC'),
            ('S-TCAC_Joint', 'S-TCAC')
        ]
        
        comp_metrics = {}
        for mode, hook_mode in scenarios:
            calib_hook = CalibrationHook(mode=hook_mode, placement='Pre-ReLU', epsilon=1e-5, alpha=0.6)
            
            # 1. Collect Expert Stats
            for task_idx, task in enumerate(task_names):
                expert = expert_models[task].to(DEVICE)
                expert.eval()
                expert_hooks = []
                for name, module in expert.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=True, task_idx=task_idx)
                        expert_hooks.append(module.register_forward_hook(hook_fn))
                
                loader = DataLoader(calib_subsets[task], batch_size=counts[task], shuffle=False)
                inputs, _ = next(iter(loader))
                inputs = inputs.to(DEVICE)
                with torch.no_grad():
                    expert(inputs)
                for h in expert_hooks: h.remove()
                
            # 2. Average original expert stats across tasks and save under 'joint'
            for name, module in base_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    means = [calib_hook.stats[(name, t_idx)]['orig_mean'] for t_idx in range(len(task_names))]
                    stds = [calib_hook.stats[(name, t_idx)]['orig_std'] for t_idx in range(len(task_names))]
                    calib_hook.stats[(name, 'joint')] = {
                        'orig_mean': torch.stack(means).mean(dim=0),
                        'orig_std': torch.stack(stds).mean(dim=0)
                    }
                    
            # 3. Collect Merged Stats (Joint Sequential)
            merged_model = copy.deepcopy(merged_wa).to(DEVICE)
            bn_names_and_modules = []
            for name, module in merged_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_names_and_modules.append((name, module))
                    
            loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)
            inputs, _ = next(iter(loader))
            inputs = inputs.to(DEVICE)
            
            calibrated_so_far = []
            for current_name, current_module in bn_names_and_modules:
                curr_merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                curr_merged_model.eval()
                curr_merged_model.fc = expert_models[task_names[0]].fc.to(DEVICE)
                
                calib_hook.target_task = 'joint'
                
                active_hooks = []
                for prev_name in calibrated_so_far:
                    prev_module = dict(curr_merged_model.named_modules())[prev_name]
                    hook_fn = calib_hook.get_inference_hook(prev_name)
                    active_hooks.append(prev_module.register_forward_hook(hook_fn))
                    
                curr_module = dict(curr_merged_model.named_modules())[current_name]
                hook_fn = calib_hook.get_calibration_hook(curr_module, current_name, is_expert=False, task_idx='joint')
                active_hooks.append(curr_module.register_forward_hook(hook_fn))
                
                with torch.no_grad():
                    curr_merged_model(inputs)
                for h in active_hooks: h.remove()
                calibrated_so_far.append(current_name)
                
            # 4. Evaluate on test sets
            scenario_accs = {}
            for task_idx, task in enumerate(task_names):
                merged_model_eval = copy.deepcopy(merged_wa).to(DEVICE)
                merged_model_eval.eval()
                merged_model_eval.fc = expert_models[task].fc.to(DEVICE)
                
                calib_hook.target_task = 'joint'
                
                inf_hooks = []
                for name, module in merged_model_eval.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        hook_fn = calib_hook.get_inference_hook(name)
                        inf_hooks.append(module.register_forward_hook(hook_fn))
                        
                scenario_accs[task] = evaluate_model(merged_model_eval, datasets[task][1])
                for h in inf_hooks: h.remove()
                
            avg_acc = np.mean(list(scenario_accs.values()))
            comp_metrics[mode] = {
                'mnist': scenario_accs['mnist'],
                'fashion': scenario_accs['fashion'],
                'cifar': scenario_accs['cifar'],
                'average': avg_acc
            }
            print(f"[{mode} under {comp_name}] Accs: {scenario_accs}, Avg: {avg_acc:.2f}%")
            
        bias_results[comp_name] = comp_metrics
        
    return bias_results

# 5.8. Multi-Seed Robustness Sweep
def run_multi_seed_robustness_sweep(expert_models, base_model, datasets, N_budgets=[16, 64], num_seeds=5):
    print("\n================== RUNNING MULTI-SEED ROBUSTNESS SWEEP ==================")
    task_names = list(expert_models.keys())
    
    scenarios = [
        ('TCAC', 'Pre-ReLU', 'all', 'sequential'),
        ('S-TCAC', 'Pre-ReLU', 'all', 'sequential'),
        ('LSC', 'Pre-ReLU', 'all', 'sequential'),
    ]
    
    experts_state_dicts = {t: expert_models[t].state_dict() for t in task_names}
    base_state_dict = base_model.state_dict()
    merged_wa = merge_models_wa(experts_state_dicts, base_state_dict)
    
    robustness_results = {}
    
    seeds = [42 + i for i in range(num_seeds)]
    
    for N_budget in N_budgets:
        robustness_results[str(N_budget)] = {}
        for mode, placement, target, collection in scenarios:
            scenario_key = f"{mode}_{placement}_{target}_{collection}"
            robustness_results[str(N_budget)][scenario_key] = []
            
        print(f"\n--- Multi-Seed Sweep for N = {N_budget} ---")
        for seed in seeds:
            print(f"Running Seed {seed}...")
            
            # Deterministic subset sampling for this seed
            calib_subsets = {}
            for task in task_names:
                g = torch.Generator()
                g.manual_seed(seed)
                indices = torch.randperm(len(datasets[task][0]), generator=g)[:N_budget].tolist()
                calib_subsets[task] = Subset(datasets[task][0], indices)
                
            for mode, placement, target, collection in scenarios:
                scenario_key = f"{mode}_{placement}_{target}_{collection}"
                # Use alpha=0.6 for S-TCAC to match our optimized value
                calib_hook = CalibrationHook(mode=mode, placement=placement, epsilon=1e-5, alpha=0.6)
                
                # Collect expert statistics
                for task_idx, task in enumerate(task_names):
                    expert = expert_models[task].to(DEVICE)
                    expert.eval()
                    expert_hooks = []
                    for name, module in expert.named_modules():
                        if isinstance(module, nn.BatchNorm2d):
                            hook_fn = calib_hook.get_calibration_hook(module, name, is_expert=True, task_idx=task_idx)
                            expert_hooks.append(module.register_forward_hook(hook_fn))
                    
                    loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                    inputs, _ = next(iter(loader))
                    inputs = inputs.to(DEVICE)
                    with torch.no_grad():
                        expert(inputs)
                    for h in expert_hooks: h.remove()
                    
                # Collect merged statistics (sequential)
                merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                bn_names_and_modules = []
                for name, module in merged_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn_names_and_modules.append((name, module))
                        
                for task_idx, task in enumerate(task_names):
                    loader = DataLoader(calib_subsets[task], batch_size=N_budget, shuffle=False)
                    inputs, _ = next(iter(loader))
                    inputs = inputs.to(DEVICE)
                    
                    calibrated_so_far = []
                    for current_name, current_module in bn_names_and_modules:
                        curr_merged_model = copy.deepcopy(merged_wa).to(DEVICE)
                        curr_merged_model.eval()
                        curr_merged_model.fc = expert_models[task].fc.to(DEVICE)
                        calib_hook.target_task = task_idx
                        
                        active_hooks = []
                        for prev_name in calibrated_so_far:
                            prev_module = dict(curr_merged_model.named_modules())[prev_name]
                            hook_fn = calib_hook.get_inference_hook(prev_name)
                            active_hooks.append(prev_module.register_forward_hook(hook_fn))
                            
                        curr_module = dict(curr_merged_model.named_modules())[current_name]
                        hook_fn = calib_hook.get_calibration_hook(curr_module, current_name, is_expert=False, task_idx=task_idx)
                        active_hooks.append(curr_module.register_forward_hook(hook_fn))
                        
                        with torch.no_grad():
                            curr_merged_model(inputs)
                        for h in active_hooks: h.remove()
                        calibrated_so_far.append(current_name)
                        
                # Evaluate on test sets
                scenario_accs = {}
                for task_idx, task in enumerate(task_names):
                    merged_model_eval = copy.deepcopy(merged_wa).to(DEVICE)
                    merged_model_eval.eval()
                    merged_model_eval.fc = expert_models[task].fc.to(DEVICE)
                    calib_hook.target_task = task_idx
                    
                    inf_hooks = []
                    for name, module in merged_model_eval.named_modules():
                        if isinstance(module, nn.BatchNorm2d):
                            hook_fn = calib_hook.get_inference_hook(name)
                            inf_hooks.append(module.register_forward_hook(hook_fn))
                            
                    scenario_accs[task] = evaluate_model(merged_model_eval, datasets[task][1])
                    for h in inf_hooks: h.remove()
                    
                avg_acc = np.mean(list(scenario_accs.values()))
                robustness_results[str(N_budget)][scenario_key].append(avg_acc)
                print(f"  [Seed {seed} - {scenario_key}] Avg Acc: {avg_acc:.2f}%")
                
    # Compile summary statistics (mean and std)
    summary_results = {}
    for N_budget in N_budgets:
        summary_results[str(N_budget)] = {}
        for mode, placement, target, collection in scenarios:
            scenario_key = f"{mode}_{placement}_{target}_{collection}"
            accs = robustness_results[str(N_budget)][scenario_key]
            summary_results[str(N_budget)][scenario_key] = {
                'all_accs': accs,
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs))
            }
            print(f"N={N_budget} | {scenario_key} | Mean: {np.mean(accs):.2f}%, Std: {np.std(accs):.2f}%")
            
    return summary_results

# 6. Main Runner
def main():
    print("Loading datasets...")
    datasets = get_datasets()
    
    # Train experts
    expert_models = {}
    base_model = get_base_model().to(DEVICE)
    
    for task in ['mnist', 'fashion', 'cifar']:
        train_data, test_data = datasets[task]
        model, test_acc = train_expert(task, train_data, test_data)
        expert_models[task] = model
        
    # Evaluate at N=128
    print("\n================== EVALUATING WITH N=128 ==================")
    results_128 = run_evaluation_suite(expert_models, base_model, datasets, N_budget=128)
    
    # Save results to JSON
    with open("results/metrics.json", "w") as f:
        json.dump(results_128, f, indent=4)
        
    # Plotting CKA Analysis
    plt.figure(figsize=(8, 5))
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    for task in ['mnist', 'fashion', 'cifar']:
        cka_vals = [results_128['CKA_WA'][task][lyr] for lyr in layers]
        plt.plot(layers, cka_vals, marker='o', label=f"{task.upper()} CKA")
    plt.title("Representational CKA Similarity between Experts & Merged Model")
    plt.xlabel("ResNet-18 Block Layers")
    plt.ylabel("Linear CKA")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.savefig("plots/cka_similarity.png", dpi=300)
    plt.close()
    
    # Plotting Calibration Comparison
    plt.figure(figsize=(13, 7))
    methods = [
        "Uncalibrated (WA)", 
        "N-TAAC (Task-Agnostic)",
        "TCAC Pre-ReLU (Parallel)", 
        "TCAC Pre-ReLU (Seq)", 
        "TCAC Post-ReLU (Parallel)", 
        "TCAC Post-ReLU (Seq)", 
        "LSC Pre-ReLU (Parallel)", 
        "LSC Pre-ReLU (Seq)", 
        "LSC Post-ReLU (Parallel)", 
        "LSC Post-ReLU (Seq)",
        "SC-TCAC Pre-ReLU (Layer 4)",
        "SC-LSC Pre-ReLU (Layer 4)",
        "TCAC on TA (Seq)",
        "Joint TCAC (Seq, Task-Agnostic)",
        "Joint S-TCAC (Seq, Task-Agnostic)"
    ]
    avg_accs = [
        results_128['WA_Avg'],
        results_128['N_TAAC_Avg'],
        results_128['TCAC_Pre-ReLU_all_parallel_Avg'],
        results_128['TCAC_Pre-ReLU_all_sequential_Avg'],
        results_128['TCAC_Post-ReLU_all_parallel_Avg'],
        results_128['TCAC_Post-ReLU_all_sequential_Avg'],
        results_128['LSC_Pre-ReLU_all_parallel_Avg'],
        results_128['LSC_Pre-ReLU_all_sequential_Avg'],
        results_128['LSC_Post-ReLU_all_parallel_Avg'],
        results_128['LSC_Post-ReLU_all_sequential_Avg'],
        results_128['TCAC_Pre-ReLU_layer4_only_parallel_Avg'],
        results_128['LSC_Pre-ReLU_layer4_only_parallel_Avg'],
        results_128['TCAC_TA_Pre-ReLU_all_sequential_Avg'],
        results_128['TCAC_Joint_Pre-ReLU_all_sequential_Avg'],
        results_128['S-TCAC_Joint_Pre-ReLU_all_sequential_Avg']
    ]
    
    bars = plt.bar(methods, avg_accs, color=['gray', 'purple', 'blue', 'dodgerblue', 'red', 'lightcoral', 'green', 'limegreen', 'orange', 'gold', 'cyan', 'teal', 'navy', 'magenta', 'orchid'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Multi-Task Accuracy (%)")
    plt.title("Deconstructing Model Merging Calibration (ResNet-18 Vision Benchmark)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    plt.savefig("plots/calibration_comparison.png", dpi=300)
    plt.close()
    
    # Run Sample Complexity Sweep
    complexity_results = run_sample_complexity_sweep(expert_models, base_model, datasets)
    
    # Save complexity results to JSON
    with open("results/metrics_complexity.json", "w") as f:
        json.dump(complexity_results, f, indent=4)
        
    # Run Alpha Ablation Sweep for S-TCAC (N=16)
    alpha_results = run_alpha_ablation_sweep(expert_models, base_model, datasets, N_budget=16)
    with open("results/metrics_alpha_ablation.json", "w") as f:
        json.dump(alpha_results, f, indent=4)
        
    # Plotting Alpha Ablation Sweep
    plt.figure(figsize=(8, 5))
    alphas_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    avg_accs_alpha = [alpha_results[str(a)]['average'] for a in alphas_list]
    plt.plot(alphas_list, avg_accs_alpha, marker='s', linewidth=2, color='purple', label='S-TCAC (N=16)')
    
    # Draw horizontal lines for baselines at N=16 (e.g. TCAC Sequential vs LSC Sequential)
    tcac_n16_baseline = complexity_results['TCAC_Pre-ReLU_all_sequential'][0]
    lsc_n16_baseline = complexity_results['LSC_Pre-ReLU_all_sequential'][0]
    plt.axhline(y=tcac_n16_baseline, color='blue', linestyle='--', alpha=0.7, label='TCAC Sequential (N=16)')
    plt.axhline(y=lsc_n16_baseline, color='orange', linestyle='--', alpha=0.7, label='LSC Sequential (N=16)')
    
    plt.xlabel("Shrinkage Factor Alpha (α)")
    plt.ylabel("Average Multi-Task Accuracy (%)")
    plt.title("Effect of Shrinkage Factor α on S-TCAC (N = 16)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/alpha_ablation.png", dpi=300)
    plt.close()

    # Plotting Sample Complexity Sweep
    plt.figure(figsize=(9, 6))
    budgets = [16, 32, 64, 128, 256]
    method_labels = {
        'TCAC_Pre-ReLU_all_parallel': 'TCAC Parallel (Pre-ReLU)',
        'TCAC_Pre-ReLU_all_sequential': 'TCAC Sequential (Pre-ReLU)',
        'S-TCAC_Pre-ReLU_all_sequential': 'S-TCAC Sequential (Ours)',
        'LSC_Pre-ReLU_all_parallel': 'LSC Parallel (Pre-ReLU)',
        'LSC_Pre-ReLU_all_sequential': 'LSC Sequential (Pre-ReLU)'
    }
    method_colors = {
        'TCAC_Pre-ReLU_all_parallel': 'red',
        'TCAC_Pre-ReLU_all_sequential': 'blue',
        'S-TCAC_Pre-ReLU_all_sequential': 'purple',
        'LSC_Pre-ReLU_all_parallel': 'green',
        'LSC_Pre-ReLU_all_sequential': 'orange'
    }
    
    for scenario_key, accs in complexity_results.items():
        label = method_labels.get(scenario_key, scenario_key)
        color = method_colors.get(scenario_key, 'gray')
        plt.plot(budgets, accs, marker='o', linewidth=2, color=color, label=label)
        
    plt.xscale('log')
    plt.xticks(budgets, [str(b) for b in budgets])
    plt.xlabel("Calibration Sample Budget (N)")
    plt.ylabel("Average Multi-Task Accuracy (%)")
    plt.title("Sample Complexity & Stability of Activation Calibration Methods")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sample_complexity.png", dpi=300)
    plt.close()
    
    # Run Composition Bias Sweep
    bias_results = run_composition_bias_sweep(expert_models, base_model, datasets)
    with open("results/metrics_composition_bias.json", "w") as f:
        json.dump(bias_results, f, indent=4)
        
    # Plotting Composition Bias Robustness
    plt.figure(figsize=(9, 5))
    comp_names = ['balanced', 'mnist_heavy', 'fashion_heavy', 'cifar_heavy']
    tcac_joint_accs = [bias_results[c]['TCAC_Joint']['average'] for c in comp_names]
    stcac_joint_accs = [bias_results[c]['S-TCAC_Joint']['average'] for c in comp_names]
    
    x = np.arange(len(comp_names))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, tcac_joint_accs, width, label='Joint TCAC (Seq)', color='magenta')
    rects2 = plt.bar(x + width/2, stcac_joint_accs, width, label='Joint S-TCAC (Seq, α=0.6)', color='orchid')
    
    plt.ylabel("Average Multi-Task Accuracy (%)")
    plt.title("Robustness of Joint Calibration to Dataset Composition Bias (N=384)")
    plt.xticks(x, [c.replace('_', '\n').title() for c in comp_names])
    plt.ylim(0, 100)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()
    plt.savefig("plots/composition_bias.png", dpi=300)
    plt.close()
    
    # Run Multi-Seed Robustness Sweep
    robustness_results = run_multi_seed_robustness_sweep(expert_models, base_model, datasets, N_budgets=[16, 64], num_seeds=5)
    with open("results/metrics_multi_seed.json", "w") as f:
        json.dump(robustness_results, f, indent=4)
        
    # Plotting Multi-Seed Robustness / Variance
    plt.figure(figsize=(9, 6))
    methods_list = [
        ('TCAC_Pre-ReLU_all_sequential', 'TCAC (Seq)', 'blue'),
        ('S-TCAC_Pre-ReLU_all_sequential', 'S-TCAC (Seq, Ours)', 'purple'),
        ('LSC_Pre-ReLU_all_sequential', 'LSC (Seq)', 'orange')
    ]
    
    x = np.arange(2) # Two budgets: N=16, N=64
    width = 0.25
    
    for idx, (scenario_key, label, color) in enumerate(methods_list):
        means = [robustness_results['16'][scenario_key]['mean'], robustness_results['64'][scenario_key]['mean']]
        stds = [robustness_results['16'][scenario_key]['std'], robustness_results['64'][scenario_key]['std']]
        rects = plt.bar(x + (idx - 1)*width, means, width, yerr=stds, label=label, color=color, capsize=5, edgecolor='black', alpha=0.8)
        
        # Add labels on top of bars
        for rect in rects:
            h = rect.get_height()
            plt.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 10), # 10 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    plt.ylabel("Average Multi-Task Accuracy (%)")
    plt.title("Statistical Robustness & Variance across 5 Random Calibration Seeds")
    plt.xticks(x, ["N = 16 (Ultra-Low Budget)", "N = 64 (Low Budget)"])
    plt.ylim(0, 100)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/calibration_variance.png", dpi=300)
    plt.close()
    
    print("\nExperiments completed successfully! Results and plots saved.")

if __name__ == '__main__':
    main()
