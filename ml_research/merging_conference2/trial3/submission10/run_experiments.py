import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import json
import random
import numpy as np

# Set random seed for reproducibility of model initialization
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
torch.backends.cudnn.enabled = False

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Dataset Setup
class GrayscaleToRGB(object):
    def __call__(self, img):
        return img.convert("RGB")

def get_dataloaders(batch_size=128):
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        GrayscaleToRGB(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("Loading MNIST...")
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    print("Loading FashionMNIST...")
    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    print("Loading CIFAR10...")
    train_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    test_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    datasets = {
        'mnist': {'train': train_mnist, 'test': test_mnist},
        'fmnist': {'train': train_fmnist, 'test': test_fmnist},
        'cifar10': {'train': train_cifar10, 'test': test_cifar10}
    }
    
    return datasets

# 2. Model Definition
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        
    def forward(self, x, task_name):
        features = self.backbone(x)
        return self.heads[task_name](features)

def create_model():
    # Load ImageNet pre-trained ResNet-18 backbone
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    resnet = torchvision.models.resnet18(weights=weights)
    
    # Extract backbone features (remove standard fc layer)
    backbone = resnet
    backbone.fc = nn.Identity()
    
    # Task-specific classification heads
    heads = {
        'mnist': nn.Linear(512, 10),
        'fmnist': nn.Linear(512, 10),
        'cifar10': nn.Linear(512, 10)
    }
    
    model = MultiTaskModel(backbone, heads)
    return model

# 3. Training Experts
def train_expert(task_name, datasets, epochs, lr=1e-4):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    model = create_model().to(DEVICE)
    train_dataset = datasets[task_name]['train']
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, task_name)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluate expert
    model.eval()
    test_dataset = datasets[task_name]['test']
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs, task_name)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = correct / total * 100.0
    print(f"Expert {task_name.upper()} Test Accuracy: {test_acc:.2f}%")
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{task_name}_expert.pt")
    return model

def load_or_train_experts(datasets):
    experts = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    epochs = {'mnist': 3, 'fmnist': 5, 'cifar10': 8}
    
    for task in tasks:
        path = f"checkpoints/{task}_expert.pt"
        if os.path.exists(path):
            print(f"Loading pre-trained expert for {task} from {path}...")
            model = create_model().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            experts[task] = model
        else:
            experts[task] = train_expert(task, datasets, epochs[task])
    return experts

# 4. Model Merging (Weight Averaging)
def merge_experts(experts):
    print("\n--- Merging Experts via Weight Averaging ---")
    merged_model = create_model().to(DEVICE)
    
    # Get pre-trained/base parameters to preserve structure
    merged_state_dict = merged_model.state_dict()
    
    # Gather state dicts from all experts
    expert_states = {t: experts[t].state_dict() for t in experts}
    
    # Perform weight averaging on the backbone weights
    for key in merged_state_dict.keys():
        if "backbone" in key:
            # Check if this parameter is in the state dict (e.g. running stats or weights)
            param_list = [expert_states[t][key].float() for t in experts]
            merged_state_dict[key] = torch.stack(param_list, dim=0).mean(dim=0).to(merged_state_dict[key].dtype)
        elif "heads" in key:
            # Heads are task-specific, copy from respective expert directly
            task_key = None
            for t in experts:
                if f"heads.{t}" in key:
                    task_key = t
                    break
            if task_key:
                merged_state_dict[key] = expert_states[task_key][key].clone()
                
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

# 5. Intercepting and Calibrating Activation Statistics
def get_bn_layers(model):
    bn_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers[name] = module
    return bn_layers

def collect_activation_stats(model, dataloader, task_name, device, mode="post_bn"):
    """
    Passes calibration samples through the model to collect statistics at the inputs or outputs of all BatchNorm2d layers.
    Returns:
        dict containing 'mean', 'std', and 'act_rate' (ReLU sparsity proxy) for each BatchNorm layer.
    """
    model.eval()
    bn_layers = get_bn_layers(model)
    
    # Dictionary to collect batch inputs or outputs
    collected_inputs = {name: [] for name in bn_layers.keys()}
    
    # Define hooks to record activations
    handles = []
    def make_hook(name):
        def hook(module, input, output):
            if mode == "pre_bn":
                # input is a tuple of (x,) where x is [B, C, H, W]
                collected_inputs[name].append(input[0].detach().cpu())
            else:
                # output is [B, C, H, W]
                collected_inputs[name].append(output.detach().cpu())
        return hook
        
    for name, module in bn_layers.items():
        handles.append(module.register_forward_hook(make_hook(name)))
        
    # Feed the entire calibration set through the model in a single/few passes
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            _ = model(imgs, task_name)
            
    # Remove hooks
    for h in handles:
        h.remove()
        
    # Compute statistics for each layer
    layer_stats = {}
    for name, inputs in collected_inputs.items():
        if len(inputs) == 0:
            continue
        X = torch.cat(inputs, dim=0) # [N, C, H, W]
        
        # Channel-wise statistics (mean and std across batch, height, width)
        mean_c = X.mean(dim=(0, 2, 3)) # [C]
        std_c = X.std(dim=(0, 2, 3)) # [C]
        
        # Proxy for active rate of ReLU (fraction of entries > 0)
        act_rate_c = (X > 0).float().mean(dim=(0, 2, 3)) # [C]
        
        # Global layer-wise standard deviation (mean of standard deviations across channels)
        global_std = std_c.mean() # scalar
        
        layer_stats[name] = {
            'mean_c': mean_c,
            'std_c': std_c,
            'act_rate_c': act_rate_c,
            'global_std': global_std
        }
        
    return layer_stats

# 6. SMACS Hook Registration
def apply_smacs_hooks(model, stats_expert, stats_merged, tau, epsilon=1e-4, mode="post_bn"):
    """
    Computes SMACS scaling factors and registers pre-hooks or forward-hooks on BatchNorm layers.
    Returns a list of hook handles that must be removed after evaluation.
    """
    bn_layers = get_bn_layers(model)
    handles = []
    
    def make_calibration_pre_hook(scale_vector):
        def hook(module, input):
            # input is a tuple of (x,) where x has shape [B, C, H, W]
            x = input[0]
            # scale_vector has shape [C], match dimensions [1, C, 1, 1]
            scaled_x = x * scale_vector.view(1, -1, 1, 1).to(x.device)
            return (scaled_x,)
        return hook

    def make_calibration_forward_hook(scale_vector):
        def hook(module, input, output):
            # output has shape [B, C, H, W]
            scaled_output = output * scale_vector.view(1, -1, 1, 1).to(output.device)
            return scaled_output
        return hook

    for name, module in bn_layers.items():
        if name not in stats_expert or name not in stats_merged:
            continue
            
        exp = stats_expert[name]
        merg = stats_merged[name]
        
        # Expert stats
        std_exp = exp['std_c']
        global_std_exp = exp['global_std']
        act_rate = exp['act_rate_c']
        
        # Merged stats
        std_merg = merg['std_c']
        global_std_merg = merg['global_std']
        
        # Initialize scale vector
        C = len(std_exp)
        scales = torch.zeros(C)
        
        # Compute layer fallback scale
        fallback_scale = global_std_exp / (global_std_merg + 1e-8)
        
        for c in range(C):
            # SMACS Rule:
            # If the channel is active >= tau and the merged std is stable (> epsilon), use channel scale
            if act_rate[c] >= tau and std_merg[c] > epsilon:
                scales[c] = std_exp[c] / std_merg[c]
            else:
                # Sparsity Trap! Fall back to layer scale
                scales[c] = fallback_scale
                
        # Register the hook on this BatchNorm module
        if mode == "pre_bn":
            handles.append(module.register_forward_pre_hook(make_calibration_pre_hook(scales)))
        else:
            handles.append(module.register_forward_hook(make_calibration_forward_hook(scales)))
        
    return handles

# 7. Evaluate Model on a Task
def evaluate_model(model, test_dataset, task_name):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs, task_name)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total * 100.0

# 8. Head Adaptation (TTA / SFT) on Calibration Set
def run_head_adaptation(model, calibration_loader, task_name, epochs=10, lr=1e-3):
    """
    Adapts ONLY the task-specific classification head on the calibration set.
    """
    # Freeze the backbone
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
        
    # Unfreeze only this task's head
    for name, param in model.heads.named_parameters():
        if f"{task_name}." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Optimizer for only the active head
    head_params = [param for name, param in model.heads.named_parameters() if f"{task_name}." in name]
    optimizer = optim.Adam(head_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Store initial weights to restore later
    initial_state = {k: v.clone() for k, v in model.heads.state_dict().items() if f"{task_name}." in k}
    
    model.train()
    for epoch in range(epochs):
        for imgs, labels in calibration_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, task_name)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Set model back to eval mode
    model.eval()
    
    # Define a cleanup function to restore the head to its initial state
    def restore_head():
        state_dict = model.heads.state_dict()
        for k, v in initial_state.items():
            state_dict[k] = v.clone()
        model.heads.load_state_dict(state_dict)
        # Re-enable backbone requires_grad
        for param in model.parameters():
            param.requires_grad = True
            
    return restore_head

# 9. Pipeline Orchestration
def run_experiments_pipeline(datasets, experts, seed=42, mode="post_bn"):
    mode_label = "Post-BN" if mode == "post_bn" else "Pre-BN"
    print(f"\n==========================================")
    print(f"Executing Calibration Sweep - Seed {seed} ({mode_label})")
    print(f"==========================================")
    
    # Set random seed for this run's calibration subset sampling
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    merged_model = merge_experts(experts)
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    calibration_size = 128
    
    # Sample calibration datasets
    calib_loaders = {}
    calib_stats_expert = {}
    calib_stats_merged = {}
    
    for task in tasks:
        train_data = datasets[task]['train']
        # Randomly sample 128 indices
        indices = random.sample(range(len(train_data)), calibration_size)
        sub_dataset = Subset(train_data, indices)
        calib_loader = DataLoader(sub_dataset, batch_size=calibration_size, shuffle=False)
        calib_loaders[task] = calib_loader
        
        # Collect statistics
        print(f"Collecting calibration stats for {task.upper()} (Seed {seed}, {mode_label})...")
        calib_stats_expert[task] = collect_activation_stats(experts[task], calib_loader, task, DEVICE, mode=mode)
        calib_stats_merged[task] = collect_activation_stats(merged_model, calib_loader, task, DEVICE, mode=mode)
        
    results = {}
    
    # Sweep of threshold tau (representing SMACS)
    tau_sweep = [1.1, 0.95, 0.90, 0.70, 0.50, 0.30, 0.10, -0.1]
    
    for tau in tau_sweep:
        if tau > 1.0:
            method_name = f"{mode_label} LSC"
        elif tau < 0:
            method_name = f"{mode_label} TCAC/SAC"
        else:
            method_name = f"{mode_label} SMACS (tau={tau:.2f})"
            
        results[method_name] = {}
        print(f"\nEvaluating: {method_name}...")
        
        accs = []
        for task in tasks:
            # Apply SMACS hooks
            handles = apply_smacs_hooks(merged_model, calib_stats_expert[task], calib_stats_merged[task], tau, mode=mode)
            
            # Evaluate test set
            acc = evaluate_model(merged_model, datasets[task]['test'], task)
            results[method_name][task] = acc
            accs.append(acc)
            
            # Remove hooks
            for h in handles:
                h.remove()
                
        results[method_name]['average'] = sum(accs) / len(accs)
        print(f"--> Average Acc: {results[method_name]['average']:.2f}% (MNIST: {results[method_name]['mnist']:.2f}%, F-MNIST: {results[method_name]['fmnist']:.2f}%, CIFAR: {results[method_name]['cifar10']:.2f}%)")
        
    # Evaluate Uncalibrated Model
    print("\nEvaluating: Uncalibrated...")
    results['Uncalibrated'] = {}
    accs = []
    for task in tasks:
        acc = evaluate_model(merged_model, datasets[task]['test'], task)
        results['Uncalibrated'][task] = acc
        accs.append(acc)
    results['Uncalibrated']['average'] = sum(accs) / len(accs)
    print(f"--> Average Acc: {results['Uncalibrated']['average']:.2f}%")
    
    # Evaluate Head-only Adaptation
    print("\nEvaluating: Head-only Adaptation...")
    results['Head-only Adaptation'] = {}
    accs = []
    for task in tasks:
        restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task)
        acc = evaluate_model(merged_model, datasets[task]['test'], task)
        results['Head-only Adaptation'][task] = acc
        accs.append(acc)
        restore_fn() # Restore classification head and backbone requires_grad
    results['Head-only Adaptation']['average'] = sum(accs) / len(accs)
    print(f"--> Average Acc: {results['Head-only Adaptation']['average']:.2f}%")
    
    # Evaluate Joint SMACS (best tau) + Head Adaptation
    # First, let's identify the best tau from the sweep
    best_tau = None
    best_avg_acc = 0.0
    for tau in tau_sweep:
        if 0 <= tau <= 1.0:
            method_name = f"{mode_label} SMACS (tau={tau:.2f})"
            if results[method_name]['average'] > best_avg_acc:
                best_avg_acc = results[method_name]['average']
                best_tau = tau
                
    if best_tau is not None:
        joint_method_name = f"{mode_label} SMACS (tau={best_tau:.2f}) + Head Adaptation"
        print(f"\nEvaluating Joint {joint_method_name}...")
        results[joint_method_name] = {}
        accs = []
        for task in tasks:
            # Apply SMACS hooks
            handles = apply_smacs_hooks(merged_model, calib_stats_expert[task], calib_stats_merged[task], best_tau, mode=mode)
            # Run Head Adaptation
            restore_fn = run_head_adaptation(merged_model, calib_loaders[task], task)
            
            acc = evaluate_model(merged_model, datasets[task]['test'], task)
            results[joint_method_name][task] = acc
            accs.append(acc)
            
            # Cleanup
            restore_fn()
            for h in handles:
                h.remove()
        results[joint_method_name]['average'] = sum(accs) / len(accs)
        print(f"--> Average Acc: {results[joint_method_name]['average']:.2f}%")
        
    return results

if __name__ == "__main__":
    datasets = get_dataloaders()
    experts = load_or_train_experts(datasets)
    
    # The Empiricist wants multi-seed validation for overwhelming empirical proof
    seeds = [42, 43, 44]
    multi_seed_results = {}
    
    for seed in seeds:
        # Run both pre_bn and post_bn calibration pipelines and combine results under this seed
        res_pre = run_experiments_pipeline(datasets, experts, seed, mode="pre_bn")
        res_post = run_experiments_pipeline(datasets, experts, seed, mode="post_bn")
        
        # Merge the dictionaries
        combined_res = {}
        combined_res.update(res_pre)
        combined_res.update(res_post)
        
        multi_seed_results[seed] = combined_res
        
    # Save the complete multi-seed results to a JSON file
    with open("results.json", "w") as f:
        json.dump(multi_seed_results, f, indent=4)
        
    print("\nExperiments complete! Results saved to results.json.")
