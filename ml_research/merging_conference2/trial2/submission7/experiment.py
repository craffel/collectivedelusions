import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

# Custom dataset wrapper to convert 1-channel images to 3-channels
class RGBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, label

# Hook class to extract activation statistics
class StatisticsHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        # Store output on CPU to avoid GPU memory growth
        self.outputs.append(output.detach().cpu())

    def clear(self):
        self.outputs = []

# Hook class to apply calibration during forward pass
class CalibrationHook:
    def __init__(self, mean_orig, std_orig, mean_merged, std_merged):
        self.mean_orig = mean_orig
        self.std_orig = std_orig
        self.mean_merged = mean_merged
        self.std_merged = std_merged
        self.enabled = True

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        # Perform channel-wise affine transformation in-place or returned
        # output shape is (B, C, H, W)
        # mean/std are of shape (1, C, 1, 1) and on correct device
        device = output.device
        mo = self.mean_orig.to(device)
        so = self.std_orig.to(device)
        mm = self.mean_merged.to(device)
        sm = self.std_merged.to(device)
        
        calibrated = (output - mm) / sm * so + mo
        return calibrated

def get_bn_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            layers.append((name, module))
    return layers

def train_expert(model, train_loader, val_loader, dataset_name, device, epochs=5):
    print(f"\n--- Training Expert for {dataset_name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Use Adam optimizer as it converges fast and reliably
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        t0 = time.time()
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
        epoch_acc = (correct / total) * 100
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {time.time()-t0:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"Expert {dataset_name} finished. Best Val Acc: {best_acc:.2f}%")
    return best_weights

def evaluate_model(model, test_loader, device):
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
    return (correct / total) * 100

def measure_inference_time(model, test_loader, device, num_batches=10):
    model.eval()
    # Warmup
    images, _ = next(iter(test_loader))
    images = images.to(device)
    for _ in range(5):
        _ = model(images)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / min(num_batches, len(test_loader)) * 1000  # ms per batch

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Prepare Data
    # For speed and stability, we use smaller subsets for training if required, but full sets for evaluation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if False else transforms.Normalize((0.5,), (0.5,))
    ])
    
    # We will build 3-channel datasets
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    fashion_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    
    # Convert MNIST and FashionMNIST to RGB (3 channels)
    mnist_train_rgb = RGBDataset(mnist_train)
    mnist_test_rgb = RGBDataset(mnist_test)
    fashion_train_rgb = RGBDataset(fashion_train)
    fashion_test_rgb = RGBDataset(fashion_test)
    
    # To train quickly, we'll sub-sample training datasets
    # ResNet-18 learns MNIST and FashionMNIST extremely fast. Let's use 10000 samples for training.
    # For CIFAR-10, let's use 15000 samples.
    sub_mnist_train = Subset(mnist_train_rgb, list(range(10000)))
    sub_fashion_train = Subset(fashion_train_rgb, list(range(10000)))
    sub_cifar_train = Subset(cifar_train, list(range(15000)))
    
    train_loaders = {
        "mnist": DataLoader(sub_mnist_train, batch_size=128, shuffle=True, num_workers=2),
        "fashion": DataLoader(sub_fashion_train, batch_size=128, shuffle=True, num_workers=2),
        "cifar10": DataLoader(sub_cifar_train, batch_size=128, shuffle=True, num_workers=2)
    }
    
    # Validation loaders (using a subset of test for speed during training, but full for final eval)
    val_loaders = {
        "mnist": DataLoader(Subset(mnist_test_rgb, list(range(1000))), batch_size=256, shuffle=False),
        "fashion": DataLoader(Subset(fashion_test_rgb, list(range(1000))), batch_size=256, shuffle=False),
        "cifar10": DataLoader(Subset(cifar_test, list(range(1000))), batch_size=256, shuffle=False)
    }
    
    test_loaders = {
        "mnist": DataLoader(mnist_test_rgb, batch_size=256, shuffle=False, num_workers=2),
        "fashion": DataLoader(fashion_test_rgb, batch_size=256, shuffle=False, num_workers=2),
        "cifar10": DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=2)
    }
    
    # Calibration loader (128 samples per task as in the paper)
    calib_loaders = {
        "mnist": DataLoader(Subset(mnist_train_rgb, list(range(args.calib_size))), batch_size=args.calib_size, shuffle=False),
        "fashion": DataLoader(Subset(fashion_train_rgb, list(range(args.calib_size))), batch_size=args.calib_size, shuffle=False),
        "cifar10": DataLoader(Subset(cifar_train, list(range(args.calib_size))), batch_size=args.calib_size, shuffle=False)
    }

    # 2. Get pretrained ResNet-18 base weights
    print("\n--- Initializing Pretrained ResNet-18 Base Model ---")
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    base_model = torchvision.models.resnet18(weights=weights)
    
    # Standardize the base model's task head to 10 classes
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_state_dict = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    
    # Save the initial base model
    torch.save(base_state_dict, os.path.join(args.checkpoint_dir, "base_model.pth"))
    
    # 3. Train or Load Experts
    expert_states = {}
    tasks = ["mnist", "fashion", "cifar10"]
    
    for task in tasks:
        ckpt_path = os.path.join(args.checkpoint_dir, f"expert_{task}.pth")
        if os.path.exists(ckpt_path) and not args.force_train:
            print(f"Loading cached expert for {task} from {ckpt_path}...")
            expert_states[task] = torch.load(ckpt_path, map_location="cpu")
        else:
            # Re-initialize with pretrained weights
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(base_state_dict)
            
            expert_states[task] = train_expert(
                model, train_loaders[task], val_loaders[task], task, device, epochs=args.epochs
            )
            torch.save(expert_states[task], ckpt_path)
            
    # 4. Evaluate individual experts on their own tasks (as sanity checks)
    print("\n--- Evaluating Expert Models on Their Respective Tasks ---")
    expert_accuracies = {}
    for task in tasks:
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(expert_states[task])
        model = model.to(device)
        acc = evaluate_model(model, test_loaders[task], device)
        expert_accuracies[task] = acc
        print(f"Expert {task} accuracy: {acc:.2f}%")
        
    # 5. Merge Models
    print(f"\n--- Merging Expert Backbones using {args.merge_method.upper()} ---")
    backbone_keys = [k for k in base_state_dict.keys() if not k.startswith("fc")]
    
    if args.merge_method == "wa":
        merged_backbone = {}
        for k in backbone_keys:
            if torch.is_floating_point(expert_states[tasks[0]][k]):
                merged_backbone[k] = torch.stack([expert_states[task][k] for task in tasks]).mean(dim=0)
            else:
                merged_backbone[k] = expert_states[tasks[0]][k].clone()
        print("Merged backbone using Weight Averaging (WA).")
        best_lambda = 0.0
    else: # ta
        # Task Vectors
        task_vectors = {task: {} for task in tasks}
        for task in tasks:
            for k in backbone_keys:
                if torch.is_floating_point(expert_states[task][k]) and not ("running_" in k or "num_batches_tracked" in k):
                    task_vectors[task][k] = expert_states[task][k] - base_state_dict[k]
                else:
                    task_vectors[task][k] = torch.zeros_like(expert_states[task][k])
                    
        lam = args.ta_lambda
        merged_backbone = {}
        for k in backbone_keys:
            if torch.is_floating_point(base_state_dict[k]):
                if "running_" in k:
                    # Average the running stats across experts rather than doing Task Arithmetic subtraction
                    merged_backbone[k] = torch.stack([expert_states[task][k] for task in tasks]).mean(dim=0)
                else:
                    merged_backbone[k] = base_state_dict[k] + lam * torch.stack([task_vectors[task][k] for task in tasks]).sum(dim=0)
            else:
                merged_backbone[k] = base_state_dict[k].clone()
        print(f"Merged backbone using Task Arithmetic (TA) with lambda = {lam}.")
        best_lambda = lam
    
    # Define the merged state dict helper
    def get_merged_model_state(task):
        state = {}
        for k in backbone_keys:
            state[k] = merged_backbone[k]
        state["fc.weight"] = expert_states[task]["fc.weight"]
        state["fc.bias"] = expert_states[task]["fc.bias"]
        return state

    # 6. Extraction of Statistics (Calibration Phase)
    print("\n--- Extraction of Activation Statistics (Calibration Phase) ---")
    # For each task, we need to extract:
    # 1. Native statistics (orig) from the expert model on task k's calibration data
    # 2. Merged statistics (merged) from the merged model on task k's calibration data
    
    # Initialize structures to hold statistics
    # Structure: stats[task_name][layer_name] = {"mean_orig":, "std_orig":, "mean_merged":, "std_merged":}
    stats = {task: {} for task in tasks}
    
    # Get BN layers to attach hooks
    dummy_model = torchvision.models.resnet18()
    bn_layers = get_bn_layers(dummy_model)
    print(f"Found {len(bn_layers)} BatchNorm2d layers for calibration.")
    
    for task in tasks:
        print(f"Extracting statistics for task: {task}...")
        calib_batch, _ = next(iter(calib_loaders[task]))
        calib_batch = calib_batch.to(device)
        
        # A. Native Expert Model Statistics
        expert_model = torchvision.models.resnet18()
        expert_model.fc = nn.Linear(expert_model.fc.in_features, 10)
        expert_model.load_state_dict(expert_states[task])
        expert_model = expert_model.to(device)
        expert_model.eval()
        
        # Register statistics hooks
        hooks = []
        hook_objects = {}
        for name, module in expert_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook_obj = StatisticsHook()
                h = module.register_forward_hook(hook_obj)
                hooks.append(h)
                hook_objects[name] = hook_obj
                
        # Forward pass
        with torch.no_grad():
            _ = expert_model(calib_batch)
            
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Compute and store native stats
        for name, hook_obj in hook_objects.items():
            # Compute channel-wise mean and std
            all_outputs = torch.cat(hook_obj.outputs, dim=0) # shape (128, C, H, W)
            C = all_outputs.shape[1]
            flat = all_outputs.transpose(0, 1).reshape(C, -1)
            mean_orig = flat.mean(dim=1).view(1, C, 1, 1)
            std_orig = flat.std(dim=1).view(1, C, 1, 1) + 1e-5
            
            stats[task][name] = {
                "mean_orig": mean_orig,
                "std_orig": std_orig
            }
            
        # B. Merged Model Statistics
        merged_model = torchvision.models.resnet18()
        merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
        merged_model.load_state_dict(get_merged_model_state(task))
        merged_model = merged_model.to(device)
        merged_model.eval()
        
        # Register statistics hooks
        hooks = []
        hook_objects = {}
        for name, module in merged_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook_obj = StatisticsHook()
                h = module.register_forward_hook(hook_obj)
                hooks.append(h)
                hook_objects[name] = hook_obj
                
        # Forward pass
        with torch.no_grad():
            _ = merged_model(calib_batch)
            
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Compute and store merged stats, and calculate the Variance Ratio
        for name, hook_obj in hook_objects.items():
            all_outputs = torch.cat(hook_obj.outputs, dim=0) # shape (128, C, H, W)
            C = all_outputs.shape[1]
            flat = all_outputs.transpose(0, 1).reshape(C, -1)
            mean_merged = flat.mean(dim=1).view(1, C, 1, 1)
            std_merged = flat.std(dim=1).view(1, C, 1, 1) + 1e-5
            
            stats[task][name]["mean_merged"] = mean_merged
            stats[task][name]["std_merged"] = std_merged
            
            # Compute variance ratio
            var_ratio = (std_merged ** 2) / (stats[task][name]["std_orig"] ** 2)
            stats[task][name]["var_ratio_mean"] = var_ratio.mean().item()
            
    # 7. Analyze and Plot Variance Collapse
    # We will log and plot how variance collapses across layers
    print("\n--- Variance Collapse Analysis ---")
    layer_names = [name for name, _ in bn_layers]
    
    plt.figure(figsize=(10, 6))
    for task in tasks:
        ratios = [stats[task][name]["var_ratio_mean"] for name in layer_names]
        plt.plot(layer_names, ratios, marker='o', label=f"{task.upper()} Variance Ratio")
        print(f"Task: {task} | Average Variance Ratio across all layers: {np.mean(ratios):.4f} | Minimum Ratio: {np.min(ratios):.4f} ({layer_names[np.argmin(ratios)]})")
        
    plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Alignment (1.0)')
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Collapse Threshold (0.5)')
    plt.xticks(rotation=90)
    plt.ylabel("Variance Ratio (Merged / Native)")
    plt.title("Variance Collapse Across Layers in Merged Backbone")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/variance_collapse_{args.merge_method}_l{best_lambda}_c{args.calib_size}.png", dpi=150)
    print(f"Saved variance collapse analysis plot to plots/variance_collapse_{args.merge_method}_l{best_lambda}_c{args.calib_size}.png")
    
    # 8. Evaluation of Calibration Strategies
    print("\n--- Evaluating Layer-Selective Calibration Strategies ---")
    
    # We will evaluate different strategies on the complete test datasets
    # Strategies:
    # 1. "no-calib" (standard merged model)
    # 2. "full-calib" (standard TCAC, calibrating all 20 layers)
    # 3. "random-k": Calibrate random k% of layers (averaged over 3 runs)
    # 4. "uniform-k": Calibrate every N-th layer
    # 5. "avcs-k": Our proposed Adaptive Variance-Collapse-driven Selection (calibrating the top-k layers with worst collapse)
    # 6. "avcs-threshold": Calibrating all layers with variance ratio < threshold delta
    
    results = {}
    
    def get_calibrated_stats(task, selected_layers):
        # selected_layers: list of layer names to calibrate
        calib_batch, _ = next(iter(calib_loaders[task]))
        calib_batch = calib_batch.to(device)
        
        calib_stats = {}
        if len(selected_layers) == 0:
            return calib_stats
            
        # Process selected_layers in the forward pass order of the model
        ordered_selected = [name for name, _ in bn_layers if name in selected_layers]
        
        for name in ordered_selected:
            # Set up merged model
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(get_merged_model_state(task))
            model = model.to(device)
            model.eval()
            
            # Register calibration hooks for ALREADY calibrated layers in calib_stats
            active_hooks = []
            for prev_name in calib_stats.keys():
                mean_orig = stats[task][prev_name]["mean_orig"]
                std_orig = stats[task][prev_name]["std_orig"]
                mean_merged = calib_stats[prev_name]["mean_merged"]
                std_merged = calib_stats[prev_name]["std_merged"]
                
                hook_obj = CalibrationHook(mean_orig, std_orig, mean_merged, std_merged)
                module = dict(model.named_modules())[prev_name]
                h = module.register_forward_hook(hook_obj)
                active_hooks.append(h)
                
            # Register statistics hook for the current layer
            stat_hook = StatisticsHook()
            module = dict(model.named_modules())[name]
            h = module.register_forward_hook(stat_hook)
            active_hooks.append(h)
            
            # Forward pass
            with torch.no_grad():
                _ = model(calib_batch)
                
            # Remove all hooks
            for h in active_hooks:
                h.remove()
                
            # Compute stats for current layer
            all_outputs = torch.cat(stat_hook.outputs, dim=0)
            C = all_outputs.shape[1]
            flat = all_outputs.transpose(0, 1).reshape(C, -1)
            mean_merged = flat.mean(dim=1).view(1, C, 1, 1)
            std_merged = flat.std(dim=1).view(1, C, 1, 1) + 1e-5
            
            calib_stats[name] = {
                "mean_merged": mean_merged,
                "std_merged": std_merged
            }
            
        return calib_stats

    def run_calibrated_eval(selected_layers_dict):
        # selected_layers_dict: {task_name: list of layer_names to calibrate}
        task_accs = {}
        task_latencies = {}
        
        for task in tasks:
            # Set up the merged model
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(get_merged_model_state(task))
            model = model.to(device)
            model.eval()
            
            layers_to_calib = selected_layers_dict[task]
            
            # Sequentially extract calibrated statistics for the selected layers
            calib_stats = get_calibrated_stats(task, layers_to_calib)
            
            # Register calibration hooks for selected layers
            active_hooks = []
            for name in layers_to_calib:
                mean_orig = stats[task][name]["mean_orig"]
                std_orig = stats[task][name]["std_orig"]
                mean_merged = calib_stats[name]["mean_merged"]
                std_merged = calib_stats[name]["std_merged"]
                
                hook_obj = CalibrationHook(mean_orig, std_orig, mean_merged, std_merged)
                module = dict(model.named_modules())[name]
                h = module.register_forward_hook(hook_obj)
                active_hooks.append(h)
                
            # Evaluate
            acc = evaluate_model(model, test_loaders[task], device)
            latency = measure_inference_time(model, test_loaders[task], device, num_batches=15)
            
            task_accs[task] = acc
            task_latencies[task] = latency
            
            # Remove hooks
            for h in active_hooks:
                h.remove()
                
        return task_accs, task_latencies

    def get_svcs_layers(task, k):
        calib_batch, _ = next(iter(calib_loaders[task]))
        calib_batch = calib_batch.to(device)
        
        selected_layers = []
        sequential_stats = {}
        
        for step in range(k):
            # Set up merged model with currently selected calibration hooks
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(get_merged_model_state(task))
            model = model.to(device)
            model.eval()
            
            calibration_hooks = []
            # Register calibration hooks for selected layers
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name in selected_layers:
                    mean_orig = stats[task][name]["mean_orig"]
                    std_orig = stats[task][name]["std_orig"]
                    mean_merged = sequential_stats[name]["mean_merged"]
                    std_merged = sequential_stats[name]["std_merged"]
                    
                    hook_obj = CalibrationHook(mean_orig, std_orig, mean_merged, std_merged)
                    h = module.register_forward_hook(hook_obj)
                    calibration_hooks.append(h)
                    
            # Register statistics hooks for all BN layers to measure current stats after active calibration
            stat_hooks = []
            stat_hook_objects = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    hook_obj = StatisticsHook()
                    h = module.register_forward_hook(hook_obj)
                    stat_hooks.append(h)
                    stat_hook_objects[name] = hook_obj
                    
            # Forward pass
            with torch.no_grad():
                _ = model(calib_batch)
                
            # Remove all hooks
            for h in calibration_hooks:
                h.remove()
            for h in stat_hooks:
                h.remove()
                
            # Compute variance ratios for remaining layers
            ratios = {}
            for name, hook_obj in stat_hook_objects.items():
                if name in selected_layers:
                    continue
                all_outputs = torch.cat(hook_obj.outputs, dim=0)
                C = all_outputs.shape[1]
                flat = all_outputs.transpose(0, 1).reshape(C, -1)
                std_current = flat.std(dim=1).view(1, C, 1, 1) + 1e-5
                
                # Compute current variance ratio
                var_ratio = (std_current ** 2) / (stats[task][name]["std_orig"] ** 2)
                ratios[name] = var_ratio.mean().item()
                
            # Select the remaining layer with the lowest variance ratio (worst collapse)
            if len(ratios) == 0:
                break
            next_layer = min(ratios, key=ratios.get)
            
            # Compute and store the sequential statistics for next_layer
            hook_obj = stat_hook_objects[next_layer]
            all_outputs = torch.cat(hook_obj.outputs, dim=0)
            C = all_outputs.shape[1]
            flat = all_outputs.transpose(0, 1).reshape(C, -1)
            mean_merged = flat.mean(dim=1).view(1, C, 1, 1)
            std_merged = flat.std(dim=1).view(1, C, 1, 1) + 1e-5
            
            sequential_stats[next_layer] = {
                "mean_merged": mean_merged,
                "std_merged": std_merged
            }
            
            selected_layers.append(next_layer)
            
        return selected_layers

    # A. Baseline: No Calibration
    print("\nRunning Baseline: No Calibration...")
    no_calib_layers = {task: [] for task in tasks}
    no_calib_accs, no_calib_lats = run_calibrated_eval(no_calib_layers)
    results["No Calibration"] = {
        "accs": no_calib_accs,
        "avg_acc": np.mean(list(no_calib_accs.values())),
        "latency": np.mean(list(no_calib_lats.values())),
        "num_calibrated_layers": 0,
        "pct_calibrated_layers": 0.0
    }
    print(f"  Avg Acc: {results['No Calibration']['avg_acc']:.2f}% | Latency: {results['No Calibration']['latency']:.2f}ms")

    # B. Baseline: Full Calibration (TCAC)
    print("\nRunning Baseline: Full Calibration (TCAC)...")
    full_calib_layers = {task: layer_names for task in tasks}
    full_calib_accs, full_calib_lats = run_calibrated_eval(full_calib_layers)
    results["Full Calibration (TCAC)"] = {
        "accs": full_calib_accs,
        "avg_acc": np.mean(list(full_calib_accs.values())),
        "latency": np.mean(list(full_calib_lats.values())),
        "num_calibrated_layers": len(layer_names),
        "pct_calibrated_layers": 100.0
    }
    print(f"  Avg Acc: {results['Full Calibration (TCAC)']['avg_acc']:.2f}% | Latency: {results['Full Calibration (TCAC)']['latency']:.2f}ms")

    # C. Sweep different values of k (percentage of layers calibrated)
    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    total_layers = len(layer_names)
    
    # Store sweep results for plotting
    sweep_data = {
        "pct": [0] + pcts + [100],
        "random_acc": [results["No Calibration"]["avg_acc"]],
        "random_lat": [results["No Calibration"]["latency"]],
        "avcs_acc": [results["No Calibration"]["avg_acc"]],
        "avcs_lat": [results["No Calibration"]["latency"]],
        "svcs_acc": [results["No Calibration"]["avg_acc"]],
        "svcs_lat": [results["No Calibration"]["latency"]]
    }
    
    for pct in pcts:
        k = int(np.round(total_layers * (pct / 100.0)))
        k = max(1, min(k, total_layers))
        print(f"\nSweeping percentage: {pct}% (Calibrating {k}/{total_layers} layers)...")
        
        # 1. RANDOM-K SELECTION
        # We average over 3 random seeds to get robust results
        rand_accs_list = []
        rand_lats_list = []
        for trial_seed in [101, 102, 103]:
            random.seed(trial_seed)
            rand_layers = {task: random.sample(layer_names, k) for task in tasks}
            accs, lats = run_calibrated_eval(rand_layers)
            rand_accs_list.append(np.mean(list(accs.values())))
            rand_lats_list.append(np.mean(list(lats.values())))
            
        avg_rand_acc = np.mean(rand_accs_list)
        avg_rand_lat = np.mean(rand_lats_list)
        sweep_data["random_acc"].append(avg_rand_acc)
        sweep_data["random_lat"].append(avg_rand_lat)
        print(f"  Random-{pct}% | Avg Acc: {avg_rand_acc:.2f}% | Latency: {avg_rand_lat:.2f}ms")
        
        # 2. ADAPTIVE VARIANCE-COLLAPSE-DRIVEN SELECTION (AVCS)
        # We select the top-k layers with the worst variance collapse (lowest variance ratio)
        avcs_layers = {}
        for task in tasks:
            # Sort layers by var_ratio_mean in ascending order
            sorted_layers = sorted(layer_names, key=lambda name: stats[task][name]["var_ratio_mean"])
            avcs_layers[task] = sorted_layers[:k]
            
        avcs_accs, avcs_lats = run_calibrated_eval(avcs_layers)
        avg_avcs_acc = np.mean(list(avcs_accs.values()))
        avg_avcs_lat = np.mean(list(avcs_lats.values()))
        sweep_data["avcs_acc"].append(avg_avcs_acc)
        sweep_data["avcs_lat"].append(avg_avcs_lat)
        print(f"  AVCS-{pct}%   | Avg Acc: {avg_avcs_acc:.2f}% | Latency: {avg_avcs_lat:.2f}ms")
        
        # 3. SEQUENTIAL VARIANCE-COLLAPSE SELECTION (SVCS) - Our proposed cascading fix
        svcs_layers = {}
        for task in tasks:
            svcs_layers[task] = get_svcs_layers(task, k)
        
        svcs_accs, svcs_lats = run_calibrated_eval(svcs_layers)
        avg_svcs_acc = np.mean(list(svcs_accs.values()))
        avg_svcs_lat = np.mean(list(svcs_lats.values()))
        sweep_data["svcs_acc"].append(avg_svcs_acc)
        sweep_data["svcs_lat"].append(avg_svcs_lat)
        print(f"  SVCS-{pct}%   | Avg Acc: {avg_svcs_acc:.2f}% | Latency: {avg_svcs_lat:.2f}ms")
        
        # Record details
        results[f"Random {pct}%"] = {
            "avg_acc": avg_rand_acc,
            "latency": avg_rand_lat,
            "num_calibrated_layers": k,
            "pct_calibrated_layers": pct
        }
        results[f"AVCS {pct}%"] = {
            "avg_acc": avg_avcs_acc,
            "latency": avg_avcs_lat,
            "num_calibrated_layers": k,
            "pct_calibrated_layers": pct
        }
        results[f"SVCS {pct}%"] = {
            "avg_acc": avg_svcs_acc,
            "latency": avg_svcs_lat,
            "num_calibrated_layers": k,
            "pct_calibrated_layers": pct
        }
        
    sweep_data["random_acc"].append(results["Full Calibration (TCAC)"]["avg_acc"])
    sweep_data["random_lat"].append(results["Full Calibration (TCAC)"]["latency"])
    sweep_data["avcs_acc"].append(results["Full Calibration (TCAC)"]["avg_acc"])
    sweep_data["avcs_lat"].append(results["Full Calibration (TCAC)"]["latency"])
    sweep_data["svcs_acc"].append(results["Full Calibration (TCAC)"]["avg_acc"])
    sweep_data["svcs_lat"].append(results["Full Calibration (TCAC)"]["latency"])
    
    # 9. Plot the Pareto Frontier: Accuracy vs. Percentage of Layers Calibrated
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_data["pct"], sweep_data["svcs_acc"], marker='s', color='orange', linewidth=2.5, label="LS-TCAC (SVCS Selection - Proposed)")
    plt.plot(sweep_data["pct"], sweep_data["avcs_acc"], marker='o', color='g', linewidth=2, label="LS-TCAC (AVCS Selection)")
    plt.plot(sweep_data["pct"], sweep_data["random_acc"], marker='x', color='b', linestyle='--', label="Random Selection Baseline")
    plt.axhline(y=results["Full Calibration (TCAC)"]["avg_acc"], color='r', linestyle=':', label="Full TCAC (100% layers)")
    plt.axhline(y=results["No Calibration"]["avg_acc"], color='black', linestyle='-.', label="No Calibration")
    
    plt.xlabel("Percentage of Layers Calibrated (%)")
    plt.ylabel("Multi-Task Average Accuracy (%)")
    plt.title(f"LS-TCAC ({args.merge_method.upper()} l={best_lambda} c={args.calib_size}): Accuracy vs. Percentage of Calibrated Layers")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/accuracy_vs_layers_{args.merge_method}_l{best_lambda}_c{args.calib_size}.png", dpi=150)
    print(f"Saved accuracy vs. layers plot to plots/accuracy_vs_layers_{args.merge_method}_l{best_lambda}_c{args.calib_size}.png")
    
    # 10. Plot Latency vs. Percentage of Layers Calibrated to prove inference speedup
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_data["pct"], sweep_data["svcs_lat"], marker='s', color='orange', linewidth=2, label="LS-TCAC (SVCS Latency)")
    plt.plot(sweep_data["pct"], sweep_data["avcs_lat"], marker='o', color='purple', linewidth=1.5, linestyle='--', label="LS-TCAC (AVCS Latency)")
    plt.axhline(y=results["No Calibration"]["latency"], color='black', linestyle='-.', label="No Calibration Latency")
    plt.axhline(y=results["Full Calibration (TCAC)"]["latency"], color='r', linestyle=':', label="Full TCAC Latency")
    plt.xlabel("Percentage of Layers Calibrated (%)")
    plt.ylabel("Inference Latency per Batch (ms)")
    plt.title(f"LS-TCAC ({args.merge_method.upper()} l={best_lambda} c={args.calib_size}): Latency vs. Percentage of Calibrated Layers")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/latency_vs_layers_{args.merge_method}_l{best_lambda}_c{args.calib_size}.png", dpi=150)
    print(f"Saved latency vs. layers plot to plots/latency_vs_layers_{args.merge_method}_l{best_lambda}_c{args.calib_size}.png")
    
    # 11. Print Summary Table
    print("\n================== EXPERIMENT SUMMARY ==================")
    print(f"{'Strategy':<30} | {'Layers Calibrated (%)':<23} | {'Avg Accuracy (%)':<18} | {'Latency (ms)':<12}")
    print("-" * 92)
    print(f"{'No Calibration':<30} | {0:<23} | {results['No Calibration']['avg_acc']:.2f}% | {results['No Calibration']['latency']:.2f}ms")
    print(f"{'Full TCAC (Baseline)':<30} | {100:<23} | {results['Full Calibration (TCAC)']['avg_acc']:.2f}% | {results['Full Calibration (TCAC)']['latency']:.2f}ms")
    print("-" * 92)
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print(f"{f'LS-TCAC (SVCS) {pct}%':<30} | {pct:<23} | {results[f'SVCS {pct}%']['avg_acc']:.2f}% | {results[f'SVCS {pct}%']['latency']:.2f}ms")
        print(f"{f'LS-TCAC (AVCS) {pct}%':<30} | {pct:<23} | {results[f'AVCS {pct}%']['avg_acc']:.2f}% | {results[f'AVCS {pct}%']['latency']:.2f}ms")
        print(f"{f'Random {pct}%':<30} | {pct:<23} | {results[f'Random {pct}%']['avg_acc']:.2f}% | {results[f'Random {pct}%']['latency']:.2f}ms")
        print("-" * 92)
    print("========================================================")
    
    # Save results to a json file
    with open(f"results_{args.merge_method}_l{best_lambda}_c{args.calib_size}.json", "w") as f:
        json.dump({
            "expert_accuracies": expert_accuracies,
            "best_lambda": best_lambda,
            "sweep_data": sweep_data,
            "individual_results": {k: {"avg_acc": v["avg_acc"], "latency": v["latency"], "num_calibrated_layers": v["num_calibrated_layers"], "pct_calibrated_layers": v["pct_calibrated_layers"]} for k, v in results.items() if k not in ["No Calibration", "Full Calibration (TCAC)"]}
        }, f, indent=4)
    print(f"Saved experiment results to results_{args.merge_method}_l{best_lambda}_c{args.calib_size}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train experts if not cached")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib_size", type=int, default=128, help="Size of calibration dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save expert weights")
    parser.add_argument("--force_train", action="store_true", help="Force retraining of experts")
    parser.add_argument("--merge_method", type=str, default="ta", choices=["ta", "wa"], help="Merging method (ta or wa)")
    parser.add_argument("--ta_lambda", type=float, default=0.5, help="Task Arithmetic scaling coefficient lambda")
    args = parser.parse_args()
    main(args)
