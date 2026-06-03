import os
import argparse
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed):
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define the Multi-Task Model structure
class MultiTaskResNet18(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        # Load pre-trained ResNet-18 backbone
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = resnet18(weights=weights)
        except Exception:
            self.backbone = resnet18(pretrained=True)
            
        # Remove the final FC layer
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create task-specific heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(self.in_features, num_classes)
            for task, num_classes in num_classes_dict.items()
        })

    def forward(self, x, task):
        features = self.backbone(x)
        logits = self.heads[task](features)
        return logits

# Custom transform to handle grayscale-to-RGB conversion and resizing
get_transform = lambda: transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
def load_data(task, data_dir="./data"):
    transform = get_transform()
    if task == "mnist":
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif task == "fmnist":
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif task == "cifar10":
        # CIFAR-10 is already RGB, but we apply Grayscale to make all datasets 100% structurally identical in domain resizing if needed,
        # or we just use same transform. Let's use same transform since Grayscale on RGB is safe or we can use custom RGB transform.
        # Grayscale on CIFAR-10 is fine or we can omit Grayscale for CIFAR10.
        # Wait, papers converted grayscale datasets to 3-channel RGB, let's make sure we do the same:
        # Grayscale(3) converts 1-channel to 3-channel. If input is 3-channel, it converts it to grayscale and then 3-channel.
        # Wait, the papers said: "convert all grayscale datasets to 3-channel RGB".
        # So CIFAR-10 should remain RGB! Let's do that:
        cifar_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=cifar_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=cifar_transform)
    else:
        raise ValueError(f"Unknown task: {task}")
    return train_dataset, test_dataset

# Train an expert model on a specific task
def train_expert(task, device, epochs=5, lr=5e-4, weight_decay=1e-4, batch_size=128):
    print(f"Training expert on {task}...")
    num_classes_dict = {"mnist": 10, "fmnist": 10, "cifar10": 10}
    model = MultiTaskResNet18(num_classes_dict).to(device)
    
    train_dataset, _ = load_data(task)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Optimize backbone + the task's specific head
    optimizer = optim.AdamW(
        list(model.backbone.parameters()) + list(model.heads[task].parameters()),
        lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, task)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({"Loss": loss.item(), "Acc": 100. * correct / total})
            
    # Save the trained expert model
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/expert_{task}.pth")
    print(f"Expert model on {task} saved.")

# Evaluate a model on a task
def evaluate_model(model, task, device, batch_size=256):
    model.eval()
    _, test_dataset = load_data(task)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, task)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    return acc

# Create merged model weight dict
def merge_weights(base_state, expert_states, mode="WA", coeff=0.3):
    merged_state = copy.deepcopy(base_state)
    tasks = list(expert_states.keys())
    
    if mode == "WA":
        # Average the backbone weights across all experts
        for key in base_state.keys():
            if "backbone" in key:
                if torch.is_floating_point(base_state[key]):
                    merged_state[key] = torch.mean(
                        torch.stack([expert_states[t][key] for t in tasks]), dim=0
                    )
                else:
                    merged_state[key] = base_state[key]
            elif "heads" in key:
                # Keep original expert heads for their respective tasks
                # Extract the task name from the key
                # e.g., heads.mnist.weight
                parts = key.split(".")
                task_head_name = parts[1]
                merged_state[key] = expert_states[task_head_name][key]
                
    elif mode == "TA":
        # Task Arithmetic: base + coeff * sum(expert - base)
        for key in base_state.keys():
            if "backbone" in key:
                if torch.is_floating_point(base_state[key]):
                    task_vectors = []
                    for t in tasks:
                        task_vectors.append(expert_states[t][key] - base_state[key])
                    merged_state[key] = base_state[key] + coeff * torch.sum(torch.stack(task_vectors), dim=0)
                else:
                    merged_state[key] = base_state[key]
            elif "heads" in key:
                parts = key.split(".")
                task_head_name = parts[1]
                merged_state[key] = expert_states[task_head_name][key]
                
    return merged_state

# Get calibration data loaders
def get_calibration_datasets(tasks, cal_size=128, seed=42):
    set_seed(seed)
    cal_subsets = {}
    
    for t in tasks:
        train_dataset, _ = load_data(t)
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        cal_indices = indices[:cal_size]
        cal_subsets[t] = Subset(train_dataset, cal_indices)
        
    return cal_subsets

# Native Task-Agnostic Activation Calibration (N-TAAC)
def run_n_taac(model, cal_subsets, device, momentum=1.0):
    print("Running Native Task-Agnostic Activation Calibration (N-TAAC)...")
    # Put model in train mode to update BatchNorm running statistics
    model.train()
    
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
        
    # Set BatchNorm layers to use the specified momentum and overwrite stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            
    # Combine calibration datasets from all tasks into a single joint dataset
    joint_dataset = ConcatDataset([cal_subsets[t] for t in cal_subsets.keys()])
    # A single forward pass with the entire joint dataset in a single batch
    # (or in chunks if GPU memory is constrained, but 3 * 128 = 384 images is tiny)
    joint_loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)
    
    # Run a single forward pass
    with torch.no_grad():
        for images, _ in joint_loader:
            images = images.to(device)
            # Forward through the backbone
            _ = model.backbone(images)
            break # Single batch is enough
            
    # Return model to eval mode
    model.eval()
    print("N-TAAC calibration complete.")
    return model

# Layer-wise Scaling-only Calibration (LSC)
class LSCOverlay:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.scaling_factors = {} # dict mapping task -> layer_idx -> factor
        self.layers = []
        
        # Locate all BatchNorm2d layers
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.layers.append((name, module))

    def compute_scaling_factors(self, expert_states, cal_subsets, device):
        print("Computing LSC scaling factors...")
        tasks = list(cal_subsets.keys())
        
        # We need to measure original activation standard deviation for each layer, for each expert
        orig_stds = {t: {} for t in tasks}
        merged_stds = {t: {} for t in tasks}
        
        # 1. Measure expert standard deviations
        for t in tasks:
            # Load expert weights
            expert_model = MultiTaskResNet18({"mnist": 10, "fmnist": 10, "cifar10": 10}).to(device)
            expert_model.load_state_dict(expert_states[t])
            expert_model.eval()
            
            # Hook to capture standard deviation of activations
            layer_outputs = {}
            def get_hook(layer_name):
                def hook(module, input, output):
                    # Compute global std across batch, channels, height, and width
                    # Var(X) = E[X^2] - E[X]^2
                    layer_outputs[layer_name] = torch.sqrt(torch.var(output) + 1e-5).item()
                return hook
                
            temp_hooks = []
            for name, module in expert_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    temp_hooks.append(module.register_forward_hook(get_hook(name)))
                    
            # Run forward pass on task t's calibration subset
            cal_loader = DataLoader(cal_subsets[t], batch_size=len(cal_subsets[t]), shuffle=False)
            with torch.no_grad():
                for images, _ in cal_loader:
                    images = images.to(device)
                    _ = expert_model(images, t)
                    break
                    
            orig_stds[t] = copy.deepcopy(layer_outputs)
            
            # Remove hooks
            for h in temp_hooks:
                h.remove()
                
        # 2. Measure merged model standard deviations on calibration subsets
        self.model.eval()
        for t in tasks:
            layer_outputs = {}
            def get_hook(layer_name):
                def hook(module, input, output):
                    layer_outputs[layer_name] = torch.sqrt(torch.var(output) + 1e-5).item()
                return hook
                
            temp_hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    temp_hooks.append(module.register_forward_hook(get_hook(name)))
                    
            cal_loader = DataLoader(cal_subsets[t], batch_size=len(cal_subsets[t]), shuffle=False)
            with torch.no_grad():
                for images, _ in cal_loader:
                    images = images.to(device)
                    _ = self.model(images, t)
                    break
                    
            merged_stds[t] = copy.deepcopy(layer_outputs)
            
            for h in temp_hooks:
                h.remove()
                
        # 3. Compute scaling factors (gamma_l_k = orig_std / merged_std)
        for t in tasks:
            self.scaling_factors[t] = {}
            for name, _ in self.layers:
                self.scaling_factors[t][name] = orig_stds[t][name] / merged_stds[t][name]
                
        print("LSC scaling factors computed.")

    def register_active_hooks(self, active_task):
        self.remove_hooks()
        # Register hooks that perform positive scalar scaling: X_hat = X * gamma
        def get_scale_hook(layer_name):
            def hook(module, input, output):
                gamma = self.scaling_factors[active_task][layer_name]
                return output * gamma
            return hook
            
        for name, module in self.layers:
            self.hooks.append(module.register_forward_hook(get_scale_hook(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# Supervised Head SFT
def run_head_sft(model, cal_subsets, device, epochs=15, lr=1e-3, active_lsc=None):
    print("Running Supervised Head SFT...")
    model.eval() # Freeze backbone in eval mode
    
    # Freeze backbone weights
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True # Classification heads are trainable
            
    tasks = list(cal_subsets.keys())
    criterion = nn.CrossEntropyLoss()
    
    # Optimize each head separately on its respective calibration set
    for t in tasks:
        # If LSC is active, we must apply LSC scaling hooks during SFT
        if active_lsc:
            active_lsc.register_active_hooks(t)
            
        head_params = list(model.heads[t].parameters())
        optimizer = optim.AdamW(head_params, lr=lr)
        
        cal_loader = DataLoader(cal_subsets[t], batch_size=16, shuffle=True)
        
        for epoch in range(epochs):
            for images, labels in cal_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, t)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        if active_lsc:
            active_lsc.remove_hooks()
            
    print("Supervised Head SFT complete.")
    return model

# Unsupervised Head TTA (soft-KL distillation from expert teachers)
def run_head_tta(model, expert_states, cal_subsets, device, epochs=15, lr=1e-3, active_lsc=None):
    print("Running Unsupervised Head TTA...")
    model.eval()
    
    # Freeze backbone weights
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            
    tasks = list(cal_subsets.keys())
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    for t in tasks:
        # Load teacher expert model
        teacher = MultiTaskResNet18({"mnist": 10, "fmnist": 10, "cifar10": 10}).to(device)
        teacher.load_state_dict(expert_states[t])
        teacher.eval()
        
        # Freeze teacher
        for p in teacher.parameters():
            p.requires_grad = False
            
        if active_lsc:
            active_lsc.register_active_hooks(t)
            
        head_params = list(model.heads[t].parameters())
        optimizer = optim.AdamW(head_params, lr=lr)
        
        cal_loader = DataLoader(cal_subsets[t], batch_size=16, shuffle=True)
        
        for epoch in range(epochs):
            for images, _ in cal_loader: # Labels are ignored! Unsupervised TTA!
                images = images.to(device)
                optimizer.zero_grad()
                
                # Student predictions
                student_logits = model(images, t)
                student_log_probs = nn.functional.log_softmax(student_logits, dim=1)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher(images, t)
                    teacher_probs = nn.functional.softmax(teacher_logits, dim=1)
                    
                loss = kl_loss(student_log_probs, teacher_probs)
                loss.backward()
                optimizer.step()
                
        if active_lsc:
            active_lsc.remove_hooks()
            
    print("Unsupervised Head TTA complete.")
    return model

# Main evaluation loop for a specific setup
def run_full_evaluation(expert_states, base_state, cal_subsets, device, merge_mode="WA", coeff=0.3, cal_size=128):
    tasks = ["mnist", "fmnist", "cifar10"]
    num_classes_dict = {t: 10 for t in tasks}
    
    results = {}
    
    # ----------------------------------------------------
    # Setup 1: Baseline (Merged Backbone, Original Heads)
    # ----------------------------------------------------
    print(f"\n--- Evaluating Baseline: {merge_mode} (coeff={coeff}) ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    merged_weights = merge_weights(base_state, expert_states, mode=merge_mode, coeff=coeff)
    model.load_state_dict(merged_weights)
    
    accs = {}
    for t in tasks:
        accs[t] = evaluate_model(model, t, device)
    accs["avg"] = np.mean(list(accs.values()))
    results["baseline"] = accs
    print(f"Baseline: {accs}")
    
    # ----------------------------------------------------
    # Setup 2: N-TAAC Calibration
    # ----------------------------------------------------
    print(f"\n--- Evaluating N-TAAC Calibration: {merge_mode} ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    model = run_n_taac(model, cal_subsets, device)
    
    accs = {}
    for t in tasks:
        accs[t] = evaluate_model(model, t, device)
    accs["avg"] = np.mean(list(accs.values()))
    results["n_taac"] = accs
    print(f"N-TAAC: {accs}")
    
    # ----------------------------------------------------
    # Setup 3: LSC Calibration
    # ----------------------------------------------------
    print(f"\n--- Evaluating LSC Calibration: {merge_mode} ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    
    lsc = LSCOverlay(model)
    lsc.compute_scaling_factors(expert_states, cal_subsets, device)
    
    accs = {}
    for t in tasks:
        lsc.register_active_hooks(t)
        accs[t] = evaluate_model(model, t, device)
        lsc.remove_hooks()
    accs["avg"] = np.mean(list(accs.values()))
    results["lsc"] = accs
    print(f"LSC: {accs}")
    
    # ----------------------------------------------------
    # Setup 4: Head SFT Only
    # ----------------------------------------------------
    print(f"\n--- Evaluating Head SFT Only: {merge_mode} ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    model = run_head_sft(model, cal_subsets, device)
    
    accs = {}
    for t in tasks:
        accs[t] = evaluate_model(model, t, device)
    accs["avg"] = np.mean(list(accs.values()))
    results["head_sft"] = accs
    print(f"Head SFT: {accs}")
    
    # ----------------------------------------------------
    # Setup 5: Head TTA Only
    # ----------------------------------------------------
    print(f"\n--- Evaluating Head TTA Only: {merge_mode} ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    model = run_head_tta(model, expert_states, cal_subsets, device)
    
    accs = {}
    for t in tasks:
        accs[t] = evaluate_model(model, t, device)
    accs["avg"] = np.mean(list(accs.values()))
    results["head_tta"] = accs
    print(f"Head TTA: {accs}")
    
    # ----------------------------------------------------
    # Setup 6: Synergistic N-TAAC + Head SFT
    # ----------------------------------------------------
    print(f"\n--- Evaluating Synergistic N-TAAC + Head SFT ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    model = run_n_taac(model, cal_subsets, device)
    model = run_head_sft(model, cal_subsets, device)
    
    accs = {}
    for t in tasks:
        accs[t] = evaluate_model(model, t, device)
    accs["avg"] = np.mean(list(accs.values()))
    results["n_taac_head_sft"] = accs
    print(f"N-TAAC + Head SFT: {accs}")
    
    # ----------------------------------------------------
    # Setup 7: Synergistic N-TAAC + Head TTA
    # ----------------------------------------------------
    print(f"\n--- Evaluating Synergistic N-TAAC + Head TTA ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    model = run_n_taac(model, cal_subsets, device)
    model = run_head_tta(model, expert_states, cal_subsets, device)
    
    accs = {}
    for t in tasks:
        accs[t] = evaluate_model(model, t, device)
    accs["avg"] = np.mean(list(accs.values()))
    results["n_taac_head_tta"] = accs
    print(f"N-TAAC + Head TTA: {accs}")
    
    # ----------------------------------------------------
    # Setup 8: Synergistic LSC + Head SFT
    # ----------------------------------------------------
    print(f"\n--- Evaluating Synergistic LSC + Head SFT ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    
    lsc = LSCOverlay(model)
    lsc.compute_scaling_factors(expert_states, cal_subsets, device)
    model = run_head_sft(model, cal_subsets, device, active_lsc=lsc)
    
    accs = {}
    for t in tasks:
        lsc.register_active_hooks(t)
        accs[t] = evaluate_model(model, t, device)
        lsc.remove_hooks()
    accs["avg"] = np.mean(list(accs.values()))
    results["lsc_head_sft"] = accs
    print(f"LSC + Head SFT: {accs}")
    
    # ----------------------------------------------------
    # Setup 9: Synergistic LSC + Head TTA
    # ----------------------------------------------------
    print(f"\n--- Evaluating Synergistic LSC + Head TTA ---")
    model = MultiTaskResNet18(num_classes_dict).to(device)
    model.load_state_dict(merged_weights)
    
    lsc = LSCOverlay(model)
    lsc.compute_scaling_factors(expert_states, cal_subsets, device)
    model = run_head_tta(model, expert_states, cal_subsets, device, active_lsc=lsc)
    
    accs = {}
    for t in tasks:
        lsc.register_active_hooks(t)
        accs[t] = evaluate_model(model, t, device)
        lsc.remove_hooks()
    accs["avg"] = np.mean(list(accs.values()))
    results["lsc_head_tta"] = accs
    print(f"LSC + Head TTA: {accs}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="evaluation", choices=["train_experts", "evaluation", "sweep"])
    parser.add_argument("--cal_size", type=int, default=128, help="Calibration sample size per task")
    parser.add_argument("--merge_mode", type=str, default="WA", choices=["WA", "TA"], help="Merging mode")
    parser.add_argument("--coeff", type=float, default=0.3, help="Task arithmetic coefficient lambda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tasks = ["mnist", "fmnist", "cifar10"]
    num_classes_dict = {t: 10 for t in tasks}
    
    if args.mode == "train_experts":
        # Train experts on MNIST, Fashion-MNIST, CIFAR-10
        # Wait, ImageNet pre-trained base model weights
        base_model = MultiTaskResNet18(num_classes_dict).to(device)
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(base_model.state_dict(), "./checkpoints/base_pretrained.pth")
        
        for t in tasks:
            train_expert(t, device, epochs=5)
            
    elif args.mode == "evaluation":
        # Load base model state and expert states
        print("Loading expert checkpoints...")
        try:
            base_state = torch.load("./checkpoints/base_pretrained.pth", map_location=device)
            expert_states = {
                t: torch.load(f"./checkpoints/expert_{t}.pth", map_location=device)
                for t in tasks
            }
        except FileNotFoundError:
            print("Expert checkpoints not found! Please run with --mode train_experts first.")
            exit(1)
            
        # Get calibration data
        cal_subsets = get_calibration_datasets(tasks, cal_size=args.cal_size, seed=args.seed)
        
        # Run evaluation
        results = run_full_evaluation(
            expert_states, base_state, cal_subsets, device,
            merge_mode=args.merge_mode, coeff=args.coeff, cal_size=args.cal_size
        )
        
        # Save results to a json file
        filename = f"results_{args.merge_mode}_{args.coeff}_N{args.cal_size}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filename}")
        
    elif args.mode == "sweep":
        # Run a multi-dimensional sweep for parallel execution in Slurm
        # We can pass different cal_size and coefficients and save separate files
        print(f"Running sweep with merge_mode={args.merge_mode}, coeff={args.coeff}, cal_size={args.cal_size}...")
        base_state = torch.load("./checkpoints/base_pretrained.pth", map_location=device)
        expert_states = {
            t: torch.load(f"./checkpoints/expert_{t}.pth", map_location=device)
            for t in tasks
        }
        cal_subsets = get_calibration_datasets(tasks, cal_size=args.cal_size, seed=args.seed)
        results = run_full_evaluation(
            expert_states, base_state, cal_subsets, device,
            merge_mode=args.merge_mode, coeff=args.coeff, cal_size=args.cal_size
        )
        filename = f"results_sweep_{args.merge_mode}_{args.coeff}_N{args.cal_size}_S{args.seed}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Sweep results saved to {filename}")
