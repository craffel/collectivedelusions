import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
import copy

# Ensure results and plot directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Dataset and DataLoader helpers
def get_dataset(task_name, is_train=True):
    transform_gray = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if task_name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=is_train, download=True, transform=transform_gray)
    elif task_name == "fashion":
        return torchvision.datasets.FashionMNIST(root="./data", train=is_train, download=True, transform=transform_gray)
    elif task_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=is_train, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task {task_name}")

def get_dataloader(task_name, batch_size=128, is_train=True):
    dataset = get_dataset(task_name, is_train=is_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2, pin_memory=True)
    return loader

# 2. Model definitions
def get_resnet18_backbone_and_head():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Identity()
    head = nn.Linear(num_features, 10)
    return model, head

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# 3. Training function for experts
def train_expert(task_name, epochs=5, lr=5e-4):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    backbone, head = get_resnet18_backbone_and_head()
    model = MultiTaskModel(backbone, head).to(device)
    
    train_loader = get_dataloader(task_name, batch_size=128, is_train=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    torch.save(backbone.state_dict(), f"models/backbone_{task_name}.pt")
    torch.save(head.state_dict(), f"models/head_{task_name}.pt")
    print(f"Saved expert model for {task_name}")
    return backbone, head

def evaluate_model(backbone, head, dataloader):
    model = MultiTaskModel(backbone, head).to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return (correct / total) * 100.0

# 4. BatchNorm Calibration (BN Calibration) implementation
def calibrate_backbone_bn(merged_backbone, calib_loader, device, expert_backbone=None, use_expert_affine=False):
    # Clone the merged backbone
    calibrated_backbone = copy.deepcopy(merged_backbone).to(device)
    
    # If use_expert_affine is True, copy expert BN weight and bias (gamma, beta)
    if use_expert_affine and expert_backbone is not None:
        expert_state = expert_backbone.state_dict()
        calibrated_state = calibrated_backbone.state_dict()
        for key in calibrated_state.keys():
            if "bn" in key and ("weight" in key or "bias" in key):
                calibrated_state[key] = expert_state[key].clone()
        calibrated_backbone.load_state_dict(calibrated_state)
        
    # Reset running stats of BN layers and configure them to track stats
    for m in calibrated_backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # Cumulative moving average
            m.track_running_stats = True
            m.training = True
            
    calibrated_backbone.eval()  # Set main layers (dropout, etc.) to eval mode
    
    # Ensure BN layers are specifically set to training mode to compute running stats
    for m in calibrated_backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            
    with torch.no_grad():
        for inputs, _ in calib_loader:
            inputs = inputs.to(device)
            _ = calibrated_backbone(inputs)
            
    # Revert BN layers to evaluation mode so they freeze their calibrated stats
    for m in calibrated_backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = False
            
    return calibrated_backbone

# 5. Main pipeline
def main():
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Step A: Train or load expert models
    experts = {}
    for task in tasks:
        backbone_path = f"models/backbone_{task}.pt"
        head_path = f"models/head_{task}.pt"
        
        if os.path.exists(backbone_path) and os.path.exists(head_path):
            print(f"Loading pre-trained expert for {task}...")
            backbone, head = get_resnet18_backbone_and_head()
            backbone = backbone.to(device)
            head = head.to(device)
            backbone.load_state_dict(torch.load(backbone_path, map_location=device))
            head.load_state_dict(torch.load(head_path, map_location=device))
            experts[task] = (backbone, head)
        else:
            backbone, head = train_expert(task, epochs=5, lr=5e-4)
            experts[task] = (backbone, head)
            
    # Evaluate individual experts (Upper Bound)
    print("\n--- Evaluating Expert Models (Individual Upper Bounds) ---")
    expert_accs = {}
    for task in tasks:
        backbone, head = experts[task]
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        acc = evaluate_model(backbone, head, test_loader)
        expert_accs[task] = acc
        print(f"Expert model for {task.upper()} accuracy: {acc:.2f}%")
        
    # Get pre-trained base model weights (to compute task vectors)
    base_backbone, _ = get_resnet18_backbone_and_head()
    base_backbone = base_backbone.to(device)
    base_state = base_backbone.state_dict()
    
    # Step B: Model Merging - Simple Weight Averaging (WA)
    print("\n--- Model Merging: Simple Weight Averaging ---")
    merged_backbone_wa, _ = get_resnet18_backbone_and_head()
    merged_backbone_wa = merged_backbone_wa.to(device)
    wa_state = {}
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            wa_state[key] = torch.stack([experts[t][0].state_dict()[key] for t in tasks]).mean(dim=0)
        else:
            wa_state[key] = experts[tasks[0]][0].state_dict()[key].clone()
    merged_backbone_wa.load_state_dict(wa_state)
    
    # Step C: Model Merging - Task Arithmetic (TA)
    print("\n--- Model Merging: Task Arithmetic ---")
    merged_backbone_ta, _ = get_resnet18_backbone_and_head()
    merged_backbone_ta = merged_backbone_ta.to(device)
    ta_state = {}
    lam = 0.4  # Standard choice
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            task_vectors = []
            for t in tasks:
                vec = experts[t][0].state_dict()[key] - base_state[key]
                task_vectors.append(vec)
            ta_state[key] = base_state[key] + lam * torch.stack(task_vectors).sum(dim=0)
        else:
            ta_state[key] = base_state[key].clone()
    merged_backbone_ta.load_state_dict(ta_state)
    
    # Step D: Evaluate Baselines
    print("\n--- Evaluating Baselines (Before Calibration) ---")
    wa_baselines = {}
    ta_baselines = {}
    for task in tasks:
        test_loader = get_dataloader(task, batch_size=256, is_train=False)
        wa_acc = evaluate_model(merged_backbone_wa, experts[task][1], test_loader)
        ta_acc = evaluate_model(merged_backbone_ta, experts[task][1], test_loader)
        wa_baselines[task] = wa_acc
        ta_baselines[task] = ta_acc
        print(f"Merged WA -> {task.upper()} baseline accuracy: {wa_acc:.2f}%")
        print(f"Merged TA -> {task.upper()} baseline accuracy: {ta_acc:.2f}%")
        
    # Step E: Apply Task-Conditional Activation Calibration (TCAC)
    # TCAC via BN Calibration
    def run_tcac_evaluation(merged_backbone, name_str):
        print(f"\n--- Running TCAC (BN Calibration) for {name_str} ---")
        tcac_shared_accs = {}
        tcac_expert_accs = {}
        
        for task in tasks:
            print(f"Calibrating for task: {task.upper()}...")
            calib_dataset = get_dataset(task, is_train=True)
            calib_subset = Subset(calib_dataset, list(range(128)))
            calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
            
            # Setting 1: TCAC (Shared Affine) -> Merged weights, merged gamma/beta, calibrated running stats
            calibrated_backbone_shared = calibrate_backbone_bn(
                merged_backbone, calib_loader, device, use_expert_affine=False
            )
            test_loader = get_dataloader(task, batch_size=256, is_train=False)
            acc_shared = evaluate_model(calibrated_backbone_shared, experts[task][1], test_loader)
            tcac_shared_accs[task] = acc_shared
            print(f"Merged {name_str} + TCAC (Shared BN Affine) -> {task.upper()} accuracy: {acc_shared:.2f}%")
            
            # Setting 2: TCAC (Task-Specific Affine) -> Merged weights, expert gamma/beta, calibrated running stats
            calibrated_backbone_expert = calibrate_backbone_bn(
                merged_backbone, calib_loader, device, expert_backbone=experts[task][0], use_expert_affine=True
            )
            acc_expert = evaluate_model(calibrated_backbone_expert, experts[task][1], test_loader)
            tcac_expert_accs[task] = acc_expert
            print(f"Merged {name_str} + TCAC (Task-Specific BN Affine) -> {task.upper()} accuracy: {acc_expert:.2f}%")
            
        return tcac_shared_accs, tcac_expert_accs

    wa_tcac_shared, wa_tcac_expert = run_tcac_evaluation(merged_backbone_wa, "Weight Averaging (WA)")
    ta_tcac_shared, ta_tcac_expert = run_tcac_evaluation(merged_backbone_ta, "Task Arithmetic (TA)")
    
    # Save the results to a json file
    output_results = {
        "expert_accs": expert_accs,
        "wa": {
            "baseline": wa_baselines,
            "tcac_shared": wa_tcac_shared,
            "tcac_expert": wa_tcac_expert
        },
        "ta": {
            "baseline": ta_baselines,
            "tcac_shared": ta_tcac_shared,
            "tcac_expert": ta_tcac_expert
        }
    }
    
    with open("results/merging_results.json", "w") as f:
        json.dump(output_results, f, indent=4)
    print("\nResults successfully saved to results/merging_results.json!")
    
    # Step F: Plot variance of activations (before and after calibration)
    # Let's extract the running var values of all BN layers under WA merging before and after calibration (MNIST)
    # We will instantiate backbones and load their stats to extract running vars
    print("\n--- Generating plots for the paper ---")
    
    # Let's calibrate MNIST again to get stats
    calib_dataset = get_dataset("mnist", is_train=True)
    calib_subset = Subset(calib_dataset, list(range(128)))
    calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False)
    
    calibrated_backbone = calibrate_backbone_bn(
        merged_backbone_wa, calib_loader, device, use_expert_affine=False
    )
    
    expert_vars = []
    merged_vars_before = []
    merged_vars_after = []
    bn_names = []
    
    expert_state = experts["mnist"][0].state_dict()
    merged_state_before = merged_backbone_wa.state_dict()
    merged_state_after = calibrated_backbone.state_dict()
    
    count = 0
    for key in expert_state.keys():
        if "bn" in key and "running_var" in key:
            ev = expert_state[key].mean().item()
            mv_bef = merged_state_before[key].mean().item()
            mv_aft = merged_state_after[key].mean().item()
            
            expert_vars.append(np.sqrt(ev))
            merged_vars_before.append(np.sqrt(mv_bef))
            merged_vars_after.append(np.sqrt(mv_aft))
            bn_names.append(f"BN_{count+1}")
            count += 1
            
    plt.figure(figsize=(12, 5))
    x = np.arange(len(bn_names))
    width = 0.25
    
    plt.bar(x - width, expert_vars, width, label='Expert (Native)', color='royalblue')
    plt.bar(x, merged_vars_before, width, label='Merged (Before Calibration)', color='tomato')
    plt.bar(x + width, merged_vars_after, width, label='Merged (TCAC Calibrated)', color='limegreen')
    
    plt.xlabel('BatchNorm Layers')
    plt.ylabel('Average Running Std Dev (Square Root of Var)')
    plt.title('Activation Variance Collapse and Recovery via TCAC (MNIST)')
    plt.xticks(x, bn_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/variance_collapse.png", dpi=300)
    print("Variance collapse plot saved to plots/variance_collapse.png!")

if __name__ == "__main__":
    main()
