import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.func import functional_call
from models import CNNBackbone, TaskHead

# Reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup directories
os.makedirs("results", exist_ok=True)

# Datasets mapping
dataset_classes = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define parameter-to-layer mapping for the CNNBackbone
def get_key_to_layer_idx(state_dict):
    key_to_layer = {}
    for key in state_dict.keys():
        if "conv1" in key:
            key_to_layer[key] = 0
        elif "bn1" in key:
            key_to_layer[key] = 1
        elif "conv2" in key:
            key_to_layer[key] = 2
        elif "bn2" in key:
            key_to_layer[key] = 3
        elif "fc1" in key:
            key_to_layer[key] = 4
        else:
            key_to_layer[key] = 4  # Default/fallback
    return key_to_layer

# Compute empirical Fisher Information for each expert
def compute_empirical_fisher(backbone, head, dataloader, num_samples=256):
    backbone.eval()
    head.eval()
    fisher = {k: torch.zeros_like(p) for k, p in backbone.named_parameters() if p.requires_grad}
    count = 0
    
    for images, _ in dataloader:
        images = images.to(device)
        features = backbone(images)
        outputs = head(features)
        preds = outputs.argmax(dim=-1)
        
        for i in range(images.size(0)):
            features_i = backbone(images[i:i+1])
            outputs_i = head(features_i)
            loss = nn.CrossEntropyLoss()(outputs_i, preds[i:i+1])
            
            grads = torch.autograd.grad(loss, [p for p in backbone.parameters() if p.requires_grad], allow_unused=True)
            for (k, p), grad in zip([(k, p) for k, p in backbone.named_parameters() if p.requires_grad], grads):
                if grad is not None:
                    fisher[k] += grad.detach() ** 2
            
            count += 1
            if count >= num_samples:
                break
        if count >= num_samples:
            break
            
    for k in fisher:
        fisher[k] /= count
    return fisher

# Dynamic weight reconstruction function
def reconstruct_merged_backbone(base_backbone, expert_backbones, key_to_layer_idx, layer_coeffs, num_layers=5):
    # Apply softmax over experts for each layer
    weights = torch.softmax(layer_coeffs, dim=1)  # (num_layers, num_experts)
    
    merged_params = {}
    for key in base_backbone.state_dict().keys():
        layer_idx = key_to_layer_idx[key]
        layer_weights = weights[layer_idx]  # shape (num_experts,)
        
        # Linearly combine expert weights for this key
        merged_w = sum(layer_weights[t] * expert_backbones[t][key] for t in range(len(expert_backbones)))
        
        # If the key represents a batchnorm buffer (running_mean, running_var, num_batches_tracked),
        # we MUST detach it so that PyTorch doesn't try to track gradients on it!
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            merged_w = merged_w.detach()
            
        merged_params[key] = merged_w
        
    return merged_params

# Define corruptions for out-of-distribution evaluation
def apply_corruption(images, corruption_type="none"):
    if corruption_type == "none":
        return images
    elif corruption_type == "noise":
        # Add strong Gaussian noise
        noise = 0.4 * torch.randn_like(images)
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == "rotation":
        # Rotate by 45 degrees
        rotated = []
        for img in images:
            rotated.append(TF.rotate(img, 45))
        return torch.stack(rotated)
    else:
        return images

# Test-Time Adaptation loop
def run_tta(task_name, method="static", corruption="none", lr_coeffs=1e-1, lr_head=1e-3, steps=10, batch_size=64, num_batches=20, gamma=15.0, calib_samples=256):
    print(f"\nEvaluating {task_name.upper()} with method: {method.upper()} under corruption: {corruption.upper()}")
    
    # Load unmerged experts
    tasks = ["mnist", "fashionmnist", "kmnist"]
    expert_backbones = []
    expert_heads = {}
    
    base_backbone = CNNBackbone()
    key_to_layer_idx = get_key_to_layer_idx(base_backbone.state_dict())
    
    for t in tasks:
        backbone_dict = torch.load(f"checkpoints/{t}_backbone.pt", map_location=device)
        expert_backbones.append(backbone_dict)
        
        head = TaskHead()
        head.load_state_dict(torch.load(f"checkpoints/{t}_head.pt", map_location=device))
        head.to(device)
        expert_heads[t] = head
        
    # Unmerged expert for self-labeling
    self_labeling_expert = TaskHead()
    self_labeling_expert.load_state_dict(torch.load(f"checkpoints/{task_name}_head.pt", map_location=device))
    self_labeling_expert.to(device)
    
    self_labeling_backbone = CNNBackbone()
    self_labeling_backbone.load_state_dict(torch.load(f"checkpoints/{task_name}_backbone.pt", map_location=device))
    self_labeling_backbone.to(device)
    self_labeling_backbone.eval()
    self_labeling_expert.eval()
    
    # Instantiate the task-specific head to be optimized
    adapted_head = copy.deepcopy(expert_heads[task_name])
    adapted_head.to(device)
    
    # Initialize merging coefficients: shape (num_layers, num_experts)
    # Zero initialization means after softmax, each expert has equal weight (1/3)
    num_layers = 5
    num_experts = len(tasks)
    layer_coeffs = nn.Parameter(torch.zeros(num_layers, num_experts, device=device))
    
    # Prepare optimization
    if method == "static":
        optimizer = None
    elif method == "adamerging":
        # AdaMerging only optimizes layer coefficients, keeping heads frozen
        optimizer = optim.Adam([
            {"params": [layer_coeffs], "lr": lr_coeffs}
        ])
    elif method == "symerge":
        # Uniform learning rate for all layer coefficients
        optimizer = optim.Adam([
            {"params": [layer_coeffs], "lr": lr_coeffs},
            {"params": adapted_head.parameters(), "lr": lr_head}
        ])
    elif method == "head-tta":
        # Only optimize classification heads, keeping merging coefficients frozen (equal 1/3 split)
        optimizer = optim.Adam([
            {"params": adapted_head.parameters(), "lr": lr_head}
        ])
    elif method == "ca-symerge":
        # Curvature-aware SyMerge
        # Load calibration data for Fisher computation
        dataset_cls = dataset_classes[task_name]
        calib_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
        calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)
        
        # Compute Fisher Information for the target expert backbone
        print("Computing Fisher Information for expert backbone...")
        fisher_dict = compute_empirical_fisher(self_labeling_backbone, self_labeling_expert, calib_loader, num_samples=calib_samples)
        
        # Summarize Fisher per layer
        layer_fisher = torch.zeros(num_layers, device=device)
        layer_counts = torch.zeros(num_layers, device=device)
        
        for key, f_val in fisher_dict.items():
            layer_idx = key_to_layer_idx[key]
            layer_fisher[layer_idx] += f_val.mean()
            layer_counts[layer_idx] += 1
            
        # Average layer-wise sensitivity
        layer_sensitivity = layer_fisher / torch.clamp(layer_counts, min=1)
        print(f"Layer sensitivities (mean Fisher): {layer_sensitivity.cpu().numpy()}")
        
        # Scale coefficient learning rates inversely by sensitivity: lr_l = lr_0 * exp(-gamma * sensitivity)
        # We set gamma parameter to provide solid scaling
        lr_scalings = torch.exp(-gamma * layer_sensitivity).detach()
        print(f"Layer coefficient learning rate scales: {lr_scalings.cpu().numpy()}")
        
        # To implement layer-wise learning rates in PyTorch Adam, we can create separate parameter groups!
        # This is incredibly clean and robust.
        param_groups = []
        # Add head parameters with standard head learning rate
        param_groups.append({"params": adapted_head.parameters(), "lr": lr_head})
        
        # Add each layer coefficient as a separate Parameter with its scaled learning rate
        # To do this safely, we can create a list of single-row Parameters or slice gradients in a custom optimizer step.
        # But slicing layer_coeffs into individual Parameter rows is incredibly elegant and works natively with PyTorch!
        layer_coeffs_list = [nn.Parameter(layer_coeffs[l].clone()) for l in range(num_layers)]
        for l in range(num_layers):
            scaled_lr = lr_coeffs * lr_scalings[l].item()
            param_groups.append({"params": [layer_coeffs_list[l]], "lr": scaled_lr})
            
        optimizer = optim.Adam(param_groups)
    else:
        raise ValueError("Unknown method")
        
    # Load test dataset
    dataset_cls = dataset_classes[task_name]
    test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # We evaluate performance batch-by-batch along the test stream
    accuracies_before = []
    accuracies_after = []
    
    criterion = nn.CrossEntropyLoss()
    
    # Iterate over test stream batches
    batch_idx = 0
    for images, labels in test_loader:
        if batch_idx >= num_batches:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        # Apply corruption to represent OOD stream
        corrupted_images = apply_corruption(images, corruption)
        
        # 1. Evaluate BEFORE adaptation on this batch
        base_backbone.eval()
        adapted_head.eval()
        with torch.no_grad():
            if method == "ca-symerge":
                # Reconstruct layer_coeffs from the list
                current_coeffs = torch.stack(layer_coeffs_list)
            else:
                current_coeffs = layer_coeffs
                
            merged_params = reconstruct_merged_backbone(base_backbone, expert_backbones, key_to_layer_idx, current_coeffs)
            features = functional_call(base_backbone, merged_params, corrupted_images)
            outputs = adapted_head(features)
            _, predicted = outputs.max(1)
            acc_before = 100. * predicted.eq(labels).sum().item() / labels.size(0)
            accuracies_before.append(acc_before)
            
        # 2. Perform test-time adaptation steps on this batch
        if method != "static":
            # Generate pseudo-labels using the stable, unmerged expert (clean images for best labels)
            with torch.no_grad():
                expert_features = self_labeling_backbone(images)
                expert_logits = self_labeling_expert(expert_features)
                pseudo_labels = expert_logits.argmax(dim=-1)
                
            # Direct adaptation steps
            for step in range(steps):
                base_backbone.train()
                adapted_head.train()
                
                if method == "ca-symerge":
                    current_coeffs = torch.stack(layer_coeffs_list)
                else:
                    current_coeffs = layer_coeffs
                    
                merged_params = reconstruct_merged_backbone(base_backbone, expert_backbones, key_to_layer_idx, current_coeffs)
                
                # Forward pass on corrupted images
                features = functional_call(base_backbone, merged_params, corrupted_images)
                outputs = adapted_head(features)
                
                if method == "adamerging":
                    # Entropy minimization loss for AdaMerging
                    probs = torch.softmax(outputs, dim=-1)
                    loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                else:
                    # Cross-entropy loss with expert pseudo-labels (SyMerge, CA-SyMerge)
                    loss = criterion(outputs, pseudo_labels)
                
                optimizer.zero_grad()
                loss.backward()
                
                # If ca-symerge, we need to map gradients from current_coeffs back to the list of Parameters
                if method == "ca-symerge":
                    for l in range(num_layers):
                        if layer_coeffs_list[l].grad is None and current_coeffs.grad is not None:
                            layer_coeffs_list[l].grad = current_coeffs.grad[l].clone()
                            
                optimizer.step()
                
        # 3. Evaluate AFTER adaptation on this batch
        base_backbone.eval()
        adapted_head.eval()
        with torch.no_grad():
            if method == "ca-symerge":
                current_coeffs = torch.stack(layer_coeffs_list)
            else:
                current_coeffs = layer_coeffs
                
            merged_params = reconstruct_merged_backbone(base_backbone, expert_backbones, key_to_layer_idx, current_coeffs)
            features = functional_call(base_backbone, merged_params, corrupted_images)
            outputs = adapted_head(features)
            _, predicted = outputs.max(1)
            acc_after = 100. * predicted.eq(labels).sum().item() / labels.size(0)
            accuracies_after.append(acc_after)
            
        print(f"Batch {batch_idx+1}/{num_batches} - Acc Before: {acc_before:.2f}%, Acc After: {acc_after:.2f}%")
        batch_idx += 1
        
    avg_before = sum(accuracies_before) / len(accuracies_before)
    avg_after = sum(accuracies_after) / len(accuracies_after)
    improvement = avg_after - avg_before
    print(f"Result - Avg Acc Before: {avg_before:.2f}%, Avg Acc After: {avg_after:.2f}% (Diff: {improvement:+.2f}%)")
    
    return avg_before, avg_after, improvement

if __name__ == "__main__":
    # Run a full set of evaluations
    tasks = ["mnist", "fashionmnist", "kmnist"]
    corruptions = ["none", "noise", "rotation"]
    methods = ["static", "adamerging", "symerge", "head-tta", "ca-symerge"]
    
    results = {}
    
    # To keep execution extremely fast but highly rigorous, let's run all tasks, methods, and corruptions!
    for task in tasks:
        for corr in corruptions:
            for method in methods:
                # Static doesn't change before/after, so we run it once and record it
                if method == "static" and corr == "none":
                    # Run static on none
                    avg_b, avg_a, diff = run_tta(task, method="static", corruption=corr, num_batches=15)
                    results[(task, corr, method)] = (avg_b, avg_a, diff)
                elif method == "static":
                    # Static with corruption
                    avg_b, avg_a, diff = run_tta(task, method="static", corruption=corr, num_batches=15)
                    results[(task, corr, method)] = (avg_b, avg_a, diff)
                else:
                    # Adaptable methods
                    avg_b, avg_a, diff = run_tta(task, method=method, corruption=corr, num_batches=15)
                    results[(task, corr, method)] = (avg_b, avg_a, diff)
                    
    # Print a beautiful final table
    print("\n==================== FINAL RESULTS TABLE ====================")
    print(f"{'Task':<15} | {'Corruption':<10} | {'Method':<12} | {'Acc Before':<10} | {'Acc After':<10} | {'Diff':<8}")
    print("-" * 75)
    for key, (b, a, d) in sorted(results.items()):
        task, corr, method = key
        print(f"{task.upper():<15} | {corr:<10} | {method:<12} | {b:.2f}%     | {a:.2f}%     | {d:+.2f}%")
    print("=============================================================")
    
    # Save results to file
    with open("results/summary.txt", "w") as f:
        f.write("Task,Corruption,Method,Acc_Before,Acc_After,Diff\n")
        for key, (b, a, d) in sorted(results.items()):
            task, corr, method = key
            f.write(f"{task},{corr},{method},{b:.2f},{a:.2f},{d:+.2f}\n")
        print("Saved summary table to results/summary.txt")
