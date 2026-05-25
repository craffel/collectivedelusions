import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False  # Avoid cuDNN initialization issues on this cluster

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
RHO = 0.05
GAMMA = 1.0  # Fisher weighting strength

print(f"Using device: {DEVICE}")

# Define transforms
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

# Create task subsets
# Task A: classes 0-4
# Task B: classes 5-9
train_targets = np.array(train_dataset.targets)
test_targets = np.array(test_dataset.targets)

indices_train_A = np.where(train_targets < 5)[0]
indices_train_B = np.where(train_targets >= 5)[0]

indices_test_A = np.where(test_targets < 5)[0]
indices_test_B = np.where(test_targets >= 5)[0]

train_set_A = Subset(train_dataset, indices_train_A)
train_set_B = Subset(train_dataset, indices_train_B)

test_set_A = Subset(test_dataset, indices_test_A)
test_set_B = Subset(test_dataset, indices_test_B)

train_loader_A = DataLoader(train_set_A, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
train_loader_B = DataLoader(train_set_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

test_loader_A = DataLoader(test_set_A, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader_B = DataLoader(test_set_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

test_loader_full = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Helper function to compute Procrustes residual norm
def compute_procrustes_residual_norm(W, W0):
    # W, W0 are expected to be 2D matrices (C, D)
    if W.dim() > 2:
        W = W.view(W.size(0), -1)
    if W0.dim() > 2:
        W0 = W0.view(W0.size(0), -1)
        
    # Row normalization to remove scale disparities
    norm_W = torch.norm(W, dim=1, keepdim=True) + 1e-8
    norm_W0 = torch.norm(W0, dim=1, keepdim=True) + 1e-8
    W_tilde = W / norm_W
    W0_tilde = W0 / norm_W0
    
    # Compute cross-correlation matrix M
    M = torch.matmul(W_tilde, W0_tilde.t())
    
    try:
        # SVD
        U, S, Vh = torch.linalg.svd(M)
        # R = U V^T = U @ Vh (since Vh is V^T)
        R = torch.matmul(U, Vh)
        # Residual = W_tilde - R @ W0_tilde
        diff = W_tilde - torch.matmul(R, W0_tilde)
        res_norm = torch.norm(diff, p='fro').item()
        return res_norm
    except Exception as e:
        print(f"SVD failed: {e}")
        return 0.0

# Function to get average Procrustes residual norm for convolutional layers
def model_procrustes_norm(model, base_model):
    norms = []
    for (name1, p1), (name2, p2) in zip(model.named_parameters(), base_model.named_parameters()):
        if p1.requires_grad and "conv" in name1.lower() and p1.dim() >= 2:
            norm = compute_procrustes_residual_norm(p1.data, p2.data)
            norms.append(norm)
    return np.mean(norms) if len(norms) > 0 else 0.0

# Function to align head of base ResNet18 model
def align_base_head(model, dataset, epochs=1):
    print("Aligning classification head on CIFAR-10 training set with backbone frozen...")
    # Freeze backbone
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    # Optimizer for head only
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Full dataset loader (shuffled)
    full_train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in full_train_loader:
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
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Alignment Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Unfreeze backbone
    for param in model.parameters():
        param.requires_grad = True
        
    return model

# Function to compute diagonal Fisher Information of the base model
def compute_base_fisher(model, dataset, num_samples=2000):
    print(f"Computing diagonal Fisher Information of aligned base model on {num_samples} samples...")
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Subset train dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
    
    num_batches = len(loader)
    
    with torch.enable_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
                
    # Average and move to CPU/keep on device
    for name in fisher:
        fisher[name] /= num_batches
        # Smooth and stabilize
        fisher[name] = torch.clamp(fisher[name], min=1e-15)
        
    print("Fisher computation complete!")
    return fisher

# Evaluation function
# Evaluation function
def evaluate(model, loader, task_name='full'):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            if task_name == 'A':
                outputs = outputs[:, :5]
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            elif task_name == 'B':
                outputs = outputs[:, 5:]
                loss = criterion(outputs, targets - 5)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets - 5).sum().item()
            else:
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            
    acc = 100. * correct / total
    avg_loss = running_loss / total
    return acc, avg_loss

# Weight merging function
def merge_weights(model_A, model_B, lmbda=0.5):
    # Create a new merged model
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = merged_model.to(DEVICE)
    
    state_dict_A = model_A.state_dict()
    state_dict_B = model_B.state_dict()
    merged_state_dict = {}
    
    for key in state_dict_A.keys():
        merged_state_dict[key] = lmbda * state_dict_A[key] + (1.0 - lmbda) * state_dict_B[key]
        
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

# Training function with support for SGD, SAM, F-SAM
def fine_tune_expert(base_model, train_loader, task_name, mode='sgd', fisher=None):
    print(f"\n--- Fine-tuning Expert on Task {task_name} using {mode.upper()} ---")
    
    # Create model copy
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)
    model.load_state_dict(base_model.state_dict())
    
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    is_task_A = "A" in task_name
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if is_task_A:
                # Task A: classes 0-4
                targets_mapped = targets
                def get_outputs(x):
                    return model(x)[:, :5]
            else:
                # Task B: classes 5-9 -> map targets to 0-4
                targets_mapped = targets - 5
                def get_outputs(x):
                    return model(x)[:, 5:]
            
            if mode == 'sgd':
                optimizer.zero_grad()
                outputs = get_outputs(inputs)
                loss = criterion(outputs, targets_mapped)
                loss.backward()
                optimizer.step()
                
            elif mode in ['sam', 'fsam', 'fsam_dir']:
                # 1. First pass: compute gradients
                outputs = get_outputs(inputs)
                loss = criterion(outputs, targets_mapped)
                optimizer.zero_grad()
                loss.backward()
                
                # 2. Perturbation
                with torch.no_grad():
                    # Calculate global gradient norm for standard SAM
                    grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)) + 1e-12
                    
                    original_params = {}
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            original_params[name] = p.data.clone()
                            
                            if mode == 'sam':
                                eps = RHO * p.grad.data / grad_norm
                            elif mode == 'fsam':
                                # F-SAM (Inverse)
                                F_i = fisher[name]
                                mean_F = F_i.mean()
                                ratio = torch.clamp(F_i / (mean_F + 1e-8), max=10.0)
                                t_i = torch.exp(-GAMMA * ratio)
                                v = t_i * p.grad.data
                                v_norm = v.norm() + 1e-12
                                eps = RHO * v / v_norm
                            elif mode == 'fsam_dir':
                                # F-SAM (Direct)
                                F_i = fisher[name]
                                mean_F = F_i.mean()
                                ratio = torch.clamp(F_i / (mean_F + 1e-8), max=10.0)
                                t_i = torch.exp(GAMMA * ratio)
                                v = t_i * p.grad.data
                                v_norm = v.norm() + 1e-12
                                eps = RHO * v / v_norm
                                
                            p.data.add_(eps)
                
                # 3. Second pass: compute gradients at perturbed parameters
                model.zero_grad()
                outputs_perturbed = get_outputs(inputs)
                loss_perturbed = criterion(outputs_perturbed, targets_mapped)
                loss_perturbed.backward()
                
                # 4. Restore original parameters and step
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            p.data.copy_(original_params[name])
                            
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets_mapped).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
    return model

# Main pipeline
def main():
    global GAMMA, RHO
    import argparse
    parser = argparse.ArgumentParser(description="Run F-SAM model merging experiments.")
    parser.add_argument('--gamma', type=float, default=1.0, help="Fisher weighting strength")
    parser.add_argument('--rho', type=float, default=0.05, help="SAM perturbation radius")
    parser.add_argument('--modes', type=str, default='sgd,sam,fsam,fsam_dir', help="Comma-separated list of modes to run")
    parser.add_argument('--output', type=str, default='results.json', help="Path to save results JSON")
    args = parser.parse_args()
    
    GAMMA = args.gamma
    RHO = args.rho
    modes = [m.strip().lower() for m in args.modes.split(',')]
    output_path = args.output
    
    print(f"Configurations - GAMMA: {GAMMA}, RHO: {RHO}, MODES: {modes}, OUTPUT: {output_path}")

    # 1. Load pre-trained ResNet18 base model
    print("Loading pre-trained ResNet18 model...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model = base_model.to(DEVICE)
    
    # 2. Align head on CIFAR-10 full training set (backbone frozen)
    base_model = align_base_head(base_model, train_dataset, epochs=1)
    
    # Evaluate base model on full test set (sanity check)
    base_acc, _ = evaluate(base_model, test_loader_full)
    print(f"Aligned Base Model Full Test Acc: {base_acc:.2f}%")
    
    # 3. Compute base Fisher
    fisher = compute_base_fisher(base_model, train_dataset, num_samples=2000)
    
    # Save base model state dict as W0
    W0_state = {k: v.clone().cpu() for k, v in base_model.state_dict().items()}
    
    results = {}
    
    for mode in modes:
        print(f"\n========================================")
        print(f"Running Experiments for MODE: {mode.upper()}")
        print(f"========================================")
        
        # Load fresh base model to start from
        fresh_base = models.resnet18(weights=None)
        fresh_base.fc = nn.Linear(fresh_base.fc.in_features, 10)
        fresh_base = fresh_base.to(DEVICE)
        fresh_base.load_state_dict(base_model.state_dict())
        
        # Train Expert A
        expert_A = fine_tune_expert(fresh_base, train_loader_A, task_name="A (0-4)", mode=mode, fisher=fisher)
        # Train Expert B
        expert_B = fine_tune_expert(fresh_base, train_loader_B, task_name="B (5-9)", mode=mode, fisher=fisher)
        
        # Evaluate individual experts
        acc_A_taskA, _ = evaluate(expert_A, test_loader_A, task_name='A')
        acc_B_taskB, _ = evaluate(expert_B, test_loader_B, task_name='B')
        
        print(f"Expert A on Task A Test Acc: {acc_A_taskA:.2f}%")
        print(f"Expert B on Task B Test Acc: {acc_B_taskB:.2f}%")
        
        # Compute Procrustes residual norms relative to aligned base model (backbone only)
        norm_A = model_procrustes_norm(expert_A, fresh_base)
        norm_B = model_procrustes_norm(expert_B, fresh_base)
        avg_norm = (norm_A + norm_B) / 2.0
        print(f"Average Procrustes Residual Norm: {avg_norm:.5f}")
        
        # Merge experts
        merged_model = merge_weights(expert_A, expert_B, lmbda=0.5)
        
        # Evaluate merged model on full CIFAR-10 test set
        merged_acc, _ = evaluate(merged_model, test_loader_full)
        print(f"Merged Model Full Test Acc: {merged_acc:.2f}%")
        
        # Save results
        results[mode] = {
            "expert_A_acc": acc_A_taskA,
            "expert_B_acc": acc_B_taskB,
            "avg_expert_acc": (acc_A_taskA + acc_B_taskB) / 2.0,
            "avg_procrustes_norm": avg_norm,
            "merged_acc": merged_acc
        }
        
    # Print summary
    print("\n================ SUMMARY OF RESULTS ================")
    print(f"{'Mode':<10} | {'Expert A':<10} | {'Expert B':<10} | {'Avg Exp':<10} | {'Proc Norm':<10} | {'Merged Acc':<10}")
    print("-" * 75)
    for mode in modes:
        res = results[mode]
        print(f"{mode.upper():<10} | {res['expert_A_acc']:<10.2f} | {res['expert_B_acc']:<10.2f} | {res['avg_expert_acc']:<10.2f} | {res['avg_procrustes_norm']:<10.5f} | {res['merged_acc']:<10.2f}")
        
    # Write results to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults successfully saved to {output_path}!")

if __name__ == "__main__":
    main()
