import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class SplitCIFAR10(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = set(classes)
        # Fast indexing
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in self.classes]
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
        
    def __len__(self):
        return len(self.indices)

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    # Task Splits
    classes_A = [0, 1, 2, 3, 4]
    classes_B = [5, 6, 7, 8, 9]
    
    train_A = SplitCIFAR10(train_dataset, classes_A)
    train_B = SplitCIFAR10(train_dataset, classes_B)
    
    test_A = SplitCIFAR10(test_dataset, classes_A)
    test_B = SplitCIFAR10(test_dataset, classes_B)
    
    loader_train_A = DataLoader(train_A, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    loader_train_B = DataLoader(train_B, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    loader_test_A = DataLoader(test_A, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_test_B = DataLoader(test_B, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_test_full = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return loader_train_A, loader_train_B, loader_test_A, loader_test_B, loader_test_full

def get_resnet18_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    return model.to(device)

def compute_fisher(model, dataloader, device, num_samples=1024):
    model.eval()
    fisher = {}
    # Enable requires_grad temporarily for weights we care about
    saved_requires_grad = {}
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 4: # conv layers
            fisher[name] = torch.zeros_like(param)
            saved_requires_grad[name] = param.requires_grad
            param.requires_grad = True
            
    count = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # backward pass
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += (param.grad ** 2) * batch_size
                    
        count += batch_size
        if count >= num_samples:
            break
            
    # Restore requires_grad
    for name, param in model.named_parameters():
        if name in saved_requires_grad:
            param.requires_grad = saved_requires_grad[name]
            
    for name in fisher:
        fisher[name] = fisher[name] / count
        # Normalize to prevent numerical underflow
        max_val = fisher[name].max() + 1e-12
        fisher[name] = fisher[name] / max_val
        
    return fisher

def compute_spor_loss(model, base_model, beta, gamma=0.0, fisher=None):
    loss = 0.0
    count = 0
    for (name, param), (name0, param0) in zip(model.named_parameters(), base_model.named_parameters()):
        if 'weight' in name and len(param.shape) == 4: # conv weight
            C_out = param.shape[0]
            W = param.view(C_out, -1)
            W0 = param0.view(C_out, -1)
            
            # Row normalization
            W_norm = torch.norm(W, p=2, dim=1, keepdim=True) + 1e-8
            W0_norm = torch.norm(W0, p=2, dim=1, keepdim=True) + 1e-8
            W_tilde = W / W_norm
            W0_tilde = W0 / W0_norm
            
            # M = W_tilde * W0_tilde^T
            M = torch.mm(W_tilde, W0_tilde.t())
            diff = torch.mm(M, M.t()) - torch.eye(C_out, device=W.device)
            
            if gamma > 0.0 and fisher is not None and name in fisher:
                # Fisher-Weighted SPOR (FW-SPOR)
                F = fisher[name].view(C_out, -1)
                F_row = torch.mean(F, dim=1) # shape (C_out,)
                F_row_mean = torch.mean(F_row) + 1e-8
                
                # Weight v_j = exp(-gamma * F_row / F_row_mean)
                v = torch.exp(-gamma * F_row / F_row_mean)
                
                # Compute Fisher-weighted deviation: v_i * v_j * diff_{i,j}^2
                loss_layer = torch.mean((v.unsqueeze(1) * diff * v.unsqueeze(0)) ** 2)
            else:
                # Standard SPOR
                loss_layer = torch.mean(diff ** 2)
                
            loss += loss_layer
            count += 1
            
    return beta * loss / count if count > 0 else 0.0

def train_expert(task, regime, beta, gamma, epochs=5, seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Expert {task} | Regime: {regime} | Beta: {beta} | Gamma: {gamma} | Device: {device}')
    
    loader_train_A, loader_train_B, _, _, _ = get_dataloaders()
    train_loader = loader_train_A if task == 'A' else loader_train_B
    
    model = get_resnet18_model(device)
    base_model = get_resnet18_model(device) # Keep frozen copy of pre-trained model
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
        
    fisher = None
    if regime == 'fw_spor' and gamma > 0.0:
        print('Computing pre-trained Fisher Information on task split...')
        fisher = compute_fisher(base_model, train_loader, device, num_samples=1024)
        
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            if regime in ['sam', 'spor', 'fw_spor']:
                # Sharpness-Aware Minimization (SAM) 2-pass update
                rho = 0.05
                is_spor = regime in ['spor', 'fw_spor']
                
                # Pass 1: unperturbed gradient
                outputs = model(images)
                loss = criterion(outputs, labels)
                if is_spor:
                    loss_spor = compute_spor_loss(model, base_model, beta, gamma if regime == 'fw_spor' else 0.0, fisher)
                    loss_total = loss + loss_spor
                else:
                    loss_total = loss
                    
                optimizer.zero_grad()
                loss_total.backward()
                
                # Save gradients and compute norm
                gradients = []
                for p in model.parameters():
                    if p.grad is not None:
                        gradients.append(p.grad.clone())
                    else:
                        gradients.append(None)
                        
                grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients if g is not None))
                
                # Perturb parameters
                with torch.no_grad():
                    for p, g in zip(model.parameters(), gradients):
                        if g is not None:
                            eps = rho * g / (grad_norm + 1e-12)
                            p.add_(eps)
                            p.eps_term = eps
                            
                # Pass 2: perturbed gradient
                outputs_perturbed = model(images)
                loss_perturbed = criterion(outputs_perturbed, labels)
                if is_spor:
                    loss_spor_perturbed = compute_spor_loss(model, base_model, beta, gamma if regime == 'fw_spor' else 0.0, fisher)
                    loss_perturbed_total = loss_perturbed + loss_spor_perturbed
                else:
                    loss_perturbed_total = loss_perturbed
                    
                optimizer.zero_grad()
                loss_perturbed_total.backward()
                
                # Restore parameters
                with torch.no_grad():
                    for p in model.parameters():
                        if hasattr(p, 'eps_term'):
                            p.sub_(p.eps_term)
                            del p.eps_term
                            
                optimizer.step()
                loss_step = loss_total.item()
                outputs_for_acc = outputs
            else:
                # Standard SGD update
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_step = loss.item()
                outputs_for_acc = outputs
                
            running_loss += loss_step * images.size(0)
            _, predicted = outputs_for_acc.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
        
    # Save trained expert model weights
    filename = f'expert_{task}_{regime}_beta_{beta}_gamma_{gamma}.pt'
    torch.save(model.state_dict(), filename)
    print(f'Saved model to {filename}\n')

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def orthomerge_layers(W_A, W_B, W0):
    # OrthoMerge calculation
    C_out = W_A.shape[0]
    
    # Row normalization for SVD Procrustes mapping
    W0_norm = torch.norm(W0, p=2, dim=1, keepdim=True) + 1e-8
    W_A_norm = torch.norm(W_A, p=2, dim=1, keepdim=True) + 1e-8
    W_B_norm = torch.norm(W_B, p=2, dim=1, keepdim=True) + 1e-8
    
    W0_tilde = W0 / W0_norm
    W_A_tilde = W_A / W_A_norm
    W_B_tilde = W_B / W_B_norm
    
    # M = W * W0^T
    M_A = torch.mm(W_A_tilde, W0_tilde.t())
    M_B = torch.mm(W_B_tilde, W0_tilde.t())
    
    # SVD to find optimal rotation
    UA, SA, VA = torch.linalg.svd(M_A)
    UB, SB, VB = torch.linalg.svd(M_B)
    
    R_A = torch.mm(UA, VA)
    R_B = torch.mm(UB, VB)
    
    # Cayley Transform
    I = torch.eye(C_out, device=W_A.device)
    Q_A = torch.matmul(R_A - I, torch.linalg.inv(R_A + I + 1e-6 * I))
    Q_B = torch.matmul(R_B - I, torch.linalg.inv(R_B + I + 1e-6 * I))
    
    # Average in Lie Algebra
    Q_merged = 0.5 * (Q_A + Q_B)
    
    # Map back to Orthogonal Group
    R_merged = torch.matmul(I + Q_merged, torch.linalg.inv(I - Q_merged + 1e-6 * I))
    
    # Reconstruction
    res_A = W_A - torch.matmul(R_A, W0)
    res_B = W_B - torch.matmul(R_B, W0)
    
    W_merged = torch.matmul(R_merged, W0) + 0.5 * (res_A + res_B)
    
    # Calculate Residual Norms
    res_norm_A = torch.norm(W_A_tilde - torch.matmul(R_A, W0_tilde), p='fro')
    res_norm_B = torch.norm(W_B_tilde - torch.matmul(R_B, W0_tilde), p='fro')
    avg_res_norm = 0.5 * (res_norm_A + res_norm_B).item()
    
    return W_merged, avg_res_norm

def merge_and_eval(regime, beta, gamma):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, loader_test_A, loader_test_B, loader_test_full = get_dataloaders()
    
    # Load Expert A, Expert B and Pre-trained Base
    model_A = get_resnet18_model(device)
    model_B = get_resnet18_model(device)
    model0 = get_resnet18_model(device) # base
    
    file_A = f'expert_A_{regime}_beta_{beta}_gamma_{gamma}.pt'
    file_B = f'expert_B_{regime}_beta_{beta}_gamma_{gamma}.pt'
    
    if not os.path.exists(file_A) or not os.path.exists(file_B):
        print(f"Error: Expert models for {regime} with beta={beta}, gamma={gamma} do not exist.")
        return None
        
    model_A.load_state_dict(torch.load(file_A, map_location=device))
    model_B.load_state_dict(torch.load(file_B, map_location=device))
    
    print(f"\n=============================================")
    print(f"MERGING AND EVALUATING: {regime} | beta={beta} | gamma={gamma}")
    print(f"=============================================")
    
    results = {}
    
    for merge_method in ['Task Arithmetic', 'C-Ortho', 'OM-All']:
        merged_model = get_resnet18_model(device)
        
        # Load state dictionary
        state_A = model_A.state_dict()
        state_B = model_B.state_dict()
        state0 = model0.state_dict()
        state_merged = merged_model.state_dict()
        
        res_norms = []
        
        for key in state_merged.keys():
            # Check if this is a weight we want to apply OrthoMerge to
            is_conv = 'weight' in key and len(state_A[key].shape) == 4
            is_fc = 'weight' in key and len(state_A[key].shape) == 2
            
            apply_orthomerge = False
            if merge_method == 'C-Ortho' and is_conv:
                apply_orthomerge = True
            elif merge_method == 'OM-All' and (is_conv or is_fc):
                apply_orthomerge = True
                
            if apply_orthomerge:
                W_A = state_A[key]
                W_B = state_B[key]
                W0 = state0[key]
                
                # Reshape if conv/fc layers
                C_out = W_A.shape[0]
                W_A_flat = W_A.view(C_out, -1)
                W_B_flat = W_B.view(C_out, -1)
                W0_flat = W0.view(C_out, -1)
                
                W_merged_flat, res_norm = orthomerge_layers(W_A_flat, W_B_flat, W0_flat)
                state_merged[key] = W_merged_flat.view_as(W_A)
                res_norms.append(res_norm)
            else:
                # Task Arithmetic for non-OrthoMerge layers
                # W_merged = 0.5 * W_A + 0.5 * W_B
                state_merged[key] = 0.5 * state_A[key] + 0.5 * state_B[key]
                
        merged_model.load_state_dict(state_merged)
        
        acc_A = evaluate_model(merged_model, loader_test_A, device)
        acc_B = evaluate_model(merged_model, loader_test_B, device)
        acc_full = evaluate_model(merged_model, loader_test_full, device)
        avg_res_norm = np.mean(res_norms) if len(res_norms) > 0 else 0.0
        
        print(f"[{merge_method}] Acc A: {acc_A:.2f}% | Acc B: {acc_B:.2f}% | Full Acc: {acc_full:.2f}% | Avg Res Norm: {avg_res_norm:.6f}")
        results[merge_method] = {
            'Acc A': acc_A,
            'Acc B': acc_B,
            'Full Acc': acc_full,
            'Avg Res Norm': avg_res_norm
        }
        
    return results

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'])
    parser.add_argument('--task', type=str, choices=['A', 'B'])
    parser.add_argument('--regime', type=str, required=True, choices=['sgd', 'sam', 'spor', 'fw_spor'])
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_expert(args.task, args.regime, args.beta, args.gamma, args.epochs)
    elif args.mode == 'eval':
        results = merge_and_eval(args.regime, args.beta, args.gamma)
        if results:
            with open(f"results_{args.regime}_beta_{args.beta}_gamma_{args.gamma}.json", 'w') as f:
                json.dump(results, f, indent=4)
