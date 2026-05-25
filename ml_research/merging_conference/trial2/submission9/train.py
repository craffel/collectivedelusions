import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np

# Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_base_model():
    # Use weights instead of pretrained parameter to avoid deprecation warnings
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace classification head to output 10 logits
    model.fc = nn.Linear(512, 10)
    # Reproducible initialization of the new head
    torch.manual_seed(42)
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity='linear')
    nn.init.zeros_(model.fc.bias)
    return model

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Task A classes: 0-4
    train_idx_a = [i for i, (_, label) in enumerate(train_set) if label < 5]
    test_idx_a = [i for i, (_, label) in enumerate(test_set) if label < 5]
    
    # Task B classes: 5-9
    train_idx_b = [i for i, (_, label) in enumerate(train_set) if label >= 5]
    test_idx_b = [i for i, (_, label) in enumerate(test_set) if label >= 5]
    
    train_loader_a = DataLoader(Subset(train_set, train_idx_a), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader_a = DataLoader(Subset(test_set, test_idx_a), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    train_loader_b = DataLoader(Subset(train_set, train_idx_b), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader_b = DataLoader(Subset(test_set, test_idx_b), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    test_loader_full = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return {
        'train_a': train_loader_a, 'test_a': test_loader_a,
        'train_b': train_loader_b, 'test_b': test_loader_b,
        'test_full': test_loader_full
    }

def train_epoch(model, base_model, dataloader, optimizer, criterion, device, opt_mode, rho, task_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Adjust targets for Task B (map 5-9 to 0-4) for CrossEntropyLoss
        if task_name == 'B':
            targets_loss = targets - 5
        else:
            targets_loss = targets
            
        if opt_mode == 'adamw':
            optimizer.zero_grad()
            outputs = model(inputs)
            # Mask logits of irrelevant classes during loss computation
            if task_name == 'A':
                loss = criterion(outputs[:, :5], targets_loss)
                preds = outputs[:, :5].argmax(dim=-1)
            else:
                loss = criterion(outputs[:, 5:], targets_loss)
                preds = outputs[:, 5:].argmax(dim=-1) + 5
                
            loss.backward()
            optimizer.step()
            
        elif opt_mode == 'sam':
            # 1. First pass (Loss and Gradient)
            optimizer.zero_grad()
            outputs = model(inputs)
            if task_name == 'A':
                loss = criterion(outputs[:, :5], targets_loss)
            else:
                loss = criterion(outputs[:, 5:], targets_loss)
            loss.backward()
            
            # Save original parameters and apply SAM perturbation tensor-wise
            original_params = {}
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        original_params[name] = p.data.clone()
                        # Tensor-wise perturbation
                        g_norm = p.grad.norm(2)
                        eps = p.grad * (rho / (g_norm + 1e-12))
                        p.add_(eps)
            
            # 2. Second pass (Perturbed Gradient)
            optimizer.zero_grad()
            outputs_pert = model(inputs)
            if task_name == 'A':
                loss_pert = criterion(outputs_pert[:, :5], targets_loss)
                preds = outputs_pert[:, :5].argmax(dim=-1)
            else:
                loss_pert = criterion(outputs_pert[:, 5:], targets_loss)
                preds = outputs_pert[:, 5:].argmax(dim=-1) + 5
            loss_pert.backward()
            
            # Restore original weights and apply optimizer step using perturbed gradient
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in original_params:
                        p.data.copy_(original_params[name])
            optimizer.step()
            loss = loss_pert
            
        elif opt_mode == 'taa_sr':
            # Task-Arithmetic-Aware Sharpness Regularization (TAA-SR)
            # 1. First pass (Loss and Gradient)
            optimizer.zero_grad()
            outputs = model(inputs)
            if task_name == 'A':
                loss = criterion(outputs[:, :5], targets_loss)
            else:
                loss = criterion(outputs[:, 5:], targets_loss)
            loss.backward()
            
            original_params = {}
            with torch.no_grad():
                for (name, p), (base_name, p_base) in zip(model.named_parameters(), base_model.named_parameters()):
                    if p.grad is not None:
                        original_params[name] = p.data.clone()
                        # Compute local task vector
                        tau = p.data - p_base.data
                        tau_norm = tau.norm(2)
                        
                        if tau_norm < 1e-6:
                            # Fallback to standard tensor-wise SAM
                            g_norm = p.grad.norm(2)
                            eps = p.grad * (rho / (g_norm + 1e-12))
                        else:
                            # TAA-SR perturbation: perturb along task vector direction
                            proj = torch.sum(p.grad * tau)
                            s = torch.sign(proj)
                            eps = s * rho * (tau / (tau_norm + 1e-12))
                        p.add_(eps)
            
            # 2. Second pass (Perturbed Gradient)
            optimizer.zero_grad()
            outputs_pert = model(inputs)
            if task_name == 'A':
                loss_pert = criterion(outputs_pert[:, :5], targets_loss)
                preds = outputs_pert[:, :5].argmax(dim=-1)
            else:
                loss_pert = criterion(outputs_pert[:, 5:], targets_loss)
                preds = outputs_pert[:, 5:].argmax(dim=-1) + 5
            loss_pert.backward()
            
            # Restore original weights and step
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in original_params:
                        p.data.copy_(original_params[name])
            optimizer.step()
            loss = loss_pert
            
        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device, task_name='full'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if task_name == 'A':
                # Evaluate on classes 0-4
                preds = outputs[:, :5].argmax(dim=-1)
                correct += preds.eq(targets).sum().item()
            elif task_name == 'B':
                # Evaluate on classes 5-9
                preds = outputs[:, 5:].argmax(dim=-1) + 5
                correct += preds.eq(targets).sum().item()
            else:
                # Full 10-class evaluation
                preds = outputs.argmax(dim=-1)
                correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
    return 100.0 * correct / total

def merge_models(model_a, model_b, base_model, merging_mode, lambda_val=0.5):
    # Initialize merged model as a copy of the base model
    merged_model = get_base_model()
    
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_0 = base_model.state_dict()
    state_merged = merged_model.state_dict()
    
    for name in state_0.keys():
        # Check if the parameter is a weight tensor and fits manifold merging
        is_conv = 'conv' in name and len(state_0[name].shape) == 4
        is_fc = 'fc' in name
        
        if merging_mode == 'TA':
            # Standard Task Arithmetic
            # W_merged = W_0 + lambda_val * (tau_a + tau_b)
            tau_a = state_a[name] - state_0[name]
            tau_b = state_b[name] - state_0[name]
            state_merged[name] = state_0[name] + lambda_val * (tau_a + tau_b)
            
        elif merging_mode == 'C-Ortho':
            # Apply OrthoMerge only to convolutional weights, linear and downsample linearly
            is_downsample = 'downsample' in name
            
            if is_conv and not is_downsample:
                # Reshape 4D tensor [C_out, C_in, K, K] to 2D [C_out, C_in * K * K]
                w_a_orig = state_a[name]
                w_b_orig = state_b[name]
                w_0_orig = state_0[name]
                
                c_out, c_in, k1, k2 = w_0_orig.shape
                w_a = w_a_orig.view(c_out, -1)
                w_b = w_b_orig.view(c_out, -1)
                w_0 = w_0_orig.view(c_out, -1)
                
                # Compute projection matrices
                Ma = torch.matmul(w_a, w_0.t())
                Mb = torch.matmul(w_b, w_0.t())
                
                # SVD
                Ua, Sa, Vha = torch.linalg.svd(Ma)
                Ub, Sb, Vhb = torch.linalg.svd(Mb)
                
                # Orthogonal rotations
                Ra = torch.matmul(Ua, Vha)
                Rb = torch.matmul(Ub, Vhb)
                
                # Map to Lie Algebra via inverse Cayley transform
                # Q = (R - I)(R + I)^-1
                eye = torch.eye(c_out, device=w_0.device)
                Qa = torch.linalg.solve(Ra + eye + 1e-6*eye, Ra - eye)
                Qb = torch.linalg.solve(Rb + eye + 1e-6*eye, Rb - eye)
                
                # Average of Lie algebra matrices with magnitude correction
                sum_Q = Qa + Qb
                sum_Q_norm = torch.linalg.norm(sum_Q, 'fro')
                Qa_norm = torch.linalg.norm(Qa, 'fro')
                Qb_norm = torch.linalg.norm(Qb, 'fro')
                
                if sum_Q_norm > 1e-12:
                    Q_merged = 0.5 * ((Qa_norm + Qb_norm) / sum_Q_norm) * sum_Q
                else:
                    Q_merged = torch.zeros_like(Qa)
                    
                # Map back to Orthogonal Group via Cayley transform
                # R_merged = (I + Q)(I - Q)^-1
                R_merged = torch.linalg.solve(eye - Q_merged + 1e-6*eye, eye + Q_merged)
                
                # Compute residuals
                rho_a = w_a - torch.matmul(Ra, w_0)
                rho_b = w_b - torch.matmul(Rb, w_0)
                rho_merged = 0.5 * (rho_a + rho_b)
                
                # Reconstruct and reshape back
                w_merged = torch.matmul(R_merged, w_0) + rho_merged
                state_merged[name] = w_merged.view(c_out, c_in, k1, k2)
            else:
                # Linear average for other parameters (e.g. classification head, batchnorm, downsample)
                tau_a = state_a[name] - state_0[name]
                tau_b = state_b[name] - state_0[name]
                state_merged[name] = state_0[name] + lambda_val * (tau_a + tau_b)
                
        elif merging_mode == 'OrthoMerge':
            # Apply OrthoMerge globally to all Conv2d and Linear layers, except 1D vectors (biases, batchnorm buffers)
            is_2d = len(state_0[name].shape) >= 2
            
            if is_2d:
                w_a_orig = state_a[name]
                w_b_orig = state_b[name]
                w_0_orig = state_0[name]
                
                orig_shape = w_0_orig.shape
                # Flatten everything to 2D
                dim0 = orig_shape[0]
                w_a = w_a_orig.view(dim0, -1)
                w_b = w_b_orig.view(dim0, -1)
                w_0 = w_0_orig.view(dim0, -1)
                
                # Compute SVD
                Ma = torch.matmul(w_a, w_0.t())
                Mb = torch.matmul(w_b, w_0.t())
                
                Ua, Sa, Vha = torch.linalg.svd(Ma)
                Ub, Sb, Vhb = torch.linalg.svd(Mb)
                
                Ra = torch.matmul(Ua, Vha)
                Rb = torch.matmul(Ub, Vhb)
                
                eye = torch.eye(dim0, device=w_0.device)
                Qa = torch.linalg.solve(Ra + eye + 1e-6*eye, Ra - eye)
                Qb = torch.linalg.solve(Rb + eye + 1e-6*eye, Rb - eye)
                
                sum_Q = Qa + Qb
                sum_Q_norm = torch.linalg.norm(sum_Q, 'fro')
                Qa_norm = torch.linalg.norm(Qa, 'fro')
                Qb_norm = torch.linalg.norm(Qb, 'fro')
                
                if sum_Q_norm > 1e-12:
                    Q_merged = 0.5 * ((Qa_norm + Qb_norm) / sum_Q_norm) * sum_Q
                else:
                    Q_merged = torch.zeros_like(Qa)
                    
                R_merged = torch.linalg.solve(eye - Q_merged + 1e-6*eye, eye + Q_merged)
                
                rho_a = w_a - torch.matmul(Ra, w_0)
                rho_b = w_b - torch.matmul(Rb, w_0)
                rho_merged = 0.5 * (rho_a + rho_b)
                
                w_merged = torch.matmul(R_merged, w_0) + rho_merged
                state_merged[name] = w_merged.view(orig_shape)
            else:
                # 1D vector linear average
                tau_a = state_a[name] - state_0[name]
                tau_b = state_b[name] - state_0[name]
                state_merged[name] = state_0[name] + lambda_val * (tau_a + tau_b)
                
    merged_model.load_state_dict(state_merged)
    return merged_model

def main():
    parser = argparse.ArgumentParser(description="Split CIFAR-10 Model Merging Experiment")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'merge', 'evaluate'], help="Execution mode")
    parser.add_argument('--task', type=str, default='A', choices=['A', 'B'], help="Task to train on")
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'sam', 'taa_sr'], help="Optimizer")
    parser.add_argument('--rho', type=float, default=0.05, help="SAM/TAA-SR perturbation scale")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--checkpoint_a', type=str, default='expert_a.pt', help="Path to Expert A checkpoint")
    parser.add_argument('--checkpoint_b', type=str, default='expert_b.pt', help="Path to Expert B checkpoint")
    parser.add_argument('--checkpoint_merged', type=str, default='merged_model.pt', help="Path to saved/loaded merged model")
    parser.add_argument('--merging_mode', type=str, default='TA', choices=['TA', 'C-Ortho', 'OrthoMerge'], help="Merging strategy")
    parser.add_argument('--lambda_val', type=float, default=0.5, help="Task vector merging coefficient")
    
    args = parser.parse_args()
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print(f"--- Training Expert {args.task} with {args.opt} ---")
        dataloaders = get_dataloaders(args.batch_size)
        train_loader = dataloaders['train_a'] if args.task == 'A' else dataloaders['train_b']
        test_loader = dataloaders['test_a'] if args.task == 'A' else dataloaders['test_b']
        
        base_model = get_base_model().to(device)
        model = get_base_model().to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        for epoch in range(1, args.epochs + 1):
            loss, acc = train_epoch(model, base_model, train_loader, optimizer, criterion, device, args.opt, args.rho, args.task)
            scheduler.step()
            test_acc = evaluate(model, test_loader, device, args.task)
            print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Test Acc: {test_acc:.2f}%")
            
        # Save model
        checkpoint_path = args.checkpoint_a if args.task == 'A' else args.checkpoint_b
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved Expert {args.task} to {checkpoint_path}")
        
    elif args.mode == 'merge':
        print(f"--- Merging Expert A and Expert B using {args.merging_mode} (lambda={args.lambda_val}) ---")
        base_model = get_base_model().to(device)
        
        model_a = get_base_model().to(device)
        model_a.load_state_dict(torch.load(args.checkpoint_a, map_location=device))
        
        model_b = get_base_model().to(device)
        model_b.load_state_dict(torch.load(args.checkpoint_b, map_location=device))
        
        merged_model = merge_models(model_a, model_b, base_model, args.merging_mode, args.lambda_val)
        torch.save(merged_model.state_dict(), args.checkpoint_merged)
        print(f"Saved merged model to {args.checkpoint_merged}")
        
        # Evaluate merged model
        dataloaders = get_dataloaders(args.batch_size)
        acc_a = evaluate(merged_model, dataloaders['test_a'], device, 'A')
        acc_b = evaluate(merged_model, dataloaders['test_b'], device, 'B')
        acc_full = evaluate(merged_model, dataloaders['test_full'], device, 'full')
        
        print(f"\n--- Merged Model Performance ---")
        print(f"Task A Accuracy: {acc_a:.2f}%")
        print(f"Task B Accuracy: {acc_b:.2f}%")
        print(f"Full CIFAR-10 Accuracy: {acc_full:.2f}%")
        
    elif args.mode == 'evaluate':
        print(f"--- Evaluating model from {args.checkpoint_merged} ---")
        model = get_base_model().to(device)
        model.load_state_dict(torch.load(args.checkpoint_merged, map_location=device))
        
        dataloaders = get_dataloaders(args.batch_size)
        acc_a = evaluate(model, dataloaders['test_a'], device, 'A')
        acc_b = evaluate(model, dataloaders['test_b'], device, 'B')
        acc_full = evaluate(model, dataloaders['test_full'], device, 'full')
        
        print(f"Task A Accuracy: {acc_a:.2f}%")
        print(f"Task B Accuracy: {acc_b:.2f}%")
        print(f"Full CIFAR-10 Accuracy: {acc_full:.2f}%")

if __name__ == '__main__':
    main()
