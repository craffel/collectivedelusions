import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.func import functional_call
import numpy as np
import copy
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"G-SAM ablation running on device: {device}")

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Model Definitions (must match train_experts.py)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc(x))
        return x

class Head(nn.Module):
    def __init__(self, num_classes=10):
        super(Head, self).__init__()
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        return self.fc(x)

from train_experts import get_datasets

# Soft Cross-Entropy Loss for Knowledge Distillation / Self-Labeling
def soft_cross_entropy_loss(student_logits, teacher_logits, temperature=2.0):
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    return -torch.sum(p_teacher * log_p_student, dim=1).mean()

# Function to add image corruptions (simulates distribution shift)
def corrupt_images(images, corruption_type="none", batch_seed=None):
    if corruption_type == "none":
        return images
    
    # Save the current state of the random number generator
    if batch_seed is not None:
        state = np.random.get_state()
        np.random.seed(batch_seed)
        
    if corruption_type == "noise":
        # Add zero-mean Gaussian noise
        noise = torch.randn_like(images) * 0.4
        res = torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == "rotation":
        # Random rotation between 20 and 45 degrees
        angle = float(np.random.uniform(20, 45))
        theta = torch.tensor([
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0]
        ], dtype=torch.float, device=images.device).unsqueeze(0).repeat(images.size(0), 1, 1)
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        res = F.grid_sample(images, grid, align_corners=False)
    else:
        res = images
        
    # Restore random state
    if batch_seed is not None:
        np.random.set_state(state)
        
    return res

# Evaluation function for sequential
def evaluate_merged_model_seq(lambdas, heads, test_loaders, corruption="none"):
    merged_params = {}
    for key in base_state.keys():
        merged_params[key] = base_state[key] + \
                             lambdas[0] * (expert_states[0][key] - base_state[key]) + \
                             lambdas[1] * (expert_states[1][key] - base_state[key]) + \
                             lambdas[2] * (expert_states[2][key] - base_state[key])
                             
    correct_all = [0, 0, 0]
    total_all = [0, 0, 0]
    
    with torch.no_grad():
        for t_idx, loader in enumerate(test_loaders):
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                b_seed = 42 + t_idx * 10000 + batch_idx
                corrupted = corrupt_images(images, corruption, batch_seed=b_seed)
                
                features = functional_call(meta_encoder, merged_params, corrupted)
                outputs = heads[t_idx](features)
                
                _, predicted = outputs.max(1)
                total_all[t_idx] += labels.size(0)
                correct_all[t_idx] += predicted.eq(labels).sum().item()
                
    accuracies = [correct_all[i] / total_all[i] for i in range(3)]
    return accuracies

# Evaluation function for independent
def evaluate_merged_model_on_task_ind(lambdas, head, loader, t_idx, corruption="none"):
    merged_params = {}
    for key in base_state.keys():
        merged_params[key] = base_state[key] + \
                             lambdas[0] * (expert_states[0][key] - base_state[key]) + \
                             lambdas[1] * (expert_states[1][key] - base_state[key]) + \
                             lambdas[2] * (expert_states[2][key] - base_state[key])
                             
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            b_seed = 42 + t_idx * 10000 + batch_idx
            corrupted = corrupt_images(images, corruption, batch_seed=b_seed)
            
            features = functional_call(meta_encoder, merged_params, corrupted)
            outputs = head(features)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return correct / total

if __name__ == "__main__":
    # Get datasets
    (m_tr, m_te), (f_tr, f_te), (k_tr, k_te), is_synthetic = get_datasets()
    
    # Load Models
    base_state = torch.load("./models/base_encoder.pt", map_location=device)
    expert_names = ["MNIST", "FashionMNIST", "KMNIST"]
    expert_states = [torch.load(f"./models/expert_encoder_{name}.pt", map_location=device) for name in expert_names]
    expert_heads = [Head(num_classes=10).to(device) for _ in range(3)]
    for i, name in enumerate(expert_names):
        expert_heads[i].load_state_dict(torch.load(f"./models/expert_head_{name}.pt", map_location=device))
        expert_heads[i].eval()
        
    meta_encoder = Encoder().to(device)
    
    test_loaders = [
        DataLoader(m_te, batch_size=64, shuffle=False),
        DataLoader(f_te, batch_size=64, shuffle=False),
        DataLoader(k_te, batch_size=64, shuffle=False)
    ]
    
    corruptions = ["none", "noise", "rotation"]
    rho = 0.05
    learning_rate = 0.01
    
    seq_results = {}
    ind_results = {}
    
    print("\n--- Running Global-SAM (G-SAM) Sequential TTA ---")
    for corruption in corruptions:
        print(f"Evaluating Sequential G-SAM under: {corruption.upper()}")
        set_seed(42)
        gsam_lambdas = torch.tensor([0.3, 0.3, 0.3], requires_grad=True, device=device)
        gsam_heads = [copy.deepcopy(h) for h in expert_heads]
        for h in gsam_heads:
            h.train()
            
        params = [gsam_lambdas]
        for h in gsam_heads:
            params.extend(list(h.parameters()))
            
        optimizer = optim.Adam(params, lr=learning_rate)
        
        for t_idx, loader in enumerate(test_loaders):
            for batch_idx, (images, _) in enumerate(loader):
                images = images.to(device)
                b_seed = 42 + t_idx * 10000 + batch_idx
                corrupted = corrupt_images(images, corruption, batch_seed=b_seed)
                
                # --- Step 1: Compute loss and original gradients ---
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_features = functional_call(meta_encoder, expert_states[t_idx], corrupted)
                    teacher_logits = expert_heads[t_idx](teacher_features)
                    
                merged_params = {}
                for key in base_state.keys():
                    merged_params[key] = base_state[key] + \
                                         gsam_lambdas[0] * (expert_states[0][key] - base_state[key]) + \
                                         gsam_lambdas[1] * (expert_states[1][key] - base_state[key]) + \
                                         gsam_lambdas[2] * (expert_states[2][key] - base_state[key])
                                         
                student_features = functional_call(meta_encoder, merged_params, corrupted)
                student_logits = gsam_heads[t_idx](student_features)
                
                loss = soft_cross_entropy_loss(student_logits, teacher_logits)
                loss.backward()
                
                # --- Step 2: Apply Global Adversarial Perturbation (G-SAM) ---
                global_grad_norm = 0.0
                for p in params:
                    if p.grad is not None:
                        global_grad_norm += torch.norm(p.grad, p=2).item() ** 2
                global_grad_norm = global_grad_norm ** 0.5
                
                perturbations = {}
                has_perturbed = False
                if global_grad_norm > 1e-12:
                    for p in params:
                        if p.grad is not None:
                            eps = rho * (p.grad / global_grad_norm)
                            perturbations[p] = eps
                            p.data.add_(eps)
                            has_perturbed = True
                            
                if has_perturbed:
                    # --- Step 3: Compute loss at the perturbed point ---
                    optimizer.zero_grad()
                    merged_params_pert = {}
                    for key in base_state.keys():
                        merged_params_pert[key] = base_state[key] + \
                                                  gsam_lambdas[0] * (expert_states[0][key] - base_state[key]) + \
                                                  gsam_lambdas[1] * (expert_states[1][key] - base_state[key]) + \
                                                  gsam_lambdas[2] * (expert_states[2][key] - base_state[key])
                                                  
                    student_features_pert = functional_call(meta_encoder, merged_params_pert, corrupted)
                    student_logits_pert = gsam_heads[t_idx](student_features_pert)
                    
                    loss_pert = soft_cross_entropy_loss(student_logits_pert, teacher_logits)
                    loss_pert.backward()
                    
                    # --- Step 4: Restore original parameters and apply update ---
                    for p in params:
                        if p in perturbations:
                            p.data.sub_(perturbations[p])
                            
                    optimizer.step()
                else:
                    optimizer.step()
                    
                with torch.no_grad():
                    gsam_lambdas.clamp_(0.0, 1.0)
                    
        gsam_accs = evaluate_merged_model_seq(gsam_lambdas.detach(), gsam_heads, test_loaders, corruption)
        print(f"Global-SAM (Seq.) | Accuracies: {gsam_accs} | Mean: {np.mean(gsam_accs):.4f}")
        seq_results[corruption] = gsam_accs

    print("\n--- Running Global-SAM (G-SAM) Independent TTA ---")
    for corruption in corruptions:
        print(f"Evaluating Independent G-SAM under: {corruption.upper()}")
        gsam_accs = []
        for t_idx, loader in enumerate(test_loaders):
            set_seed(42)
            gsam_lambdas = torch.tensor([0.3, 0.3, 0.3], requires_grad=True, device=device)
            gsam_head = copy.deepcopy(expert_heads[t_idx])
            gsam_head.train()
            
            params = [gsam_lambdas] + list(gsam_head.parameters())
            optimizer = optim.Adam(params, lr=learning_rate)
            
            for batch_idx, (images, _) in enumerate(loader):
                images = images.to(device)
                b_seed = 42 + t_idx * 10000 + batch_idx
                corrupted = corrupt_images(images, corruption, batch_seed=b_seed)
                
                # --- Step 1: Compute loss and original gradients ---
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_features = functional_call(meta_encoder, expert_states[t_idx], corrupted)
                    teacher_logits = expert_heads[t_idx](teacher_features)
                    
                merged_params = {}
                for key in base_state.keys():
                    merged_params[key] = base_state[key] + \
                                         gsam_lambdas[0] * (expert_states[0][key] - base_state[key]) + \
                                         gsam_lambdas[1] * (expert_states[1][key] - base_state[key]) + \
                                         gsam_lambdas[2] * (expert_states[2][key] - base_state[key])
                                         
                student_features = functional_call(meta_encoder, merged_params, corrupted)
                student_logits = gsam_head(student_features)
                
                loss = soft_cross_entropy_loss(student_logits, teacher_logits)
                loss.backward()
                
                # --- Step 2: Apply Global Adversarial Perturbation (G-SAM) ---
                global_grad_norm = 0.0
                for p in params:
                    if p.grad is not None:
                        global_grad_norm += torch.norm(p.grad, p=2).item() ** 2
                global_grad_norm = global_grad_norm ** 0.5
                
                perturbations = {}
                has_perturbed = False
                if global_grad_norm > 1e-12:
                    for p in params:
                        if p.grad is not None:
                            eps = rho * (p.grad / global_grad_norm)
                            perturbations[p] = eps
                            p.data.add_(eps)
                            has_perturbed = True
                            
                if has_perturbed:
                    # --- Step 3: Compute loss at the perturbed point ---
                    optimizer.zero_grad()
                    merged_params_pert = {}
                    for key in base_state.keys():
                        merged_params_pert[key] = base_state[key] + \
                                                  gsam_lambdas[0] * (expert_states[0][key] - base_state[key]) + \
                                                  gsam_lambdas[1] * (expert_states[1][key] - base_state[key]) + \
                                                  gsam_lambdas[2] * (expert_states[2][key] - base_state[key])
                                                  
                    student_features_pert = functional_call(meta_encoder, merged_params_pert, corrupted)
                    student_logits_pert = gsam_head(student_features_pert)
                    
                    loss_pert = soft_cross_entropy_loss(student_logits_pert, teacher_logits)
                    loss_pert.backward()
                    
                    # --- Step 4: Restore original parameters and apply update ---
                    for p in params:
                        if p in perturbations:
                            p.data.sub_(perturbations[p])
                            
                    optimizer.step()
                else:
                    optimizer.step()
                    
                with torch.no_grad():
                    gsam_lambdas.clamp_(0.0, 1.0)
                    
            task_acc = evaluate_merged_model_on_task_ind(gsam_lambdas.detach(), gsam_head, loader, t_idx, corruption)
            gsam_accs.append(task_acc)
            
        print(f"Global-SAM (Ind.)  | Accuracies: {gsam_accs} | Mean: {np.mean(gsam_accs):.4f}")
        ind_results[corruption] = gsam_accs

    # Print a final summary of G-SAM vs SATT-Merge (from results in progress.md)
    print("\n--- G-SAM vs SATT-Merge Summary ---")
    for c in corruptions:
        print(f"Corruption: {c.upper()}")
        print(f"  Sequential G-SAM Mean:  {np.mean(seq_results[c]):.4f}")
        print(f"  Independent G-SAM Mean: {np.mean(ind_results[c]):.4f}")
