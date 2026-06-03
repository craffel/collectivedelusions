import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
import json

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False

# Define a simple Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Helper function to get subset loaders for CIFAR-10 tasks
def get_task_loaders(dataset_name="CIFAR10", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download CIFAR10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Task splits:
    # Task A: Classes 0, 1, 2 (airplane, automobile, bird)
    # Task B: Classes 3, 4, 5 (cat, deer, dog)
    # Task C: Classes 6, 7, 8, 9 (frog, horse, ship, truck)
    task_splits = {
        'A': [0, 1, 2],
        'B': [3, 4, 5],
        'C': [6, 7, 8, 9]
    }
    
    loaders = {}
    for task_name, classes in task_splits.items():
        # Filter train indices
        train_indices = [i for i, label in enumerate(train_dataset.targets) if label in classes]
        # Filter test indices
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
        
        train_sub = Subset(train_dataset, train_indices)
        test_sub = Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False, num_workers=2)
        
        loaders[task_name] = {'train': train_loader, 'test': test_loader}
        
    return loaders

# Function to train a model on a specific task
def train_model(model, loaders, task_name, epochs=10, lr=0.001, device='cpu'):
    print(f"\n--- Training on Task {task_name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_loader = loaders[task_name]['train']
    for epoch in range(epochs):
        model.train()
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
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    return model

# Function to evaluate a model on a specific task or all tasks
def evaluate_model(model, loaders, task_name=None, device='cpu'):
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    tasks = [task_name] if task_name is not None else list(loaders.keys())
    
    with torch.no_grad():
        for t in tasks:
            test_loader = loaders[t]['test']
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            test_loss = running_loss / len(test_loader.dataset)
            test_acc = 100. * correct / total
            results[t] = {'loss': test_loss, 'acc': test_acc}
            
    return results

# Model Merging Methods

def merge_task_arithmetic(base_model, experts, lam=0.5):
    """
    Standard Task Arithmetic.
    W_merged = W_base + lam * sum(W_expert - W_base)
    """
    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    for key in base_state.keys():
        if base_state[key].dtype in [torch.float32, torch.float64]:
            task_vectors = [expert_state[key] - base_state[key] for expert_state in expert_states]
            sum_vector = torch.stack(task_vectors).sum(dim=0)
            merged_state[key] = base_state[key] + lam * sum_vector
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_relativistic(base_model, experts, lam=0.5, eta=1.5):
    """
    LorentzMerge: Relativistic Model Merging.
    Treats each task update as moving with a 'parameter velocity' v_k = ||tv||_F.
    The speed of light is c = eta * ||W_0||_F.
    Applies Lorentz contraction to contract high-velocity (overfitted/extreme) expert updates,
    preventing them from dominating and causing multi-task interference.
    """
    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    for key in base_state.keys():
        if base_state[key].dtype not in [torch.float32, torch.float64]:
            continue
            
        task_vectors = [expert_state[key] - base_state[key] for expert_state in expert_states]
        
        # Calculate base norm
        base_norm = torch.norm(base_state[key]).item()
        
        # Speed of light c
        c = eta * base_norm if base_norm > 0 else 1.0
        
        processed_task_vectors = []
        for tv in task_vectors:
            tv_norm = torch.norm(tv).item()
            # Relative velocity beta
            beta = min(tv_norm / c, 0.999) if c > 0 else 0.0
            
            # Lorentz contraction factor (1 / gamma)
            inv_gamma = (1.0 - beta**2)**0.5
            
            # Lorentz warped update
            tv_warped = tv * inv_gamma
            processed_task_vectors.append(tv_warped)
            
        sum_vector = torch.stack(processed_task_vectors).sum(dim=0)
        merged_state[key] = base_state[key] + lam * sum_vector
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_ties(base_model, experts, lam=0.5, fraction=0.2):
    """
    TIES-Merging Baseline.
    """
    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    for key in base_state.keys():
        if base_state[key].dtype not in [torch.float32, torch.float64]:
            continue
            
        task_vectors = [expert_state[key] - base_state[key] for expert_state in expert_states]
        
        # 1. Truncate task vectors (keep top fraction by magnitude)
        truncated_vectors = []
        for tv in task_vectors:
            flat_tv = tv.view(-1)
            k = int(fraction * flat_tv.numel())
            if k > 0:
                threshold = torch.topk(flat_tv.abs(), k).values[-1]
                mask = flat_tv.abs() >= threshold
                truncated_flat = flat_tv * mask
                truncated_vectors.append(truncated_flat.view_as(tv))
            else:
                truncated_vectors.append(tv)
                
        # 2. Sign consensus
        stacked_truncated = torch.stack(truncated_vectors)
        signs = torch.sign(stacked_truncated)
        sum_signs = signs.sum(dim=0)
        consensus_sign = torch.sign(sum_signs)
        
        # 3. Disagreement elimination
        matching_mask = (signs == consensus_sign.unsqueeze(0)) & (consensus_sign.unsqueeze(0) != 0)
        eliminated_stacked = stacked_truncated * matching_mask
        
        # 4. Average and rescale
        num_matching = matching_mask.sum(dim=0).float()
        sum_vectors = eliminated_stacked.sum(dim=0)
        
        average_vector = torch.zeros_like(sum_vectors)
        mask_any = num_matching > 0
        average_vector[mask_any] = sum_vectors[mask_any] / num_matching[mask_any]
        
        merged_state[key] = base_state[key] + lam * average_vector
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_dare(base_model, experts, lam=0.5, drop_rate=0.9):
    """
    DARE (Drop And REscale) Baseline.
    """
    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    for key in base_state.keys():
        if base_state[key].dtype not in [torch.float32, torch.float64]:
            continue
            
        task_vectors = [expert_state[key] - base_state[key] for expert_state in expert_states]
        
        processed_vectors = []
        for tv in task_vectors:
            mask = (torch.rand_like(tv) > drop_rate).float()
            scaled_tv = (tv * mask) / (1.0 - drop_rate)
            processed_vectors.append(scaled_tv)
            
        sum_vector = torch.stack(processed_vectors).sum(dim=0)
        merged_state[key] = base_state[key] + lam * sum_vector
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def regularize_spectrum(tensor, stat_type="fd_suppress", T=1.0):
    """
    Core spectrum regularizer using SVD and various statistical thermodynamic formulations.
    """
    orig_shape = tensor.shape
    if tensor.dim() < 2:
        return tensor
        
    flat_vector = tensor.view(orig_shape[0], -1)
    
    try:
        U, S, Vh = torch.linalg.svd(flat_vector, full_matrices=False)
        
        # Calculate energy levels (lower energy = larger singular value)
        energies = -S
        
        if stat_type == "fd_suppress":
            # Original ThermoMerge formula: suppresses largest, keeps tail
            cur_mu = S.mean().item()
            exponent = (S - cur_mu) / max(T, 1e-5)
            exponent = torch.clamp(exponent, -50.0, 50.0)
            S_new = 1.0 / (torch.exp(exponent) + 1.0)
            
        elif stat_type == "fd_cap":
            # Correct Fermi-Dirac: caps largest at 1.0, suppresses tail (Pauli exclusion principle)
            cur_mu = energies.mean().item()
            exponent = (energies - cur_mu) / max(T, 1e-5)
            exponent = torch.clamp(exponent, -50.0, 50.0)
            S_new = 1.0 / (torch.exp(exponent) + 1.0)
            
        elif stat_type == "be":
            # Bose-Einstein: exhibits extreme condensation on ground state (largest singular value)
            min_energy = energies.min().item()
            mu = min_energy - 1.0  # chemical potential < ground state energy
            exponent = (energies - mu) / max(T, 1e-5)
            exponent = torch.clamp(exponent, 1e-5, 50.0)
            S_new = 1.0 / (torch.exp(exponent) - 1.0)
            
        elif stat_type == "mb":
            # Maxwell-Boltzmann: classical Boltzmann energy distribution
            exponent = -energies / max(T, 1e-5)
            exponent = torch.clamp(exponent, -50.0, 50.0)
            S_new = torch.exp(exponent)
            
        elif stat_type == "iso":
            # Isotropic (completely flat)
            S_new = torch.ones_like(S)
            
        else:
            # Identity (Task Arithmetic fallback)
            S_new = S
            
        # Scale to preserve Frobenius norm (energy conservation)
        sum_orig_sq = torch.sum(S**2)
        sum_new_sq = torch.sum(S_new**2)
        if sum_new_sq > 0:
            scale = (sum_orig_sq / sum_new_sq).sqrt()
            S_scaled = S_new * scale
        else:
            S_scaled = S
            
        reconstructed = U @ torch.diag(S_scaled) @ Vh
        return reconstructed.view(orig_shape)
    except Exception as e:
        return tensor

def merge_spectrum_balancing(base_model, experts, lam=0.5, stat_type="fd_suppress", T=1.0, target_layers="classifier_only"):
    """
    Spectral Balancing Merging supporting different statistics and layer targets.
    """
    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    for key in base_state.keys():
        if base_state[key].dtype not in [torch.float32, torch.float64]:
            continue
            
        task_vectors = [expert_state[key] - base_state[key] for expert_state in expert_states]
        sum_vector = torch.stack(task_vectors).sum(dim=0)
        
        apply_svd = False
        if target_layers == "classifier_only" and key == "classifier.3.weight":
            apply_svd = True
        elif target_layers == "classifier_all" and key in ["classifier.0.weight", "classifier.3.weight"]:
            apply_svd = True
        elif target_layers == "all" and base_state[key].dim() >= 2:
            apply_svd = True
            
        if apply_svd:
            sum_vector_processed = regularize_spectrum(sum_vector, stat_type=stat_type, T=T)
            merged_state[key] = base_state[key] + lam * sum_vector_processed
        else:
            merged_state[key] = base_state[key] + lam * sum_vector
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_quantum_coherence(base_model, experts, lam=0.5, alpha=0.5, target_layers="classifier_only"):
    """
    Quantum Coherence Merging (QC-Merge).
    Projects task vectors onto the SVD consensus basis of the joint update,
    and filters the off-diagonal elements (cross-task interference) using coherence parameter alpha.
    """
    merged_model = copy.deepcopy(base_model)
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    expert_states = [expert.state_dict() for expert in experts]
    
    for key in base_state.keys():
        if base_state[key].dtype not in [torch.float32, torch.float64]:
            continue
            
        task_vectors = [expert_state[key] - base_state[key] for expert_state in expert_states]
        
        apply_qc = False
        if target_layers == "classifier_only" and key == "classifier.3.weight":
            apply_qc = True
        elif target_layers == "classifier_all" and key in ["classifier.0.weight", "classifier.3.weight"]:
            apply_qc = True
        elif target_layers == "all" and base_state[key].dim() >= 2:
            apply_qc = True
            
        if apply_qc and base_state[key].dim() >= 2:
            orig_shape = base_state[key].shape
            
            # Flatten task vectors to 2D
            task_vectors_2d = [tv.view(orig_shape[0], -1) for tv in task_vectors]
            
            # Joint update
            joint_vector_2d = torch.stack(task_vectors_2d).sum(dim=0)
            
            try:
                # SVD on joint update to find the consensus basis
                U_joint, S_joint, Vh_joint = torch.linalg.svd(joint_vector_2d, full_matrices=False)
                
                processed_task_vectors_2d = []
                for tv_2d in task_vectors_2d:
                    # Project onto the consensus basis: C = U_joint^T * tv_2d * V_joint_T
                    C = U_joint.T @ tv_2d @ Vh_joint.T
                    
                    # Extract diagonal: diag(C)
                    C_diag = torch.diag(torch.diag(C))
                    
                    # Quantum decoherence filter: C_alpha = alpha * C + (1 - alpha) * diag(C)
                    C_alpha = alpha * C + (1.0 - alpha) * C_diag
                    
                    # Reconstruct back to original space: tv_reconstructed = U_joint * C_alpha * Vh_joint
                    tv_reconstructed = U_joint @ C_alpha @ Vh_joint
                    
                    # Individual Frobenius norm (energy) conservation
                    norm_orig = torch.norm(tv_2d)
                    norm_recon = torch.norm(tv_reconstructed)
                    if norm_recon > 0:
                        tv_reconstructed = tv_reconstructed * (norm_orig / norm_recon)
                        
                    processed_task_vectors_2d.append(tv_reconstructed)
                    
                # Reconstruct back to the original parameter shape and sum
                sum_vector_2d = torch.stack(processed_task_vectors_2d).sum(dim=0)
                sum_vector = sum_vector_2d.view(orig_shape)
                
                merged_state[key] = base_state[key] + lam * sum_vector
            except Exception as e:
                # Fallback to standard sum
                print(f"SVD failed for {key}: {e}. Falling back to standard sum.")
                sum_vector = torch.stack(task_vectors).sum(dim=0)
                merged_state[key] = base_state[key] + lam * sum_vector
        else:
            sum_vector = torch.stack(task_vectors).sum(dim=0)
            merged_state[key] = base_state[key] + lam * sum_vector
            
    merged_model.load_state_dict(merged_state)
    return merged_model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading datasets...")
    loaders = get_task_loaders(batch_size=128)
    
    # 2. Initialize Base Model
    print("Initializing base model...")
    base_model = SimpleCNN(num_classes=10).to(device)
    
    # Check if we have pre-trained experts. If not, train them.
    # We start from a common random initialization as our base model.
    base_results = evaluate_model(base_model, loaders)
    print("\nPre-trained (Base) Model Evaluation:")
    for task, res in base_results.items():
        print(f"Task {task} - Loss: {res['loss']:.4f} - Acc: {res['acc']:.2f}%")
        
    # 3. Train/Load Experts
    experts = {}
    for task_name in ['A', 'B', 'C']:
        expert_path = f"expert_{task_name}.pt"
        expert_model = copy.deepcopy(base_model)
        if os.path.exists(expert_path):
            print(f"Loading pre-trained expert for Task {task_name}...")
            expert_model.load_state_dict(torch.load(expert_path, map_location=device))
        else:
            print(f"Training expert for Task {task_name}...")
            expert_model = train_model(expert_model, loaders, task_name, epochs=6, lr=0.001, device=device)
            torch.save(expert_model.state_dict(), expert_path)
            
        experts[task_name] = expert_model
        exp_results = evaluate_model(expert_model, loaders)
        print(f"\nExpert {task_name} Evaluation on all tasks:")
        for t, res in exp_results.items():
            print(f"  Task {t} - Loss: {res['loss']:.4f} - Acc: {res['acc']:.2f}%")
            
    expert_list = [experts['A'], experts['B'], experts['C']]
    
    # 4. Comprehensive Experimental Suite
    print("\n==============================================")
    print("      COMPREHENSIVE EXPERIMENT RUN            ")
    print("==============================================")
    
    results_db = []
    
    for lam in [0.3, 0.5, 0.7]:
        print(f"\n==============================================")
        print(f"Evaluating with Merge Coefficient lambda = {lam}")
        print(f"==============================================")
        
        # A. Task Arithmetic
        ta_model = merge_task_arithmetic(base_model, expert_list, lam=lam)
        ta_res = evaluate_model(ta_model, loaders)
        ta_accs = [ta_res[t]['acc'] for t in ['A', 'B', 'C']]
        avg_ta = np.mean(ta_accs)
        bal_ta = min(ta_accs) / max(ta_accs) if max(ta_accs) > 0 else 0
        print(f"Task Arithmetic:  Task A: {ta_accs[0]:.2f}%, Task B: {ta_accs[1]:.2f}%, Task C: {ta_accs[2]:.2f}% | Avg Acc: {avg_ta:.2f}% | Balance: {bal_ta:.4f}")
        results_db.append({"lam": lam, "method": "Task Arithmetic", "target_layers": "none", "T": 0.0, "accs": ta_accs, "avg_acc": avg_ta, "balance": bal_ta})
        
        # LorentzMerge (Relativistic Model Merging)
        for eta in [1.1, 1.3, 1.5, 2.0, 5.0]:
            lm_model = merge_relativistic(base_model, expert_list, lam=lam, eta=eta)
            lm_res = evaluate_model(lm_model, loaders)
            lm_accs = [lm_res[t]['acc'] for t in ['A', 'B', 'C']]
            avg_lm = np.mean(lm_accs)
            bal_lm = min(lm_accs) / max(lm_accs) if max(lm_accs) > 0 else 0
            print(f"LorentzMerge (eta={eta}): Task A: {lm_accs[0]:.2f}%, Task B: {lm_accs[1]:.2f}%, Task C: {lm_accs[2]:.2f}% | Avg Acc: {avg_lm:.2f}% | Balance: {bal_lm:.4f}")
            results_db.append({"lam": lam, "method": f"LorentzMerge", "target_layers": "all", "T": eta, "accs": lm_accs, "avg_acc": avg_lm, "balance": bal_lm})
        
        # B. TIES-Merging
        for frac in [0.2, 0.4]:
            ties_model = merge_ties(base_model, expert_list, lam=lam, fraction=frac)
            ties_res = evaluate_model(ties_model, loaders)
            ties_accs = [ties_res[t]['acc'] for t in ['A', 'B', 'C']]
            avg_ties = np.mean(ties_accs)
            bal_ties = min(ties_accs) / max(ties_accs) if max(ties_accs) > 0 else 0
            print(f"TIES (frac={frac}):  Task A: {ties_accs[0]:.2f}%, Task B: {ties_accs[1]:.2f}%, Task C: {ties_accs[2]:.2f}% | Avg Acc: {avg_ties:.2f}% | Balance: {bal_ties:.4f}")
            results_db.append({"lam": lam, "method": f"TIES (frac={frac})", "target_layers": "none", "T": 0.0, "accs": ties_accs, "avg_acc": avg_ties, "balance": bal_ties})
            
        # C. DARE
        for dr in [0.2, 0.5, 0.8]:
            dare_model = merge_dare(base_model, expert_list, lam=lam, drop_rate=dr)
            dare_res = evaluate_model(dare_model, loaders)
            dare_accs = [dare_res[t]['acc'] for t in ['A', 'B', 'C']]
            avg_dare = np.mean(dare_accs)
            bal_dare = min(dare_accs) / max(dare_accs) if max(dare_accs) > 0 else 0
            print(f"DARE (drop={dr}):  Task A: {dare_accs[0]:.2f}%, Task B: {dare_accs[1]:.2f}%, Task C: {dare_accs[2]:.2f}% | Avg Acc: {avg_dare:.2f}% | Balance: {bal_dare:.4f}")
            results_db.append({"lam": lam, "method": f"DARE (drop={dr})", "target_layers": "none", "T": 0.0, "accs": dare_accs, "avg_acc": avg_dare, "balance": bal_dare})
            
        # D. Spectral Merging (Baselines & Thermodynamic Formulations)
        # We sweep over target layer strategies
        for target_l in ["classifier_only", "classifier_all", "all"]:
            print(f"\n--- Layer selection strategy: {target_l} ---")
            
            # 1. Isotropic Baseline (Flat Spectrum)
            iso_model = merge_spectrum_balancing(base_model, expert_list, lam=lam, stat_type="iso", target_layers=target_l)
            iso_res = evaluate_model(iso_model, loaders)
            iso_accs = [iso_res[t]['acc'] for t in ['A', 'B', 'C']]
            avg_iso = np.mean(iso_accs)
            bal_iso = min(iso_accs) / max(iso_accs) if max(iso_accs) > 0 else 0
            print(f"  Isotropic:        Task A: {iso_accs[0]:.2f}%, Task B: {iso_accs[1]:.2f}%, Task C: {iso_accs[2]:.2f}% | Avg Acc: {avg_iso:.2f}% | Balance: {bal_iso:.4f}")
            results_db.append({"lam": lam, "method": "Isotropic", "target_layers": target_l, "T": 0.0, "accs": iso_accs, "avg_acc": avg_iso, "balance": bal_iso})
            
            # 2. Thermodynamic Statistics Sweeps over Temperature T
            stats_to_sweep = {
                "fd_suppress": "ThermoMerge (FD-Suppress)",
                "fd_cap": "ThermoMerge (FD-Cap)",
                "be": "BEMerge (Bose-Einstein)",
                "mb": "MBMerge (Maxwell-Boltzmann)"
            }
            
            for stat_key, stat_name in stats_to_sweep.items():
                # For a concise sweep, we do 3 key temperatures representing low, mid, and high temp regimes
                for T in [0.1, 1.0, 5.0, 10.0]:
                    tm_model = merge_spectrum_balancing(base_model, expert_list, lam=lam, stat_type=stat_key, T=T, target_layers=target_l)
                    tm_res = evaluate_model(tm_model, loaders)
                    tm_accs = [tm_res[t]['acc'] for t in ['A', 'B', 'C']]
                    avg_tm = np.mean(tm_accs)
                    bal_tm = min(tm_accs) / max(tm_accs) if max(tm_accs) > 0 else 0
                    print(f"  {stat_name} (T={T:3.1f}): Task A: {tm_accs[0]:.2f}%, Task B: {tm_accs[1]:.2f}%, Task C: {tm_accs[2]:.2f}% | Avg Acc: {avg_tm:.2f}% | Balance: {bal_tm:.4f}")
                    results_db.append({"lam": lam, "method": stat_name, "target_layers": target_l, "T": T, "accs": tm_accs, "avg_acc": avg_tm, "balance": bal_tm})

            # 3. Quantum Coherence Merging (QC-Merge)
            for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
                qcm_model = merge_quantum_coherence(base_model, expert_list, lam=lam, alpha=alpha, target_layers=target_l)
                qcm_res = evaluate_model(qcm_model, loaders)
                qcm_accs = [qcm_res[t]['acc'] for t in ['A', 'B', 'C']]
                avg_qcm = np.mean(qcm_accs)
                bal_qcm = min(qcm_accs) / max(qcm_accs) if max(qcm_accs) > 0 else 0
                print(f"  QC-Merge (alpha={alpha:3.1f}):  Task A: {qcm_accs[0]:.2f}%, Task B: {qcm_accs[1]:.2f}%, Task C: {qcm_accs[2]:.2f}% | Avg Acc: {avg_qcm:.2f}% | Balance: {bal_qcm:.4f}")
                results_db.append({"lam": lam, "method": f"QC-Merge (alpha={alpha})", "target_layers": target_l, "T": alpha, "accs": qcm_accs, "avg_acc": avg_qcm, "balance": bal_qcm})

    # Save all results to a JSON file for precise documentation
    with open("results_ablation.json", "w") as f:
        json.dump(results_db, f, indent=4)
    print("\nSaved full results database to results_ablation.json")
    print("Experiments complete.")
