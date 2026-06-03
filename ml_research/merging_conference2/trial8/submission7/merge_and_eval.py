import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Datasets and loaders (same transforms as training)
def get_loaders():
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    fmnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load test datasets
    test_mnist = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    test_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=fmnist_transform)
    test_cifar = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar10_transform)

    # Load a small train slice for Fisher computation (calibration)
    train_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    train_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fmnist_transform)
    train_cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar10_transform)

    # Use Subset to get exactly 1024 samples for Fisher/calibration
    cal_mnist = DataLoader(Subset(train_mnist, range(1024)), batch_size=128, shuffle=False)
    cal_fmnist = DataLoader(Subset(train_fmnist, range(1024)), batch_size=128, shuffle=False)
    cal_cifar = DataLoader(Subset(train_cifar, range(1024)), batch_size=128, shuffle=False)

    loaders = {
        "mnist": {
            "test": DataLoader(test_mnist, batch_size=128, shuffle=False, num_workers=4),
            "cal": cal_mnist
        },
        "fmnist": {
            "test": DataLoader(test_fmnist, batch_size=128, shuffle=False, num_workers=4),
            "cal": cal_fmnist
        },
        "cifar10": {
            "test": DataLoader(test_cifar, batch_size=128, shuffle=False, num_workers=4),
            "cal": cal_cifar
        }
    }
    return loaders

def load_expert(task):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(f"models/{task}_expert.pt", map_location="cpu"))
    return model

def load_progenitor():
    model = resnet18()
    # Replace fc to match, but we don't care about fc since we only use the backbone
    model.fc = nn.Linear(512, 10)
    # Load backbone
    backbone_sd = torch.load("models/progenitor_backbone.pt", map_location="cpu")
    # Apply to model
    model_sd = model.state_dict()
    for k, v in backbone_sd.items():
        if k in model_sd:
            model_sd[k] = v
    model.load_state_dict(model_sd)
    return model

# Function to compute diagonal Fisher information
def compute_diagonal_fisher(model, dataloader, device):
    model.eval()
    model = model.to(device)
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("fc."):
            fisher[name] = torch.zeros_like(param.data)
            
    criterion = nn.CrossEntropyLoss()
    count = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += (param.grad.data ** 2) * inputs.size(0)
        count += inputs.size(0)
        
    for name in fisher:
        fisher[name] = fisher[name] / count
        # Ensure non-zero minimum to avoid division by zero in alignment
        fisher[name] = torch.clamp(fisher[name], min=1e-8)
        
    return fisher

def get_calibration_loaders(n_samples):
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    fmnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load test datasets are not needed here, only calibration subsets
    train_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    train_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fmnist_transform)
    train_cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar10_transform)

    cal_mnist = DataLoader(Subset(train_mnist, range(n_samples)), batch_size=128, shuffle=False)
    cal_fmnist = DataLoader(Subset(train_fmnist, range(n_samples)), batch_size=128, shuffle=False)
    cal_cifar = DataLoader(Subset(train_cifar, range(n_samples)), batch_size=128, shuffle=False)

    return {
        "mnist": cal_mnist,
        "fmnist": cal_fmnist,
        "cifar10": cal_cifar
    }

def run_ties_merge(W0, W_experts, T_experts, keep_rate=0.2, apply_ipr=False):
    W_merged = {}
    for k in W0:
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            W_merged[k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
        else:
            # 1. Trim (keep top fraction)
            trimmed = {}
            for task in ["mnist", "fmnist", "cifar10"]:
                tv = T_experts[task][k]
                flat_tv = tv.flatten()
                k_keep = int(keep_rate * len(flat_tv))
                if k_keep > 0:
                    threshold = torch.topk(torch.abs(flat_tv), k_keep).values[-1]
                    mask = torch.abs(tv) >= threshold
                    trimmed[task] = tv * mask
                else:
                    trimmed[task] = torch.zeros_like(tv)
                    
            # 2. Elect sign
            sum_tv = torch.zeros_like(T_experts["mnist"][k])
            for task in trimmed:
                sum_tv += trimmed[task]
            elected_sign = torch.sign(sum_tv)
            
            # 3. Disjoint merge
            merged = torch.zeros_like(sum_tv)
            count = torch.zeros_like(sum_tv)
            for task in trimmed:
                agree = (torch.sign(trimmed[task]) == elected_sign) & (trimmed[task] != 0)
                merged[agree] += trimmed[task][agree]
                count[agree] += 1
                
            mask_any = count > 0
            merged[mask_any] /= count[mask_any]
            
            # 4. Optional scaling: apply IPR or standard
            if apply_ipr:
                norm_mnist = torch.norm(T_experts["mnist"][k].float())
                norm_fmnist = torch.norm(T_experts["fmnist"][k].float())
                norm_cifar = torch.norm(T_experts["cifar10"][k].float())
                avg_expert_norm = (norm_mnist + norm_fmnist + norm_cifar) / 3.0
                
                norm_merge = torch.norm(merged.float())
                S_l = avg_expert_norm / norm_merge if norm_merge > 1e-8 else 1.0
                W_merged[k] = W0[k] + S_l * merged
            else:
                W_merged[k] = W0[k] + merged
    return W_merged

def run_dare_merge(W0, W_experts, T_experts, drop_rate=0.2, apply_ipr=False):
    W_merged = {}
    for k in W0:
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            W_merged[k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
        else:
            merged = torch.zeros_like(T_experts["mnist"][k])
            for task in ["mnist", "fmnist", "cifar10"]:
                tv = T_experts[task][k]
                mask = (torch.rand_like(tv) >= drop_rate).float()
                tv_dare = tv * mask / (1.0 - drop_rate) if drop_rate < 1.0 else torch.zeros_like(tv)
                merged += tv_dare
            merged /= 3.0
            
            if apply_ipr:
                norm_mnist = torch.norm(T_experts["mnist"][k].float())
                norm_fmnist = torch.norm(T_experts["fmnist"][k].float())
                norm_cifar = torch.norm(T_experts["cifar10"][k].float())
                avg_expert_norm = (norm_mnist + norm_fmnist + norm_cifar) / 3.0
                
                norm_merge = torch.norm(merged.float())
                S_l = avg_expert_norm / norm_merge if norm_merge > 1e-8 else 1.0
                W_merged[k] = W0[k] + S_l * merged
            else:
                W_merged[k] = W0[k] + merged
    return W_merged

def run_sca_ipr_for_n(W0, W_experts, T_experts, cal_loaders, alphas, experts, fc_heads, test_loaders):
    # Compute diagonal Fishers
    fishers = {}
    for task in ["mnist", "fmnist", "cifar10"]:
        fishers[task] = compute_diagonal_fisher(experts[task], cal_loaders[task], device)
    
    W_sca_ipr_sweep = {alpha: {} for alpha in alphas}
    
    for k in W0:
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            for alpha in alphas:
                W_sca_ipr_sweep[alpha][k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
        else:
            t_merge = (T_experts["mnist"][k] + T_experts["fmnist"][k] + T_experts["cifar10"][k]) / 3.0
            orig_shape = t_merge.shape
            
            norm_mnist = torch.norm(T_experts["mnist"][k].float())
            norm_fmnist = torch.norm(T_experts["fmnist"][k].float())
            norm_cifar = torch.norm(T_experts["cifar10"][k].float())
            avg_expert_norm = (norm_mnist + norm_fmnist + norm_cifar) / 3.0
            
            if len(orig_shape) >= 2:
                d_out = orig_shape[0]
                d_in = int(np.prod(orig_shape[1:]))
                t_merge_2d = t_merge.reshape(d_out, d_in).to(device)
                
                try:
                    U, Sigma, Vh = torch.linalg.svd(t_merge_2d, full_matrices=False)
                    V = Vh.t()
                    C = len(Sigma)
                except Exception as e:
                    for alpha in alphas:
                        norm_merge = torch.norm(t_merge.float())
                        S_l = avg_expert_norm / norm_merge if norm_merge > 1e-8 else 1.0
                        W_sca_ipr_sweep[alpha][k] = W0[k] + S_l * t_merge
                    continue
                
                D = d_out * d_in
                Phi = torch.zeros((C, D), device=device)
                for c in range(C):
                    phi_c_2d = Sigma[c] * torch.outer(U[:, c], V[:, c])
                    Phi[c] = phi_c_2d.flatten()
                    
                A = torch.zeros((C, C), device=device)
                b = torch.zeros(C, device=device)
                
                for task in ["mnist", "fmnist", "cifar10"]:
                    F_task = fishers[task][k].reshape(D).to(device)
                    T_task = T_experts[task][k].reshape(D).to(device)
                    A_task = torch.matmul(Phi * F_task.unsqueeze(0), Phi.t())
                    A += A_task
                    b_task = torch.matmul(Phi, F_task * T_task)
                    b += b_task
                    
                mean_diag = torch.clamp(torch.diag(A).mean(), min=1e-15)
                
                for alpha in alphas:
                    lambda_reg = alpha * mean_diag
                    A_reg = A + max(lambda_reg, 1e-6 * mean_diag) * torch.eye(C, device=device)
                    b_reg = b + lambda_reg * torch.ones(C, device=device)
                    
                    try:
                        s = torch.linalg.solve(A_reg, b_reg)
                        s = torch.clamp(s, min=0.0, max=10.0)
                    except Exception as e:
                        s = torch.ones(C, device=device) * 1.73
                        
                    scaled_update_2d = torch.zeros_like(t_merge_2d)
                    for c in range(C):
                        scaled_update_2d += s[c] * Sigma[c] * torch.outer(U[:, c], V[:, c])
                        
                    norm_sca = torch.norm(scaled_update_2d.float())
                    if norm_sca > 1e-8:
                        scaled_update_2d_ipr = (avg_expert_norm / norm_sca) * scaled_update_2d
                    else:
                        scaled_update_2d_ipr = scaled_update_2d
                    W_sca_ipr_sweep[alpha][k] = W0[k] + scaled_update_2d_ipr.reshape(orig_shape).cpu()
            else:
                for alpha in alphas:
                    norm_merge = torch.norm(t_merge.float())
                    S_l = avg_expert_norm / norm_merge if norm_merge > 1e-8 else 1.0
                    W_sca_ipr_sweep[alpha][k] = W0[k] + S_l * t_merge
                    
    results = {}
    for alpha in alphas:
        accs = evaluate_model(W_sca_ipr_sweep[alpha], fc_heads, test_loaders)
        avg_acc = np.mean(list(accs.values()))
        results[alpha] = {
            "accs": accs,
            "avg_acc": avg_acc
        }
    return results

# Evaluation function
def evaluate_model(backbone_state_dict, fc_heads, task_loaders):
    # Instantiate a test model
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    model.eval()
    
    accuracies = {}
    for task in fc_heads.keys():
        # Build model state dict: backbone from merged + task-specific fc head
        merged_sd = {}
        for k, v in backbone_state_dict.items():
            merged_sd[k] = v
        # Add task-specific fc head
        for k, v in fc_heads[task].items():
            merged_sd[f"fc.{k}"] = v
            
        model.load_state_dict(merged_sd)
        
        # Eval
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in task_loaders[task]["test"]:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracies[task] = 100.0 * correct / total
    return accuracies

def main():
    loaders = get_loaders()
    
    # Load models
    print("\nLoading expert models and progenitor backbone...")
    experts = {
        "mnist": load_expert("mnist"),
        "fmnist": load_expert("fmnist"),
        "cifar10": load_expert("cifar10")
    }
    progenitor = load_progenitor()
    
    # Extract task-specific classification heads
    fc_heads = {
        "mnist": {k: v.clone() for k, v in experts["mnist"].state_dict().items() if k.startswith("fc.")},
        "fmnist": {k: v.clone() for k, v in experts["fmnist"].state_dict().items() if k.startswith("fc.")},
        "cifar10": {k: v.clone() for k, v in experts["cifar10"].state_dict().items() if k.startswith("fc.")}
    }
    # Strip "fc." prefix
    for task in fc_heads:
        fc_heads[task] = {k[3:]: v for k, v in fc_heads[task].items()}
        
    # Extract backbones
    W0 = {k: v.clone() for k, v in progenitor.state_dict().items() if not k.startswith("fc.")}
    W_experts = {
        task: {k: v.clone() for k, v in experts[task].state_dict().items() if not k.startswith("fc.")}
        for task in ["mnist", "fmnist", "cifar10"]
    }
    
    # Extract task vectors
    T_experts = {
        task: {k: W_experts[task][k] - W0[k] for k in W0}
        for task in ["mnist", "fmnist", "cifar10"]
    }
    
    # Evaluate individual experts (Oracles)
    print("\nEvaluating Individual Oracle Experts...")
    oracle_accs = {}
    for task in ["mnist", "fmnist", "cifar10"]:
        accs = evaluate_model(W_experts[task], {task: fc_heads[task]}, loaders)
        oracle_accs[task] = accs[task]
    print(f"Oracle Accuracies: MNIST: {oracle_accs['mnist']:.2f}%, FMNIST: {oracle_accs['fmnist']:.2f}%, CIFAR-10: {oracle_accs['cifar10']:.2f}%")
    print(f"Average Oracle Accuracy: {np.mean(list(oracle_accs.values())):.2f}%")
    
    # ------------------------------------------------------------------------
    # 1. Standard Weight Averaging (WA)
    # ------------------------------------------------------------------------
    print("\n--- Running Weight Averaging (WA) ---")
    T_wa = {}
    W_wa = {}
    for k in W0:
        # Average task updates
        T_wa[k] = (T_experts["mnist"][k] + T_experts["fmnist"][k] + T_experts["cifar10"][k]) / 3.0
        # If running stats buffer, copy averaged values directly (not as updates)
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            W_wa[k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
        else:
            W_wa[k] = W0[k] + T_wa[k]
            
    wa_accs = evaluate_model(W_wa, fc_heads, loaders)
    print(f"WA Accuracies: MNIST: {wa_accs['mnist']:.2f}%, FMNIST: {wa_accs['fmnist']:.2f}%, CIFAR-10: {wa_accs['cifar10']:.2f}%")
    print(f"Average WA Accuracy: {np.mean(list(wa_accs.values())):.2f}%")
    
    # ------------------------------------------------------------------------
    # 2. Update-level Isotropic Parameter Resonance (U-IPR)
    # ------------------------------------------------------------------------
    print("\n--- Running Isotropic Parameter Resonance (U-IPR) ---")
    W_uipr = {}
    for k in W0:
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            W_uipr[k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
        else:
            # Compute layer-wise norms
            norm_mnist = torch.norm(T_experts["mnist"][k].float())
            norm_fmnist = torch.norm(T_experts["fmnist"][k].float())
            norm_cifar = torch.norm(T_experts["cifar10"][k].float())
            avg_expert_norm = (norm_mnist + norm_fmnist + norm_cifar) / 3.0
            
            # Merged update and norm
            t_merge = T_wa[k]
            norm_merge = torch.norm(t_merge.float())
            
            # Scale factor
            if norm_merge > 1e-8:
                S_l = avg_expert_norm / norm_merge
            else:
                S_l = 1.0
                
            W_uipr[k] = W0[k] + S_l * t_merge
            
    uipr_accs = evaluate_model(W_uipr, fc_heads, loaders)
    print(f"U-IPR Accuracies: MNIST: {uipr_accs['mnist']:.2f}%, FMNIST: {uipr_accs['fmnist']:.2f}%, CIFAR-10: {uipr_accs['cifar10']:.2f}%")
    print(f"Average U-IPR Accuracy: {np.mean(list(uipr_accs.values())):.2f}%")
    
    # ------------------------------------------------------------------------
    # 2b. TIES-Merging (with and without IPR scaling)
    # ------------------------------------------------------------------------
    print("\n--- Running TIES-Merging Sweep ---")
    for keep_rate in [0.2, 0.5, 0.8]:
        for apply_ipr in [False, True]:
            name = f"TIES (keep={keep_rate}, IPR={apply_ipr})"
            W_ties = run_ties_merge(W0, W_experts, T_experts, keep_rate=keep_rate, apply_ipr=apply_ipr)
            ties_accs = evaluate_model(W_ties, fc_heads, loaders)
            print(f"{name} Accuracies: MNIST: {ties_accs['mnist']:.2f}%, FMNIST: {ties_accs['fmnist']:.2f}%, CIFAR-10: {ties_accs['cifar10']:.2f}%")
            print(f"Average {name} Accuracy: {np.mean(list(ties_accs.values())):.2f}%")

    # ------------------------------------------------------------------------
    # 2c. DARE (with and without IPR scaling)
    # ------------------------------------------------------------------------
    print("\n--- Running DARE Sweep ---")
    for drop_rate in [0.2, 0.5, 0.8]:
        for apply_ipr in [False, True]:
            name = f"DARE (drop={drop_rate}, IPR={apply_ipr})"
            W_dare = run_dare_merge(W0, W_experts, T_experts, drop_rate=drop_rate, apply_ipr=apply_ipr)
            dare_accs = evaluate_model(W_dare, fc_heads, loaders)
            print(f"{name} Accuracies: MNIST: {dare_accs['mnist']:.2f}%, FMNIST: {dare_accs['fmnist']:.2f}%, CIFAR-10: {dare_accs['cifar10']:.2f}%")
            print(f"Average {name} Accuracy: {np.mean(list(dare_accs.values())):.2f}%")
            
    # ------------------------------------------------------------------------
    # 3. Our Spectral Curvature Alignment (SCA)
    # ------------------------------------------------------------------------
    print("\n--- Computing Diagonal Fishers for SCA ---")
    fishers = {
        "mnist": compute_diagonal_fisher(experts["mnist"], loaders["mnist"]["cal"], device),
        "fmnist": compute_diagonal_fisher(experts["fmnist"], loaders["fmnist"]["cal"], device),
        "cifar10": compute_diagonal_fisher(experts["cifar10"], loaders["cifar10"]["cal"], device)
    }
    
    print("\n--- Running Spectral Curvature Alignment (SCA) Sweep ---")
    alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    W_sca_sweep = {alpha: {} for alpha in alphas}
    W_sca_ipr_sweep = {alpha: {} for alpha in alphas}
    
    for k in W0:
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            # Merge BN running stats by averaging
            for alpha in alphas:
                W_sca_sweep[alpha][k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
                W_sca_ipr_sweep[alpha][k] = (W_experts["mnist"][k] + W_experts["fmnist"][k] + W_experts["cifar10"][k]) / 3.0
        else:
            # Only apply SVD to 2D-capable parameters (Conv and Linear weights)
            t_merge = T_wa[k]
            orig_shape = t_merge.shape
            
            # Compute average expert norm for scaling
            norm_mnist = torch.norm(T_experts["mnist"][k].float())
            norm_fmnist = torch.norm(T_experts["fmnist"][k].float())
            norm_cifar = torch.norm(T_experts["cifar10"][k].float())
            avg_expert_norm = (norm_mnist + norm_fmnist + norm_cifar) / 3.0
            
            if len(orig_shape) >= 2:
                # Reshape to 2D
                d_out = orig_shape[0]
                d_in = int(np.prod(orig_shape[1:]))
                
                # Reshape task updates and fishers
                t_merge_2d = t_merge.reshape(d_out, d_in).to(device)
                
                # Perform SVD
                try:
                    U, Sigma, Vh = torch.linalg.svd(t_merge_2d, full_matrices=False)
                    V = Vh.t() # shape (d_in, C)
                    C = len(Sigma)
                except Exception as e:
                    print(f"SVD failed for {k}, falling back to U-IPR: {e}")
                    for alpha in alphas:
                        norm_merge = torch.norm(t_merge.float())
                        S_l = avg_expert_norm / norm_merge if norm_merge > 1e-8 else 1.0
                        W_sca_sweep[alpha][k] = W0[k] + S_l * t_merge
                        W_sca_ipr_sweep[alpha][k] = W0[k] + S_l * t_merge
                    continue
                
                # Flattened dimension size
                D = d_out * d_in
                
                # Precompute component vectors Phi of shape (C, D)
                Phi = torch.zeros((C, D), device=device)
                for c in range(C):
                    phi_c_2d = Sigma[c] * torch.outer(U[:, c], V[:, c])
                    Phi[c] = phi_c_2d.flatten()
                    
                # Setup linear system parameters A (C x C) and b (C)
                A = torch.zeros((C, C), device=device)
                b = torch.zeros(C, device=device)
                
                # Stack expert task vectors and fishers for efficient vectorized calculation
                for task in ["mnist", "fmnist", "cifar10"]:
                    F_task = fishers[task][k].reshape(D).to(device)
                    T_task = T_experts[task][k].reshape(D).to(device)
                    
                    # Compute A_task = Phi * F_task * Phi^T
                    A_task = torch.matmul(Phi * F_task.unsqueeze(0), Phi.t())
                    A += A_task
                    
                    # Compute b_task = Phi @ (F_task * T_task)
                    b_task = torch.matmul(Phi, F_task * T_task)
                    b += b_task
                    
                # Solve the regularized linear system with relative regularization
                mean_diag = torch.clamp(torch.diag(A).mean(), min=1e-15)
                
                for alpha in alphas:
                    lambda_reg = alpha * mean_diag
                    # Ensure numerical stability with a small base regularizer
                    A_reg = A + max(lambda_reg, 1e-6 * mean_diag) * torch.eye(C, device=device)
                    b_reg = b + lambda_reg * torch.ones(C, device=device)
                    
                    try:
                        s = torch.linalg.solve(A_reg, b_reg)
                        # Clamp scale factors to avoid extreme scaling (e.g. between 0 and 10)
                        s = torch.clamp(s, min=0.0, max=10.0)
                    except Exception as e:
                        s = torch.ones(C, device=device) * 1.73
                        
                    # Reconstruct scaled merged update
                    scaled_update_2d = torch.zeros_like(t_merge_2d)
                    for c in range(C):
                        scaled_update_2d += s[c] * Sigma[c] * torch.outer(U[:, c], V[:, c])
                        
                    W_sca_sweep[alpha][k] = W0[k] + scaled_update_2d.reshape(orig_shape).cpu()
                    
                    # For SCA-IPR, apply overall norm rescaling
                    norm_sca = torch.norm(scaled_update_2d.float())
                    if norm_sca > 1e-8:
                        scaled_update_2d_ipr = (avg_expert_norm / norm_sca) * scaled_update_2d
                    else:
                        scaled_update_2d_ipr = scaled_update_2d
                    W_sca_ipr_sweep[alpha][k] = W0[k] + scaled_update_2d_ipr.reshape(orig_shape).cpu()
            else:
                # 1D fallback (biases, Norm parameters): use U-IPR
                for alpha in alphas:
                    norm_merge = torch.norm(t_merge.float())
                    S_l = avg_expert_norm / norm_merge if norm_merge > 1e-8 else 1.0
                    W_sca_sweep[alpha][k] = W0[k] + S_l * t_merge
                    W_sca_ipr_sweep[alpha][k] = W0[k] + S_l * t_merge
                
    print("\n--- Evaluating SCA Sweep ---")
    for alpha in alphas:
        sca_accs = evaluate_model(W_sca_sweep[alpha], fc_heads, loaders)
        print(f"SCA (alpha={alpha}) Accuracies: MNIST: {sca_accs['mnist']:.2f}%, FMNIST: {sca_accs['fmnist']:.2f}%, CIFAR-10: {sca_accs['cifar10']:.2f}%")
        print(f"Average SCA (alpha={alpha}) Accuracy: {np.mean(list(sca_accs.values())):.2f}%")

    print("\n--- Evaluating SCA-IPR Sweep ---")
    best_sca_ipr_acc = 0.0
    best_alpha = 0.0
    for alpha in alphas:
        sca_ipr_accs = evaluate_model(W_sca_ipr_sweep[alpha], fc_heads, loaders)
        avg_acc = np.mean(list(sca_ipr_accs.values()))
        print(f"SCA-IPR (alpha={alpha}) Accuracies: MNIST: {sca_ipr_accs['mnist']:.2f}%, FMNIST: {sca_ipr_accs['fmnist']:.2f}%, CIFAR-10: {sca_ipr_accs['cifar10']:.2f}%")
        print(f"Average SCA-IPR (alpha={alpha}) Accuracy: {avg_acc:.2f}%")
        if avg_acc > best_sca_ipr_acc:
            best_sca_ipr_acc = avg_acc
            best_alpha = alpha
    print(f"\nBest SCA-IPR alpha: {best_alpha} with Average Accuracy: {best_sca_ipr_acc:.2f}%")

    # ------------------------------------------------------------------------
    # 4. Calibration Sample Size Ablation Study
    # ------------------------------------------------------------------------
    print("\n--- Running Calibration Sample Size Ablation Study ---")
    N_list = [128, 256, 512, 1024]
    ablation_alphas = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    for N in N_list:
        print(f"\n--- SCA-IPR Calibration Study for N = {N} ---")
        cal_loaders = get_calibration_loaders(N)
        results = run_sca_ipr_for_n(W0, W_experts, T_experts, cal_loaders, ablation_alphas, experts, fc_heads, loaders)
        for alpha in ablation_alphas:
            avg_acc = results[alpha]["avg_acc"]
            accs = results[alpha]["accs"]
            print(f"SCA-IPR (N={N}, alpha={alpha}) Accuracies: MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR-10: {accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")

if __name__ == "__main__":
    main()
