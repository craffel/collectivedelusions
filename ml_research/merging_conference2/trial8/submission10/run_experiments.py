import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on the GPU nodes
torch.backends.cudnn.enabled = False

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Dataset Preparation
# ==========================================
def get_dataloaders(batch_size=256):
    print("Preparing datasets...")
    
    # Transforms
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert 1 channel to 3 channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    color_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Datasets
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=gray_transform)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=gray_transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=gray_transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=gray_transform)
    
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=color_transform)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=color_transform)
    
    # Dataloaders
    loaders = {
        "mnist": {
            "train": DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            "test": DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
            "raw_train": mnist_train
        },
        "fmnist": {
            "train": DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            "test": DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
            "raw_train": fmnist_train
        },
        "cifar": {
            "train": DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2),
            "test": DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2),
            "raw_train": cifar_train
        }
    }
    return loaders

# ==========================================
# 2. Model Definitions
# ==========================================
class ExpertModelResNet(nn.Module):
    def __init__(self, progenitor=None):
        super().__init__()
        # Load standard ResNet-18
        if progenitor is not None:
            self.backbone = torchvision.models.resnet18(weights=None)
            self.backbone.fc = nn.Identity()
            self.backbone.load_state_dict(progenitor.backbone.state_dict())
        else:
            self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 10)

class ExpertModelMLP(nn.Module):
    def __init__(self, progenitor=None):
        super().__init__()
        if progenitor is not None:
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3072, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
            )
            self.backbone.load_state_dict(progenitor.backbone.state_dict())
        else:
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3072, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
            )
        self.fc = nn.Linear(512, 10)

# ==========================================
# 3. Training Utilities
# ==========================================
def train_model(model, train_loader, epochs=5, lr=1e-3, weight_decay=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.backbone(images)
            outputs = model.fc(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
    return model

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.backbone(images)
            outputs = model.fc(outputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

# ==========================================
# 4. Merging and Calibration Algorithms
# ==========================================
def compute_weight_averaging(progenitor_state, expert_backbones):
    # Standard Weight Averaging: W_init + 1/K * \sum T_k
    K = len(expert_backbones)
    merged_state = {}
    for key in progenitor_state.keys():
        w_init = progenitor_state[key]
        t_sum = sum(expert[key] - w_init for expert in expert_backbones)
        merged_state[key] = w_init + (t_sum / K)
    return merged_state

def compute_task_arithmetic(progenitor_state, expert_backbones, lmbda):
    # Task Arithmetic: W_init + \lambda * \sum T_k
    merged_state = {}
    for key in progenitor_state.keys():
        w_init = progenitor_state[key]
        t_sum = sum(expert[key] - w_init for expert in expert_backbones)
        merged_state[key] = w_init + lmbda * t_sum
    return merged_state

def apply_hns(progenitor_state, expert_backbones, merged_backbone_state, task_idx):
    # Holographic Norm Scaling (HNS) for a specific task
    reconstructed_state = {}
    for key in progenitor_state.keys():
        w_init = progenitor_state[key]
        w_merged = merged_backbone_state[key]
        w_expert = expert_backbones[task_idx][key]
        
        # If weight matrix with Cout dimension (length >= 2)
        if len(w_init.shape) >= 2:
            t_expert = w_expert - w_init
            t_merged = w_merged - w_init
            
            # Channel-wise norms (dim 0 is output channels)
            t_expert_flat = t_expert.flatten(1)
            t_merged_flat = t_merged.flatten(1)
            
            norm_expert = torch.norm(t_expert_flat, p=2, dim=1)
            norm_merged = torch.norm(t_merged_flat, p=2, dim=1)
            
            gamma = norm_expert / (norm_merged + 1e-8)
            gamma = torch.clamp(gamma, min=0.1, max=10.0)
            
            # Broadcast gamma
            view_shape = [w_init.shape[0]] + [1] * (len(w_init.shape) - 1)
            gamma_view = gamma.view(view_shape)
            
            w_rec = w_init + gamma_view * t_merged
            reconstructed_state[key] = w_rec.to(w_merged.dtype)
        else:
            # For 1D parameters, HNS uses the expert's own parameter
            reconstructed_state[key] = w_expert
    return reconstructed_state

def apply_u_ipr(progenitor_state, expert_backbones, merged_backbone_state):
    # Update-level Isotropic Parameter Resonance (U-IPR)
    corrected_state = {}
    K = len(expert_backbones)
    
    for key in progenitor_state.keys():
        w_init = progenitor_state[key]
        w_merged = merged_backbone_state[key]
        
        # Keep averaged buffers as is
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            corrected_state[key] = w_merged
            continue
            
        t_experts = [expert[key] - w_init for expert in expert_backbones]
        t_merged = w_merged - w_init
        
        norm_experts = [torch.norm(t_expert.float(), p="fro") for t_expert in t_experts]
        norm_merged = torch.norm(t_merged.float(), p="fro")
        
        avg_norm_experts = sum(norm_experts) / K
        s_l = avg_norm_experts / (norm_merged + 1e-8)
        s_l = torch.clamp(s_l, min=0.1, max=10.0)
        
        w_rec = w_init + s_l * t_merged
        corrected_state[key] = w_rec.to(w_merged.dtype)
    return corrected_state

def apply_s_ipr(progenitor_state, expert_backbones, merged_backbone_state):
    # Spectral Parameter Resonance (S-IPR)
    corrected_state = {}
    K = len(expert_backbones)
    
    for key in progenitor_state.keys():
        w_init = progenitor_state[key]
        w_merged = merged_backbone_state[key]
        
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            corrected_state[key] = w_merged
            continue
            
        if len(w_init.shape) < 2:
            # Fallback to U-IPR
            t_experts = [expert[key] - w_init for expert in expert_backbones]
            t_merged = w_merged - w_init
            norm_experts = [torch.norm(t_expert.float(), p="fro") for t_expert in t_experts]
            norm_merged = torch.norm(t_merged.float(), p="fro")
            avg_norm_experts = sum(norm_experts) / K
            s_l = avg_norm_experts / (norm_merged + 1e-8)
            s_l = torch.clamp(s_l, min=0.1, max=10.0)
            corrected_state[key] = (w_init + s_l * t_merged).to(w_merged.dtype)
            continue
            
        orig_shape = w_init.shape
        R = orig_shape[0]
        C = w_init.numel() // R
        
        t_experts = [(expert[key] - w_init).view(R, C).float() for expert in expert_backbones]
        t_merged = (w_merged - w_init).view(R, C).float()
        
        expert_singular_values = []
        for t_expert in t_experts:
            try:
                _, S, _ = torch.linalg.svd(t_expert, full_matrices=False)
                expert_singular_values.append(S)
            except RuntimeError:
                expert_singular_values.append(torch.norm(t_expert) / (R * C)**0.5 * torch.ones(min(R, C), device=w_init.device))
                
        avg_S = sum(expert_singular_values) / K
        
        try:
            U_merged, S_merged, Vh_merged = torch.linalg.svd(t_merged, full_matrices=False)
            t_corrected = U_merged @ torch.diag(avg_S) @ Vh_merged
        except RuntimeError:
            t_corrected = t_merged
            
        w_rec = w_init + t_corrected.view(orig_shape)
        corrected_state[key] = w_rec.to(w_merged.dtype)
    return corrected_state

def apply_sa_ipr(progenitor_state, expert_backbones, merged_backbone_state, alpha=0.5):
    # Subspace-Aligned Isotropic Parameter Resonance (SA-IPR)
    corrected_state = {}
    K = len(expert_backbones)
    
    for key in progenitor_state.keys():
        w_init = progenitor_state[key]
        w_merged = merged_backbone_state[key]
        
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            corrected_state[key] = w_merged
            continue
            
        if len(w_init.shape) < 2:
            # Fallback to U-IPR
            t_experts = [expert[key] - w_init for expert in expert_backbones]
            t_merged = w_merged - w_init
            norm_experts = [torch.norm(t_expert.float(), p="fro") for t_expert in t_experts]
            norm_merged = torch.norm(t_merged.float(), p="fro")
            avg_norm_experts = sum(norm_experts) / K
            s_l = avg_norm_experts / (norm_merged + 1e-8)
            s_l = torch.clamp(s_l, min=0.1, max=10.0)
            corrected_state[key] = (w_init + s_l * t_merged).to(w_merged.dtype)
            continue
            
        orig_shape = w_init.shape
        R = orig_shape[0]
        C = w_init.numel() // R
        min_dim = min(R, C)
        
        t_experts = [(expert[key] - w_init).view(R, C).float() for expert in expert_backbones]
        
        U_list = []
        V_list = []
        expert_norms = []
        for t_expert in t_experts:
            expert_norms.append(torch.norm(t_expert, p="fro"))
            try:
                U_k, _, Vh_k = torch.linalg.svd(t_expert, full_matrices=False)
                U_list.append(U_k)
                V_list.append(Vh_k.T)
            except RuntimeError:
                U_list.append(torch.eye(R, min_dim, device=w_init.device))
                V_list.append(torch.eye(C, min_dim, device=w_init.device))
                
        r = max(1, int(alpha * min_dim))
        U_concat = torch.cat([U_k[:, :r] for U_k in U_list], dim=1)
        V_concat = torch.cat([V_k[:, :r] for V_k in V_list], dim=1)
        
        try:
            U_joint, _, _ = torch.linalg.svd(U_concat, full_matrices=False)
            V_joint, _, _ = torch.linalg.svd(V_concat, full_matrices=False)
            
            d = min(min_dim, r)
            P_U = U_joint[:, :d]
            P_V = V_joint[:, :d]
            
            t_experts_aligned = []
            for t_expert in t_experts:
                aligned = P_U @ (P_U.T @ t_expert @ P_V) @ P_V.T
                t_experts_aligned.append(aligned)
                
            t_merged_aligned = sum(t_experts_aligned) / K
        except RuntimeError:
            t_merged_aligned = sum(t_experts) / K
            
        norm_merged_aligned = torch.norm(t_merged_aligned, p="fro")
        avg_norm_experts = sum(expert_norms) / K
        s_l_aligned = avg_norm_experts / (norm_merged_aligned + 1e-8)
        s_l_aligned = torch.clamp(s_l_aligned, min=0.1, max=10.0)
        
        t_corrected = s_l_aligned * t_merged_aligned
        w_rec = w_init + t_corrected.view(orig_shape)
        corrected_state[key] = w_rec.to(w_merged.dtype)
    return corrected_state

def apply_de_bn(model, raw_train_dataset, num_samples=32):
    # Data-Efficient BatchNorm Calibration
    # We disable grad, put in train, reset running stats, run num_samples, restore momentum
    model.train()
    torch.set_grad_enabled(False)
    
    # Reset stats
    bn_modules = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            bn_modules.append(m)
            
    if not bn_modules:
        torch.set_grad_enabled(True)
        return model # No BatchNorm layers
        
    # Get random subset of training data
    indices = random.sample(range(len(raw_train_dataset)), min(num_samples, len(raw_train_dataset)))
    subset = Subset(raw_train_dataset, indices)
    calib_loader = DataLoader(subset, batch_size=min(32, num_samples), shuffle=False)
    
    for images, _ in calib_loader:
        images = images.to(device)
        model.backbone(images)
        
    for m in bn_modules:
        m.momentum = 0.1
        
    torch.set_grad_enabled(True)
    return model

# ==========================================
# 5. Pipeline Orchestrator
# ==========================================
def run_all_experiments():
    loaders = get_dataloaders()
    tasks = ["mnist", "fmnist", "cifar"]
    
    # We will run this for both ResNet-18 and MLP
    architectures = ["resnet", "mlp"]
    results = {arch: {} for arch in architectures}
    
    for arch in architectures:
        print(f"\n==========================================")
        print(f"RUNNING ARCHITECTURE: {arch.upper()}")
        print(f"==========================================")
        
        # Initialize progenitor and save weights
        if arch == "resnet":
            progenitor = ExpertModelResNet().to(device)
        else:
            progenitor = ExpertModelMLP().to(device)
            
        progenitor_path = f"{arch}_progenitor.pt"
        if os.path.exists(progenitor_path):
            print(f"Loading progenitor from {progenitor_path}...")
            progenitor.backbone.load_state_dict(torch.load(progenitor_path, map_location=device))
        else:
            progenitor_state = copy.deepcopy(progenitor.backbone.state_dict())
            torch.save(progenitor_state, progenitor_path)
            
        progenitor_state = copy.deepcopy(progenitor.backbone.state_dict())
        
        # Train or load experts
        experts = []
        expert_backbones = []
        expert_heads = []
        expert_accs = {}
        
        for task in tasks:
            expert_path = f"{arch}_expert_{task}.pt"
            if arch == "resnet":
                model = ExpertModelResNet(progenitor).to(device)
            else:
                model = ExpertModelMLP(progenitor).to(device)
                
            if os.path.exists(expert_path):
                print(f"Loading expert from {expert_path}...")
                checkpoint = torch.load(expert_path, map_location=device)
                model.backbone.load_state_dict(checkpoint["backbone"])
                model.fc.load_state_dict(checkpoint["fc"])
                acc = evaluate_model(model, loaders[task]["test"])
                print(f"Expert {task.upper()} loaded. Test Accuracy: {acc:.2f}%")
            else:
                print(f"Training Expert for {task.upper()}...")
                model = train_model(model, loaders[task]["train"], epochs=5)
                acc = evaluate_model(model, loaders[task]["test"])
                print(f"Expert {task.upper()} Test Accuracy: {acc:.2f}%")
                
                # Save checkpoint
                torch.save({
                    "backbone": model.backbone.state_dict(),
                    "fc": model.fc.state_dict()
                }, expert_path)
            
            experts.append(model)
            expert_backbones.append(copy.deepcopy(model.backbone.state_dict()))
            expert_heads.append(copy.deepcopy(model.fc.state_dict()))
            expert_accs[task] = acc
            
        results[arch]["expert_oracles"] = expert_accs
        
        # ==========================================
        # Evaluator Helper
        # ==========================================
        def evaluate_merged_backbone(backbone_state, calib_samples=0, task_to_calib=None):
            # Evaluate the given backbone state on all three tasks using task-specific heads
            accs = {}
            for i, task in enumerate(tasks):
                if arch == "resnet":
                    test_model = ExpertModelResNet().to(device)
                else:
                    test_model = ExpertModelMLP().to(device)
                    
                test_model.backbone.load_state_dict(backbone_state)
                test_model.fc.load_state_dict(expert_heads[i])
                
                # Apply DE-BN calibration if requested
                if calib_samples > 0 and arch == "resnet" and (task_to_calib is None or task_to_calib == task):
                    test_model = apply_de_bn(test_model, loaders[task]["raw_train"], calib_samples)
                    
                accs[task] = evaluate_model(test_model, loaders[task]["test"])
            return accs
            
        # 1. Weight Averaging (WA) Uncalibrated
        print("Evaluating Weight Averaging...")
        wa_backbone = compute_weight_averaging(progenitor_state, expert_backbones)
        results[arch]["wa_uncalibrated"] = evaluate_merged_backbone(wa_backbone)
        
        # 2. Task Arithmetic (TA) Grid-Search
        print("Evaluating Task Arithmetic Grid-Search...")
        ta_results = []
        lambdas = np.linspace(0.1, 1.5, 15)
        for lmbda in lambdas:
            ta_backbone = compute_task_arithmetic(progenitor_state, expert_backbones, lmbda)
            accs = evaluate_merged_backbone(ta_backbone)
            avg_acc = sum(accs.values()) / len(accs)
            ta_results.append((lmbda, accs, avg_acc))
            print(f"  Lambda: {lmbda:.2f} - Avg Acc: {avg_acc:.2f}% (MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR: {accs['cifar']:.2f}%)")
            
        # Best uncalibrated TA
        best_ta = max(ta_results, key=lambda x: x[2])
        results[arch]["best_ta_uncalibrated"] = {
            "lambda": best_ta[0],
            "accs": best_ta[1],
            "avg_acc": best_ta[2]
        }
        
        # 3. Holographic Norm Scaling (HNS)
        # HNS is task-specific! So we reconstruct the backbone specifically for each task i
        print("Evaluating HNS (Task-Specific)...")
        hns_accs = {}
        for i, task in enumerate(tasks):
            hns_backbone_i = apply_hns(progenitor_state, expert_backbones, wa_backbone, task_idx=i)
            # Evaluate ONLY on the specific task using task's head
            if arch == "resnet":
                test_model = ExpertModelResNet().to(device)
            else:
                test_model = ExpertModelMLP().to(device)
                
            test_model.backbone.load_state_dict(hns_backbone_i)
            test_model.fc.load_state_dict(expert_heads[i])
            hns_accs[task] = evaluate_model(test_model, loaders[task]["test"])
            
        results[arch]["hns"] = hns_accs
        
        # 4. Update-level Isotropic Parameter Resonance (U-IPR)
        print("Evaluating U-IPR...")
        u_ipr_backbone = apply_u_ipr(progenitor_state, expert_backbones, wa_backbone)
        results[arch]["u_ipr"] = evaluate_merged_backbone(u_ipr_backbone)
        
        # 5. Spectral Parameter Resonance (S-IPR)
        print("Evaluating S-IPR...")
        s_ipr_backbone = apply_s_ipr(progenitor_state, expert_backbones, wa_backbone)
        results[arch]["s_ipr"] = evaluate_merged_backbone(s_ipr_backbone)
        
        # 6. Subspace-Aligned Isotropic Parameter Resonance (SA-IPR)
        print("Evaluating SA-IPR...")
        for alpha in [0.3, 0.5, 0.7]:
            sa_ipr_backbone = apply_sa_ipr(progenitor_state, expert_backbones, wa_backbone, alpha=alpha)
            results[arch][f"sa_ipr_alpha_{alpha}"] = evaluate_merged_backbone(sa_ipr_backbone)
            
        # 7. Data-Efficient BatchNorm Calibration (DE-BN) on top of WA
        if arch == "resnet":
            print("Evaluating DE-BN on top of Weight Averaging...")
            results[arch]["de_bn_wa"] = {}
            for N in [8, 16, 32, 64, 128]:
                results[arch]["de_bn_wa"][N] = evaluate_merged_backbone(wa_backbone, calib_samples=N)
                
            print("Evaluating DE-BN on top of Best Task Arithmetic...")
            results[arch]["de_bn_ta"] = {}
            best_ta_backbone = compute_task_arithmetic(progenitor_state, expert_backbones, best_ta[0])
            for N in [8, 16, 32, 64, 128]:
                results[arch]["de_bn_ta"][N] = evaluate_merged_backbone(best_ta_backbone, calib_samples=N)
                
    # ==========================================
    # Print Synthesis
    # ==========================================
    print("\n==========================================")
    print("FINAL EXPERIMENTAL SUMMARY")
    print("==========================================")
    for arch in architectures:
        print(f"\nArchitecture: {arch.upper()}")
        print(f"  Oracle Experts:    "
              f"MNIST: {results[arch]['expert_oracles']['mnist']:.2f}%, "
              f"FMNIST: {results[arch]['expert_oracles']['fmnist']:.2f}%, "
              f"CIFAR: {results[arch]['expert_oracles']['cifar']:.2f}%, "
              f"Avg: {sum(results[arch]['expert_oracles'].values())/3:.2f}%")
              
        print(f"  Weight Averaging:  "
              f"MNIST: {results[arch]['wa_uncalibrated']['mnist']:.2f}%, "
              f"FMNIST: {results[arch]['wa_uncalibrated']['fmnist']:.2f}%, "
              f"CIFAR: {results[arch]['wa_uncalibrated']['cifar']:.2f}%, "
              f"Avg: {sum(results[arch]['wa_uncalibrated'].values())/3:.2f}%")
              
        best_ta_lambda = results[arch]['best_ta_uncalibrated']['lambda']
        print(f"  Tuned TA (\u03bb={best_ta_lambda:.2f}):   "
              f"MNIST: {results[arch]['best_ta_uncalibrated']['accs']['mnist']:.2f}%, "
              f"FMNIST: {results[arch]['best_ta_uncalibrated']['accs']['fmnist']:.2f}%, "
              f"CIFAR: {results[arch]['best_ta_uncalibrated']['accs']['cifar']:.2f}%, "
              f"Avg: {results[arch]['best_ta_uncalibrated']['avg_acc']:.2f}%")
              
        print(f"  HNS:               "
              f"MNIST: {results[arch]['hns']['mnist']:.2f}%, "
              f"FMNIST: {results[arch]['hns']['fmnist']:.2f}%, "
              f"CIFAR: {results[arch]['hns']['cifar']:.2f}%, "
              f"Avg: {sum(results[arch]['hns'].values())/3:.2f}%")
              
        print(f"  U-IPR (Ours):      "
              f"MNIST: {results[arch]['u_ipr']['mnist']:.2f}%, "
              f"FMNIST: {results[arch]['u_ipr']['fmnist']:.2f}%, "
              f"CIFAR: {results[arch]['u_ipr']['cifar']:.2f}%, "
              f"Avg: {sum(results[arch]['u_ipr'].values())/3:.2f}%")
              
        print(f"  S-IPR (Ours):      "
              f"MNIST: {results[arch]['s_ipr']['mnist']:.2f}%, "
              f"FMNIST: {results[arch]['s_ipr']['fmnist']:.2f}%, "
              f"CIFAR: {results[arch]['s_ipr']['cifar']:.2f}%, "
              f"Avg: {sum(results[arch]['s_ipr'].values())/3:.2f}%")
              
        for alpha in [0.3, 0.5, 0.7]:
            sa_key = f"sa_ipr_alpha_{alpha}"
            print(f"  SA-IPR (\u03b1={alpha:.1f}):     "
                  f"MNIST: {results[arch][sa_key]['mnist']:.2f}%, "
                  f"FMNIST: {results[arch][sa_key]['fmnist']:.2f}%, "
                  f"CIFAR: {results[arch][sa_key]['cifar']:.2f}%, "
                  f"Avg: {sum(results[arch][sa_key].values())/3:.2f}%")
                  
        if arch == "resnet":
            print("  --- Calibration Methods ---")
            for N in [8, 16, 32, 64, 128]:
                print(f"  DE-BN (WA, N={N}):   "
                      f"MNIST: {results[arch]['de_bn_wa'][N]['mnist']:.2f}%, "
                      f"FMNIST: {results[arch]['de_bn_wa'][N]['fmnist']:.2f}%, "
                      f"CIFAR: {results[arch]['de_bn_wa'][N]['cifar']:.2f}%, "
                      f"Avg: {sum(results[arch]['de_bn_wa'][N].values())/3:.2f}%")
                print(f"  DE-BN (TA, N={N}):   "
                      f"MNIST: {results[arch]['de_bn_ta'][N]['mnist']:.2f}%, "
                      f"FMNIST: {results[arch]['de_bn_ta'][N]['fmnist']:.2f}%, "
                      f"CIFAR: {results[arch]['de_bn_ta'][N]['cifar']:.2f}%, "
                      f"Avg: {sum(results[arch]['de_bn_ta'][N].values())/3:.2f}%")
                      
    # Save the results dictionary for further analysis and plotting
    torch.save(results, "all_experimental_results.pt")
    
    # ==========================================
    # 6. Plotting Results
    # ==========================================
    print("Generating plots...")
    generate_plots(results)
    print("All done!")

def generate_plots(results):
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Bar Chart comparing methods on ResNet-18 vs MLP
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ["expert_oracles", "wa_uncalibrated", "best_ta_uncalibrated", "hns", "u_ipr", "s_ipr"]
    method_labels = ["Oracles", "WA Uncal.", "Tuned TA", "HNS (Task-Spec.)", "U-IPR", "S-IPR"]
    
    resnet_vals = []
    mlp_vals = []
    
    for m in methods:
        if m == "expert_oracles":
            resnet_vals.append(sum(results["resnet"]["expert_oracles"].values()) / 3)
            mlp_vals.append(sum(results["mlp"]["expert_oracles"].values()) / 3)
        elif m == "best_ta_uncalibrated":
            resnet_vals.append(results["resnet"]["best_ta_uncalibrated"]["avg_acc"])
            mlp_vals.append(results["mlp"]["best_ta_uncalibrated"]["avg_acc"])
        else:
            resnet_vals.append(sum(results["resnet"][m].values()) / 3)
            mlp_vals.append(sum(results["mlp"][m].values()) / 3)
            
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, resnet_vals, width, label='ResNet-18 (with BatchNorm)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, mlp_vals, width, label='MLP (no BatchNorm)', color='#ff7f0e')
    
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Model Merging Performance: ResNet-18 vs MLP')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Annotate bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig("plots/comparison_resnet_vs_mlp.png", dpi=300)
    plt.close()
    
    # Plot 2: Pareto Frontier of DE-BN Calibration Samples vs Accuracy
    if "de_bn_wa" in results["resnet"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        N_samples = [8, 16, 32, 64, 128]
        
        wa_de_bn_accs = [sum(results["resnet"]["de_bn_wa"][N].values()) / 3 for N in N_samples]
        ta_de_bn_accs = [sum(results["resnet"]["de_bn_ta"][N].values()) / 3 for N in N_samples]
        
        # Horizontal lines for comparison
        u_ipr_resnet = sum(results["resnet"]["u_ipr"].values()) / 3
        hns_resnet = sum(results["resnet"]["hns"].values()) / 3
        best_ta_resnet = results["resnet"]["best_ta_uncalibrated"]["avg_acc"]
        wa_resnet = sum(results["resnet"]["wa_uncalibrated"].values()) / 3
        
        ax.plot(N_samples, wa_de_bn_accs, marker='o', label='DE-BN on top of WA', color='#2ca02c', linewidth=2)
        ax.plot(N_samples, ta_de_bn_accs, marker='s', label='DE-BN on top of Best TA', color='#d62728', linewidth=2)
        
        ax.axhline(y=u_ipr_resnet, color='blue', linestyle='--', label='U-IPR (Data-Free)')
        ax.axhline(y=hns_resnet, color='purple', linestyle='-.', label='HNS (Data-Free)')
        ax.axhline(y=best_ta_resnet, color='gray', linestyle=':', label='Tuned TA (Data-Free)')
        ax.axhline(y=wa_resnet, color='black', linestyle='-', alpha=0.5, label='WA (Uncalibrated)')
        
        ax.set_xlabel('Number of Calibration Samples (N)')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title('Pareto Frontier: Data-Efficient BN Calibration vs Data-Free Merging')
        ax.set_xticks(N_samples)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right')
        
        fig.tight_layout()
        plt.savefig("plots/pareto_frontier_calibration.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    run_all_experiments()
