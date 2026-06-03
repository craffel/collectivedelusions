import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy
import numpy as np

# Set seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Prepare Datasets & DataLoaders
print("Preparing datasets...")
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

train_datasets = {
    "mnist": datasets.MNIST(root="./data", train=True, transform=mnist_transform, download=False),
    "fmnist": datasets.FashionMNIST(root="./data", train=True, transform=fmnist_transform, download=False),
    "cifar10": datasets.CIFAR10(root="./data", train=True, transform=cifar_transform, download=False)
}

test_datasets = {
    "mnist": datasets.MNIST(root="./data", train=False, transform=mnist_transform, download=False),
    "fmnist": datasets.FashionMNIST(root="./data", train=False, transform=fmnist_transform, download=False),
    "cifar10": datasets.CIFAR10(root="./data", train=False, transform=cifar_transform, download=False)
}

train_loaders = {k: DataLoader(v, batch_size=128, shuffle=True, num_workers=4) for k, v in train_datasets.items()}
test_loaders = {k: DataLoader(v, batch_size=128, shuffle=False, num_workers=4) for k, v in test_datasets.items()}


# 2. Define Architecture
def get_resnet18_with_head():
    # Load ImageNet pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # We will wrap the backbone and separate head
    backbone = nn.Sequential(*list(model.children())[:-1]) # outputs [B, 512, 1, 1]
    head = nn.Linear(512, 10)
    # Re-initialize head to be clean
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return backbone, head

# Training function for experts
def train_expert(name, train_loader, epochs=5):
    print(f"\n--- Training Expert for {name.upper()} ---")
    backbone, head = get_resnet18_with_head()
    backbone.to(device)
    head.to(device)
    
    optimizer = optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        backbone.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feats = backbone(x).squeeze(-1).squeeze(-1)
            outputs = head(feats)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {acc:.2f}%")
        
    return backbone.cpu(), head.cpu()


# 3. Save / Load Expert Models
backbones = {}
heads = {}

os.makedirs("checkpoints", exist_ok=True)
for task in ["mnist", "fmnist", "cifar10"]:
    bb_path = f"checkpoints/{task}_backbone.pth"
    hd_path = f"checkpoints/{task}_head.pth"
    if os.path.exists(bb_path) and os.path.exists(hd_path):
        print(f"Loading pretrained expert for {task}...")
        backbone, head = get_resnet18_with_head()
        backbone.load_state_dict(torch.load(bb_path))
        head.load_state_dict(torch.load(hd_path))
        backbones[task] = backbone
        heads[task] = head
    else:
        bb, hd = train_expert(task, train_loaders[task], epochs=5)
        torch.save(bb.state_dict(), bb_path)
        torch.save(hd.state_dict(), hd_path)
        backbones[task] = bb
        heads[task] = hd

# Progenitor Model (Unmodified ImageNet model)
progenitor_backbone, _ = get_resnet18_with_head()
# Keep progenitor_backbone on CPU for reference


# 4. Evaluation Function
def evaluate_model(backbone, task, loader):
    backbone.eval()
    backbone.to(device)
    head = heads[task].to(device)
    head.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = backbone(x).squeeze(-1).squeeze(-1)
            outputs = head(feats)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    acc = 100.0 * correct / total
    backbone.cpu()
    head.cpu()
    return acc


# Individual Experts Test Accuracy Check
print("\n--- Evaluating Individual Experts on their own tasks (Upper Bounds) ---")
for task in ["mnist", "fmnist", "cifar10"]:
    acc = evaluate_model(backbones[task], task, test_loaders[task])
    print(f"Expert {task.upper()} Accuracy: {acc:.2f}%")


# 5. Merging Paradigms & Algorithms

# Helper to copy state dict
def copy_backbone(src):
    dst = copy.deepcopy(progenitor_backbone)
    dst.load_state_dict(src.state_dict())
    return dst

# WEIGHT AVERAGING (WA)
def merge_wa(backbones):
    print("\n[WA] Merging backbones with standard Weight Averaging...")
    merged = copy_backbone(progenitor_backbone)
    merged_sd = merged.state_dict()
    
    keys = list(merged_sd.keys())
    for key in keys:
        if "num_batches_tracked" in key:
            continue
        tensors = [backbones[task].state_dict()[key].float() for task in backbones]
        merged_sd[key] = torch.stack(tensors, dim=0).mean(dim=0).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged

# TASK ARITHMETIC (TA)
def merge_ta(backbones, progenitor, lam=1.0):
    print(f"\n[TA] Merging backbones with Task Arithmetic (lam={lam})...")
    merged = copy_backbone(progenitor_backbone)
    merged_sd = merged.state_dict()
    prog_sd = progenitor.state_dict()
    
    keys = list(merged_sd.keys())
    for key in keys:
        if "num_batches_tracked" in key:
            continue
        updates = []
        for task in backbones:
            update = backbones[task].state_dict()[key].float() - prog_sd[key].float()
            updates.append(update)
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        merged_sd[key] = (prog_sd[key].float() + lam * merged_update).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged

# HOLOGRAPHIC NORM SCALING (HNS)
def evaluate_hns(backbones, progenitor):
    print("\n[HNS] Evaluating Holographic Norm Scaling...")
    # Since HNS is task-specific, we create a custom merged backbone for each task
    # and then evaluate on that task.
    results = {}
    prog_sd = progenitor.state_dict()
    
    # Pre-calculate merged updates (as in WA)
    merged_updates = {}
    keys = list(prog_sd.keys())
    for key in keys:
        if "num_batches_tracked" in key:
            continue
        updates = []
        for task in backbones:
            update = backbones[task].state_dict()[key].float() - prog_sd[key].float()
            updates.append(update)
        merged_updates[key] = torch.stack(updates, dim=0).mean(dim=0)
        
    # Evaluate for each task
    for target_task in backbones:
        task_backbone = copy_backbone(progenitor_backbone)
        task_sd = task_backbone.state_dict()
        
        for key in keys:
            if "num_batches_tracked" in key:
                continue
            
            delta_merged = merged_updates[key]
            delta_expert = backbones[target_task].state_dict()[key].float() - prog_sd[key].float()
            
            if delta_merged.dim() >= 2:
                # Channel-wise scaling along dim=0 (output channels)
                gamma = []
                for c in range(delta_merged.size(0)):
                    norm_expert = torch.norm(delta_expert[c])
                    norm_merged = torch.norm(delta_merged[c])
                    scale = norm_expert / (norm_merged + 1e-8)
                    gamma.append(scale)
                gamma = torch.tensor(gamma).view(-1, *([1] * (delta_merged.dim() - 1)))
                delta_corrected = gamma * delta_merged
            else:
                # Scalar scaling for 1D parameters
                norm_expert = torch.norm(delta_expert)
                norm_merged = torch.norm(delta_merged)
                scale = norm_expert / (norm_merged + 1e-8)
                delta_corrected = scale * delta_merged
                
            task_sd[key] = (prog_sd[key].float() + delta_corrected).to(task_sd[key].dtype)
            
        task_backbone.load_state_dict(task_sd)
        results[target_task] = evaluate_model(task_backbone, target_task, test_loaders[target_task])
        
    return results

# UPDATE-LEVEL ISOTROPIC PARAMETER RESONANCE (U-IPR)
def merge_u_ipr(backbones, progenitor):
    print("\n[U-IPR] Merging backbones with Update-level Isotropic Parameter Resonance...")
    merged = copy_backbone(progenitor_backbone)
    merged_sd = merged.state_dict()
    prog_sd = progenitor.state_dict()
    
    keys = list(prog_sd.keys())
    for key in keys:
        if "num_batches_tracked" in key:
            continue
        
        updates = []
        for task in backbones:
            update = backbones[task].state_dict()[key].float() - prog_sd[key].float()
            updates.append(update)
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        # U-IPR calculations
        norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
        norm_merged = torch.norm(merged_update)
        
        scale = norm_experts / (norm_merged + 1e-8)
        # Clamping as specified in Algorithm 1
        scale = torch.clamp(scale, min=0.1, max=10.0)
        
        corrected_update = scale * merged_update
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged

# SPECTRAL PARAMETER RESONANCE (S-IPR)
def merge_s_ipr(backbones, progenitor):
    print("\n[S-IPR] Merging backbones with Spectral Parameter Resonance...")
    merged = copy_backbone(progenitor_backbone)
    merged_sd = merged.state_dict()
    prog_sd = progenitor.state_dict()
    
    keys = list(prog_sd.keys())
    for key in keys:
        if "num_batches_tracked" in key:
            continue
        
        updates = []
        for task in backbones:
            update = backbones[task].state_dict()[key].float() - prog_sd[key].float()
            updates.append(update)
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        # If 2D weight matrix (or 4D convolutional weight flattened to 2D)
        if merged_update.dim() >= 2:
            orig_shape = merged_update.shape
            
            # Reshape 4D convolutions to 2D: [out_channels, in_channels * kh * kw]
            if merged_update.dim() == 4:
                flat_shape = (orig_shape[0], -1)
                updates_2d = [u.view(flat_shape) for u in updates]
                merged_2d = merged_update.view(flat_shape)
            else:
                updates_2d = updates
                merged_2d = merged_update
                
            try:
                # SVD of experts
                expert_singular_values = []
                for u_2d in updates_2d:
                    _, S_exp, _ = torch.linalg.svd(u_2d, full_matrices=False)
                    expert_singular_values.append(S_exp)
                
                # Average expert spectrum
                avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                
                # SVD of merged
                U_m, S_m, Vh_m = torch.linalg.svd(merged_2d, full_matrices=False)
                
                # Reconstruct with average spectrum
                corrected_2d = U_m @ torch.diag(avg_spectrum) @ Vh_m
                corrected_update = corrected_2d.view(orig_shape)
                
            except Exception as e:
                # SVD can sometimes fail to converge, fallback to U-IPR
                print(f"SVD failed for {key} ({e}), falling back to U-IPR.")
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            # 1D parameters, fallback to U-IPR
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged

# HIGHER-ORDER TENSOR PARAMETER RESONANCE (HOT-PR) - OUR PROPOSED METHOD
def hosvd(T):
    # T is a 4D tensor of shape [d1, d2, d3, d4]
    # SVD on mode 1 unfolding
    T1 = T.permute(0, 1, 2, 3).reshape(T.shape[0], -1)
    U1, _, _ = torch.linalg.svd(T1, full_matrices=False)
    
    # SVD on mode 2 unfolding
    T2 = T.permute(1, 0, 2, 3).reshape(T.shape[1], -1)
    U2, _, _ = torch.linalg.svd(T2, full_matrices=False)
    
    # SVD on mode 3 unfolding
    T3 = T.permute(2, 0, 1, 3).reshape(T.shape[2], -1)
    U3, _, _ = torch.linalg.svd(T3, full_matrices=False)
    
    # SVD on mode 4 unfolding
    T4 = T.permute(3, 0, 1, 2).reshape(T.shape[3], -1)
    U4, _, _ = torch.linalg.svd(T4, full_matrices=False)
    
    # Project to get the core tensor G
    G = torch.einsum('abcd,ai,bj,ck,dl->ijkl', T, U1, U2, U3, U4)
    return G, U1, U2, U3, U4

def reconstruct_hosvd(G, U1, U2, U3, U4):
    return torch.einsum('ijkl,ai,bj,ck,dl->abcd', G, U1, U2, U3, U4)

def merge_hot_pr(backbones, progenitor):
    print("\n[HOT-PR] Merging backbones with proposed Higher-Order Tensor Parameter Resonance...")
    merged = copy_backbone(progenitor_backbone)
    merged_sd = merged.state_dict()
    prog_sd = progenitor.state_dict()
    
    keys = list(prog_sd.keys())
    for key in keys:
        if "num_batches_tracked" in key:
            continue
            
        updates = []
        for task in backbones:
            update = backbones[task].state_dict()[key].float() - prog_sd[key].float()
            updates.append(update)
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        # Apply HOT-PR to 4D tensors (convolutional layers)
        if merged_update.dim() == 4:
            try:
                # Compute HOSVD for each expert
                expert_cores = []
                for u_tensor in updates:
                    G_exp, _, _, _, _ = hosvd(u_tensor)
                    expert_cores.append(G_exp)
                    
                # Average core tensor magnitude
                avg_core_mag = torch.stack([G.abs() for G in expert_cores], dim=0).mean(dim=0)
                
                # Compute HOSVD for merged update
                G_m, U1_m, U2_m, U3_m, U4_m = hosvd(merged_update)
                
                # Construct corrected core tensor
                G_corrected = torch.sign(G_m) * avg_core_mag
                
                # Reconstruct corrected tensor
                corrected_update = reconstruct_hosvd(G_corrected, U1_m, U2_m, U3_m, U4_m)
                
            except Exception as e:
                print(f"HOSVD failed for {key} ({e}), falling back to S-IPR.")
                # Fallback to S-IPR / SVD
                orig_shape = merged_update.shape
                flat_shape = (orig_shape[0], -1)
                updates_2d = [u.view(flat_shape) for u in updates]
                merged_2d = merged_update.view(flat_shape)
                try:
                    expert_singular_values = []
                    for u_2d in updates_2d:
                        _, S_exp, _ = torch.linalg.svd(u_2d, full_matrices=False)
                        expert_singular_values.append(S_exp)
                    avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                    U_m, S_m, Vh_m = torch.linalg.svd(merged_2d, full_matrices=False)
                    corrected_2d = U_m @ torch.diag(avg_spectrum) @ Vh_m
                    corrected_update = corrected_2d.view(orig_shape)
                except Exception as e2:
                    norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                    norm_merged = torch.norm(merged_update)
                    scale = norm_experts / (norm_merged + 1e-8)
                    scale = torch.clamp(scale, min=0.1, max=10.0)
                    corrected_update = scale * merged_update
                    
        # Apply S-IPR to 2D tensors (linear layers)
        elif merged_update.dim() == 2:
            try:
                expert_singular_values = []
                for u in updates:
                    _, S_exp, _ = torch.linalg.svd(u, full_matrices=False)
                    expert_singular_values.append(S_exp)
                avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                U_m, S_m, Vh_m = torch.linalg.svd(merged_update, full_matrices=False)
                corrected_update = U_m @ torch.diag(avg_spectrum) @ Vh_m
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
                
        else:
            # 1D tensors (biases, BatchNorm running statistics) fallback to U-IPR
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged


# 6. Run Evaluations
results = {}

# Evaluate WA
wa_model = merge_wa(backbones)
results["WA"] = {task: evaluate_model(wa_model, task, test_loaders[task]) for task in backbones}

# Evaluate TA (lam=1.0)
ta_model_1 = merge_ta(backbones, progenitor_backbone, lam=1.0)
results["TA (lam=1.0)"] = {task: evaluate_model(ta_model_1, task, test_loaders[task]) for task in backbones}

# Evaluate TA (lam=0.5)
ta_model_05 = merge_ta(backbones, progenitor_backbone, lam=0.5)
results["TA (lam=0.5)"] = {task: evaluate_model(ta_model_05, task, test_loaders[task]) for task in backbones}

# Evaluate HNS
results["HNS"] = evaluate_hns(backbones, progenitor_backbone)

# Evaluate U-IPR
u_ipr_model = merge_u_ipr(backbones, progenitor_backbone)
results["U-IPR"] = {task: evaluate_model(u_ipr_model, task, test_loaders[task]) for task in backbones}

# Evaluate S-IPR
s_ipr_model = merge_s_ipr(backbones, progenitor_backbone)
results["S-IPR"] = {task: evaluate_model(s_ipr_model, task, test_loaders[task]) for task in backbones}

# Evaluate HOT-PR (Ours)
hot_pr_model = merge_hot_pr(backbones, progenitor_backbone)
results["HOT-PR (Ours)"] = {task: evaluate_model(hot_pr_model, task, test_loaders[task]) for task in backbones}


# 7. Print Results Table
print("\n" + "="*80)
print(f"{'Method':<20} | {'MNIST':<10} | {'FMNIST':<10} | {'CIFAR-10':<10} | {'Average':<10}")
print("="*80)
for method, task_accs in results.items():
    mnist_acc = task_accs["mnist"]
    fmnist_acc = task_accs["fmnist"]
    cifar_acc = task_accs["cifar10"]
    avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3.0
    print(f"{method:<20} | {mnist_acc:<10.2f}% | {fmnist_acc:<10.2f}% | {cifar_acc:<10.2f}% | {avg_acc:<10.2f}%")
print("="*80)

# 8. Save results to a report file
with open("experimental_results.txt", "w") as f:
    f.write("Multi-Task Model Merging Experimental Results\n")
    f.write("="*80 + "\n")
    f.write(f"{'Method':<20} | {'MNIST':<10} | {'FMNIST':<10} | {'CIFAR-10':<10} | {'Average':<10}\n")
    f.write("="*80 + "\n")
    for method, task_accs in results.items():
        mnist_acc = task_accs["mnist"]
        fmnist_acc = task_accs["fmnist"]
        cifar_acc = task_accs["cifar10"]
        avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3.0
        f.write(f"{method:<20} | {mnist_acc:<10.2f}% | {fmnist_acc:<10.2f}% | {cifar_acc:<10.2f}% | {avg_acc:<10.2f}%\n")
    f.write("="*80 + "\n")
print("Results saved to experimental_results.txt!")
