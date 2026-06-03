import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import copy

# Helper to flatten and unflatten layers for SVD
def flatten_weight(w):
    # Keep the first dimension (out_features/out_channels) and flatten the rest
    return w.view(w.shape[0], -1), w.shape

def unflatten_weight(w_flat, original_shape):
    return w_flat.view(original_shape)

# SVD and Procrustes analysis for orthogonal component extraction
def solve_procrustes(W_target, W_0):
    # W_target: [d_out, d_in], W_0: [d_out, d_in]
    # SVD of W_target @ W_0.T
    # M = W_target @ W_0.T of size [d_out, d_out]
    M = torch.matmul(W_target, W_0.t())
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = torch.matmul(U, Vh)  # R = U @ V^T
    return R

# Inverse Cayley Transform: R -> Q (skew-symmetric)
def inv_cayley(R, eps=1e-6):
    d = R.shape[0]
    I = torch.eye(d, device=R.device)
    # Q = (R - I) @ inv(R + I)
    # Add small diagonal perturbation for numerical stability
    R_plus_I = R + I + eps * I
    Q = torch.matmul(R - I, torch.linalg.inv(R_plus_I))
    # Enforce skew-symmetry: Q = 0.5 * (Q - Q.T)
    Q = 0.5 * (Q - Q.t())
    return Q

# Cayley Transform: Q -> R (orthogonal)
def cayley(Q):
    d = Q.shape[0]
    I = torch.eye(d, device=Q.device)
    # R = (I + Q) @ inv(I - Q)
    R = torch.matmul(I + Q, torch.linalg.inv(I - Q))
    return R

# Get dataloader for evaluation or calibration
def get_eval_dataloader(task, train=False, batch_size=128, image_size=32):
    if task == 'cifar10':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)
    elif task == 'svhn':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        split = "train" if train else "test"
        dataset = torchvision.datasets.SVHN(root="./data", split=split, download=False, transform=transform)
    elif task == 'fmnist':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
        ])
        dataset = torchvision.datasets.FashionMNIST(root="./data", train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader

# Instantiate a fresh model with a given task head
def build_expert_model(arch, backbone_sd, head_sd, head_name):
    if arch == 'resnet18':
        model = torchvision.models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10)
    elif arch == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, 10)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    # Combine backbone and head
    full_sd = {**backbone_sd, **head_sd}
    model.load_state_dict(full_sd)
    return model

# Basic evaluation function
def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total

# --- Merging Algorithms ---

# 1. Arithmetic Merging
def merge_arithmetic(base_sd, expert_sds, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        # Compute average task vector
        task_vectors = [expert_sds[i][k] - base_sd[k] for i in range(N)]
        avg_task_vector = torch.stack(task_vectors).mean(dim=0)
        merged_sd[k] = base_sd[k] + alpha * avg_task_vector
    return merged_sd

# 2. TIES Merging
def merge_ties(base_sd, expert_sds, fraction=0.2, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        
        # If 1D parameter, use simple average
        if base_sd[k].dim() < 2:
            task_vectors = [expert_sds[i][k] - base_sd[k] for i in range(N)]
            avg_task_vector = torch.stack(task_vectors).mean(dim=0)
            merged_sd[k] = base_sd[k] + alpha * avg_task_vector
            continue
            
        # Task vectors
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]
        
        # 1. Trim (Sparsify)
        trimmed_tvs = []
        for tv in tvs:
            if tv.numel() == 0 or tv.dim() == 0:
                trimmed_tvs.append(tv)
                continue
            flat_tv = tv.view(-1)
            k_val = max(1, int(fraction * flat_tv.numel()))
            threshold = torch.topk(flat_tv.abs(), k_val).values[-1]
            mask = flat_tv.abs() >= threshold
            trimmed_flat = flat_tv * mask
            trimmed_tvs.append(trimmed_flat.view(tv.shape))
            
        # 2. Elect Sign
        stacked = torch.stack(trimmed_tvs)
        sign_sum = stacked.sign().sum(dim=0)
        elected_sign = sign_sum.sign()
        
        # 3. Disjoint Merge
        filtered_tvs = []
        for tv in trimmed_tvs:
            # Keep values where sign matches the elected sign
            mask = (tv.sign() == elected_sign) & (elected_sign != 0)
            filtered_tvs.append(tv * mask)
            
        # Average the matching signs
        # Count non-zero elements
        counts = torch.stack([(tv != 0).float() for tv in filtered_tvs]).sum(dim=0)
        summed = torch.stack(filtered_tvs).sum(dim=0)
        avg_tv = torch.where(counts > 0, summed / counts, torch.zeros_like(summed))
        
        merged_sd[k] = base_sd[k] + alpha * avg_tv
    return merged_sd

# 3. DARE Merging
def merge_dare(base_sd, expert_sds, drop_rate=0.9, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
            
        # If 1D parameter, use simple average
        if base_sd[k].dim() < 2:
            task_vectors = [expert_sds[i][k] - base_sd[k] for i in range(N)]
            avg_task_vector = torch.stack(task_vectors).mean(dim=0)
            merged_sd[k] = base_sd[k] + alpha * avg_task_vector
            continue
            
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]
        
        scaled_tvs = []
        for tv in tvs:
            if tv.numel() == 0 or tv.dim() == 0:
                scaled_tvs.append(tv)
                continue
            # Drop randomly and scale
            mask = (torch.rand_like(tv.float()) >= drop_rate).to(tv.device)
            scaled_tv = (tv * mask.float()) / (1.0 - drop_rate)
            scaled_tvs.append(scaled_tv)
            
        avg_tv = torch.stack(scaled_tvs).mean(dim=0)
        merged_sd[k] = base_sd[k] + alpha * avg_tv
    return merged_sd

# 4. OrthoMerge
def merge_orthomerge(base_sd, expert_sds, alpha=0.5, device='cpu'):
    merged_sd = {}
    N = len(expert_sds)
    
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        base_w = base_sd[k].to(device)
        expert_ws = [expert_sds[i][k].to(device) for i in range(N)]
        
        # We decouple parameters that are 2D or can be reshaped to 2D
        if base_w.dim() >= 2:
            original_shape = base_w.shape
            base_flat, _ = flatten_weight(base_w)
            
            R_list = []
            rho_list = []
            
            for i in range(N):
                expert_flat, _ = flatten_weight(expert_ws[i])
                # 1. Procrustes extraction
                R_i = solve_procrustes(expert_flat, base_flat)
                R_list.append(R_i)
                # 2. Residual
                rho_i = expert_flat - torch.matmul(R_i, base_flat)
                rho_list.append(rho_i)
                
            # 3. Magnitude-corrected Orthogonal Merging
            Q_list = []
            for R_i in R_list:
                Q_i = inv_cayley(R_i)
                Q_list.append(Q_i)
                
            Q_sum = torch.stack(Q_list).sum(dim=0)
            Q_sum_norm = torch.linalg.norm(Q_sum, 'fro')
            
            if Q_sum_norm > 1e-8:
                sum_norms = sum(torch.linalg.norm(Q_i, 'fro') for Q_i in Q_list)
                c = sum_norms / Q_sum_norm
                Q_merged = (c / N) * Q_sum
            else:
                Q_merged = torch.zeros_like(Q_sum)
                
            R_merged = cayley(Q_merged)
            
            # 4. Merge residuals
            rho_merged = torch.stack(rho_list).mean(dim=0)
            
            # 5. Hybrid Merging
            W_final_flat = torch.matmul(R_merged, base_flat) + alpha * rho_merged
            merged_sd[k] = unflatten_weight(W_final_flat, original_shape).cpu()
        else:
            # 1D parameters are scaled consistently by alpha
            tvs = [expert_ws[i] - base_w for i in range(N)]
            avg_tv = torch.stack(tvs).mean(dim=0)
            merged_sd[k] = (base_w + alpha * avg_tv).cpu()
            
    return merged_sd

# 5. SAIM (Isotropic Merging)
def merge_saim(base_sd, expert_sds, alpha=0.5, gamma=0.5, device='cpu'):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        base_w = base_sd[k].to(device)
        expert_ws = [expert_sds[i][k].to(device) for i in range(N)]
        
        if base_w.dim() >= 2:
            original_shape = base_w.shape
            base_flat, _ = flatten_weight(base_w)
            
            # Average task vector
            tvs = [flatten_weight(expert_ws[i])[0] - base_flat for i in range(N)]
            tv_com = torch.stack(tvs).mean(dim=0)
            
            # SVD
            U, S, Vh = torch.linalg.svd(tv_com, full_matrices=False)
            
            # Isotropic balancing
            mean_s = S.mean()
            S_balanced = mean_s + (S - mean_s) * gamma
            
            # Reconstruct
            tv_merged = torch.matmul(U, torch.matmul(torch.diag(S_balanced), Vh))
            
            W_final_flat = base_flat + alpha * tv_merged
            merged_sd[k] = unflatten_weight(W_final_flat, original_shape).cpu()
        else:
            # 1D parameters are scaled consistently by alpha
            tvs = [expert_ws[i] - base_w for i in range(N)]
            avg_tv = torch.stack(tvs).mean(dim=0)
            merged_sd[k] = (base_w + alpha * avg_tv).cpu()
            
    return merged_sd

# 6. DOR-SAIM (Our Proposed Method)
def merge_dor_saim(base_sd, expert_sds, alpha=0.5, gamma=0.5, device='cpu'):
    # Decoupled Orthogonal-Residual Sharpness-Aware Isotropic Merging
    merged_sd = {}
    N = len(expert_sds)
    
    decoupled_info = {} # Keep track of R, base, rho for post-hoc scaling
    
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        base_w = base_sd[k].to(device)
        expert_ws = [expert_sds[i][k].to(device) for i in range(N)]
        
        if base_w.dim() >= 2:
            original_shape = base_w.shape
            base_flat, _ = flatten_weight(base_w)
            
            R_list = []
            rho_list = []
            
            for i in range(N):
                expert_flat, _ = flatten_weight(expert_ws[i])
                R_i = solve_procrustes(expert_flat, base_flat)
                R_list.append(R_i)
                rho_i = expert_flat - torch.matmul(R_i, base_flat)
                rho_list.append(rho_i)
                
            # Merge Orthogonal Components
            Q_list = []
            for R_i in R_list:
                Q_i = inv_cayley(R_i)
                Q_list.append(Q_i)
                
            Q_sum = torch.stack(Q_list).sum(dim=0)
            Q_sum_norm = torch.linalg.norm(Q_sum, 'fro')
            
            if Q_sum_norm > 1e-8:
                sum_norms = sum(torch.linalg.norm(Q_i, 'fro') for Q_i in Q_list)
                c = sum_norms / Q_sum_norm
                Q_merged = (c / N) * Q_sum
            else:
                Q_merged = torch.zeros_like(Q_sum)
                
            R_merged = cayley(Q_merged)
            
            # Merge Residual Components using Isotropic Merging
            rho_com = torch.stack(rho_list).mean(dim=0)
            U, S, Vh = torch.linalg.svd(rho_com, full_matrices=False)
            mean_s = S.mean()
            S_balanced = mean_s + (S - mean_s) * gamma
            rho_merged = torch.matmul(U, torch.matmul(torch.diag(S_balanced), Vh))
            
            # Save decoupled components for post-hoc optimization
            decoupled_info[k] = {
                'original_shape': original_shape,
                'R_merged': R_merged.cpu(),
                'base_flat': base_flat.cpu(),
                'rho_merged': rho_merged.cpu(),
                'alpha': alpha
            }
            
            W_final_flat = torch.matmul(R_merged, base_flat) + alpha * rho_merged
            merged_sd[k] = unflatten_weight(W_final_flat, original_shape).cpu()
        else:
            # 1D parameters are scaled consistently by alpha
            tvs = [expert_ws[i] - base_w for i in range(N)]
            avg_tv = torch.stack(tvs).mean(dim=0)
            merged_sd[k] = (base_w + alpha * avg_tv).cpu()
            
    return merged_sd, decoupled_info


# --- Post-Hoc Sharpness-Aware Calibration ---
class LayerwiseScaleModule(nn.Module):
    def __init__(self, decoupled_info, merged_sd, head_sds, tasks, arch, head_name, device):
        super().__init__()
        self.decoupled_info = decoupled_info
        self.merged_sd = merged_sd
        self.head_sds = head_sds
        self.tasks = tasks
        self.arch = arch
        self.head_name = head_name
        self.device = device
        
        # We define a scaling factor parameter for each merged layer
        self.keys = list(decoupled_info.keys())
        # Scaling factors initialized to 1.0
        self.scales = nn.ParameterDict({
            k.replace('.', '_'): nn.Parameter(torch.tensor(1.0, device=device)) for k in self.keys
        })
        
        # Pre-load static/non-merged parts of state dict
        self.static_sd = {}
        for k, v in merged_sd.items():
            if k not in decoupled_info:
                # Use merged static layers!
                self.static_sd[k] = v.to(device)
                
        # Pre-load decoupled info tensors to device and pre-compute static matmul
        self.device_decoupled_info = {}
        for k, info in decoupled_info.items():
            R_merged = info['R_merged'].to(device)
            base_flat = info['base_flat'].to(device)
            R_base_flat = torch.matmul(R_merged, base_flat)
            self.device_decoupled_info[k] = {
                'original_shape': info['original_shape'],
                'R_base_flat': R_base_flat,
                'rho_merged': info['rho_merged'].to(device),
                'alpha': info['alpha']
            }
                
        # Instantiate base models for each task once
        self.base_models = {}
        for task in tasks:
            head_sd = {k: v.to(device) for k, v in head_sds[task].items()}
            model = build_expert_model(arch, merged_sd, head_sd, head_name)
            model = model.to(device)
            model.eval()
            self.base_models[task] = model
                
    def get_merged_backbone_sd(self):
        backbone_sd = {}
        # Static layers
        for k, v in self.static_sd.items():
            backbone_sd[k] = v
            
        # Merged/scaled layers
        for k in self.keys:
            info = self.device_decoupled_info[k]
            R_base_flat = info['R_base_flat']
            rho_merged = info['rho_merged']
            alpha = info['alpha']
            
            # Apply optimized scaling factor
            scale_name = k.replace('.', '_')
            s = self.scales[scale_name]
            
            W_final_flat = R_base_flat + s * alpha * rho_merged
            backbone_sd[k] = unflatten_weight(W_final_flat, info['original_shape'])
            
        return backbone_sd

    def forward(self, inputs, task_idx):
        from torch.func import functional_call
        backbone_sd = self.get_merged_backbone_sd()
        task = self.tasks[task_idx]
        model = self.base_models[task]
        return functional_call(model, backbone_sd, inputs)


def post_hoc_sharpness_aware_optimization(decoupled_info, merged_sd, head_sds, tasks, arch, head_name, device, steps=10, lr=0.01, rho_sam=0.05):
    print("Initializing Post-Hoc Sharpness-Aware Calibration (DOR-SAIM)...")
    
    img_size = 224 if arch == 'vit_b_16' else 32
    # 1. Create a tiny combined calibration dataset
    calib_loader_dict = {}
    for task in tasks:
        # Load train set to use as calibration
        loader = get_eval_dataloader(task, train=True, batch_size=32, image_size=img_size)
        calib_loader_dict[task] = loader
        
    # Get a batch of calibration data for each task
    calib_batches = []
    for t_idx, task in enumerate(tasks):
        for inputs, labels in calib_loader_dict[task]:
            calib_batches.append((inputs, labels, t_idx))
            break  # Just take 1 batch of size 32 per task
            
    # Combine calibration batches (96 images total for 3 tasks)
    print(f"Calibration dataset size: {len(calib_batches) * 32} images across {len(tasks)} tasks.")
    
    # Instantiate the wrapper module
    module = LayerwiseScaleModule(decoupled_info, merged_sd, head_sds, tasks, arch, head_name, device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Sharpness-Aware Optimization Loop (SAM-style updates)
    for step in range(1, steps + 1):
        # 1. Compute normal gradient
        optimizer.zero_grad()
        loss_val = 0.0
        
        # Accumulate loss over all calibration batches
        for inputs, labels, t_idx in calib_batches:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = module(inputs, t_idx)
            loss = criterion(outputs, labels)
            loss_val += loss
            
        loss_val = loss_val / len(calib_batches)
        loss_val.backward()
        
        # Save current scales and gradients for the perturbation step
        scales_grad = {}
        for name, param in module.scales.items():
            if param.grad is not None:
                scales_grad[name] = param.grad.clone()
                
        # 2. Perturbation Step (Worst-case direction search)
        # Compute L2 norm of the scales' gradients
        if len(scales_grad) > 0:
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in scales_grad.values()))
        else:
            grad_norm = torch.tensor(0.0, device=device)
        
        if grad_norm > 1e-8:
            # Apply worst-case perturbation: s_perturbed = s + rho * grad / norm(grad)
            with torch.no_grad():
                for name, param in module.scales.items():
                    if name in scales_grad:
                        perturbation = rho_sam * scales_grad[name] / grad_norm
                        param.add_(perturbation)
                        
            # Compute gradient at perturbed point
            optimizer.zero_grad()
            perturbed_loss = 0.0
            for inputs, labels, t_idx in calib_batches:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = module(inputs, t_idx)
                loss = criterion(outputs, labels)
                perturbed_loss += loss
                
            perturbed_loss = perturbed_loss / len(calib_batches)
            perturbed_loss.backward()
            
            # Restore original scales (remove perturbation) before optimizer step
            with torch.no_grad():
                for name, param in module.scales.items():
                    if name in scales_grad:
                        perturbation = rho_sam * scales_grad[name] / grad_norm
                        param.sub_(perturbation)
                        
        # 3. Optimizer Step (update scales using gradients from the perturbed point)
        optimizer.step()
        print(f"Calibration Step {step}/{steps} - Base Loss: {loss_val.item():.4f}")
        
    # Get the final calibrated backbone state dict
    with torch.no_grad():
        calibrated_backbone_sd = module.get_merged_backbone_sd()
        
    # Clean up GPU memory
    del module
    torch.cuda.empty_cache()
    
    return {k: v.cpu() for k, v in calibrated_backbone_sd.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vit_b_16'])
    parser.add_argument('--alpha', type=float, default=0.5, help='Merging coefficient/scale')
    parser.add_argument('--gamma', type=float, default=0.5, help='Isotropic balancing coefficient')
    parser.add_argument('--method', type=str, default='dor_saim', 
                        choices=['arithmetic', 'ties', 'dare', 'orthomerge', 'saim', 'dor_saim'])
    parser.add_argument('--ties_fraction', type=float, default=0.2)
    parser.add_argument('--dare_drop_rate', type=float, default=0.9)
    parser.add_argument('--calib_steps', type=int, default=10, help='Steps of post-hoc SAM optimization')
    parser.add_argument('--calib_lr', type=float, default=0.01)
    parser.add_argument('--rho_sam', type=float, default=0.05)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid initialization errors
    print(f"Running evaluation on device: {device}")
    
    # 1. Paths & Heads
    tasks = ['cifar10', 'svhn', 'fmnist']
    head_name = "fc" if args.arch == 'resnet18' else "heads.head"
    
    # 2. Load Base Model and Expert Models
    print("Loading base pre-trained model and expert task models...")
    base_backbone_path = f"checkpoints/{args.arch}_base_backbone.pt"
    if not os.path.exists(base_backbone_path):
        raise FileNotFoundError(f"Base backbone not found at {base_backbone_path}. Train experts first!")
        
    base_backbone_sd = torch.load(base_backbone_path, map_location='cpu')
    
    expert_backbone_sds = []
    expert_head_sds = {}
    
    for task in tasks:
        b_path = f"checkpoints/{args.arch}_{task}_backbone.pt"
        h_path = f"checkpoints/{args.arch}_{task}_head.pt"
        if not os.path.exists(b_path) or not os.path.exists(h_path):
            raise FileNotFoundError(f"Expert weights for {task} not found. Please train all experts first!")
        
        expert_backbone_sds.append(torch.load(b_path, map_location='cpu'))
        expert_head_sds[task] = torch.load(h_path, map_location='cpu')
        
    # 3. Perform Merging
    print(f"Applying merging method: '{args.method.upper()}' (alpha={args.alpha}, gamma={args.gamma})...")
    
    if args.method == 'arithmetic':
        merged_backbone_sd = merge_arithmetic(base_backbone_sd, expert_backbone_sds, alpha=args.alpha)
    elif args.method == 'ties':
        merged_backbone_sd = merge_ties(base_backbone_sd, expert_backbone_sds, fraction=args.ties_fraction, alpha=args.alpha)
    elif args.method == 'dare':
        merged_backbone_sd = merge_dare(base_backbone_sd, expert_backbone_sds, drop_rate=args.dare_drop_rate, alpha=args.alpha)
    elif args.method == 'orthomerge':
        merged_backbone_sd = merge_orthomerge(base_backbone_sd, expert_backbone_sds, alpha=args.alpha, device=device)
    elif args.method == 'saim':
        merged_backbone_sd = merge_saim(base_backbone_sd, expert_backbone_sds, alpha=args.alpha, gamma=args.gamma, device=device)
    elif args.method == 'dor_saim':
        merged_backbone_sd, decoupled_info = merge_dor_saim(base_backbone_sd, expert_backbone_sds, alpha=args.alpha, gamma=args.gamma, device=device)
        # Perform post-hoc sharpness-aware calibration
        merged_backbone_sd = post_hoc_sharpness_aware_optimization(
            decoupled_info, merged_backbone_sd, expert_head_sds, tasks, args.arch, head_name, device,
            steps=args.calib_steps, lr=args.calib_lr, rho_sam=args.rho_sam
        )
        
    # Save merged model backbone
    os.makedirs("merged_models", exist_ok=True)
    merged_path = f"merged_models/{args.arch}_{args.method}_merged.pt"
    torch.save(merged_backbone_sd, merged_path)
    print(f"Saved merged backbone to {merged_path}")
    
    # 4. Evaluate Merged Model on all Tasks
    print("\n--- Evaluating Merged Model ---")
    avg_acc = 0.0
    results = {}
    img_size = 224 if args.arch == 'vit_b_16' else 32
    for task in tasks:
        loader = get_eval_dataloader(task, train=False, batch_size=128, image_size=img_size)
        # Reconstruct expert model with merged backbone and task-specific head
        model = build_expert_model(args.arch, merged_backbone_sd, expert_head_sds[task], head_name)
        model = model.to(device)
        
        loss, acc = evaluate_model(model, loader, device)
        results[task] = acc
        avg_acc += acc
        print(f"Task: {task:8s} | Accuracy: {acc:.2f}% | Loss: {loss:.4f}")
        
    avg_acc = avg_acc / len(tasks)
    print(f"Average Accuracy across tasks: {avg_acc:.2f}%")
    print("--------------------------------")
    
    # Save metrics to a file
    with open(f"merged_models/{args.arch}_{args.method}_results.txt", "w") as f:
        f.write(f"Method: {args.method}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Gamma: {args.gamma}\n")
        for task, acc in results.items():
            f.write(f"{task}: {acc:.2f}%\n")
        f.write(f"Average: {avg_acc:.2f}%\n")

if __name__ == "__main__":
    main()
