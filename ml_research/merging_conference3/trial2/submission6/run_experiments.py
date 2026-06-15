import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import timm
import numpy as np
import copy
from torch.func import functional_call
import json

# Setup directories
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define dataset names
TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

# Helper to load data
def get_dataset(task, train=True, transform=None):
    if task == "MNIST":
        return dset.MNIST(root="./data", train=train, download=True, transform=transform)
    elif task == "FashionMNIST":
        return dset.FashionMNIST(root="./data", train=train, download=True, transform=transform)
    elif task == "CIFAR10":
        return dset.CIFAR10(root="./data", train=train, download=True, transform=transform)
    elif task == "SVHN":
        split = "train" if train else "test"
        return dset.SVHN(root="./data", split=split, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task {task}")

# Image transformation to fit ViT-Tiny
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Gray-to-RGB transform for MNIST/FashionMNIST
class GrayToRGB(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

transform_gray = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create expert classifier module
class ExpertModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features, num_classes)
        # Ensure head weights are initialized nicely
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.backbone.forward_head(features, pre_logits=True)
        return self.head(features)

# Define STE rounding for first-order gradient flow through quantization
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

round_ste = RoundSTE.apply

def quantize_tensor(tensor, bits=8, channel_wise=True):
    if bits is None:
        return tensor
    if not channel_wise or tensor.dim() <= 1:
        # Per-tensor symmetric uniform quantization
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            return tensor
        
        qmin = -(2**(bits - 1))
        qmax = 2**(bits - 1) - 1
        
        scale = max_val / qmax
        
        quantized = round_ste(tensor / scale)
        clipped = torch.clamp(quantized, qmin, qmax)
        dequantized = clipped * scale
        return dequantized
    else:
        # Per-channel symmetric uniform quantization (dim 0 represents output channel/feature)
        orig_shape = tensor.shape
        flat_tensor = tensor.view(orig_shape[0], -1)
        max_vals = torch.max(torch.abs(flat_tensor), dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        max_vals = torch.clamp(max_vals, min=1e-9)
        
        qmin = -(2**(bits - 1))
        qmax = 2**(bits - 1) - 1
        
        scale = max_vals / qmax
        scale_shaped = scale.view(orig_shape[0], *([1] * (len(orig_shape) - 1)))
        
        quantized = round_ste(tensor / scale_shaped)
        clipped = torch.clamp(quantized, qmin, qmax)
        dequantized = clipped * scale_shaped
        return dequantized

# Parameter grouping function
def group_parameters(model):
    # Groups backbone parameters layer-wise (block-wise)
    # L = 14 groups
    # Group 0: patch_embed, pos_embed, cls_token
    # Group 1-12: blocks 0-11
    # Group 13: norm and other global parameters
    # Returns a list of lists of parameter names
    groups = [[] for _ in range(14)]
    for name, param in model.named_parameters():
        if "head" in name:
            continue
        if "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
            groups[0].append(name)
        elif "blocks." in name:
            parts = name.split(".")
            block_idx = int(parts[1])
            groups[block_idx + 1].append(name)
        elif "norm" in name:
            groups[13].append(name)
        else:
            # Fallback to group 13
            groups[13].append(name)
    return groups

# Function to compute Shannon entropy of predictions
def compute_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return torch.mean(entropy)

# Train single task expert
def train_expert(seed, task):
    checkpoint_path = f"checkpoints/seed_{seed}_{task}_expert.pt"
    # Create base model
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model = ExpertModel(backbone).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading cached expert for seed {seed}, task {task}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model

    print(f"Training expert for seed {seed}, task {task}...")
    
    # Get dataset splits
    t_gray = transform_gray if task in ["MNIST", "FashionMNIST"] else transform
    train_dataset = get_dataset(task, train=True, transform=t_gray)
    eval_dataset = get_dataset(task, train=False, transform=t_gray)
    
    # Deterministic subsetting
    rng = np.random.default_rng(seed)
    train_indices = rng.permutation(len(train_dataset))[:512]
    eval_indices = rng.permutation(len(eval_dataset))[:512]
    
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=32, shuffle=True)
    eval_loader = DataLoader(Subset(eval_dataset, eval_indices), batch_size=32, shuffle=False)
    
    # Separate learning rates for backbone and head
    optimizer = optim.Adam([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 1e-3}
    ])
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        # print(f"  Epoch {epoch+1}/5 - Loss: {epoch_loss/512:.4f}")
        
    # Evaluate convergence
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=-1) == y).sum().item()
    acc = correct / 512
    print(f"  Task {task} Expert Accuracy: {acc*100:.2f}%")
    
    # Cache checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    return model

# Evaluate merged model performance on a specific task
def evaluate_merged(merged_backbone_params, head, task, seed, bits=None, quantize_head=False):
    t_gray = transform_gray if task in ["MNIST", "FashionMNIST"] else transform
    eval_dataset = get_dataset(task, train=False, transform=t_gray)
    rng = np.random.default_rng(seed)
    eval_indices = rng.permutation(len(eval_dataset))[:512]
    eval_loader = DataLoader(Subset(eval_dataset, eval_indices), batch_size=32, shuffle=False)
    
    # Prepare quantized parameters
    quant_backbone_params = {}
    for name, param in merged_backbone_params.items():
        if "weight" in name and param.dim() > 1:
            quant_backbone_params[name] = quantize_tensor(param, bits)
        else:
            quant_backbone_params[name] = param
            
    # Combine quantized backbone and head for functional call
    combined_params = {f"backbone.{k}": v for k, v in quant_backbone_params.items()}
    for k, v in head.state_dict().items():
        if quantize_head and "weight" in k and v.dim() > 1:
            combined_params[f"head.{k}"] = quantize_tensor(v, 8)
        else:
            combined_params[f"head.{k}"] = v
        
    # Standard dummy shell to run forward pass functionally
    dummy_backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    dummy_expert = ExpertModel(dummy_backbone).to(device)
    dummy_expert.eval()
    
    correct = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            out = functional_call(dummy_expert, combined_params, x)
            correct += (out.argmax(dim=-1) == y).sum().item()
    return correct / 512

# Multi-task calibration loss (joint prediction entropy)
def compute_calibration_loss(lambdas, base_backbone_params, task_vectors, heads, calib_loaders, param_groups, bits=None):
    # lambdas: tensor of shape (K, L)
    # Reconstruct merged backbone parameters
    merged_backbone = {}
    for name in base_backbone_params.keys():
        merged_backbone[name] = base_backbone_params[name].clone()
        
    # Map each parameter name to its group index
    # (To be efficient and bypass string match inside inner loop)
    name_to_group = {}
    for g_idx, group_names in enumerate(param_groups):
        for name in group_names:
            name_to_group[name] = g_idx
            
    # Accumulate task vectors scaled by task/layer-specific lambdas
    for name, base_param in base_backbone_params.items():
        g_idx = name_to_group.get(name, 13)
        for k in range(len(task_vectors)):
            coef = lambdas[k, g_idx]
            merged_backbone[name] = merged_backbone[name] + coef * task_vectors[k][name]
            
    # Quantize weights
    quant_backbone = {}
    for name, param in merged_backbone.items():
        if "weight" in name and param.dim() > 1:
            quant_backbone[name] = quantize_tensor(param, bits)
        else:
            quant_backbone[name] = param
            
    # Set up dummy shell for evaluation
    dummy_backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    dummy_expert = ExpertModel(dummy_backbone).to(device)
    dummy_expert.eval()
    
    total_entropy = 0.0
    for k, task in enumerate(TASKS):
        # Build parameter dictionary for functional call
        combined_params = {f"backbone.{name}": v for name, v in quant_backbone.items()}
        for hk, hv in heads[k].state_dict().items():
            combined_params[f"head.{hk}"] = hv
            
        # Run forward pass on calibration batch
        task_entropy = 0.0
        count = 0
        for x, _ in calib_loaders[k]:
            x = x.to(device)
            logits = functional_call(dummy_expert, combined_params, x)
            task_entropy += compute_entropy(logits)
            count += 1
        total_entropy += (task_entropy / count)
        
    return total_entropy / len(TASKS)


def run_seed_experiments(seed):
    print(f"\n==================== SEED {seed} ====================")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load baseline model and train experts
    base_backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True).to(device)
    base_params = {k: v.clone().detach() for k, v in base_backbone.state_dict().items() if "head" not in k}
    
    experts = []
    heads = []
    for task in TASKS:
        expert = train_expert(seed, task)
        experts.append(expert)
        heads.append(expert.head)
        
    # 2. Extract task vectors
    task_vectors = []
    for k, task in enumerate(TASKS):
        expert_state = experts[k].backbone.state_dict()
        t_vec = {}
        for name, base_p in base_params.items():
            t_vec[name] = expert_state[name].clone().detach() - base_p
        task_vectors.append(t_vec)
        
    # 3. Parameter groups for layer-wise coefficient assignment
    param_groups = group_parameters(base_backbone)
    
    # 4. Construct calibration loaders (64 images per task)
    calib_loaders = []
    for k, task in enumerate(TASKS):
        t_gray = transform_gray if task in ["MNIST", "FashionMNIST"] else transform
        dataset = get_dataset(task, train=True, transform=t_gray)
        rng = np.random.default_rng(seed)
        calib_indices = rng.permutation(len(dataset))[1024:1024+16]
        loader = DataLoader(Subset(dataset, calib_indices), batch_size=32, shuffle=False)
        calib_loaders.append(loader)
        
    # We will evaluate both INT8 and INT4 configurations
    results = {8: {}, 4: {}}
    
    for bits in [8, 4]:
        print(f"\n--- Running Quantization Bit-Width: {bits}-bit ---")
        
        # 4.1 baseline: FP16 Merged model (No quantization, standard uniform lambda = 0.3)
        # Reconstruct standard merged weights (FP16 upper bound)
        fp16_backbone_params = {}
        for name, base_p in base_params.items():
            g_idx = next((g for g, names in enumerate(param_groups) if name in names), 13)
            fp16_backbone_params[name] = base_p.clone()
            for k in range(len(TASKS)):
                fp16_backbone_params[name] += 0.3 * task_vectors[k][name]
                
        # Evaluate FP16 model (No quantization)
        fp16_accs = {}
        for k, task in enumerate(TASKS):
            fp16_accs[task] = evaluate_merged(fp16_backbone_params, heads[k], task, seed, bits=None)
        fp16_avg = np.mean(list(fp16_accs.values()))
        print(f"FP16 Merged Model (No Quantization) Accuracies: {fp16_accs}, Avg: {fp16_avg*100:.2f}%")
        
        # 4.2 Baseline: Quantize-then-Merge (Q-then-M)
        # Quantize each expert's backbone weights, then merge with lambda = 0.3
        q_then_m_backbone_params = {}
        # First compute quantized experts
        quant_experts = []
        for k in range(len(TASKS)):
            q_expert = {}
            expert_state = experts[k].backbone.state_dict()
            for name, param in expert_state.items():
                if "weight" in name and param.dim() > 1:
                    q_expert[name] = quantize_tensor(param, bits)
                else:
                    q_expert[name] = param
            quant_experts.append(q_expert)
            
        # Re-quantize base weights
        quant_base = {}
        for name, param in base_params.items():
            if "weight" in name and param.dim() > 1:
                quant_base[name] = quantize_tensor(param, bits)
            else:
                quant_base[name] = param
                
        # Merge pre-quantized parameters
        for name in base_params.keys():
            q_then_m_backbone_params[name] = quant_base[name].clone()
            for k in range(len(TASKS)):
                q_then_m_backbone_params[name] += 0.3 * (quant_experts[k][name] - quant_base[name])
                
        q_then_m_accs = {}
        for k, task in enumerate(TASKS):
            q_then_m_accs[task] = evaluate_merged(q_then_m_backbone_params, heads[k], task, seed, bits=None)
        q_then_m_avg = np.mean(list(q_then_m_accs.values()))
        print(f"Q-then-M (Quantize-then-Merge) Accuracies: {q_then_m_accs}, Avg: {q_then_m_avg*100:.2f}%")
        
        # 4.3 Baseline: Merge-then-Quantize (M-then-Q)
        # Fusing experts with standard uniform lambda = 0.3, then quantizing the resulting weights
        m_then_q_accs = {}
        for k, task in enumerate(TASKS):
            m_then_q_accs[task] = evaluate_merged(fp16_backbone_params, heads[k], task, seed, bits=bits)
        m_then_q_avg = np.mean(list(m_then_q_accs.values()))
        print(f"M-then-Q (Merge-then-Quantize) Accuracies: {m_then_q_accs}, Avg: {m_then_q_avg*100:.2f}%")
        
        # 4.4 Baseline: FP16 Optimized AdaMerging, then Quantized
        # Step 1: Optimize merging coefficients on the FP16 model (no quantization)
        print("Running AdaMerging (FP16 Optimized)...")
        fp16_lambdas = torch.full((len(TASKS), 14), 0.3, device=device)
        # Run 1+1 ES on FP16 model to find optimal lambdas
        best_fp16_lambdas = fp16_lambdas.clone()
        best_fp16_loss = compute_calibration_loss(best_fp16_lambdas, base_params, task_vectors, heads, calib_loaders, param_groups, bits=None)
        
        # Fast 1+1 ES on FP16
        sigma = 0.05
        for step in range(40):
            mutation = torch.randn_like(best_fp16_lambdas) * sigma
            candidate = torch.clamp(best_fp16_lambdas + mutation, 0.0, 1.0)
            loss = compute_calibration_loss(candidate, base_params, task_vectors, heads, calib_loaders, param_groups, bits=None)
            if loss < best_fp16_loss:
                best_fp16_loss = loss
                best_fp16_lambdas = candidate
                
        # Reconstruct FP16-Optimized backbone parameters
        fp16_opt_backbone = {}
        for name, base_p in base_params.items():
            g_idx = next((g for g, names in enumerate(param_groups) if name in names), 13)
            fp16_opt_backbone[name] = base_p.clone()
            for k in range(len(TASKS)):
                coef = best_fp16_lambdas[k, g_idx]
                fp16_opt_backbone[name] += coef * task_vectors[k][name]
                
        # Evaluate FP16 Optimized model without Quantization (True Unquantized Ceiling)
        fp16_opt_unquant_accs = {}
        for k, task in enumerate(TASKS):
            fp16_opt_unquant_accs[task] = evaluate_merged(fp16_opt_backbone, heads[k], task, seed, bits=None)
        fp16_opt_unquant_avg = np.mean(list(fp16_opt_unquant_accs.values()))
        print(f"AdaMerging (FP16 Optimized, Unquantized) Accuracies: {fp16_opt_unquant_accs}, Avg: {fp16_opt_unquant_avg*100:.2f}%")
        
        # Evaluate FP16 Optimized model under Quantization (M-then-Q with optimized coefficients)
        fp16_opt_quant_accs = {}
        for k, task in enumerate(TASKS):
            fp16_opt_quant_accs[task] = evaluate_merged(fp16_opt_backbone, heads[k], task, seed, bits=bits)
        fp16_opt_quant_avg = np.mean(list(fp16_opt_quant_accs.values()))
        print(f"AdaMerging (FP16 Optimized, then Quantized) Accuracies: {fp16_opt_quant_accs}, Avg: {fp16_opt_quant_avg*100:.2f}%")
        
        # 4.4.1 Baseline: FP16 Optimized AdaMerging with Adam GD (Unquantized & Quantized)
        print("Running AdaMerging (FP16 Optimized with Adam GD)...")
        fp16_adam_lambdas = torch.full((len(TASKS), 14), 0.3, device=device, requires_grad=True)
        optimizer_fp16 = optim.Adam([fp16_adam_lambdas], lr=0.02)
        
        for step in range(20):
            optimizer_fp16.zero_grad()
            loss = compute_calibration_loss(fp16_adam_lambdas, base_params, task_vectors, heads, calib_loaders, param_groups, bits=None)
            loss.backward()
            optimizer_fp16.step()
            with torch.no_grad():
                fp16_adam_lambdas.clamp_(0.0, 1.0)
                
        # Reconstruct FP16-Optimized with Adam GD backbone parameters
        fp16_opt_adam_backbone = {}
        with torch.no_grad():
            for name, base_p in base_params.items():
                g_idx = next((g for g, names in enumerate(param_groups) if name in names), 13)
                fp16_opt_adam_backbone[name] = base_p.clone()
                for k in range(len(TASKS)):
                    coef = fp16_adam_lambdas[k, g_idx]
                    fp16_opt_adam_backbone[name] += coef * task_vectors[k][name]
                    
        # Evaluate FP16 Optimized with Adam GD without Quantization
        fp16_opt_adam_unquant_accs = {}
        for k, task in enumerate(TASKS):
            fp16_opt_adam_unquant_accs[task] = evaluate_merged(fp16_opt_adam_backbone, heads[k], task, seed, bits=None)
        fp16_opt_adam_unquant_avg = np.mean(list(fp16_opt_adam_unquant_accs.values()))
        print(f"AdaMerging (FP16 Optimized with Adam, Unquantized) Accuracies: {fp16_opt_adam_unquant_accs}, Avg: {fp16_opt_adam_unquant_avg*100:.2f}%")
        
        # Evaluate FP16 Optimized with Adam GD under Quantization (post-hoc quantized)
        fp16_opt_adam_quant_accs = {}
        for k, task in enumerate(TASKS):
            fp16_opt_adam_quant_accs[task] = evaluate_merged(fp16_opt_adam_backbone, heads[k], task, seed, bits=bits)
        fp16_opt_adam_quant_avg = np.mean(list(fp16_opt_adam_quant_accs.values()))
        print(f"AdaMerging (FP16 Optimized with Adam, Quantized) Accuracies: {fp16_opt_adam_quant_accs}, Avg: {fp16_opt_adam_quant_avg*100:.2f}%")
        
        # 4.5 Q-Merge (1+1 ES Optimization directly under quantization operator)
        print("Optimizing Q-Merge (1+1 ES)...")
        qmerge_es_lambdas = torch.full((len(TASKS), 14), 0.3, device=device)
        best_es_lambdas = qmerge_es_lambdas.clone()
        best_es_loss = compute_calibration_loss(best_es_lambdas, base_params, task_vectors, heads, calib_loaders, param_groups, bits=bits)
        
        for step in range(40):
            mutation = torch.randn_like(best_es_lambdas) * sigma
            candidate = torch.clamp(best_es_lambdas + mutation, 0.0, 1.0)
            loss = compute_calibration_loss(candidate, base_params, task_vectors, heads, calib_loaders, param_groups, bits=bits)
            if loss < best_es_loss:
                best_es_loss = loss
                best_es_lambdas = candidate
                
        # Reconstruct Q-Merge (1+1 ES) weights
        qmerge_es_backbone = {}
        for name, base_p in base_params.items():
            g_idx = next((g for g, names in enumerate(param_groups) if name in names), 13)
            qmerge_es_backbone[name] = base_p.clone()
            for k in range(len(TASKS)):
                coef = best_es_lambdas[k, g_idx]
                qmerge_es_backbone[name] += coef * task_vectors[k][name]
                
        # Evaluate Q-Merge (1+1 ES)
        qmerge_es_accs = {}
        for k, task in enumerate(TASKS):
            qmerge_es_accs[task] = evaluate_merged(qmerge_es_backbone, heads[k], task, seed, bits=bits)
        qmerge_es_avg = np.mean(list(qmerge_es_accs.values()))
        print(f"Q-Merge (1+1 ES) Accuracies: {qmerge_es_accs}, Avg: {qmerge_es_avg*100:.2f}%")
        
        # 4.6 Q-Merge (Adam GD Optimization with STE directly under quantization operator)
        print("Optimizing Q-Merge (Adam GD with STE)...")
        qmerge_adam_lambdas = torch.full((len(TASKS), 14), 0.3, device=device, requires_grad=True)
        optimizer = optim.Adam([qmerge_adam_lambdas], lr=0.02)
        
        for step in range(20):
            optimizer.zero_grad()
            loss = compute_calibration_loss(qmerge_adam_lambdas, base_params, task_vectors, heads, calib_loaders, param_groups, bits=bits)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                qmerge_adam_lambdas.clamp_(0.0, 1.0)
                
        # Reconstruct Q-Merge (Adam GD) weights
        qmerge_adam_backbone = {}
        with torch.no_grad():
            for name, base_p in base_params.items():
                g_idx = next((g for g, names in enumerate(param_groups) if name in names), 13)
                qmerge_adam_backbone[name] = base_p.clone()
                for k in range(len(TASKS)):
                    coef = qmerge_adam_lambdas[k, g_idx]
                    qmerge_adam_backbone[name] += coef * task_vectors[k][name]
                    
        # Evaluate Q-Merge (Adam GD)
        qmerge_adam_accs = {}
        for k, task in enumerate(TASKS):
            qmerge_adam_accs[task] = evaluate_merged(qmerge_adam_backbone, heads[k], task, seed, bits=bits)
        qmerge_adam_avg = np.mean(list(qmerge_adam_accs.values()))
        print(f"Q-Merge (Adam GD) Accuracies: {qmerge_adam_accs}, Avg: {qmerge_adam_avg*100:.2f}%")
        
        # Evaluate Q-Merge (Adam GD) with fully quantized heads (8-bit)
        qmerge_adam_quant_head_accs = {}
        for k, task in enumerate(TASKS):
            qmerge_adam_quant_head_accs[task] = evaluate_merged(qmerge_adam_backbone, heads[k], task, seed, bits=bits, quantize_head=True)
        qmerge_adam_quant_head_avg = np.mean(list(qmerge_adam_quant_head_accs.values()))
        print(f"Q-Merge (Adam GD, Quantized Heads) Accuracies: {qmerge_adam_quant_head_accs}, Avg: {qmerge_adam_quant_head_avg*100:.2f}%")
        
        results[bits] = {
            "fp16": fp16_accs,
            "fp16_optimized_unquantized": fp16_opt_unquant_accs,
            "fp16_optimized_adam_unquantized": fp16_opt_adam_unquant_accs,
            "q_then_m": q_then_m_accs,
            "m_then_q": m_then_q_accs,
            "fp16_optimized_quantized": fp16_opt_quant_accs,
            "fp16_optimized_adam_quantized": fp16_opt_adam_quant_accs,
            "qmerge_es": qmerge_es_accs,
            "qmerge_adam": qmerge_adam_accs,
            "qmerge_adam_quant_head": qmerge_adam_quant_head_accs
        }
        
    return results

# Main execution loop
if __name__ == "__main__":
    seeds = [42, 100, 2026]
    all_results = {}
    
    for seed in seeds:
        all_results[seed] = run_seed_experiments(seed)
        
    # Collate results across seeds to calculate means and standard deviations
    final_metrics = {8: {}, 4: {}}
    for bits in [8, 4]:
        methods = [
            "fp16",
            "fp16_optimized_unquantized",
            "fp16_optimized_adam_unquantized",
            "q_then_m",
            "m_then_q",
            "fp16_optimized_quantized",
            "fp16_optimized_adam_quantized",
            "qmerge_es",
            "qmerge_adam",
            "qmerge_adam_quant_head"
        ]
        for method in methods:
            final_metrics[bits][method] = {}
            for task in TASKS + ["Average"]:
                final_metrics[bits][method][task] = []
                
        for seed in seeds:
            for method in methods:
                seed_res = all_results[seed][bits][method]
                for task in TASKS:
                    final_metrics[bits][method][task].append(seed_res[task])
                final_metrics[bits][method]["Average"].append(np.mean(list(seed_res.values())))
                
        # Calculate mean and std dev
        for method in methods:
            for task in TASKS + ["Average"]:
                vals = final_metrics[bits][method][task]
                final_metrics[bits][method][task] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals))
                }
                
    # Save metrics to json
    with open("results/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("\nAll experiments successfully completed and saved to results/metrics.json!")
