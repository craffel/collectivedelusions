import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# --- Configurations ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# --- Dataset Loading ---
def get_datasets(data_dir="./data"):
    # Grayscale conversion and resizing to 32x32 for 3-channel input
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform_gray)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_gray)
    
    cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_rgb)
    cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_rgb)
    
    return {
        "MNIST": (mnist_train, mnist_test),
        "FashionMNIST": (fmnist_train, fmnist_test),
        "CIFAR10": (cifar_train, cifar_test)
    }

# --- Model Definitions ---
def get_pretrained_resnet18():
    # Load ImageNet pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return model

# --- Fine-Tuning Experts ---
def train_expert(task_name, dataset_train, save_path):
    print(f"--- Fine-Tuning Expert for {task_name} ---")
    model = get_pretrained_resnet18()
    # Replace final linear layer with 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)
    
    # Deterministic subset of 5,000 samples
    train_subset = Subset(dataset_train, list(range(5000)))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    
    # AdamW with lr=5e-4 and weight_decay=1e-4
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert model checkpoint to {save_path}")
    return model

# --- Evaluation Helper ---
@torch.no_grad()
def evaluate_model(backbone, heads, datasets, batch_size=128):
    # backbone: ResNet-18 model with fc = nn.Identity()
    # heads: dict of nn.Linear task-specific heads
    backbone.eval()
    accuracies = {}
    
    for task_name, (_, dataset_test) in datasets.items():
        head = heads[task_name].to(DEVICE)
        head.eval()
        loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
        
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            feats = backbone(x)
            outputs = head(feats)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        acc = 100.0 * correct / total
        accuracies[task_name] = acc
        
    accuracies["Average"] = sum(accuracies.values()) / len(accuracies)
    return accuracies

# --- Calibration Implementations ---

# 1. SP-TAAC (Sparsity-Preserving Task-Agnostic Calibration)
def run_sp_taac(merged_backbone, expert_backbones, calibration_sets, N, epsilon=1e-8):
    print(f"Running SP-TAAC (N={N})...")
    # Identify BatchNorm2d layers in the merged model and experts
    bn_layers_merged = [m for m in merged_backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_layers_experts = [[m for m in exp.modules() if isinstance(m, nn.BatchNorm2d)] for exp in expert_backbones]
    
    num_bn = len(bn_layers_merged)
    print(f"Found {num_bn} BatchNorm2d layers to calibrate.")
    
    # Joint calibration set
    joint_cal_samples = []
    for task_name, d_cal in calibration_sets.items():
        joint_cal_samples.append(d_cal)
    joint_cal_tensor = torch.cat(joint_cal_samples, dim=0).to(DEVICE) # Shape: [K * N, 3, 32, 32]
    
    # We calibrate layers sequentially from shallow to deep
    for l in range(num_bn):
        # 1. Capture activations of layer l in each expert model
        # To do this, we register a forward hook on layer l, run the calibration set, then remove the hook
        expert_stds = []
        for k, exp in enumerate(expert_backbones):
            exp.eval()
            task_name = list(calibration_sets.keys())[k]
            x_cal = calibration_sets[task_name].to(DEVICE) # Shape: [N, 3, 32, 32]
            
            activations = []
            def hook(module, input, output):
                activations.append(output.detach())
                
            handle = bn_layers_experts[k][l].register_forward_hook(hook)
            with torch.no_grad():
                _ = exp(x_cal)
            handle.remove()
            
            act_tensor = torch.cat(activations, dim=0) # [N, C, H, W]
            # Compute global variance over batches, channels, and spatial dims
            var = torch.var(act_tensor)
            std = torch.sqrt(var + epsilon).item()
            expert_stds.append(std)
            
        # Target standard deviation: average of experts
        target_std = sum(expert_stds) / len(expert_stds)
        
        # 2. Capture activations of layer l in the merged model on the joint calibration set
        merged_backbone.eval()
        activations_merged = []
        def hook_merged(module, input, output):
            activations_merged.append(output.detach())
            
        handle_merged = bn_layers_merged[l].register_forward_hook(hook_merged)
        with torch.no_grad():
            _ = merged_backbone(joint_cal_tensor)
        handle_merged.remove()
        
        act_merged_tensor = torch.cat(activations_merged, dim=0)
        var_merged = torch.var(act_merged_tensor)
        merged_std = torch.sqrt(var_merged + epsilon).item()
        
        # Scaling factor gamma_l
        gamma_l = target_std / merged_std
        
        # Apply in-place scaling to the merged model's BatchNorm layer parameters (weight and bias)
        bn_layers_merged[l].weight.data.mul_(gamma_l)
        bn_layers_merged[l].bias.data.mul_(gamma_l)
        
    print("SP-TAAC complete.")

# 2. TAAC (Channel-Wise Task-Agnostic Activation Calibration)
def run_taac(merged_backbone, expert_backbones, calibration_sets, N, epsilon=1e-8):
    print(f"Running TAAC (N={N})...")
    bn_layers_merged = [m for m in merged_backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_layers_experts = [[m for m in exp.modules() if isinstance(m, nn.BatchNorm2d)] for exp in expert_backbones]
    num_bn = len(bn_layers_merged)
    
    joint_cal_samples = []
    for task_name, d_cal in calibration_sets.items():
        joint_cal_samples.append(d_cal)
    joint_cal_tensor = torch.cat(joint_cal_samples, dim=0).to(DEVICE)
    
    for l in range(num_bn):
        expert_means = []
        expert_vars = []
        for k, exp in enumerate(expert_backbones):
            exp.eval()
            task_name = list(calibration_sets.keys())[k]
            x_cal = calibration_sets[task_name].to(DEVICE)
            
            activations = []
            def hook(module, input, output):
                activations.append(output.detach())
            handle = bn_layers_experts[k][l].register_forward_hook(hook)
            with torch.no_grad():
                _ = exp(x_cal)
            handle.remove()
            
            act_tensor = torch.cat(activations, dim=0) # [N, C, H, W]
            # Channel-wise mean and var (reduce over B, H, W)
            # act_tensor has shape [B, C, H, W]
            mean = act_tensor.mean(dim=(0, 2, 3)) # [C]
            var = act_tensor.var(dim=(0, 2, 3), unbiased=False) # [C]
            expert_means.append(mean)
            expert_vars.append(var)
            
        # Target mean and var: average of experts
        target_mean = torch.stack(expert_means, dim=0).mean(dim=0) # [C]
        target_std = torch.stack([torch.sqrt(v + epsilon) for v in expert_vars], dim=0).mean(dim=0) # [C]
        
        # Merged model activations
        merged_backbone.eval()
        activations_merged = []
        def hook_merged(module, input, output):
            activations_merged.append(output.detach())
        handle_merged = bn_layers_merged[l].register_forward_hook(hook_merged)
        with torch.no_grad():
            _ = merged_backbone(joint_cal_tensor)
        handle_merged.remove()
        
        act_merged_tensor = torch.cat(activations_merged, dim=0)
        merged_mean = act_merged_tensor.mean(dim=(0, 2, 3)) # [C]
        merged_var = act_merged_tensor.var(dim=(0, 2, 3), unbiased=False) # [C]
        merged_std = torch.sqrt(merged_var + epsilon) # [C]
        
        # Scaling factor s_c and bias b_c per channel
        s_c = target_std / merged_std
        b_c = target_mean - s_c * merged_mean
        
        # Apply channel-wise in-place parameters update
        bn_layers_merged[l].weight.data.copy_(bn_layers_merged[l].weight.data * s_c)
        bn_layers_merged[l].bias.data.copy_(bn_layers_merged[l].bias.data * s_c + b_c)
        
    print("TAAC complete.")

# 3. FWMM (Feature-Wise Moment Matching) for Classification Heads
def run_fwmm(calibrated_backbone, expert_backbones, heads, calibration_sets, N, epsilon=1e-8):
    print("Running Closed-Form Feature-Wise Moment Matching (FWMM) on classification heads...")
    # Create copies of experts and merged backbones with Identity fc
    # (Since we pass the linear heads dictionary separately, we temporarily make backbone.fc an Identity layer)
    cal_backbone_temp = copy.deepcopy(calibrated_backbone)
    cal_backbone_temp.fc = nn.Identity()
    cal_backbone_temp.eval()
    
    expert_backbones_temp = []
    for exp in expert_backbones:
        eb = copy.deepcopy(exp)
        eb.fc = nn.Identity()
        eb.eval()
        expert_backbones_temp.append(eb)
        
    cal_heads = {}
    
    # For each task, compute statistics of the features before the FC layer and update head parameters in closed form
    for k, task_name in enumerate(calibration_sets.keys()):
        x_cal = calibration_sets[task_name].to(DEVICE)
        
        # Get expert features
        with torch.no_grad():
            feats_expert = expert_backbones_temp[k](x_cal) # [N, 512]
        mean_expert = feats_expert.mean(dim=0) # [512]
        std_expert = torch.sqrt(feats_expert.var(dim=0) + epsilon) # [512]
        
        # Get merged calibrated backbone features
        with torch.no_grad():
            feats_merged = cal_backbone_temp(x_cal) # [N, 512]
        mean_merged = feats_merged.mean(dim=0) # [512]
        std_merged = torch.sqrt(feats_merged.var(dim=0) + epsilon) # [512]
        
        # Closed-form feature correction factors (scale and shift)
        S = std_expert / std_merged
        
        # Update head weights and bias in closed-form
        head_orig = heads[task_name].to(DEVICE)
        head_new = nn.Linear(head_orig.in_features, head_orig.out_features).to(DEVICE)
        
        W_orig = head_orig.weight.data.clone()
        b_orig = head_orig.bias.data.clone()
        
        # Scale weights column-wise
        W_new = W_orig * S.unsqueeze(0)
        # Shift bias based on means (using original weight)
        b_new = b_orig + torch.mv(W_orig, mean_expert - mean_merged * S)
        
        head_new.weight.data.copy_(W_new)
        head_new.bias.data.copy_(b_new)
        cal_heads[task_name] = head_new
        
    print("FWMM on classification heads complete.")
    return cal_heads

# 3.5. FWMM with Bayesian Shrinkage (Sparsity and Sign Preserving)
def run_fwmm_shrinkage(calibrated_backbone, expert_backbones, heads, calibration_sets, N, N0=16, use_shift=True, epsilon=1e-8):
    print(f"Running Closed-Form FWMM with Bayesian Shrinkage (N={N}, N0={N0}, use_shift={use_shift})...")
    # Create copies of experts and merged backbones with Identity fc
    cal_backbone_temp = copy.deepcopy(calibrated_backbone)
    cal_backbone_temp.fc = nn.Identity()
    cal_backbone_temp.eval()
    
    expert_backbones_temp = []
    for exp in expert_backbones:
        eb = copy.deepcopy(exp)
        eb.fc = nn.Identity()
        eb.eval()
        expert_backbones_temp.append(eb)
        
    cal_heads = {}
    
    # Shrinkage factor lambda
    lam = N / (N + N0)
    
    # For each task, compute statistics of the features before the FC layer and update head parameters in closed form
    for k, task_name in enumerate(calibration_sets.keys()):
        x_cal = calibration_sets[task_name].to(DEVICE)
        
        # Get expert features
        with torch.no_grad():
            feats_expert = expert_backbones_temp[k](x_cal) # [N, 512]
        mean_expert = feats_expert.mean(dim=0) # [512]
        std_expert = torch.sqrt(feats_expert.var(dim=0) + epsilon) # [512]
        
        # Get merged calibrated backbone features
        with torch.no_grad():
            feats_merged = cal_backbone_temp(x_cal) # [N, 512]
        mean_merged = feats_merged.mean(dim=0) # [512]
        std_merged = torch.sqrt(feats_merged.var(dim=0) + epsilon) # [512]
        
        # Regularize standard deviations by shrinking towards their global mean
        std_expert_reg = (1.0 - lam) * std_expert.mean() + lam * std_expert
        std_merged_reg = (1.0 - lam) * std_merged.mean() + lam * std_merged
        
        # Closed-form feature correction factors (scale and shift) using regularized stds
        S = std_expert_reg / std_merged_reg
        
        # Update head weights and bias in closed-form
        head_orig = heads[task_name].to(DEVICE)
        head_new = nn.Linear(head_orig.in_features, head_orig.out_features).to(DEVICE)
        
        W_orig = head_orig.weight.data.clone()
        b_orig = head_orig.bias.data.clone()
        
        # Scale weights column-wise
        W_new = W_orig * S.unsqueeze(0)
        
        # Shift bias based on means (using original weight)
        if use_shift:
            # Shift bias is scaled by lambda
            mean_diff = lam * (mean_expert - mean_merged * S)
            b_new = b_orig + torch.mv(W_orig, mean_diff)
        else:
            b_new = b_orig
            
        head_new.weight.data.copy_(W_new)
        head_new.bias.data.copy_(b_new)
        cal_heads[task_name] = head_new
        
    print("FWMM with Bayesian Shrinkage complete.")
    return cal_heads

# 3.6. MOMO-Merge with Orthogonal Procrustes Head Alignment (CF-PHA)
def run_procrustes_alignment(calibrated_backbone, expert_backbones, heads, calibration_sets, N, N0_var=16, N0_cov=512, epsilon=1e-8):
    print(f"Running Closed-Form Procrustes Head Alignment (CF-PHA) with Dual-Scale Bayesian Shrinkage (N={N}, N0_var={N0_var}, N0_cov={N0_cov})...")
    cal_backbone_temp = copy.deepcopy(calibrated_backbone)
    cal_backbone_temp.fc = nn.Identity()
    cal_backbone_temp.eval()
    
    expert_backbones_temp = []
    for exp in expert_backbones:
        eb = copy.deepcopy(exp)
        eb.fc = nn.Identity()
        eb.eval()
        expert_backbones_temp.append(eb)
        
    cal_heads = {}
    lam_var = N / (N + N0_var)
    lam_cov = N / (N + N0_cov)
    
    for k, task_name in enumerate(calibration_sets.keys()):
        x_cal = calibration_sets[task_name].to(DEVICE)
        
        # Get expert features
        with torch.no_grad():
            feats_expert = expert_backbones_temp[k](x_cal) # [N, 512]
        mean_expert = feats_expert.mean(dim=0) # [512]
        std_expert = torch.sqrt(feats_expert.var(dim=0) + epsilon) # [512]
        
        # Get merged calibrated backbone features
        with torch.no_grad():
            feats_merged = cal_backbone_temp(x_cal) # [N, 512]
        mean_merged = feats_merged.mean(dim=0) # [512]
        std_merged = torch.sqrt(feats_merged.var(dim=0) + epsilon) # [512]
        
        # Shrink standard deviations
        std_expert_reg = (1.0 - lam_var) * std_expert.mean() + lam_var * std_expert
        std_merged_reg = (1.0 - lam_var) * std_merged.mean() + lam_var * std_merged
        
        # Centered and standardized features
        X_centered = (feats_merged - mean_merged) / std_merged_reg.unsqueeze(0)
        Z_centered = (feats_expert - mean_expert) / std_expert_reg.unsqueeze(0)
        
        # Cross-covariance matrix C
        C = torch.matmul(X_centered.t(), Z_centered) / N # [512, 512]
        
        # Shrink cross-covariance C toward Identity
        C_reg = (1.0 - lam_cov) * torch.eye(512, device=DEVICE) + lam_cov * C
        
        # Singular Value Decomposition on C_reg
        U, S_vals, V = torch.linalg.svd(C_reg)
        
        # Optimal orthogonal rotation matrix R
        R = torch.matmul(U, V) # [512, 512]
        
        # Original head
        head_orig = heads[task_name].to(DEVICE)
        head_new = nn.Linear(head_orig.in_features, head_orig.out_features).to(DEVICE)
        
        W_orig = head_orig.weight.data.clone()
        b_orig = head_orig.bias.data.clone()
        
        # Transform weight: W_new = W_orig * diag(std_expert_reg) * R^T * diag(1/std_merged_reg)
        W_scaled = W_orig * std_expert_reg.unsqueeze(0) # scale columns by expert std
        W_rotated = torch.matmul(W_scaled, R.t())       # apply rotation R^T
        W_new = W_rotated / std_merged_reg.unsqueeze(0) # divide columns by merged std
        
        # Transform bias (using lam_var for shift shrinkage)
        b_new = b_orig + torch.mv(W_orig, lam_var * mean_expert) - torch.mv(W_new, lam_var * mean_merged)
        
        head_new.weight.data.copy_(W_new)
        head_new.bias.data.copy_(b_new)
        cal_heads[task_name] = head_new
        
    print("CF-PHA complete.")
    return cal_heads

# 4. REDA-SFT (Supervised Head Fine-Tuning)
def run_reda_sft(calibrated_backbone, heads, calibration_sets, N, epochs=15, lr=1e-3):
    print(f"Running REDA-SFT Head Fine-Tuning (epochs={epochs}, lr={lr})...")
    cal_backbone_temp = copy.deepcopy(calibrated_backbone)
    cal_backbone_temp.fc = nn.Identity()
    cal_backbone_temp.eval()
    
    cal_heads = {}
    
    for task_name, x_cal in calibration_sets.items():
        # Get target labels (dummy target label for simplicity? No, we need supervised labels!
        # Wait, how do we get labels for calibration set?
        # Let's extract both images and labels from the calibration set!)
        # We need both x and y. Let's make sure our calibration sets contain both!
        pass
        
    # Let's rewrite how calibration sets are stored: they should contain both (x, y) tensors!
    # Yes, we will define calibration_sets as a dict of (x_tensor, y_tensor)!
    # This is much better.

# --- Main Experimental Flow ---
def main():
    datasets = get_datasets()
    
    # Check if expert checkpoints exist
    expert_paths = {
        "MNIST": "expert_mnist.pth",
        "FashionMNIST": "expert_fashion.pth",
        "CIFAR10": "expert_cifar.pth"
    }
    
    experts = {}
    for task_name, (train_dataset, _) in datasets.items():
        save_path = expert_paths[task_name]
        if not os.path.exists(save_path):
            print(f"Expert checkpoint for {task_name} not found. Training from scratch...")
            train_expert(task_name, train_dataset, save_path)
        
        # Load expert model
        model = get_pretrained_resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        experts[task_name] = model
        
    print("--- Evaluating Individual Experts (Baselines) ---")
    for task_name, exp in experts.items():
        # Replace fc with Identity to separate backbone and head
        exp_backbone = copy.deepcopy(exp)
        exp_backbone.fc = nn.Identity()
        heads_dict = {task_name: exp.fc}
        
        # Evaluate on the specific task's dataset only
        accs = evaluate_model(exp_backbone, heads_dict, {task_name: datasets[task_name]})
        print(f"Expert {task_name} accuracy: {accs[task_name]:.2f}%")
        
    # Let's construct a list of calibration sample sizes to sweep
    N_sizes = [4, 8, 16, 32, 64, 128, 256]
    
    # We will run sweeps over Weight Averaging first
    results_table = []
    
    # Pre-extract calibration samples and labels for each task
    # We construct calibration sets for the maximum size 256, then slice them as needed
    calibration_data = {}
    for task_name, (train_dataset, _) in datasets.items():
        # Deterministic subset of training set
        loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        x_cal, y_cal = next(iter(loader))
        calibration_data[task_name] = (x_cal, y_cal)
        
    # --- Weight Averaging Merging ---
    print("\n" + "="*50)
    print("--- Weight Averaging (WA) Merging ---")
    print("="*50)
    
    # Create the base merged model by averaging expert weights (excluding heads)
    merged_model_wa = get_pretrained_resnet18()
    merged_model_wa.fc = nn.Identity() # The merged model has no fc layer or uses Identity backbone
    merged_model_wa = merged_model_wa.to(DEVICE)
    
    expert_backbones = []
    for task_name in ["MNIST", "FashionMNIST", "CIFAR10"]:
        exp = experts[task_name]
        exp_backbone = copy.deepcopy(exp)
        exp_backbone.fc = nn.Identity()
        expert_backbones.append(exp_backbone)
        
    # Average backbone weights
    merged_state_dict = copy.deepcopy(merged_model_wa.state_dict())
    expert_state_dicts = [exp.state_dict() for exp in expert_backbones]
    
    for key in merged_state_dict.keys():
        # Compute average across experts
        stacked = torch.stack([sd[key].float() for sd in expert_state_dicts], dim=0)
        merged_state_dict[key].copy_(stacked.mean(dim=0))
        
    merged_model_wa.load_state_dict(merged_state_dict)
    
    # Classification heads are the original expert heads
    original_heads = {
        "MNIST": experts["MNIST"].fc,
        "FashionMNIST": experts["FashionMNIST"].fc,
        "CIFAR10": experts["CIFAR10"].fc
    }
    
    # Evaluate uncalibrated WA
    acc_uncal = evaluate_model(merged_model_wa, original_heads, datasets)
    print(f"Uncalibrated WA - MNIST: {acc_uncal['MNIST']:.2f}% | F-MNIST: {acc_uncal['FashionMNIST']:.2f}% | CIFAR-10: {acc_uncal['CIFAR10']:.2f}% | Avg: {acc_uncal['Average']:.2f}%")
    results_table.append({
        "N": "N/A",
        "Method": "Uncalibrated WA",
        "MNIST": acc_uncal["MNIST"],
        "FashionMNIST": acc_uncal["FashionMNIST"],
        "CIFAR-10": acc_uncal["CIFAR10"],
        "Average": acc_uncal["Average"]
    })
    
    # Sweep calibration size N
    for N in N_sizes:
        print(f"\n--- Sweeping Calibration Size N = {N} ---")
        # Slice calibration sets of size N
        cal_sets_x = {task: calibration_data[task][0][:N] for task in ["MNIST", "FashionMNIST", "CIFAR10"]}
        cal_sets_y = {task: calibration_data[task][1][:N] for task in ["MNIST", "FashionMNIST", "CIFAR10"]}
        
        # --- 1. SP-TAAC Only (Backbone Calibration, original heads) ---
        backbone_sp_taac = copy.deepcopy(merged_model_wa)
        run_sp_taac(backbone_sp_taac, expert_backbones, cal_sets_x, N)
        acc_sp_taac = evaluate_model(backbone_sp_taac, original_heads, datasets)
        print(f"SP-TAAC Only - MNIST: {acc_sp_taac['MNIST']:.2f}% | F-MNIST: {acc_sp_taac['FashionMNIST']:.2f}% | CIFAR-10: {acc_sp_taac['CIFAR10']:.2f}% | Avg: {acc_sp_taac['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "SP-TAAC Only",
            "MNIST": acc_sp_taac["MNIST"],
            "FashionMNIST": acc_sp_taac["FashionMNIST"],
            "CIFAR-10": acc_sp_taac["CIFAR10"],
            "Average": acc_sp_taac["Average"]
        })
        
        # --- 2. TAAC Only (Channel-wise Backbone Calibration, original heads) ---
        backbone_taac = copy.deepcopy(merged_model_wa)
        run_taac(backbone_taac, expert_backbones, cal_sets_x, N)
        acc_taac = evaluate_model(backbone_taac, original_heads, datasets)
        print(f"TAAC Only - MNIST: {acc_taac['MNIST']:.2f}% | F-MNIST: {acc_taac['FashionMNIST']:.2f}% | CIFAR-10: {acc_taac['CIFAR10']:.2f}% | Avg: {acc_taac['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "TAAC Only",
            "MNIST": acc_taac["MNIST"],
            "FashionMNIST": acc_taac["FashionMNIST"],
            "CIFAR-10": acc_taac["CIFAR10"],
            "Average": acc_taac["Average"]
        })
        
        # --- 3. FWMM Only (No Backbone Calibration, Head Calibration) ---
        heads_fwmm = run_fwmm(merged_model_wa, expert_backbones, original_heads, cal_sets_x, N)
        acc_fwmm_only = evaluate_model(merged_model_wa, heads_fwmm, datasets)
        print(f"FWMM Only - MNIST: {acc_fwmm_only['MNIST']:.2f}% | F-MNIST: {acc_fwmm_only['FashionMNIST']:.2f}% | CIFAR-10: {acc_fwmm_only['CIFAR10']:.2f}% | Avg: {acc_fwmm_only['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "FWMM Only (Head)",
            "MNIST": acc_fwmm_only["MNIST"],
            "FashionMNIST": acc_fwmm_only["FashionMNIST"],
            "CIFAR-10": acc_fwmm_only["CIFAR10"],
            "Average": acc_fwmm_only["Average"]
        })
        
        # --- 4. MOMO-Merge (SP-TAAC + FWMM) (Ours, Minimalist) ---
        backbone_momo = copy.deepcopy(merged_model_wa)
        run_sp_taac(backbone_momo, expert_backbones, cal_sets_x, N)
        
        # 4a. Original MOMO-Merge
        heads_momo_orig = run_fwmm(backbone_momo, expert_backbones, original_heads, cal_sets_x, N)
        acc_momo_orig = evaluate_model(backbone_momo, heads_momo_orig, datasets)
        print(f"MOMO-Merge (Orig) - MNIST: {acc_momo_orig['MNIST']:.2f}% | F-MNIST: {acc_momo_orig['FashionMNIST']:.2f}% | CIFAR-10: {acc_momo_orig['CIFAR10']:.2f}% | Avg: {acc_momo_orig['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "MOMO-Merge (Orig)",
            "MNIST": acc_momo_orig["MNIST"],
            "FashionMNIST": acc_momo_orig["FashionMNIST"],
            "CIFAR-10": acc_momo_orig["CIFAR10"],
            "Average": acc_momo_orig["Average"]
        })
        
        # 4b. MOMO-Merge with Shrinkage + Shift
        heads_momo_shrink_shift = run_fwmm_shrinkage(backbone_momo, expert_backbones, original_heads, cal_sets_x, N, N0=16, use_shift=True)
        acc_momo_shrink_shift = evaluate_model(backbone_momo, heads_momo_shrink_shift, datasets)
        print(f"MOMO-Merge (Shrink-Shift) - MNIST: {acc_momo_shrink_shift['MNIST']:.2f}% | F-MNIST: {acc_momo_shrink_shift['FashionMNIST']:.2f}% | CIFAR-10: {acc_momo_shrink_shift['CIFAR10']:.2f}% | Avg: {acc_momo_shrink_shift['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "MOMO-Merge (Shrink-Shift)",
            "MNIST": acc_momo_shrink_shift["MNIST"],
            "FashionMNIST": acc_momo_shrink_shift["FashionMNIST"],
            "CIFAR-10": acc_momo_shrink_shift["CIFAR10"],
            "Average": acc_momo_shrink_shift["Average"]
        })
        
        # 4c. MOMO-Merge with Shrinkage but NO Shift (Sparsity and Sign Preserving)
        heads_momo_shrink_noshift = run_fwmm_shrinkage(backbone_momo, expert_backbones, original_heads, cal_sets_x, N, N0=16, use_shift=False)
        acc_momo_shrink_noshift = evaluate_model(backbone_momo, heads_momo_shrink_noshift, datasets)
        print(f"MOMO-Merge (Shrink-NoShift) - MNIST: {acc_momo_shrink_noshift['MNIST']:.2f}% | F-MNIST: {acc_momo_shrink_noshift['FashionMNIST']:.2f}% | CIFAR-10: {acc_momo_shrink_noshift['CIFAR10']:.2f}% | Avg: {acc_momo_shrink_noshift['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "MOMO-Merge (Shrink-NoShift)",
            "MNIST": acc_momo_shrink_noshift["MNIST"],
            "FashionMNIST": acc_momo_shrink_noshift["FashionMNIST"],
            "CIFAR-10": acc_momo_shrink_noshift["CIFAR10"],
            "Average": acc_momo_shrink_noshift["Average"]
        })
        
        # 4d. MOMO-Merge with Orthogonal Procrustes Head Alignment (CF-PHA) (Ours, Non-diagonal)
        heads_momo_cf_pha = run_procrustes_alignment(backbone_momo, expert_backbones, original_heads, cal_sets_x, N, N0_var=16, N0_cov=512)
        acc_momo_cf_pha = evaluate_model(backbone_momo, heads_momo_cf_pha, datasets)
        print(f"MOMO-Merge (CF-PHA) - MNIST: {acc_momo_cf_pha['MNIST']:.2f}% | F-MNIST: {acc_momo_cf_pha['FashionMNIST']:.2f}% | CIFAR-10: {acc_momo_cf_pha['CIFAR10']:.2f}% | Avg: {acc_momo_cf_pha['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "MOMO-Merge (CF-PHA)",
            "MNIST": acc_momo_cf_pha["MNIST"],
            "FashionMNIST": acc_momo_cf_pha["FashionMNIST"],
            "CIFAR-10": acc_momo_cf_pha["CIFAR10"],
            "Average": acc_momo_cf_pha["Average"]
        })
        
        # --- 5. REDA-SFT (SP-TAAC + Gradient SFT Head) ---
        print("Running REDA-SFT (SP-TAAC + Gradient Head SFT)...")
        backbone_reda = copy.deepcopy(merged_model_wa)
        run_sp_taac(backbone_reda, expert_backbones, cal_sets_x, N)
        
        # Fine-tune each task's head on its calibration set
        backbone_reda_temp = copy.deepcopy(backbone_reda)
        backbone_reda_temp.fc = nn.Identity()
        backbone_reda_temp.eval()
        
        heads_reda = {}
        for task_name in ["MNIST", "FashionMNIST", "CIFAR10"]:
            orig_head = original_heads[task_name].to(DEVICE)
            # Create a clean head copy
            head_sft = nn.Linear(orig_head.in_features, orig_head.out_features).to(DEVICE)
            head_sft.load_state_dict(orig_head.state_dict())
            head_sft.train()
            
            optimizer = optim.AdamW(head_sft.parameters(), lr=1e-3, weight_decay=0.0)
            criterion = nn.CrossEntropyLoss()
            
            x_c = cal_sets_x[task_name].to(DEVICE)
            y_c = cal_sets_y[task_name].to(DEVICE)
            
            # Supervised training for 15 epochs
            for epoch in range(15):
                optimizer.zero_grad()
                with torch.no_grad():
                    feats = backbone_reda_temp(x_c)
                logits = head_sft(feats)
                loss = criterion(logits, y_c)
                loss.backward()
                optimizer.step()
                
            heads_reda[task_name] = head_sft
            
        acc_reda = evaluate_model(backbone_reda, heads_reda, datasets)
        print(f"REDA-SFT - MNIST: {acc_reda['MNIST']:.2f}% | F-MNIST: {acc_reda['FashionMNIST']:.2f}% | CIFAR-10: {acc_reda['CIFAR10']:.2f}% | Avg: {acc_reda['Average']:.2f}%")
        results_table.append({
            "N": N,
            "Method": "REDA-SFT (Gradient)",
            "MNIST": acc_reda["MNIST"],
            "FashionMNIST": acc_reda["FashionMNIST"],
            "CIFAR-10": acc_reda["CIFAR10"],
            "Average": acc_reda["Average"]
        })

    # --- Print and save the results ---
    print("\n" + "="*86)
    print("--- EXPERIMENTAL RESULTS SUMMARY ---")
    print("="*86)
    print(f"{'N':<5} | {'Method':<28} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*86)
    for row in results_table:
        mnist_str = f"{row['MNIST']:.2f}%"
        fmnist_str = f"{row['FashionMNIST']:.2f}%"
        cifar_str = f"{row['CIFAR-10']:.2f}%"
        avg_str = f"{row['Average']:.2f}%"
        print(f"{str(row['N']):<5} | {row['Method']:<28} | {mnist_str:<8} | {fmnist_str:<8} | {cifar_str:<8} | {avg_str:<8}")
    print("="*86)
    
    # Save results to a markdown text file
    with open("results.txt", "w") as f:
        f.write("# Model Merging Experimental Results\n\n")
        f.write(f"{'N':<5} | {'Method':<28} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}\n")
        f.write("-" * 86 + "\n")
        for row in results_table:
            mnist_str = f"{row['MNIST']:.2f}%"
            fmnist_str = f"{row['FashionMNIST']:.2f}%"
            cifar_str = f"{row['CIFAR-10']:.2f}%"
            avg_str = f"{row['Average']:.2f}%"
            f.write(f"{str(row['N']):<5} | {row['Method']:<28} | {mnist_str:<8} | {fmnist_str:<8} | {cifar_str:<8} | {avg_str:<8}\n")

if __name__ == "__main__":
    main()
