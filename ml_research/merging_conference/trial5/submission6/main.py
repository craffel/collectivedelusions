import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call
import copy
import numpy as np
import random
import os

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.backends.cudnn.enabled = False

# ==========================================
# 1. Dataset Preparation
# ==========================================
print("Preparing datasets...")
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
svhn_train = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transform_train)

cifar_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
svhn_test = torchvision.datasets.SVHN('./data', split='test', download=True, transform=transform_test)

# Subset 5000 images for fast, high-quality training
cifar_indices = torch.randperm(len(cifar_train))[:5000].tolist()
svhn_indices = torch.randperm(len(svhn_train))[:5000].tolist()

cifar_train_subset = Subset(cifar_train, cifar_indices)
svhn_train_subset = Subset(svhn_train, svhn_indices)

cifar_train_loader = DataLoader(cifar_train_subset, batch_size=64, shuffle=True)
svhn_train_loader = DataLoader(svhn_train_subset, batch_size=64, shuffle=True)

# Construct deterministic test streams of 32 batches of size 64 (2048 images total, 1024 per task)
cifar_test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)
svhn_test_loader = DataLoader(svhn_test, batch_size=64, shuffle=False)

cifar_test_batches = []
for x, y in cifar_test_loader:
    cifar_test_batches.append((x, y, 0)) # 0 for CIFAR-10 task
    if len(cifar_test_batches) == 16:
        break

svhn_test_batches = []
for x, y in svhn_test_loader:
    svhn_test_batches.append((x, y, 1)) # 1 for SVHN task
    if len(svhn_test_batches) == 16:
        break

# Alternating Stream (High Frequency): C10, SVHN, C10, SVHN...
alt_stream = []
for i in range(16):
    alt_stream.append(cifar_test_batches[i])
    alt_stream.append(svhn_test_batches[i])

# Sequential Stream (Low Frequency): C10, C10..., SVHN, SVHN...
seq_stream = []
for i in range(16):
    seq_stream.append(cifar_test_batches[i])
for i in range(16):
    seq_stream.append(svhn_test_batches[i])

# ==========================================
# 2. Expert Fine-Tuning
# ==========================================
def train_expert(name, train_loader, test_loader):
    model_path = f"expert_{name}.pth"
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    base_model = base_model.to(device)
    
    if os.path.exists(model_path):
        print(f"Loading existing expert for {name} from {model_path}...")
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        return base_model
    
    print(f"Fine-tuning expert model for {name}...")
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    base_model.train()
    for epoch in range(3):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = base_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += outputs.argmax(dim=-1).eq(y).sum().item()
            total += x.size(0)
        print(f"Epoch {epoch+1}/3 - Loss: {total_loss/total:.4f}, Accuracy: {correct/total*100:.2f}%")
        
    # Evaluate standalone accuracy
    base_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = base_model(x)
            correct += outputs.argmax(dim=-1).eq(y).sum().item()
            total += x.size(0)
    print(f"Standalone Accuracy of {name} Expert: {correct/total*100:.2f}%")
    torch.save(base_model.state_dict(), model_path)
    return base_model

print("="*40)
expert_cifar10 = train_expert("cifar10", cifar_train_loader, cifar_test_batches)
print("="*40)
expert_svhn = train_expert("svhn", svhn_train_loader, svhn_test_batches)
print("="*40)

# ==========================================
# 3. Compute joint layer-wise Fisher prior
# ==========================================
print("Computing joint parameter-level diagonal Fisher Information Prior...")
# We use a calibration loader combining CIFAR-10 and SVHN
cal_cifar_iter = iter(DataLoader(cifar_train_subset, batch_size=32, shuffle=True))
cal_svhn_iter = iter(DataLoader(svhn_train_subset, batch_size=32, shuffle=True))

base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
base_resnet.eval()

fisher_dict = {n: torch.zeros_like(p) for n, p in base_resnet.named_parameters() if 'fc' not in n}

# Collect 256 samples in total (4 batches of 64)
for _ in range(4):
    c10_x, _ = next(cal_cifar_iter)
    svhn_x, _ = next(cal_svhn_iter)
    x = torch.cat([c10_x, svhn_x], dim=0).to(device) # size 64
    
    outputs = base_resnet(x)
    log_probs = torch.log_softmax(outputs, dim=-1)
    # Empirical Fisher: gradient of log-likelihood of predicted class
    preds = outputs.argmax(dim=-1)
    loss = torch.gather(log_probs, 1, preds.unsqueeze(1)).mean()
    
    base_resnet.zero_grad()
    loss.backward()
    
    for n, p in base_resnet.named_parameters():
        if 'fc' not in n and p.grad is not None:
            fisher_dict[n] += (p.grad.data ** 2) / 4.0

# Compute joint layer-wise Fisher sensitivity
joint_fisher = {}
for n, p_fish in fisher_dict.items():
    # Mean Fisher value over the parameter tensor
    joint_fisher[n] = p_fish.mean().item()

# Print out some key layers and their Fisher Information
sorted_fisher = sorted(joint_fisher.items(), key=lambda x: x[1], reverse=True)
print("Top 5 most sensitive layers (highest Fisher):")
for n, f in sorted_fisher[:5]:
    print(f"  {n}: {f:.6f}")
print("Top 5 most robust layers (lowest Fisher):")
for n, f in sorted_fisher[-5:]:
    print(f"  {n}: {f:.6f}")

# Extract expert parameters and buffers
expert1_params = {n: p.clone().detach() for n, p in expert_cifar10.named_parameters()}
expert2_params = {n: p.clone().detach() for n, p in expert_svhn.named_parameters()}
expert1_buffers = {n: b.clone().detach() for n, b in expert_cifar10.named_buffers()}
expert2_buffers = {n: b.clone().detach() for n, b in expert_svhn.named_buffers()}

# Get keys for the encoder (excluding fc)
encoder_keys = [n for n in expert1_params.keys() if 'fc' not in n]

# ==========================================
# 4. Evaluation Function
# ==========================================
def evaluate_method(method_name, optimizer_name, stream, eta, alpha=0.5, theta_F=None, reset_threshold=2.5, noise_std=0.0, oracle_grouping=False):
    """
    Evaluates a test-time model merging method on a test stream.
    method_name: 'static', 'standard_tta', 'lfwa', 'pc_merge', 'lf_proj'
    optimizer_name: 'adam', 'sgd'
    stream: list of batches (x, y, task_id)
    eta: learning rate
    alpha: LFWA sensitivity power
    theta_F: Fisher projection threshold
    oracle_grouping: Use true labels instead of predicted labels for class grouping
    """
    set_seed(42)
    
    # Initialize trainable logit coefficients (lambda_w = 0.0 corresponds to 50/50 merge)
    lambda_logits = {n: torch.zeros((), requires_grad=True, device=device) for n in encoder_keys}
    
    # Optimizer setup
    params_list = list(lambda_logits.values())
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=eta, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(params_list, lr=eta)
        
    # EMA loss for OPR resets (PC-Merge)
    ema_loss = None
    beta_ema = 0.9
    
    total_correct = 0
    total_samples = 0
    
    # For recording trajectories
    coef_trajectory = []
    
    # Evaluate batch by batch
    for step, (x, y, task_id) in enumerate(stream):
        x, y = x.to(device), y.to(device)
        if noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std
        
        # 1. Prediction under current coefficients (before update)
        # Construct merged parameters and buffers for prediction
        merged_params = {}
        for n, p in base_resnet.named_parameters():
            if 'fc' not in n:
                prob1 = torch.sigmoid(lambda_logits[n])
                prob2 = 1.0 - prob1
                merged_params[n] = prob1 * expert1_params[n] + prob2 * expert2_params[n]
            else:
                # Use active task's classification head
                if task_id == 0:
                    merged_params[n] = expert1_params[n]
                else:
                    merged_params[n] = expert2_params[n]
                    
        # Buffers are assigned based on the active task
        merged_buffers = {}
        for n, buf in base_resnet.named_buffers():
            if 'fc' not in n:
                if task_id == 0:
                    merged_buffers[n] = expert1_buffers[n]
                else:
                    merged_buffers[n] = expert2_buffers[n]
                    
        # Forward pass on merged model
        state_dict = {**merged_params, **merged_buffers}
        with torch.no_grad():
            outputs = functional_call(base_resnet, state_dict, x)
            preds = outputs.argmax(dim=-1)
            correct = preds.eq(y).sum().item()
            total_correct += correct
            total_samples += x.size(0)
            
        # Record trajectory (mean probability of expert 1)
        mean_prob1 = torch.stack([torch.sigmoid(l) for l in lambda_logits.values()]).mean().item()
        coef_trajectory.append(mean_prob1)
        
        # 2. Adaptation Step
        if method_name == 'static':
            # No updates for static
            continue
            
        # Get predictions of active frozen expert
        active_expert = expert_cifar10 if task_id == 0 else expert_svhn
        with torch.no_grad():
            expert_outputs = active_expert(x)
            p_expert = F.softmax(expert_outputs, dim=-1)
            
        # Define self-labeling loss function on a subset of batch indices
        def compute_subset_kl_loss(batch_indices):
            if len(batch_indices) == 0:
                return torch.tensor(0.0, device=device)
            sub_x = x[batch_indices]
            sub_p_expert = p_expert[batch_indices]
            
            # Merged parameters
            sub_merged_params = {}
            for n, p in base_resnet.named_parameters():
                if 'fc' not in n:
                    prob1 = torch.sigmoid(lambda_logits[n])
                    prob2 = 1.0 - prob1
                    sub_merged_params[n] = prob1 * expert1_params[n] + prob2 * expert2_params[n]
                else:
                    sub_merged_params[n] = expert1_params[n] if task_id == 0 else expert2_params[n]
                    
            sub_state_dict = {**sub_merged_params, **merged_buffers}
            sub_merged_outputs = functional_call(base_resnet, sub_state_dict, sub_x)
            sub_log_p_merged = F.log_softmax(sub_merged_outputs, dim=-1)
            
            # KL divergence
            kl = F.kl_div(sub_log_p_merged, sub_p_expert, reduction='batchmean')
            return kl

        # OPR (Optimizer & Parameter Reset) check for PC-Merge and LF-Proj
        if method_name in ['pc_merge', 'lf_proj']:
            # Evaluate full loss first
            full_loss_val = compute_subset_kl_loss(list(range(x.size(0))))
            
            if ema_loss is None:
                ema_loss = full_loss_val.item()
            else:
                # Check for spike indicating task boundary
                if full_loss_val.item() > reset_threshold * ema_loss:
                    # Reset logits to 0.0 (uniform merge)
                    for n in encoder_keys:
                        lambda_logits[n].data.fill_(0.0)
                    # Re-initialize optimizer
                    if optimizer_name == 'sgd':
                        optimizer = torch.optim.SGD(params_list, lr=eta, momentum=0.9)
                    else:
                        optimizer = torch.optim.Adam(params_list, lr=eta)
                    ema_loss = full_loss_val.item()
                else:
                    ema_loss = beta_ema * ema_loss + (1 - beta_ema) * full_loss_val.item()

        # Update implementation for standard_tta and lfwa
        if method_name in ['standard_tta', 'lfwa']:
            loss = compute_subset_kl_loss(list(range(x.size(0))))
            optimizer.zero_grad()
            loss.backward()
            
            # Gradients application
            if method_name == 'lfwa':
                for n, logit in lambda_logits.items():
                    if logit.grad is not None:
                        # Scale learning rate by Fisher sensitivity, clamping to prevent explosion in SGD
                        scale = min(1.0, (joint_fisher[n] + 1e-8) ** (-alpha))
                        logit.grad.data *= scale
            optimizer.step()
            
        # Update implementation for pc_merge and our proposed lf_proj
        elif method_name in ['pc_merge', 'lf_proj']:
            # Group batch samples by predicted class (or pseudo labels) or true labels (oracle)
            if oracle_grouping:
                group_classes = y.cpu().numpy()
            else:
                group_classes = preds.cpu().numpy()
            unique_classes = np.unique(group_classes)
            
            # Compute class-specific gradients
            class_grads = []
            for c in unique_classes:
                class_indices = np.where(group_classes == c)[0].tolist()
                c_loss = compute_subset_kl_loss(class_indices)
                
                optimizer.zero_grad()
                c_loss.backward()
                
                # Store class gradient dictionary
                c_grad_dict = {}
                for n, logit in lambda_logits.items():
                    if logit.grad is not None:
                        c_grad_dict[n] = logit.grad.clone()
                    else:
                        c_grad_dict[n] = torch.zeros_like(logit)
                class_grads.append(c_grad_dict)
                
            if len(class_grads) > 0:
                # Perform gradient projection
                if method_name == 'pc_merge':
                    # Global PCGrad projection
                    projected_grads = pcgrad_project_dict(class_grads)
                    # Sum the projected gradients
                    final_grad = {n: torch.zeros_like(lambda_logits[n]) for n in encoder_keys}
                    for g_dict in projected_grads:
                        for n in encoder_keys:
                            final_grad[n] += g_dict[n]
                else:
                    # LF-Proj (Our Proposed Method): Fisher-weighted gradient projection + learning rate scaling
                    keys = sorted(encoder_keys)
                    num_grads = len(class_grads)
                    
                    # Compute joint Fisher values as a PyTorch tensor on the correct device
                    fisher_tensor = torch.tensor([joint_fisher[k] for k in keys], device=device)
                    proj_strength_tensor = torch.clamp(fisher_tensor / theta_F, max=1.0)
                    
                    # Stack original class gradients to shape (num_grads, num_keys)
                    original_grads_stacked = torch.stack([
                        torch.stack([g_dict[k] for k in keys]) for g_dict in class_grads
                    ])
                    
                    # Initialize projected gradients stacked tensor
                    projected_grads_stacked = original_grads_stacked.clone()
                    
                    for i in range(num_grads):
                        for j in range(num_grads):
                            if i != j:
                                g_i = projected_grads_stacked[i]  # shape: (num_keys,)
                                g_j = original_grads_stacked[j]   # shape: (num_keys,)
                                
                                dot_prod = g_i * g_j              # shape: (num_keys,)
                                conflict = dot_prod < 0           # shape: (num_keys,)
                                
                                g_j_norm = g_j * g_j + 1e-12      # shape: (num_keys,)
                                
                                proj = proj_strength_tensor * (dot_prod / g_j_norm) * g_j
                                projected_grads_stacked[i] = torch.where(conflict, g_i - proj, g_i)
                                
                    # Reconstruct final_grad dict and apply Fisher prior scaling
                    final_grad = {}
                    summed_grads = projected_grads_stacked.sum(dim=0) # shape: (num_keys,)
                    for idx, n in enumerate(keys):
                        scale = min(1.0, (joint_fisher[n] + 1e-8) ** (-alpha))
                        final_grad[n] = summed_grads[idx] * scale
                        
                # Manually write gradients and step optimizer
                optimizer.zero_grad()
                for n, logit in lambda_logits.items():
                    logit.grad = final_grad[n]
                optimizer.step()
                
    acc = total_correct / total_samples * 100
    return acc, coef_trajectory

def pcgrad_project_dict(class_grads):
    keys = sorted(class_grads[0].keys())
    # Flatten each dictionary of gradients into a 1D tensor
    flattened_grads = []
    for g_dict in class_grads:
        flat = torch.stack([g_dict[k] for k in keys])
        flattened_grads.append(flat)
    
    num_grads = len(flattened_grads)
    projected_flat = [g.clone() for g in flattened_grads]
    indices = list(range(num_grads))
    
    for i in indices:
        for j in indices:
            if i != j:
                dot_prod = torch.dot(projected_flat[i], projected_flat[j])
                if dot_prod < 0:
                    g_j_norm = torch.dot(projected_flat[j], projected_flat[j]) + 1e-12
                    projected_flat[i] -= (dot_prod / g_j_norm) * projected_flat[j]
                    
    # Unflatten back to dictionaries
    projected_dicts = []
    for flat in projected_flat:
        p_dict = {}
        for idx, k in enumerate(keys):
            p_dict[k] = flat[idx]
        projected_dicts.append(p_dict)
    return projected_dicts

# Compute default theta_F as the median of joint Fisher
fisher_values = list(joint_fisher.values())
theta_F_val = np.median(fisher_values)
print(f"Median Fisher (theta_F): {theta_F_val:.6f}")

# ==========================================
# 5. Run Evaluations Across All Configurations
# ==========================================
methods = ['static', 'standard_tta', 'lfwa', 'pc_merge', 'lf_proj']
optimizers = ['sgd', 'adam']
streams = {'Alternating': alt_stream, 'Sequential': seq_stream}
learning_rates = {
    'static': [0.0], # Static does not use LR
    'standard_tta': [0.01, 0.1, 1.0],
    'lfwa': [0.01, 0.1, 1.0],
    'pc_merge': [0.01, 0.1, 1.0],
    'lf_proj': [0.01, 0.1, 1.0]
}

results = {}
best_lrs = {}

for opt in optimizers:
    results[opt] = {}
    best_lrs[opt] = {}
    for stream_name, stream in streams.items():
        results[opt][stream_name] = {}
        best_lrs[opt][stream_name] = {}
        print(f"\nEvaluating stream: {stream_name} with optimizer: {opt.upper()} (CLEAN)")
        print("-" * 60)
        
        for method in methods:
            best_acc = 0.0
            best_lr = 0.0
            best_traj = None
            
            lrs_to_test = learning_rates[method]
            for lr in lrs_to_test:
                acc, traj = evaluate_method(
                    method_name=method,
                    optimizer_name=opt,
                    stream=stream,
                    eta=lr,
                    alpha=0.5, # Alpha for LFWA and LF-Proj
                    theta_F=theta_F_val, # Threshold for LF-Proj
                    reset_threshold=2.5 # Reset threshold for PC-Merge / LF-Proj
                )
                print(f"Method: {method:<15} | LR: {lr:<5} | Accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
                    best_traj = traj
                    
            results[opt][stream_name][method] = {
                'accuracy': best_acc,
                'lr': best_lr,
                'trajectory': best_traj
            }
            best_lrs[opt][stream_name][method] = best_lr

# ==========================================
# 5.1 Noisy Evaluations (noise_std = 0.15)
# ==========================================
print("\n" + "="*80)
print("RUNNING EVALUATIONS WITH NOISE LEVEL 0.15")
print("="*80)
results_noise_015 = {}
for opt in optimizers:
    results_noise_015[opt] = {}
    for stream_name, stream in streams.items():
        results_noise_015[opt][stream_name] = {}
        for method in methods:
            best_lr = best_lrs[opt][stream_name][method]
            acc, _ = evaluate_method(
                method_name=method,
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.15
            )
            print(f"Opt: {opt.upper()} | Stream: {stream_name:<11} | Method: {method:<15} | LR: {best_lr:<5} | Accuracy: {acc:.2f}%")
            results_noise_015[opt][stream_name][method] = acc

# ==========================================
# 5.2 Noisy Evaluations (noise_std = 0.30)
# ==========================================
print("\n" + "="*80)
print("RUNNING EVALUATIONS WITH NOISE LEVEL 0.30")
print("="*80)
results_noise_030 = {}
for opt in optimizers:
    results_noise_030[opt] = {}
    for stream_name, stream in streams.items():
        results_noise_030[opt][stream_name] = {}
        for method in methods:
            best_lr = best_lrs[opt][stream_name][method]
            acc, _ = evaluate_method(
                method_name=method,
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.30
            )
            print(f"Opt: {opt.upper()} | Stream: {stream_name:<11} | Method: {method:<15} | LR: {best_lr:<5} | Accuracy: {acc:.2f}%")
            results_noise_030[opt][stream_name][method] = acc

# ==========================================
# 5.3 LF-Proj Alpha Sensitivity Sweep
# ==========================================
print("\n" + "="*80)
print("RUNNING LF-PROJ ALPHA SENSITIVITY SWEEP")
print("="*80)
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
results_alpha_sweep = {}
for opt in optimizers:
    results_alpha_sweep[opt] = {}
    for stream_name, stream in streams.items():
        results_alpha_sweep[opt][stream_name] = {}
        best_lr = best_lrs[opt][stream_name]['lf_proj']
        for alpha in alpha_values:
            acc, _ = evaluate_method(
                method_name='lf_proj',
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=alpha,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.0
            )
            print(f"Opt: {opt.upper()} | Stream: {stream_name:<11} | Alpha: {alpha:<5} | Accuracy: {acc:.2f}%")
            results_alpha_sweep[opt][stream_name][alpha] = acc

# ==========================================
# 5.4 LF-Proj Theta_F Percentile Sweep
# ==========================================
print("\n" + "="*80)
print("RUNNING LF-PROJ THETA_F PERCENTILE SWEEP")
print("="*80)
percentiles = [10, 25, 50, 75, 90]
results_percentile_sweep = {}
fisher_vals = list(joint_fisher.values())
for opt in optimizers:
    results_percentile_sweep[opt] = {}
    for stream_name, stream in streams.items():
        results_percentile_sweep[opt][stream_name] = {}
        best_lr = best_lrs[opt][stream_name]['lf_proj']
        for pct in percentiles:
            theta_F_pct = np.percentile(fisher_vals, pct)
            acc, _ = evaluate_method(
                method_name='lf_proj',
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_pct,
                reset_threshold=2.5,
                noise_std=0.0
            )
            print(f"Opt: {opt.upper()} | Stream: {stream_name:<11} | Percentile: {pct}% (Theta_F: {theta_F_pct:.6f}) | Accuracy: {acc:.2f}%")
            results_percentile_sweep[opt][stream_name][pct] = acc

# ==========================================
# 5.5 LF-Proj Reset Threshold Gamma Sensitivity Sweep
# ==========================================
print("\n" + "="*80)
print("RUNNING LF-PROJ RESET THRESHOLD GAMMA SENSITIVITY SWEEP")
print("="*80)
gamma_values = [1.5, 2.0, 2.5, 3.0, 3.5]
results_gamma_sweep = {}
for opt in optimizers:
    results_gamma_sweep[opt] = {}
    for stream_name, stream in streams.items():
        results_gamma_sweep[opt][stream_name] = {}
        best_lr = best_lrs[opt][stream_name]['lf_proj']
        for gamma in gamma_values:
            acc, _ = evaluate_method(
                method_name='lf_proj',
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=gamma,
                noise_std=0.0
            )
            print(f"Opt: {opt.upper()} | Stream: {stream_name:<11} | Gamma: {gamma:<5} | Accuracy: {acc:.2f}%")
            results_gamma_sweep[opt][stream_name][gamma] = acc

# ==========================================
# 5.6 Oracle vs. Predicted Class Grouping Sweep
# ==========================================
print("\n" + "="*80)
print("RUNNING ORACLE VS. PREDICTED GROUPING SWEEP")
print("="*80)
results_oracle = {}
for opt in optimizers:
    results_oracle[opt] = {}
    for stream_name, stream in streams.items():
        results_oracle[opt][stream_name] = {}
        for method in ['pc_merge', 'lf_proj']:
            best_lr = best_lrs[opt][stream_name][method]
            
            # Predict (Clean)
            acc_pred, _ = evaluate_method(
                method_name=method,
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.0,
                oracle_grouping=False
            )
            # Oracle (Clean)
            acc_oracle, _ = evaluate_method(
                method_name=method,
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.0,
                oracle_grouping=True
            )
            
            # Predict (Noisy 0.30)
            acc_pred_noisy, _ = evaluate_method(
                method_name=method,
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.30,
                oracle_grouping=False
            )
            # Oracle (Noisy 0.30)
            acc_oracle_noisy, _ = evaluate_method(
                method_name=method,
                optimizer_name=opt,
                stream=stream,
                eta=best_lr,
                alpha=0.5,
                theta_F=theta_F_val,
                reset_threshold=2.5,
                noise_std=0.30,
                oracle_grouping=True
            )
            
            print(f"Opt: {opt.upper()} | Stream: {stream_name:<11} | Method: {method:<15} | Clean: Pred={acc_pred:.2f}%, Oracle={acc_oracle:.2f}% | Noisy(0.30): Pred={acc_pred_noisy:.2f}%, Oracle={acc_oracle_noisy:.2f}%")
            results_oracle[opt][stream_name][method] = {
                'clean_pred': acc_pred,
                'clean_oracle': acc_oracle,
                'noisy_pred': acc_pred_noisy,
                'noisy_oracle': acc_oracle_noisy
            }

# ==========================================
# 6. Print Summary Table
# ==========================================
print("\n" + "="*80)
print("                                SUMMARY OF RESULTS")
print("="*80)
print(f"{'Method':<18} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}")
print("-" * 80)
for method in methods:
    sgd_alt = results['sgd']['Alternating'][method]['accuracy']
    sgd_seq = results['sgd']['Sequential'][method]['accuracy']
    adam_alt = results['adam']['Alternating'][method]['accuracy']
    adam_seq = results['adam']['Sequential'][method]['accuracy']
    
    print(f"{method:<18} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%")
print("="*80)

# Write results to results.txt for persistence
with open("results.txt", "w") as f:
    f.write("Test-Time Model Merging Experimental Results & Ablations\n")
    f.write("="*80 + "\n\n")
    
    # Clean Table
    f.write("1. CLEAN DATASET RESULTS (noise_std = 0.0)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Method':<18} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}\n")
    f.write("-" * 80 + "\n")
    for method in methods:
        sgd_alt = results['sgd']['Alternating'][method]['accuracy']
        sgd_seq = results['sgd']['Sequential'][method]['accuracy']
        adam_alt = results['adam']['Alternating'][method]['accuracy']
        adam_seq = results['adam']['Sequential'][method]['accuracy']
        f.write(f"{method:<18} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%\n")
    f.write("="*80 + "\n\n")

    # Noisy 0.15 Table
    f.write("2. NOISY DATASET RESULTS (Gaussian noise_std = 0.15)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Method':<18} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}\n")
    f.write("-" * 80 + "\n")
    for method in methods:
        sgd_alt = results_noise_015['sgd']['Alternating'][method]
        sgd_seq = results_noise_015['sgd']['Sequential'][method]
        adam_alt = results_noise_015['adam']['Alternating'][method]
        adam_seq = results_noise_015['adam']['Sequential'][method]
        f.write(f"{method:<18} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%\n")
    f.write("="*80 + "\n\n")

    # Noisy 0.30 Table
    f.write("3. NOISY DATASET RESULTS (Gaussian noise_std = 0.30)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Method':<18} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}\n")
    f.write("-" * 80 + "\n")
    for method in methods:
        sgd_alt = results_noise_030['sgd']['Alternating'][method]
        sgd_seq = results_noise_030['sgd']['Sequential'][method]
        adam_alt = results_noise_030['adam']['Alternating'][method]
        adam_seq = results_noise_030['adam']['Sequential'][method]
        f.write(f"{method:<18} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%\n")
    f.write("="*80 + "\n\n")

    # Alpha Sweep Table
    f.write("4. LF-PROJ HYPERPARAMETER ABLATION: ALPHA SWEEP (on clean data)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Alpha':<10} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}\n")
    f.write("-" * 80 + "\n")
    for alpha in alpha_values:
        sgd_alt = results_alpha_sweep['sgd']['Alternating'][alpha]
        sgd_seq = results_alpha_sweep['sgd']['Sequential'][alpha]
        adam_alt = results_alpha_sweep['adam']['Alternating'][alpha]
        adam_seq = results_alpha_sweep['adam']['Sequential'][alpha]
        f.write(f"{alpha:<10.1f} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%\n")
    f.write("="*80 + "\n\n")

    # Theta_F Percentile Sweep Table
    f.write("5. LF-PROJ HYPERPARAMETER ABLATION: THETA_F PERCENTILE SWEEP (on clean data)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Percentile':<10} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}\n")
    f.write("-" * 80 + "\n")
    for pct in percentiles:
        sgd_alt = results_percentile_sweep['sgd']['Alternating'][pct]
        sgd_seq = results_percentile_sweep['sgd']['Sequential'][pct]
        adam_alt = results_percentile_sweep['adam']['Alternating'][pct]
        adam_seq = results_percentile_sweep['adam']['Sequential'][pct]
        f.write(f"{str(pct)+'%':<10} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%\n")
    f.write("="*80 + "\n\n")

    # OPR Reset Threshold Gamma Sweep Table
    f.write("6. LF-PROJ HYPERPARAMETER ABLATION: OPR RESET THRESHOLD GAMMA SENSITIVITY SWEEP (on clean data)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Gamma':<10} | {'SGD Alt':<10} | {'SGD Seq':<10} | {'Adam Alt':<10} | {'Adam Seq':<10}\n")
    f.write("-" * 80 + "\n")
    for gamma in gamma_values:
        sgd_alt = results_gamma_sweep['sgd']['Alternating'][gamma]
        sgd_seq = results_gamma_sweep['sgd']['Sequential'][gamma]
        adam_alt = results_gamma_sweep['adam']['Alternating'][gamma]
        adam_seq = results_gamma_sweep['adam']['Sequential'][gamma]
        f.write(f"{gamma:<10.1f} | {sgd_alt:5.2f}%    | {sgd_seq:5.2f}%    | {adam_alt:5.2f}%    | {adam_seq:5.2f}%\n")
    f.write("="*80 + "\n\n")

    # Oracle Sweep Table
    f.write("7. ORACLE VS. PREDICTED CLASS GROUPING COMPARISON (Clean vs Noisy 0.30)\n")
    f.write("="*80 + "\n")
    f.write(f"{'Method/Opt/Stream':<35} | {'Clean Pred':<11} | {'Clean Oracle':<12} | {'Noisy Pred':<11} | {'Noisy Oracle':<12}\n")
    f.write("-" * 80 + "\n")
    for opt in optimizers:
        for stream_name in ['Alternating', 'Sequential']:
            for method in ['pc_merge', 'lf_proj']:
                r = results_oracle[opt][stream_name][method]
                name = f"{method}_{opt}_{stream_name}"
                f.write(f"{name:<35} | {r['clean_pred']:10.2f}% | {r['clean_oracle']:11.2f}% | {r['noisy_pred']:10.2f}% | {r['noisy_oracle']:11.2f}%\n")
    f.write("="*80 + "\n")

print("Results successfully saved to results.txt.")

# ==========================================
# 7. Generate Trajectory Plots
# ==========================================
print("\n" + "="*80)
print("GENERATING TRAJECTORY PLOTS")
print("="*80)
try:
    import matplotlib.pyplot as plt
    
    # Create a nice 1x2 panel plot (SGD and Adam)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    methods_to_plot = ['static', 'standard_tta', 'lfwa', 'pc_merge', 'lf_proj']
    method_labels = {
        'static': 'Static Merging',
        'standard_tta': 'Standard TTA',
        'lfwa': 'LFWA (Stabilized)',
        'pc_merge': 'PC-Merge',
        'lf_proj': 'LF-Proj (Ours)'
    }
    method_colors = {
        'static': '#7f8c8d',
        'standard_tta': '#e74c3c',
        'lfwa': '#e67e22',
        'pc_merge': '#2980b9',
        'lf_proj': '#27ae60'
    }
    method_styles = {
        'static': '--',
        'standard_tta': ':',
        'lfwa': '-.',
        'pc_merge': '-',
        'lf_proj': '-'
    }
    
    for idx, opt in enumerate(['sgd', 'adam']):
        ax = axes[idx]
        ax.axvline(x=16, color='black', linestyle=':', label='Task Boundary (C10 -> SVHN)')
        
        for method in methods_to_plot:
            traj = results[opt]['Sequential'][method]['trajectory']
            if traj is not None:
                steps = np.arange(1, len(traj) + 1)
                ax.plot(steps, traj, label=method_labels[method], 
                        color=method_colors[method], linestyle=method_styles[method], linewidth=2.5)
                
        ax.set_title(f"{opt.upper()} Optimizer", fontsize=14, fontweight='bold')
        ax.set_xlabel("Adaptation Steps (Batches)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Expert 1 (CIFAR-10) Coefficient Weight", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(1, 32)
        ax.set_ylim(-0.05, 1.05)
        
    plt.suptitle("Test-Time Merging Coefficient Trajectories under Sequential Stream (CIFAR-10 -> SVHN)", fontsize=15, fontweight='bold', y=0.98)
    axes[1].legend(loc='lower left', bbox_to_anchor=(1.02, 0.1), fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig("trajectories_sequential.pdf", bbox_inches='tight')
    plt.savefig("trajectories_sequential.png", bbox_inches='tight', dpi=300)
    print("Trajectory plots successfully saved to trajectories_sequential.pdf and trajectories_sequential.png.")
except Exception as e:
    print(f"Error generating plots: {e}")

