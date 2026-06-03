import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import json

# Set seeds for reproducibility
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2026)
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------------------------------------------
# 1. Dataset Loading and Preprocessing
# -------------------------------------------------------------------

def get_datasets(data_dir='./data', sample_train_size=5000):
    # Standard transform: resize to 32x32, convert 1-channel to 3-channel, normalize using ImageNet stats
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Train datasets
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    fashion_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    cifar10_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    # Test datasets
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    # Take random subsets of training data for expert training
    def get_subset(dataset, size):
        indices = np.random.choice(len(dataset), size, replace=False)
        return Subset(dataset, indices)
        
    mnist_train_sub = get_subset(mnist_train, sample_train_size)
    fashion_train_sub = get_subset(fashion_train, sample_train_size)
    cifar10_train_sub = get_subset(cifar10_train, sample_train_size)
    
    return {
        'train': {
            'mnist': mnist_train_sub,
            'fashion': fashion_train_sub,
            'cifar10': cifar10_train_sub
        },
        'test': {
            'mnist': mnist_test,
            'fashion': fashion_test,
            'cifar10': cifar10_test
        }
    }

# -------------------------------------------------------------------
# 2. Model Architecture
# -------------------------------------------------------------------

class MultiTaskResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace classification head with Identity to extract 512-dim features
        self.backbone.fc = nn.Identity()
        
        # Task-specific classification heads
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(512, 10),
            'fashion': nn.Linear(512, 10),
            'cifar10': nn.Linear(512, 10)
        })
        
    def forward(self, x, task_name=None):
        features = self.backbone(x)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

# -------------------------------------------------------------------
# 3. Helper Functions for State Dict Manipulation & Merging
# -------------------------------------------------------------------

def clone_model(model):
    cloned = MultiTaskResNet18(pretrained=False)
    cloned.load_state_dict(model.state_dict())
    return cloned.to(device)

def get_backbone_state_dict(model):
    return {k: v.clone() for k, v in model.backbone.state_dict().items()}

def merge_weights_wa(experts):
    """
    Weight Averaging of the backbones of the experts.
    """
    merged_backbone = {}
    keys = list(experts[0].backbone.state_dict().keys())
    for key in keys:
        tensors = [expert.backbone.state_dict()[key] for expert in experts]
        if tensors[0].dtype.is_floating_point:
            merged_backbone[key] = torch.stack(tensors).mean(dim=0)
        else:
            merged_backbone[key] = tensors[0].clone()
    return merged_backbone

def merge_weights_ta(base_model, experts, scaling_factor=0.3):
    """
    Task Arithmetic merging of backbones.
    theta = theta_0 + lambda * sum(theta_k - theta_0)
    """
    merged_backbone = {}
    base_state = base_model.backbone.state_dict()
    keys = list(base_state.keys())
    
    for key in keys:
        if base_state[key].dtype.is_floating_point:
            task_vectors = []
            for expert in experts:
                update = expert.backbone.state_dict()[key] - base_state[key]
                task_vectors.append(update)
            merged_backbone[key] = base_state[key] + scaling_factor * torch.stack(task_vectors).sum(dim=0)
        else:
            # Non-floating point parameters (e.g. tracking buffers) are kept from base model or averaged
            merged_backbone[key] = base_state[key].clone()
            
    return merged_backbone

# -------------------------------------------------------------------
# 4. Training Expert Models
# -------------------------------------------------------------------

def train_expert(task_name, train_dataset, epochs=5, lr=5e-4, wd=1e-4, batch_size=128):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    model = MultiTaskResNet18(pretrained=True).to(device)
    
    # Freeze other heads, optimize only backbone and task head
    for name, param in model.named_parameters():
        if 'heads' in name and task_name not in name:
            param.requires_grad = False
            
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=wd
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, task_name)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    return model

# -------------------------------------------------------------------
# 5. Hook Management for Calibration
# -------------------------------------------------------------------

def get_bn_modules(model):
    return {name: module for name, module in model.backbone.named_modules() if isinstance(module, nn.BatchNorm2d)}

def collect_activations(model, dataloader, num_samples=128):
    model.eval()
    bn_modules = get_bn_modules(model)
    activations = {name: [] for name in bn_modules.keys()}
    
    def get_hook(name):
        def hook(module, input, output):
            activations[name].append(output.detach().cpu())
        return hook
        
    handles = []
    for name, module in bn_modules.items():
        handles.append(module.register_forward_hook(get_hook(name)))
        
    samples_collected = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            samples_collected += images.size(0)
            if samples_collected >= num_samples:
                break
                
    for handle in handles:
        handle.remove()
        
    # Concatenate along batch dimension up to num_samples
    for name in activations.keys():
        activations[name] = torch.cat(activations[name], dim=0)[:num_samples]
        
    return activations

# -------------------------------------------------------------------
# 6. Implementation of Calibration and Adaptation Methods
# -------------------------------------------------------------------

def compute_lsc_factors(expert_activations, merged_activations):
    """
    LSC scaling: gamma = std(expert) / std(merged)
    """
    scaling_factors = {}
    for name in expert_activations.keys():
        std_expert = torch.std(expert_activations[name]).item()
        std_merged = torch.std(merged_activations[name]).item()
        scaling_factors[name] = std_expert / (std_merged + 1e-5)
    return scaling_factors

def compute_optimal_scaling_factors(expert_activations, merged_activations):
    """
    Optimal scaling: gamma* = E[Y^T X] / E[X^2]
    Representational correlation: rho = E[Y^T X] / (||Y||_2 * ||X||_2)
    """
    optimal_factors = {}
    correlations = {}
    for name in expert_activations.keys():
        Y = expert_activations[name]  # expert
        X = merged_activations[name]  # merged
        
        # Flatten tensors to vector representation
        Y_flat = Y.view(-1)
        X_flat = X.view(-1)
        
        numerator = torch.dot(Y_flat, X_flat).item()
        denom_opt = torch.dot(X_flat, X_flat).item()
        denom_rho = (torch.norm(Y_flat) * torch.norm(X_flat)).item()
        
        optimal_factors[name] = numerator / (denom_opt + 1e-5)
        correlations[name] = numerator / (denom_rho + 1e-5)
        
    return optimal_factors, correlations

def compute_linear_cka(Y, X):
    """
    Computes Centered Linear Kernel Alignment (CKA) between expert representation Y and merged representation X.
    Formulation:
    CKA(Y, X) = HSIC(Y Y^T, X X^T) / sqrt(HSIC(Y Y^T, Y Y^T) HSIC(X X^T, X X^T))
    """
    # Flatten activations to (N, D) where N is batch size (num_samples)
    N = Y.size(0)
    Y_flat = Y.view(N, -1).float()
    X_flat = X.view(N, -1).float()
    
    # Center the features along the batch dimension
    Y_centered = Y_flat - Y_flat.mean(dim=0, keepdim=True)
    X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)
    
    # Linear kernels K = Y Y^T, L = X X^T
    K = torch.matmul(Y_centered, Y_centered.t())
    L = torch.matmul(X_centered, X_centered.t())
    
    # Double centering helper
    def double_center(H):
        one = torch.ones(N, N, device=H.device) / N
        return H - torch.matmul(one, H) - torch.matmul(H, one) + torch.matmul(torch.matmul(one, H), one)
        
    K_centered = double_center(K)
    L_centered = double_center(L)
    
    hsic_KL = torch.sum(K_centered * L_centered).item()
    hsic_KK = torch.sum(K_centered * K_centered).item()
    hsic_LL = torch.sum(L_centered * L_centered).item()
    
    cka = hsic_KL / (np.sqrt(hsic_KK * hsic_LL) + 1e-5)
    return cka

def apply_scaling_hooks(model, scaling_factors):
    bn_modules = get_bn_modules(model)
    handles = []
    
    def get_hook(name):
        def hook(module, input, output):
            return output * scaling_factors[name]
        return hook
        
    for name, module in bn_modules.items():
        if name in scaling_factors:
            handles.append(module.register_forward_hook(get_hook(name)))
    return handles

def run_ntaac(model, joint_dataloader):
    """
    Native Task-Agnostic Activation Calibration (N-TAAC)
    Natively calibrates the BatchNorm modules of the merged backbone by running a forward pass in train() mode
    with all learnable weights/biases frozen and BN momentum set to 1.0.
    """
    calibrated_model = clone_model(model)
    calibrated_model.train()
    
    # Freeze all learnable parameters
    for param in calibrated_model.parameters():
        param.requires_grad = False
        
    # Configure BatchNorm layers
    bn_modules = get_bn_modules(calibrated_model)
    for name, module in bn_modules.items():
        module.momentum = 1.0
        
    # Run a single batch of joint calibration data
    images, _ = next(iter(joint_dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        _ = calibrated_model(images)
        
    calibrated_model.eval()
    return calibrated_model

def adapt_heads_sft(model, task_loaders, epochs=15, lr=1e-3):
    """
    Supervised Head SFT: Fine-tune the classification heads of the frozen merged model.
    """
    adapted_model = clone_model(model)
    # Freeze backbone
    for param in adapted_model.backbone.parameters():
        param.requires_grad = False
        
    # SFT separately for each head
    criterion = nn.CrossEntropyLoss()
    for task_name, loader in task_loaders.items():
        head = adapted_model.heads[task_name]
        # Optimize only the specific task head
        optimizer = optim.AdamW(head.parameters(), lr=lr)
        
        for epoch in range(epochs):
            adapted_model.train()
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = adapted_model(images, task_name)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
    adapted_model.eval()
    return adapted_model

def adapt_heads_tta(model, experts, task_loaders, epochs=15, lr=1e-3):
    """
    Unsupervised Head TTA: Adapt the heads using self-distillation (KL-divergence) from frozen experts.
    """
    adapted_model = clone_model(model)
    # Freeze backbone
    for param in adapted_model.backbone.parameters():
        param.requires_grad = False
        
    # TTA separately for each head
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    for task_name, loader in task_loaders.items():
        expert = experts[task_name]
        expert.eval()
        head = adapted_model.heads[task_name]
        optimizer = optim.AdamW(head.parameters(), lr=lr)
        
        for epoch in range(epochs):
            adapted_model.train()
            for images, _ in loader:
                images = images.to(device)
                optimizer.zero_grad()
                
                # Student predictions
                student_outputs = adapted_model(images, task_name)
                student_log_probs = torch.log_softmax(student_outputs, dim=1)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = expert(images, task_name)
                    teacher_probs = torch.softmax(teacher_outputs, dim=1)
                    
                loss = kl_loss(student_log_probs, teacher_probs)
                loss.backward()
                optimizer.step()
                
    adapted_model.eval()
    return adapted_model

# -------------------------------------------------------------------
# 7. Evaluation Suite
# -------------------------------------------------------------------

def evaluate_model(model, test_datasets, task_scaling_hooks=None):
    model.eval()
    results = {}
    
    for task_name, dataset in test_datasets.items():
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
        
        # Register task-specific scaling hooks if present
        handles = []
        if task_scaling_hooks is not None and task_name in task_scaling_hooks:
            handles = apply_scaling_hooks(model, task_scaling_hooks[task_name])
            
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_name)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        acc = correct / total * 100
        results[task_name] = acc
        
    results['avg'] = np.mean([results[t] for t in test_datasets.keys()])
    return results

# -------------------------------------------------------------------
# 8. Main Pipeline Execution
# -------------------------------------------------------------------

def main():
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Load data
    datasets = get_datasets(sample_train_size=5000)
    test_datasets = datasets['test']
    
    tasks = ['mnist', 'fashion', 'cifar10']
    experts = {}
    
    # 1. Train or load expert models
    for task in tasks:
        ckpt_path = f'./checkpoints/{task}_expert.pt'
        if os.path.exists(ckpt_path):
            print(f"Loading pre-trained expert for {task.upper()} from {ckpt_path}")
            model = MultiTaskResNet18(pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            experts[task] = model
        else:
            experts[task] = train_expert(task, datasets['train'][task], epochs=5, lr=5e-4, wd=1e-4)
            torch.save(experts[task].state_dict(), ckpt_path)
            print(f"Saved expert for {task.upper()} to {ckpt_path}")
            
    # Print expert accuracies
    print("\nEvaluating expert models:")
    expert_results = {}
    for task in tasks:
        loader = DataLoader(test_datasets[task], batch_size=256, shuffle=False)
        correct = 0
        total = 0
        experts[task].eval()
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = experts[task](images, task)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        expert_results[task] = correct / total * 100
    expert_results['avg'] = np.mean([expert_results[t] for t in tasks])
    print(f"Expert accuracies: {expert_results}")
    
    # 2. Extract Calibration Data (N = 128)
    N_cal = 128
    calibration_loaders = {}
    calibration_subsets = {}
    for task in tasks:
        # Create tiny calibration subset
        indices = np.random.choice(len(datasets['train'][task]), N_cal, replace=False)
        subset = Subset(datasets['train'][task], indices)
        calibration_subsets[task] = subset
        calibration_loaders[task] = DataLoader(subset, batch_size=N_cal, shuffle=False)
        
    # Joint calibration dataset for N-TAAC
    joint_indices = []
    joint_dataset_list = []
    for task in tasks:
        joint_dataset_list.append(calibration_subsets[task])
    # Create combined joint dataset
    class JointDataset(torch.utils.data.Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.lengths = [len(d) for d in datasets]
            
        def __len__(self):
            return sum(self.lengths)
            
        def __getitem__(self, idx):
            # Interleave samples
            dataset_idx = idx % len(self.datasets)
            sample_idx = idx // len(self.datasets)
            if sample_idx >= len(self.datasets[dataset_idx]):
                sample_idx = random.randint(0, len(self.datasets[dataset_idx])-1)
            return self.datasets[dataset_idx][sample_idx]
            
    joint_dataset = JointDataset(joint_dataset_list)
    # Batch size is K * N = 384
    joint_loader = DataLoader(joint_dataset, batch_size=len(tasks)*N_cal, shuffle=True)
    
    # Create Base Pretrained Model
    base_model = MultiTaskResNet18(pretrained=True).to(device)
    # Copy heads from experts to base model
    for task in tasks:
        base_model.heads[task].load_state_dict(experts[task].heads[task].state_dict())
        
    # Collect activations for each expert on their calibration data
    expert_acts = {}
    for task in tasks:
        expert_acts[task] = collect_activations(experts[task], calibration_loaders[task], N_cal)
        
    # -------------------------------------------------------------------
    # Evaluation under Weight Averaging (WA) and Task Arithmetic (TA)
    # -------------------------------------------------------------------
    
    results_record = {}
    
    for merge_mode in ['WA', 'TA']:
        print(f"\n=======================================================")
        print(f"Running Experiments for Merge Mode: {merge_mode}")
        print(f"=======================================================")
        
        merged_model = MultiTaskResNet18(pretrained=False).to(device)
        # Load heads from experts
        for task in tasks:
            merged_model.heads[task].load_state_dict(experts[task].heads[task].state_dict())
            
        if merge_mode == 'WA':
            merged_state = merge_weights_wa([experts[t] for t in tasks])
            merged_model.backbone.load_state_dict(merged_state)
        else: # TA with lambda = 0.3
            merged_state = merge_weights_ta(base_model, [experts[t] for t in tasks], scaling_factor=0.3)
            merged_model.backbone.load_state_dict(merged_state)
            
        # Collect merged activations for each task
        merged_acts = {}
        for task in tasks:
            merged_acts[task] = collect_activations(merged_model, calibration_loaders[task], N_cal)
            
        # 3. Compute Calibration Parameters
        lsc_factors = {}
        opt_factors = {}
        correlations = {}
        cka_similarities = {}
        
        for task in tasks:
            lsc_factors[task] = compute_lsc_factors(expert_acts[task], merged_acts[task])
            opt_f, corr = compute_optimal_scaling_factors(expert_acts[task], merged_acts[task])
            opt_factors[task] = opt_f
            correlations[task] = corr
            
            # Compute Centered Linear CKA
            cka_similarities[task] = {}
            for name in expert_acts[task].keys():
                cka_similarities[task][name] = compute_linear_cka(expert_acts[task][name], merged_acts[task][name])
            
        # Print Correlation & Optimal Scaling stats for MNIST & CIFAR10 to verify our theory
        print(f"\n--- Representational Correlation (rho) and Linear CKA for {merge_mode} ---")
        for task in ['mnist', 'cifar10']:
            mean_rho = np.mean(list(correlations[task].values()))
            mean_cka = np.mean(list(cka_similarities[task].values()))
            print(f"Task {task.upper()} average representation correlation (rho): {mean_rho:.4f} | Linear CKA: {mean_cka:.4f}")
            
        # 4. Evaluate Different Pipelines
        pipelines = {}
        
        # Pipeline 1: Uncalibrated baseline (NONE)
        pipelines['NONE'] = evaluate_model(merged_model, test_datasets)
        
        # Pipeline 2: Heuristic LSC
        pipelines['LSC (Heuristic)'] = evaluate_model(merged_model, test_datasets, task_scaling_hooks=lsc_factors)
        
        # Pipeline 3: Mathematically Optimal Scaling (Our proposed)
        pipelines['LSC (Optimal)'] = evaluate_model(merged_model, test_datasets, task_scaling_hooks=opt_factors)
        
        # Pipeline 4: N-TAAC
        ntaac_model = run_ntaac(merged_model, joint_loader)
        pipelines['N-TAAC'] = evaluate_model(ntaac_model, test_datasets)
        
        # Pipeline 5: Head SFT (supervised)
        sft_model = adapt_heads_sft(merged_model, calibration_loaders)
        pipelines['Head SFT'] = evaluate_model(sft_model, test_datasets)
        
        # Pipeline 6: Head TTA (unsupervised)
        tta_model = adapt_heads_tta(merged_model, experts, calibration_loaders)
        pipelines['Head TTA'] = evaluate_model(tta_model, test_datasets)
        
        # Pipeline 7: Unified LSC (Optimal) + Head SFT (Our Unified Supervised Method)
        opt_sft_model = adapt_heads_sft(merged_model, calibration_loaders)
        pipelines['LSC (Optimal) + Head SFT'] = evaluate_model(opt_sft_model, test_datasets, task_scaling_hooks=opt_factors)
        
        # Pipeline 8: Unified LSC (Optimal) + Head TTA (Our Unified Unsupervised Method)
        opt_tta_model = adapt_heads_tta(merged_model, experts, calibration_loaders)
        pipelines['LSC (Optimal) + Head TTA'] = evaluate_model(opt_tta_model, test_datasets, task_scaling_hooks=opt_factors)
        
        # Pipeline 9: Unified N-TAAC + Head SFT
        ntaac_sft_model = adapt_heads_sft(ntaac_model, calibration_loaders)
        pipelines['N-TAAC + Head SFT'] = evaluate_model(ntaac_sft_model, test_datasets)
        
        results_record[merge_mode] = pipelines
        
        # Print summary
        print(f"\nResults for Merge Mode: {merge_mode}")
        print(f"{'Method':<30} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-" * 75)
        for name, res in pipelines.items():
            print(f"{name:<30} | {res['mnist']:<8.2f} | {res['fashion']:<8.2f} | {res['cifar10']:<8.2f} | {res['avg']:<8.2f}")
            
        # 5. Generate Plots to verify the Optimal Activation Scaling Theorem
        layers = list(correlations['mnist'].keys())
        layer_indices = range(len(layers))
        
        # Plot 1: Correlation (rho) and Linear CKA across layers
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(layer_indices, [correlations['mnist'][l] for l in layers], label=r'MNIST ($\rho$)', marker='o', linestyle='-')
        plt.plot(layer_indices, [cka_similarities['mnist'][l] for l in layers], label='MNIST (Linear CKA)', marker='o', linestyle='--')
        plt.plot(layer_indices, [correlations['cifar10'][l] for l in layers], label=r'CIFAR-10 ($\rho$)', marker='s', linestyle='-')
        plt.plot(layer_indices, [cka_similarities['cifar10'][l] for l in layers], label='CIFAR-10 (Linear CKA)', marker='s', linestyle='--')
        plt.xlabel('BatchNorm Layer Index')
        plt.ylabel('Representational Similarity')
        plt.title('Representational Similarity Across Layers')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(layer_indices, [lsc_factors['mnist'][l] for l in layers], label='LSC Heuristic (ratio of stds)', marker='o', linestyle='--')
        plt.plot(layer_indices, [opt_factors['mnist'][l] for l in layers], label='Optimal scaling factor', marker='x', linestyle='-')
        plt.xlabel('BatchNorm Layer Index')
        plt.ylabel('Scaling Factor')
        plt.title('Heuristic vs Optimal Scaling (MNIST)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./plots/theory_verification_{merge_mode}.png')
        plt.close()
        
    # Write results to JSON
    with open('./checkpoints/results.json', 'w') as f:
        json.dump(results_record, f, indent=4)
        
    print("\nMain experiments complete! All results saved to checkpoints/results.json and plots saved to plots/")
    
    # Run hyperparameter sweeps
    run_sweeps(experts, datasets, test_datasets, base_model, tasks)

def run_sweeps(experts, datasets, test_datasets, base_model, tasks):
    print("\n=======================================================")
    print("Running Hyperparameter Sweeps (N and Lambda)")
    print("=======================================================")
    
    # 1. Sweep over calibration budget N
    N_list = [16, 32, 64, 128, 256, 512]
    sweep_N_results = {
        'LSC (Heuristic)': [],
        'LSC (Optimal)': [],
        'N-TAAC': [],
        'LSC (Optimal) + Head TTA': []
    }
    
    class JointDataset(torch.utils.data.Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.lengths = [len(d) for d in datasets]
        def __len__(self):
            return sum(self.lengths)
        def __getitem__(self, idx):
            dataset_idx = idx % len(self.datasets)
            sample_idx = idx // len(self.datasets)
            if sample_idx >= len(self.datasets[dataset_idx]):
                sample_idx = random.randint(0, len(self.datasets[dataset_idx])-1)
            return self.datasets[dataset_idx][sample_idx]

    # We use Weight Averaging for the N sweep
    merged_model_wa = MultiTaskResNet18(pretrained=False).to(device)
    for task in tasks:
        merged_model_wa.heads[task].load_state_dict(experts[task].heads[task].state_dict())
    merged_state_wa = merge_weights_wa([experts[t] for t in tasks])
    merged_model_wa.backbone.load_state_dict(merged_state_wa)

    for N in N_list:
        print(f"\nEvaluating N = {N} ...")
        # Extract calibration loaders
        calibration_loaders = {}
        calibration_subsets = {}
        for task in tasks:
            indices = np.random.choice(len(datasets['train'][task]), N, replace=False)
            subset = Subset(datasets['train'][task], indices)
            calibration_subsets[task] = subset
            calibration_loaders[task] = DataLoader(subset, batch_size=N, shuffle=False)
            
        joint_dataset_list = [calibration_subsets[t] for t in tasks]
        joint_dataset = JointDataset(joint_dataset_list)
        joint_loader = DataLoader(joint_dataset, batch_size=len(tasks)*N, shuffle=True)
        
        # Collect expert activations
        expert_acts = {}
        for task in tasks:
            expert_acts[task] = collect_activations(experts[task], calibration_loaders[task], N)
            
        # Collect merged activations
        merged_acts = {}
        for task in tasks:
            merged_acts[task] = collect_activations(merged_model_wa, calibration_loaders[task], N)
            
        # Compute scaling factors
        lsc_factors = {}
        opt_factors = {}
        for task in tasks:
            lsc_factors[task] = compute_lsc_factors(expert_acts[task], merged_acts[task])
            opt_f, _ = compute_optimal_scaling_factors(expert_acts[task], merged_acts[task])
            opt_factors[task] = opt_f
            
        # Evaluate LSC Heuristic
        acc_lsc_h = evaluate_model(merged_model_wa, test_datasets, task_scaling_hooks=lsc_factors)['avg']
        sweep_N_results['LSC (Heuristic)'].append(acc_lsc_h)
        
        # Evaluate LSC Optimal
        acc_lsc_o = evaluate_model(merged_model_wa, test_datasets, task_scaling_hooks=opt_factors)['avg']
        sweep_N_results['LSC (Optimal)'].append(acc_lsc_o)
        
        # Evaluate N-TAAC
        ntaac_model = run_ntaac(merged_model_wa, joint_loader)
        acc_ntaac = evaluate_model(ntaac_model, test_datasets)['avg']
        sweep_N_results['N-TAAC'].append(acc_ntaac)
        
        # Evaluate LSC (Optimal) + Head TTA
        opt_tta_model = adapt_heads_tta(merged_model_wa, experts, calibration_loaders, epochs=15)
        acc_opt_tta = evaluate_model(opt_tta_model, test_datasets, task_scaling_hooks=opt_factors)['avg']
        sweep_N_results['LSC (Optimal) + Head TTA'].append(acc_opt_tta)
        
        print(f"N={N} Results: LSC(H)={acc_lsc_h:.2f}%, LSC(O)={acc_lsc_o:.2f}%, N-TAAC={acc_ntaac:.2f}%, LSC(O)+HeadTTA={acc_opt_tta:.2f}%")

    # 2. Sweep over Task Arithmetic Lambda
    lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    sweep_lambda_results = {
        'NONE': [],
        'LSC (Heuristic)': [],
        'N-TAAC': [],
        'LSC (Optimal) + Head TTA': []
    }
    
    # We use N=128 for the lambda sweep
    N_lambda = 128
    calibration_loaders_l = {}
    calibration_subsets_l = {}
    for task in tasks:
        indices = np.random.choice(len(datasets['train'][task]), N_lambda, replace=False)
        subset = Subset(datasets['train'][task], indices)
        calibration_subsets_l[task] = subset
        calibration_loaders_l[task] = DataLoader(subset, batch_size=N_lambda, shuffle=False)
        
    joint_dataset_list_l = [calibration_subsets_l[t] for t in tasks]
    joint_dataset_l = JointDataset(joint_dataset_list_l)
    joint_loader_l = DataLoader(joint_dataset_l, batch_size=len(tasks)*N_lambda, shuffle=True)
    
    # Collect expert activations
    expert_acts_l = {}
    for task in tasks:
        expert_acts_l[task] = collect_activations(experts[task], calibration_loaders_l[task], N_lambda)

    for lam in lambda_list:
        print(f"\nEvaluating Lambda = {lam} ...")
        
        # Build TA merged model
        merged_model_ta = MultiTaskResNet18(pretrained=False).to(device)
        for task in tasks:
            merged_model_ta.heads[task].load_state_dict(experts[task].heads[task].state_dict())
        merged_state_ta = merge_weights_ta(base_model, [experts[t] for t in tasks], scaling_factor=lam)
        merged_model_ta.backbone.load_state_dict(merged_state_ta)
        
        # Collect merged activations
        merged_acts_l = {}
        for task in tasks:
            merged_acts_l[task] = collect_activations(merged_model_ta, calibration_loaders_l[task], N_lambda)
            
        # Compute scaling factors
        lsc_factors_l = {}
        opt_factors_l = {}
        for task in tasks:
            lsc_factors_l[task] = compute_lsc_factors(expert_acts_l[task], merged_acts_l[task])
            opt_f, _ = compute_optimal_scaling_factors(expert_acts_l[task], merged_acts_l[task])
            opt_factors_l[task] = opt_f
            
        # Evaluate NONE
        acc_none = evaluate_model(merged_model_ta, test_datasets)['avg']
        sweep_lambda_results['NONE'].append(acc_none)
        
        # Evaluate LSC Heuristic
        acc_lsc_h = evaluate_model(merged_model_ta, test_datasets, task_scaling_hooks=lsc_factors_l)['avg']
        sweep_lambda_results['LSC (Heuristic)'].append(acc_lsc_h)
        
        # Evaluate N-TAAC
        ntaac_model = run_ntaac(merged_model_ta, joint_loader_l)
        acc_ntaac = evaluate_model(ntaac_model, test_datasets)['avg']
        sweep_lambda_results['N-TAAC'].append(acc_ntaac)
        
        # Evaluate LSC (Optimal) + Head TTA
        opt_tta_model = adapt_heads_tta(merged_model_ta, experts, calibration_loaders_l, epochs=15)
        acc_opt_tta = evaluate_model(opt_tta_model, test_datasets, task_scaling_hooks=opt_factors_l)['avg']
        sweep_lambda_results['LSC (Optimal) + Head TTA'].append(acc_opt_tta)
        
        print(f"Lambda={lam} Results: NONE={acc_none:.2f}%, LSC(H)={acc_lsc_h:.2f}%, N-TAAC={acc_ntaac:.2f}%, LSC(O)+HeadTTA={acc_opt_tta:.2f}%")

    # Save results to json
    sweeps_record = {
        'N_sweep': {
            'N_list': N_list,
            'results': sweep_N_results
        },
        'lambda_sweep': {
            'lambda_list': lambda_list,
            'results': sweep_lambda_results
        }
    }
    with open('./checkpoints/results_sweeps.json', 'w') as f:
        json.dump(sweeps_record, f, indent=4)
        
    # Generate Plots
    # Plot N sweep
    plt.figure(figsize=(6, 5))
    for name, accs in sweep_N_results.items():
        plt.plot(N_list, accs, marker='o', label=name)
    plt.xlabel('Calibration Sample Size (N)')
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Effect of Calibration Sample Size (N) under WA')
    plt.xscale('log')
    plt.xticks(N_list, N_list)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sweep_N.png')
    plt.close()
    
    # Plot Lambda sweep
    plt.figure(figsize=(6, 5))
    for name, accs in sweep_lambda_results.items():
        plt.plot(lambda_list, accs, marker='o', label=name)
    plt.xlabel('Task Arithmetic Scaling Factor (lambda)')
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Effect of Task Arithmetic Scaling Factor (lambda)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sweep_lambda.png')
    plt.close()
    
    print("\nHyperparameter sweeps complete! Sweeps results saved to checkpoints/results_sweeps.json and plots saved to plots/")

if __name__ == '__main__':
    main()
