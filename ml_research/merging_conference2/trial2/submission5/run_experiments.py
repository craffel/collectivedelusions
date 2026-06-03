import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import copy
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on cluster nodes
    torch.backends.cudnn.enabled = False

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", 5))
LR = 5e-4
WEIGHT_DECAY = 1e-4
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", 128))

# Custom dataset wrapper to handle grayscale to 3-channel conversion and resizing
class GrayscaleToRGBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convert PIL Image to RGB if it is not already RGB
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Data preprocessing and loaders
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    # MNIST
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test_raw = datasets.MNIST(root='./data', train=False, download=True)
    mnist_train = GrayscaleToRGBDataset(mnist_train_raw, transform_rgb)
    mnist_test = GrayscaleToRGBDataset(mnist_test_raw, transform_rgb)

    # Fashion-MNIST
    fmnist_train_raw = datasets.FashionMNIST(root='./data', train=True, download=True)
    fmnist_test_raw = datasets.FashionMNIST(root='./data', train=False, download=True)
    fmnist_train = GrayscaleToRGBDataset(fmnist_train_raw, transform_rgb)
    fmnist_test = GrayscaleToRGBDataset(fmnist_test_raw, transform_rgb)

    # CIFAR-10
    cifar_train_raw = datasets.CIFAR10(root='./data', train=True, download=True)
    cifar_test_raw = datasets.CIFAR10(root='./data', train=False, download=True)
    cifar_train = GrayscaleToRGBDataset(cifar_train_raw, transform_rgb)
    cifar_test = GrayscaleToRGBDataset(cifar_test_raw, transform_rgb)

    # For rapid and stable training on CPU/GPU, we can subset training datasets
    # Let's use 5000 samples for fine-tuning each expert
    train_subset_size = int(os.environ.get("TRAIN_SUBSET_SIZE", 5000))
    
    mnist_train_sub = Subset(mnist_train, np.random.choice(len(mnist_train), train_subset_size, replace=False))
    fmnist_train_sub = Subset(fmnist_train, np.random.choice(len(fmnist_train), train_subset_size, replace=False))
    cifar_train_sub = Subset(cifar_train, np.random.choice(len(cifar_train), train_subset_size, replace=False))

    # Support dry-run subsetting for test set to ensure super fast CPU dry-run
    if os.environ.get("DRY_RUN", "False") == "True":
        mnist_test = Subset(mnist_test, np.arange(100))
        fmnist_test = Subset(fmnist_test, np.arange(100))
        cifar_test = Subset(cifar_test, np.arange(100))

    loaders = {
        'MNIST': {
            'train': DataLoader(mnist_train_sub, batch_size=BATCH_SIZE, shuffle=True),
            'test': DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False),
            'cal': DataLoader(Subset(mnist_train, np.arange(NUM_CALIBRATION_SAMPLES)), batch_size=NUM_CALIBRATION_SAMPLES, shuffle=False)
        },
        'FashionMNIST': {
            'train': DataLoader(fmnist_train_sub, batch_size=BATCH_SIZE, shuffle=True),
            'test': DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False),
            'cal': DataLoader(Subset(fmnist_train, np.arange(NUM_CALIBRATION_SAMPLES)), batch_size=NUM_CALIBRATION_SAMPLES, shuffle=False)
        },
        'CIFAR10': {
            'train': DataLoader(cifar_train_sub, batch_size=BATCH_SIZE, shuffle=True),
            'test': DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=False),
            'cal': DataLoader(Subset(cifar_train, np.arange(NUM_CALIBRATION_SAMPLES)), batch_size=NUM_CALIBRATION_SAMPLES, shuffle=False)
        }
    }
    return loaders

# Define Multi-Task Architecture
class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove the default classification head
        
        # Task-specific classification heads
        self.heads = nn.ModuleDict({
            'MNIST': nn.Linear(self.num_features, NUM_CLASSES),
            'FashionMNIST': nn.Linear(self.num_features, NUM_CLASSES),
            'CIFAR10': nn.Linear(self.num_features, NUM_CLASSES)
        })

    def forward(self, x, task):
        features = self.backbone(x)
        return self.heads[task](features)

# Fine-tuning function
def train_expert(model, dataloaders, task, epochs=EPOCHS):
    print(f"--- Fine-tuning Expert for Task: {task} ---")
    model.train()
    # Freeze other heads, keep only backbone and task head trainable
    for name, param in model.named_parameters():
        if "heads" in name:
            if task in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in dataloaders[task]['train']:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, task)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    return model

# Evaluation function
def evaluate_model(model, dataloaders, task):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloaders[task]['test']:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs, task)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.0 * correct / total
    return acc

# Helper to clone models
def clone_model(model):
    return copy.deepcopy(model)

# Dictionary of all BatchNorm layers to hook
def get_bn_layers(model):
    bn_layers = {}
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers[name] = module
    return bn_layers

# Robust eigenvalue decomposition for symmetric matrices
def robust_eigen_decomp(cov, epsilon=1e-5):
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    except torch._C._LinAlgError:
        # If eigh fails due to extreme ill-conditioning (e.g. when representation collapse occurs),
        # we add a stronger diagonal perturbation to force convergence.
        cov_reg = cov + 1e-3 * torch.eye(cov.shape[0], device=cov.device)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)
        except torch._C._LinAlgError:
            # Absolute fallback: treat as diagonal
            var = torch.clamp(torch.diag(cov), min=1e-6)
            eigenvalues = var
            eigenvectors = torch.eye(cov.shape[0], device=cov.device)
    return eigenvalues, eigenvectors

# Optimal Diagonal Calibration Optimization Function
def optimize_diagonal_calibration(cov_orig, cov_merged, lr=1e-2, steps=200):
    # cov_orig: [C, C]
    # cov_merged: [C, C]
    # Find d (shape [C]) to minimize || diag(d) @ cov_merged @ diag(d) - cov_orig ||_F^2
    # Initialize d with standard diagonal calibration: d = sqrt(diag(cov_orig) / diag(cov_merged))
    device = cov_orig.device
    d_init = torch.sqrt(torch.clamp(torch.diag(cov_orig), min=1e-6) / torch.clamp(torch.diag(cov_merged), min=1e-6))
    
    # We optimize d in log-space to ensure it remains positive
    log_d = torch.nn.Parameter(torch.log(d_init).clone().detach().to(device).requires_grad_(True))
    optimizer = torch.optim.Adam([log_d], lr=lr)
    
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            d = torch.exp(log_d)
            # E = D @ cov_merged @ D - cov_orig
            E = (d.unsqueeze(1) * cov_merged) * d.unsqueeze(0) - cov_orig
            loss = torch.sum(E ** 2)
            loss.backward()
            optimizer.step()
        
    return torch.exp(log_d).detach()

# Hook classes for Calibration
class CalibrationHook:
    def __init__(self, name, mode='diagonal', alpha=0.5, epsilon=1e-5):
        self.name = name
        self.mode = mode
        self.alpha = alpha
        self.epsilon = epsilon
        self.recording = False
        self.calibrating = False
        
        # Statistics
        self.mean_orig = None
        self.var_orig = None      # For diagonal
        self.cov_orig_sqrt = None # For multivariate
        self.cov_orig = None      # Full covariance of expert
        
        self.mean_merged = None
        self.var_merged = None
        self.cov_merged = None
        self.cov_merged_inv_sqrt = None
        self.opt_d = None
        self.M_reg = None
        
        # Distance metric (for theoretical analysis)
        self.covariance_distance_before = None
        self.covariance_distance_after = None

    def __call__(self, module, input, output):
        # output shape: [B, C, H, W]
        B, C, H, W = output.shape
        Y = output.permute(0, 2, 3, 1).reshape(-1, C) # Shape: [B*H*W, C]
        
        if self.recording:
            # We are recording the original expert statistics
            mean = Y.mean(dim=0) # [C]
            self.mean_orig = mean.detach()
            
            # Always record diagonal variance
            var = Y.var(dim=0, unbiased=False) # [C]
            self.var_orig = var.detach()
            
            # Always record multivariate covariance matrix
            Y_centered = Y - mean.unsqueeze(0)
            cov = (Y_centered.T @ Y_centered) / (Y_centered.shape[0] - 1) # [C, C]
            self.cov_orig = cov.detach()
            
            # Eigenvalue decomposition of symmetric matrix for square root
            eigenvalues, eigenvectors = robust_eigen_decomp(cov, self.epsilon)
            eigenvalues = torch.clamp(eigenvalues, min=0.0)
            S_sqrt = torch.sqrt(eigenvalues + self.epsilon)
            self.cov_orig_sqrt = (eigenvectors @ torch.diag(S_sqrt) @ eigenvectors.T).detach()
                
        elif self.calibrating:
            # We are calibrating the merged model's activations
            if self.mean_merged is None:
                # First pass: record merged model's statistics on calibration set
                mean = Y.mean(dim=0)
                self.mean_merged = mean.detach()
                
                if self.mode == 'diagonal':
                    var = Y.var(dim=0, unbiased=False)
                    self.var_merged = var.detach()
                elif self.mode == 'optimal_diagonal':
                    Y_centered = Y - mean.unsqueeze(0)
                    cov = (Y_centered.T @ Y_centered) / (Y_centered.shape[0] - 1)
                    self.cov_merged = cov.detach()
                    cov_orig_val = self.cov_orig if self.cov_orig is not None else (self.cov_orig_sqrt @ self.cov_orig_sqrt)
                    self.opt_d = optimize_diagonal_calibration(cov_orig_val, self.cov_merged, steps=200)
                elif self.mode == 'regularized_multivariate':
                    Y_centered = Y - mean.unsqueeze(0)
                    cov = (Y_centered.T @ Y_centered) / (Y_centered.shape[0] - 1)
                    self.cov_merged = cov.detach()
                    
                    cov_orig_val = self.cov_orig if self.cov_orig is not None else (self.cov_orig_sqrt @ self.cov_orig_sqrt)
                    
                    # Compute regularized covariance matrices for both orig and merged
                    diag_orig = torch.diag(torch.clamp(torch.diag(cov_orig_val), min=1e-6))
                    cov_orig_reg = (1.0 - self.alpha) * cov_orig_val + self.alpha * diag_orig
                    
                    diag_merged = torch.diag(torch.clamp(torch.diag(self.cov_merged), min=1e-6))
                    cov_merged_reg = (1.0 - self.alpha) * self.cov_merged + self.alpha * diag_merged
                    
                    # Compute Square Roots of regularized matrices
                    eigenvalues_orig, eigenvectors_orig = robust_eigen_decomp(cov_orig_reg, self.epsilon)
                    eigenvalues_orig = torch.clamp(eigenvalues_orig, min=0.0)
                    S_sqrt_orig = torch.sqrt(eigenvalues_orig + self.epsilon)
                    cov_orig_sqrt_reg = eigenvectors_orig @ torch.diag(S_sqrt_orig) @ eigenvectors_orig.T
                    
                    eigenvalues_merged, eigenvectors_merged = robust_eigen_decomp(cov_merged_reg, self.epsilon)
                    eigenvalues_merged = torch.clamp(eigenvalues_merged, min=0.0)
                    S_inv_sqrt_merged = 1.0 / torch.sqrt(eigenvalues_merged + self.epsilon)
                    cov_merged_inv_sqrt_reg = eigenvectors_merged @ torch.diag(S_inv_sqrt_merged) @ eigenvectors_merged.T
                    
                    self.M_reg = (cov_orig_sqrt_reg @ cov_merged_inv_sqrt_reg).detach()
                else: # multivariate
                    Y_centered = Y - mean.unsqueeze(0)
                    cov = (Y_centered.T @ Y_centered) / (Y_centered.shape[0] - 1)
                    
                    eigenvalues, eigenvectors = robust_eigen_decomp(cov, self.epsilon)
                    eigenvalues = torch.clamp(eigenvalues, min=0.0)
                    S_inv_sqrt = 1.0 / torch.sqrt(eigenvalues + self.epsilon)
                    self.cov_merged_inv_sqrt = (eigenvectors @ torch.diag(S_inv_sqrt) @ eigenvectors.T).detach()
                    
                    # Compute theoretical metric: distance between original and merged covariance
                    # Record the original covariance matrix for distance check
                    cov_merged = cov.detach()
                    
                    # Recalculate original covariance matrix
                    cov_orig = self.cov_orig_sqrt @ self.cov_orig_sqrt
                    self.covariance_distance_before = torch.norm(cov_orig - cov_merged, p='fro').item() / torch.norm(cov_orig, p='fro').item()

            # Apply transformation
            if self.mode == 'diagonal':
                # Apply channel-wise (diagonal) rescaling
                scale = torch.sqrt((self.var_orig + self.epsilon) / (self.var_merged + self.epsilon))
                reshaped_mean_merged = self.mean_merged.view(1, C, 1, 1)
                reshaped_scale = scale.view(1, C, 1, 1)
                reshaped_mean_orig = self.mean_orig.view(1, C, 1, 1)
                
                calibrated_output = (output - reshaped_mean_merged) * reshaped_scale + reshaped_mean_orig
                
                # Check covariance after calibration if we want to measure
                if self.covariance_distance_after is None:
                    Y_cal = calibrated_output.permute(0, 2, 3, 1).reshape(-1, C)
                    mean_cal = Y_cal.mean(dim=0)
                    Y_cal_centered = Y_cal - mean_cal.unsqueeze(0)
                    cov_cal = (Y_cal_centered.T @ Y_cal_centered) / (Y_cal_centered.shape[0] - 1)
                    cov_orig = self.cov_orig_sqrt @ self.cov_orig_sqrt if self.cov_orig_sqrt is not None else torch.diag(self.var_orig)
                    self.covariance_distance_after = torch.norm(cov_orig - cov_cal, p='fro').item() / torch.norm(cov_orig, p='fro').item()
                    
                return calibrated_output
            elif self.mode == 'optimal_diagonal':
                reshaped_mean_merged = self.mean_merged.view(1, C, 1, 1)
                reshaped_scale = self.opt_d.view(1, C, 1, 1)
                reshaped_mean_orig = self.mean_orig.view(1, C, 1, 1)
                
                calibrated_output = (output - reshaped_mean_merged) * reshaped_scale + reshaped_mean_orig
                
                if self.covariance_distance_after is None:
                    Y_cal = calibrated_output.permute(0, 2, 3, 1).reshape(-1, C)
                    mean_cal = Y_cal.mean(dim=0)
                    Y_cal_centered = Y_cal - mean_cal.unsqueeze(0)
                    cov_cal = (Y_cal_centered.T @ Y_cal_centered) / (Y_cal_centered.shape[0] - 1)
                    cov_orig_val = self.cov_orig if self.cov_orig is not None else (self.cov_orig_sqrt @ self.cov_orig_sqrt)
                    self.covariance_distance_after = torch.norm(cov_orig_val - cov_cal, p='fro').item() / torch.norm(cov_orig_val, p='fro').item()
                    
                return calibrated_output
            elif self.mode == 'regularized_multivariate':
                # Apply transformation
                Y_centered = Y - self.mean_merged.unsqueeze(0)
                Y_calibrated = Y_centered @ self.M_reg.T + self.mean_orig.unsqueeze(0)
                calibrated_output = Y_calibrated.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
                if self.covariance_distance_after is None:
                    Y_cal = Y_calibrated
                    mean_cal = Y_cal.mean(dim=0)
                    Y_cal_centered = Y_cal - mean_cal.unsqueeze(0)
                    cov_cal = (Y_cal_centered.T @ Y_cal_centered) / (Y_cal_centered.shape[0] - 1)
                    cov_orig_val = self.cov_orig if self.cov_orig is not None else (self.cov_orig_sqrt @ self.cov_orig_sqrt)
                    self.covariance_distance_after = torch.norm(cov_orig_val - cov_cal, p='fro').item() / torch.norm(cov_orig_val, p='fro').item()
                    
                return calibrated_output
            else:
                # Apply multivariate covariance calibration
                # Y_calibrated = (Y - mean_merged) @ inv_sqrt_merged.T @ sqrt_orig.T + mean_orig
                # Transform is: M = cov_orig_sqrt @ cov_merged_inv_sqrt
                M = self.cov_orig_sqrt @ self.cov_merged_inv_sqrt
                
                # Apply to Y
                Y_centered = Y - self.mean_merged.unsqueeze(0)
                Y_calibrated = Y_centered @ M.T + self.mean_orig.unsqueeze(0)
                
                # Reshape back to [B, C, H, W]
                calibrated_output = Y_calibrated.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
                if self.covariance_distance_after is None:
                    # Let's verify that after M-CAC, covariance distance is extremely close to 0!
                    cov_orig = self.cov_orig_sqrt @ self.cov_orig_sqrt
                    # Compute calibrated covariance
                    Y_cal = Y_calibrated
                    mean_cal = Y_cal.mean(dim=0)
                    Y_cal_centered = Y_cal - mean_cal.unsqueeze(0)
                    cov_cal = (Y_cal_centered.T @ Y_cal_centered) / (Y_cal_centered.shape[0] - 1)
                    self.covariance_distance_after = torch.norm(cov_orig - cov_cal, p='fro').item() / torch.norm(cov_orig, p='fro').item()
                    
                return calibrated_output
        return output

# Setup hooks on all BN layers
def register_calibration_hooks(model, mode='diagonal', alpha=0.5):
    bn_layers = get_bn_layers(model)
    hooks = {}
    handles = []
    for name, layer in bn_layers.items():
        hook = CalibrationHook(name, mode=mode, alpha=alpha)
        handle = layer.register_forward_hook(hook)
        hooks[name] = hook
        handles.append(handle)
    return hooks, handles

# Model Merging Algorithms
def merge_weight_averaging(models_dict):
    # simple arithmetic mean of backbone parameters
    merged = clone_model(list(models_dict.values())[0])
    state_dicts = [m.state_dict() for m in models_dict.values()]
    merged_state = merged.state_dict()
    
    for key in merged_state.keys():
        if "backbone" in key:
            if torch.is_floating_point(merged_state[key]):
                merged_state[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
            else:
                merged_state[key] = state_dicts[0][key]
        else:
            # Classification heads remain expert-specific, copy them directly from original experts
            for task, model in models_dict.items():
                if f"heads.{task}" in key:
                    merged_state[key] = model.state_dict()[key]
    merged.load_state_dict(merged_state)
    return merged

def merge_task_arithmetic(base_model, models_dict, scaling_factor=0.4):
    # base model weights + scaling_factor * sum(task vectors)
    merged = clone_model(base_model)
    base_state = base_model.state_dict()
    expert_states = {task: m.state_dict() for task, m in models_dict.items()}
    merged_state = merged.state_dict()
    
    for key in merged_state.keys():
        if "backbone" in key:
            if torch.is_floating_point(merged_state[key]):
                task_vectors = []
                for task in models_dict.keys():
                    tv = expert_states[task][key] - base_state[key]
                    task_vectors.append(tv)
                merged_state[key] = base_state[key] + scaling_factor * torch.stack(task_vectors).sum(dim=0)
            else:
                merged_state[key] = base_state[key]
        else:
            # Classification heads remain expert-specific
            for task, model in models_dict.items():
                if f"heads.{task}" in key:
                    merged_state[key] = model.state_dict()[key]
    merged.load_state_dict(merged_state)
    return merged

# Main Experiment runner
def main():
    print("=========================================")
    print("Starting Model Merging & Calibration Experiment")
    print("=========================================")
    
    # 1. Load Data
    loaders = get_dataloaders()
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10']
    
    # 2. Instantiate and Save Pretrained Base Model
    base_model = MultiTaskResNet18().to(DEVICE)
    print("Pretrained base model initialized.")
    
    # 3. Train task-specific experts
    experts = {}
    for task in tasks:
        # Start from a clean pretrained model for each expert
        expert_model = clone_model(base_model)
        expert_model = train_expert(expert_model, loaders, task, epochs=EPOCHS)
        experts[task] = expert_model
        
    # Evaluate experts
    print("\n--- Evaluating Individual Expert Performance (Upper Bounds) ---")
    expert_accs = {}
    for task in tasks:
        acc = evaluate_model(experts[task], loaders, task)
        expert_accs[task] = acc
        print(f"Expert {task} Accuracy: {acc:.2f}%")
        
    avg_expert_acc = np.mean(list(expert_accs.values()))
    print(f"Average Expert Accuracy: {avg_expert_acc:.2f}%")
    
    # 4. Calibration Statistics Collection (Offline Phase)
    # Register hooks on experts to record original statistics
    expert_hooks = {}
    expert_handles = {}
    for task in tasks:
        expert_hooks[task], expert_handles[task] = register_calibration_hooks(experts[task], mode='multivariate')
        
        # Turn on recording
        for hook in expert_hooks[task].values():
            hook.recording = True
            
        # Run one calibration batch forward pass to collect statistics
        imgs, _ = next(iter(loaders[task]['cal']))
        imgs = imgs.to(DEVICE)
        experts[task](imgs, task)
        
        # Turn off recording and remove hooks from experts
        for handle in expert_handles[task]:
            handle.remove()
        for hook in expert_hooks[task].values():
            hook.recording = False
            
    print("\nRecorded original activation statistics (means and covariance matrices) for all experts.")

    # 5. Define Merging & Calibration Configurations to Evaluate
    # We sweep scaling factors for Task Arithmetic to find the peak performance
    scaling_factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    
    results = []
    
    # Define Task-Conditional Evaluation inside main()
    def evaluate_with_task_conditional_calibration(model, hooks, mode='diagonal', alpha=0.5):
        accs = {}
        cov_dist_before = []
        cov_dist_after = []
        
        for task in tasks:
            # 1. Setup task-specific calibration targets in hooks
            for name, hook in hooks.items():
                hook.mode = mode
                hook.alpha = alpha
                hook.mean_orig = expert_hooks[task][name].mean_orig.clone()
                hook.var_orig = expert_hooks[task][name].var_orig.clone()
                hook.cov_orig_sqrt = expert_hooks[task][name].cov_orig_sqrt.clone()
                if hasattr(expert_hooks[task][name], 'cov_orig') and expert_hooks[task][name].cov_orig is not None:
                    hook.cov_orig = expert_hooks[task][name].cov_orig.clone()
                
                # Reset merged model's recorded statistics for this new model/task
                hook.mean_merged = None
                hook.var_merged = None
                hook.cov_merged = None
                hook.cov_merged_inv_sqrt = None
                hook.opt_d = None
                hook.M_reg = None
                hook.calibrating = True
                
            # 2. Run calibration forward pass to record merged statistics on task k
            model.eval()
            with torch.no_grad():
                imgs, _ = next(iter(loaders[task]['cal']))
                imgs = imgs.to(DEVICE)
                model(imgs, task)
                
            # 3. Evaluate on task k's test set
            accs[task] = evaluate_model(model, loaders, task)
            
            # Record average representation restoration metric across layers
            for hook in hooks.values():
                if hook.covariance_distance_before is not None:
                    cov_dist_before.append(hook.covariance_distance_before)
                if hook.covariance_distance_after is not None:
                    cov_dist_after.append(hook.covariance_distance_after)
                    
        avg_dist_before = np.mean(cov_dist_before) if cov_dist_before else 0.0
        avg_dist_after = np.mean(cov_dist_after) if cov_dist_after else 0.0
        return accs, avg_dist_before, avg_dist_after

    # --- Weight Averaging (WA) Experiments ---
    print("\n=== Evaluating Weight Averaging (WA) Merging ===")
    wa_model = merge_weight_averaging(experts)
    
    # Evaluate baseline WA (uncalibrated)
    wa_baseline_accs = {}
    for task in tasks:
        wa_baseline_accs[task] = evaluate_model(wa_model, loaders, task)
    print(f"WA Baseline Accuracies: {wa_baseline_accs} | Avg: {np.mean(list(wa_baseline_accs.values())):.2f}%")
    results.append({
        'Method': 'Weight Averaging',
        'Scale': 0.0,
        'Calibration': 'None',
        'Accuracies': wa_baseline_accs,
        'Avg': np.mean(list(wa_baseline_accs.values()))
    })
    
    # Evaluate WA + TCAC (Diagonal)
    print("\nRunning WA + TCAC (Diagonal)...")
    wa_tcac_model = clone_model(wa_model)
    tcac_hooks, tcac_handles = register_calibration_hooks(wa_tcac_model, mode='diagonal')
    wa_tcac_accs, tcac_dist_bef, tcac_dist_aft = evaluate_with_task_conditional_calibration(wa_tcac_model, tcac_hooks, mode='diagonal')
    for handle in tcac_handles:
        handle.remove()
    print(f"WA + TCAC Accuracies: {wa_tcac_accs} | Avg: {np.mean(list(wa_tcac_accs.values())):.2f}%")
    print(f"TCAC Covariance Distortion - Before: {tcac_dist_bef:.4f} | After: {tcac_dist_aft:.4f}")
    results.append({
        'Method': 'Weight Averaging',
        'Scale': 0.0,
        'Calibration': 'TCAC (Diagonal)',
        'Accuracies': wa_tcac_accs,
        'Avg': np.mean(list(wa_tcac_accs.values())),
        'DistBefore': tcac_dist_bef,
        'DistAfter': tcac_dist_aft
    })

    # Evaluate WA + M-CAC (Multivariate)
    print("\nRunning WA + M-CAC (Multivariate)...")
    wa_mcac_model = clone_model(wa_model)
    mcac_hooks, mcac_handles = register_calibration_hooks(wa_mcac_model, mode='multivariate')
    wa_mcac_accs, mcac_dist_bef, mcac_dist_aft = evaluate_with_task_conditional_calibration(wa_mcac_model, mcac_hooks, mode='multivariate')
    for handle in mcac_handles:
        handle.remove()
    print(f"WA + M-CAC Accuracies: {wa_mcac_accs} | Avg: {np.mean(list(wa_mcac_accs.values())):.2f}%")
    print(f"M-CAC Covariance Distortion - Before: {mcac_dist_bef:.4f} | After: {mcac_dist_aft:.4f}")
    results.append({
        'Method': 'Weight Averaging',
        'Scale': 0.0,
        'Calibration': 'M-CAC (Multivariate)',
        'Accuracies': wa_mcac_accs,
        'Avg': np.mean(list(wa_mcac_accs.values())),
        'DistBefore': mcac_dist_bef,
        'DistAfter': mcac_dist_aft
    })

    # Sweep alpha on R-MCAC
    alpha_values = [0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
    best_alpha = 0.95
    best_alpha_acc = 0.0
    for alpha in alpha_values:
        print(f"\nRunning WA + R-MCAC (Regularized Multivariate, alpha={alpha})...")
        wa_rmcac_model = clone_model(wa_model)
        rmcac_hooks, rmcac_handles = register_calibration_hooks(wa_rmcac_model, mode='regularized_multivariate', alpha=alpha)
        wa_rmcac_accs, rmcac_dist_bef, rmcac_dist_aft = evaluate_with_task_conditional_calibration(wa_rmcac_model, rmcac_hooks, mode='regularized_multivariate', alpha=alpha)
        for handle in rmcac_handles:
            handle.remove()
        avg_acc = np.mean(list(wa_rmcac_accs.values()))
        print(f"WA + R-MCAC (alpha={alpha}) Accuracies: {wa_rmcac_accs} | Avg: {avg_acc:.2f}%")
        print(f"R-MCAC Covariance Distortion - Before: {rmcac_dist_bef:.4f} | After: {rmcac_dist_aft:.4f}")
        results.append({
            'Method': 'Weight Averaging',
            'Scale': 0.0,
            'Calibration': f'R-MCAC (alpha={alpha})',
            'Accuracies': wa_rmcac_accs,
            'Avg': avg_acc,
            'DistBefore': rmcac_dist_bef,
            'DistAfter': rmcac_dist_aft
        })
        if avg_acc > best_alpha_acc:
            best_alpha_acc = avg_acc
            best_alpha = alpha

    print(f"\nBest alpha found for R-MCAC: {best_alpha} with accuracy {best_alpha_acc:.2f}%")

    # Evaluate WA + ODC (Optimal Diagonal Calibration)
    print("\nRunning WA + ODC (Optimal Diagonal Calibration)...")
    wa_odc_model = clone_model(wa_model)
    odc_hooks, odc_handles = register_calibration_hooks(wa_odc_model, mode='optimal_diagonal')
    wa_odc_accs, odc_dist_bef, odc_dist_aft = evaluate_with_task_conditional_calibration(wa_odc_model, odc_hooks, mode='optimal_diagonal')
    for handle in odc_handles:
        handle.remove()
    print(f"WA + ODC Accuracies: {wa_odc_accs} | Avg: {np.mean(list(wa_odc_accs.values())):.2f}%")
    print(f"ODC Covariance Distortion - Before: {odc_dist_bef:.4f} | After: {odc_dist_aft:.4f}")
    results.append({
        'Method': 'Weight Averaging',
        'Scale': 0.0,
        'Calibration': 'ODC (Optimal Diagonal)',
        'Accuracies': wa_odc_accs,
        'Avg': np.mean(list(wa_odc_accs.values())),
        'DistBefore': odc_dist_bef,
        'DistAfter': odc_dist_aft
    })

    # --- Task Arithmetic (TA) Experiments ---
    print("\n=== Evaluating Task Arithmetic (TA) Merging with Sweeps ===")
    # Finer scaling factors to focus on active region
    ta_scaling_factors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    for scale in ta_scaling_factors:
        print(f"\n--- TA with Scale = {scale} ---")
        ta_model = merge_task_arithmetic(base_model, experts, scaling_factor=scale)
        
        # 1. TA Baseline (uncalibrated)
        ta_baseline_accs = {}
        for task in tasks:
            ta_baseline_accs[task] = evaluate_model(ta_model, loaders, task)
        avg_ta_base = np.mean(list(ta_baseline_accs.values()))
        print(f"TA Baseline Accs (Scale={scale}): {ta_baseline_accs} | Avg: {avg_ta_base:.2f}%")
        results.append({
            'Method': 'Task Arithmetic',
            'Scale': scale,
            'Calibration': 'None',
            'Accuracies': ta_baseline_accs,
            'Avg': avg_ta_base
        })
        
        # 2. TA + TCAC (Diagonal)
        ta_tcac_model = clone_model(ta_model)
        tcac_hooks, tcac_handles = register_calibration_hooks(ta_tcac_model, mode='diagonal')
        ta_tcac_accs, dist_bef, dist_aft = evaluate_with_task_conditional_calibration(ta_tcac_model, tcac_hooks, mode='diagonal')
        for handle in tcac_handles:
            handle.remove()
        avg_ta_tcac = np.mean(list(ta_tcac_accs.values()))
        print(f"TA + TCAC Accs (Scale={scale}): {ta_tcac_accs} | Avg: {avg_ta_tcac:.2f}%")
        results.append({
            'Method': 'Task Arithmetic',
            'Scale': scale,
            'Calibration': 'TCAC (Diagonal)',
            'Accuracies': ta_tcac_accs,
            'Avg': avg_ta_tcac,
            'DistBefore': dist_bef,
            'DistAfter': dist_aft
        })
        
        # 3. TA + R-MCAC with Best Alpha
        ta_rmcac_model = clone_model(ta_model)
        rmcac_hooks, rmcac_handles = register_calibration_hooks(ta_rmcac_model, mode='regularized_multivariate', alpha=best_alpha)
        ta_rmcac_accs, m_dist_bef, m_dist_aft = evaluate_with_task_conditional_calibration(ta_rmcac_model, rmcac_hooks, mode='regularized_multivariate', alpha=best_alpha)
        for handle in rmcac_handles:
            handle.remove()
        avg_ta_rmcac = np.mean(list(ta_rmcac_accs.values()))
        print(f"TA + R-MCAC (alpha={best_alpha}) Accs (Scale={scale}): {ta_rmcac_accs} | Avg: {avg_ta_rmcac:.2f}%")
        results.append({
            'Method': 'Task Arithmetic',
            'Scale': scale,
            'Calibration': f'R-MCAC (alpha={best_alpha})',
            'Accuracies': ta_rmcac_accs,
            'Avg': avg_ta_rmcac,
            'DistBefore': m_dist_bef,
            'DistAfter': m_dist_aft
        })

        # 4. TA + ODC (Optimal Diagonal)
        ta_odc_model = clone_model(ta_model)
        odc_hooks, odc_handles = register_calibration_hooks(ta_odc_model, mode='optimal_diagonal')
        ta_odc_accs, o_dist_bef, o_dist_aft = evaluate_with_task_conditional_calibration(ta_odc_model, odc_hooks, mode='optimal_diagonal')
        for handle in odc_handles:
            handle.remove()
        avg_ta_odc = np.mean(list(ta_odc_accs.values()))
        print(f"TA + ODC Accs (Scale={scale}): {ta_odc_accs} | Avg: {avg_ta_odc:.2f}%")
        results.append({
            'Method': 'Task Arithmetic',
            'Scale': scale,
            'Calibration': 'ODC (Optimal Diagonal)',
            'Accuracies': ta_odc_accs,
            'Avg': avg_ta_odc,
            'DistBefore': o_dist_bef,
            'DistAfter': o_dist_aft
        })

    # Print a beautiful summarized comparison table
    print("\n" + "="*115)
    print("FINAL EXPERIMENTAL RESULTS SUMMARY")
    print("="*115)
    print(f"{'Method':<20} | {'Scale':<5} | {'Calibration':<25} | {'MNIST':<7} | {'F-MNIST':<7} | {'CIFAR10':<7} | {'Average':<7} | {'Dist-Bef':<8} | {'Dist-Aft':<8}")
    print("-"*120)
    
    # First print Individual Experts as Upper Bound
    print(f"{'Individual Experts':<20} | {'-':<5} | {'-':<25} | {expert_accs['MNIST']:<7.2f} | {expert_accs['FashionMNIST']:<7.2f} | {expert_accs['CIFAR10']:<7.2f} | {avg_expert_acc:<7.2f} | {'-':<8} | {'-':<8}")
    print("-"*120)
    
    for r in results:
        scale_str = f"{r['Scale']:.2f}" if r['Method'] == 'Task Arithmetic' else '-'
        bef_str = f"{r['DistBefore']:.4f}" if 'DistBefore' in r else '-'
        aft_str = f"{r['DistAfter']:.4f}" if 'DistAfter' in r else '-'
        print(f"{r['Method']:<20} | {scale_str:<5} | {r['Calibration']:<25} | {r['Accuracies']['MNIST']:<7.2f} | {r['Accuracies']['FashionMNIST']:<7.2f} | {r['Accuracies']['CIFAR10']:<7.2f} | {r['Avg']:<7.2f} | {bef_str:<8} | {aft_str:<8}")
        
    print("="*120)

    # Save results to a text file for Latex integration
    with open('results_summary.txt', 'w') as f:
        f.write("Method,Scale,Calibration,MNIST,FashionMNIST,CIFAR10,Average,DistBefore,DistAfter\n")
        f.write(f"Individual Experts,-,-,{expert_accs['MNIST']:.2f},{expert_accs['FashionMNIST']:.2f},{expert_accs['CIFAR10']:.2f},{avg_expert_acc:.2f},-,-\n")
        for r in results:
            scale_val = r['Scale']
            bef_val = r.get('DistBefore', -1.0)
            aft_val = r.get('DistAfter', -1.0)
            f.write(f"{r['Method']},{scale_val},{r['Calibration']},{r['Accuracies']['MNIST']:.2f},{r['Accuracies']['FashionMNIST']:.2f},{r['Accuracies']['CIFAR10']:.2f},{r['Avg']:.2f},{bef_val:.4f},{aft_val:.4f}\n")

    print("\nResults successfully written to results_summary.txt.")

if __name__ == '__main__':
    main()
