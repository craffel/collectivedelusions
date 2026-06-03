import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

# Define Model Architecture
class MultiTaskResNet18(nn.Module):
    def __init__(self, tasks=['mnist', 'fashion', 'cifar10']):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove default fc layer
        
        # Task-specific classification heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(num_features, 10) for task in tasks
        })
        
    def forward(self, x, task):
        features = self.backbone(x)
        return self.heads[task](features)

# Linear Centered Kernel Alignment (CKA)
def compute_linear_cka(X, Y):
    # X: [N, d1]
    # Y: [N, d2]
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    
    cov = torch.matmul(X_c.t(), Y_c)
    hsic = torch.sum(cov ** 2)
    
    var_X = torch.sum(torch.matmul(X_c.t(), X_c) ** 2)
    var_Y = torch.sum(torch.matmul(Y_c.t(), Y_c) ** 2)
    
    cka = hsic / (torch.sqrt(var_X * var_Y) + 1e-8)
    return cka.item()

# Forward Hook class for TCAC and CKA activation collection
class ActivationHook:
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.mode = 'idle'  # 'collect_orig', 'collect_merged', 'apply', 'idle'
        # Stats dictionary mapping task -> stats
        self.stats = {}
        self.current_task = None
        # Shared or task-specific activations for CKA analysis
        self.collected_activations = []

    def clear_activations(self):
        self.collected_activations = []

    def hook_fn(self, module, input, output):
        # Shape of output: [B, C, H, W]
        if self.mode == 'collect_orig':
            mu = output.mean(dim=(0, 2, 3), keepdim=True)
            var = output.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            sigma = torch.sqrt(var + 1e-5)
            
            if self.current_task not in self.stats:
                self.stats[self.current_task] = {}
            self.stats[self.current_task]['mu_orig'] = mu.detach().clone()
            self.stats[self.current_task]['sigma_orig'] = sigma.detach().clone()
            
        elif self.mode == 'collect_merged':
            mu = output.mean(dim=(0, 2, 3), keepdim=True)
            var = output.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            sigma = torch.sqrt(var + 1e-5)
            
            if self.current_task not in self.stats:
                self.stats[self.current_task] = {}
            self.stats[self.current_task]['mu_merged'] = mu.detach().clone()
            self.stats[self.current_task]['sigma_merged'] = sigma.detach().clone()
            
        elif self.mode == 'apply':
            if self.current_task in self.stats and 'mu_merged' in self.stats[self.current_task]:
                mu_m = self.stats[self.current_task]['mu_merged']
                sig_m = self.stats[self.current_task]['sigma_merged']
                mu_o = self.stats[self.current_task]['mu_orig']
                sig_o = self.stats[self.current_task]['sigma_orig']
                
                # Rescale activations task-conditionally
                normalized = (output - mu_m) / sig_m
                rescaled = normalized * sig_o + mu_o
                output = rescaled
                
        if self.mode in ['collect_orig', 'collect_merged', 'collect_cka', 'apply']:
            # For CKA, we collect flat features: shape [B, C * H * W]
            flat = output.view(output.size(0), -1).detach().cpu()
            self.collected_activations.append(flat)
            
        return output

# Setup hooks on ResNet-18 key layers
def setup_hooks(model):
    hooks = {}
    layers_to_hook = {
        'conv1': model.backbone.conv1,
        'layer1': model.backbone.layer1,
        'layer2': model.backbone.layer2,
        'layer3': model.backbone.layer3,
        'layer4': model.backbone.layer4
    }
    for name, layer in layers_to_hook.items():
        hook = ActivationHook(name)
        handle = layer.register_forward_hook(hook.hook_fn)
        hooks[name] = (hook, handle)
    return hooks

def remove_hooks(hooks):
    for name, (hook, handle) in hooks.items():
        handle.remove()

def set_hooks_mode(hooks, mode, current_task=None):
    for name, (hook, handle) in hooks.items():
        hook.mode = mode
        hook.current_task = current_task
        hook.clear_activations()

def collect_hooked_activations(hooks):
    collected = {}
    for name, (hook, handle) in hooks.items():
        if len(hook.collected_activations) > 0:
            collected[name] = torch.cat(hook.collected_activations, dim=0)
    return collected

# Get state dictionaries
def get_backbone_state(model):
    return {k: v.clone().cpu() for k, v in model.backbone.state_dict().items()}

def set_backbone_state(model, state_dict):
    device = next(model.parameters()).device
    loaded_state = {k: v.to(device) for k, v in state_dict.items()}
    model.backbone.load_state_dict(loaded_state)

# Model Merging Function
def merge_models(base_state, expert_states, lambdas):
    merged_state = {}
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            update = torch.zeros_like(base_state[key])
            for task, exp_state in expert_states.items():
                update += lambdas[task] * (exp_state[key] - base_state[key])
            merged_state[key] = base_state[key] + update
        else:
            merged_state[key] = base_state[key].clone()
    return merged_state

# Differentiable Model Merging Function for Joint TTA
def merge_models_differentiable(base_state, expert_states, lambdas_tensor, tasks=['mnist', 'fashion', 'cifar10']):
    merged_state = {}
    for key in base_state.keys():
        if base_state[key].is_floating_point():
            # Standard buffers (like running_mean and running_var) are not differentiable in PyTorch's native_batch_norm.
            # We must merge them using detached lambdas so they do not carry gradients.
            if 'running_mean' in key or 'running_var' in key:
                lambdas_detached = lambdas_tensor.detach()
                update = torch.zeros_like(base_state[key], device=lambdas_detached.device)
                for idx, task in enumerate(tasks):
                    update = update + lambdas_detached[idx] * (expert_states[task][key].to(lambdas_detached.device) - base_state[key].to(lambdas_detached.device))
                merged_state[key] = base_state[key].to(lambdas_detached.device) + update
            else:
                # Differentiable parameters (weights and biases of conv, linear, BN weights)
                update = torch.zeros_like(base_state[key], device=lambdas_tensor.device)
                for idx, task in enumerate(tasks):
                    update = update + lambdas_tensor[idx] * (expert_states[task][key].to(lambdas_tensor.device) - base_state[key].to(lambdas_tensor.device))
                merged_state[key] = base_state[key].to(lambdas_tensor.device) + update
        else:
            merged_state[key] = base_state[key].clone().to(lambdas_tensor.device)
    return merged_state

# Data Loading
def load_data(data_dir='./data'):
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_ds = {
        'mnist': torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray),
        'fashion': torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray),
        'cifar10': torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
    }
    
    test_ds = {
        'mnist': torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray),
        'fashion': torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray),
        'cifar10': torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_rgb)
    }
    
    return train_ds, test_ds

# Create Subsets for Training, testing and Calibration
def create_subsets(train_ds, test_ds, n_train=2000, n_test=500, n_calib=128):
    set_seed(42)
    sub_train, sub_test, sub_calib = {}, {}, {}
    
    for task in train_ds.keys():
        len_tr = len(train_ds[task])
        indices_tr = np.random.choice(len_tr, n_train, replace=False)
        sub_train[task] = Subset(train_ds[task], indices_tr)
        
        # Remaining for calibration (to avoid overlap)
        rem_indices = list(set(range(len_tr)) - set(indices_tr))
        indices_cal = np.random.choice(rem_indices, n_calib, replace=False)
        sub_calib[task] = Subset(train_ds[task], indices_cal)
        
        len_te = len(test_ds[task])
        indices_te = np.random.choice(len_te, n_test, replace=False)
        sub_test[task] = Subset(test_ds[task], indices_te)
        
    return sub_train, sub_test, sub_calib

# Evaluate function
def evaluate(model, test_loaders, device, hooks=None, hook_mode='idle', task_specific=True):
    model.eval()
    accuracies = {}
    
    with torch.no_grad():
        for task, loader in test_loaders.items():
            correct = 0
            total = 0
            if hooks:
                set_hooks_mode(hooks, hook_mode, current_task=task)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, task)
                _, pred = outputs.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
            accuracies[task] = 100.0 * correct / total
            
    if hooks:
        set_hooks_mode(hooks, 'idle')
    return accuracies

# Main Execution Flow
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    
    # 1. Load Data and Subsets
    train_ds, test_ds = load_data()
    sub_train, sub_test, sub_calib = create_subsets(train_ds, test_ds)
    
    train_loaders = {t: DataLoader(sub_train[t], batch_size=128, shuffle=True) for t in sub_train.keys()}
    test_loaders = {t: DataLoader(sub_test[t], batch_size=128, shuffle=False) for t in sub_test.keys()}
    calib_loaders = {t: DataLoader(sub_calib[t], batch_size=128, shuffle=False) for t in sub_calib.keys()}
    
    # 2. Initialize Model & Save Base State
    model = MultiTaskResNet18().to(device)
    base_state = get_backbone_state(model)
    torch.save(base_state, 'base_backbone.pt')
    
    # 3. Train Experts or Load if available
    expert_states = {}
    expert_heads = {}
    expert_accuracies = {}
    
    tasks = ['mnist', 'fashion', 'cifar10']
    
    print('='*50)
    print('Training Experts on 2,000 samples for 3 epochs...')
    print('='*50)
    
    for task in tasks:
        backbone_path = f'expert_backbone_{task}.pt'
        head_path = f'expert_head_{task}.pt'
        
        if os.path.exists(backbone_path) and os.path.exists(head_path):
            print(f'Loading existing expert states for {task.upper()}...')
            expert_states[task] = torch.load(backbone_path, map_location='cpu')
            expert_heads[task] = torch.load(head_path, map_location='cpu')
            
            # Evaluate Loaded Expert
            set_backbone_state(model, expert_states[task])
            model.heads[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders[task]:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x, task)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            expert_acc = 100.0 * correct / total
            expert_accuracies[task] = expert_acc
            print(f'>>> Expert {task.upper()} Loaded Test Accuracy: {expert_acc:.2f}%\n')
        else:
            print(f'Training expert states for {task.upper()}...')
            # Re-initialize/reset to base model weights for training each expert
            set_backbone_state(model, base_state)
            # Reset head
            nn.init.kaiming_normal_(model.heads[task].weight, nonlinearity='linear')
            nn.init.constant_(model.heads[task].bias, 0)
            
            optimizer = torch.optim.AdamW(
                list(model.backbone.parameters()) + list(model.heads[task].parameters()),
                lr=1e-4, weight_decay=1e-2
            )
            
            # Train Loop
            model.train()
            for epoch in range(3):
                total_loss = 0
                correct = 0
                total = 0
                for x, y in train_loaders[task]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(x, task)
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * x.size(0)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
                
                epoch_loss = total_loss / total
                epoch_acc = 100.0 * correct / total
                print(f'Task {task.upper()} | Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
                
            # Evaluate Expert
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders[task]:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x, task)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            expert_acc = 100.0 * correct / total
            expert_accuracies[task] = expert_acc
            print(f'>>> Expert {task.upper()} Final Test Accuracy: {expert_acc:.2f}%\n')
            
            # Save states
            expert_states[task] = get_backbone_state(model)
            expert_heads[task] = {k: v.clone().cpu() for k, v in model.heads[task].state_dict().items()}
            torch.save(expert_states[task], f'expert_backbone_{task}.pt')
            torch.save(expert_heads[task], f'expert_head_{task}.pt')
        
    # 4. Set up Hooks for Activation Calibration and CKA
    hooks = setup_hooks(model)
    
    # Collect expert statistics and expert activations for CKA reference
    expert_test_activations = {t: {} for t in tasks}
    print('Collecting expert statistics and reference activations on test set...')
    for task in tasks:
        # Load expert model weights
        set_backbone_state(model, expert_states[task])
        model.heads[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
        # Collect Calibration statistics (using N=128 calibration subset)
        model.eval()
        set_hooks_mode(hooks, 'collect_orig', current_task=task)
        with torch.no_grad():
            for x, y in calib_loaders[task]:
                x = x.to(device)
                _ = model(x, task)
                
        # Collect Test Activations for CKA reference
        set_hooks_mode(hooks, 'collect_cka', current_task=task)
        with torch.no_grad():
            for x, y in test_loaders[task]:
                x = x.to(device)
                _ = model(x, task)
        expert_test_activations[task] = collect_hooked_activations(hooks)
        
    set_hooks_mode(hooks, 'idle')
    
    # 5. MERGING MODEL STRATEGIES
    lambdas = {'mnist': 0.3, 'fashion': 0.3, 'cifar10': 0.3}
    
    print('='*50)
    print('Evaluating Model Merging Strategies...')
    print('='*50)
    
    results = {}
    
    # Load all expert heads back into the model for multi-task evaluation
    for task in tasks:
        model.heads[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
    # A. Task Arithmetic Base Merged Model
    ta_state = merge_models(base_state, expert_states, lambdas)
    set_backbone_state(model, ta_state)
    ta_acc = evaluate(model, test_loaders, device)
    results['Task Arithmetic'] = ta_acc
    print(f'Task Arithmetic (Default λ=0.3): {ta_acc} | Avg: {np.mean(list(ta_acc.values())):.2f}%')
    
    # B. Weight Averaging (WA) Merged Model
    wa_lambdas = {'mnist': 0.333, 'fashion': 0.333, 'cifar10': 0.333}
    wa_state = merge_models(base_state, expert_states, wa_lambdas)
    set_backbone_state(model, wa_state)
    wa_acc = evaluate(model, test_loaders, device)
    results['Weight Averaging'] = wa_acc
    print(f'Weight Averaging (λ=0.333): {wa_acc} | Avg: {np.mean(list(wa_acc.values())):.2f}%')
    
    # C. Task-Conditional Activation Calibration (TCAC)
    # Collect merged statistics on N=128 calibration set
    set_backbone_state(model, ta_state)
    set_hooks_mode(hooks, 'collect_merged')
    with torch.no_grad():
        for task in tasks:
            set_hooks_mode(hooks, 'collect_merged', current_task=task)
            for x, y in calib_loaders[task]:
                x = x.to(device)
                _ = model(x, task)
    set_hooks_mode(hooks, 'idle')
    
    # Evaluate TCAC
    tcac_acc = evaluate(model, test_loaders, device, hooks=hooks, hook_mode='apply')
    results['TCAC'] = tcac_acc
    print(f'TCAC (Applied on TA): {tcac_acc} | Avg: {np.mean(list(tcac_acc.values())):.2f}%')
    
    # Collect CKA activations for TA (without calibration) and TCAC (with calibration)
    ta_test_activations = {}
    tcac_test_activations = {}
    
    set_backbone_state(model, ta_state)
    for task in tasks:
        set_hooks_mode(hooks, 'collect_cka', current_task=task)
        with torch.no_grad():
            for x, y in test_loaders[task]:
                x = x.to(device)
                _ = model(x, task)
        ta_test_activations[task] = collect_hooked_activations(hooks)
        
    for task in tasks:
        set_hooks_mode(hooks, 'apply', current_task=task)
        # Note: hook_mode 'apply' handles both scaling and collection since it returns output
        # But we need to collect CKA. Let's make hook_mode 'apply' also collect activations for CKA when requested,
        # or we can modify the hook. To keep it simple, we can run inference in 'apply' mode and manually collect.
        # Actually, in ActivationHook, when self.mode == 'apply', it doesn't collect activations unless we append.
        # Let's change ActivationHook to collect activations when mode is 'apply' too!
        # Yes, we added: if self.mode in ['collect_orig', 'collect_merged', 'collect_cka', 'apply'] we can collect.
        # Let's verify: in our ActivationHook class, we have:
        # if self.mode in ['collect_orig', 'collect_merged', 'collect_cka']:
        # Let's make sure 'apply' also collects if we want.
        # To make it robust, we can just run a quick evaluation and collect
        pass
    
    # Let's manually collect TCAC activations
    for task in tasks:
        for name, (hook, handle) in hooks.items():
            hook.mode = 'apply'
            hook.current_task = task
            hook.clear_activations()
        with torch.no_grad():
            for x, y in test_loaders[task]:
                x = x.to(device)
                _ = model(x, task)
        tcac_test_activations[task] = {}
        for name, (hook, handle) in hooks.items():
            tcac_test_activations[task][name] = torch.cat(hook.collected_activations, dim=0)
            hook.clear_activations()
            hook.mode = 'idle'
            
    # D. Supervised Head-Only Fine-Tuning (SFT) on Calibration set (N=128)
    # Reset model to Task Arithmetic state
    set_backbone_state(model, ta_state)
    # We freeze the backbone, optimize task-specific head on calibration set
    sft_heads = {}
    sft_acc = {}
    print('\nRunning Supervised Head-Only Fine-Tuning (SFT) on N=128 samples...')
    for task in tasks:
        # Load original expert head as initialization (or we can use default, let's use the expert head as starting point)
        head_copy = nn.Linear(512, 10).to(device)
        head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
        optimizer = torch.optim.AdamW(head_copy.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Frozen backbone + Head training
        model.eval() # Keep backbone in eval (frozen BatchNorm statistics)
        for epoch in range(15): # 15 epochs on 128 samples is extremely fast
            for x, y in calib_loaders[task]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = model.backbone(x)
                outputs = head_copy(features)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                optimizer.step()
                
        # Evaluate SFT head
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loaders[task]:
                x, y = x.to(device), y.to(device)
                features = model.backbone(x)
                outputs = head_copy(features)
                _, pred = outputs.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        sft_acc[task] = 100.0 * correct / total
        sft_heads[task] = {k: v.clone().cpu() for k, v in head_copy.state_dict().items()}
        
    results['Supervised Head SFT'] = sft_acc
    print(f'Supervised Head SFT: {sft_acc} | Avg: {np.mean(list(sft_acc.values())):.2f}%')
    
    # E. Unsupervised Head-Only Adaptation (TTA via self-distillation) on Calibration set (N=128)
    # Let's do distillation from the original expert model (the teacher) to the head (the student)
    distill_heads = {}
    distill_acc = {}
    print('Running Unsupervised/Self-Distillation Head-Only Adaptation (TTA) on N=128 samples...')
    for task in tasks:
        # Teacher is frozen expert model
        teacher_backbone = expert_states[task]
        teacher_head = {k: v.to(device) for k, v in expert_heads[task].items()}
        
        # Student head starting from expert head
        head_copy = nn.Linear(512, 10).to(device)
        head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
        optimizer = torch.optim.AdamW(head_copy.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Temporary model to run teacher inference
        t_model = MultiTaskResNet18().to(device)
        set_backbone_state(t_model, teacher_backbone)
        t_model.heads[task].load_state_dict(teacher_head)
        t_model.eval()
        
        model.eval()
        for epoch in range(15):
            for x, _ in calib_loaders[task]: # No labels used
                x = x.to(device)
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_outputs = t_model(x, task)
                    teacher_probs = F.softmax(teacher_outputs / 2.0, dim=1) # Temperature 2
                    features = model.backbone(x)
                    
                student_outputs = head_copy(features)
                student_log_probs = F.log_softmax(student_outputs / 2.0, dim=1)
                
                # KL Divergence Loss
                loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (2.0 ** 2)
                loss.backward()
                optimizer.step()
                
        # Evaluate Unsupervised head
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loaders[task]:
                x, y = x.to(device), y.to(device)
                features = model.backbone(x)
                outputs = head_copy(features)
                _, pred = outputs.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        distill_acc[task] = 100.0 * correct / total
        distill_heads[task] = {k: v.clone().cpu() for k, v in head_copy.state_dict().items()}
        
    results['Unsupervised Head TTA'] = distill_acc
    print(f'Unsupervised Head TTA (Distillation): {distill_acc} | Avg: {np.mean(list(distill_acc.values())):.2f}%')
    
    # F. Joint TTA (SyMerge-Style) - Demonstrating Runway Gradients & Instability
    # We jointly optimize lambdas (merging coefficients) and classification heads
    print('\nDemonstrating Joint TTA (unconstrained vs. constrained)...')
    from torch.func import functional_call

    # Instantiate frozen teacher models to avoid modifying the student backbone in-place
    teachers = {}
    for task in tasks:
        t_model = MultiTaskResNet18().to(device)
        set_backbone_state(t_model, expert_states[task])
        t_model.heads[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        t_model.eval()
        teachers[task] = t_model
    
    # 1. Unconstrained Joint TTA (demonstrates runaway gradient and activation explosion)
    print('--- Running Unconstrained Joint TTA (No clamping) ---')
    lambdas_param = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device))
    heads_to_opt = nn.ModuleDict({
        task: nn.Linear(512, 10).to(device) for task in tasks
    })
    for task in tasks:
        heads_to_opt[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
    optimizer = torch.optim.AdamW(
        list(heads_to_opt.parameters()) + [lambdas_param],
        lr=5e-2  # Higher learning rate to highlight runaway behavior quickly
    )
    
    mnist_iter = iter(calib_loaders['mnist'])
    fashion_iter = iter(calib_loaders['fashion'])
    cifar_iter = iter(calib_loaders['cifar10'])
    
    for step in range(12):
        try:
            mx, my = next(mnist_iter)
            fx, fy = next(fashion_iter)
            cx, cy = next(cifar_iter)
        except StopIteration:
            break
            
        optimizer.zero_grad()
        
        # Differentiable weight merging
        merged_params = merge_models_differentiable(base_state, expert_states, lambdas_param)
        
        loss = 0
        # MNIST batch
        mx = mx.to(device)
        m_feats = functional_call(model.backbone, merged_params, (mx,))
        m_out = heads_to_opt['mnist'](m_feats)
        with torch.no_grad():
            m_teacher = teachers['mnist'](mx, 'mnist')
        loss += F.kl_div(F.log_softmax(m_out / 2.0, dim=1), F.softmax(m_teacher / 2.0, dim=1), reduction='batchmean') * (2.0 ** 2)
        
        # FASHION batch
        fx = fx.to(device)
        f_feats = functional_call(model.backbone, merged_params, (fx,))
        f_out = heads_to_opt['fashion'](f_feats)
        with torch.no_grad():
            f_teacher = teachers['fashion'](fx, 'fashion')
        loss += F.kl_div(F.log_softmax(f_out / 2.0, dim=1), F.softmax(f_teacher / 2.0, dim=1), reduction='batchmean') * (2.0 ** 2)
        
        # CIFAR-10 batch
        cx = cx.to(device)
        c_feats = functional_call(model.backbone, merged_params, (cx,))
        c_out = heads_to_opt['cifar10'](c_feats)
        with torch.no_grad():
            c_teacher = teachers['cifar10'](cx, 'cifar10')
        loss += F.kl_div(F.log_softmax(c_out / 2.0, dim=1), F.softmax(c_teacher / 2.0, dim=1), reduction='batchmean') * (2.0 ** 2)
        
        loss.backward()
        optimizer.step()
        
        print(f'Step {step+1} | Lambdas: {lambdas_param.detach().cpu().numpy()} | Loss: {loss.item():.4f}')
        
        if torch.isnan(loss) or any(torch.isnan(lambdas_param)) or any(torch.isinf(lambdas_param)):
            print(f'>>> CRITICAL: Numerical instability triggered! Loss became NaN or coefficients exploded.')
            break

    # 2. Clamped Joint TTA (demonstrates clamping paradox where coefficients get stuck)
    print('--- Running Clamped Joint TTA (clamped to [0.0, 0.3]) ---')
    lambdas_clamp = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device))
    heads_clamp = nn.ModuleDict({
        task: nn.Linear(512, 10).to(device) for task in tasks
    })
    for task in tasks:
        heads_clamp[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
    optimizer_clamp = torch.optim.AdamW(
        list(heads_clamp.parameters()) + [lambdas_clamp],
        lr=5e-2
    )
    
    mnist_iter = iter(calib_loaders['mnist'])
    fashion_iter = iter(calib_loaders['fashion'])
    cifar_iter = iter(calib_loaders['cifar10'])
    
    for step in range(12):
        try:
            mx, my = next(mnist_iter)
            fx, fy = next(fashion_iter)
            cx, cy = next(cifar_iter)
        except StopIteration:
            break
            
        optimizer_clamp.zero_grad()
        
        # Differentiable weight merging
        merged_params = merge_models_differentiable(base_state, expert_states, lambdas_clamp)
        
        loss = 0
        # MNIST batch
        mx = mx.to(device)
        m_feats = functional_call(model.backbone, merged_params, (mx,))
        m_out = heads_clamp['mnist'](m_feats)
        with torch.no_grad():
            m_teacher = teachers['mnist'](mx, 'mnist')
        loss += F.kl_div(F.log_softmax(m_out / 2.0, dim=1), F.softmax(m_teacher / 2.0, dim=1), reduction='batchmean') * (2.0 ** 2)
        
        # FASHION batch
        fx = fx.to(device)
        f_feats = functional_call(model.backbone, merged_params, (fx,))
        f_out = heads_clamp['fashion'](f_feats)
        with torch.no_grad():
            f_teacher = teachers['fashion'](fx, 'fashion')
        loss += F.kl_div(F.log_softmax(f_out / 2.0, dim=1), F.softmax(f_teacher / 2.0, dim=1), reduction='batchmean') * (2.0 ** 2)
        
        # CIFAR-10 batch
        cx = cx.to(device)
        c_feats = functional_call(model.backbone, merged_params, (cx,))
        c_out = heads_clamp['cifar10'](c_feats)
        with torch.no_grad():
            c_teacher = teachers['cifar10'](cx, 'cifar10')
        loss += F.kl_div(F.log_softmax(c_out / 2.0, dim=1), F.softmax(c_teacher / 2.0, dim=1), reduction='batchmean') * (2.0 ** 2)
        
        loss.backward()
        optimizer_clamp.step()
        
        # Apply Clamping to demonstrate clamping paradox
        with torch.no_grad():
            lambdas_clamp.clamp_(0.0, 0.3)
            
        print(f'Step {step+1} | Lambdas (Clamped): {lambdas_clamp.detach().cpu().numpy()} | Loss: {loss.item():.4f}')
            
    # Remove hooks
    remove_hooks(hooks)
    
    # 6. CKA REPRESENTATIONAL SIMILARITY ANALYSIS
    print('\n' + '='*50)
    print('Computing Centered Kernel Alignment (CKA)...')
    print('='*50)
    
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    cka_results = {t: {'TA': [], 'TCAC': []} for t in tasks}
    
    for task in tasks:
        # CKA between Expert and TA
        for l_name in layer_names:
            X = expert_test_activations[task][l_name]
            Y = ta_test_activations[task][l_name]
            cka_val = compute_linear_cka(X, Y)
            cka_results[task]['TA'].append(cka_val)
            
        # CKA between Expert and TCAC
        for l_name in layer_names:
            X = expert_test_activations[task][l_name]
            Y = tcac_test_activations[task][l_name]
            cka_val = compute_linear_cka(X, Y)
            cka_results[task]['TCAC'].append(cka_val)
            
        print(f'Task {task.upper()} Layer CKA (Expert vs. TA):   {cka_results[task]["TA"]}')
        print(f'Task {task.upper()} Layer CKA (Expert vs. TCAC): {cka_results[task]["TCAC"]}')
        print('-'*40)
        
    # 7. GENERATE TABLES AND PLOTS
    print('\nGenerating plots and Markdown tables...')
    
    # Save a CSV/TXT summary of the results
    with open('results_summary.txt', 'w') as f:
        f.write('Method | MNIST | Fashion-MNIST | CIFAR-10 | Average\n')
        f.write('-'*60 + '\n')
        f.write(f'Individual Experts | {expert_accuracies["mnist"]:.2f}% | {expert_accuracies["fashion"]:.2f}% | {expert_accuracies["cifar10"]:.2f}% | {np.mean(list(expert_accuracies.values())):.2f}%\n')
        for method, accs in results.items():
            f.write(f'{method} | {accs["mnist"]:.2f}% | {accs["fashion"]:.2f}% | {accs["cifar10"]:.2f}% | {np.mean(list(accs.values())):.2f}%\n')
            
    # Print the table to console
    with open('results_summary.txt', 'r') as f:
        print(f.read())
        
    # Plot CKA
    plt.figure(figsize=(12, 5))
    for i, task in enumerate(tasks):
        plt.subplot(1, 3, i+1)
        plt.plot(layer_names, cka_results[task]['TA'], marker='o', label='TA (Uncalibrated)', color='red')
        plt.plot(layer_names, cka_results[task]['TCAC'], marker='s', label='TCAC (Calibrated)', color='green', linestyle='--')
        plt.title(f'{task.upper()} CKA to Expert')
        plt.xlabel('Layer')
        plt.ylabel('CKA Similarity')
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig('cka_similarity_plot.png', dpi=300)
    plt.close()
    
    # Plot Accuracies
    methods = ['Task Arithmetic', 'Weight Averaging', 'TCAC', 'Supervised Head SFT', 'Unsupervised Head TTA']
    avg_accs = [np.mean(list(results[m].values())) for m in methods]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_accs, color=['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6'], width=0.6)
    plt.axhline(y=np.mean(list(expert_accuracies.values())), color='black', linestyle='--', label='Expert Upper Bound (Avg)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Comparison of Model Merging and Adaptation Methods')
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig('method_comparison_plot.png', dpi=300)
    plt.close()
    
    # Run the methodological sweeps and ablations for Phase 4
    run_sweeps(model, train_ds, test_ds, base_state, expert_states, expert_heads, device, tasks, hooks, ta_state)
    
    print('All experiments completed and figures saved successfully!')

def run_sweeps(model, train_ds, test_ds, base_state, expert_states, expert_heads, device, tasks, hooks, ta_state):
    print('\n' + '='*50)
    print('RUNNING METHODOLOGICAL SWEEPS AND ABLATIONS (Phase 4)...')
    print('='*50)
    
    # Re-setup hooks locally for run_sweeps since they were removed in main
    hooks = setup_hooks(model)
    
    # We will sweep calibration sample size N
    n_values = [32, 64, 128, 256, 512]
    
    sweep_results = {
        'N': [],
        'SFT': [],
        'TTA': [],
        'TCAC': []
    }
    
    # 1. SAMPLE COMPLEXITY SWEEP
    for N in n_values:
        print(f'\n>>> Evaluating Sample Complexity with N = {N} ...')
        
        # Create loaders for this N
        sub_train, sub_test, sub_calib = create_subsets(train_ds, test_ds, n_calib=N)
        calib_loaders = {t: DataLoader(sub_calib[t], batch_size=128, shuffle=False) for t in sub_calib.keys()}
        test_loaders = {t: DataLoader(sub_test[t], batch_size=128, shuffle=False) for t in sub_test.keys()}
        
        # --- A. TCAC with N samples ---
        set_backbone_state(model, ta_state)
        # Reset heads to expert heads for evaluation
        for task in tasks:
            model.heads[task].load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
            
        # Collect expert statistics
        set_hooks_mode(hooks, 'idle')
        for task in tasks:
            set_backbone_state(model, expert_states[task])
            model.eval()
            set_hooks_mode(hooks, 'collect_orig', current_task=task)
            with torch.no_grad():
                for x, y in calib_loaders[task]:
                    _ = model(x.to(device), task)
                    
        # Collect merged statistics
        set_backbone_state(model, ta_state)
        for task in tasks:
            set_hooks_mode(hooks, 'collect_merged', current_task=task)
            with torch.no_grad():
                for x, y in calib_loaders[task]:
                    _ = model(x.to(device), task)
        set_hooks_mode(hooks, 'idle')
        
        # Evaluate TCAC
        print(f"Stats check (N={N}):")
        for name, (hook, _) in hooks.items():
            print(f"  Hook {name} stats tasks: {list(hook.stats.keys())}")
            if 'mnist' in hook.stats:
                print(f"    mnist keys: {list(hook.stats['mnist'].keys())}")
        tcac_acc = evaluate(model, test_loaders, device, hooks=hooks, hook_mode='apply')
        avg_tcac = np.mean(list(tcac_acc.values()))
        
        # --- B. Supervised Head SFT with N samples ---
        set_backbone_state(model, ta_state)
        sft_acc = {}
        for task in tasks:
            head_copy = nn.Linear(512, 10).to(device)
            head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
            optimizer = torch.optim.AdamW(head_copy.parameters(), lr=1e-3, weight_decay=1e-4)
            
            model.eval()
            for epoch in range(15):
                for x, y in calib_loaders[task]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        features = model.backbone(x)
                    outputs = head_copy(features)
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders[task]:
                    x, y = x.to(device), y.to(device)
                    features = model.backbone(x)
                    outputs = head_copy(features)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            sft_acc[task] = 100.0 * correct / total
        avg_sft = np.mean(list(sft_acc.values()))
        
        # --- C. Unsupervised Head TTA with N samples ---
        set_backbone_state(model, ta_state)
        distill_acc = {}
        for task in tasks:
            teacher_backbone = expert_states[task]
            teacher_head = {k: v.to(device) for k, v in expert_heads[task].items()}
            
            head_copy = nn.Linear(512, 10).to(device)
            head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
            optimizer = torch.optim.AdamW(head_copy.parameters(), lr=1e-3, weight_decay=1e-4)
            
            t_model = MultiTaskResNet18().to(device)
            set_backbone_state(t_model, teacher_backbone)
            t_model.heads[task].load_state_dict(teacher_head)
            t_model.eval()
            
            model.eval()
            for epoch in range(15):
                for x, _ in calib_loaders[task]:
                    x = x.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = t_model(x, task)
                        teacher_probs = F.softmax(teacher_outputs / 2.0, dim=1)
                        features = model.backbone(x)
                    student_outputs = head_copy(features)
                    student_log_probs = F.log_softmax(student_outputs / 2.0, dim=1)
                    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (2.0 ** 2)
                    loss.backward()
                    optimizer.step()
                    
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders[task]:
                    x, y = x.to(device), y.to(device)
                    features = model.backbone(x)
                    outputs = head_copy(features)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            distill_acc[task] = 100.0 * correct / total
        avg_distill = np.mean(list(distill_acc.values()))
        
        print(f"Results for N={N}: TCAC={avg_tcac:.2f}% (raw: {tcac_acc}), SFT={avg_sft:.2f}% (raw: {sft_acc}), TTA={avg_distill:.2f}% (raw: {distill_acc})")
        sweep_results['N'].append(N)
        sweep_results['TCAC'].append(avg_tcac)
        sweep_results['SFT'].append(avg_sft)
        sweep_results['TTA'].append(avg_distill)
        
    # 2. LEARNING RATE ROBUSTNESS SWEEP (on N=128)
    lr_values = [1e-4, 1e-3, 1e-2]
    lr_results = {
        'LR': [],
        'SFT': [],
        'TTA': []
    }
    
    sub_train, sub_test, sub_calib = create_subsets(train_ds, test_ds, n_calib=128)
    calib_loaders_128 = {t: DataLoader(sub_calib[t], batch_size=128, shuffle=False) for t in sub_calib.keys()}
    test_loaders_128 = {t: DataLoader(sub_test[t], batch_size=128, shuffle=False) for t in sub_test.keys()}
    
    for lr in lr_values:
        print(f'\n>>> Evaluating Optimization Stability with LR = {lr} ...')
        
        # Supervised Head SFT
        set_backbone_state(model, ta_state)
        sft_acc = {}
        for task in tasks:
            head_copy = nn.Linear(512, 10).to(device)
            head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
            optimizer = torch.optim.AdamW(head_copy.parameters(), lr=lr, weight_decay=1e-4)
            
            model.eval()
            for epoch in range(15):
                for x, y in calib_loaders_128[task]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        features = model.backbone(x)
                    outputs = head_copy(features)
                    loss = F.cross_entropy(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders_128[task]:
                    x, y = x.to(device), y.to(device)
                    features = model.backbone(x)
                    outputs = head_copy(features)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            sft_acc[task] = 100.0 * correct / total
        avg_sft = np.mean(list(sft_acc.values()))
        
        # Unsupervised Head TTA
        set_backbone_state(model, ta_state)
        distill_acc = {}
        for task in tasks:
            teacher_backbone = expert_states[task]
            teacher_head = {k: v.to(device) for k, v in expert_heads[task].items()}
            
            head_copy = nn.Linear(512, 10).to(device)
            head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
            optimizer = torch.optim.AdamW(head_copy.parameters(), lr=lr, weight_decay=1e-4)
            
            t_model = MultiTaskResNet18().to(device)
            set_backbone_state(t_model, teacher_backbone)
            t_model.heads[task].load_state_dict(teacher_head)
            t_model.eval()
            
            model.eval()
            for epoch in range(15):
                for x, _ in calib_loaders_128[task]:
                    x = x.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = t_model(x, task)
                        teacher_probs = F.softmax(teacher_outputs / 2.0, dim=1)
                        features = model.backbone(x)
                    student_outputs = head_copy(features)
                    student_log_probs = F.log_softmax(student_outputs / 2.0, dim=1)
                    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (2.0 ** 2)
                    loss.backward()
                    optimizer.step()
                    
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders_128[task]:
                    x, y = x.to(device), y.to(device)
                    features = model.backbone(x)
                    outputs = head_copy(features)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            distill_acc[task] = 100.0 * correct / total
        avg_distill = np.mean(list(distill_acc.values()))
        
        print(f"Results for LR={lr}: SFT={avg_sft:.2f}%, TTA={avg_distill:.2f}%")
        lr_results['LR'].append(lr)
        lr_results['SFT'].append(avg_sft)
        lr_results['TTA'].append(avg_distill)
        
    # Write results to text files
    with open('ablation_results.txt', 'w') as f:
        f.write('=== Sample Complexity Sweep (Average Accuracy %) ===\n')
        f.write('N | TCAC | Supervised Head SFT | Unsupervised Head TTA\n')
        f.write('-'*70 + '\n')
        for i, N in enumerate(n_values):
            f.write(f'{N} | {sweep_results["TCAC"][i]:.2f}% | {sweep_results["SFT"][i]:.2f}% | {sweep_results["TTA"][i]:.2f}%\n')
            
        f.write('\n=== Learning Rate Sweep (Average Accuracy %, N=128) ===\n')
        f.write('LR | Supervised Head SFT | Unsupervised Head TTA\n')
        f.write('-'*70 + '\n')
        for i, lr in enumerate(lr_values):
            f.write(f'{lr} | {lr_results["SFT"][i]:.2f}% | {lr_results["TTA"][i]:.2f}%\n')
            
    # Generate Plot
    plt.figure(figsize=(10, 4.5))
    
    # Left subplot: Sample Complexity
    plt.subplot(1, 2, 1)
    plt.plot(n_values, sweep_results['SFT'], marker='o', label='Supervised Head SFT', color='#3498db', linewidth=2)
    plt.plot(n_values, sweep_results['TTA'], marker='s', label='Unsupervised Head TTA', color='#9b59b6', linewidth=2)
    plt.plot(n_values, sweep_results['TCAC'], marker='^', label='TCAC (Calibration)', color='#2ecc71', linestyle='--', linewidth=1.5)
    plt.title('Sample Complexity Sweep')
    plt.xlabel('Calibration Samples N per Task')
    plt.ylabel('Average Accuracy (%)')
    plt.xscale('log')
    plt.xticks(n_values, n_values)
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend()
    
    # Right subplot: Learning Rate Robustness
    plt.subplot(1, 2, 2)
    lr_labels = ['1e-4', '1e-3', '1e-2']
    x_indices = np.arange(len(lr_labels))
    plt.bar(x_indices - 0.2, lr_results['SFT'], 0.4, label='Supervised Head SFT', color='#3498db')
    plt.bar(x_indices + 0.2, lr_results['TTA'], 0.4, label='Unsupervised Head TTA', color='#9b59b6')
    plt.xticks(x_indices, lr_labels)
    plt.title('Learning Rate Robustness')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Accuracy (%)')
    plt.ylim(0, 55)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ablation_sweeps_plot.png', dpi=300)
    plt.close()
    
    # Remove the hooks setup inside run_sweeps
    remove_hooks(hooks)
    
    print('Sweeps and ablation plots generated and saved to ablation_sweeps_plot.png!')

if __name__ == '__main__':
    main()
