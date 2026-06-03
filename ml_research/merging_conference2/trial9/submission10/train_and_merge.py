import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ----------------------------------------------------
# 1. Dataset Preparation
# ----------------------------------------------------
def get_dataloaders(batch_size=256):
    # Grayscale transforms (MNIST, FMNIST)
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Duplicate channels to 3
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 transforms
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Datasets
    train_mnist = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_gray)
    test_mnist = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_gray)

    train_fmnist = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_gray)
    test_fmnist = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_gray)

    train_cifar = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_cifar)
    test_cifar = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_cifar)

    # Loaders
    loaders = {
        'mnist': {
            'train': DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        },
        'fmnist': {
            'train': DataLoader(train_fmnist, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        },
        'cifar10': {
            'train': DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        }
    }
    return loaders

# ----------------------------------------------------
# 2. Model Definitions
# ----------------------------------------------------
class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3 * 32 * 32, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(512, 512)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        return x

def get_progenitor(arch_type):
    if arch_type == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()  # Replace fc layer with identity mapping
        return model
    elif arch_type == 'mlp':
        return MLPBackbone()
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

class ExpertModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ----------------------------------------------------
# 3. Training and Evaluation Functions
# ----------------------------------------------------
def train_expert(arch_type, task_name, train_loader, epochs=5):
    print(f"\n--- Training Expert: {arch_type} on {task_name} ---")
    
    # Get standard progenitor backbone
    if arch_type == 'resnet18':
        # Load progenitor weights from checkpoints if already saved to ensure common start,
        # otherwise download and save.
        prog_path = f'checkpoints/{arch_type}_progenitor.pth'
        if os.path.exists(prog_path):
            backbone = get_progenitor(arch_type)
            backbone.load_state_dict(torch.load(prog_path))
        else:
            backbone = get_progenitor(arch_type)
            torch.save(backbone.state_dict(), prog_path)
    else: # mlp
        # MLP progenitor must be randomly initialized but SHARED among experts!
        prog_path = f'checkpoints/{arch_type}_progenitor.pth'
        backbone = get_progenitor(arch_type)
        if os.path.exists(prog_path):
            backbone.load_state_dict(torch.load(prog_path))
        else:
            torch.save(backbone.state_dict(), prog_path)

    head = nn.Linear(512, 10).to(device)
    backbone = backbone.to(device)
    model = ExpertModel(backbone, head).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # Save expert backbone and head
    torch.save(backbone.state_dict(), f'checkpoints/{arch_type}_{task_name}_backbone.pth')
    torch.save(head.state_dict(), f'checkpoints/{arch_type}_{task_name}_head.pth')
    print(f"Expert saved: checkpoints/{arch_type}_{task_name}_(backbone/head).pth")

def evaluate_model(backbone, head, test_loader):
    backbone.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = head(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

# ----------------------------------------------------
# 4. Merging and Calibration Methods
# ----------------------------------------------------
def apply_cmva(merged_backbone, expert_backbones, mode='both'):
    """
    Applies Channel-wise Mean-Variance Alignment (CMVA) to the merged model's weights.
    mode can be 'both', 'std_only', or 'mean_only'.
    """
    with torch.no_grad():
        # Load state dicts
        merged_state = merged_backbone.state_dict()
        expert_states = [expert.state_dict() for expert in expert_backbones]
        
        for name, param in merged_backbone.named_parameters():
            # Apply to weight tensors of backbone (exclude non-weight 1D parameters like biases/BN scales)
            if 'fc' in name or 'head' in name or param.ndim < 2:
                continue
            
            # Reduce dims represent all dimensions except the first (output channel/feature)
            reduce_dims = list(range(1, param.ndim))
            
            # Compute expert statistics per channel
            expert_means = []
            expert_stds = []
            for state_k in expert_states:
                p_expert = state_k[name]
                m = p_expert.mean(dim=reduce_dims, keepdim=True)
                s = p_expert.std(dim=reduce_dims, keepdim=True)
                expert_means.append(m)
                expert_stds.append(s)
            
            # Target is the average of individual expert moments
            target_mean = torch.stack(expert_means).mean(dim=0)
            target_std = torch.stack(expert_stds).mean(dim=0)
            
            # Compute merged (source) statistics
            source_mean = param.mean(dim=reduce_dims, keepdim=True)
            source_std = param.std(dim=reduce_dims, keepdim=True)
            
            # Parametric alignment
            if mode == 'both':
                calibrated = target_mean + (target_std / (source_std + 1e-8)) * (param - source_mean)
            elif mode == 'std_only':
                calibrated = (target_std / (source_std + 1e-8)) * param
            elif mode == 'mean_only':
                calibrated = target_mean + (param - source_mean)
            else:
                calibrated = param
            
            param.copy_(calibrated)

def calibrate_bn(backbone, dataloaders, num_samples=64):
    """
    Data-efficient BatchNorm Calibration (DE-BN)
    """
    backbone.train()
    # Reset running statistics and enable running mean/var calculation
    for m in backbone.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            m.reset_running_stats()
            m.momentum = None  # Cumulative average
            m.train()
            
    # Gather a small calibration set from all tasks
    cal_images = []
    samples_per_task = num_samples // len(dataloaders)
    for task_name, loader in dataloaders.items():
        count = 0
        for images, _ in loader['train']:
            cal_images.append(images)
            count += images.size(0)
            if count >= samples_per_task:
                break
                
    cal_images = torch.cat(cal_images, dim=0)[:num_samples].to(device)
    
    # Run forward passes to update BN running statistics
    with torch.no_grad():
        backbone(cal_images)
        
    # Restore standard evaluation mode
    backbone.eval()
    for m in backbone.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            m.momentum = 0.1  # Restore default

# ----------------------------------------------------
# 5. Main Execution Flow
# ----------------------------------------------------
def run_experiment(arch_type, loaders):
    print(f"\n====================================================")
    print(f"Running Experiments for Architecture: {arch_type.upper()}")
    print(f"====================================================")
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    # Train experts if not already trained
    for task in tasks:
        backbone_path = f'checkpoints/{arch_type}_{task}_backbone.pth'
        head_path = f'checkpoints/{arch_type}_{task}_head.pth'
        if not (os.path.exists(backbone_path) and os.path.exists(head_path)):
            train_expert(arch_type, task, loaders[task]['train'], epochs=5)
            
    # Load progenitor
    progenitor_backbone = get_progenitor(arch_type).to(device)
    progenitor_backbone.load_state_dict(torch.load(f'checkpoints/{arch_type}_progenitor.pth'))
    
    # Load expert backbones and heads
    expert_backbones = []
    expert_heads = {}
    for task in tasks:
        exp_back = get_progenitor(arch_type).to(device)
        exp_back.load_state_dict(torch.load(f'checkpoints/{arch_type}_{task}_backbone.pth'))
        expert_backbones.append(exp_back)
        
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(torch.load(f'checkpoints/{arch_type}_{task}_head.pth'))
        expert_heads[task] = head

    # Evaluate individual expert baselines (Self-Evaluation)
    print("\n--- Individual Expert Accuracy (Self) ---")
    for task, exp_back in zip(tasks, expert_backbones):
        acc = evaluate_model(exp_back, expert_heads[task], loaders[task]['test'])
        print(f"Expert on {task.upper()}: {acc:.2f}%")

    # Evaluate individual experts on other tasks (Cross-Evaluation)
    print("\n--- Cross-Task Accuracy Matrix ---")
    for i, t_exp in enumerate(tasks):
        for j, t_eval in enumerate(tasks):
            acc = evaluate_model(expert_backbones[i], expert_heads[t_eval], loaders[t_eval]['test'])
            print(f"Expert {t_exp.upper()} evaluated on {t_eval.upper()}: {acc:.2f}%")

    # ----------------------------------------------------
    # Baseline 1: Weight Averaging (WA)
    # ----------------------------------------------------
    print("\n--- Merging Method: Weight Averaging (WA) ---")
    merged_backbone_wa = get_progenitor(arch_type).to(device)
    
    # Compute average weights
    wa_state = {}
    for name, param in progenitor_backbone.state_dict().items():
        expert_params = [back.state_dict()[name] for back in expert_backbones]
        if torch.is_floating_point(param) or torch.is_complex(param):
            wa_state[name] = torch.stack(expert_params).mean(dim=0)
        else:
            wa_state[name] = expert_params[0].clone()
    merged_backbone_wa.load_state_dict(wa_state)
    
    wa_accs = {}
    for task in tasks:
        acc = evaluate_model(merged_backbone_wa, expert_heads[task], loaders[task]['test'])
        wa_accs[task] = acc
    wa_avg = np.mean(list(wa_accs.values()))
    print(f"WA Accuracy: MNIST: {wa_accs['mnist']:.2f}%, FMNIST: {wa_accs['fmnist']:.2f}%, CIFAR10: {wa_accs['cifar10']:.2f}% | Avg: {wa_avg:.2f}%")

    # ----------------------------------------------------
    # Baseline 1b: Weight Averaging with DE-BN (For ResNet-18)
    # ----------------------------------------------------
    wa_debn_avg = None
    if arch_type == 'resnet18':
        print("\n--- Merging Method: Weight Averaging + DE-BN Calibration (N=64) ---")
        merged_backbone_wa_debn = get_progenitor(arch_type).to(device)
        merged_backbone_wa_debn.load_state_dict(wa_state)
        calibrate_bn(merged_backbone_wa_debn, loaders, num_samples=64)
        
        wa_debn_accs = {}
        for task in tasks:
            acc = evaluate_model(merged_backbone_wa_debn, expert_heads[task], loaders[task]['test'])
            wa_debn_accs[task] = acc
        wa_debn_avg = np.mean(list(wa_debn_accs.values()))
        print(f"WA + DE-BN Accuracy: MNIST: {wa_debn_accs['mnist']:.2f}%, FMNIST: {wa_debn_accs['fmnist']:.2f}%, CIFAR10: {wa_debn_accs['cifar10']:.2f}% | Avg: {wa_debn_avg:.2f}%")

    # ----------------------------------------------------
    # Baseline 2: Task Arithmetic (TA) Sweep
    # ----------------------------------------------------
    print("\n--- Merging Method: Task Arithmetic (TA) Sweep ---")
    lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    ta_results = []
    
    init_state = progenitor_backbone.state_dict()
    
    # Compute task vectors: W_k - W_init
    task_vectors = []
    for back in expert_backbones:
        tv = {}
        for name, param in init_state.items():
            if torch.is_floating_point(param) or torch.is_complex(param):
                tv[name] = back.state_dict()[name] - param
            else:
                tv[name] = torch.zeros_like(param)
        task_vectors.append(tv)
        
    for lmbda in lambda_list:
        merged_backbone_ta = get_progenitor(arch_type).to(device)
        ta_state = {}
        for name, param in init_state.items():
            if torch.is_floating_point(param) or torch.is_complex(param):
                ta_state[name] = param + lmbda * torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
            else:
                ta_state[name] = param.clone()
        merged_backbone_ta.load_state_dict(ta_state)
        
        accs = {}
        for task in tasks:
            acc = evaluate_model(merged_backbone_ta, expert_heads[task], loaders[task]['test'])
            accs[task] = acc
        avg_acc = np.mean(list(accs.values()))
        ta_results.append({'lambda': lmbda, 'accs': accs, 'avg': avg_acc})
        print(f"TA λ={lmbda:.2f} | MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")

    # ----------------------------------------------------
    # Proposed Method: CMVA (with TA Sweep)
    # ----------------------------------------------------
    print("\n--- Merging Method: CMVA Calibration (Both Mean & Std) ---")
    cmva_both_results = []
    for lmbda in lambda_list:
        merged_backbone_cmva = get_progenitor(arch_type).to(device)
        ta_state = {}
        for name, param in init_state.items():
            if torch.is_floating_point(param) or torch.is_complex(param):
                ta_state[name] = param + lmbda * torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
            else:
                ta_state[name] = param.clone()
        merged_backbone_cmva.load_state_dict(ta_state)
        
        # Apply CMVA calibration
        apply_cmva(merged_backbone_cmva, expert_backbones, mode='both')
        
        accs = {}
        for task in tasks:
            acc = evaluate_model(merged_backbone_cmva, expert_heads[task], loaders[task]['test'])
            accs[task] = acc
        avg_acc = np.mean(list(accs.values()))
        cmva_both_results.append({'lambda': lmbda, 'accs': accs, 'avg': avg_acc})
        print(f"CMVA (Both) λ={lmbda:.2f} | MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")

    # ----------------------------------------------------
    # Ablation 1: CMVA (Std Only Calibration)
    # ----------------------------------------------------
    print("\n--- Ablation: CMVA (Std Only Calibration) ---")
    cmva_std_results = []
    for lmbda in lambda_list:
        merged_backbone_cmva = get_progenitor(arch_type).to(device)
        ta_state = {}
        for name, param in init_state.items():
            if torch.is_floating_point(param) or torch.is_complex(param):
                ta_state[name] = param + lmbda * torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
            else:
                ta_state[name] = param.clone()
        merged_backbone_cmva.load_state_dict(ta_state)
        
        apply_cmva(merged_backbone_cmva, expert_backbones, mode='std_only')
        
        accs = {}
        for task in tasks:
            acc = evaluate_model(merged_backbone_cmva, expert_heads[task], loaders[task]['test'])
            accs[task] = acc
        avg_acc = np.mean(list(accs.values()))
        cmva_std_results.append({'lambda': lmbda, 'accs': accs, 'avg': avg_acc})
        print(f"CMVA (Std Only) λ={lmbda:.2f} | MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")

    # ----------------------------------------------------
    # Ablation 2: CMVA (Mean Only Calibration)
    # ----------------------------------------------------
    print("\n--- Ablation: CMVA (Mean Only Calibration) ---")
    cmva_mean_results = []
    for lmbda in lambda_list:
        merged_backbone_cmva = get_progenitor(arch_type).to(device)
        ta_state = {}
        for name, param in init_state.items():
            if torch.is_floating_point(param) or torch.is_complex(param):
                ta_state[name] = param + lmbda * torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
            else:
                ta_state[name] = param.clone()
        merged_backbone_cmva.load_state_dict(ta_state)
        
        apply_cmva(merged_backbone_cmva, expert_backbones, mode='mean_only')
        
        accs = {}
        for task in tasks:
            acc = evaluate_model(merged_backbone_cmva, expert_heads[task], loaders[task]['test'])
            accs[task] = acc
        avg_acc = np.mean(list(accs.values()))
        cmva_mean_results.append({'lambda': lmbda, 'accs': accs, 'avg': avg_acc})
        print(f"CMVA (Mean Only) λ={lmbda:.2f} | MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")

    # Save metrics
    metrics = {
        'wa': {'accs': wa_accs, 'avg': wa_avg},
        'wa_debn': {'accs': None, 'avg': wa_debn_avg} if wa_debn_avg is None else {'accs': wa_debn_accs, 'avg': wa_debn_avg},
        'ta': ta_results,
        'cmva_both': cmva_both_results,
        'cmva_std': cmva_std_results,
        'cmva_mean': cmva_mean_results
    }
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_list, [r['avg'] for r in ta_results], label='Task Arithmetic (TA)', marker='o', linewidth=2)
    plt.plot(lambda_list, [r['avg'] for r in cmva_both_results], label='TA + CMVA (Both)', marker='s', linewidth=2)
    plt.plot(lambda_list, [r['avg'] for r in cmva_std_results], label='TA + CMVA (Std Only)', marker='^', linestyle='--')
    plt.plot(lambda_list, [r['avg'] for r in cmva_mean_results], label='TA + CMVA (Mean Only)', marker='v', linestyle='--')
    
    # Add horizontal lines for WA
    plt.axhline(y=wa_avg, color='r', linestyle=':', label='Weight Averaging (WA)')
    if wa_debn_avg is not None:
        plt.axhline(y=wa_debn_avg, color='g', linestyle='-.', label='WA + DE-BN')
        
    plt.title(f'Merging Performance on {arch_type.upper()}', fontsize=14)
    plt.xlabel('Scaling Parameter (lambda)', fontsize=12)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'plots/{arch_type}_merging_performance.png', dpi=300)
    plt.close()
    
    return metrics

if __name__ == '__main__':
    loaders = get_dataloaders()
    
    results = {}
    for arch in ['mlp', 'resnet18']:
        results[arch] = run_experiment(arch, loaders)
        
    with open('checkpoints/experimental_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nAll experiments completed successfully!")
