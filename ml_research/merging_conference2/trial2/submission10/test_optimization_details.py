import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Model Definition
class MultiTaskResNet18(nn.Module):
    def __init__(self, tasks=['mnist', 'fashion', 'cifar10']):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in tasks
        })
    def forward(self, x, task):
        features = self.backbone(x)
        return self.heads[task](features)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir='./data'):
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
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

def create_subsets(train_ds, test_ds, n_train=2000, n_test=500, n_calib=128, seed=42):
    set_seed(seed)
    sub_train, sub_test, sub_calib = {}, {}, {}
    for task in train_ds.keys():
        len_tr = len(train_ds[task])
        indices_tr = np.random.choice(len_tr, n_train, replace=False)
        sub_train[task] = Subset(train_ds[task], indices_tr)
        rem_indices = list(set(range(len_tr)) - set(indices_tr))
        indices_cal = np.random.choice(rem_indices, n_calib, replace=False)
        sub_calib[task] = Subset(train_ds[task], indices_cal)
        len_te = len(test_ds[task])
        indices_te = np.random.choice(len_te, n_test, replace=False)
        sub_test[task] = Subset(test_ds[task], indices_te)
    return sub_train, sub_test, sub_calib

def get_backbone_state(model):
    return {k: v.clone().cpu() for k, v in model.backbone.state_dict().items()}

def set_backbone_state(model, state_dict):
    model.backbone.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in state_dict.items()})

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

def run_sft_exp(model, ta_state, expert_heads, calib_loaders, test_loaders, tasks, device, 
                epochs=15, lr=1e-3, weight_decay=1e-4, optimizer_name='AdamW'):
    set_backbone_state(model, ta_state)
    sft_acc = {}
    for task in tasks:
        head_copy = nn.Linear(512, 10).to(device)
        head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(head_copy.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(head_copy.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(head_copy.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")
            
        model.eval()
        for epoch in range(epochs):
            for x, y in calib_loaders[task]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = model.backbone(x)
                outputs = head_copy(features)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                optimizer.step()
                
        # Evaluate
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
    return np.mean(list(sft_acc.values()))

def run_tta_exp(model, ta_state, expert_states, expert_heads, calib_loaders, test_loaders, tasks, device, 
                epochs=15, lr=1e-3, weight_decay=1e-4, optimizer_name='AdamW'):
    set_backbone_state(model, ta_state)
    distill_acc = {}
    for task in tasks:
        # Teacher model
        teacher_backbone = expert_states[task]
        teacher_head = {k: v.to(device) for k, v in expert_heads[task].items()}
        
        head_copy = nn.Linear(512, 10).to(device)
        head_copy.load_state_dict({k: v.to(device) for k, v in expert_heads[task].items()})
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(head_copy.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(head_copy.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(head_copy.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")
            
        t_model = MultiTaskResNet18().to(device)
        set_backbone_state(t_model, teacher_backbone)
        t_model.heads[task].load_state_dict(teacher_head)
        t_model.eval()
        
        model.eval()
        for epoch in range(epochs):
            for x, _ in calib_loaders[task]:
                x = x.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_features = t_model.backbone(x)
                    teacher_logits = t_model.heads[task](teacher_features)
                    student_features = model.backbone(x)
                student_logits = head_copy(student_features)
                
                # Soft KL loss
                loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits / 2.0, dim=-1),
                    reduction='batchmean'
                ) * (2.0 ** 2)
                loss.backward()
                optimizer.step()
                
        # Evaluate
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
    return np.mean(list(distill_acc.values()))

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    train_ds, test_ds = load_data()
    tasks = ['mnist', 'fashion', 'cifar10']

    # Load Base and Expert Weights
    base_state = torch.load('base_backbone.pt', map_location='cpu')
    expert_states = {}
    expert_heads = {}
    for task in tasks:
        expert_states[task] = torch.load(f'expert_backbone_{task}.pt', map_location='cpu')
        expert_heads[task] = torch.load(f'expert_head_{task}.pt', map_location='cpu')

    model = MultiTaskResNet18().to(device)
    lambdas = {'mnist': 0.3, 'fashion': 0.3, 'cifar10': 0.3}
    ta_state = merge_models(base_state, expert_states, lambdas)

    # Use standard N = 128
    sub_train, sub_test, sub_calib = create_subsets(train_ds, test_ds, n_calib=128)
    calib_loaders = {t: DataLoader(sub_calib[t], batch_size=128, shuffle=False) for t in sub_calib.keys()}
    test_loaders = {t: DataLoader(sub_test[t], batch_size=128, shuffle=False) for t in sub_test.keys()}

    # 1. Sweep Optimization Epochs
    epoch_values = [1, 3, 5, 10, 15, 20, 30]
    epoch_sft = []
    epoch_tta = []
    print("\n--- Sweeping Optimization Epochs ---")
    for ep in epoch_values:
        sft_acc = run_sft_exp(model, ta_state, expert_heads, calib_loaders, test_loaders, tasks, device, epochs=ep)
        tta_acc = run_tta_exp(model, ta_state, expert_states, expert_heads, calib_loaders, test_loaders, tasks, device, epochs=ep)
        epoch_sft.append(sft_acc)
        epoch_tta.append(tta_acc)
        print(f"  Epochs={ep}: SFT={sft_acc:.2f}%, TTA={tta_acc:.2f}%")

    # 2. Sweep Weight Decay
    wd_values = [0, 1e-6, 1e-4, 1e-2, 1e-1]
    wd_sft = []
    wd_tta = []
    print("\n--- Sweeping Weight Decay ---")
    for wd in wd_values:
        sft_acc = run_sft_exp(model, ta_state, expert_heads, calib_loaders, test_loaders, tasks, device, weight_decay=wd)
        tta_acc = run_tta_exp(model, ta_state, expert_states, expert_heads, calib_loaders, test_loaders, tasks, device, weight_decay=wd)
        wd_sft.append(sft_acc)
        wd_tta.append(tta_acc)
        print(f"  Weight Decay={wd}: SFT={sft_acc:.2f}%, TTA={tta_acc:.2f}%")

    # 3. Compare Optimizers
    opt_names = ['AdamW', 'SGD', 'RMSprop']
    opt_sft = []
    opt_tta = []
    print("\n--- Comparing Optimizers ---")
    for opt in opt_names:
        sft_acc = run_sft_exp(model, ta_state, expert_heads, calib_loaders, test_loaders, tasks, device, optimizer_name=opt)
        tta_acc = run_tta_exp(model, ta_state, expert_states, expert_heads, calib_loaders, test_loaders, tasks, device, optimizer_name=opt)
        opt_sft.append(sft_acc)
        opt_tta.append(tta_acc)
        print(f"  Optimizer={opt}: SFT={sft_acc:.2f}%, TTA={tta_acc:.2f}%")

    # Save to text file
    with open('optimization_details.txt', 'w') as f:
        f.write('=== Optimization Convergence Sweeps (Average Accuracy %) ===\n')
        f.write('Epochs | Supervised Head SFT | Unsupervised Head TTA\n')
        for i, ep in enumerate(epoch_values):
            f.write(f'{ep} | {epoch_sft[i]:.2f}% | {epoch_tta[i]:.2f}%\n')
            
        f.write('\n=== Weight Decay Robustness Sweeps ===\n')
        f.write('Weight Decay | Supervised Head SFT | Unsupervised Head TTA\n')
        for i, wd in enumerate(wd_values):
            f.write(f'{wd} | {wd_sft[i]:.2f}% | {wd_tta[i]:.2f}%\n')
            
        f.write('\n=== Optimizer Robustness Sweeps ===\n')
        f.write('Optimizer | Supervised Head SFT | Unsupervised Head TTA\n')
        for i, opt in enumerate(opt_names):
            f.write(f'{opt} | {opt_sft[i]:.2f}% | {opt_tta[i]:.2f}%\n')

    # Generate Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left: Convergence
    axes[0].plot(epoch_values, epoch_sft, marker='o', label='Supervised Head SFT', color='#3498db', linewidth=2)
    axes[0].plot(epoch_values, epoch_tta, marker='s', label='Unsupervised Head TTA', color='#2ecc71', linewidth=2)
    axes[0].set_xlabel('Optimization Epochs')
    axes[0].set_ylabel('Average Accuracy (%)')
    axes[0].set_title('Convergence Speed (N=128)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Middle: Weight Decay
    axes[1].plot([str(wd) for wd in wd_values], wd_sft, marker='o', label='Supervised Head SFT', color='#3498db', linewidth=2)
    axes[1].plot([str(wd) for wd in wd_values], wd_tta, marker='s', label='Unsupervised Head TTA', color='#2ecc71', linewidth=2)
    axes[1].set_xlabel('Weight Decay Parameter')
    axes[1].set_ylabel('Average Accuracy (%)')
    axes[1].set_title('Sensitivity to Weight Decay (N=128)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    # Right: Optimizer Comparison
    x = np.arange(len(opt_names))
    axes[2].bar(x - 0.2, opt_sft, 0.4, label='Supervised Head SFT', color='#3498db')
    axes[2].bar(x + 0.2, opt_tta, 0.4, label='Unsupervised Head TTA', color='#2ecc71')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(opt_names)
    axes[2].set_xlabel('Optimizer Type')
    axes[2].set_ylabel('Average Accuracy (%)')
    axes[2].set_title('Robustness to Optimizer Choice (N=128)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('optimization_details_plot.png', dpi=300)
    print("Optimization details swept and saved successfully!")

if __name__ == '__main__':
    main()
