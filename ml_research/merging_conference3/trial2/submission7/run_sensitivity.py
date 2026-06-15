import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Set random seed for reproducibility
torch.manual_seed(20260613)
np.random.seed(20260613)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = os.path.expanduser('~/data')
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 128
TRAIN_SUBSET_SIZE = 2000
TEST_SUBSET_SIZE = 500
FINE_TUNE_EPOCHS = 3
TTA_STEPS = 50
LR_TTA = 0.01

BACKBONE_TYPE = 'resnet18'

transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataloaders():
    mnist_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_rgb)
    mnist_test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform_rgb)
    mnist_train = Subset(mnist_train, np.random.choice(len(mnist_train), TRAIN_SUBSET_SIZE, replace=False))
    mnist_test = Subset(mnist_test, np.random.choice(len(mnist_test), TEST_SUBSET_SIZE, replace=False))
    
    fmnist_train = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform_rgb)
    fmnist_test = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform_rgb)
    fmnist_train = Subset(fmnist_train, np.random.choice(len(fmnist_train), TRAIN_SUBSET_SIZE, replace=False))
    fmnist_test = Subset(fmnist_test, np.random.choice(len(fmnist_test), TEST_SUBSET_SIZE, replace=False))
    
    cifar_train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_color)
    cifar_train = Subset(cifar_train, np.random.choice(len(cifar_train), TRAIN_SUBSET_SIZE, replace=False))
    cifar_test = Subset(cifar_test, np.random.choice(len(cifar_test), TEST_SUBSET_SIZE, replace=False))
    
    svhn_train = torchvision.datasets.SVHN(root=DATA_DIR, split='train', download=True, transform=transform_color)
    svhn_test = torchvision.datasets.SVHN(root=DATA_DIR, split='test', download=True, transform=transform_color)
    svhn_train = Subset(svhn_train, np.random.choice(len(svhn_train), TRAIN_SUBSET_SIZE, replace=False))
    svhn_test = Subset(svhn_test, np.random.choice(len(svhn_test), TEST_SUBSET_SIZE, replace=False))
    
    train_loaders = {
        'MNIST': DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True),
        'FashionMNIST': DataLoader(fmnist_train, batch_size=BATCH_SIZE, shuffle=True),
        'CIFAR10': DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True),
        'SVHN': DataLoader(svhn_train, batch_size=BATCH_SIZE, shuffle=True)
    }
    
    test_loaders = {
        'MNIST': DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False),
        'FashionMNIST': DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False),
        'CIFAR10': DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=False),
        'SVHN': DataLoader(svhn_test, batch_size=BATCH_SIZE, shuffle=False)
    }
    
    return train_loaders, test_loaders

class SimpleCNNBackbone(nn.Module):
    def __init__(self):
        super(SimpleCNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, 128)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc(x))
        return x

def get_backbone():
    if BACKBONE_TYPE == 'resnet18':
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        return backbone
    else:
        return SimpleCNNBackbone()

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=128):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, 10)
        
    def forward(self, x):
        return self.fc(x)

def train_experts(train_loaders, test_loaders):
    print("Initializing shared backbone and task experts...")
    base_backbone = get_backbone().to(DEVICE)
    
    os.makedirs("checkpoints_cache", exist_ok=True)
    base_backbone_path = f"checkpoints_cache/base_backbone_{BACKBONE_TYPE}.pt"
    if not os.path.exists(base_backbone_path):
        torch.save(base_backbone.state_dict(), base_backbone_path)
    else:
        base_backbone.load_state_dict(torch.load(base_backbone_path, map_location=DEVICE))
        
    experts = {}
    heads = {}
    
    input_dim = 512 if BACKBONE_TYPE == 'resnet18' else 128
    
    for task_name, loader in train_loaders.items():
        backbone_path = f"checkpoints_cache/expert_{task_name}_backbone_{BACKBONE_TYPE}.pt"
        head_path = f"checkpoints_cache/expert_{task_name}_head_{BACKBONE_TYPE}.pt"
        
        backbone = get_backbone().to(DEVICE)
        head = ClassificationHead(input_dim).to(DEVICE)
        
        if os.path.exists(backbone_path) and os.path.exists(head_path):
            print(f"Loading cached Expert for {task_name}...")
            backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
            head.load_state_dict(torch.load(head_path, map_location=DEVICE))
        else:
            print(f"\n--- Fine-tuning Expert for {task_name} ---")
            backbone.load_state_dict(base_backbone.state_dict())
            
            if BACKBONE_TYPE == 'resnet18':
                for name, param in backbone.named_parameters():
                    if 'layer4' not in name:
                        param.requires_grad = False
                optimizer = torch.optim.Adam(list(backbone.layer4.parameters()) + list(head.parameters()), lr=1e-3)
            else:
                optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)
                
            criterion = nn.CrossEntropyLoss()
            
            backbone.train()
            head.train()
            for epoch in range(FINE_TUNE_EPOCHS):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(head(backbone(x)), y)
                    loss.backward()
                    optimizer.step()
                    
            # Save fine-tuned experts
            torch.save(backbone.state_dict(), backbone_path)
            torch.save(head.state_dict(), head_path)
            
        backbone.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loaders[task_name]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = head(backbone(x))
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        print(f"Expert {task_name} Test Accuracy: {100. * correct / total:.2f}%")
        experts[task_name] = backbone
        heads[task_name] = head
    return base_backbone, experts, heads

def evaluate_merged_model(merged_state_dict, heads, test_loaders):
    backbone = get_backbone().to(DEVICE)
    backbone.load_state_dict(merged_state_dict, strict=False)
    backbone.eval()
    accuracies = {}
    for task_name, loader in test_loaders.items():
        head = heads[task_name]
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = head(backbone(x))
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracies[task_name] = 100. * correct / total
    return accuracies

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def run_thermomerge_with_params(base_backbone, experts, heads, train_loaders, test_loaders, T_start, beta):
    tasks = list(experts.keys())
    num_tasks = len(tasks)
    
    dummy_model = get_backbone().to(DEVICE)
    param_names = make_functional(dummy_model)
    num_layers = len(param_names)
    
    base_state = base_backbone.state_dict()
    base_params = [base_state[name].clone().detach().to(DEVICE) for name in param_names]
    
    expert_params_list = []
    for task in tasks:
        task_state = experts[task].state_dict()
        task_params = [task_state[name].clone().detach().to(DEVICE) for name in param_names]
        expert_params_list.append(task_params)
        
    streaming_iterators = {task: iter(train_loaders[task]) for task in tasks}
    def get_streaming_batch(task):
        nonlocal streaming_iterators
        try:
            x, _ = next(streaming_iterators[task])
        except StopIteration:
            streaming_iterators[task] = iter(train_loaders[task])
            x, _ = next(streaming_iterators[task])
        if x.size(0) < BATCH_SIZE:
            streaming_iterators[task] = iter(train_loaders[task])
            x, _ = next(streaming_iterators[task])
        return x.to(DEVICE)

    # ThermoMerge Optimization
    lambdas_raw_tm = (torch.ones(num_layers, num_tasks, device=DEVICE) * 0.3).detach().requires_grad_(True)
    tau_raw_tm = (torch.ones(num_tasks, device=DEVICE) * 1.0).detach().requires_grad_(True)
    optimizer_tm = torch.optim.Adam([lambdas_raw_tm, tau_raw_tm], lr=LR_TTA)
    
    T_end = 1.0
    for step in range(TTA_STEPS):
        optimizer_tm.zero_grad()
        loss = 0
        step_batches = {task: get_streaming_batch(task) for task in tasks}
        T_t = T_end + (T_start - T_end) * np.exp(-beta * step)
        lambdas = torch.clamp(lambdas_raw_tm, 0.0, 1.0)
        tau_k = torch.clamp(tau_raw_tm, 0.2, 5.0)
        
        merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx, task in enumerate(tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update = update + lambdas[l_idx, k_idx] * task_update
            merged_params.append(base_p + update)
            
        backbone_temp = get_backbone().to(DEVICE)
        _ = make_functional(backbone_temp)
        load_weights(backbone_temp, param_names, merged_params)
        
        for task_idx, task in enumerate(tasks):
            x = step_batches[task]
            T_k = tau_k[task_idx] * T_t
            with torch.no_grad():
                experts[task].eval()
                heads[task].eval()
                p_expert = F.softmax(heads[task](experts[task](x)) / T_k, dim=-1)
            logits = heads[task](backbone_temp(x))
            log_q_merged = F.log_softmax(logits / T_k, dim=-1)
            kl = F.kl_div(log_q_merged, p_expert, reduction='batchmean')
            loss += T_k * kl
            
        loss.backward()
        optimizer_tm.step()
        
    with torch.no_grad():
        lambdas_final = torch.clamp(lambdas_raw_tm, 0.0, 1.0)
        tm_merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx, task in enumerate(tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update += lambdas_final[l_idx, k_idx] * task_update
            tm_merged_params.append(base_p + update)
        tm_merged_state = {name: p for name, p in zip(param_names, tm_merged_params)}
        tm_acc = evaluate_merged_model(tm_merged_state, heads, test_loaders)
    return np.mean(list(tm_acc.values()))

if __name__ == "__main__":
    train_loaders, test_loaders = get_dataloaders()
    base_backbone, experts, heads = train_experts(train_loaders, test_loaders)
    
    # 1. Sensitivity Analysis for T_start
    print("\nEvaluating Sensitivity to T_start...")
    t_start_vals = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    t_start_accs = []
    for t_val in t_start_vals:
        acc = run_thermomerge_with_params(base_backbone, experts, heads, train_loaders, test_loaders, T_start=t_val, beta=0.05)
        print(f"T_start = {t_val:.1f} | Average Accuracy: {acc:.2f}%")
        t_start_accs.append(acc)
        
    # 2. Sensitivity Analysis for Beta (cooling rate)
    print("\nEvaluating Sensitivity to Cooling Rate beta...")
    beta_vals = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    beta_accs = []
    for b_val in beta_vals:
        acc = run_thermomerge_with_params(base_backbone, experts, heads, train_loaders, test_loaders, T_start=5.0, beta=b_val)
        print(f"beta = {b_val:.2f} | Average Accuracy: {acc:.2f}%")
        beta_accs.append(acc)
        
    # Plotting both
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: T_start
    plt.subplot(1, 2, 1)
    plt.plot(t_start_vals, t_start_accs, marker='o', linewidth=2, color='blue')
    plt.xlabel('Initial Temperature ($T_{start}$)')
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Sensitivity to Initial Temperature')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Beta
    plt.subplot(1, 2, 2)
    plt.plot(beta_vals, beta_accs, marker='s', linewidth=2, color='green')
    plt.xlabel('Cooling Rate ($\\beta$)')
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('Sensitivity to Cooling Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('sensitivity_plot.png', dpi=300)
    plt.close()
    print("Sensitivity analysis complete! Figure saved as 'sensitivity_plot.png'")
