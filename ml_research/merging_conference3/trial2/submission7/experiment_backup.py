import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Set random seed for reproducibility
torch.manual_seed(20260613)
np.random.seed(20260613)

# Configuration and Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = os.path.expanduser('~/data')
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 128
TRAIN_SUBSET_SIZE = 2000
TEST_SUBSET_SIZE = 500
FINE_TUNE_EPOCHS = 3
TTA_STEPS = 100
LR_TTA = 0.01

# 1. Dataset Preprocessing & Loading
# Grayscale to RGB transformation for MNIST and FashionMNIST
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
    print("Loading datasets and creating fast subsets...")
    
    # MNIST
    mnist_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_rgb)
    mnist_test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform_rgb)
    mnist_train = Subset(mnist_train, np.random.choice(len(mnist_train), TRAIN_SUBSET_SIZE, replace=False))
    mnist_test = Subset(mnist_test, np.random.choice(len(mnist_test), TEST_SUBSET_SIZE, replace=False))
    
    # FashionMNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform_rgb)
    fmnist_test = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform_rgb)
    fmnist_train = Subset(fmnist_train, np.random.choice(len(fmnist_train), TRAIN_SUBSET_SIZE, replace=False))
    fmnist_test = Subset(fmnist_test, np.random.choice(len(fmnist_test), TEST_SUBSET_SIZE, replace=False))
    
    # CIFAR-10
    cifar_train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_color)
    cifar_train = Subset(cifar_train, np.random.choice(len(cifar_train), TRAIN_SUBSET_SIZE, replace=False))
    cifar_test = Subset(cifar_test, np.random.choice(len(cifar_test), TEST_SUBSET_SIZE, replace=False))
    
    # SVHN
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

# 2. Model Architecture
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

class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.fc(x)

# 3. Model Training (Fine-tuning downstream experts)
def train_experts(train_loaders, test_loaders):
    print("Initializing shared backbone and task experts...")
    base_backbone = SimpleCNNBackbone().to(DEVICE)
    
    experts = {}
    heads = {}
    
    for task_name, loader in train_loaders.items():
        print(f"\n--- Fine-tuning Expert for {task_name} ---")
        # Initialize expert as copy of base backbone
        backbone = SimpleCNNBackbone().to(DEVICE)
        backbone.load_state_dict(base_backbone.state_dict())
        head = ClassificationHead().to(DEVICE)
        
        optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        backbone.train()
        head.train()
        for epoch in range(FINE_TUNE_EPOCHS):
            total_loss = 0
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                features = backbone(x)
                logits = head(features)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * x.size(0)
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            
            epoch_loss = total_loss / total
            epoch_acc = 100. * correct / total
            print(f"Epoch {epoch+1}/{FINE_TUNE_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
            
        # Evaluation on test subset
        backbone.eval()
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loaders[task_name]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                features = backbone(x)
                logits = head(features)
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        test_acc = 100. * correct / total
        print(f"Expert {task_name} Test Accuracy: {test_acc:.2f}%")
        
        experts[task_name] = backbone
        heads[task_name] = head
        
    return base_backbone, experts, heads

# 4. Merging Infrastructure (Layer-wise)
def get_layer_names():
    # Target trainable layers of SimpleCNNBackbone
    return ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc.weight', 'fc.bias']

def merge_parameters(base_state_dict, expert_state_dicts, lambdas, layer_names):
    """
    Fuses layers layer-by-layer:
    theta_MTL = theta_base + sum_k lambdas[l, k] * (theta_expert_k - theta_base)
    """
    merged_state_dict = {}
    tasks = list(expert_state_dicts.keys())
    
    for key in base_state_dict.keys():
        if key in layer_names:
            l_idx = layer_names.index(key)
            base_val = base_state_dict[key]
            update = torch.zeros_like(base_val)
            for k_idx, task in enumerate(tasks):
                task_update = expert_state_dicts[task][key] - base_val
                update += lambdas[l_idx, k_idx] * task_update
            merged_state_dict[key] = base_val + update
        else:
            merged_state_dict[key] = base_state_dict[key].clone()
            
    return merged_state_dict

def evaluate_merged_model(merged_state_dict, heads, test_loaders):
    backbone = SimpleCNNBackbone().to(DEVICE)
    backbone.load_state_dict(merged_state_dict)
    backbone.eval()
    
    accuracies = {}
    for task_name, loader in test_loaders.items():
        head = heads[task_name]
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                features = backbone(x)
                logits = head(features)
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracies[task_name] = 100. * correct / total
    return accuracies

# Differentiable parameter mapping helpers
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

# Entropy calculation for AdaMerging
def softmax_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

# 5. TEST-TIME ADAPTATION BASELINES & proposed ThermoMerge
def run_model_merging_experiments(base_backbone, experts, heads, train_loaders, test_loaders):
    tasks = list(experts.keys())
    num_tasks = len(tasks)
    
    # 1. Get parameter names from a dummy backbone
    dummy_model = SimpleCNNBackbone().to(DEVICE)
    param_names = make_functional(dummy_model)
    num_layers = len(param_names)
    
    # 2. Extract base and expert parameters
    base_state = base_backbone.state_dict()
    base_params = [base_state[name].clone().detach().to(DEVICE) for name in param_names]
    
    expert_params_list = []
    for task in tasks:
        task_state = experts[task].state_dict()
        task_params = [task_state[name].clone().detach().to(DEVICE) for name in param_names]
        expert_params_list.append(task_params)
        
    results = {}
    trajectories = {}
    
    # -------------------------------------------------------------
    # BASELINE 1: Task Arithmetic (Linear Averaging, uniform lambda = 0.3)
    # -------------------------------------------------------------
    print("\n--- Running Baseline 1: Task Arithmetic (uniform 0.3) ---")
    lambdas_ta = torch.ones(num_layers, num_tasks, device=DEVICE) * 0.3
    
    ta_merged_params = []
    for l_idx in range(num_layers):
        base_p = base_params[l_idx]
        update = torch.zeros_like(base_p)
        for k_idx in range(num_tasks):
            task_update = expert_params_list[k_idx][l_idx] - base_p
            update += lambdas_ta[l_idx, k_idx] * task_update
        ta_merged_params.append(base_p + update)
        
    ta_merged_state = {name: p for name, p in zip(param_names, ta_merged_params)}
    ta_acc = evaluate_merged_model(ta_merged_state, heads, test_loaders)
    results['Task_Arithmetic'] = ta_acc
    print(f"Task Arithmetic Avg Acc: {np.mean(list(ta_acc.values())):.2f}%")
    
    # Prepare batch data for fast unsupervised TTA
    tta_batches = {task: next(iter(train_loaders[task]))[0].to(DEVICE) for task in tasks}
    
    # -------------------------------------------------------------
    # BASELINE 2: AdaMerging (Entropy Minimization TTA)
    # -------------------------------------------------------------
    print("\n--- Running Baseline 2: AdaMerging (Entropy Min) ---")
    # Trainable parameter: raw lambdas
    lambdas_raw = (torch.ones(num_layers, num_tasks, device=DEVICE) * 0.3).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([lambdas_raw], lr=LR_TTA)
    
    traj_ada = []
    for step in range(TTA_STEPS):
        optimizer.zero_grad()
        loss = 0
        
        # Clamp lambdas to [0, 1]
        lambdas = torch.clamp(lambdas_raw, 0.0, 1.0)
        
        # Merge parameters differentiably
        merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx in range(num_tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update = update + lambdas[l_idx, k_idx] * task_update
            merged_params.append(base_p + update)
            
        # Create fresh functional model for this forward pass
        backbone_temp = SimpleCNNBackbone().to(DEVICE)
        _ = make_functional(backbone_temp)
        load_weights(backbone_temp, param_names, merged_params)
        backbone_temp.train()
        
        # Calculate entropy on unlabeled batches across all tasks
        for task_idx, task in enumerate(tasks):
            x = tta_batches[task]
            features = backbone_temp(x)
            logits = heads[task](features)
            loss += softmax_entropy(logits)
            
        loss.backward()
        optimizer.step()
        traj_ada.append(loss.item())
        
        if (step+1) % 20 == 0:
            print(f"Step {step+1}/{TTA_STEPS} | Unsupervised Entropy Loss: {loss.item():.4f}")
            
    # Final evaluation
    with torch.no_grad():
        lambdas_final = torch.clamp(lambdas_raw, 0.0, 1.0)
        ada_merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx in range(num_tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update += lambdas_final[l_idx, k_idx] * task_update
            ada_merged_params.append(base_p + update)
        ada_merged_state = {name: p for name, p in zip(param_names, ada_merged_params)}
        ada_acc = evaluate_merged_model(ada_merged_state, heads, test_loaders)
    results['AdaMerging'] = ada_acc
    trajectories['AdaMerging'] = traj_ada
    print(f"AdaMerging Avg Acc: {np.mean(list(ada_acc.values())):.2f}%")
    
    # -------------------------------------------------------------
    # BASELINE 3: SyMerge (Teacher Alignment / Self-Labeling TTA)
    # -------------------------------------------------------------
    print("\n--- Running Baseline 3: SyMerge (Teacher-Student Alignment) ---")
    lambdas_raw_sy = (torch.ones(num_layers, num_tasks, device=DEVICE) * 0.3).detach().requires_grad_(True)
    optimizer_sy = torch.optim.Adam([lambdas_raw_sy], lr=LR_TTA)
    
    # Compute teacher (expert) targets once for stability
    teacher_preds = {}
    with torch.no_grad():
        for task in tasks:
            x = tta_batches[task]
            experts[task].eval()
            heads[task].eval()
            features = experts[task](x)
            logits = heads[task](features)
            teacher_preds[task] = F.softmax(logits, dim=-1)
            
    traj_sy = []
    for step in range(TTA_STEPS):
        optimizer_sy.zero_grad()
        loss = 0
        
        lambdas = torch.clamp(lambdas_raw_sy, 0.0, 1.0)
        merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx in range(num_tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update = update + lambdas[l_idx, k_idx] * task_update
            merged_params.append(base_p + update)
            
        backbone_temp = SimpleCNNBackbone().to(DEVICE)
        _ = make_functional(backbone_temp)
        load_weights(backbone_temp, param_names, merged_params)
        
        for task_idx, task in enumerate(tasks):
            x = tta_batches[task]
            features = backbone_temp(x)
            logits = heads[task](features)
            log_q = F.log_softmax(logits, dim=-1)
            p = teacher_preds[task]
            loss += F.kl_div(log_q, p, reduction='batchmean')
            
        loss.backward()
        optimizer_sy.step()
        traj_sy.append(loss.item())
        
        if (step+1) % 20 == 0:
            print(f"Step {step+1}/{TTA_STEPS} | Alignment KL Loss: {loss.item():.4f}")
            
    with torch.no_grad():
        lambdas_final = torch.clamp(lambdas_raw_sy, 0.0, 1.0)
        sy_merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx in range(num_tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update += lambdas_final[l_idx, k_idx] * task_update
            sy_merged_params.append(base_p + update)
        sy_merged_state = {name: p for name, p in zip(param_names, sy_merged_params)}
        sy_acc = evaluate_merged_model(sy_merged_state, heads, test_loaders)
    results['SyMerge'] = sy_acc
    trajectories['SyMerge'] = traj_sy
    print(f"SyMerge Avg Acc: {np.mean(list(sy_acc.values())):.2f}%")
    
    # -------------------------------------------------------------
    # PROPOSED: ThermoMerge (Thermodynamic Free Energy Minimization)
    # -------------------------------------------------------------
    print("\n--- Running Proposed: ThermoMerge (Thermodynamic Model Merging) ---")
    lambdas_raw_tm = (torch.ones(num_layers, num_tasks, device=DEVICE) * 0.3).detach().requires_grad_(True)
    optimizer_tm = torch.optim.Adam([lambdas_raw_tm], lr=LR_TTA)
    
    # Pre-compute expert logits for free energy calculation
    expert_logits = {}
    with torch.no_grad():
        for task in tasks:
            x = tta_batches[task]
            features = experts[task](x)
            expert_logits[task] = heads[task](features)
            
    traj_tm = []
    
    # Annealing Schedule parameters
    T_start = 5.0
    T_end = 1.0
    beta = 0.05 # Cooling rate
    
    for step in range(TTA_STEPS):
        optimizer_tm.zero_grad()
        loss = 0
        
        # Determine physical temperature at step t
        T_t = T_end + (T_start - T_end) * np.exp(-beta * step)
        
        lambdas = torch.clamp(lambdas_raw_tm, 0.0, 1.0)
        merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx in range(num_tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update = update + lambdas[l_idx, k_idx] * task_update
            merged_params.append(base_p + update)
            
        backbone_temp = SimpleCNNBackbone().to(DEVICE)
        _ = make_functional(backbone_temp)
        load_weights(backbone_temp, param_names, merged_params)
        
        for task_idx, task in enumerate(tasks):
            x = tta_batches[task]
            features = backbone_temp(x)
            merged_logits = heads[task](features)
            
            # Compute expert Boltzmann distribution at physical temperature T_t
            p_expert = F.softmax(expert_logits[task] / T_t, dim=-1)
            # Compute merged model Boltzmann log-distribution
            log_q_merged = F.log_softmax(merged_logits / T_t, dim=-1)
            
            # Physical Helmholtz Free Energy Discrepancy (proportional to T_t * KL)
            kl = F.kl_div(log_q_merged, p_expert, reduction='batchmean')
            loss += T_t * kl
            
        loss.backward()
        optimizer_tm.step()
        traj_tm.append(loss.item())
        
        if (step+1) % 20 == 0:
            print(f"Step {step+1}/{TTA_STEPS} | T={T_t:.2f} | Free Energy Discrepancy: {loss.item():.4f}")
            
    with torch.no_grad():
        lambdas_final = torch.clamp(lambdas_raw_tm, 0.0, 1.0)
        tm_merged_params = []
        for l_idx in range(num_layers):
            base_p = base_params[l_idx]
            update = torch.zeros_like(base_p)
            for k_idx in range(num_tasks):
                task_update = expert_params_list[k_idx][l_idx] - base_p
                update += lambdas_final[l_idx, k_idx] * task_update
            tm_merged_params.append(base_p + update)
        tm_merged_state = {name: p for name, p in zip(param_names, tm_merged_params)}
        tm_acc = evaluate_merged_model(tm_merged_state, heads, test_loaders)
    results['ThermoMerge'] = tm_acc
    trajectories['ThermoMerge'] = traj_tm
    print(f"ThermoMerge Avg Acc: {np.mean(list(tm_acc.values())):.2f}%")
    
    return results, trajectories

# 6. Main execution flow
if __name__ == "__main__":
    start_time = time.time()
    
    # Step 1: Load and partition datasets
    train_loaders, test_loaders = get_dataloaders()
    
    # Step 2: Fine-tune task-specific experts
    base_backbone, experts, heads = train_experts(train_loaders, test_loaders)
    
    # Step 3: Run model merging TTA experiments
    results, trajectories = run_model_merging_experiments(base_backbone, experts, heads, train_loaders, test_loaders)
    
    # Step 4: Write and save results
    print("\n--- Final Consolidated Multi-Task Merging Results ---")
    headers = ["Method", "MNIST", "FashionMNIST", "CIFAR-10", "SVHN", "Average"]
    print("=" * 75)
    print(f"{headers[0]:<18} | {headers[1]:<8} | {headers[2]:<12} | {headers[3]:<8} | {headers[4]:<8} | {headers[5]:<8}")
    print("-" * 75)
    
    json_results = {}
    for method, accs in results.items():
        mnist = accs['MNIST']
        fmnist = accs['FashionMNIST']
        cifar = accs['CIFAR10']
        svhn = accs['SVHN']
        avg = np.mean([mnist, fmnist, cifar, svhn])
        print(f"{method:<18} | {mnist:>7.2f}% | {fmnist:>11.2f}% | {cifar:>7.2f}% | {svhn:>7.2f}% | {avg:>7.2f}%")
        
        json_results[method] = {
            'MNIST': mnist,
            'FashionMNIST': fmnist,
            'CIFAR10': cifar,
            'SVHN': svhn,
            'Average': avg
        }
    print("=" * 75)
    
    # Save metrics.json
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Metrics saved to '{metrics_path}'")
    
    # Step 5: Generate and save figures
    print("\nGenerating figures...")
    
    # Fig 1: Optimization Trajectory
    plt.figure(figsize=(8, 5))
    plt.plot(trajectories['AdaMerging'], label='AdaMerging (Entropy Min)', color='orange')
    plt.plot(trajectories['SyMerge'], label='SyMerge (Teacher KL)', color='green')
    plt.plot(trajectories['ThermoMerge'], label='ThermoMerge (Ours, Free Energy)', color='blue', linewidth=2)
    plt.xlabel('Test-Time Optimization Steps')
    plt.ylabel('Unsupervised Objective Loss')
    plt.title('Optimization Trajectories of Unsupervised TTA Merging')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('optimization_trajectory.png', dpi=300)
    plt.close()
    
    # Fig 2: Accuracy Comparison
    methods = list(json_results.keys())
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN', 'Average']
    x = np.arange(len(tasks))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        acc_values = [json_results[method][task] for task in tasks]
        plt.bar(x + i*width - 1.5*width, acc_values, width, label=method)
        
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Performance Comparison Across Downstream Vision Tasks')
    plt.xticks(x, tasks)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    plt.close()
    
    print("Figures saved as 'optimization_trajectory.png' and 'accuracy_comparison.png'")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
