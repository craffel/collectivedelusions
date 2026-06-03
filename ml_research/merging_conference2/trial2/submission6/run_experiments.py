import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to bypass initialization/mismatch errors on this cluster
    torch.backends.cudnn.enabled = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Dataset and Dataloader Setup
def get_dataloaders(batch_size=128):
    # Grayscale conversion and normalization
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Convert 1-channel to 3-channel
    ])

    fashion_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Convert 1-channel to 3-channel
    ])

    cifar_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Datasets
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=fashion_transform)
    test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=fashion_transform)

    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

    # Dataloaders
    loaders = {
        'mnist': {
            'train': DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'test': DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            'train_dataset': train_mnist
        },
        'fashion': {
            'train': DataLoader(train_fashion, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'test': DataLoader(test_fashion, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            'train_dataset': train_fashion
        },
        'cifar': {
            'train': DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'test': DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            'train_dataset': train_cifar
        }
    }
    return loaders

# 2. Model Structure Definition
def get_base_backbone():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity() # Shared backbone outputs 512-dim features
    return model

class ExpertModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# 3. Training Experts
def train_expert(name, train_loader, val_loader, epochs=5):
    print(f"\n--- Training Expert for {name.upper()} ---")
    backbone = get_base_backbone().to(device)
    head = nn.Linear(512, 10).to(device)
    model = ExpertModel(backbone, head)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

    # Evaluate on test set
    test_acc = evaluate_model(model, val_loader)
    print(f"Expert {name.upper()} Test Accuracy: {test_acc:.2f}%")
    return model, test_acc

def evaluate_model(model, loader, bn_mode='eval'):
    model.eval()
    
    # Special BN Mode handling
    if bn_mode == 'train':
        # Force all BatchNorm2d modules to be in training mode (using batch statistics)
        # while keeping momentum = 0.0 to prevent changing the running stats
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.training = True
                module.momentum = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

# 4. Calibration Helpers for TCAC and REPAIR
class StatsTracker:
    def __init__(self):
        self.outputs = []
    def hook(self, module, input, output):
        self.outputs.append(output.detach().cpu())
    def get_stats(self):
        all_outputs = torch.cat(self.outputs, dim=0) # (N, C, H, W)
        mean = all_outputs.mean(dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.sqrt(all_outputs.var(dim=(0, 2, 3), keepdim=True) + 1e-5).to(device)
        return mean, std

class CalibratorHook:
    def __init__(self, mean_orig, std_orig, mean_merged, std_merged):
        self.mean_orig = mean_orig
        self.std_orig = std_orig
        self.mean_merged = mean_merged
        self.std_merged = std_merged
    def hook(self, module, input, output):
        return (output - self.mean_merged) / self.std_merged * self.std_orig + self.mean_orig

def collect_calibration_stats(model, cal_loader, num_samples=128):
    model.eval()
    trackers = {}
    hooks = []
    
    # Register trackers on all BatchNorm2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            tracker = StatsTracker()
            trackers[name] = tracker
            hooks.append(module.register_forward_hook(tracker.hook))
            
    # Run calibration set
    samples_collected = 0
    with torch.no_grad():
        for images, _ in cal_loader:
            images = images.to(device)
            model(images)
            samples_collected += images.size(0)
            if samples_collected >= num_samples:
                break
                
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Get statistics
    stats = {}
    for name, tracker in trackers.items():
        mean, std = tracker.get_stats()
        stats[name] = (mean, std)
    return stats

def recalibrate_bn_exact(model, cal_loader):
    model.train()
    # Set momentum = 1.0 to exactly replace running stats with batch stats of the calibration set
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.training = True
            module.momentum = 1.0
            
    with torch.no_grad():
        for images, _ in cal_loader:
            images = images.to(device)
            model(images)

# 5. Merging Functions
def merge_weight_averaging(expert_states):
    merged_state = {}
    for key in expert_states[0].keys():
        if expert_states[0][key].dtype.is_floating_point:
            stacked = torch.stack([state[key] for state in expert_states], dim=0)
            merged_state[key] = stacked.mean(dim=0)
        else:
            merged_state[key] = expert_states[0][key].clone()
    return merged_state

def merge_task_arithmetic(pretrained_state, expert_states, lmbda):
    merged_state = {}
    for key in pretrained_state.keys():
        if pretrained_state[key].dtype.is_floating_point:
            # Move to the device of the expert weights
            dev = expert_states[0][key].device
            p_val = pretrained_state[key].to(dev)
            update = torch.zeros_like(p_val, device=dev)
            for exp_state in expert_states:
                update += (exp_state[key] - p_val)
            merged_state[key] = p_val + lmbda * update
        else:
            merged_state[key] = pretrained_state[key].clone()
    return merged_state

def apply_tsbm(model, merged_backbone_state, expert_k_backbone_state):
    # Load the merged backbone weights
    model.backbone.load_state_dict(merged_backbone_state, strict=True)
    # Keep BN parameters task-specific
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module_state = {}
            prefix = name + '.' if name else ''
            for key, val in expert_k_backbone_state.items():
                if key.startswith(prefix):
                    sub_key = key[len(prefix):]
                    if '.' not in sub_key:
                        module_state[sub_key] = val
            module.load_state_dict(module_state, strict=True)

# Main Execution Flow
def main():
    loaders = get_dataloaders()
    tasks = ['mnist', 'fashion', 'cifar']

    # 1. Train or load experts
    expert_models = {}
    expert_backbone_states = []
    expert_accs = {}
    
    pretrained_backbone = get_base_backbone()
    pretrained_backbone_state = copy.deepcopy(pretrained_backbone.state_dict())

    for task in tasks:
        ckpt_path = f"expert_{task}.pt"
        if os.path.exists(ckpt_path):
            print(f"Loading expert for {task.upper()} from checkpoint...")
            backbone = get_base_backbone().to(device)
            head = nn.Linear(512, 10).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            backbone.load_state_dict(ckpt['backbone'])
            head.load_state_dict(ckpt['head'])
            model = ExpertModel(backbone, head)
            expert_models[task] = model
            test_acc = evaluate_model(model, loaders[task]['test'])
            expert_accs[task] = test_acc
            print(f"Loaded {task.upper()} Expert Test Accuracy: {test_acc:.2f}%")
        else:
            # Train and save
            model, test_acc = train_expert(task, loaders[task]['train'], loaders[task]['test'], epochs=5)
            expert_models[task] = model
            expert_accs[task] = test_acc
            torch.save({
                'backbone': model.backbone.state_dict(),
                'head': model.head.state_dict()
            }, ckpt_path)
            
        expert_backbone_states.append(model.backbone.state_dict())

    # Create a base merged model to run evaluations on
    eval_backbone = get_base_backbone().to(device)
    eval_heads = {task: expert_models[task].head for task in tasks}
    
    # 2. Baseline Weight Averaging (WA)
    wa_backbone_state = merge_weight_averaging(expert_backbone_states)
    
    wa_results = {}
    print("\nEvaluating Weight Averaging (WA) baseline:")
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, eval_heads[task])
        eval_model.backbone.load_state_dict(wa_backbone_state)
        acc = evaluate_model(eval_model, loaders[task]['test'])
        wa_results[task] = acc
        print(f"WA -> {task.upper()}: {acc:.2f}%")
    wa_avg = np.mean(list(wa_results.values()))
    print(f"WA Average Accuracy: {wa_avg:.2f}%")

    # 3. Weight Averaging + Test-Time BatchNorm (WA + TTBN)
    wa_ttbn_results = {}
    print("\nEvaluating Weight Averaging + Test-Time BatchNorm (WA + TTBN):")
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, eval_heads[task])
        eval_model.backbone.load_state_dict(wa_backbone_state)
        acc = evaluate_model(eval_model, loaders[task]['test'], bn_mode='train')
        wa_ttbn_results[task] = acc
        print(f"WA + TTBN -> {task.upper()}: {acc:.2f}%")
    wa_ttbn_avg = np.mean(list(wa_ttbn_results.values()))
    print(f"WA + TTBN Average Accuracy: {wa_ttbn_avg:.2f}%")

    # 4. Sweep lambdas for Task Arithmetic (TA) and Task-Specific BatchNorm Merging (TSBM)
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    ta_sweep_results = []
    tsbm_sweep_results = []
    ta_ttbn_sweep_results = []
    
    print("\nRunning Task Arithmetic and TSBM sweeps across lambda...")
    for lmbda in lambdas:
        ta_state = merge_task_arithmetic(pretrained_backbone_state, expert_backbone_states, lmbda)
        
        # Eval TA
        ta_accs = {}
        for task in tasks:
            eval_model = ExpertModel(eval_backbone, eval_heads[task])
            eval_model.backbone.load_state_dict(ta_state)
            acc = evaluate_model(eval_model, loaders[task]['test'])
            ta_accs[task] = acc
        ta_avg = np.mean(list(ta_accs.values()))
        ta_sweep_results.append((lmbda, ta_accs, ta_avg))
        
        # Eval TA + TTBN
        ta_ttbn_accs = {}
        for task in tasks:
            eval_model = ExpertModel(eval_backbone, eval_heads[task])
            eval_model.backbone.load_state_dict(ta_state)
            acc = evaluate_model(eval_model, loaders[task]['test'], bn_mode='train')
            ta_ttbn_accs[task] = acc
        ta_ttbn_avg = np.mean(list(ta_ttbn_accs.values()))
        ta_ttbn_sweep_results.append((lmbda, ta_ttbn_accs, ta_ttbn_avg))
        
        # Eval TSBM (Our Proposed Method)
        tsbm_accs = {}
        for task in tasks:
            eval_model = ExpertModel(eval_backbone, eval_heads[task])
            apply_tsbm(eval_model, ta_state, expert_models[task].backbone.state_dict())
            acc = evaluate_model(eval_model, loaders[task]['test'])
            tsbm_accs[task] = acc
        tsbm_avg = np.mean(list(tsbm_accs.values()))
        tsbm_sweep_results.append((lmbda, tsbm_accs, tsbm_avg))
        
        print(f"Lambda {lmbda:.1f} | TA Avg: {ta_avg:.2f}% | TA+TTBN Avg: {ta_ttbn_avg:.2f}% | TSBM Avg: {tsbm_avg:.2f}%")

    # Find best lambdas
    best_ta = max(ta_sweep_results, key=lambda x: x[2])
    best_ta_ttbn = max(ta_ttbn_sweep_results, key=lambda x: x[2])
    best_tsbm = max(tsbm_sweep_results, key=lambda x: x[2])
    
    print(f"\nBest Task Arithmetic: Lambda={best_ta[0]:.1f}, Avg Acc={best_ta[2]:.2f}%")
    print(f"Best TA + TTBN: Lambda={best_ta_ttbn[0]:.1f}, Avg Acc={best_ta_ttbn[2]:.2f}%")
    print(f"Best TSBM: Lambda={best_tsbm[0]:.1f}, Avg Acc={best_tsbm[2]:.2f}%")

    # 5. SOTA Comparisons (TCAC and REPAIR)
    # We will use the best performing lambda from standard Task Arithmetic as the base for TCAC and REPAIR
    best_ta_lmbda = best_ta[0]
    best_ta_state = merge_task_arithmetic(pretrained_backbone_state, expert_backbone_states, best_ta_lmbda)
    
    # Let's collect original stats for each task
    print("\nCollecting original expert statistics for TCAC/REPAIR...")
    orig_stats = {}
    for task in tasks:
        # Create a calibration dataloader (subset of 128 samples from train)
        cal_dataset = Subset(loaders[task]['train_dataset'], list(range(128)))
        cal_loader = DataLoader(cal_dataset, batch_size=128, shuffle=False)
        orig_stats[task] = collect_calibration_stats(expert_models[task], cal_loader)
        
    # Let's collect merged stats for each task
    print("Collecting merged model statistics for TCAC/REPAIR...")
    merged_stats = {}
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, eval_heads[task])
        eval_model.backbone.load_state_dict(best_ta_state)
        cal_dataset = Subset(loaders[task]['train_dataset'], list(range(128)))
        cal_loader = DataLoader(cal_dataset, batch_size=128, shuffle=False)
        merged_stats[task] = collect_calibration_stats(eval_model, cal_loader)
        
    # Standard REPAIR: Average the original stats across tasks to form target stats
    repair_target_stats = {}
    # Get BN names from first task
    bn_names = list(orig_stats[tasks[0]].keys())
    for name in bn_names:
        means = [orig_stats[task][name][0] for task in tasks]
        stds = [orig_stats[task][name][1] for task in tasks]
        mean_target = torch.stack(means, dim=0).mean(dim=0)
        std_target = torch.stack(stds, dim=0).mean(dim=0)
        repair_target_stats[name] = (mean_target, std_target)

    # Evaluate Multi-Task REPAIR (Original simultaneous implementation)
    print("\nEvaluating Multi-Task REPAIR (Simultaneous):")
    repair_results_sim = {}
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, eval_heads[task])
        eval_model.backbone.load_state_dict(best_ta_state)
        
        # Register calibration hooks
        hooks = []
        for name, module in eval_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                m_mean, m_std = merged_stats[task][name]
                t_mean, t_std = repair_target_stats[name]
                cal_hook = CalibratorHook(t_mean, t_std, m_mean, m_std)
                hooks.append(module.register_forward_hook(cal_hook.hook))
                
        acc = evaluate_model(eval_model, loaders[task]['test'])
        repair_results_sim[task] = acc
        print(f"REPAIR (Simultaneous) -> {task.upper()}: {acc:.2f}%")
        
        # Remove hooks
        for h in hooks:
            h.remove()
            
    repair_avg_sim = np.mean(list(repair_results_sim.values()))
    print(f"REPAIR (Simultaneous) Average Accuracy: {repair_avg_sim:.2f}%")

    # Evaluate TCAC (Original simultaneous implementation)
    print("\nEvaluating Task-Conditional Activation Calibration (TCAC, Simultaneous):")
    tcac_results_sim = {}
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, eval_heads[task])
        eval_model.backbone.load_state_dict(best_ta_state)
        
        # Register calibration hooks
        hooks = []
        for name, module in eval_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                m_mean, m_std = merged_stats[task][name]
                o_mean, o_std = orig_stats[task][name]
                cal_hook = CalibratorHook(o_mean, o_std, m_mean, m_std)
                hooks.append(module.register_forward_hook(cal_hook.hook))
                
        acc = evaluate_model(eval_model, loaders[task]['test'])
        tcac_results_sim[task] = acc
        print(f"TCAC (Simultaneous) -> {task.upper()}: {acc:.2f}%")
        
        # Remove hooks
        for h in hooks:
            h.remove()
            
    tcac_avg_sim = np.mean(list(tcac_results_sim.values()))
    print(f"TCAC (Simultaneous) Average Accuracy: {tcac_avg_sim:.2f}%")

    # Evaluate Corrected Offline BN Recalibration (Corrected REPAIR/TCAC baseline)
    print("\nEvaluating Corrected Offline BN Recalibration (REPAIR - Corrected):")
    recal_results = {}
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, eval_heads[task])
        eval_model.backbone.load_state_dict(copy.deepcopy(best_ta_state))
        
        # Create calibration dataloader (exactly 128 samples, batch_size=128)
        cal_dataset = Subset(loaders[task]['train_dataset'], list(range(128)))
        cal_loader = DataLoader(cal_dataset, batch_size=128, shuffle=False)
        
        recalibrate_bn_exact(eval_model, cal_loader)
        
        acc = evaluate_model(eval_model, loaders[task]['test'], bn_mode='eval')
        recal_results[task] = acc
        print(f"REPAIR (Corrected) -> {task.upper()}: {acc:.2f}%")
    recal_avg = np.mean(list(recal_results.values()))
    print(f"REPAIR (Corrected) Average Accuracy: {recal_avg:.2f}%")

    # 6. Build the Final Results Table
    print("\n" + "="*50)
    print("FINAL MODEL MERGING RESULTS SUMMARY")
    print("="*50)
    print(f"{'Method':<20} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*50)
    print(f"{'Individual Experts':<20} | {expert_accs['mnist']:<8.2f} | {expert_accs['fashion']:<8.2f} | {expert_accs['cifar']:<8.2f} | {np.mean(list(expert_accs.values())):<8.2f}")
    print(f"{'Weight Averaging (WA)':<20} | {wa_results['mnist']:<8.2f} | {wa_results['fashion']:<8.2f} | {wa_results['cifar']:<8.2f} | {wa_avg:<8.2f}")
    print(f"{'WA + TTBN (Proposed)':<20} | {wa_ttbn_results['mnist']:<8.2f} | {wa_ttbn_results['fashion']:<8.2f} | {wa_ttbn_results['cifar']:<8.2f} | {wa_ttbn_avg:<8.2f}")
    print(f"{'Task Arithmetic (TA)':<20} | {best_ta[1]['mnist']:<8.2f} | {best_ta[1]['fashion']:<8.2f} | {best_ta[1]['cifar']:<8.2f} | {best_ta[2]:<8.2f}")
    print(f"{'TA + TTBN (Proposed)':<20} | {best_ta_ttbn[1]['mnist']:<8.2f} | {best_ta_ttbn[1]['fashion']:<8.2f} | {best_ta_ttbn[1]['cifar']:<8.2f} | {best_ta_ttbn[2]:<8.2f}")
    print(f"{'REPAIR (Multi-Task)':<20} | {recal_results['mnist']:<8.2f} | {recal_results['fashion']:<8.2f} | {recal_results['cifar']:<8.2f} | {recal_avg:<8.2f}")
    print(f"{'TCAC (SOTA Baseline)':<20} | {tcac_results_sim['mnist']:<8.2f} | {tcac_results_sim['fashion']:<8.2f} | {tcac_results_sim['cifar']:<8.2f} | {tcac_avg_sim:<8.2f}")
    print(f"{'TSBM (Ours, Proposed)':<20} | {best_tsbm[1]['mnist']:<8.2f} | {best_tsbm[1]['fashion']:<8.2f} | {best_tsbm[1]['cifar']:<8.2f} | {best_tsbm[2]:<8.2f}")
    print("="*50)

    # 7. Plotting the results
    plt.figure(figsize=(10, 6))
    lmbda_vals = lambdas
    ta_avgs = [r[2] for r in ta_sweep_results]
    tsbm_avgs = [r[2] for r in tsbm_sweep_results]
    ta_ttbn_avgs = [r[2] for r in ta_ttbn_sweep_results]
    
    plt.plot(lmbda_vals, ta_avgs, marker='o', linestyle='-', label='Task Arithmetic (TA)')
    plt.plot(lmbda_vals, ta_ttbn_avgs, marker='s', linestyle='--', label='TA + Test-Time BN (TTBN)')
    plt.plot(lmbda_vals, tsbm_avgs, marker='^', linestyle='-', linewidth=2, label='Task-Specific BN Merging (TSBM)')
    
    plt.axhline(y=tcac_avg_sim, color='r', linestyle='-.', label='TCAC (Simultaneous, Broken)')
    plt.axhline(y=recal_avg, color='g', linestyle='-.', label='REPAIR/BN-Recalibration (Corrected)')
    plt.axhline(y=wa_avg, color='gray', linestyle=':', label='Weight Averaging')
    
    plt.xlabel('Merging Coefficient (Lambda)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Multi-Task Model Merging Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('merging_performance_sweep.png', dpi=300)
    print("Performance comparison plot saved to merging_performance_sweep.png")

    # 8. Ablation study on Test Batch Size for TTBN
    print("\n" + "="*50)
    print("ABLATION STUDY: EFFECT OF TEST BATCH SIZE ON TTBN")
    print("="*50)
    print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*50)
    
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    best_ta_lmbda = best_ta_ttbn[0]
    best_ta_state = merge_task_arithmetic(pretrained_backbone_state, expert_backbone_states, best_ta_lmbda)
    
    for bs in batch_sizes:
        # Create dataloaders for this batch size
        bs_loaders = get_dataloaders(batch_size=bs)
        bs_results = {}
        for task in tasks:
            eval_model = ExpertModel(eval_backbone, eval_heads[task])
            eval_model.backbone.load_state_dict(best_ta_state)
            acc = evaluate_model(eval_model, bs_loaders[task]['test'], bn_mode='train')
            bs_results[task] = acc
        bs_avg = np.mean(list(bs_results.values()))
        print(f"{bs:<12} | {bs_results['mnist']:<8.2f} | {bs_results['fashion']:<8.2f} | {bs_results['cifar']:<8.2f} | {bs_avg:<8.2f}")
    print("="*50)

if __name__ == '__main__':
    main()
