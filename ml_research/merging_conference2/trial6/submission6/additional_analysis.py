import os
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set complete determinism
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

set_seed(42)

# Helper Dataset to replicate grayscale to 3 channels
class ReplicateChannelsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x, y

def get_dataloaders(batch_size=64, num_train_samples=5000, num_cal_samples=256):
    # Transforms
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download full datasets
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)

    # Deterministic subset selection
    def get_subsets(full_dataset):
        indices = list(range(len(full_dataset)))
        random.seed(42)
        random.shuffle(indices)
        train_indices = indices[:num_train_samples]
        cal_indices = indices[num_train_samples:num_train_samples + num_cal_samples]
        return Subset(full_dataset, train_indices), Subset(full_dataset, cal_indices)

    mnist_train, mnist_cal = get_subsets(mnist_train_full)
    fmnist_train, fmnist_cal = get_subsets(fmnist_train_full)
    cifar_train, cifar_cal = get_subsets(cifar_train_full)

    # Data loaders
    loaders = {
        'mnist': {
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
            'cal': DataLoader(mnist_cal, batch_size=num_cal_samples, shuffle=False),
            'test': DataLoader(mnist_test, batch_size=256, shuffle=False)
        },
        'fmnist': {
            'train': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True),
            'cal': DataLoader(fmnist_cal, batch_size=num_cal_samples, shuffle=False),
            'test': DataLoader(fmnist_test, batch_size=256, shuffle=False)
        },
        'cifar': {
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True),
            'cal': DataLoader(cifar_cal, batch_size=num_cal_samples, shuffle=False),
            'test': DataLoader(cifar_test, batch_size=256, shuffle=False)
        }
    }
    return loaders

def create_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace classification head with Dropout + Linear
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
    return model

def train_expert(model, init_model, train_loader, device, epochs=5, lr=1e-4, weight_decay=1e-4, l2_sp_lambda=0.0):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Store initial weights for L2-SP
    if l2_sp_lambda > 0.0:
        init_params = {name: param.clone().to(device) for name, param in init_model.named_parameters() if param.requires_grad}

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Add L2-SP penalty if applicable
            if l2_sp_lambda > 0.0:
                l2_sp_penalty = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and 'weight' in name and name in init_params:
                        l2_sp_penalty += torch.sum((param - init_params[name]) ** 2)
                loss += l2_sp_lambda * l2_sp_penalty
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return model

def evaluate(model, loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

def get_conv_bn_pairs(model):
    submodules_ordered = []
    submodules_ordered.append(('conv1', model.conv1, 'bn1', model.bn1))
    
    def process_block(block_name, block):
        submodules_ordered.append((f'{block_name}.conv1', block.conv1, f'{block_name}.bn1', block.bn1))
        if block.downsample is not None:
            submodules_ordered.append((f'{block_name}.downsample.0', block.downsample[0], f'{block_name}.downsample.1', block.downsample[1]))
        submodules_ordered.append((f'{block_name}.conv2', block.conv2, f'{block_name}.bn2', block.bn2))

    for idx, block in enumerate(model.layer1):
        process_block(f'layer1.{idx}', block)
    for idx, block in enumerate(model.layer2):
        process_block(f'layer2.{idx}', block)
    for idx, block in enumerate(model.layer3):
        process_block(f'layer3.{idx}', block)
    for idx, block in enumerate(model.layer4):
        process_block(f'layer4.{idx}', block)
        
    return submodules_ordered

def merge_models_wa(experts, init_model):
    merged = create_model()
    merged_state = merged.state_dict()
    expert_states = {t: expert.state_dict() for t, expert in experts.items()}
    
    for name in merged_state.keys():
        if 'fc' in name:
            continue
        if torch.is_floating_point(merged_state[name]) or torch.is_complex(merged_state[name]):
            merged_state[name] = torch.stack([expert_states[t][name] for t in experts.keys()]).mean(dim=0)
        else:
            first_task = list(experts.keys())[0]
            merged_state[name] = expert_states[first_task][name].clone()
        
    merged.load_state_dict(merged_state)
    return merged

class ActivationHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = input[0].detach()

    def remove(self):
        self.hook.remove()

class OutputActivationHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output.detach()

    def remove(self):
        self.hook.remove()

def calibrate_model(merged, experts, cal_loaders, device, method='none', r=4, reg=0.5, N=128):
    if method == 'none':
        return merged

    set_seed(42)
    merged_cal = copy.deepcopy(merged).to(device).eval()
    experts_cal = {t: copy.deepcopy(expert).to(device).eval() for t, expert in experts.items()}
    
    merged_pairs = get_conv_bn_pairs(merged_cal)
    expert_pairs = {t: get_conv_bn_pairs(experts_cal[t]) for t in experts.keys()}
    num_layers = len(merged_pairs)

    cal_batches = {}
    for t in experts.keys():
        for x, _ in cal_loaders[t]['cal']:
            cal_batches[t] = x[:N].to(device)
            break
            
    for l_idx in range(num_layers):
        conv_name, conv_m, bn_name, bn_m = merged_pairs[l_idx]
        is_deep = any(k in conv_name for k in ['layer3', 'layer4'])
        is_deep_calibration = is_deep and method == 'hybrid'
        
        if is_deep_calibration:
            hook_X = ActivationHook(conv_m)
            hooks_H = {t: OutputActivationHook(expert_pairs[t][l_idx][3]) for t in experts.keys()}
            X_tasks = {}
            H_target_tasks = {}
            with torch.no_grad():
                for t in experts.keys():
                    merged_cal.fc = experts_cal[t].fc
                    _ = merged_cal(cal_batches[t])
                    X_tasks[t] = hook_X.activation.clone()
                    _ = experts_cal[t](cal_batches[t])
                    H_target_tasks[t] = hooks_H[t].activation.clone()
            hook_X.remove()
            for t in experts.keys():
                hooks_H[t].remove()
        else:
            hook_V_merged = OutputActivationHook(conv_m)
            hooks_V_expert = {t: OutputActivationHook(expert_pairs[t][l_idx][1]) for t in experts.keys()}
            V_merged_tasks = []
            V_expert_tasks = []
            with torch.no_grad():
                for t in experts.keys():
                    merged_cal.fc = experts_cal[t].fc
                    _ = merged_cal(cal_batches[t])
                    V_merged_tasks.append(hook_V_merged.activation.clone())
                    _ = experts_cal[t](cal_batches[t])
                    V_expert_tasks.append(hooks_V_expert[t].activation.clone())
            hook_V_merged.remove()
            for t in experts.keys():
                hooks_V_expert[t].remove()
            
        if not is_deep_calibration:
            V_all = torch.cat(V_merged_tasks, dim=0)
            V_expert_all = torch.cat(V_expert_tasks, dim=0)
            C = V_all.shape[1]
            V_flat = V_all.permute(0, 2, 3, 1).reshape(-1, C)
            V_expert_flat = V_expert_all.permute(0, 2, 3, 1).reshape(-1, C)
            sigma_merged = V_flat.std(dim=0)
            sigma_target = V_expert_flat.std(dim=0)
            gamma = sigma_target / (sigma_merged + 1e-5)
            gamma = torch.clamp(gamma, min=0.1, max=10.0)
            bn_m.weight.data.copy_(bn_m.weight.data * gamma)
            bn_m.bias.data.copy_(bn_m.bias.data * gamma)
        else:
            C_out, C_in, Kh, Kw = conv_m.weight.shape
            V_target_tasks = []
            for t in experts.keys():
                H_t = H_target_tasks[t]
                exp_bn = expert_pairs[t][l_idx][3]
                w = exp_bn.weight.view(1, C_out, 1, 1)
                b = exp_bn.bias.view(1, C_out, 1, 1)
                mu = exp_bn.running_mean.view(1, C_out, 1, 1)
                var = exp_bn.running_var.view(1, C_out, 1, 1)
                eps = exp_bn.eps
                V_t = (H_t - b) / (w + 1e-5) * torch.sqrt(var + eps) + mu
                V_target_tasks.append(V_t)
                
            X_unfolded_tasks = []
            for t in experts.keys():
                X_t = X_tasks[t]
                X_unf = F.unfold(X_t, kernel_size=(Kh, Kw), dilation=conv_m.dilation, padding=conv_m.padding, stride=conv_m.stride)
                X_unfolded_tasks.append(X_unf)
                
            X_unfolded = torch.cat(X_unfolded_tasks, dim=0)
            d_in = X_unfolded.shape[1]
            X_matrix = X_unfolded.transpose(0, 1).reshape(d_in, -1)
            
            V_target_all = torch.cat(V_target_tasks, dim=0)
            V_target = V_target_all.transpose(0, 1).reshape(C_out, -1)
            
            M = X_matrix.shape[1]
            lmbda = reg * M
            
            W_curr = conv_m.weight.view(C_out, -1)
            E = V_target - torch.matmul(W_curr, X_matrix)
            
            cov = torch.matmul(X_matrix, X_matrix.T)
            cov.add_(torch.eye(d_in, device=device) * lmbda)
            inv_cov = torch.linalg.inv(cov)
            
            dW_star = torch.matmul(torch.matmul(E, X_matrix.T), inv_cov)
            U, S, Vh = torch.linalg.svd(dW_star, full_matrices=False)
            dW_r = torch.matmul(U[:, :r], torch.matmul(torch.diag(S[:r]), Vh[:r, :]))
            
            conv_m.weight.data.copy_(conv_m.weight.data + dW_r.view(conv_m.weight.shape))
            
            W_new = conv_m.weight.view(C_out, -1)
            V_new = torch.matmul(W_new, X_matrix)
            
            mu_run = V_new.mean(dim=1)
            var_run = V_new.var(dim=1, unbiased=False)
            bn_m.running_mean.copy_(mu_run)
            bn_m.running_var.copy_(var_run)
            
            V_norm = (V_new - mu_run.unsqueeze(1)) / torch.sqrt(var_run.unsqueeze(1) + bn_m.eps)
            H_target_all = torch.cat([H_target_tasks[t] for t in experts.keys()], dim=0)
            H_target = H_target_all.transpose(0, 1).reshape(C_out, -1)
            
            w_c = (V_norm * H_target).mean(dim=1)
            b_c = H_target.mean(dim=1)
            w_c = torch.clamp(w_c, min=0.1, max=10.0)
            
            bn_m.weight.data.copy_(w_c)
            bn_m.bias.data.copy_(b_c)
            
    merged.load_state_dict(merged_cal.state_dict())
    return merged

def run_diagnostic_and_ablation(device):
    loaders = get_dataloaders()
    set_seed(42)
    
    # We will use Scenario A (Low Reg) and Scenario C (High Reg) to analyze
    # how rank r and reg interact with calibration
    scenarios_config = {
        'A_low_reg': {'weight_decay': 0.0, 'l2_sp_lambda': 0.0},
        'C_high_reg': {'weight_decay': 1e-2, 'l2_sp_lambda': 0.0}
    }
    
    # 1. Train experts for A and C (if we run from scratch, it will take minutes)
    scenarios_experts = {}
    init_model = create_model().to(device)
    
    for sc_name, config in scenarios_config.items():
        print(f"\nTraining experts for diagnostic: {sc_name}...")
        experts = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            set_seed(42)
            model = create_model()
            model = train_expert(
                model=model,
                init_model=init_model,
                train_loader=loaders[task]['train'],
                device=device,
                epochs=5,
                lr=1e-4,
                weight_decay=config['weight_decay'],
                l2_sp_lambda=config['l2_sp_lambda']
            )
            experts[task] = model
        scenarios_experts[sc_name] = experts

    # 2. Hyperparameter Grid Search on Scenario A and C
    ranks = [1, 2, 4, 8]
    regs = [0.1, 0.5, 1.0]
    
    grid_results = {}
    for sc_name in scenarios_experts.keys():
        experts = scenarios_experts[sc_name]
        merged_base = merge_models_wa(experts, init_model)
        
        sc_grid = []
        for r in ranks:
            for reg in regs:
                print(f"Grid search {sc_name} | rank={r}, reg={reg}")
                merged_cal = copy.deepcopy(merged_base)
                merged_cal = calibrate_model(
                    merged=merged_cal,
                    experts=experts,
                    cal_loaders=loaders,
                    device=device,
                    method='hybrid',
                    r=r,
                    reg=reg
                )
                
                # Evaluate
                accs = []
                for task in ['mnist', 'fmnist', 'cifar']:
                    merged_cal.fc = experts[task].fc
                    accs.append(evaluate(merged_cal, loaders[task]['test'], device))
                avg_acc = sum(accs) / len(accs)
                sc_grid.append({'rank': r, 'reg': reg, 'accuracy': avg_acc})
                print(f"  Accuracy: {avg_acc:.2f}%")
        grid_results[sc_name] = sc_grid
        
    with open('grid_search_results.json', 'w') as f:
        json.dump(grid_results, f, indent=4)
        
    # 3. Analyze Activation Variance and Spectrum Decay at layer4.1.bn2
    # We will collect activations of the deepest block's final BN output: 'layer4.1.bn2'
    # We will compare:
    # (a) Individual task experts on their own data (Oracle / target)
    # (b) Uncalibrated merged model (WA)
    # (c) Corrected Hybrid Calibrated model
    # We use Scenario A experts for this.
    experts = scenarios_experts['A_low_reg']
    merged_base = merge_models_wa(experts, init_model)
    merged_cal = copy.deepcopy(merged_base)
    merged_cal = calibrate_model(
        merged=merged_cal,
        experts=experts,
        cal_loaders=loaders,
        device=device,
        method='hybrid',
        r=4,
        reg=0.5
    )
    
    # We use a combined batch of size 128 from each loader to collect activations
    cal_batches = {}
    for t in experts.keys():
        for x, _ in loaders[t]['cal']:
            cal_batches[t] = x.to(device)
            break
            
    # Hooks to capture 'layer4.1.bn2' output
    # (In ResNet-18, the last basic block is model.layer4[1], and its BN layer is bn2)
    # Let's extract activations
    def collect_activations(model, x, head):
        model = model.to(device)
        model.eval()
        model.fc = head.to(device)
        hook = OutputActivationHook(model.layer4[1].bn2)
        with torch.no_grad():
            _ = model(x)
        act = hook.activation.clone()
        hook.remove()
        return act
        
    # Capture Oracle activations
    oracle_acts = []
    for t in experts.keys():
        act = collect_activations(experts[t], cal_batches[t], experts[t].fc)
        oracle_acts.append(act)
    oracle_act_all = torch.cat(oracle_acts, dim=0) # (3*N, C, H, W)
    
    # Capture Uncalibrated acts
    uncal_acts = []
    for t in experts.keys():
        act = collect_activations(merged_base, cal_batches[t], experts[t].fc)
        uncal_acts.append(act)
    uncal_act_all = torch.cat(uncal_acts, dim=0)
    
    # Capture Calibrated acts
    cal_acts = []
    for t in experts.keys():
        act = collect_activations(merged_cal, cal_batches[t], experts[t].fc)
        cal_acts.append(act)
    cal_act_all = torch.cat(cal_acts, dim=0)
    
    # Compute standard deviations across channels
    # Flatten over batch and spatial dimensions: (B*H*W, C)
    C = oracle_act_all.shape[1]
    oracle_flat = oracle_act_all.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
    uncal_flat = uncal_act_all.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
    cal_flat = cal_act_all.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
    
    std_oracle = np.std(oracle_flat, axis=0)
    std_uncal = np.std(uncal_flat, axis=0)
    std_cal = np.std(cal_flat, axis=0)
    
    # Plot 1: Standard Deviation across Channels (Variance Collapse)
    plt.figure(figsize=(7, 4.5))
    plt.style.use('seaborn-v0_8-whitegrid')
    # Sort for visual clarity
    sort_idx = np.argsort(std_oracle)[::-1]
    plt.plot(std_oracle[sort_idx], label='Oracle Target (Experts Avg)', color='#C44E52', linestyle='--')
    plt.plot(std_uncal[sort_idx], label='Uncalibrated Merging', color='#8C8C8C')
    plt.plot(std_cal[sort_idx], label='Hybrid SLR-WBC (Ours)', color='#4C72B0')
    plt.ylabel('Standard Deviation of Activations')
    plt.xlabel('Channels (Sorted by std)')
    plt.title('Representation Variance Collapse Recovery at layer4.1.bn2')
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('variance_collapse_recovery.png', dpi=300)
    plt.close()
    
    # Plot 2: Feature Spectrum Decay (SVD)
    # We flatten the activation maps to (B, C*H*W) and perform SVD to see dimensional collapse
    def compute_spectrum(act_tensor):
        B = act_tensor.shape[0]
        flat = act_tensor.reshape(B, -1).cpu().numpy()
        # SVD
        _, S, _ = np.linalg.svd(flat, full_matrices=False)
        # Normalize by top singular value
        S = S / S[0]
        return S
        
    spec_oracle = compute_spectrum(oracle_act_all)
    spec_uncal = compute_spectrum(uncal_act_all)
    spec_cal = compute_spectrum(cal_act_all)
    
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(spec_oracle, label='Oracle Target', color='#C44E52', linestyle='--')
    plt.semilogy(spec_uncal, label='Uncalibrated Merging', color='#8C8C8C')
    plt.semilogy(spec_cal, label='Hybrid SLR-WBC (Ours)', color='#4C72B0')
    plt.ylabel('Normalized Singular Values (S/S0)')
    plt.xlabel('Singular Value Rank')
    plt.title('Feature Spectrum Decay (Dimensionality)')
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('spectrum_decay_recovery.png', dpi=300)
    plt.close()
    
    print("Diagnostics completed! Plots saved as variance_collapse_recovery.png and spectrum_decay_recovery.png!")
    
    # Run the calibration budget analysis sweep
    run_calibration_budget_analysis(device, scenarios_experts, loaders)

def run_calibration_budget_analysis(device, scenarios_experts, loaders):
    print("\nStarting Calibration Budget and Regularization Sweep...")
    
    Ns = [16, 32, 64, 128, 256]
    regs = [0.01, 0.1, 0.5, 1.0, 5.0]
    
    sweep_results = {}
    init_model = create_model().to(device)
    
    for sc_name in ['A_low_reg', 'C_high_reg']:
        experts = scenarios_experts[sc_name]
        merged_base = merge_models_wa(experts, init_model)
        
        sc_results = []
        for reg in regs:
            reg_accs = []
            for N in Ns:
                print(f"Sweep {sc_name} | reg={reg}, N={N}")
                merged_cal = copy.deepcopy(merged_base)
                merged_cal = calibrate_model(
                    merged=merged_cal,
                    experts=experts,
                    cal_loaders=loaders,
                    device=device,
                    method='hybrid',
                    r=4,
                    reg=reg,
                    N=N
                )
                
                # Evaluate
                accs = []
                for task in ['mnist', 'fmnist', 'cifar']:
                    merged_cal.fc = experts[task].fc
                    accs.append(evaluate(merged_cal, loaders[task]['test'], device))
                avg_acc = sum(accs) / len(accs)
                reg_accs.append(avg_acc)
                print(f"  Accuracy: {avg_acc:.2f}%")
                
            sc_results.append({
                'reg': reg,
                'N_values': Ns,
                'accuracies': reg_accs
            })
        sweep_results[sc_name] = sc_results
        
    # Save raw results
    with open('calibration_budget_analysis.json', 'w') as f:
        json.dump(sweep_results, f, indent=4)
        
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    sc_titles = {
        'A_low_reg': 'Scenario A (Low Reg / WD=0)',
        'C_high_reg': 'Scenario C (High Reg / WD=1e-2)'
    }
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
    
    for ax_idx, sc_name in enumerate(['A_low_reg', 'C_high_reg']):
        ax = axes[ax_idx]
        sc_data = sweep_results[sc_name]
        
        for r_idx, item in enumerate(sc_data):
            reg = item['reg']
            accuracies = item['accuracies']
            ax.plot(Ns, accuracies, marker='o', color=colors[r_idx], label=f'reg={reg}')
            
        ax.set_title(sc_titles[sc_name], fontsize=13, fontweight='bold')
        ax.set_xlabel('Calibration Sample Size (N)', fontsize=12)
        ax.set_xscale('log')
        ax.set_xticks(Ns)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if ax_idx == 0:
            ax.set_ylabel('Average Merged Test Accuracy (%)', fontsize=12)
        ax.legend(frameon=True, facecolor='white', loc='lower right')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
    plt.suptitle('Generalization Performance across Calibration Sizes (N) and SVD Regularization Strengths', fontsize=14, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig('calibration_budget_analysis.png', dpi=300)
    plt.close()
    
    print("Sweep completed! Plots saved as calibration_budget_analysis.png!")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running diagnostic on device:", device)
    run_diagnostic_and_ablation(device)
