import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ReplicateChannel(object):
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ReplicateChannel(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=mnist_transform)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=mnist_transform)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# Dataloaders for evaluation
test_loaders = {
    'mnist': DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=4),
    'fashion': DataLoader(test_fashion, batch_size=256, shuffle=False, num_workers=4),
    'cifar': DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=4)
}

# Create calibration datasets
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=mnist_transform)
train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=mnist_transform)
train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

def get_calibration_loader(N, seed=42):
    # N total, N // 3 per dataset
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_per_task = N // 3
    
    # Randomly sample indices
    mnist_idx = np.random.choice(len(train_mnist), n_per_task, replace=False)
    fashion_idx = np.random.choice(len(train_fashion), n_per_task, replace=False)
    cifar_idx = np.random.choice(len(train_cifar), n_per_task, replace=False)
    
    mnist_sub = Subset(train_mnist, mnist_idx)
    fashion_sub = Subset(train_fashion, fashion_idx)
    cifar_sub = Subset(train_cifar, cifar_idx)
    
    # Return subsets and combined loaders
    joint_loader = DataLoader(
        torch.utils.data.ConcatDataset([mnist_sub, fashion_sub, cifar_sub]),
        batch_size=N, shuffle=False
    )
    
    task_loaders = {
        'mnist': DataLoader(mnist_sub, batch_size=n_per_task, shuffle=False),
        'fashion': DataLoader(fashion_sub, batch_size=n_per_task, shuffle=False),
        'cifar': DataLoader(cifar_sub, batch_size=n_per_task, shuffle=False)
    }
    
    return joint_loader, task_loaders

# Helper function to load model with structure
def load_resnet18_with_fc(state_dict_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    return model.to(device)

def get_weight_averaged_backbone():
    print("\nCreating Weight Averaged (WA) Merged Model...")
    mnist_sd = torch.load('mnist_expert.pth', map_location=device)
    fashion_sd = torch.load('fashion_expert.pth', map_location=device)
    cifar_sd = torch.load('cifar_expert.pth', map_location=device)
    
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10) # Dummy fc
    merged_model = merged_model.to(device)
    
    merged_sd = merged_model.state_dict()
    for key in merged_sd.keys():
        if 'fc' in key:
            continue
        # Average the weights
        merged_sd[key] = (mnist_sd[key] + fashion_sd[key] + cifar_sd[key]) / 3.0
        
    merged_model.load_state_dict(merged_sd)
    return merged_model

def get_task_arithmetic_backbone(lam=0.5):
    print(f"\nCreating Task Arithmetic (TA) Merged Model (lambda={lam})...")
    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrained_sd = pretrained_model.state_dict()
    
    mnist_sd = torch.load('mnist_expert.pth', map_location=device)
    fashion_sd = torch.load('fashion_expert.pth', map_location=device)
    cifar_sd = torch.load('cifar_expert.pth', map_location=device)
    
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10) # Dummy fc
    merged_model = merged_model.to(device)
    
    merged_sd = merged_model.state_dict()
    for key in merged_sd.keys():
        if 'fc' in key:
            continue
        # Compute task vectors
        tau_mnist = mnist_sd[key] - pretrained_sd[key].to(device)
        tau_fashion = fashion_sd[key] - pretrained_sd[key].to(device)
        tau_cifar = cifar_sd[key] - pretrained_sd[key].to(device)
        # Combine
        merged_sd[key] = pretrained_sd[key].to(device) + lam * (tau_mnist + tau_fashion + tau_cifar)
        
    merged_model.load_state_dict(merged_sd)
    return merged_model

# Evaluation
def evaluate_multi_task(backbone, experts_fc_paths, test_loaders):
    backbone.eval()
    accuracies = {}
    
    for task_name, test_loader in test_loaders.items():
        # Load expert fc
        expert_sd = torch.load(experts_fc_paths[task_name], map_location=device)
        # Temporarily replace backbone fc with expert's fc
        old_fc = copy.deepcopy(backbone.fc)
        backbone.fc.weight.data.copy_(expert_sd['fc.weight'])
        backbone.fc.bias.data.copy_(expert_sd['fc.bias'])
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = backbone(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracies[task_name] = correct / total
        # Restore old fc
        backbone.fc = old_fc
        
    accuracies['avg'] = sum(accuracies.values()) / len(accuracies)
    return accuracies

# Compute SP-TAAC Scaling Factors
def compute_sp_taac_scales(merged_model, experts_paths, joint_loader, task_loaders):
    merged_model.eval()
    
    # Load expert models
    experts = {}
    for name, path in experts_paths.items():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        experts[name] = model.to(device)
        
    # Find all BatchNorm2d layers
    bn_modules_merged = []
    for m in merged_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_modules_merged.append(m)
            
    bn_modules_experts = {name: [] for name in experts.keys()}
    for name, model in experts.items():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules_experts[name].append(m)
                
    num_bn = len(bn_modules_merged)
    print(f"Found {num_bn} BatchNorm2d layers.")
    
    # Register hooks on merged model to collect activations over joint_loader
    merged_acts = [[] for _ in range(num_bn)]
    def get_collect_hook(storage):
        def hook(module, input, output):
            storage.append(output.detach().cpu())
        return hook
        
    hooks = []
    for idx, m in enumerate(bn_modules_merged):
        hooks.append(m.register_forward_hook(get_collect_hook(merged_acts[idx])))
        
    with torch.no_grad():
        for inputs, _ in joint_loader:
            inputs = inputs.to(device)
            _ = merged_model(inputs)
            
    for h in hooks:
        h.remove()
        
    # Now run each expert on its own task calibration loader
    expert_acts = {name: [[] for _ in range(num_bn)] for name in experts.keys()}
    for name, model in experts.items():
        expert_hooks = []
        for idx, m in enumerate(bn_modules_experts[name]):
            expert_hooks.append(m.register_forward_hook(get_collect_hook(expert_acts[name][idx])))
            
        with torch.no_grad():
            for inputs, _ in task_loaders[name]:
                inputs = inputs.to(device)
                _ = model(inputs)
                
        for h in expert_hooks:
            h.remove()
            
    # Compute global layer-wise scaling factors
    gammas = []
    for idx in range(num_bn):
        # Concat along batch dimension
        m_act = torch.cat(merged_acts[idx], dim=0) # (N, C, H, W)
        sigma_merged = torch.std(m_act, dim=(0, 1, 2, 3)).item() + 1e-8
        
        sigma_experts = []
        for name in experts.keys():
            e_act = torch.cat(expert_acts[name][idx], dim=0)
            sigma_expert = torch.std(e_act, dim=(0, 1, 2, 3)).item() + 1e-8
            sigma_experts.append(sigma_expert)
            
        sigma_target = np.mean(sigma_experts)
        gamma = sigma_target / sigma_merged
        gammas.append(gamma)
        
    return gammas

# Apply scaling via forward hooks (SP-TAAC baseline)
def apply_sp_taac_hooks(model, gammas):
    bn_modules = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_modules.append(m)
            
    hooks = []
    def make_scaling_hook(gamma):
        def hook(module, input, output):
            return output * gamma
        return hook
        
    for idx, m in enumerate(bn_modules):
        h = m.register_forward_hook(make_scaling_hook(gammas[idx]))
        hooks.append(h)
    return hooks

# Apply proposed Weight-Folded Calibration (WFC)
def apply_wfc_folding(model, gammas):
    folded_model = copy.deepcopy(model)
    bn_modules = []
    for m in folded_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_modules.append(m)
            
    # Directly modify the weight and bias parameters of the BatchNorm layers
    with torch.no_grad():
        for idx, m in enumerate(bn_modules):
            m.weight.mul_(gammas[idx])
            m.bias.mul_(gammas[idx])
            # Also scale the running mean and variance?
            # Running stats are used for normalization prior to affine transform.
            # If we scale running_mean and running_var, it changes the normalization.
            # Since the input is not scaled, we should ONLY scale weight, bias and running_mean/var?
            # Wait, the output is: y = (x - mean) / sqrt(var + eps) * weight + bias.
            # If we scale weight and bias by gamma, the output becomes:
            # y_scaled = (x - mean) / sqrt(var + eps) * (gamma * weight) + (gamma * bias)
            #          = gamma * y
            # This is EXACTLY mathematically equivalent to multiplying the output of the layer by gamma!
            # Therefore, we ONLY need to scale the weight and bias!
            pass
            
    return folded_model

# Proposed Least-Squares Head Alignment (LSHA)
# Proposed Prior-Regularized Least-Squares Head Alignment (PR-LSHA)
def apply_lsha(backbone, experts_paths, task_loaders, reg=1.0):
    backbone.eval()
    
    # Dictionary to hold the aligned head weights
    aligned_heads = {}
    
    for task_name, loader in task_loaders.items():
        # Load expert model to get target logits and original head parameters (prior)
        expert = models.resnet18(weights=None)
        expert.fc = nn.Linear(expert.fc.in_features, 10)
        expert.load_state_dict(torch.load(experts_paths[task_name], map_location=device))
        expert = expert.to(device)
        expert.eval()
        
        W0_weight = expert.fc.weight.data # (10, 512)
        W0_bias = expert.fc.bias.data # (10,)
        # Combine into W0_bias_param of shape (10, 513) where last column is bias
        W0 = torch.cat([W0_weight, W0_bias.unsqueeze(1)], dim=1) # (10, 513)
        
        # Collect backbone features and expert logits
        target_logits_list = []
        
        # We need a hook to collect features right before the fc layer
        features = []
        def feat_hook(module, input, output):
            # output is (B, 512, 1, 1) after avgpool
            features.append(output.flatten(1).detach())
            
        hook = backbone.avgpool.register_forward_hook(feat_hook)
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _ = backbone(inputs)
                logits = expert(inputs)
                target_logits_list.append(logits.detach())
                
        hook.remove()
        
        F = torch.cat(features, dim=0) # (N, 512)
        Y = torch.cat(target_logits_list, dim=0) # (N, 10)
        
        # Append ones for bias: F_bias is (N, 513)
        F_bias = torch.cat([F, torch.ones(F.size(0), 1, device=device)], dim=1)
        
        # Compute residual Y_tilde = Y - F_bias @ W0.t()
        Y_tilde = Y - F_bias @ W0.t()
        
        # Solve for correction V^T = (F_bias^T @ F_bias + reg * I)^-1 F_bias^T @ Y_tilde
        I = torch.eye(F_bias.size(1), device=device)
        lhs = F_bias.t() @ F_bias + reg * I
        rhs = F_bias.t() @ Y_tilde
        V_bias = torch.linalg.solve(lhs, rhs) # (513, 10)
        
        # Final head: W^T = W0^T + V^T
        W_bias = W0.t() + V_bias # (513, 10)
        
        W_new = W_bias[:-1, :].t() # (10, 512)
        b_new = W_bias[-1, :] # (10,)
        
        aligned_heads[task_name] = {
            'weight': W_new,
            'bias': b_new
        }
        
    return aligned_heads

# Evaluate with aligned heads
def evaluate_multi_task_aligned_heads(backbone, aligned_heads, test_loaders):
    backbone.eval()
    accuracies = {}
    
    for task_name, test_loader in test_loaders.items():
        head_params = aligned_heads[task_name]
        
        old_fc = copy.deepcopy(backbone.fc)
        backbone.fc.weight.data.copy_(head_params['weight'])
        backbone.fc.bias.data.copy_(head_params['bias'])
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = backbone(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracies[task_name] = correct / total
        backbone.fc = old_fc
        
    accuracies['avg'] = sum(accuracies.values()) / len(accuracies)
    return accuracies


if __name__ == '__main__':
    experts_paths = {
        'mnist': 'mnist_expert.pth',
        'fashion': 'fashion_expert.pth',
        'cifar': 'cifar_expert.pth'
    }
    
    # 1. Base Experts evaluation
    print("Evaluating individual expert models on their own tasks...")
    for name, path in experts_paths.items():
        if os.path.exists(path):
            m = load_resnet18_with_fc(path, 10)
            acc = evaluate_multi_task(m, {name: path}, {name: test_loaders[name]})
            print(f"{name.upper()} Expert Accuracy: {acc[name]*100:.2f}%")
        else:
            print(f"Expert model path {path} not found!")
            
    # 2. Weight Averaging Baseline
    wa_backbone = get_weight_averaged_backbone()
    wa_accs = evaluate_multi_task(wa_backbone, experts_paths, test_loaders)
    print("\n--- Weight Averaging Baseline ---")
    print(f"MNIST: {wa_accs['mnist']*100:.2f}% | Fashion: {wa_accs['fashion']*100:.2f}% | CIFAR: {wa_accs['cifar']*100:.2f}%")
    print(f"Average: {wa_accs['avg']*100:.2f}%")
    
    # 3. Task Arithmetic Baseline
    ta_backbone = get_task_arithmetic_backbone(lam=0.5)
    ta_accs = evaluate_multi_task(ta_backbone, experts_paths, test_loaders)
    print("\n--- Task Arithmetic Baseline (lambda=0.5) ---")
    print(f"MNIST: {ta_accs['mnist']*100:.2f}% | Fashion: {ta_accs['fashion']*100:.2f}% | CIFAR: {ta_accs['cifar']*100:.2f}%")
    print(f"Average: {ta_accs['avg']*100:.2f}%")
    
    # 4. Calibration & Proposal Evaluation
    print("\n========================== CALIBRATION EXPERIMENTS ==========================")
    for N in [16, 64, 256]:
        print(f"\n--- Running Calibration with N={N} ---")
        joint_loader, task_loaders_cal = get_calibration_loader(N, seed=42)
        
        # Compute scaling factors gammas
        gammas = compute_sp_taac_scales(wa_backbone, experts_paths, joint_loader, task_loaders_cal)
        print("Scaling factors computed:", [f"{g:.4f}" for g in gammas[:5]], "... total layers:", len(gammas))
        
        # Test hook-based SP-TAAC
        hooks = apply_sp_taac_hooks(wa_backbone, gammas)
        sptaac_accs = evaluate_multi_task(wa_backbone, experts_paths, test_loaders)
        # Remove hooks
        for h in hooks:
            h.remove()
            
        print(f"SP-TAAC (Hooks) - MNIST: {sptaac_accs['mnist']*100:.2f}% | Fashion: {sptaac_accs['fashion']*100:.2f}% | CIFAR: {sptaac_accs['cifar']*100:.2f}% | Avg: {sptaac_accs['avg']*100:.2f}%")
        
        # Test Weight-Folded Calibration (WFC - Proposed)
        wfc_backbone = apply_wfc_folding(wa_backbone, gammas)
        wfc_accs = evaluate_multi_task(wfc_backbone, experts_paths, test_loaders)
        print(f"WFC (Folded-Ours) - MNIST: {wfc_accs['mnist']*100:.2f}% | Fashion: {wfc_accs['fashion']*100:.2f}% | CIFAR: {wfc_accs['cifar']*100:.2f}% | Avg: {wfc_accs['avg']*100:.2f}%")
        
        # Test mathematical equivalence of Hook-based SP-TAAC and Folded WFC
        diff = 0.0
        for name in ['mnist', 'fashion', 'cifar']:
            diff += abs(sptaac_accs[name] - wfc_accs[name])
        print(f"Mathematical Equivalence Verification: Mean Absolute Difference = {diff:.6f}")
        
        # Test proposed Least-Squares Head Alignment (LSHA - Proposed) on top of WFC backbone
        for reg in [0.01, 0.1, 1.0, 10.0]:
            aligned_heads = apply_lsha(wfc_backbone, experts_paths, task_loaders_cal, reg=reg)
            lsha_accs = evaluate_multi_task_aligned_heads(wfc_backbone, aligned_heads, test_loaders)
            print(f"WFC + LSHA (Ours, reg={reg}) - MNIST: {lsha_accs['mnist']*100:.2f}% | Fashion: {lsha_accs['fashion']*100:.2f}% | CIFAR: {lsha_accs['cifar']*100:.2f}% | Avg: {lsha_accs['avg']*100:.2f}%")
            
        print("-" * 50)
