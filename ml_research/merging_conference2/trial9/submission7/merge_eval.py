import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import os
import copy
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Datasets & Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

print("Loading test datasets...")
mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=mnist_transform)
fmnist_test = datasets.FashionMNIST(root='data', train=False, download=True, transform=fmnist_transform)
cifar_test = datasets.CIFAR10(root='data', train=False, download=True, transform=cifar_transform)

batch_size = 256
mnist_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
fmnist_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
cifar_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)

loaders = {
    'mnist': mnist_loader,
    'fmnist': fmnist_loader,
    'cifar10': cifar_loader
}

# Model Definitions
def get_resnet_progenitor():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Identity()
    return model

class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class JointModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.fc = head
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def evaluate_model(backbone, head, loader):
    backbone = copy.deepcopy(backbone).to(device)
    head = copy.deepcopy(head).to(device)
    model = JointModel(backbone, head)
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
            
    return correct / total * 100.0

# Merging functions
def merge_weight_averaging(state_dicts):
    merged = copy.deepcopy(state_dicts[0])
    for key in merged.keys():
        if merged[key].is_floating_point():
            merged[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return merged

def merge_task_arithmetic(progenitor_state, expert_states, lmbda):
    merged = copy.deepcopy(progenitor_state)
    for key in merged.keys():
        if merged[key].is_floating_point():
            # tau_k = W_k - W_init
            tau_sum = sum(sd[key] - progenitor_state[key] for sd in expert_states)
            merged[key] = progenitor_state[key] + lmbda * tau_sum
    return merged

def get_bn_to_conv_map(model):
    bn_to_conv = {}
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            last_conv_name = name
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if last_conv_name is not None:
                bn_to_conv[name] = last_conv_name
    return bn_to_conv

def merge_weight_averaging_crvs(state_dicts, gamma):
    merged = merge_weight_averaging(state_dicts)
    for key in merged.keys():
        if 'running_var' in key:
            merged[key] = merged[key] * gamma
    return merged

def merge_weight_averaging_cos_rvs(expert_models, state_dicts, min_gamma=0.1, max_gamma=1.0, beta=1.0):
    K = len(state_dicts)
    merged = merge_weight_averaging(state_dicts)
    bn_to_conv = get_bn_to_conv_map(expert_models[0])
    
    for bn_name, conv_name in bn_to_conv.items():
        bn_var_key = f"{bn_name}.running_var"
        var_key = None
        for k in merged.keys():
            if k.endswith(bn_var_key):
                var_key = k
                break
                
        conv_key = None
        for k in merged.keys():
            if k.endswith(f"{conv_name}.weight"):
                conv_key = k
                break
                
        if var_key is not None and conv_key is not None:
            weights = [sd[conv_key] for sd in state_dicts]
            num_channels = weights[0].shape[0]
            flattened_weights = [w.reshape(num_channels, -1) for w in weights]
            
            gammas = []
            for c in range(num_channels):
                cos_sims = []
                for i in range(K):
                    for j in range(i + 1, K):
                        w_i = flattened_weights[i][c]
                        w_j = flattened_weights[j][c]
                        norm_i = torch.norm(w_i)
                        norm_j = torch.norm(w_j)
                        if norm_i > 1e-8 and norm_j > 1e-8:
                            sim = torch.dot(w_i, w_j) / (norm_i * norm_j)
                        else:
                            sim = torch.tensor(0.0, device=w_i.device)
                        cos_sims.append(sim)
                gamma_c = 1.0 / K + (2.0 / (K * K)) * sum(cos_sims)
                gamma_c = gamma_c * beta
                gamma_c = torch.clamp(gamma_c, min_gamma, max_gamma)
                gammas.append(gamma_c)
                
            gammas = torch.stack(gammas).to(merged[var_key].device)
            merged[var_key] = merged[var_key] * gammas
    return merged

def merge_task_arithmetic_crvs(progenitor_state, expert_states, lmbda, gamma):
    merged = merge_task_arithmetic(progenitor_state, expert_states, lmbda)
    for key in merged.keys():
        if 'running_var' in key:
            merged[key] = merged[key] * gamma
    return merged

def merge_task_arithmetic_cos_rvs(expert_models, progenitor_state, expert_states, lmbda, min_gamma=0.1, max_gamma=1.0, beta=1.0):
    K = len(expert_states)
    merged = merge_task_arithmetic(progenitor_state, expert_states, lmbda)
    bn_to_conv = get_bn_to_conv_map(expert_models[0])
    
    for bn_name, conv_name in bn_to_conv.items():
        bn_var_key = f"{bn_name}.running_var"
        var_key = None
        for k in merged.keys():
            if k.endswith(bn_var_key):
                var_key = k
                break
                
        conv_key = None
        for k in merged.keys():
            if k.endswith(f"{conv_name}.weight"):
                conv_key = k
                break
                
        if var_key is not None and conv_key is not None:
            weights = [sd[conv_key] for sd in expert_states]
            num_channels = weights[0].shape[0]
            flattened_weights = [w.reshape(num_channels, -1) for w in weights]
            
            gammas = []
            for c in range(num_channels):
                cos_sims = []
                for i in range(K):
                    for j in range(i + 1, K):
                        w_i = flattened_weights[i][c]
                        w_j = flattened_weights[j][c]
                        norm_i = torch.norm(w_i)
                        norm_j = torch.norm(w_j)
                        if norm_i > 1e-8 and norm_j > 1e-8:
                            sim = torch.dot(w_i, w_j) / (norm_i * norm_j)
                        else:
                            sim = torch.tensor(0.0, device=w_i.device)
                        cos_sims.append(sim)
                gamma_c = 1.0 / K + (2.0 / (K * K)) * sum(cos_sims)
                gamma_c = gamma_c * beta
                gamma_c = torch.clamp(gamma_c, min_gamma, max_gamma)
                gammas.append(gamma_c)
                
            gammas = torch.stack(gammas).to(merged[var_key].device)
            merged[var_key] = merged[var_key] * gammas
    return merged

# BatchNorm Calibration functions
def calibrate_bn_real(model, calibration_loaders, num_samples_per_task):
    model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = False
        
    orig_momentums = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            orig_momentums[name] = m.momentum
            m.momentum = 1.0  # Force exact statistics on the current batch
            
    # Gather a combined calibration set from all tasks
    cal_images = []
    for loader in calibration_loaders:
        collected = 0
        for x, _ in loader:
            cal_images.append(x)
            collected += x.size(0)
            if collected >= num_samples_per_task:
                break
                
    if len(cal_images) > 0:
        cal_x = torch.cat(cal_images, dim=0).to(device)
        with torch.no_grad():
            _ = model(cal_x)
            
    # Restore original momentums
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = orig_momentums[name]

def calibrate_bn_synthetic(model, num_samples, noise_type='gaussian', std=1.0):
    model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = False
        
    orig_momentums = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            orig_momentums[name] = m.momentum
            m.momentum = 1.0  # Force exact statistics on the current batch
            
    if noise_type == 'gaussian':
        synthetic_x = torch.randn(num_samples, 3, 32, 32).to(device) * std
    elif noise_type == 'uniform':
        synthetic_x = (torch.rand(num_samples, 3, 32, 32).to(device) * 2.0 - 1.0) * std
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
        
    with torch.no_grad():
        _ = model(synthetic_x)
        
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = orig_momentums[name]

# Main execution
def run_evaluation():
    # 1. Load trained models
    print("\nLoading trained expert models...")
    
    # ResNet-18 Progenitor & Experts
    resnet_progenitor = get_resnet_progenitor()
    resnet_progenitor.load_state_dict(torch.load('checkpoints/resnet_progenitor.pt'))
    
    resnet_experts = {}
    resnet_heads = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        backbone = get_resnet_progenitor()
        backbone.load_state_dict(torch.load(f'checkpoints/resnet_{task}_backbone.pt'))
        resnet_experts[task] = backbone
        
        fc = nn.Linear(512, 10)
        fc.load_state_dict(torch.load(f'checkpoints/resnet_{task}_fc.pt'))
        resnet_heads[task] = fc
        
    # MLP Progenitor & Experts
    mlp_progenitor = MLPBackbone()
    mlp_progenitor.load_state_dict(torch.load('checkpoints/mlp_progenitor.pt'))
    
    mlp_experts = {}
    mlp_heads = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        backbone = MLPBackbone()
        backbone.load_state_dict(torch.load(f'checkpoints/mlp_{task}_backbone.pt'))
        mlp_experts[task] = backbone
        
        fc = nn.Linear(512, 10)
        fc.load_state_dict(torch.load(f'checkpoints/mlp_{task}_fc.pt'))
        mlp_heads[task] = fc

    # Load calibration loaders (use a small batch from test loaders for DE-BN)
    cal_loaders = [loaders['mnist'], loaders['fmnist'], loaders['cifar10']]

    print("\n================== RESNET-18 MERGING RESULTS ==================")
    # Oracle accuracies
    oracle_accs = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        acc = evaluate_model(resnet_experts[task], resnet_heads[task], loaders[task])
        oracle_accs[task] = acc
        print(f"ResNet-18 {task.upper()} Oracle: {acc:.2f}%")
    print(f"Average Oracle Accuracy: {np.mean(list(oracle_accs.values())):.2f}%")

    # Weight Averaging (WA) Uncalibrated
    wa_state = merge_weight_averaging([resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']])
    wa_backbone = get_resnet_progenitor()
    wa_backbone.load_state_dict(wa_state)
    
    wa_accs = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        wa_accs[task] = evaluate_model(wa_backbone, resnet_heads[task], loaders[task])
    avg_wa = np.mean(list(wa_accs.values()))
    print(f"\nWA Uncalibrated: MNIST={wa_accs['mnist']:.2f}%, FMNIST={wa_accs['fmnist']:.2f}%, CIFAR10={wa_accs['cifar10']:.2f}%, Average={avg_wa:.2f}%")

    # C-RVS Calibration on WA
    print("\nEvaluating C-RVS (Constant Running Variance Scaling) on Weight Averaging...")
    for gamma in [0.05, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        crvs_state = merge_weight_averaging_crvs([resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], gamma)
        crvs_backbone = get_resnet_progenitor()
        crvs_backbone.load_state_dict(crvs_state)
        
        crvs_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            crvs_accs[task] = evaluate_model(crvs_backbone, resnet_heads[task], loaders[task])
        avg_crvs = np.mean(list(crvs_accs.values()))
        print(f"  gamma={gamma:.2f}: MNIST={crvs_accs['mnist']:.2f}%, FMNIST={crvs_accs['fmnist']:.2f}%, CIFAR10={crvs_accs['cifar10']:.2f}%, Average={avg_crvs:.2f}%")

    # Cos-RVS Calibration on WA
    print("\nEvaluating Cos-RVS (Cosine-Adaptive Running Variance Scaling) on Weight Averaging...")
    for min_g in [0.05, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5]:
        cos_rvs_state = merge_weight_averaging_cos_rvs([resnet_experts[t] for t in ['mnist', 'fmnist', 'cifar10']], 
                                                       [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                                       min_gamma=min_g)
        cos_rvs_backbone = get_resnet_progenitor()
        cos_rvs_backbone.load_state_dict(cos_rvs_state)
        
        cos_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cos_accs[task] = evaluate_model(cos_rvs_backbone, resnet_heads[task], loaders[task])
        avg_cos = np.mean(list(cos_accs.values()))
        print(f"  min_gamma={min_g:.2f}: MNIST={cos_accs['mnist']:.2f}%, FMNIST={cos_accs['fmnist']:.2f}%, CIFAR10={cos_accs['cifar10']:.2f}%, Average={avg_cos:.2f}%")

    # S-Cos-RVS (Scaled Cos-RVS) Calibration on WA
    print("\nEvaluating S-Cos-RVS (Scaled Cosine-Adaptive Running Variance Scaling) on Weight Averaging...")
    for beta in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        cos_rvs_state = merge_weight_averaging_cos_rvs([resnet_experts[t] for t in ['mnist', 'fmnist', 'cifar10']], 
                                                       [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                                       min_gamma=0.1, beta=beta)
        cos_rvs_backbone = get_resnet_progenitor()
        cos_rvs_backbone.load_state_dict(cos_rvs_state)
        
        cos_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cos_accs[task] = evaluate_model(cos_rvs_backbone, resnet_heads[task], loaders[task])
        avg_cos = np.mean(list(cos_accs.values()))
        print(f"  beta={beta:.2f}: MNIST={cos_accs['mnist']:.2f}%, FMNIST={cos_accs['fmnist']:.2f}%, CIFAR10={cos_accs['cifar10']:.2f}%, Average={avg_cos:.2f}%")

    # Tuned Task Arithmetic
    best_ta_lambda = 0.1
    best_ta_acc = 0.0
    best_ta_backbone = None
    best_ta_accs = {}
    
    print("\nSweeping Task Arithmetic lambda...")
    for lmbda in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        ta_state = merge_task_arithmetic(resnet_progenitor.state_dict(), 
                                         [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                         lmbda)
        ta_backbone = get_resnet_progenitor()
        ta_backbone.load_state_dict(ta_state)
        
        ta_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            ta_accs[task] = evaluate_model(ta_backbone, resnet_heads[task], loaders[task])
        avg_ta = np.mean(list(ta_accs.values()))
        print(f"  lambda={lmbda:.2f}: MNIST={ta_accs['mnist']:.2f}%, FMNIST={ta_accs['fmnist']:.2f}%, CIFAR10={ta_accs['cifar10']:.2f}%, Average={avg_ta:.2f}%")
        
        if avg_ta > best_ta_acc:
            best_ta_acc = avg_ta
            best_ta_lambda = lmbda
            best_ta_backbone = ta_backbone
            best_ta_accs = ta_accs
            
    print(f"Best ResNet-18 Task Arithmetic: lambda={best_ta_lambda:.2f}, Average Accuracy={best_ta_acc:.2f}%")

    # C-RVS on Best Tuned TA
    print(f"\nEvaluating C-RVS on Best Tuned TA (lambda={best_ta_lambda:.2f})...")
    for gamma in [0.05, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        crvs_ta_state = merge_task_arithmetic_crvs(resnet_progenitor.state_dict(), 
                                                   [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                                   best_ta_lambda, gamma)
        crvs_ta_backbone = get_resnet_progenitor()
        crvs_ta_backbone.load_state_dict(crvs_ta_state)
        
        crvs_ta_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            crvs_ta_accs[task] = evaluate_model(crvs_ta_backbone, resnet_heads[task], loaders[task])
        avg_crvs_ta = np.mean(list(crvs_ta_accs.values()))
        print(f"  gamma={gamma:.2f}: MNIST={crvs_ta_accs['mnist']:.2f}%, FMNIST={crvs_ta_accs['fmnist']:.2f}%, CIFAR10={crvs_ta_accs['cifar10']:.2f}%, Average={avg_crvs_ta:.2f}%")

    # Cos-RVS on Best Tuned TA
    print(f"\nEvaluating Cos-RVS on Best Tuned TA (lambda={best_ta_lambda:.2f})...")
    for min_g in [0.05, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5]:
        cos_ta_state = merge_task_arithmetic_cos_rvs([resnet_experts[t] for t in ['mnist', 'fmnist', 'cifar10']], 
                                                     resnet_progenitor.state_dict(), 
                                                     [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                                     best_ta_lambda, min_gamma=min_g)
        cos_ta_backbone = get_resnet_progenitor()
        cos_ta_backbone.load_state_dict(cos_ta_state)
        
        cos_ta_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cos_ta_accs[task] = evaluate_model(cos_ta_backbone, resnet_heads[task], loaders[task])
        avg_cos_ta = np.mean(list(cos_ta_accs.values()))
        print(f"  min_gamma={min_g:.2f}: MNIST={cos_ta_accs['mnist']:.2f}%, FMNIST={cos_ta_accs['fmnist']:.2f}%, CIFAR10={cos_ta_accs['cifar10']:.2f}%, Average={avg_cos_ta:.2f}%")

    # S-Cos-RVS (Scaled Cos-RVS) on Best Tuned TA
    print(f"\nEvaluating S-Cos-RVS on Best Tuned TA (lambda={best_ta_lambda:.2f})...")
    for beta in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        cos_ta_state = merge_task_arithmetic_cos_rvs([resnet_experts[t] for t in ['mnist', 'fmnist', 'cifar10']], 
                                                     resnet_progenitor.state_dict(), 
                                                     [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                                     best_ta_lambda, min_gamma=0.1, beta=beta)
        cos_ta_backbone = get_resnet_progenitor()
        cos_ta_backbone.load_state_dict(cos_ta_state)
        
        cos_ta_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cos_ta_accs[task] = evaluate_model(cos_ta_backbone, resnet_heads[task], loaders[task])
        avg_cos_ta = np.mean(list(cos_ta_accs.values()))
        print(f"  beta={beta:.2f}: MNIST={cos_ta_accs['mnist']:.2f}%, FMNIST={cos_ta_accs['fmnist']:.2f}%, CIFAR10={cos_ta_accs['cifar10']:.2f}%, Average={avg_cos_ta:.2f}%")

    # DE-BN Calibration on WA
    print("\nEvaluating DE-BN on Weight Averaging...")
    for n in [8, 16, 32, 64, 128]:
        cal_backbone = copy.deepcopy(wa_backbone)
        calibrate_bn_real(cal_backbone, cal_loaders, n)
        
        cal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cal_accs[task] = evaluate_model(cal_backbone, resnet_heads[task], loaders[task])
        avg_cal = np.mean(list(cal_accs.values()))
        print(f"  DE-BN (N={n} per task): MNIST={cal_accs['mnist']:.2f}%, FMNIST={cal_accs['fmnist']:.2f}%, CIFAR10={cal_accs['cifar10']:.2f}%, Average={avg_cal:.2f}%")

    # DF-SBC (Gaussian Noise Calibration) on WA
    print("\nEvaluating DF-SBC (Gaussian Noise) on Weight Averaging...")
    for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
        cal_backbone = copy.deepcopy(wa_backbone)
        calibrate_bn_synthetic(cal_backbone, n, noise_type='gaussian', std=1.0)
        
        cal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cal_accs[task] = evaluate_model(cal_backbone, resnet_heads[task], loaders[task])
        avg_cal = np.mean(list(cal_accs.values()))
        print(f"  DF-SBC (N={n} samples, standard Gaussian): MNIST={cal_accs['mnist']:.2f}%, FMNIST={cal_accs['fmnist']:.2f}%, CIFAR10={cal_accs['cifar10']:.2f}%, Average={avg_cal:.2f}%")

    # DF-SBC (Uniform Noise Calibration) on WA
    print("\nEvaluating DF-SBC (Uniform Noise) on Weight Averaging...")
    for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
        cal_backbone = copy.deepcopy(wa_backbone)
        calibrate_bn_synthetic(cal_backbone, n, noise_type='uniform', std=1.0)
        
        cal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cal_accs[task] = evaluate_model(cal_backbone, resnet_heads[task], loaders[task])
        avg_cal = np.mean(list(cal_accs.values()))
        print(f"  DF-SBC (N={n} samples, standard Uniform): MNIST={cal_accs['mnist']:.2f}%, FMNIST={cal_accs['fmnist']:.2f}%, CIFAR10={cal_accs['cifar10']:.2f}%, Average={avg_cal:.2f}%")

    # DF-SBC on Tuned TA
    print(f"\nEvaluating DF-SBC (Gaussian Noise) on Best Tuned TA (lambda={best_ta_lambda:.2f})...")
    for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
        cal_backbone = copy.deepcopy(best_ta_backbone)
        calibrate_bn_synthetic(cal_backbone, n, noise_type='gaussian', std=1.0)
        
        cal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            cal_accs[task] = evaluate_model(cal_backbone, resnet_heads[task], loaders[task])
        avg_cal = np.mean(list(cal_accs.values()))
        print(f"  DF-SBC on TA (N={n} samples): MNIST={cal_accs['mnist']:.2f}%, FMNIST={cal_accs['fmnist']:.2f}%, CIFAR10={cal_accs['cifar10']:.2f}%, Average={avg_cal:.2f}%")


    print("\n================== MLP MERGING RESULTS ==================")
    # MLP Oracle accuracies
    mlp_oracle_accs = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        acc = evaluate_model(mlp_experts[task], mlp_heads[task], loaders[task])
        mlp_oracle_accs[task] = acc
        print(f"MLP {task.upper()} Oracle: {acc:.2f}%")
    print(f"MLP Average Oracle Accuracy: {np.mean(list(mlp_oracle_accs.values())):.2f}%")

    # MLP Weight Averaging (WA) Uncalibrated
    mlp_wa_state = merge_weight_averaging([mlp_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']])
    mlp_wa_backbone = MLPBackbone()
    mlp_wa_backbone.load_state_dict(mlp_wa_state)
    
    mlp_wa_accs = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        mlp_wa_accs[task] = evaluate_model(mlp_wa_backbone, mlp_heads[task], loaders[task])
    avg_mlp_wa = np.mean(list(mlp_wa_accs.values()))
    print(f"\nMLP WA Uncalibrated: MNIST={mlp_wa_accs['mnist']:.2f}%, FMNIST={mlp_wa_accs['fmnist']:.2f}%, CIFAR10={mlp_wa_accs['cifar10']:.2f}%, Average={avg_mlp_wa:.2f}%")

    # MLP Tuned Task Arithmetic
    best_mlp_ta_lambda = 0.1
    best_mlp_ta_acc = 0.0
    best_mlp_ta_accs = {}
    
    print("\nSweeping MLP Task Arithmetic lambda...")
    for lmbda in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        ta_state = merge_task_arithmetic(mlp_progenitor.state_dict(), 
                                         [mlp_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']], 
                                         lmbda)
        ta_backbone = MLPBackbone()
        ta_backbone.load_state_dict(ta_state)
        
        ta_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            ta_accs[task] = evaluate_model(ta_backbone, mlp_heads[task], loaders[task])
        avg_ta = np.mean(list(ta_accs.values()))
        print(f"  lambda={lmbda:.2f}: MNIST={ta_accs['mnist']:.2f}%, FMNIST={ta_accs['fmnist']:.2f}%, CIFAR10={ta_accs['cifar10']:.2f}%, Average={avg_ta:.2f}%")
        
        if avg_ta > best_mlp_ta_acc:
            best_mlp_ta_acc = avg_ta
            best_mlp_ta_lambda = lmbda
            best_mlp_ta_accs = ta_accs
            
    print(f"Best MLP Task Arithmetic: lambda={best_mlp_ta_lambda:.2f}, Average Accuracy={best_mlp_ta_acc:.2f}%")

if __name__ == '__main__':
    run_evaluation()
