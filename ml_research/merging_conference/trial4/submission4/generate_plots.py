import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

class ResNetBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_dataset(name, train=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if name == 'mnist':
        return torchvision.datasets.MNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'fashion':
        return torchvision.datasets.FashionMNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'kmnist':
        return torchvision.datasets.KMNIST(root='./data', train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")

def project_to_simplex(v):
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(n_features, device=v.device) + 1
    cond = u - cssv / ind > 0
    nonzero_indices = torch.nonzero(cond)
    if len(nonzero_indices) == 0:
        return torch.ones_like(v) / n_features
    rho = nonzero_indices[-1].item()
    theta = cssv[rho] / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    return w

def run_evaluation_with_history(
    batches, 
    method, 
    base_backbone, 
    base_backbone_params, 
    expert_backbones, 
    expert_heads, 
    task_vectors, 
    fisher_priors, 
    lr_head, 
    lr_lambda, 
    gamma, 
    device
):
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads]
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    
    lambda_history = []
    
    for step, ((inputs, targets), task_idx) in enumerate(batches):
        inputs, targets = inputs.to(device), targets.to(device)
        
        active_head = adapted_heads[task_idx]
        for p in active_head.parameters():
            p.requires_grad = True
            
        if method == 'static':
            static_lambda = torch.tensor([1/3, 1/3, 1/3], device=device)
            lambda_history.append(static_lambda.tolist())
            continue

        elif method == 'unconstrained_tta':
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            probs = F.softmax(logits, dim=-1)
            loss = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head.zero_grad()
            lambda_history.append(lambda_val.tolist())

        elif method == 'tf_ewc_dts': # Our Expert-Free EWC-TTA
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            probs = F.softmax(logits, dim=-1)
            loss_ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            
            loss_ewc = 0.0
            fim = fisher_priors[task_idx]
            init_head = expert_heads[task_idx]
            for p_name, p in active_head.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
                
            loss = loss_ent + gamma * loss_ewc
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head.zero_grad()
            lambda_history.append(lambda_val.tolist())

        elif method == 'eg_tvr_static':
            with torch.no_grad():
                entropies = []
                for k in range(3):
                    merged_params = {}
                    for param_key in base_backbone_params.keys():
                        merged_params[param_key] = base_backbone_params[param_key] + task_vectors[k][param_key]
                    features = functional_call(base_backbone, merged_params, inputs)
                    logits = adapted_heads[k](features)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
                    entropies.append(entropy)
            detected_k = entropies.index(min(entropies))
            hist_lambda = [0.0, 0.0, 0.0]
            hist_lambda[detected_k] = 1.0
            lambda_history.append(hist_lambda)

        elif method == 'eg_tvr_adaptive':
            with torch.no_grad():
                entropies = []
                for k in range(3):
                    merged_params = {}
                    for param_key in base_backbone_params.keys():
                        merged_params[param_key] = base_backbone_params[param_key] + task_vectors[k][param_key]
                    features = functional_call(base_backbone, merged_params, inputs)
                    logits = adapted_heads[k](features)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
                    entropies.append(entropy)
            detected_k = entropies.index(min(entropies))
            init_lambda = [0.1, 0.1, 0.1]
            init_lambda[detected_k] = 0.8
            lambda_val = torch.tensor(init_lambda, device=device, requires_grad=True)
            active_head_adapted = adapted_heads[detected_k]
            for p in active_head_adapted.parameters():
                p.requires_grad = True
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head_adapted(features)
            probs = F.softmax(logits, dim=-1)
            loss_ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            loss_ewc = 0.0
            fim = fisher_priors[detected_k]
            init_head = expert_heads[detected_k]
            for p_name, p in active_head_adapted.named_parameters():
                init_p = getattr(init_head, p_name)
                f_weight = fim[p_name]
                loss_ewc += 0.5 * (f_weight * (p - init_p) ** 2).sum()
            loss = loss_ent + gamma * loss_ewc
            loss.backward()
            with torch.no_grad():
                lambda_val.data -= lr_lambda * lambda_val.grad
                lambda_val.data = project_to_simplex(lambda_val.data)
                for p in active_head_adapted.parameters():
                    if p.grad is not None:
                        p.data -= lr_head * p.grad
            lambda_val.grad = None
            active_head_adapted.zero_grad()
            lambda_history.append(lambda_val.tolist())

    return np.array(lambda_history)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    base_backbone = ResNetBackbone(base_model).to(device)
    base_backbone_params = {k: v.clone().detach() for k, v in base_backbone.named_parameters()}
    
    expert_backbones = []
    expert_heads = []
    task_vectors = []
    fisher_priors = []
    
    for i in range(3):
        ckpt_path = f'checkpoints/expert_{i}.pt'
        fim_path = f'checkpoints/fim_{i}.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        m_expert = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        eb = ResNetBackbone(m_expert).to(device)
        eb.load_state_dict(checkpoint['backbone_state_dict'])
        eb.eval()
        expert_backbones.append(eb)
        
        eh = nn.Linear(512, 10).to(device)
        eh.load_state_dict(checkpoint['head_state_dict'])
        eh.eval()
        expert_heads.append(eh)
        
        eb_params = {k: v.clone().detach() for k, v in eb.named_parameters()}
        tv = {k: eb_params[k] - base_backbone_params[k] for k in base_backbone_params.keys()}
        task_vectors.append(tv)
        
        fim = torch.load(fim_path, map_location=device)
        fisher_priors.append(fim)
        
    mnist_test = get_dataset('mnist', train=False)
    fashion_test = get_dataset('fashion', train=False)
    kmnist_test = get_dataset('kmnist', train=False)
    
    mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=32, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=32, shuffle=False)
    
    mnist_batches = []
    for i, batch in enumerate(mnist_loader):
        if i >= 50: break
        mnist_batches.append((batch, 0))
        
    fashion_batches = []
    for i, batch in enumerate(fashion_loader):
        if i >= 50: break
        fashion_batches.append((batch, 1))
        
    kmnist_batches = []
    for i, batch in enumerate(kmnist_loader):
        if i >= 50: break
        kmnist_batches.append((batch, 2))
        
    sequential_batches = mnist_batches + fashion_batches + kmnist_batches
    alternating_batches = []
    for i in range(50):
        alternating_batches.append(mnist_batches[i])
        alternating_batches.append(fashion_batches[i])
        alternating_batches.append(kmnist_batches[i])

    print("Running methods to generate coefficient history...")
    
    # 1. Bar Chart of accuracies
    methods_bar = ['Static (Uniform)', 'Unconstrained TTA', 'EWC-TTA (Baseline)', 'Expert-Free EWC-TTA (Ours)', 'EG-TVR (Static, Ours)', 'EG-TVR (Adaptive, Ours)']
    seq_accs = [89.27, 93.92, 96.33, 93.92, 96.58, 96.62]
    alt_accs = [89.27, 49.46, 94.29, 53.88, 96.58, 96.62]
    
    x = np.arange(len(methods_bar))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, seq_accs, width, label='Sequential Stream', color='#1f77b4')
    rects2 = ax.bar(x + width/2, alt_accs, width, label='Alternating Stream', color='#ff7f0e')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Multi-task Classification Accuracy Across Test Streams')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_bar, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('results_bar_chart.pdf', dpi=300)
    plt.close()
    print("Saved results_bar_chart.pdf")

    # 2. Tracking Lambda for Sequential Stream
    print("Running Sequential Stream on TTA and EG-TVR...")
    hist_tta_seq = run_evaluation_with_history(
        sequential_batches, 'tf_ewc_dts',
        base_backbone, base_backbone_params, expert_backbones, expert_heads,
        task_vectors, fisher_priors, 1e-4, 0.5, 10.0, device
    )
    hist_r_seq = run_evaluation_with_history(
        sequential_batches, 'eg_tvr_adaptive',
        base_backbone, base_backbone_params, expert_backbones, expert_heads,
        task_vectors, fisher_priors, 1e-4, 0.5, 10.0, device
    )
    
    # Plotting Sequential Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    batches_x = np.arange(150)
    
    # Ax1: Expert-Free EWC-TTA Sequential
    ax1.plot(batches_x, hist_tta_seq[:, 0], label='MNIST Expert ($\lambda_0$)', color='g', alpha=0.8)
    ax1.plot(batches_x, hist_tta_seq[:, 1], label='Fashion Expert ($\lambda_1$)', color='r', alpha=0.8)
    ax1.plot(batches_x, hist_tta_seq[:, 2], label='KMNIST Expert ($\lambda_2$)', color='b', alpha=0.8)
    ax1.set_ylabel('Coefficients ($\lambda$)')
    ax1.set_title('Merging Coefficients Evolution: Expert-Free EWC-TTA (Sequential)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Ax2: EG-TVR (Adaptive) Sequential
    ax2.plot(batches_x, hist_r_seq[:, 0], label='MNIST Expert ($\lambda_0$)', color='g', alpha=0.8)
    ax2.plot(batches_x, hist_r_seq[:, 1], label='Fashion Expert ($\lambda_1$)', color='r', alpha=0.8)
    ax2.plot(batches_x, hist_r_seq[:, 2], label='KMNIST Expert ($\lambda_2$)', color='b', alpha=0.8)
    ax2.set_ylabel('Coefficients ($\lambda$)')
    ax2.set_xlabel('Batch Index')
    ax2.set_title('Merging Coefficients Evolution: EG-TVR (Adaptive, Sequential)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Vertical lines for task boundaries
    for ax_cur in [ax1, ax2]:
        ax_cur.axvline(x=50, color='black', linestyle=':', alpha=0.7)
        ax_cur.axvline(x=100, color='black', linestyle=':', alpha=0.7)
        ax_cur.text(25, 0.5, 'MNIST', fontsize=10, ha='center', fontweight='bold')
        ax_cur.text(75, 0.5, 'Fashion', fontsize=10, ha='center', fontweight='bold')
        ax_cur.text(125, 0.5, 'KMNIST', fontsize=10, ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('lambda_evolution_seq.pdf', dpi=300)
    plt.close()
    print("Saved lambda_evolution_seq.pdf")

    # 3. Tracking Lambda for Alternating Stream
    print("Running Alternating Stream on TTA and EG-TVR...")
    hist_tta_alt = run_evaluation_with_history(
        alternating_batches, 'tf_ewc_dts',
        base_backbone, base_backbone_params, expert_backbones, expert_heads,
        task_vectors, fisher_priors, 1e-4, 0.5, 10.0, device
    )
    hist_r_alt = run_evaluation_with_history(
        alternating_batches, 'eg_tvr_adaptive',
        base_backbone, base_backbone_params, expert_backbones, expert_heads,
        task_vectors, fisher_priors, 1e-4, 0.5, 10.0, device
    )
    
    # Plotting Alternating Comparison (first 30 batches for better readability)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    batches_sub = np.arange(30)
    
    ax1.plot(batches_sub, hist_tta_alt[:30, 0], 'o-', label='MNIST Expert ($\lambda_0$)', color='g', alpha=0.8)
    ax1.plot(batches_sub, hist_tta_alt[:30, 1], 's-', label='Fashion Expert ($\lambda_1$)', color='r', alpha=0.8)
    ax1.plot(batches_sub, hist_tta_alt[:30, 2], '^-', label='KMNIST Expert ($\lambda_2$)', color='b', alpha=0.8)
    ax1.set_ylabel('Coefficients ($\lambda$)')
    ax1.set_title('Merging Coefficients Evolution: Expert-Free EWC-TTA (Alternating, first 30 batches)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(batches_sub, hist_r_alt[:30, 0], 'o-', label='MNIST Expert ($\lambda_0$)', color='g', alpha=0.8)
    ax2.plot(batches_sub, hist_r_alt[:30, 1], 's-', label='Fashion Expert ($\lambda_1$)', color='r', alpha=0.8)
    ax2.plot(batches_sub, hist_r_alt[:30, 2], '^-', label='KMNIST Expert ($\lambda_2$)', color='b', alpha=0.8)
    ax2.set_ylabel('Coefficients ($\lambda$)')
    ax2.set_xlabel('Batch Index')
    ax2.set_title('Merging Coefficients Evolution: EG-TVR (Adaptive, Alternating, first 30 batches)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Shade background to show active tasks
    # alternating is Task 0, 1, 2, 0, 1, 2...
    for i in range(30):
        task_type = i % 3
        if task_type == 0:
            color_shade = 'g'
        elif task_type == 1:
            color_shade = 'r'
        else:
            color_shade = 'b'
        ax1.axvspan(i - 0.5, i + 0.5, color=color_shade, alpha=0.08)
        ax2.axvspan(i - 0.5, i + 0.5, color=color_shade, alpha=0.08)

    plt.tight_layout()
    plt.savefig('lambda_evolution_alt.pdf', dpi=300)
    plt.close()
    print("Saved lambda_evolution_alt.pdf")
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
