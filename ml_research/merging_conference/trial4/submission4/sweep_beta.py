import os
import argparse
import copy
import json
import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights

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

def run_evaluation(
    stream_name, 
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
    dts_alpha, 
    dts_beta, 
    device
):
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads]
    lambda_val = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    
    dts_ema_entropy = 1.0
    correct_count = 0
    total_count = 0
    
    for step, ((inputs, targets), task_idx) in enumerate(batches):
        inputs, targets = inputs.to(device), targets.to(device)
        active_head = adapted_heads[task_idx]
        for p in active_head.parameters():
            p.requires_grad = True
            
        merged_params = {}
        for k in base_backbone_params.keys():
            merged_params[k] = base_backbone_params[k] + sum(
                lambda_val[i] * task_vectors[i][k] for i in range(3)
            )
            
        features = functional_call(base_backbone, merged_params, inputs)
        logits = active_head(features)
        
        # Standard entropy to update running estimate
        with torch.no_grad():
            probs_std = F.softmax(logits, dim=-1)
            batch_entropy = -(probs_std * torch.log(probs_std + 1e-12)).sum(dim=-1).mean().item()
        dts_ema_entropy = dts_alpha * dts_ema_entropy + (1 - dts_alpha) * batch_entropy
        
        # Scale logits
        T = 1.0 + dts_beta * dts_ema_entropy
        probs_scaled = F.softmax(logits / T, dim=-1)
        
        # Loss
        loss_ent = -(probs_scaled * torch.log(probs_scaled + 1e-12)).sum(dim=-1).mean()
        
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
        
        # Final inference
        with torch.no_grad():
            merged_params = {}
            for k in base_backbone_params.keys():
                merged_params[k] = base_backbone_params[k] + sum(
                    lambda_val[i] * task_vectors[i][k] for i in range(3)
                )
            features = functional_call(base_backbone, merged_params, inputs)
            logits = active_head(features)
            _, predicted = logits.max(1)
            correct_count += predicted.eq(targets).sum().item()
            total_count += targets.size(0)
            
    accuracy = 100.0 * correct_count / total_count
    return accuracy

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
        
    print("\n--- SWEEPING TEMPERATURE BETA AND GAMMA ---")
    betas = [0.0, 0.05, 0.1, 0.5]
    gammas = [10.0, 100.0]
    lr_lambdas = [0.5]
    
    for beta in betas:
        for gamma in gammas:
            for lr_lam in lr_lambdas:
                seq_acc = run_evaluation(
                    'Sequential', sequential_batches, 'tf_ewc_dts',
                    base_backbone, base_backbone_params, expert_backbones, expert_heads,
                    task_vectors, fisher_priors, 1e-4, lr_lam, gamma, 0.9, beta, device
                )
                alt_acc = run_evaluation(
                    'Alternating', alternating_batches, 'tf_ewc_dts',
                    base_backbone, base_backbone_params, expert_backbones, expert_heads,
                    task_vectors, fisher_priors, 1e-4, lr_lam, gamma, 0.9, beta, device
                )
                print(f"beta={beta:<5} | gamma={gamma:<5} | lr_lam={lr_lam:<5} | Seq: {seq_acc:.2f}% | Alt: {alt_acc:.2f}%")

if __name__ == "__main__":
    main()
