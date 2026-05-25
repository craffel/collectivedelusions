import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

set_seed(42)

# Define Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_bn_modules(model):
    bn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_modules[name] = module
    return bn_modules

def fuse_bn_buffers(merged_model, experts, posterior_weights):
    with torch.no_grad():
        merged_bn = get_bn_modules(merged_model)
        expert_bns = [get_bn_modules(expert) for expert in experts]
        
        for name, m_bn in merged_bn.items():
            means = [e_bn[name].running_mean for e_bn in expert_bns]
            vars = [e_bn[name].running_var for e_bn in expert_bns]
            
            fused_mean = sum(posterior_weights[k] * means[k] for k in range(len(experts)))
            fused_var = sum(posterior_weights[k] * (vars[k] + (means[k] - fused_mean)**2) for k in range(len(experts)))
            
            m_bn.running_mean.copy_(fused_mean)
            m_bn.running_var.copy_(fused_var)

def hard_merge_bn_buffers(merged_model, experts, active_expert_idx):
    with torch.no_grad():
        merged_bn = get_bn_modules(merged_model)
        active_bn = get_bn_modules(experts[active_expert_idx])
        for name, m_bn in merged_bn.items():
            m_bn.running_mean.copy_(active_bn[name].running_mean)
            m_bn.running_var.copy_(active_bn[name].running_var)

def get_merged_state_dict(experts_sds, lambdas, keys_to_merge):
    merged_sd = {}
    first_sd = experts_sds[0]
    for key in first_sd.keys():
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        
        if key in keys_to_merge:
            coefs = torch.softmax(lambdas[key], dim=0)
            merged_sd[key] = sum(coefs[k] * experts_sds[k][key] for k in range(len(experts_sds)))
        else:
            coefs = torch.softmax(lambdas['global'], dim=0)
            merged_sd[key] = sum(coefs[k] * experts_sds[k][key] for k in range(len(experts_sds)))
            
    return merged_sd

def compute_test_time_fisher(model, x, device):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            
    logits = model(x)
    pseudo_labels = logits.argmax(dim=-1)
    loss = F.cross_entropy(logits, pseudo_labels)
    
    model.zero_grad()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher[name].copy_(param.grad.data ** 2 + 1e-5)
            
    return fisher

def get_layer_fisher_sensitivity(fisher_dict, keys_to_merge):
    layer_sensitivities = {}
    for key in keys_to_merge:
        if key in fisher_dict:
            layer_sensitivities[key] = fisher_dict[key].sum().item()
        else:
            layer_sensitivities[key] = 1.0
    return layer_sensitivities

def evaluate_custom_ours(stream_batches, experts, experts_sds, keys_to_merge, device,
                         bn_fusion_strategy="soft", # "soft", "hard", "uniform"
                         beta=0.5,
                         gamma=15.0,
                         inner_steps=3,
                         tau_n=1.2,
                         global_mode="learnable"): # "learnable", "average"
    K = len(experts)
    lambdas = {key: torch.zeros(K, device=device, requires_grad=True) for key in keys_to_merge}
    if global_mode == "learnable":
        lambdas['global'] = torch.zeros(K, device=device, requires_grad=True)
    else:
        lambdas['global'] = torch.stack([lambdas[k] for k in keys_to_merge]).mean(dim=0)
    
    prev_lambdas_val = {key: torch.zeros(K, device=device) for key in keys_to_merge}
    prev_lambdas_val['global'] = torch.zeros(K, device=device)
    
    merged_model = SimpleCNN().to(device)
    correct_total = 0
    samples_total = 0
    
    for step, (x, y, domain_lbl) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # 1. Soft posterior estimation (using predictive entropy)
        with torch.no_grad():
            entropies = []
            for expert in experts:
                expert_logits = expert(x)
                expert_probs = F.softmax(expert_logits, dim=-1)
                expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10), dim=-1).mean().item()
                entropies.append(expert_entropy)
                
            exp_neg_ent = np.exp(-gamma * np.array(entropies))
            posterior_weights = exp_neg_ent / np.sum(exp_neg_ent)
            active_expert_idx = np.argmin(entropies)
            
        avg_expert_entropy = np.mean(entropies)
        is_novel = False
        if avg_expert_entropy > tau_n:
            is_novel = True
            
        # 2. Initialize merging coefficients based on posterior or uniform if novel
        with torch.no_grad():
            for key in keys_to_merge:
                if is_novel:
                    lambdas[key].copy_(torch.zeros(K, device=device))
                else:
                    soft_logits = torch.log(torch.tensor(posterior_weights, device=device) + 1e-10)
                    lambdas[key].copy_(soft_logits)
            
            if global_mode == "learnable":
                if is_novel:
                    lambdas['global'].copy_(torch.zeros(K, device=device))
                else:
                    soft_logits = torch.log(torch.tensor(posterior_weights, device=device) + 1e-10)
                    lambdas['global'].copy_(soft_logits)
            else:
                lambdas['global'] = torch.stack([lambdas[k] for k in keys_to_merge]).mean(dim=0)
                    
        # Load merged weights and perform selected BN Buffer fusion
        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
        merged_model.load_state_dict(sd, strict=False)
        
        def apply_bn_fusion(weights):
            if is_novel:
                fuse_bn_buffers(merged_model, experts, [1.0/K]*K)
            else:
                if bn_fusion_strategy == "soft":
                    fuse_bn_buffers(merged_model, experts, weights)
                elif bn_fusion_strategy == "hard":
                    hard_merge_bn_buffers(merged_model, experts, active_expert_idx)
                elif bn_fusion_strategy == "uniform":
                    fuse_bn_buffers(merged_model, experts, [1.0/K]*K)
        
        apply_bn_fusion(posterior_weights)
        
        # Compute Test-Time Fisher
        tt_fisher = compute_test_time_fisher(merged_model, x, device)
        layer_sensitivities = get_layer_fisher_sensitivity(tt_fisher, keys_to_merge)
        
        if global_mode == "learnable":
            optimizer = optim.Adam(list(lambdas.values()), lr=1e-2)
        else:
            optimizer = optim.Adam([lambdas[k] for k in keys_to_merge], lr=1e-2)
        
        # Fine-tuning steps (MAP estimation)
        for step_inner in range(inner_steps):
            optimizer.zero_grad()
            if global_mode == "average":
                lambdas['global'] = torch.stack([lambdas[k] for k in keys_to_merge]).mean(dim=0)
                
            sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
            merged_model.load_state_dict(sd, strict=False)
            
            with torch.no_grad():
                current_weights = torch.softmax(lambdas['global'], dim=0).cpu().numpy().tolist()
            
            apply_bn_fusion(current_weights)
                
            logits = merged_model(x)
            probs = F.softmax(logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            # Prior L2 regularization to prevent catastrophic drift
            prior_reg = 0.0
            for key in keys_to_merge:
                prior_reg += torch.sum((lambdas[key] - prev_lambdas_val[key]) ** 2)
            if global_mode == "learnable":
                prior_reg += torch.sum((lambdas['global'] - prev_lambdas_val['global']) ** 2)
            else:
                # Use current derived global and previous derived global
                prior_reg += torch.sum((lambdas['global'] - prev_lambdas_val['global']) ** 2)
            
            total_loss = entropy_loss + (beta / 2.0) * prior_reg
            total_loss.backward()
            
            # Apply Fisher Preconditioning
            with torch.no_grad():
                for key in keys_to_merge:
                    if lambdas[key].grad is not None:
                        scale = 1.0 / (layer_sensitivities[key] + 1e-5)
                        lambdas[key].grad.copy_(lambdas[key].grad * scale)
                        
            optimizer.step()
            
        with torch.no_grad():
            for key in keys_to_merge:
                prev_lambdas_val[key].copy_(lambdas[key].detach())
            if global_mode == "learnable":
                prev_lambdas_val['global'].copy_(lambdas['global'].detach())
            else:
                lambdas['global'] = torch.stack([lambdas[k] for k in keys_to_merge]).mean(dim=0)
                prev_lambdas_val['global'].copy_(lambdas['global'].detach())
                
        # Final evaluation on the batch
        with torch.no_grad():
            if global_mode == "average":
                lambdas['global'] = torch.stack([lambdas[k] for k in keys_to_merge]).mean(dim=0)
            sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
            merged_model.load_state_dict(sd, strict=False)
            current_weights = torch.softmax(lambdas['global'], dim=0).cpu().numpy().tolist()
            apply_bn_fusion(current_weights)
                
            logits = merged_model(x)
            preds = logits.argmax(dim=-1)
            correct = preds.eq(y).sum().item()
            correct_total += correct
            samples_total += len(y)
            
    return correct_total / samples_total

def get_streams(batch_size, mnist_test, kmnist_test, fmnist_test):
    # Standardize evaluation on exactly 960 samples per domain
    total_samples_per_domain = 960
    num_batches = total_samples_per_domain // batch_size
    
    mnist_test_subset = Subset(mnist_test, list(range(total_samples_per_domain)))
    kmnist_test_subset = Subset(kmnist_test, list(range(total_samples_per_domain)))
    fmnist_test_subset = Subset(fmnist_test, list(range(total_samples_per_domain)))
    
    mnist_test_loader = DataLoader(mnist_test_subset, batch_size=batch_size, shuffle=False)
    kmnist_test_loader = DataLoader(kmnist_test_subset, batch_size=batch_size, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test_subset, batch_size=batch_size, shuffle=False)
    
    mnist_batches = list(mnist_test_loader)
    kmnist_batches = list(kmnist_test_loader)
    fmnist_batches = list(fmnist_test_loader)
    
    closed_seq_stream = []
    for x, y in mnist_batches[:num_batches]:
        closed_seq_stream.append((x, y, 0))
    for x, y in kmnist_batches[:num_batches]:
        closed_seq_stream.append((x, y, 1))
        
    closed_alt_stream = []
    for idx in range(num_batches):
        closed_alt_stream.append((mnist_batches[idx][0], mnist_batches[idx][1], 0))
        closed_alt_stream.append((kmnist_batches[idx][0], kmnist_batches[idx][1], 1))
        
    # Open world stream evaluates on 640 samples per domain (1920 total samples)
    total_samples_ow = 640
    num_batches_ow = total_samples_ow // batch_size
    open_world_stream = []
    for x, y in mnist_batches[:num_batches_ow]:
        open_world_stream.append((x, y, 0))
    for x, y in kmnist_batches[:num_batches_ow]:
        open_world_stream.append((x, y, 1))
    for x, y in fmnist_batches[:num_batches_ow]:
        open_world_stream.append((x, y, 2))
        
    return {
        "Closed Sequential": closed_seq_stream,
        "Closed Alternating": closed_alt_stream,
        "Open-World": open_world_stream
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    # Standard batch size 64 for initial ablations
    streams = get_streams(64, mnist_test, kmnist_test, fmnist_test)
    
    # Load Experts
    mnist_expert = SimpleCNN().to(device)
    mnist_expert.load_state_dict(torch.load("checkpoints/mnist_expert.pt", map_location=device))
    mnist_expert.eval()
    kmnist_expert = SimpleCNN().to(device)
    kmnist_expert.load_state_dict(torch.load("checkpoints/kmnist_expert.pt", map_location=device))
    kmnist_expert.eval()
    
    experts = [mnist_expert, kmnist_expert]
    experts_sds = [expert.state_dict() for expert in experts]
    
    keys_to_merge = []
    for key, val in experts_sds[0].items():
        if "running_mean" not in key and "running_var" not in key and "num_batches_tracked" not in key:
            keys_to_merge.append(key)
            
    print("\n--- ABLATION 1: BN Fusion Strategy ---")
    bn_strategies = ["soft", "hard", "uniform"]
    for strategy in bn_strategies:
        set_seed(42)
        print(f"\nStrategy: {strategy.upper()}")
        for name, stream in streams.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       bn_fusion_strategy=strategy)
            print(f"  {name} Stream: {acc*100:.2f}%")
            
    print("\n--- ABLATION 2: Temporal Prior Regularization (Beta) ---")
    betas = [0.0, 0.1, 0.5, 2.0, 10.0]
    for b in betas:
        set_seed(42)
        print(f"\nBeta: {b}")
        for name, stream in streams.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       beta=b)
            print(f"  {name} Stream: {acc*100:.2f}%")
            
    print("\n--- ABLATION 3: Bayesian Temperature (Gamma) ---")
    gammas = [1.0, 5.0, 15.0, 50.0]
    for g in gammas:
        set_seed(42)
        print(f"\nGamma: {g}")
        for name, stream in streams.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       gamma=g)
            print(f"  {name} Stream: {acc*100:.2f}%")

    print("\n--- ABLATION 4: Number of Inner Adaptation Steps (M) ---")
    inner_steps_list = [0, 1, 2, 3, 5]
    for m in inner_steps_list:
        set_seed(42)
        print(f"\nInner Steps M: {m}")
        for name, stream in streams.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       inner_steps=m)
            print(f"  {name} Stream: {acc*100:.2f}%")

    print("\n--- ABLATION 5: Novelty Detection Threshold (Tau_N) ---")
    tau_ns = [0.5, 0.8, 1.0, 1.2, 1.5]
    for t in tau_ns:
        set_seed(42)
        print(f"\nTau_N: {t}")
        for name, stream in streams.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       tau_n=t)
            print(f"  {name} Stream: {acc*100:.2f}%")

    print("\n--- ABLATION 6: Impact of Test-Time Batch Size (B) ---")
    batch_sizes = [16, 32, 64, 128]
    for bs in batch_sizes:
        set_seed(42)
        print(f"\nBatch Size B: {bs}")
        # Build stream dynamically for this batch size
        stream_bs = get_streams(bs, mnist_test, kmnist_test, fmnist_test)
        for name, stream in stream_bs.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       tau_n=1.2)
            print(f"  {name} Stream: {acc*100:.2f}%")

    print("\n--- ABLATION 7: Origin of Global BN Fusion Weights (Learnable vs. Average) ---")
    global_modes = ["learnable", "average"]
    for mode in global_modes:
        set_seed(42)
        print(f"\nGlobal BN Mode: {mode.upper()}")
        for name, stream in streams.items():
            acc = evaluate_custom_ours(stream, experts, experts_sds, keys_to_merge, device,
                                       global_mode=mode)
            print(f"  {name} Stream: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
