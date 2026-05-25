import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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

# Define ResNet-18 modified for CIFAR-10
class CIFARResNet18(nn.Module):
    def __init__(self):
        super(CIFARResNet18, self).__init__()
        self.model = models.resnet18(num_classes=10)
        # Modify the first conv layer to support 32x32 images instead of 224x224
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity() # Remove maxpool as it downsizes the 32x32 image too much

    def forward(self, x):
        return self.model(x)

# Help manage expert BN buffers
def get_bn_modules(model):
    bn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_modules[name] = module
    return bn_modules

# Soft Batch Normalization buffer fusion (Mixture of Gaussians matching)
def fuse_bn_buffers(merged_model, experts, posterior_weights):
    with torch.no_grad():
        merged_bn = get_bn_modules(merged_model)
        expert_bns = [get_bn_modules(expert) for expert in experts]
        
        for name, m_bn in merged_bn.items():
            means = [e_bn[name].running_mean for e_bn in expert_bns]
            vars = [e_bn[name].running_var for e_bn in expert_bns]
            
            # Weighted average of means
            fused_mean = sum(posterior_weights[k] * means[k] for k in range(len(experts)))
            
            # Weighted average of variances plus cross-term covariance
            fused_var = sum(posterior_weights[k] * (vars[k] + (means[k] - fused_mean)**2) for k in range(len(experts)))
            
            m_bn.running_mean.copy_(fused_mean)
            m_bn.running_var.copy_(fused_var)

# Hard Batch Normalization buffer merging (EBER style)
def hard_merge_bn_buffers(merged_model, experts, active_expert_idx):
    with torch.no_grad():
        merged_bn = get_bn_modules(merged_model)
        active_bn = get_bn_modules(experts[active_expert_idx])
        for name, m_bn in merged_bn.items():
            m_bn.running_mean.copy_(active_bn[name].running_mean)
            m_bn.running_var.copy_(active_bn[name].running_var)

# Differentiable Weight Merging State Dict Generator
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

# Compute Test-Time Fisher Information on a batch
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

def train_expert(name, train_loader, device, epochs=5):
    print(f"Training ResNet-18 Expert on {name}...")
    model = CIFARResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} Accuracy: {100.*correct/total:.2f}%")
                
    model.eval()
    print(f"Finished Training ResNet-18 Expert on {name}.")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Transforms
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    transform_rotated = transforms.Compose([
        transforms.RandomRotation((180, 180)),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_grayscale = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])
    
    os.makedirs("data", exist_ok=True)
    
    # Download datasets with different transforms
    cifar_train_clean = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_clean)
    cifar_train_rotated = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_rotated)
    
    cifar_test_clean = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_clean)
    cifar_test_rotated = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_rotated)
    cifar_test_grayscale = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_grayscale)
    
    clean_loader = DataLoader(cifar_train_clean, batch_size=256, shuffle=True)
    rotated_loader = DataLoader(cifar_train_rotated, batch_size=256, shuffle=True)
    
    # 2. Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    if os.path.exists("checkpoints/cifar_clean_resnet18.pt") and os.path.exists("checkpoints/cifar_rotated_resnet18.pt"):
        print("Loading existing CIFAR ResNet-18 experts...")
        clean_expert = CIFARResNet18().to(device)
        clean_expert.load_state_dict(torch.load("checkpoints/cifar_clean_resnet18.pt", map_location=device))
        rotated_expert = CIFARResNet18().to(device)
        rotated_expert.load_state_dict(torch.load("checkpoints/cifar_rotated_resnet18.pt", map_location=device))
    else:
        clean_expert = train_expert("Clean CIFAR-10", clean_loader, device, epochs=5)
        rotated_expert = train_expert("Rotated CIFAR-10", rotated_loader, device, epochs=5)
        
        torch.save(clean_expert.state_dict(), "checkpoints/cifar_clean_resnet18.pt")
        torch.save(rotated_expert.state_dict(), "checkpoints/cifar_rotated_resnet18.pt")
        
    clean_expert.eval()
    rotated_expert.eval()
    
    experts = [clean_expert, rotated_expert]
    experts_sds = [expert.state_dict() for expert in experts]
    
    keys_to_merge = []
    for key, val in experts_sds[0].items():
        if "running_mean" not in key and "running_var" not in key and "num_batches_tracked" not in key:
            keys_to_merge.append(key)
            
    # 3. Create Test Streams
    # For speed and high quality, we construct sequential, alternating, and open-world streams
    batch_size = 64
    num_batches_per_domain = 15
    
    clean_test_subset = Subset(cifar_test_clean, list(range(num_batches_per_domain * batch_size)))
    rotated_test_subset = Subset(cifar_test_rotated, list(range(num_batches_per_domain * batch_size)))
    grayscale_test_subset = Subset(cifar_test_grayscale, list(range(num_batches_per_domain * batch_size)))
    
    clean_test_loader = DataLoader(clean_test_subset, batch_size=batch_size, shuffle=False)
    rotated_test_loader = DataLoader(rotated_test_subset, batch_size=batch_size, shuffle=False)
    grayscale_test_loader = DataLoader(grayscale_test_subset, batch_size=batch_size, shuffle=False)
    
    clean_batches = list(clean_test_loader)
    rotated_batches = list(rotated_test_loader)
    grayscale_batches = list(grayscale_test_loader)
    
    # A) Closed Sequential: Clean CIFAR-10, then Rotated CIFAR-10
    closed_seq_stream = []
    for x, y in clean_batches[:num_batches_per_domain]:
        closed_seq_stream.append((x, y, 0))
    for x, y in rotated_batches[:num_batches_per_domain]:
        closed_seq_stream.append((x, y, 1))
        
    # B) Closed Alternating
    closed_alt_stream = []
    for idx in range(num_batches_per_domain):
        closed_alt_stream.append((clean_batches[idx][0], clean_batches[idx][1], 0))
        closed_alt_stream.append((rotated_batches[idx][0], rotated_batches[idx][1], 1))
        
    # C) Open-World Stream: Clean, then Rotated, then Grayscale (Novel!)
    open_world_stream = []
    for x, y in clean_batches[:10]:
        open_world_stream.append((x, y, 0))
    for x, y in rotated_batches[:10]:
        open_world_stream.append((x, y, 1))
    for x, y in grayscale_batches[:10]:
        open_world_stream.append((x, y, 2))
        
    streams = {
        "Closed Sequential": closed_seq_stream,
        "Closed Alternating": closed_alt_stream,
        "Open-World": open_world_stream
    }
    
    methods = ["Static Merging", "AdaMerging", "DR-Fisher", "DF-Bayes-TTMM (Ours)"]
    stream_accuracies = {stream_name: {m: 0.0 for m in methods} for stream_name in streams}
    
    for stream_name, stream_batches in streams.items():
        print(f"\n--- Stream: {stream_name} ---")
        K = len(experts)
        
        for method in methods:
            set_seed(42)
            print(f"Running Method: {method}")
            lambdas = {key: torch.zeros(K, device=device, requires_grad=True) for key in keys_to_merge}
            lambdas['global'] = torch.zeros(K, device=device, requires_grad=True)
            
            prev_lambdas_val = {key: torch.zeros(K, device=device) for key in keys_to_merge}
            prev_lambdas_val['global'] = torch.zeros(K, device=device)
            
            merged_model = CIFARResNet18().to(device)
            
            correct_total = 0
            samples_total = 0
            
            for step, (x, y, domain_lbl) in enumerate(stream_batches):
                x, y = x.to(device), y.to(device)
                
                # Evaluation and adaptation step
                if method == "Static Merging":
                    # Static uniform merging
                    with torch.no_grad():
                        lambdas_static = {key: torch.zeros(K, device=device) for key in keys_to_merge}
                        lambdas_static['global'] = torch.zeros(K, device=device)
                        merged_model.load_state_dict(get_merged_state_dict(experts_sds, lambdas_static, keys_to_merge), strict=False)
                        # Uniform BN stats fusion
                        fuse_bn_buffers(merged_model, experts, [0.5, 0.5])
                        
                        merged_model.eval()
                        outputs = merged_model(x)
                        correct_total += outputs.argmax(dim=-1).eq(y).sum().item()
                        samples_total += y.size(0)
                        
                elif method == "AdaMerging":
                    # Gradient-based entropy minimization without BN stats update (activation mismatch!)
                    # For fairness, we run 1 inner step
                    optimizer = optim.Adam([lambdas[key] for key in keys_to_merge] + [lambdas['global']], lr=1e-2)
                    optimizer.zero_grad()
                    
                    # Merge weights and evaluate
                    merged_model.load_state_dict(get_merged_state_dict(experts_sds, lambdas, keys_to_merge), strict=False)
                    # Uniform BN statistics
                    fuse_bn_buffers(merged_model, experts, [0.5, 0.5])
                    
                    merged_model.eval()
                    outputs = merged_model(x)
                    probs = F.softmax(outputs, dim=-1)
                    loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate accuracy
                    with torch.no_grad():
                        outputs_eval = merged_model(x)
                        correct_total += outputs_eval.argmax(dim=-1).eq(y).sum().item()
                        samples_total += y.size(0)
                        
                elif method == "DR-Fisher":
                    # Hard-routing EBER style + test-time fisher preconditioning (oracle on clean, hard routing)
                    with torch.no_grad():
                        # Determine active expert by predictive entropy
                        entropies = []
                        for expert in experts:
                            expert_logits = expert(x)
                            expert_probs = F.softmax(expert_logits, dim=-1)
                            expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10), dim=-1).mean().item()
                            entropies.append(expert_entropy)
                        
                        # Hard routing active expert idx
                        active_idx = np.argmin(entropies)
                        
                    # Initialize coefficients close to active expert
                    with torch.no_grad():
                        for key in keys_to_merge:
                            lambdas[key].fill_(-10.0)
                            lambdas[key][active_idx] = 10.0
                        lambdas['global'].fill_(-10.0)
                        lambdas['global'][active_idx] = 10.0
                        
                    # Perform 1 inner step under Fisher preconditioning
                    # Compute test-time fisher
                    temp_model = CIFARResNet18().to(device)
                    temp_model.load_state_dict(get_merged_state_dict(experts_sds, lambdas, keys_to_merge), strict=False)
                    hard_merge_bn_buffers(temp_model, experts, active_idx)
                    fisher = compute_test_time_fisher(temp_model, x, device)
                    layer_sens = get_layer_fisher_sensitivity(fisher, keys_to_merge)
                    
                    # Step
                    optimizer = optim.Adam([lambdas[key] for key in keys_to_merge] + [lambdas['global']], lr=1e-2)
                    optimizer.zero_grad()
                    merged_model.load_state_dict(get_merged_state_dict(experts_sds, lambdas, keys_to_merge), strict=False)
                    hard_merge_bn_buffers(merged_model, experts, active_idx)
                    
                    merged_model.eval()
                    outputs = merged_model(x)
                    probs = F.softmax(outputs, dim=-1)
                    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                    
                    # Regularization
                    reg_loss = 0.0
                    for key in keys_to_merge:
                        # Scaled by test-time fisher layer sensitivity
                        weight_sens = layer_sens[key]
                        reg_loss += 0.5 * 0.5 * weight_sens * torch.sum((lambdas[key] - prev_lambdas_val[key])**2)
                    
                    total_loss = entropy_loss + reg_loss
                    total_loss.backward()
                    optimizer.step()
                    
                    # Save for next
                    with torch.no_grad():
                        for key in keys_to_merge:
                            prev_lambdas_val[key].copy_(lambdas[key])
                        prev_lambdas_val['global'].copy_(lambdas['global'])
                        
                        outputs_eval = merged_model(x)
                        correct_total += outputs_eval.argmax(dim=-1).eq(y).sum().item()
                        samples_total += y.size(0)
                        
                elif method == "DF-Bayes-TTMM (Ours)":
                    # Fully data-free Bayesian Test-Time Model Merging
                    with torch.no_grad():
                        entropies = []
                        for expert in experts:
                            expert_logits = expert(x)
                            expert_probs = F.softmax(expert_logits, dim=-1)
                            expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10), dim=-1).mean().item()
                            entropies.append(expert_entropy)
                        
                        avg_expert_entropy = np.mean(entropies)
                        
                    # Check for novelty
                    is_novel = avg_expert_entropy > 1.1  # CIFAR entropy-based novelty threshold
                    
                    if is_novel:
                        # Novel domain: use uniform prior fallback
                        posterior_weights = [0.5, 0.5]
                    else:
                        # Known domain: Soft Bayesian posterior
                        gamma = 15.0
                        exp_neg_ent = np.exp(-gamma * np.array(entropies))
                        posterior_weights = exp_neg_ent / np.sum(exp_neg_ent)
                        
                    # Initialize lambdas in logit space
                    with torch.no_grad():
                        for key in keys_to_merge:
                            for k in range(K):
                                lambdas[key][k] = np.log(posterior_weights[k] + 1e-5)
                        for k in range(K):
                            lambdas['global'][k] = np.log(posterior_weights[k] + 1e-5)
                            
                    # We perform 1 inner step
                    # Compute test-time fisher on temp
                    temp_model = CIFARResNet18().to(device)
                    temp_model.load_state_dict(get_merged_state_dict(experts_sds, lambdas, keys_to_merge), strict=False)
                    fuse_bn_buffers(temp_model, experts, posterior_weights)
                    fisher = compute_test_time_fisher(temp_model, x, device)
                    layer_sens = get_layer_fisher_sensitivity(fisher, keys_to_merge)
                    
                    # Gradient step
                    optimizer = optim.Adam([lambdas[key] for key in keys_to_merge] + [lambdas['global']], lr=1e-2)
                    optimizer.zero_grad()
                    
                    merged_model.load_state_dict(get_merged_state_dict(experts_sds, lambdas, keys_to_merge), strict=False)
                    
                    # Soft BN stats fusion based on current global coefficients
                    with torch.no_grad():
                        curr_global_weights = torch.softmax(lambdas['global'], dim=0).cpu().numpy()
                    fuse_bn_buffers(merged_model, experts, curr_global_weights)
                    
                    merged_model.eval()
                    outputs = merged_model(x)
                    probs = F.softmax(outputs, dim=-1)
                    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                    
                    # Temporal prior regularization
                    reg_loss = 0.0
                    for key in keys_to_merge:
                        weight_sens = layer_sens[key]
                        reg_loss += 0.5 * 0.5 * weight_sens * torch.sum((lambdas[key] - prev_lambdas_val[key])**2)
                        
                    total_loss = entropy_loss + reg_loss
                    total_loss.backward()
                    optimizer.step()
                    
                    # Update previous for temporal prior tracking
                    with torch.no_grad():
                        for key in keys_to_merge:
                            prev_lambdas_val[key].copy_(lambdas[key])
                        prev_lambdas_val['global'].copy_(lambdas['global'])
                        
                        outputs_eval = merged_model(x)
                        correct_total += outputs_eval.argmax(dim=-1).eq(y).sum().item()
                        samples_total += y.size(0)
            
            final_acc = 100. * correct_total / samples_total
            print(f"Finished method {method} on {stream_name}. Accuracy: {final_acc:.2f}%")
            stream_accuracies[stream_name][method] = final_acc
            
    print("\n=== FINAL RESULTS SUMMARY FOR CIFAR-10 RESNET-18 ===")
    for stream_name in streams:
        print(f"\nStream: {stream_name}")
        for m in methods:
            print(f"  {m}: {stream_accuracies[stream_name][m]:.2f}%")

if __name__ == "__main__":
    main()
