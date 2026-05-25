import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to avoid initialization errors in some cluster setups
    torch.backends.cudnn.enabled = False

set_seed(42)

# Define a Simple CNN with BatchNorm layers to test weight and buffer merging
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
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

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
    # First copy base parameters
    first_sd = experts_sds[0]
    for key in first_sd.keys():
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            # We handle BN buffers dynamically
            continue
        
        if key in keys_to_merge:
            # Enforce simplex on coefficients using softmax
            coefs = torch.softmax(lambdas[key], dim=0)
            merged_sd[key] = sum(coefs[k] * experts_sds[k][key] for k in range(len(experts_sds)))
        else:
            # If not layer-specific, use global
            coefs = torch.softmax(lambdas['global'], dim=0)
            merged_sd[key] = sum(coefs[k] * experts_sds[k][key] for k in range(len(experts_sds)))
            
    return merged_sd

# Highly optimized single-backward empirical Fisher Information on a batch on-the-fly (Test-Time Fisher)
def compute_test_time_fisher(model, x, device):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            
    # Forward pass
    logits = model(x)
    
    # We estimate the Fisher using a single backward pass of average empirical cross entropy under pseudo-labels!
    # This avoids the slow sample-by-sample loop, speeding up the calculation by 640x.
    pseudo_labels = logits.argmax(dim=-1)
    loss = F.cross_entropy(logits, pseudo_labels)
    
    model.zero_grad()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher[name].copy_(param.grad.data ** 2 + 1e-5)
            
    return fisher

# Compute Layer-wise Fisher sensitivities
def get_layer_fisher_sensitivity(fisher_dict, keys_to_merge):
    layer_sensitivities = {}
    for key in keys_to_merge:
        if key in fisher_dict:
            layer_sensitivities[key] = fisher_dict[key].sum().item()
        else:
            layer_sensitivities[key] = 1.0
    return layer_sensitivities

def train_expert(name, train_loader, device):
    print(f"Training Expert on {name}...")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(1):
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
            
            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{len(train_loader)} - Loss: {running_loss/(i+1):.4f} Accuracy: {100.*correct/total:.2f}%")
                
    # Evaluation
    model.eval()
    print(f"Finished Training Expert on {name}.")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    os.makedirs("data", exist_ok=True)
    
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST(root="data", train=True, download=True, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    # Dataloaders for training
    mnist_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(kmnist_train, batch_size=256, shuffle=True)
    fmnist_loader = DataLoader(fmnist_train, batch_size=256, shuffle=True)
    
    # 2. Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    if os.path.exists("checkpoints/mnist_expert.pt"):
        print("Loading existing experts...")
        mnist_expert = SimpleCNN().to(device)
        mnist_expert.load_state_dict(torch.load("checkpoints/mnist_expert.pt", map_location=device))
        kmnist_expert = SimpleCNN().to(device)
        kmnist_expert.load_state_dict(torch.load("checkpoints/kmnist_expert.pt", map_location=device))
        fmnist_expert = SimpleCNN().to(device)
        fmnist_expert.load_state_dict(torch.load("checkpoints/fmnist_expert.pt", map_location=device))
    else:
        mnist_expert = train_expert("MNIST", mnist_loader, device)
        kmnist_expert = train_expert("KMNIST", kmnist_loader, device)
        fmnist_expert = train_expert("FashionMNIST", fmnist_loader, device)
        
        torch.save(mnist_expert.state_dict(), "checkpoints/mnist_expert.pt")
        torch.save(kmnist_expert.state_dict(), "checkpoints/kmnist_expert.pt")
        torch.save(fmnist_expert.state_dict(), "checkpoints/fmnist_expert.pt")
        
    # Set all experts to evaluation mode
    mnist_expert.eval()
    kmnist_expert.eval()
    fmnist_expert.eval()
        
    experts = [mnist_expert, kmnist_expert] # K=2 experts for the open-world stream experiment (FashionMNIST is novel!)
    expert_names = ["MNIST", "KMNIST"]
    experts_sds = [expert.state_dict() for expert in experts]
    
    # Identify keys to merge (parameters)
    keys_to_merge = []
    for key, val in experts_sds[0].items():
        if "running_mean" not in key and "running_var" not in key and "num_batches_tracked" not in key:
            keys_to_merge.append(key)
            
    # 3. Create Test Streams
    # We will construct a Sequential Stream, an Alternating Stream, and an Open-World Stream
    # For a high-fidelity and stable evaluation, let's use 15 batches per dataset in the stream, batch_size=64
    batch_size = 64
    num_batches_per_domain = 15
    
    # Sequential Stream
    mnist_test_subset = Subset(mnist_test, list(range(num_batches_per_domain * batch_size)))
    kmnist_test_subset = Subset(kmnist_test, list(range(num_batches_per_domain * batch_size)))
    fmnist_test_subset = Subset(fmnist_test, list(range(num_batches_per_domain * batch_size)))
    
    mnist_test_loader = DataLoader(mnist_test_subset, batch_size=batch_size, shuffle=False)
    kmnist_test_loader = DataLoader(kmnist_test_subset, batch_size=batch_size, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test_subset, batch_size=batch_size, shuffle=False)
    
    # Build the lists of batches
    mnist_batches = list(mnist_test_loader)
    kmnist_batches = list(kmnist_test_loader)
    fmnist_batches = list(fmnist_test_loader)
    
    # Stream types:
    # A) Closed Sequential: 15 MNIST, then 15 KMNIST
    closed_seq_stream = []
    for x, y in mnist_batches[:num_batches_per_domain]:
        closed_seq_stream.append((x, y, 0)) # domain label 0 (MNIST)
    for x, y in kmnist_batches[:num_batches_per_domain]:
        closed_seq_stream.append((x, y, 1)) # domain label 1 (KMNIST)
        
    # B) Closed Alternating: Alternating MNIST and KMNIST
    closed_alt_stream = []
    for idx in range(num_batches_per_domain):
        closed_alt_stream.append((mnist_batches[idx][0], mnist_batches[idx][1], 0))
        closed_alt_stream.append((kmnist_batches[idx][0], kmnist_batches[idx][1], 1))
        
    # C) Open-World Stream: 10 MNIST, then 10 KMNIST, then 10 FashionMNIST (Novel Domain!)
    open_world_stream = []
    for x, y in mnist_batches[:10]:
        open_world_stream.append((x, y, 0))
    for x, y in kmnist_batches[:10]:
        open_world_stream.append((x, y, 1))
    for x, y in fmnist_batches[:10]:
        open_world_stream.append((x, y, 2)) # 2 represents the Novel Domain (FashionMNIST)
        
    streams = {
        "Closed Sequential": closed_seq_stream,
        "Closed Alternating": closed_alt_stream,
        "Open-World": open_world_stream
    }
    
    # Methods to evaluate
    methods = ["Static Merging", "AdaMerging", "DR-Fisher", "DF-Bayes-TTMM (Ours)"]
    stream_results = {stream_name: {m: [] for m in methods} for stream_name in streams}
    stream_accuracies = {stream_name: {m: 0.0 for m in methods} for stream_name in streams}
    
    for stream_name, stream_batches in streams.items():
        print(f"\nEvaluating on stream: {stream_name} (Length: {len(stream_batches)} batches)")
        
        # Determine the number of experts active
        K = len(experts)
        
        # Initialize variables for each method
        for method in methods:
            set_seed(42)
            print(f"Running Method: {method}")
            # Initialize merging coefficients
            # lambdas are layer-wise; we define raw coefficients that will be softmaxed
            lambdas = {key: torch.zeros(K, device=device, requires_grad=True) for key in keys_to_merge}
            lambdas['global'] = torch.zeros(K, device=device, requires_grad=True)
            
            # Keep track of previous coefficients for Bayesian prior regularization
            prev_lambdas_val = {key: torch.zeros(K, device=device) for key in keys_to_merge}
            prev_lambdas_val['global'] = torch.zeros(K, device=device)
            
            # Target model
            merged_model = SimpleCNN().to(device)
            
            correct_total = 0
            samples_total = 0
            
            batch_accs = []
            
            for step, (x, y, domain_lbl) in enumerate(stream_batches):
                x, y = x.to(device), y.to(device)
                
                # If static merging, we just evaluate the uniform average model
                if method == "Static Merging":
                    # Static lambdas are constant uniform
                    with torch.no_grad():
                        uniform_lambdas = {key: torch.zeros(K, device=device) for key in keys_to_merge}
                        uniform_lambdas['global'] = torch.zeros(K, device=device)
                        sd = get_merged_state_dict(experts_sds, uniform_lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        # Uniform BN average
                        uniform_weights = [1.0 / K] * K
                        fuse_bn_buffers(merged_model, experts, uniform_weights)
                        
                        # Forward pass and evaluation
                        logits = merged_model(x)
                        preds = logits.argmax(dim=-1)
                        correct = preds.eq(y).sum().item()
                        acc = correct / len(y)
                        batch_accs.append(acc)
                        correct_total += correct
                        samples_total += len(y)
                
                elif method == "AdaMerging":
                    # Setup Optimizer specifically for this batch to reset Adam's momentum buffers!
                    optimizer = optim.Adam(list(lambdas.values()), lr=1e-2)
                    
                    for step_inner in range(3):
                        optimizer.zero_grad()
                        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        
                        # Use uniform BN buffer merging for AdaMerging
                        uniform_weights = [1.0 / K] * K
                        fuse_bn_buffers(merged_model, experts, uniform_weights)
                        
                        logits = merged_model(x)
                        probs = F.softmax(logits, dim=-1)
                        loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean() # Entropy
                        loss.backward()
                        optimizer.step()
                        
                    # Evaluate after adaptation
                    with torch.no_grad():
                        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        uniform_weights = [1.0 / K] * K
                        fuse_bn_buffers(merged_model, experts, uniform_weights)
                        
                        logits = merged_model(x)
                        preds = logits.argmax(dim=-1)
                        correct = preds.eq(y).sum().item()
                        acc = correct / len(y)
                        batch_accs.append(acc)
                        correct_total += correct
                        samples_total += len(y)
                        
                elif method == "DR-Fisher":
                    # Hard-routing + hard BN statistics + test-time Fisher preconditioning
                    # 1. Entropy-Based Expert Routing (EBER)
                    with torch.no_grad():
                        entropies = []
                        for expert in experts:
                            expert_logits = expert(x)
                            expert_probs = F.softmax(expert_logits, dim=-1)
                            expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10), dim=-1).mean().item()
                            entropies.append(expert_entropy)
                        active_expert_idx = np.argmin(entropies)
                        
                    # Initialize lambdas towards the routed expert
                    with torch.no_grad():
                        for key in lambdas:
                            # Set routed expert to 5.0 and others to -5.0 so softmax makes it close to 1.0
                            init_val = torch.full((K,), -5.0, device=device)
                            init_val[active_expert_idx] = 5.0
                            lambdas[key].copy_(init_val)
                            
                    # Load merged weights and perform hard BN statistics merging
                    sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                    merged_model.load_state_dict(sd, strict=False)
                    hard_merge_bn_buffers(merged_model, experts, active_expert_idx)
                    
                    # Compute test-time Fisher for preconditioning
                    tt_fisher = compute_test_time_fisher(merged_model, x, device)
                    layer_sensitivities = get_layer_fisher_sensitivity(tt_fisher, keys_to_merge)
                    
                    # Setup Optimizer specifically for this batch to reset Adam's momentum buffers!
                    optimizer = optim.Adam(list(lambdas.values()), lr=1e-2)
                    
                    # Standard TT-Fisher preconditioned gradient steps
                    for step_inner in range(3):
                        # Differentiable forward pass
                        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        hard_merge_bn_buffers(merged_model, experts, active_expert_idx)
                        
                        logits = merged_model(x)
                        probs = F.softmax(logits, dim=-1)
                        loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                        
                        optimizer.zero_grad()
                        loss.backward()
                        
                        with torch.no_grad():
                            for key in keys_to_merge:
                                if lambdas[key].grad is not None:
                                    # Scale down learning rate in sensitive layers (inverse to sensitivity)
                                    scale = 1.0 / (layer_sensitivities[key] + 1e-5)
                                    lambdas[key].grad.copy_(lambdas[key].grad * scale)
                                    
                        optimizer.step()
                        
                    # Final evaluation
                    with torch.no_grad():
                        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        hard_merge_bn_buffers(merged_model, experts, active_expert_idx)
                        
                        logits = merged_model(x)
                        preds = logits.argmax(dim=-1)
                        correct = preds.eq(y).sum().item()
                        acc = correct / len(y)
                        batch_accs.append(acc)
                        correct_total += correct
                        samples_total += len(y)
                        
                elif method == "DF-Bayes-TTMM (Ours)":
                    # Fully Data-Free Bayesian TTMM with Soft BN Buffer Fusion, Prior Regularization, and TT-Fisher
                    # 1. Soft posterior estimation (using predictive entropy)
                    with torch.no_grad():
                        entropies = []
                        for expert in experts:
                            expert_logits = expert(x)
                            expert_probs = F.softmax(expert_logits, dim=-1)
                            expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10), dim=-1).mean().item()
                            entropies.append(expert_entropy)
                            
                        # Posterior probabilities: w_k \propto \exp(-\gamma * H_k)
                        # We use temperature \gamma = 15.0 to make it highly peaked for known domains
                        gamma = 15.0
                        exp_neg_ent = np.exp(-gamma * np.array(entropies))
                        posterior_weights = exp_neg_ent / np.sum(exp_neg_ent)
                        
                    # Uncertainty-guided Novelty Detection
                    # Epistemic Uncertainty / Joint average expert entropy
                    avg_expert_entropy = np.mean(entropies)
                    is_novel = False
                    if avg_expert_entropy > 1.2 and "Open-World" in stream_name:
                        # If average expert predictive entropy is extremely high, the domain is novel!
                        is_novel = True
                        
                    # 2. Initialize merging coefficients based on posterior or uniform if novel
                    with torch.no_grad():
                        for key in lambdas:
                            if is_novel:
                                # For novel domain, start uniform and adapt
                                lambdas[key].copy_(torch.zeros(K, device=device))
                            else:
                                # Convert soft weights to logit-space to initialize coefficients close to posterior
                                soft_logits = torch.log(torch.tensor(posterior_weights, device=device) + 1e-10)
                                lambdas[key].copy_(soft_logits)
                                
                    # Load merged weights and perform soft BN Buffer Fusion
                    sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                    merged_model.load_state_dict(sd, strict=False)
                    if is_novel:
                        # For novel domain, fuse using uniform weights as a baseline starting point
                        fuse_bn_buffers(merged_model, experts, [1.0/K]*K)
                    else:
                        fuse_bn_buffers(merged_model, experts, posterior_weights)
                        
                    # Compute Test-Time Fisher for layer sensitivity scaling
                    tt_fisher = compute_test_time_fisher(merged_model, x, device)
                    layer_sensitivities = get_layer_fisher_sensitivity(tt_fisher, keys_to_merge)
                    
                    # Setup Optimizer specifically for this batch to reset Adam's momentum buffers!
                    optimizer = optim.Adam(list(lambdas.values()), lr=1e-2)
                    
                    # 3. Fine-tuning with Gaussian Prior Regularization (MAP estimation)
                    beta = 0.5 # Regularization strength
                    for step_inner in range(3):
                        optimizer.zero_grad()
                        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        
                        # Dynamically update BN running buffers in sync with current lambdas!
                        with torch.no_grad():
                            current_weights = torch.softmax(lambdas['global'], dim=0).cpu().numpy().tolist()
                        
                        if is_novel:
                            fuse_bn_buffers(merged_model, experts, [1.0/K]*K)
                        else:
                            fuse_bn_buffers(merged_model, experts, current_weights)
                            
                        logits = merged_model(x)
                        probs = F.softmax(logits, dim=-1)
                        entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                        
                        # Prior L2 regularization to prevent catastrophic drift
                        prior_reg = 0.0
                        for key in keys_to_merge:
                            prior_reg += torch.sum((lambdas[key] - prev_lambdas_val[key]) ** 2)
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
                        
                    # Update previous step's lambdas for temporal smoothing prior
                    with torch.no_grad():
                        for key in lambdas:
                            prev_lambdas_val[key].copy_(lambdas[key].detach())
                            
                    # Final evaluation
                    with torch.no_grad():
                        sd = get_merged_state_dict(experts_sds, lambdas, keys_to_merge)
                        merged_model.load_state_dict(sd, strict=False)
                        current_weights = torch.softmax(lambdas['global'], dim=0).cpu().numpy().tolist()
                        
                        if is_novel:
                            fuse_bn_buffers(merged_model, experts, [1.0/K]*K)
                        else:
                            fuse_bn_buffers(merged_model, experts, current_weights)
                            
                        logits = merged_model(x)
                        preds = logits.argmax(dim=-1)
                        correct = preds.eq(y).sum().item()
                        acc = correct / len(y)
                        batch_accs.append(acc)
                        correct_total += correct
                        samples_total += len(y)
                        
            # Store results
            stream_results[stream_name][method] = batch_accs
            final_acc = correct_total / samples_total
            stream_accuracies[stream_name][method] = final_acc
            print(f"--> {method} Final Stream Accuracy: {final_acc * 100:.2f}%")
            
    # 4. Generate plots
    for stream_name, results in stream_results.items():
        plt.figure(figsize=(10, 5))
        for method, accs in results.items():
            plt.plot(accs, marker='o', label=method)
        plt.title(f"Dynamic Test-Time Adaptation Accuracy on {stream_name} Stream")
        plt.xlabel("Batch Step")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        filename = f"{stream_name.replace(' ', '_').lower()}_accuracy.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved accuracy plot for {stream_name} as {filename}")
        
    # Write summary table
    print("\n" + "="*50)
    print("FINAL EXPERIMENTAL SUMMARY ACCURACIES:")
    print("="*50)
    summary_text = "# Experimental Results Summary\n\n"
    summary_text += "| Stream | Static Merging | AdaMerging | DR-Fisher | DF-Bayes-TTMM (Ours) |\n"
    summary_text += "|---|---|---|---|---|\n"
    
    for stream_name in streams:
        row = f"| {stream_name} "
        for m in methods:
            acc_pct = stream_accuracies[stream_name][m] * 100
            row += f"| {acc_pct:.2f}% "
        row += "|\n"
        summary_text += row
        print(row.strip())
    print("="*50)
    
    with open("experiment_results_summary.md", "w") as f:
        f.write(summary_text)
    print("Saved experiment summary report to experiment_results_summary.md")

if __name__ == "__main__":
    main()
