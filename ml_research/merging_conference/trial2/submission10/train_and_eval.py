import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision.models import resnet18, ResNet18_Weights

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Disable cuDNN to prevent cluster-specific cuDNN initialization errors
        torch.backends.cudnn.enabled = False

class MultiTaskModel(nn.Module):
    def __init__(self, num_tasks=3, num_classes=10):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # We extract encoder layers up to avgpool
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # Multi-task linear heads
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_tasks)
        ])
        
    def forward(self, x, task_id):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        return self.heads[task_id](features)

def get_datasets(data_dir="./data"):
    # ResNet18 expects 3 channels. MNIST/FashionMNIST/KMNIST are 1 channel.
    # We repeat the channel 3 times to fit ResNet18 expected shape.
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_mnist = MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_mnist = MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_fmnist = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_fmnist = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_kmnist = KMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_kmnist = KMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    return {
        "train": [train_mnist, train_fmnist, train_kmnist],
        "test": [test_mnist, test_fmnist, test_kmnist]
    }

def train_expert(model, train_loader, task_id, epochs, device, lr=1e-4):
    # Freezes other heads, only trains encoder and task-specific head
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.heads[task_id].parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, task_id)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return epoch_acc

def evaluate_expert(model, test_loader, task_id, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_id)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total

def get_merged_state_dict(model, encoder_pre_state, encoder_pre_params, task_vectors, lambdas, adapted_heads_params, num_tasks):
    S = {}
    
    # Merge encoder parameters and buffers
    for name, pre_val in encoder_pre_state.items():
        # Reconstruct encoder.name
        if pre_val.is_floating_point() and "num_batches_tracked" not in name:
            is_param = name in encoder_pre_params
            val = pre_val.clone()
            for k in range(num_tasks):
                # Use coeff with gradients for parameters, and detached for buffers
                coeff = lambdas[k] if is_param else lambdas[k].detach()
                val = val + coeff * task_vectors[k][name]
        else:
            # Buffer (like num_batches_tracked or integer)
            val = pre_val.clone()
        S["encoder." + name] = val
        
    # Set heads
    for k in range(num_tasks):
        S[f"heads.{k}.weight"] = adapted_heads_params[f"heads.{k}.weight"]
        S[f"heads.{k}.bias"] = adapted_heads_params[f"heads.{k}.bias"]
        
    return S

def get_expert_state_dict(encoder_pre_state, encoder_k_state, expert_heads_weights, k, num_tasks):
    S = {}
    for name in encoder_pre_state.keys():
        S["encoder." + name] = encoder_k_state[name]
    for task_id in range(num_tasks):
        S[f"heads.{task_id}.weight"] = expert_heads_weights[task_id]["weight"]
        S[f"heads.{task_id}.bias"] = expert_heads_weights[task_id]["bias"]
    return S

# Image corruptions
def apply_corruption(x, corruption_type, device):
    if corruption_type == "none":
        return x
    elif corruption_type == "gaussian_noise":
        return x + torch.randn_like(x, device=device) * 0.4
    elif corruption_type == "gaussian_blur":
        # Approximate Gaussian blur with 5x5 average pool as a robust pytorch native alternative
        return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    elif corruption_type == "contrast_reduction":
        return x * 0.25
    elif corruption_type == "image_rotation":
        # Rotate by 90 degrees as a simple rotation test
        return torch.rot90(x, k=1, dims=[2, 3])
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")

def compute_clean_fisher_estimates(
    model, 
    encoder_pre_state, 
    encoder_pre_params,
    encoder_experts_states, 
    expert_heads_weights, 
    task_vectors,
    train_loaders, 
    device, 
    num_tasks=3, 
    num_samples=512
):
    print(f"Computing clean Fisher Information estimates on training datasets using {num_samples} samples per task...")
    fisher_estimates = {}
    
    # Initialize head parameter Fisher estimates
    for k in range(num_tasks):
        fisher_estimates[f"heads.{k}.weight"] = torch.zeros_like(expert_heads_weights[k]["weight"], device=device)
        fisher_estimates[f"heads.{k}.bias"] = torch.zeros_like(expert_heads_weights[k]["bias"], device=device)
    # Initialize lambdas Fisher estimate
    fisher_estimates["lambdas"] = torch.zeros(num_tasks, device=device)
    
    model.eval()
    
    # We will estimate Fisher for each head k on task k's train loader
    for k in range(num_tasks):
        expert_S = get_expert_state_dict(encoder_pre_state, encoder_experts_states[k], expert_heads_weights, k, num_tasks)
        
        # Get head parameter placeholders
        weight_param = expert_heads_weights[k]["weight"].clone().detach().to(device).requires_grad_(True)
        bias_param = expert_heads_weights[k]["bias"].clone().detach().to(device).requires_grad_(True)
        params = [weight_param, bias_param]
        
        accumulated_samples = 0
        loader = train_loaders[k]
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            # Inject active params so we can compute gradients with respect to them
            expert_S_active = expert_S.copy()
            expert_S_active[f"heads.{k}.weight"] = weight_param
            expert_S_active[f"heads.{k}.bias"] = bias_param
            
            logits = functional_call(model, expert_S_active, args=(x, k))
            loss = F.cross_entropy(logits, y)
            
            grads = torch.autograd.grad(loss, params, retain_graph=False)
            
            fisher_estimates[f"heads.{k}.weight"] += (grads[0] ** 2) * batch_size
            fisher_estimates[f"heads.{k}.bias"] += (grads[1] ** 2) * batch_size
            
            accumulated_samples += batch_size
            if accumulated_samples >= num_samples:
                break
                
        fisher_estimates[f"heads.{k}.weight"] /= accumulated_samples
        fisher_estimates[f"heads.{k}.bias"] /= accumulated_samples
        print(f"  Task {k} head Fisher computed on {accumulated_samples} samples.")
        
    # Now estimate Fisher for lambdas
    lambdas_val = torch.tensor([0.3, 0.3, 0.3], device=device, requires_grad=True)
    adapted_heads_params = {}
    for k in range(num_tasks):
        adapted_heads_params[f"heads.{k}.weight"] = expert_heads_weights[k]["weight"].clone().detach().to(device)
        adapted_heads_params[f"heads.{k}.bias"] = expert_heads_weights[k]["bias"].clone().detach().to(device)
        
    accumulated_samples_lambdas = 0
    for k in range(num_tasks):
        loader = train_loaders[k]
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            S = get_merged_state_dict(model, encoder_pre_state, encoder_pre_params, task_vectors, lambdas_val, adapted_heads_params, num_tasks)
            logits = functional_call(model, S, args=(x, k))
            loss = F.cross_entropy(logits, y)
            
            grads_lambdas = torch.autograd.grad(loss, [lambdas_val], retain_graph=False)
            fisher_estimates["lambdas"] += (grads_lambdas[0] ** 2) * batch_size
            
            accumulated_samples_lambdas += batch_size
            if accumulated_samples_lambdas >= num_samples:
                break
                
    fisher_estimates["lambdas"] /= accumulated_samples_lambdas
    print(f"  Lambdas Fisher computed on {accumulated_samples_lambdas} samples across tasks: {fisher_estimates['lambdas'].cpu().numpy()}")
    
    return fisher_estimates

def run_test_time_adaptation(
    model, 
    encoder_pre_state, 
    encoder_pre_params,
    encoder_experts_states, 
    expert_heads_weights, 
    task_vectors,
    tta_loaders,
    method,
    device,
    clean_fisher_estimates=None,
    num_tasks=3,
    steps=10,
    rho=0.05,
    eta=0.01,
    beta=0.9,
    gamma=1e-4,
    lr_lambda=0.001,
    lr_head=0.01,
    mu_fisher=500.0,
    alpha_fisher=10.0,
    conf_threshold=0.4
):
    # Initialize merging coefficients
    lambdas = torch.tensor([0.3, 0.3, 0.3], device=device, requires_grad=True)
    
    # Initialize adapted heads
    adapted_heads_params = {}
    for k in range(num_tasks):
        adapted_heads_params[f"heads.{k}.weight"] = expert_heads_weights[k]["weight"].clone().detach().requires_grad_(True)
        adapted_heads_params[f"heads.{k}.bias"] = expert_heads_weights[k]["bias"].clone().detach().requires_grad_(True)
        
    # Active parameters
    active_params = [lambdas] + list(adapted_heads_params.values())
    optimizer = torch.optim.Adam([
        {"params": [lambdas], "lr": lr_lambda},
        {"params": list(adapted_heads_params.values()), "lr": lr_head}
    ])
    
    # Initialize Running Fisher Information Matrix estimates
    if clean_fisher_estimates is not None:
        fisher_estimates = clean_fisher_estimates
    else:
        fisher_estimates = {}
        if method in ["f-samerge", "f-asam", "r-f-sam"]:
            for name, param in adapted_heads_params.items():
                fisher_estimates[name] = torch.zeros_like(param, device=device)
            fisher_estimates["lambdas"] = torch.zeros_like(lambdas, device=device)
        
    # Run TTA for specified steps
    model.eval()
    
    # We iterate through the loaders in parallel
    iterators = [iter(loader) for loader in tta_loaders]
    
    for step in range(steps):
        # 1. Fetch next batch for each task
        batches = []
        for k in range(num_tasks):
            try:
                x_k, _ = next(iterators[k])
            except StopIteration:
                # Cycle if we run out of data
                iterators[k] = iter(tta_loaders[k])
                x_k, _ = next(iterators[k])
            batches.append(x_k.to(device))
            
        # 2. Get expert soft labels
        P_experts = []
        with torch.no_grad():
            for k in range(num_tasks):
                expert_S = get_expert_state_dict(encoder_pre_state, encoder_experts_states[k], expert_heads_weights, k, num_tasks)
                expert_logits = functional_call(model, expert_S, args=(batches[k], k))
                P_experts.append(torch.softmax(expert_logits, dim=-1))
                
        # 3. Define forward pass and loss helper
        def compute_loss(lambdas_val, heads_val):
            S = get_merged_state_dict(model, encoder_pre_state, encoder_pre_params, task_vectors, lambdas_val, heads_val, num_tasks)
            kl_loss = 0.0
            for k in range(num_tasks):
                logits = functional_call(model, S, args=(batches[k], k))
                log_P_merged = F.log_softmax(logits, dim=-1)
                
                if method == "r-f-sam":
                    # Confidence-weighted Self-Labeling (CWSL)
                    kl_elements = F.kl_div(log_P_merged, P_experts[k], reduction="none")
                    kl_sample = torch.sum(kl_elements, dim=-1)
                    
                    entropy = -torch.sum(P_experts[k] * torch.log(P_experts[k] + 1e-12), dim=-1)
                    log_C = np.log(P_experts[k].size(-1))
                    norm_entropy = entropy / log_C
                    weight = (1.0 - norm_entropy).clamp(min=0.0)
                    
                    max_prob = P_experts[k].max(dim=-1)[0]
                    # Soft filter: downweight uncertain, hard filter: zero out extremely uncertain
                    weight = weight * (max_prob >= conf_threshold).float()
                    
                    weighted_kl = torch.sum(weight * kl_sample) / (torch.sum(weight) + 1e-12)
                    kl_loss += weighted_kl
                else:
                    # Standard unweighted KL
                    kl = F.kl_div(log_P_merged, P_experts[k], reduction="batchmean")
                    kl_loss += kl
                
            # If f-samerge, add online Fisher-anchored EWC penalty
            if method == "f-samerge":
                penalty = 0.0
                for name, param in heads_val.items():
                    task_idx = int(name.split(".")[1])
                    init_param = expert_heads_weights[task_idx]["weight" if "weight" in name else "bias"]
                    penalty += torch.sum(fisher_estimates[name] * (param - init_param)**2)
                # Lambdas penalty
                init_lambdas = torch.tensor([0.3, 0.3, 0.3], device=device)
                penalty += torch.sum(fisher_estimates["lambdas"] * (lambdas_val - init_lambdas)**2)
                
                kl_loss = kl_loss + mu_fisher * penalty
                
            return kl_loss
            
        # If static, no updates needed
        if method == "static":
            break
            
        # 4. optimization step based on method
        if method == "symerge":
            # Standard backprop
            optimizer.zero_grad()
            loss = compute_loss(lambdas, adapted_heads_params)
            loss.backward()
            optimizer.step()
            
        elif method in ["sam", "asam", "f-samerge", "f-asam", "r-f-sam"]:
            # First pass: compute loss and gradients
            loss = compute_loss(lambdas, adapted_heads_params)
            
            # Clear grads
            for p in active_params:
                p.grad = None
                
            loss.backward()
            
            # Extract gradients
            grads = {}
            for name, param in adapted_heads_params.items():
                if param.grad is not None:
                    grads[name] = param.grad.clone()
                else:
                    grads[name] = torch.zeros_like(param)
                    
            if lambdas.grad is not None:
                grads["lambdas"] = lambdas.grad.clone()
            else:
                grads["lambdas"] = torch.zeros_like(lambdas)
                
            # Compute perturbations
            perturbations = {}
            
            if method == "sam":
                # Compute global L2 norm of gradients
                grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads.values()) + 1e-12)
                for name, param in adapted_heads_params.items():
                    perturbations[name] = rho * grads[name] / grad_norm
                perturbations["lambdas"] = rho * grads["lambdas"] / grad_norm
                
            elif method in ["asam", "f-samerge", "f-asam", "r-f-sam"]:
                if method in ["f-samerge", "f-asam", "r-f-sam"] and clean_fisher_estimates is None:
                    # Update running Fisher estimate with pre-perturbed gradients
                    for name, param in adapted_heads_params.items():
                        fisher_estimates[name] = beta * fisher_estimates[name] + (1.0 - beta) * (grads[name]**2)
                    fisher_estimates["lambdas"] = beta * fisher_estimates["lambdas"] + (1.0 - beta) * (grads["lambdas"]**2)
                
                if method in ["f-asam", "r-f-sam"]:
                    # Bounded Fisher-weighted Adaptive SAM (BF-ASAM)
                    # To prevent division-by-zero or low-Fisher explosion, we scale the ASAM
                    # perturbation divisor by (1 + alpha_fisher * fisher_estimates)
                    # This ensures that low-Fisher weights are perturbed like standard ASAM,
                    # while high-Fisher (critical) weights are perturbed less.
                    denom = 0.0
                    for name, param in adapted_heads_params.items():
                        scale = ((torch.abs(param) + eta)**2) / (1.0 + alpha_fisher * fisher_estimates[name])
                        denom += torch.sum(scale * grads[name]**2)
                    scale_lambdas = ((torch.abs(lambdas) + eta)**2) / (1.0 + alpha_fisher * fisher_estimates["lambdas"])
                    denom += torch.sum(scale_lambdas * grads["lambdas"]**2)
                    denom = torch.sqrt(denom + 1e-12)
                    
                    for name, param in adapted_heads_params.items():
                        scale = ((torch.abs(param) + eta)**2) / (1.0 + alpha_fisher * fisher_estimates[name])
                        perturbations[name] = rho * (scale * grads[name]) / denom
                    scale_lambdas = ((torch.abs(lambdas) + eta)**2) / (1.0 + alpha_fisher * fisher_estimates["lambdas"])
                    perturbations["lambdas"] = rho * (scale_lambdas * grads["lambdas"]) / denom
                else:
                    # Standard scale-invariant ASAM perturbation
                    denom = 0.0
                    for name, param in adapted_heads_params.items():
                        denom += torch.sum(((torch.abs(param) + eta) * grads[name])**2)
                    denom += torch.sum(((torch.abs(lambdas) + eta) * grads["lambdas"])**2)
                    denom = torch.sqrt(denom + 1e-12)
                    
                    for name, param in adapted_heads_params.items():
                        perturbations[name] = rho * ((torch.abs(param) + eta)**2) * grads[name] / denom
                    perturbations["lambdas"] = rho * ((torch.abs(lambdas) + eta)**2) * grads["lambdas"] / denom
                
            # Apply perturbation
            perturbed_lambdas = (lambdas + perturbations["lambdas"]).detach().requires_grad_(True)
            perturbed_heads = {}
            for name, param in adapted_heads_params.items():
                perturbed_heads[name] = (param + perturbations[name]).detach().requires_grad_(True)
                
            # Second pass: compute perturbed loss
            loss_perturbed = compute_loss(perturbed_lambdas, perturbed_heads)
            
            # Clear perturbed grads
            perturbed_params_list = [perturbed_lambdas] + list(perturbed_heads.values())
            for p in perturbed_params_list:
                p.grad = None
                
            loss_perturbed.backward()
            
            # Apply perturbed gradients to original active parameters
            optimizer.zero_grad()
            for name, param in adapted_heads_params.items():
                if perturbed_heads[name].grad is not None:
                    param.grad = perturbed_heads[name].grad.clone()
            if perturbed_lambdas.grad is not None:
                lambdas.grad = perturbed_lambdas.grad.clone()
                
            # Update weights using original optimizer and perturbed gradients
            optimizer.step()
            
    # Return adapted parameters
    return lambdas.detach(), {k: v.detach() for k, v in adapted_heads_params.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run on CPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Expert fine-tuning epochs")
    parser.add_argument("--steps", type=int, default=10, help="TTA optimization steps")
    parser.add_argument("--rho", type=type(0.05), default=0.05, help="SAM/ASAM/Fisher perturbation scale")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--mu-fisher", type=float, default=500.0, help="Fisher-anchoring regularizer strength")
    parser.add_argument("--alpha-fisher", type=float, default=10.0, help="Fisher weighting scaling coefficient")
    parser.add_argument("--conf-threshold", type=float, default=0.4, help="Confidence threshold for R-BF-SAM / R-FA-SAM")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Datasets
    print("Loading datasets...")
    datasets = get_datasets(args.data_dir)
    num_tasks = len(datasets["train"])
    
    if args.dry_run:
        print("Dry-run mode: using small subset of datasets and 1 step")
        args.epochs = 1
        args.steps = 2
        # Use subsets of 128 images for training and evaluation
        for k in range(num_tasks):
            datasets["train"][k] = Subset(datasets["train"][k], range(128))
            datasets["test"][k] = Subset(datasets["test"][k], range(128))
            
    # Create DataLoaders
    train_loaders = [DataLoader(d, batch_size=128, shuffle=True) for d in datasets["train"]]
    # For TTA, batch size is 32
    tta_loaders = [DataLoader(d, batch_size=32, shuffle=True) for d in datasets["test"]]
    # For evaluation, batch size is 128
    eval_loaders = [DataLoader(d, batch_size=128, shuffle=False) for d in datasets["test"]]
    
    # 2. Initialize Model
    print("Initializing Multi-Task ResNet18 Model...")
    model = MultiTaskModel(num_tasks=num_tasks).to(device)
    
    # Capture pre-trained encoder state
    encoder_pre_state = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    encoder_pre_params = set(name for name, _ in model.encoder.named_parameters())
    
    # 3. Train task-specific experts
    print("Fine-tuning expert models...")
    encoder_experts_states = []
    expert_heads_weights = []
    
    for k in range(num_tasks):
        print(f"--- Training Expert {k+1}/{num_tasks} (Task {k}) ---")
        # Reset model to pre-trained weights for each training
        model.encoder.load_state_dict(encoder_pre_state)
        # Re-initialize head k
        nn.init.kaiming_normal_(model.heads[k].weight)
        nn.init.zeros_(model.heads[k].bias)
        
        # Train
        train_expert(model, train_loaders[k], task_id=k, epochs=args.epochs, device=device)
        
        # Save fine-tuned encoder and head
        encoder_experts_states.append({name: param.clone() for name, param in model.encoder.state_dict().items()})
        expert_heads_weights.append({
            "weight": model.heads[k].weight.clone(),
            "bias": model.heads[k].bias.clone()
        })
        
        # Eval expert on clean test set
        clean_acc = evaluate_expert(model, eval_loaders[k], task_id=k, device=device)
        print(f"Expert {k+1} Clean Test Accuracy: {clean_acc*100:.2f}%")
        
    # Calculate task vectors
    task_vectors = []
    for k in range(num_tasks):
        vec = {}
        for name in encoder_pre_state.keys():
            # If buffer or integer, task vector is zero
            if encoder_pre_state[name].is_floating_point() and "num_batches_tracked" not in name:
                vec[name] = encoder_experts_states[k][name] - encoder_pre_state[name]
            else:
                vec[name] = torch.zeros_like(encoder_pre_state[name])
        task_vectors.append(vec)
        
    # Calculate Clean Fisher Estimates on Training datasets
    num_fisher_samples = 32 if args.dry_run else 512
    clean_fisher_estimates = compute_clean_fisher_estimates(
        model=model,
        encoder_pre_state=encoder_pre_state,
        encoder_pre_params=encoder_pre_params,
        encoder_experts_states=encoder_experts_states,
        expert_heads_weights=expert_heads_weights,
        task_vectors=task_vectors,
        train_loaders=train_loaders,
        device=device,
        num_tasks=num_tasks,
        num_samples=num_fisher_samples
    )
        
    # 4. Evaluation Loop across corruptions and adaptation methods
    corruptions = ["none", "gaussian_noise", "gaussian_blur", "contrast_reduction", "image_rotation"]
    methods = ["static", "symerge", "sam", "asam", "f-samerge", "f-asam", "r-f-sam"]
    
    results = {method: {} for method in methods}
    
    print("\n--- Running Test-Time Adaptation and Evaluation ---")
    for corruption in corruptions:
        print(f"\nEvaluating Corruption: {corruption}")
        
        # Prepare corrupted TTA and eval loaders
        corrupted_tta_loaders = []
        corrupted_eval_loaders = []
        
        for k in range(num_tasks):
            # We apply corruption on the fly via a custom DataLoader wrapper or preprocessing
            # Let's write a simple wrapper for Subset or Dataset that applies corruption
            class CorruptedDataset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, corr_type, dev):
                    self.base = base_dataset
                    self.corr = corr_type
                    self.dev = dev
                def __len__(self):
                    return len(self.base)
                def __getitem__(self, idx):
                    x, y = self.base[idx]
                    # Put on CPU/GPU, corrupt, put on CPU
                    x_corr = apply_corruption(x.unsqueeze(0).to(self.dev), self.corr, self.dev).squeeze(0).cpu()
                    return x_corr, y
            
            corr_tta_ds = CorruptedDataset(datasets["test"][k], corruption, device)
            corr_eval_ds = CorruptedDataset(datasets["test"][k], corruption, device)
            
            corrupted_tta_loaders.append(DataLoader(corr_tta_ds, batch_size=32, shuffle=True))
            corrupted_eval_loaders.append(DataLoader(corr_eval_ds, batch_size=128, shuffle=False))
            
        for method in methods:
            print(f"  Method: {method}")
            # Run Test-Time Adaptation to obtain adapted lambdas and heads
            adapted_lambdas, adapted_heads_params = run_test_time_adaptation(
                model=model,
                encoder_pre_state=encoder_pre_state,
                encoder_pre_params=encoder_pre_params,
                encoder_experts_states=encoder_experts_states,
                expert_heads_weights=expert_heads_weights,
                task_vectors=task_vectors,
                tta_loaders=corrupted_tta_loaders,
                method=method,
                device=device,
                clean_fisher_estimates=clean_fisher_estimates,
                num_tasks=num_tasks,
                steps=args.steps,
                rho=args.rho,
                mu_fisher=args.mu_fisher,
                alpha_fisher=args.alpha_fisher,
                conf_threshold=args.conf_threshold
            )
            
            print(f"    Adapted Lambdas: {adapted_lambdas.cpu().numpy()}")
            
            # Evaluate the adapted model on the corrupted test set
            merged_state = get_merged_state_dict(model, encoder_pre_state, encoder_pre_params, task_vectors, adapted_lambdas, adapted_heads_params, num_tasks)
            
            task_accuracies = []
            for k in range(num_tasks):
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in corrupted_eval_loaders[k]:
                        x, y = x.to(device), y.to(device)
                        logits = functional_call(model, merged_state, args=(x, k))
                        preds = logits.argmax(dim=-1)
                        correct += (preds == y).sum().item()
                        total += x.size(0)
                acc = correct / total
                task_accuracies.append(acc)
                
            avg_acc = sum(task_accuracies) / num_tasks
            results[method][corruption] = {
                "tasks": task_accuracies,
                "avg": avg_acc
            }
            print(f"    Task Accuracies: {[f'{a*100:.2f}%' for a in task_accuracies]} | Avg: {avg_acc*100:.2f}%")
            
    # 5. Print final comparative results
    print("\n" + "="*50)
    print("FINAL SUMMARY OF RESULTS (Average Multitask Accuracy)")
    print("="*50)
    header = f"{'Method':<12} | " + " | ".join(f"{c[:8]:<8}" for c in corruptions) + " | OOD Avg"
    print(header)
    print("-"*len(header))
    
    for method in methods:
        row = f"{method:<12} | "
        ood_sum = 0.0
        ood_count = 0
        for corruption in corruptions:
            acc = results[method][corruption]["avg"] * 100
            row += f"{acc:.2f}%   | "
            if corruption != "none":
                ood_sum += acc
                ood_count += 1
        row += f"{ood_sum / ood_count:.2f}%"
        print(row)
    print("="*50)

if __name__ == "__main__":
    main()
