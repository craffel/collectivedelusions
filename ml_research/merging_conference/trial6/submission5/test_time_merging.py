import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.func import functional_call
import numpy as np
import matplotlib.pyplot as plt

# Disable cuDNN to bypass cluster initialization errors
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluating on device:", device)

def get_resnet18_expert():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def load_experts():
    experts = {}
    datasets = ["mnist", "fashionmnist", "kmnist"]
    for d in datasets:
        path = f"./models/expert_{d}.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert checkpoint not found at {path}. Please run train_experts.py first.")
        model = get_resnet18_expert()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        experts[d] = model
    return experts

def get_base_model():
    # Base model is the pre-trained ResNet-18 modified to accept 1-channel grayscale input
    model = resnet18(pretrained=True)
    pretrained_conv1 = model.conv1.weight.data
    new_conv1_weight = pretrained_conv1.sum(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_conv1_weight
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

def get_datasets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    return {"mnist": mnist_test, "fashionmnist": fmnist_test, "kmnist": kmnist_test}

# Simplex projection
def project_simplex_batch(v):
    u, _ = torch.sort(v, descending=True, dim=-1)
    cssv = torch.cumsum(u, dim=-1)
    indices = torch.arange(1, v.size(-1) + 1, device=v.device).float()
    cond = u - (cssv - 1.0) / indices > 0
    # argmax of indices * cond gives largest index satisfying the condition
    rho = torch.argmax(indices * cond, dim=-1)
    cssv_at_rho = cssv[torch.arange(v.size(0)), rho]
    theta = (cssv_at_rho - 1.0) / (rho + 1)
    w = torch.clamp(v - theta.unsqueeze(-1), min=0)
    return w

# Corruption helper
def apply_corruption(images, corruption, std=0.3, factor=0.4):
    if corruption == "clean":
        return images
    elif corruption == "gaussian_noise":
        # Convert back from [-1, 1] to [0, 1] for raw image space
        raw = images * 0.5 + 0.5
        raw = raw + torch.randn_like(raw) * std
        raw = torch.clamp(raw, 0.0, 1.0)
        return (raw - 0.5) / 0.5
    elif corruption == "contrast_shift":
        raw = images * 0.5 + 0.5
        raw = raw * factor
        raw = torch.clamp(raw, 0.0, 1.0)
        return (raw - 0.5) / 0.5
    return images

# Extract prototypes using 500 calibration samples per task
def extract_prototypes(experts, datasets):
    print("Extracting class prototypes...")
    prototypes = {}
    for d, model in experts.items():
        # Get first 500 samples of dataset d
        dataset = datasets[d]
        subset = Subset(dataset, list(range(500)))
        loader = DataLoader(subset, batch_size=32, shuffle=False)
        
        # We need the 512-dim features before the final linear layer (fc)
        # To get features, we register a forward hook or just call model.forward up to fc
        features_list = []
        labels_list = []
        
        def hook_fn(module, input, output):
            features_list.append(input[0].cpu()) # output of avgpool is input to fc
            
        hook = model.fc.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                model(images)
                labels_list.append(labels)
                
        hook.remove()
        
        # Concatenate features and labels
        features = torch.cat(features_list, dim=0).squeeze() # shape: (500, 512)
        labels = torch.cat(labels_list, dim=0)
        
        # Compute mean feature vector per class
        class_prototypes = []
        for c in range(10):
            mask = (labels == c)
            if mask.sum() > 0:
                class_feat = features[mask].mean(dim=0)
                class_feat = class_feat / class_feat.norm(p=2) # L2-normalized
                class_prototypes.append(class_feat.to(device))
            else:
                class_prototypes.append(torch.zeros(512).to(device))
                
        # Shape: (10, 512)
        prototypes[d] = torch.stack(class_prototypes, dim=0)
    print("Prototypes extracted successfully!")
    return prototypes

# Precompute offline Fisher Information
def precompute_offline_fisher(base_model, experts, datasets):
    print("Precomputing offline diagonal Fisher Information...")
    fisher = {}
    # Use 500 samples per task
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
            
    criterion = nn.CrossEntropyLoss()
    total_count = 0
    
    for d, model in experts.items():
        dataset = datasets[d]
        subset = Subset(dataset, list(range(500)))
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            total_count += 1
            
    # Compute tensor-level averages
    layer_fisher = {}
    for name in fisher:
        fisher[name] /= total_count
        layer_fisher[name] = fisher[name].mean().item()
        
    print("Offline Fisher Information computed successfully!")
    return layer_fisher

# Test stream constructor
def construct_streams(datasets):
    mnist_test = datasets["mnist"]
    fmnist_test = datasets["fashionmnist"]
    kmnist_test = datasets["kmnist"]
    
    # We need 1,600 samples per task (total 4,800 samples)
    mnist_sub = Subset(mnist_test, list(range(1600)))
    fmnist_sub = Subset(fmnist_test, list(range(1600)))
    kmnist_sub = Subset(kmnist_test, list(range(1600)))
    
    # Batch size 32 means 50 batches per task
    mnist_loader = DataLoader(mnist_sub, batch_size=32, shuffle=False)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=32, shuffle=False)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=32, shuffle=False)
    
    mnist_batches = list(mnist_loader)
    fmnist_batches = list(fmnist_loader)
    kmnist_batches = list(kmnist_loader)
    
    # 1. Alternating Stream: batches alternate on every step
    alternating_stream = []
    for i in range(50):
        alternating_stream.append((mnist_batches[i][0], mnist_batches[i][1], "mnist"))
        alternating_stream.append((fmnist_batches[i][0], fmnist_batches[i][1], "fashionmnist"))
        alternating_stream.append((kmnist_batches[i][0], kmnist_batches[i][1], "kmnist"))
        
    # 2. Sequential Stream: blocks of 50 batches per task
    sequential_stream = []
    for i in range(50):
        sequential_stream.append((mnist_batches[i][0], mnist_batches[i][1], "mnist"))
    for i in range(50):
        sequential_stream.append((fmnist_batches[i][0], fmnist_batches[i][1], "fashionmnist"))
    for i in range(50):
        sequential_stream.append((kmnist_batches[i][0], kmnist_batches[i][1], "kmnist"))
        
    return {"alternating": alternating_stream, "sequential": sequential_stream}

# Main Evaluator
class TestTimeMergingEvaluator:
    def __init__(self, base_model, experts, prototypes, offline_fisher_scalar):
        self.base_model = base_model
        self.experts = experts
        self.prototypes = prototypes
        self.offline_fisher_scalar = offline_fisher_scalar
        
        # Prepare task vectors: v_k = \theta_k - \theta_base
        self.task_vectors = {}
        for d, model in experts.items():
            task_dict = {}
            for name, param in model.named_parameters():
                base_param = self.base_model.state_dict()[name]
                task_dict[name] = param.data - base_param.data
            self.task_vectors[d] = task_dict
            
        # Get parameter names that are trainable
        self.trainable_names = [name for name, param in base_model.named_parameters() if param.requires_grad]
        
    def merge_weights_and_buffers(self, lambdas):
        # lambdas: dict of shape {layer_name: tensor(3)}
        # Merge trainable parameters
        merged_params = {}
        for name in self.trainable_names:
            base_p = self.base_model.state_dict()[name]
            l = lambdas[name]
            # l: [lambda_mnist, lambda_fmnist, lambda_kmnist]
            v_mnist = self.task_vectors["mnist"][name]
            v_fmnist = self.task_vectors["fashionmnist"][name]
            v_kmnist = self.task_vectors["kmnist"][name]
            merged_params[name] = base_p + l[0] * v_mnist + l[1] * v_fmnist + l[2] * v_kmnist
            
        # Merge batchnorm buffers using average lambda across all trainable layers
        avg_lambda = torch.stack(list(lambdas.values())).mean(dim=0).detach()
        merged_buffers = {}
        for name, buf in self.base_model.named_buffers():
            if "running_mean" in name or "running_var" in name:
                buf_mnist = self.experts["mnist"].state_dict()[name]
                buf_fmnist = self.experts["fashionmnist"].state_dict()[name]
                buf_kmnist = self.experts["kmnist"].state_dict()[name]
                merged_buffers[name] = buf + avg_lambda[0] * (buf_mnist - buf) + avg_lambda[1] * (buf_fmnist - buf) + avg_lambda[2] * (buf_kmnist - buf)
            else:
                merged_buffers[name] = buf
                
        return merged_params, merged_buffers

    def evaluate_static(self, stream, corruption):
        # Static merging uses uniform lambdas [1/3, 1/3, 1/3]
        lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in self.trainable_names}
        merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
        
        correct = 0
        total = 0
        for images, labels, _ in stream:
            images = apply_corruption(images.to(device), corruption)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return correct / total

    def evaluate_adamerging(self, stream, corruption, lr=1e-3, steps=1):
        # Initialize lambdas to uniform
        lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in self.trainable_names}
        
        correct = 0
        total = 0
        trajectory = []
        
        for images, labels, _ in stream:
            images = apply_corruption(images.to(device), corruption)
            labels = labels.to(device)
            
            # Record trajectory (average of lambdas across layers)
            avg_l = torch.stack([l.data for l in lambdas.values()]).mean(dim=0).cpu().numpy()
            trajectory.append(avg_l)
            
            # Adapt merging coefficients for "steps" steps using entropy minimization
            for _ in range(steps):
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                probs = torch.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                
                # Compute gradients
                grads = torch.autograd.grad(entropy, list(lambdas.values()))
                
                # Gradient update and projection to simplex
                with torch.no_grad():
                    for name, grad in zip(self.trainable_names, grads):
                        # Update and project
                        v = lambdas[name] - lr * grad
                        lambdas[name].copy_(project_simplex_batch(v.unsqueeze(0)).squeeze(0))
                        
            # Forward pass with adapted weights for evaluation
            with torch.no_grad():
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return correct / total, trajectory

    def evaluate_lfwa(self, stream, corruption, lr=1e-3, steps=1, alpha=1.0, eps=1e-6):
        # LFWA: entropy minimization preconditioned with offline Fisher
        lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in self.trainable_names}
        
        correct = 0
        total = 0
        trajectory = []
        
        for images, labels, _ in stream:
            images = apply_corruption(images.to(device), corruption)
            labels = labels.to(device)
            
            avg_l = torch.stack([l.data for l in lambdas.values()]).mean(dim=0).cpu().numpy()
            trajectory.append(avg_l)
            
            for _ in range(steps):
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                probs = torch.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                
                grads = torch.autograd.grad(entropy, list(lambdas.values()))
                
                with torch.no_grad():
                    for name, grad in zip(self.trainable_names, grads):
                        f_scalar = self.offline_fisher_scalar[name]
                        eta_w = lr * ((f_scalar + eps) ** (-alpha))
                        # Clip learning rate to avoid exploding updates in insensitive layers
                        eta_w = min(eta_w, 1.0)
                        v = lambdas[name] - eta_w * grad
                        lambdas[name].copy_(project_simplex_batch(v.unsqueeze(0)).squeeze(0))
                        
            with torch.no_grad():
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return correct / total, trajectory

    def evaluate_ao_lfwa(self, stream, corruption, lr=1e-3, steps=1, alpha=1.0, eps=1e-6, gamma=0.1):
        # AO-LFWA (Proposed): online active estimation of diagonal Fisher
        # Initialize lambdas
        lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in self.trainable_names}
        # Initialize online Fisher to all ones
        online_fisher_scalar = {name: 1.0 for name in self.trainable_names}
        
        correct = 0
        total = 0
        trajectory = []
        
        for images, labels, _ in stream:
            images = apply_corruption(images.to(device), corruption)
            labels = labels.to(device)
            
            avg_l = torch.stack([l.data for l in lambdas.values()]).mean(dim=0).cpu().numpy()
            trajectory.append(avg_l)
            
            for _ in range(steps):
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                
                # We need to compute gradients of the loss with respect to both:
                # 1. The merged parameter tensors themselves (to update online Fisher)
                # 2. The merging coefficients (to update lambdas)
                # To make merged parameters differentiable, we reconstruct them with requires_grad=True
                active_params = {}
                for name in self.trainable_names:
                    active_params[name] = merged_params[name].clone().detach().requires_grad_(True)
                    
                outputs = functional_call(self.base_model, {**active_params, **merged_buffers}, images)
                probs = torch.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                
                # Compute gradient with respect to merged parameters
                param_grads = torch.autograd.grad(entropy, list(active_params.values()), retain_graph=True, allow_unused=True)
                
                # Update online Fisher running estimates using EMA of squared gradients
                with torch.no_grad():
                    for name, p_grad in zip(self.trainable_names, param_grads):
                        if p_grad is not None:
                            grad_mean_sq = p_grad.pow(2).mean().item()
                            online_fisher_scalar[name] = (1 - gamma) * online_fisher_scalar[name] + gamma * grad_mean_sq
                            
                # Now compute gradients of entropy with respect to lambdas
                # Using the original merged_params which are differentiable w.r.t. lambdas
                outputs_lam = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                probs_lam = torch.softmax(outputs_lam, dim=-1)
                entropy_lam = -torch.sum(probs_lam * torch.log(probs_lam + 1e-8), dim=-1).mean()
                grads = torch.autograd.grad(entropy_lam, list(lambdas.values()))
                
                # Update lambdas using preconditioned online Fisher
                with torch.no_grad():
                    for name, grad in zip(self.trainable_names, grads):
                        f_scalar = online_fisher_scalar[name]
                        eta_w = lr * ((f_scalar + eps) ** (-alpha))
                        eta_w = min(eta_w, 1.0) # clip to avoid numerical instability
                        v = lambdas[name] - eta_w * grad
                        lambdas[name].copy_(project_simplex_batch(v.unsqueeze(0)).squeeze(0))
                        
            with torch.no_grad():
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return correct / total, trajectory

    def evaluate_fpca(self, stream, corruption, lr=1e-2, steps=1, alpha=1.0, eps=1e-6, beta=0.1, gamma_mask=0.85):
        # FP-CA baseline: Prototype Routing, Masked Contrastive Loss, and Offline Fisher Preconditioning
        lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in self.trainable_names}
        
        correct = 0
        total = 0
        trajectory = []
        
        # To compute task affinity, we need active task prototypes
        # We find the 512-dim feature embedding by registering a hook on the fc layer of the model
        for step_idx, (images, labels, task_name) in enumerate(stream):
            images = apply_corruption(images.to(device), corruption)
            labels = labels.to(device)
            
            avg_l = torch.stack([l.data for l in lambdas.values()]).mean(dim=0).cpu().numpy()
            trajectory.append(avg_l)
            
            # --- 1. Prototype-driven Dynamic Routing (PD-Routing) ---
            # Forward pass with uniform model to detect active task
            with torch.no_grad():
                static_lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in self.trainable_names}
                st_params, st_bufs = self.merge_weights_and_buffers(static_lambdas)
                
                # Extract features before fc
                features_list = []
                def hook_fn(module, input, output):
                    features_list.append(input[0].clone())
                hook = self.base_model.fc.register_forward_hook(hook_fn)
                functional_call(self.base_model, {**st_params, **st_bufs}, images)
                hook.remove()
                
                feats = features_list[0].squeeze() # shape: (batch_size, 512)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                
                # Compute cosine similarity with prototypes of each expert
                similarities = {}
                for d, proto in self.prototypes.items():
                    # proto shape: (10, 512)
                    # Compute sim of each sample features with class prototypes of domain d
                    sim = torch.matmul(feats, proto.t()) # shape: (batch_size, 10)
                    similarities[d] = sim.max(dim=-1)[0].mean().item()
                    
                # Routing selection
                best_task = max(similarities, key=similarities.get)
                # Prior prior
                task_map = {"mnist": 0, "fashionmnist": 1, "kmnist": 2}
                prior_score = similarities[best_task]
                
                # If confidence is high, reset lambdas
                if prior_score > 0.65: # threshold lower than clean as noise degrades features
                    prior_vec = torch.zeros(3, device=device)
                    prior_vec[task_map[best_task]] = 1.0
                    for name in self.trainable_names:
                        lambdas[name].copy_(prior_vec)
                        
            # --- 2. Adaptation Step ---
            for _ in range(steps):
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                
                # Extract features for contrastive alignment
                features_list = []
                hook = self.base_model.fc.register_forward_hook(hook_fn)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                hook.remove()
                
                probs = torch.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                
                # Masked Contrastive Loss
                feats = features_list[0].squeeze()
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                
                # Get soft/hard prediction class
                max_prob, pred_class = probs.max(dim=-1)
                mask = (max_prob > gamma_mask)
                
                if mask.sum() > 0:
                    # Active task prototypes
                    active_proto = self.prototypes[best_task] # shape: (10, 512)
                    masked_feats = feats[mask] # shape: (M, 512)
                    masked_pred = pred_class[mask] # shape: (M,)
                    
                    # Compute cosine similarity matrix
                    sim_matrix = torch.matmul(masked_feats, active_proto.t()) / 0.1 # temp 0.1
                    
                    # Cross-entropy contrastive loss
                    criterion_contra = nn.CrossEntropyLoss()
                    loss_contra = criterion_contra(sim_matrix, masked_pred)
                    loss_total = entropy + beta * loss_contra
                else:
                    loss_total = entropy
                    
                grads = torch.autograd.grad(loss_total, list(lambdas.values()))
                
                with torch.no_grad():
                    for name, grad in zip(self.trainable_names, grads):
                        f_scalar = self.offline_fisher_scalar[name]
                        eta_w = lr * ((f_scalar + eps) ** (-alpha))
                        eta_w = min(eta_w, 1.0)
                        v = lambdas[name] - eta_w * grad
                        lambdas[name].copy_(project_simplex_batch(v.unsqueeze(0)).squeeze(0))
                        
            with torch.no_grad():
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return correct / total, trajectory

    def evaluate_ao_fpca(self, stream, corruption, lr=1e-2, steps=1, alpha=1.0, eps=1e-6, beta=0.1, gamma_mask=0.85, gamma_fish=0.1):
        # AO-FP-CA (Our Proposed Method): Active Online Fisher + Prototype Routing & Masked Contrastive Loss
        lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in self.trainable_names}
        online_fisher_scalar = {name: 1.0 for name in self.trainable_names}
        
        correct = 0
        total = 0
        trajectory = []
        
        for step_idx, (images, labels, task_name) in enumerate(stream):
            images = apply_corruption(images.to(device), corruption)
            labels = labels.to(device)
            
            avg_l = torch.stack([l.data for l in lambdas.values()]).mean(dim=0).cpu().numpy()
            trajectory.append(avg_l)
            
            # --- 1. Prototype-driven Dynamic Routing (PD-Routing) ---
            with torch.no_grad():
                static_lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in self.trainable_names}
                st_params, st_bufs = self.merge_weights_and_buffers(static_lambdas)
                
                features_list = []
                def hook_fn(module, input, output):
                    features_list.append(input[0].clone())
                hook = self.base_model.fc.register_forward_hook(hook_fn)
                functional_call(self.base_model, {**st_params, **st_bufs}, images)
                hook.remove()
                
                feats = features_list[0].squeeze()
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                
                similarities = {}
                for d, proto in self.prototypes.items():
                    sim = torch.matmul(feats, proto.t())
                    similarities[d] = sim.max(dim=-1)[0].mean().item()
                    
                best_task = max(similarities, key=similarities.get)
                task_map = {"mnist": 0, "fashionmnist": 1, "kmnist": 2}
                prior_score = similarities[best_task]
                
                if prior_score > 0.65:
                    prior_vec = torch.zeros(3, device=device)
                    prior_vec[task_map[best_task]] = 1.0
                    for name in self.trainable_names:
                        lambdas[name].copy_(prior_vec)
                        
            # --- 2. Adaptation Step ---
            for _ in range(steps):
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                
                # A. Update online Fisher estimates using differentiable active_params
                active_params = {}
                for name in self.trainable_names:
                    active_params[name] = merged_params[name].clone().detach().requires_grad_(True)
                    
                features_list = []
                hook = self.base_model.fc.register_forward_hook(hook_fn)
                outputs = functional_call(self.base_model, {**active_params, **merged_buffers}, images)
                hook.remove()
                
                probs = torch.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                
                # Compute gradient with respect to merged parameters
                param_grads = torch.autograd.grad(entropy, list(active_params.values()), retain_graph=True, allow_unused=True)
                
                # Update online Fisher running estimates using EMA of squared gradients
                with torch.no_grad():
                    for name, p_grad in zip(self.trainable_names, param_grads):
                        if p_grad is not None:
                            grad_mean_sq = p_grad.pow(2).mean().item()
                            online_fisher_scalar[name] = (1 - gamma_fish) * online_fisher_scalar[name] + gamma_fish * grad_mean_sq
                            
                # B. Now compute actual gradient of total loss with respect to lambdas
                # Differentiable features hook for contrastive loss
                features_list = []
                hook = self.base_model.fc.register_forward_hook(hook_fn)
                outputs_lam = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                hook.remove()
                
                probs_lam = torch.softmax(outputs_lam, dim=-1)
                entropy_lam = -torch.sum(probs_lam * torch.log(probs_lam + 1e-8), dim=-1).mean()
                
                feats = features_list[0].squeeze()
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                
                max_prob, pred_class = probs_lam.max(dim=-1)
                mask = (max_prob > gamma_mask)
                
                if mask.sum() > 0:
                    active_proto = self.prototypes[best_task]
                    masked_feats = feats[mask]
                    masked_pred = pred_class[mask]
                    sim_matrix = torch.matmul(masked_feats, active_proto.t()) / 0.1
                    criterion_contra = nn.CrossEntropyLoss()
                    loss_contra = criterion_contra(sim_matrix, masked_pred)
                    loss_total = entropy_lam + beta * loss_contra
                else:
                    loss_total = entropy_lam
                    
                grads = torch.autograd.grad(loss_total, list(lambdas.values()))
                
                # Update lambdas using preconditioned online Fisher
                with torch.no_grad():
                    for name, grad in zip(self.trainable_names, grads):
                        f_scalar = online_fisher_scalar[name]
                        eta_w = lr * ((f_scalar + eps) ** (-alpha))
                        eta_w = min(eta_w, 1.0)
                        v = lambdas[name] - eta_w * grad
                        lambdas[name].copy_(project_simplex_batch(v.unsqueeze(0)).squeeze(0))
                        
            with torch.no_grad():
                merged_params, merged_buffers = self.merge_weights_and_buffers(lambdas)
                outputs = functional_call(self.base_model, {**merged_params, **merged_buffers}, images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return correct / total, trajectory

def run_experiment_suite():
    print("\nLoading datasets and experts...")
    datasets = get_datasets()
    experts = load_experts()
    base_model = get_base_model()
    
    # Extract prototypes
    prototypes = extract_prototypes(experts, datasets)
    
    # Precompute offline Fisher Information
    offline_fisher = precompute_offline_fisher(base_model, experts, datasets)
    
    # Construct test streams
    streams = construct_streams(datasets)
    
    # Evaluator
    evaluator = TestTimeMergingEvaluator(base_model, experts, prototypes, offline_fisher)
    
    results = {}
    trajectories = {}
    
    # Evaluate across streams and corruptions
    for stream_name, stream in streams.items():
        results[stream_name] = {}
        trajectories[stream_name] = {}
        for corruption in ["clean", "gaussian_noise", "contrast_shift"]:
            print(f"\nEvaluating stream: {stream_name} under corruption: {corruption}")
            results[stream_name][corruption] = {}
            trajectories[stream_name][corruption] = {}
            
            # 1. Static Merging
            acc_static = evaluator.evaluate_static(stream, corruption)
            print(f"Static Merging Accuracy: {acc_static*100:.2f}%")
            results[stream_name][corruption]["Static"] = acc_static
            
            # 2. AdaMerging (uniform lr)
            acc_ada, traj_ada = evaluator.evaluate_adamerging(stream, corruption)
            print(f"AdaMerging Accuracy: {acc_ada*100:.2f}%")
            results[stream_name][corruption]["AdaMerging"] = acc_ada
            trajectories[stream_name][corruption]["AdaMerging"] = traj_ada
            
            # 3. LFWA (offline Fisher)
            acc_lfwa, traj_lfwa = evaluator.evaluate_lfwa(stream, corruption)
            print(f"LFWA Accuracy: {acc_lfwa*100:.2f}%")
            results[stream_name][corruption]["LFWA"] = acc_lfwa
            trajectories[stream_name][corruption]["LFWA"] = traj_lfwa
            
            # 4. AO-LFWA (Our Proposed Online Fisher)
            acc_ao_lfwa, traj_ao_lfwa = evaluator.evaluate_ao_lfwa(stream, corruption)
            print(f"AO-LFWA (Proposed) Accuracy: {acc_ao_lfwa*100:.2f}%")
            results[stream_name][corruption]["AO-LFWA"] = acc_ao_lfwa
            trajectories[stream_name][corruption]["AO-LFWA"] = traj_ao_lfwa
            
            # 5. FP-CA (offline Fisher + Prototypes)
            acc_fpca, traj_fpca = evaluator.evaluate_fpca(stream, corruption)
            print(f"FP-CA Accuracy: {acc_fpca*100:.2f}%")
            results[stream_name][corruption]["FP-CA"] = acc_fpca
            trajectories[stream_name][corruption]["FP-CA"] = traj_fpca
            
            # 6. AO-FP-CA (Our Proposed Method)
            acc_ao_fpca, traj_ao_fpca = evaluator.evaluate_ao_fpca(stream, corruption)
            print(f"AO-FP-CA (Proposed) Accuracy: {acc_ao_fpca*100:.2f}%")
            results[stream_name][corruption]["AO-FP-CA"] = acc_ao_fpca
            trajectories[stream_name][corruption]["AO-FP-CA"] = traj_ao_fpca
            
    # Output latex-formatted results summary table
    print("\n--- LATEX TABLE FORMATTED RESULTS ---")
    for stream_name in ["alternating", "sequential"]:
        print(f"\nStream: {stream_name.capitalize()}")
        print("Method & Clean & Gaussian Noise & Contrast Shift \\\\")
        print("\\hline")
        for method in ["Static", "AdaMerging", "LFWA", "AO-LFWA", "FP-CA", "AO-FP-CA"]:
            c_clean = results[stream_name]["clean"][method] * 100
            c_noise = results[stream_name]["gaussian_noise"][method] * 100
            c_contrast = results[stream_name]["contrast_shift"][method] * 100
            print(f"{method} & {c_clean:.2f}\\% & {c_noise:.2f}\\% & {c_contrast:.2f}\\% \\\\")
            
    # Save Trajectory Plot
    print("\nPlotting coefficient trajectories for sequential stream clean...")
    # Plot traj for sequential stream clean: CPA-Merge (which is FP-CA) vs our AO-FP-CA
    # traj is list of shape (150, 3) representing the average lambda_mnist, lambda_fmnist, lambda_kmnist
    t_fpca = np.array(trajectories["sequential"]["clean"]["FP-CA"])
    t_ao_fpca = np.array(trajectories["sequential"]["clean"]["AO-FP-CA"])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_fpca[:, 0], label="$\lambda_{MNIST}$", color="red")
    plt.plot(t_fpca[:, 1], label="$\lambda_{FashionMNIST}$", color="green")
    plt.plot(t_fpca[:, 2], label="$\lambda_{KMNIST}$", color="blue")
    plt.axvline(x=50, color="gray", linestyle="--")
    plt.axvline(x=100, color="gray", linestyle="--")
    plt.title("FP-CA (Offline Fisher) Trajectory")
    plt.xlabel("Test Batch Index")
    plt.ylabel("Average Merging Coefficient")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(t_ao_fpca[:, 0], label="$\lambda_{MNIST}$", color="red")
    plt.plot(t_ao_fpca[:, 1], label="$\lambda_{FashionMNIST}$", color="green")
    plt.plot(t_ao_fpca[:, 2], label="$\lambda_{KMNIST}$", color="blue")
    plt.axvline(x=50, color="gray", linestyle="--")
    plt.axvline(x=100, color="gray", linestyle="--")
    plt.title("AO-FP-CA (Proposed Online Fisher) Trajectory")
    plt.xlabel("Test Batch Index")
    plt.ylabel("Average Merging Coefficient")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("coefficient_trajectories.png", dpi=300)
    print("Saved coefficient_trajectories.png")
    
    # Run hyperparameter sweeps
    run_sweeps(evaluator, streams)

def run_sweeps(evaluator, streams):
    print("\n--- RUNNING HYPERPARAMETER SWEEPS ---")
    alternating_stream = streams["alternating"]
    
    # 1. Sweep gamma_fish in [0.01, 0.05, 0.1, 0.3, 0.5] (holding alpha = 1.0)
    gamma_vals = [0.01, 0.05, 0.1, 0.3, 0.5]
    gamma_results = {g: {} for g in gamma_vals}
    for g in gamma_vals:
        print(f"Sweeping gamma_fish = {g}...")
        for corruption in ["clean", "gaussian_noise", "contrast_shift"]:
            acc, _ = evaluator.evaluate_ao_fpca(alternating_stream, corruption, alpha=1.0, gamma_fish=g)
            gamma_results[g][corruption] = acc
            
    print("\nSweep Gamma_fish Results Table (LaTeX):")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("$\\gamma_{\\text{fish}}$ & Clean & Gaussian Noise & Contrast Shift \\\\")
    print("\\midrule")
    for g in gamma_vals:
        c_clean = gamma_results[g]["clean"] * 100
        c_noise = gamma_results[g]["gaussian_noise"] * 100
        c_contrast = gamma_results[g]["contrast_shift"] * 100
        print(f"{g} & {c_clean:.2f}\\% & {c_noise:.2f}\\% & {c_contrast:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # 2. Sweep alpha in [0.0, 0.5, 1.0, 1.5] (holding gamma_fish = 0.1)
    alpha_vals = [0.0, 0.5, 1.0, 1.5]
    alpha_results = {a: {} for a in alpha_vals}
    for a in alpha_vals:
        print(f"Sweeping alpha = {a}...")
        for corruption in ["clean", "gaussian_noise", "contrast_shift"]:
            acc, _ = evaluator.evaluate_ao_fpca(alternating_stream, corruption, alpha=a, gamma_fish=0.1)
            alpha_results[a][corruption] = acc
            
    print("\nSweep Alpha Results Table (LaTeX):")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("$\\alpha$ & Clean & Gaussian Noise & Contrast Shift \\\\")
    print("\\midrule")
    for a in alpha_vals:
        c_clean = alpha_results[a]["clean"] * 100
        c_noise = alpha_results[a]["gaussian_noise"] * 100
        c_contrast = alpha_results[a]["contrast_shift"] * 100
        print(f"{a:.1f} & {c_clean:.2f}\\% & {c_noise:.2f}\\% & {c_contrast:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
if __name__ == "__main__":
    run_experiment_suite()
