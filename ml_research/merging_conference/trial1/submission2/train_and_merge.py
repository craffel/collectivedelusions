import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, GaussianBlur, ColorJitter
import open_clip
import numpy as np
from torch.func import functional_call
import argparse

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on cluster nodes.")

# Helper class for SAM (Sharpness-Aware Minimization)
class SAM_Optimizer:
    def __init__(self, params, base_optimizer, rho=0.05, parameter_wise=False, selective=False):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.parameter_wise = parameter_wise
        self.selective = selective

    @torch.no_grad()
    def first_step(self):
        if self.parameter_wise:
            # Parameter-wise perturbation (each tensor is scaled by its own gradient norm)
            for p in self.params:
                if p.grad is None:
                    continue
                p.old_p = p.data.clone()
                if self.selective and p.numel() <= 10:
                    continue
                grad_norm = p.grad.norm(2)
                scale = self.rho / (grad_norm + 1e-12)
                e_w = p.grad * scale
                p.add_(e_w)
        else:
            # Calculate global norm of gradients across all parameters
            params_to_perturb = []
            grad_norms = []
            for p in self.params:
                if p.grad is None:
                    continue
                if self.selective and p.numel() <= 10:
                    p.old_p = p.data.clone()
                    continue
                params_to_perturb.append(p)
                grad_norms.append(p.grad.norm(2))
                
            if len(grad_norms) > 0:
                grad_norm = torch.norm(torch.stack(grad_norms), 2)
                scale = self.rho / (grad_norm + 1e-12)
                for p in params_to_perturb:
                    p.old_p = p.data.clone()
                    e_w = p.grad * scale
                    p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data.copy_(p.old_p)
        self.base_optimizer.step()

# Load CLIP model and text encoder
def get_model_and_tokenizer():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, tokenizer

# Classification templates and zero-shot heads
def create_zero_shot_heads(model, tokenizer):
    task_classes = {
        "mnist": [str(i) for i in range(10)],
        "fashion": ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
        "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    }
    
    heads = {}
    with torch.no_grad():
        for task, classes in task_classes.items():
            templates = [f"a photo of a {c}" for c in classes]
            tokens = tokenizer(templates).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            heads[task] = text_features.clone()
    return heads

# Add corruption to images for robustness evaluation
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# Build datasets
def get_datasets():
    # Base transforms
    transform_clean = Compose([
        Resize((224, 224)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    transform_mnist_corrupt = Compose([
        Resize((224, 224)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        AddGaussianNoise(0., 0.3)
    ])
    
    transform_fashion_corrupt = Compose([
        Resize((224, 224)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        GaussianBlur(kernel_size=15, sigma=4.0)
    ])
    
    transform_cifar_corrupt = Compose([
        Resize((224, 224)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ColorJitter(brightness=0.8, contrast=0.8)
    ])
    
    datasets = {}
    
    # MNIST
    mnist_train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_clean)
    mnist_test_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_clean)
    mnist_test_corr = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist_corrupt)
    
    # FashionMNIST
    fashion_train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_clean)
    fashion_test_full = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_clean)
    fashion_test_corr = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_fashion_corrupt)
    
    # CIFAR10
    cifar_train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_clean)
    cifar_test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_clean)
    cifar_test_corr = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar_corrupt)
    
    # Subsets to make it extremely fast
    np.random.seed(42)
    mnist_train_idx = np.random.choice(len(mnist_train_full), 5000, replace=False)
    mnist_cal_idx = np.random.choice(len(mnist_test_full), 200, replace=False)
    mnist_test_idx = np.random.choice(len(mnist_test_full), 1000, replace=False)
    
    fashion_train_idx = np.random.choice(len(fashion_train_full), 5000, replace=False)
    fashion_cal_idx = np.random.choice(len(fashion_test_full), 200, replace=False)
    fashion_test_idx = np.random.choice(len(fashion_test_full), 1000, replace=False)
    
    cifar_train_idx = np.random.choice(len(cifar_train_full), 5000, replace=False)
    cifar_cal_idx = np.random.choice(len(cifar_test_full), 200, replace=False)
    cifar_test_idx = np.random.choice(len(cifar_test_full), 1000, replace=False)
    
    datasets["mnist"] = {
        "train": Subset(mnist_train_full, mnist_train_idx),
        "cal": Subset(mnist_test_full, mnist_cal_idx),
        "test_clean": Subset(mnist_test_full, mnist_test_idx),
        "test_corr": Subset(mnist_test_corr, mnist_test_idx),
    }
    
    datasets["fashion"] = {
        "train": Subset(fashion_train_full, fashion_train_idx),
        "cal": Subset(fashion_test_full, fashion_cal_idx),
        "test_clean": Subset(fashion_test_full, fashion_test_idx),
        "test_corr": Subset(fashion_test_corr, fashion_test_idx),
    }
    
    datasets["cifar10"] = {
        "train": Subset(cifar_train_full, cifar_train_idx),
        "cal": Subset(cifar_test_full, cifar_cal_idx),
        "test_clean": Subset(cifar_test_full, cifar_test_idx),
        "test_corr": Subset(cifar_test_corr, cifar_test_idx),
    }
    
    return datasets

# Evaluate a model on a loader
def evaluate_model(model, loader, text_head):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_head.T)
            preds = similarity.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Train an expert model on a task
def train_expert(task, train_loader, val_loader, text_head):
    print(f"\n--- Training Expert for {task} ---")
    model, _ = get_model_and_tokenizer()
    model = model.to(device)
    
    # Freeze text encoder
    for param in model.transformer.parameters():
        param.requires_grad = False
    if hasattr(model, 'text_projection') and model.text_projection is not None:
        model.text_projection.requires_grad = False
        
    # We fine-tune the image encoder
    for name, param in model.visual.named_parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.visual.parameters()), lr=1e-5, weight_decay=0.1)
    
    # Clean validation accuracy before training
    val_acc_before = evaluate_model(model, val_loader, text_head)
    print(f"Zero-shot validation accuracy: {val_acc_before:.4f}")
    
    for epoch in range(2):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_head.T)
            
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch+1} - Loss: {total_loss/total:.4f}, Train Acc: {correct/total:.4f}")
        
    val_acc_after = evaluate_model(model, val_loader, text_head)
    print(f"Fine-tuned validation accuracy: {val_acc_after:.4f}")
    return model.visual.state_dict()

# Construct the visual params dictionary for functional_call
def get_visual_params_dict(visual_model, pretrained_visual_state, task_vectors, coeffs, task_projs=None, active_task=None):
    params_dict = {**dict(visual_model.named_parameters()), **dict(visual_model.named_buffers())}
    
    for name in pretrained_visual_state.keys():
        if task_projs is not None and active_task is not None and name == 'proj':
            params_dict[name] = task_projs[active_task]
        else:
            val = pretrained_visual_state[name].to(device)
            merged_val = val
            for task_idx, (task_name, tv) in enumerate(task_vectors.items()):
                if name in tv:
                    c = coeffs[task_idx] if isinstance(coeffs, (list, tuple, np.ndarray, torch.Tensor)) else coeffs
                    merged_val = merged_val + c * tv[name].to(device)
            params_dict[name] = merged_val
            
    return params_dict

# Get model with merged weights (loads weights into _shared_model in place)
_shared_model = None
def get_merged_visual_model(pretrained_state, task_vectors, coeffs, task_proj_states=None, active_task=None):
    global _shared_model
    if _shared_model is None:
        raise ValueError("Global _shared_model is not set. Initialize it in main().")
        
    # Compute merged state dict
    merged_state = copy.deepcopy(pretrained_state)
    for name in merged_state.keys():
        if task_proj_states is not None and name == 'proj':
            continue
            
        # Add task vectors weighted by coefficients
        for task_idx, (task_name, tv) in enumerate(task_vectors.items()):
            if name in tv:
                c = coeffs[task_idx] if isinstance(coeffs, (list, tuple, np.ndarray, torch.Tensor)) else coeffs
                merged_state[name] = merged_state[name].to(device) + c * tv[name].to(device)
                
    # Load merged state into model's visual encoder
    _shared_model.visual.load_state_dict(merged_state, strict=False)
    
    # Load task-specific proj if available
    if task_proj_states is not None and active_task is not None:
        _shared_model.visual.proj.data.copy_(task_proj_states[active_task].to(device))
        
    return _shared_model

# Main experimental run
def main():
    global _shared_model
    
    parser = argparse.ArgumentParser(description="U-SASLA model merging experiments")
    parser.add_argument("--rho", type=float, default=0.05, help="SAM rho parameter")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of calibration steps")
    parser.add_argument("--output_file", type=str, default="experiment_results.json", help="Output file path")
    parser.add_argument("--selective_sam", action="store_true", help="Apply SAM only to projection layers, keeping merging coeffs un-perturbed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Set seed to {args.seed}. Selective SAM is: {args.selective_sam}")
    print("Preparing datasets...")
    datasets = get_datasets()
    
    # Load base model
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    _shared_model = model  # Cache globally for in-place loading
    
    pretrained_visual_state = copy.deepcopy(model.visual.state_dict())
    
    # Save pretrained weights to file
    torch.save(pretrained_visual_state, "pretrained_clip.pt")
    
    # Create text heads
    text_heads = create_zero_shot_heads(model, tokenizer)
    
    tasks = ["mnist", "fashion", "cifar10"]
    expert_states = {}
    
    # Check if expert weights already exist, otherwise train them
    for t in tasks:
        weight_path = f"expert_{t}.pt"
        train_loader = DataLoader(datasets[t]["train"], batch_size=128, shuffle=True)
        val_loader = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        
        if os.path.exists(weight_path):
            print(f"Loading pre-trained expert for {t} from {weight_path}")
            expert_states[t] = torch.load(weight_path, map_location=device)
        else:
            state = train_expert(t, train_loader, val_loader, text_heads[t])
            torch.save(state, weight_path)
            expert_states[t] = state

    # Create task vectors (for visual encoder parameters)
    task_vectors = {}
    for t in tasks:
        tv = {}
        for name in pretrained_visual_state.keys():
            if name in expert_states[t]:
                tv[name] = expert_states[t][name] - pretrained_visual_state[name]
        task_vectors[t] = tv

    # Evaluate zero-shot base model
    print("\nEvaluating Pretrained Base Model...")
    base_accs_clean = {}
    base_accs_corr = {}
    for t in tasks:
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        base_accs_clean[t] = evaluate_model(model, loader_clean, text_heads[t])
        base_accs_corr[t] = evaluate_model(model, loader_corr, text_heads[t])
        print(f"Task {t} - Clean: {base_accs_clean[t]:.4f}, Corrupted: {base_accs_corr[t]:.4f}")

    # Evaluate individual experts (as upper bounds)
    print("\nEvaluating Individual Expert Models...")
    expert_accs_clean = {}
    expert_accs_corr = {}
    for t in tasks:
        expert_model = get_merged_visual_model(pretrained_visual_state, {t: task_vectors[t]}, [1.0])
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        expert_accs_clean[t] = evaluate_model(expert_model, loader_clean, text_heads[t])
        expert_accs_corr[t] = evaluate_model(expert_model, loader_corr, text_heads[t])
        print(f"Expert {t} - Clean: {expert_accs_clean[t]:.4f}, Corrupted: {expert_accs_corr[t]:.4f}")

    # Method 1: Task Arithmetic (with alpha=0.3)
    print("\nEvaluating Task Arithmetic (alpha=0.3)...")
    ta_model = get_merged_visual_model(pretrained_visual_state, task_vectors, [0.3, 0.3, 0.3])
    ta_accs_clean = {}
    ta_accs_corr = {}
    for t in tasks:
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        ta_accs_clean[t] = evaluate_model(ta_model, loader_clean, text_heads[t])
        ta_accs_corr[t] = evaluate_model(ta_model, loader_corr, text_heads[t])
        print(f"TA {t} - Clean: {ta_accs_clean[t]:.4f}, Corrupted: {ta_accs_corr[t]:.4f}")

    # Prepare Calibration Data Loaders for test-time adaptive methods
    cal_loaders = {t: DataLoader(datasets[t]["cal"], batch_size=32, shuffle=True) for t in tasks}

    # Helper function for self-labeled optimization (SyMerge style vs U-SASLA) using functional_call
    def run_calibration(use_sam=False, parameter_wise=False, selective=False, num_steps=40, rho=0.05, lr=2e-3):
        # Initialize coefficients: 0.3 for each task
        coeffs = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
        
        # Initialize task-specific projection layers with experts' proj layers
        task_proj_states = {t: copy.deepcopy(expert_states[t]['proj']).to(device) for t in tasks}
        task_projs = {t: nn.Parameter(copy.deepcopy(task_proj_states[t])) for t in tasks}
        
        # Trainable parameters
        params_to_opt = [coeffs] + list(task_projs.values())
        optimizer = torch.optim.AdamW(params_to_opt, lr=lr)
        
        if use_sam:
            sam_opt = SAM_Optimizer(params_to_opt, optimizer, rho=rho, parameter_wise=parameter_wise, selective=selective)
            
        # Optimization loop
        for step in range(num_steps):
            # We cycle through tasks
            for task_idx, t in enumerate(tasks):
                # Get a calibration batch
                images, _ = next(iter(cal_loaders[t]))
                images = images.to(device)
                
                # Get expert soft labels using functional_call
                expert_params = get_visual_params_dict(
                    model.visual, pretrained_visual_state, {t: task_vectors[t]}, torch.tensor([1.0], device=device)
                )
                with torch.no_grad():
                    expert_feats = functional_call(model.visual, expert_params, (images,))
                    expert_feats = expert_feats / expert_feats.norm(dim=-1, keepdim=True)
                    expert_logits = 100.0 * expert_feats @ text_heads[t].T
                    expert_probs = F.softmax(expert_logits, dim=-1)
                
                # Forward pass function for merged model
                def compute_loss(c_values, p_dict):
                    merged_params = get_visual_params_dict(
                        model.visual, pretrained_visual_state, task_vectors, c_values, p_dict, active_task=t
                    )
                    merged_feats = functional_call(model.visual, merged_params, (images,))
                    merged_feats = merged_feats / merged_feats.norm(dim=-1, keepdim=True)
                    merged_logits = 100.0 * merged_feats @ text_heads[t].T
                    merged_log_probs = F.log_softmax(merged_logits, dim=-1)
                    
                    # KL Divergence / Distillation loss
                    loss = F.kl_div(merged_log_probs, expert_probs, reduction='batchmean')
                    return loss

                if use_sam:
                    # SAM Step 1
                    optimizer.zero_grad()
                    loss = compute_loss(coeffs, task_projs)
                    loss.backward()
                    sam_opt.first_step()
                    
                    # SAM Step 2
                    optimizer.zero_grad()
                    loss_perturbed = compute_loss(coeffs, task_projs)
                    loss_perturbed.backward()
                    sam_opt.second_step()
                else:
                    # Standard gradient step
                    optimizer.zero_grad()
                    loss = compute_loss(coeffs, task_projs)
                    loss.backward()
                    optimizer.step()
                    
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps} - Coeffs: {coeffs.detach().cpu().numpy().tolist()}")
                
        # Detach and save optimized weights
        final_coeffs = coeffs.detach().clone()
        final_projs = {t: task_projs[t].detach().clone() for t in tasks}
        return final_coeffs, final_projs

    # Method 2: SyMerge Calibration
    print("\nRunning SyMerge Calibration (Standard self-labeling adaptation)...")
    symerge_coeffs, symerge_projs = run_calibration(use_sam=False, num_steps=args.num_steps, lr=args.lr)
    print(f"SyMerge Optimized Coefficients: {symerge_coeffs.cpu().numpy().tolist()}")

    # Evaluate SyMerge
    print("\nEvaluating SyMerge...")
    symerge_accs_clean = {}
    symerge_accs_corr = {}
    for t in tasks:
        # Create task-specific merged model for evaluation
        sy_model = get_merged_visual_model(
            pretrained_visual_state, task_vectors, symerge_coeffs, symerge_projs, active_task=t
        )
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        symerge_accs_clean[t] = evaluate_model(sy_model, loader_clean, text_heads[t])
        symerge_accs_corr[t] = evaluate_model(sy_model, loader_corr, text_heads[t])
        print(f"SyMerge {t} - Clean: {symerge_accs_clean[t]:.4f}, Corrupted: {symerge_accs_corr[t]:.4f}")

    # Method 3: Our U-SASLA Calibration (Sharpness-Aware Single-Layer Adaptation - Global SAM)
    print(f"\nRunning U-SASLA Calibration (Sharpness-Aware adaptation) with rho={args.rho}, lr={args.lr}...")
    sasla_coeffs, sasla_projs = run_calibration(use_sam=True, parameter_wise=False, selective=args.selective_sam, rho=args.rho, lr=args.lr, num_steps=args.num_steps)
    print(f"U-SASLA Optimized Coefficients: {sasla_coeffs.cpu().numpy().tolist()}")

    # Evaluate U-SASLA
    print("\nEvaluating U-SASLA...")
    sasla_accs_clean = {}
    sasla_accs_corr = {}
    for t in tasks:
        # Create task-specific merged model for evaluation
        sasla_model = get_merged_visual_model(
            pretrained_visual_state, task_vectors, sasla_coeffs, sasla_projs, active_task=t
        )
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        sasla_accs_clean[t] = evaluate_model(sasla_model, loader_clean, text_heads[t])
        sasla_accs_corr[t] = evaluate_model(sasla_model, loader_corr, text_heads[t])
        print(f"U-SASLA {t} - Clean: {sasla_accs_clean[t]:.4f}, Corrupted: {sasla_accs_corr[t]:.4f}")

    # Method 3b: Our PW-SASLA Calibration (Parameter-wise Sharpness-Aware Single-Layer Adaptation)
    print(f"\nRunning PW-SASLA Calibration (Parameter-wise Sharpness-Aware adaptation) with rho={args.rho}, lr={args.lr}...")
    pwsasla_coeffs, pwsasla_projs = run_calibration(use_sam=True, parameter_wise=True, selective=args.selective_sam, rho=args.rho, lr=args.lr, num_steps=args.num_steps)
    print(f"PW-SASLA Optimized Coefficients: {pwsasla_coeffs.cpu().numpy().tolist()}")

    # Evaluate PW-SASLA
    print("\nEvaluating PW-SASLA...")
    pwsasla_accs_clean = {}
    pwsasla_accs_corr = {}
    for t in tasks:
        # Create task-specific merged model for evaluation
        pwsasla_model = get_merged_visual_model(
            pretrained_visual_state, task_vectors, pwsasla_coeffs, pwsasla_projs, active_task=t
        )
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        pwsasla_accs_clean[t] = evaluate_model(pwsasla_model, loader_clean, text_heads[t])
        pwsasla_accs_corr[t] = evaluate_model(pwsasla_model, loader_corr, text_heads[t])
        print(f"PW-SASLA {t} - Clean: {pwsasla_accs_clean[t]:.4f}, Corrupted: {pwsasla_accs_corr[t]:.4f}")

    # Method 4: AdaMerging (learning only coefficients, keeping projection layers frozen)
    print("\nRunning AdaMerging Calibration (Learning only coefficients)...")
    def run_adamerging_cal(num_steps=40, lr=2e-3):
        coeffs = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
        optimizer = torch.optim.AdamW([coeffs], lr=lr)
        
        for step in range(num_steps):
            for task_idx, t in enumerate(tasks):
                images, _ = next(iter(cal_loaders[t]))
                images = images.to(device)
                
                # Get expert soft labels using functional_call
                expert_params = get_visual_params_dict(
                    model.visual, pretrained_visual_state, {t: task_vectors[t]}, torch.tensor([1.0], device=device)
                )
                with torch.no_grad():
                    expert_feats = functional_call(model.visual, expert_params, (images,))
                    expert_feats = expert_feats / expert_feats.norm(dim=-1, keepdim=True)
                    expert_logits = 100.0 * expert_feats @ text_heads[t].T
                    expert_probs = F.softmax(expert_logits, dim=-1)
                
                optimizer.zero_grad()
                # Merged parameters (without task-specific projs)
                merged_params = get_visual_params_dict(
                    model.visual, pretrained_visual_state, task_vectors, coeffs
                )
                merged_feats = functional_call(model.visual, merged_params, (images,))
                merged_feats = merged_feats / merged_feats.norm(dim=-1, keepdim=True)
                merged_logits = 100.0 * merged_feats @ text_heads[t].T
                merged_log_probs = F.log_softmax(merged_logits, dim=-1)
                
                loss = F.kl_div(merged_log_probs, expert_probs, reduction='batchmean')
                loss.backward()
                optimizer.step()
        return coeffs.detach().clone()

    adamerging_coeffs = run_adamerging_cal(num_steps=args.num_steps, lr=args.lr)
    print(f"AdaMerging Optimized Coefficients: {adamerging_coeffs.cpu().numpy().tolist()}")

    print("\nEvaluating AdaMerging...")
    adamerging_accs_clean = {}
    adamerging_accs_corr = {}
    for t in tasks:
        ada_model = get_merged_visual_model(pretrained_visual_state, task_vectors, adamerging_coeffs)
        loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
        loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
        adamerging_accs_clean[t] = evaluate_model(ada_model, loader_clean, text_heads[t])
        adamerging_accs_corr[t] = evaluate_model(ada_model, loader_corr, text_heads[t])
        print(f"AdaMerging {t} - Clean: {adamerging_accs_clean[t]:.4f}, Corrupted: {adamerging_accs_corr[t]:.4f}")

    # Compile all results
    results = {
        "tasks": tasks,
        "pretrained": {"clean": base_accs_clean, "corrupted": base_accs_corr},
        "expert": {"clean": expert_accs_clean, "corrupted": expert_accs_corr},
        "task_arithmetic": {"clean": ta_accs_clean, "corrupted": ta_accs_corr},
        "adamerging": {"clean": adamerging_accs_clean, "corrupted": adamerging_accs_corr},
        "symerge": {"clean": symerge_accs_clean, "corrupted": symerge_accs_corr},
        "u_sasla": {"clean": sasla_accs_clean, "corrupted": sasla_accs_corr},
        "pw_sasla": {"clean": pwsasla_accs_clean, "corrupted": pwsasla_accs_corr}
    }
    
    # Save results to json
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll experiments complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
