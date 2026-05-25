import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn.bert_padding'] = None
sys.modules['flash_attn_2_cuda'] = None

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

def get_dataloaders(subset_size=10000, batch_size=64, seed=42):
    # Transforms for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transforms for test
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Loading datasets...")
    # CIFAR-10
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # SVHN
    svhn_train_full = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
    svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
    
    # Get subsets for fast training
    g = torch.Generator().manual_seed(seed)
    
    cifar_indices = torch.randperm(len(cifar_train_full), generator=g)[:subset_size].tolist()
    cifar_train = Subset(cifar_train_full, cifar_indices)
    
    svhn_indices = torch.randperm(len(svhn_train_full), generator=g)[:subset_size].tolist()
    svhn_train = Subset(svhn_train_full, svhn_indices)
    
    cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
    cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
    
    svhn_train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=2)
    svhn_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Datasets configured: CIFAR-10 train subset={len(cifar_train)}, test={len(cifar_test)}")
    print(f"Datasets configured: SVHN train subset={len(svhn_train)}, test={len(svhn_test)}")
    
    return cifar_train_loader, cifar_test_loader, svhn_train_loader, svhn_test_loader

def train_expert(task_name, train_loader, test_loader, epochs=3, lr=5e-4, device='cuda'):
    print(f"\n--- Training Expert for {task_name} ---")
    # Load base ViT
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)
    
    # Configure LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save the best weights
            # We want to save both the LoRA weights and the classification head weights
            torch.save(model.state_dict(), f"{task_name}_best_model.pt")
            
    print(f"Finished training {task_name}. Best Test Acc: {best_acc:.2f}%")
    return best_acc

def extract_lora_and_head(state_dict):
    """
    Extracts only the LoRA weights and the classification head weights from the state dict.
    """
    lora_dict = {}
    head_dict = {}
    for k, v in state_dict.items():
        if "lora_" in k:
            lora_dict[k] = v.cpu()
        elif "classifier" in k:
            clean_key = k.replace("base_model.model.", "")
            head_dict[clean_key] = v.cpu()
    return lora_dict, head_dict

def get_base_model_and_params(device='cuda'):
    """
    Loads the clean base ViT model (without PEFT wrapper),
    and returns its parameter and buffer state dictionary.
    """
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)
    model.eval()
    model.to(device)
    base_params = {k: v.clone() for k, v in model.named_parameters()}
    base_buffers = {k: v.clone() for k, v in model.named_buffers()}
    return model, base_params, base_buffers

def construct_merged_weights(base_params, lora1, head1, lora2, head2, coefficients, current_task=1):
    """
    Constructs the merged model parameters differentiably w.r.t the layer-wise coefficients.
    coefficients: tensor of shape (24,) containing the layer-wise merging coefficients.
    current_task: 1 for Task 1 (CIFAR-10), 2 for Task 2 (SVHN) to determine which classification head to use.
    """
    merged_params = {}
    # 1. Start with the base model parameters
    for k, v in base_params.items():
        if k == 'classifier.weight':
            # Use the merged or selected classification head
            merged_params[k] = head1['classifier.weight'].to(v.device) if current_task == 1 else head2['classifier.weight'].to(v.device)
        elif k == 'classifier.bias':
            merged_params[k] = head1['classifier.bias'].to(v.device) if current_task == 1 else head2['classifier.bias'].to(v.device)
        else:
            merged_params[k] = v.clone()

    # 2. Add the dynamic merged LoRA weights
    # We have 12 transformer layers, each has attention.attention.query and attention.attention.value.
    # Total of 24 layers with LoRA.
    # The coefficients tensor has length 24.
    
    # Let's map parameter names to their corresponding LoRA keys and coefficient index.
    coeff_idx = 0
    alpha = 16
    r = 8
    scale = alpha / r
    
    for layer in range(12):
        for module in ["query", "value"]:
            # Base parameter key: e.g. "vit.encoder.layer.0.attention.attention.query.weight"
            weight_key = f"vit.encoder.layer.{layer}.attention.attention.{module}.weight"
            
            # LoRA parameter keys:
            # lora_A: e.g. "base_model.model.vit.encoder.layer.0.attention.attention.query.lora_A.default.weight"
            # lora_B: e.g. "base_model.model.vit.encoder.layer.0.attention.attention.query.lora_B.default.weight"
            lora_A_key = f"base_model.model.vit.encoder.layer.{layer}.attention.attention.{module}.lora_A.default.weight"
            lora_B_key = f"base_model.model.vit.encoder.layer.{layer}.attention.attention.{module}.lora_B.default.weight"
            
            if lora_A_key in lora1 and lora_B_key in lora1:
                # Expert 1 LoRA matrices
                A1 = lora1[lora_A_key].to(base_params[weight_key].device)
                B1 = lora1[lora_B_key].to(base_params[weight_key].device)
                
                # Expert 2 LoRA matrices
                A2 = lora2[lora_A_key].to(base_params[weight_key].device)
                B2 = lora2[lora_B_key].to(base_params[weight_key].device)
                
                # Get the coefficient for this specific module
                lam = coefficients[coeff_idx]
                
                # Compute the weight delta
                delta_W1 = B1 @ A1
                delta_W2 = B2 @ A2
                merged_delta = scale * (lam * delta_W1 + (1.0 - lam) * delta_W2)
                
                # Add to the base weight
                merged_params[weight_key] = base_params[weight_key] + merged_delta
                
                coeff_idx += 1
                
    return merged_params

def evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, coefficients, test_loader, task_idx=1, device='cuda'):
    base_model.eval()
    correct = 0
    total = 0
    
    # Construct the state dict dynamically
    with torch.no_grad():
        merged_params = construct_merged_weights(base_params, lora1, head1, lora2, head2, coefficients, current_task=task_idx)
        all_state = {**merged_params, **base_buffers}
        
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = functional_call(base_model, all_state, (images,)).logits
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

def compute_entropy_loss(logits):
    p = F.softmax(logits, dim=-1)
    entropy = - (p * torch.log(p + 1e-12)).sum(dim=-1).mean()
    return entropy

def compute_sosr_loss(coefficients, lora1, lora2):
    """
    Computes Soft-Orthogonality Spectral Regularization (SOSR) penalty
    on the merged LoRA adapters to ensure low-rank alignment.
    """
    loss = 0.0
    coeff_idx = 0
    epsilon = 1e-6
    for layer in range(12):
        for module in ["query", "value"]:
            lora_A_key = f"base_model.model.vit.encoder.layer.{layer}.attention.attention.{module}.lora_A.default.weight"
            lora_B_key = f"base_model.model.vit.encoder.layer.{layer}.attention.attention.{module}.lora_B.default.weight"
            
            if lora_A_key in lora1 and lora_B_key in lora1:
                A1 = lora1[lora_A_key].cuda()
                B1 = lora1[lora_B_key].cuda()
                A2 = lora2[lora_A_key].cuda()
                B2 = lora2[lora_B_key].cuda()
                
                lam = coefficients[coeff_idx]
                
                # Construct merged matrices
                A_merged = lam * A1 + (1.0 - lam) * A2
                B_merged = lam * B1 + (1.0 - lam) * B2
                
                # BT * B
                BTB = B_merged.t() @ B_merged
                diag_BTB = torch.diag(BTB)
                BTB_off = BTB - torch.diag_embed(diag_BTB)
                sosr_B = BTB_off.pow(2).sum() / (diag_BTB.pow(2).sum() + epsilon)
                
                # A * AT
                AAT = A_merged @ A_merged.t()
                diag_AAT = torch.diag(AAT)
                AAT_off = AAT - torch.diag_embed(diag_AAT)
                sosr_A = AAT_off.pow(2).sum() / (diag_AAT.pow(2).sum() + epsilon)
                
                loss += (sosr_B + sosr_A)
                coeff_idx += 1
    return loss / max(coeff_idx, 1)

def run_tta(base_model, base_params, base_buffers, lora1, head1, lora2, head2, 
            cifar_test_loader, svhn_test_loader, 
            steps=100, batch_size=32, lr=0.01, rho=0.05, use_sam=True, use_sosr=True, sosr_weight=0.1, stream_type='alternating', device='cuda'):
    
    print(f"\n--- Running Test-Time Adaptation (SATA-TTA: use_sam={use_sam}, use_sosr={use_sosr}, stream_type={stream_type}) ---")
    
    # Initialize merging coefficients: length 24, all initialized to 0.5
    coefficients = torch.tensor([0.5] * 24, device=device, requires_grad=True)
    
    # We will also adapt the task-specific classification heads!
    # Let's clone them so we don't mutate the originals
    head1_adapted = {k: v.clone().to(device).requires_grad_(True) for k, v in head1.items()}
    head2_adapted = {k: v.clone().to(device).requires_grad_(True) for k, v in head2.items()}
    
    adapted_params = [coefficients] + list(head1_adapted.values()) + list(head2_adapted.values())
    optimizer = torch.optim.Adam(adapted_params, lr=lr)
    
    # Prepare test streams (unlabeled test data)
    cifar_iter = iter(cifar_test_loader)
    svhn_iter = iter(svhn_test_loader)
    
    for step in range(steps):
        # Sample test stream batch.
        if stream_type == 'alternating':
            task_idx = 1 if step % 2 == 0 else 2
        elif stream_type == 'sequential':
            # first half is CIFAR-10, second half is SVHN
            task_idx = 1 if step < (steps // 2) else 2
        elif stream_type == 'skewed':
            # 80% CIFAR-10, 20% SVHN
            task_idx = 1 if (step % 5) < 4 else 2
        else:
            raise ValueError(f"Unknown stream_type: {stream_type}")
            
        loader_iter = cifar_iter if task_idx == 1 else svhn_iter
        
        try:
            images, _ = next(loader_iter)
        except StopIteration:
            # Refresh iterator if exhausted
            if task_idx == 1:
                cifar_iter = iter(cifar_test_loader)
                images, _ = next(cifar_iter)
            else:
                svhn_iter = iter(svhn_test_loader)
                images, _ = next(svhn_iter)
                
        images = images.to(device)
        
        # Helper function to compute the objective loss on a batch
        def get_loss(coeff_val, h1_val, h2_val):
            # Compute expert predictions as soft labels
            with torch.no_grad():
                expert_coeff = torch.tensor([1.0 if task_idx == 1 else 0.0] * 24, device=device)
                expert_params = construct_merged_weights(base_params, lora1, head1, lora2, head2, expert_coeff, current_task=task_idx)
                expert_state = {**expert_params, **base_buffers}
                expert_logits = functional_call(base_model, expert_state, (images,)).logits
                soft_labels = F.softmax(expert_logits, dim=-1)

            # Reconstruct merged weights statelessly
            merged_params = construct_merged_weights(base_params, lora1, h1_val, lora2, h2_val, coeff_val, current_task=task_idx)
            all_state = {**merged_params, **base_buffers}
            
            # Forward pass
            logits = functional_call(base_model, all_state, (images,)).logits
            
            # KL divergence loss (Self-labeling)
            log_p = F.log_softmax(logits, dim=-1)
            loss_sl = F.kl_div(log_p, soft_labels, reduction='batchmean')
            
            # Apply SOSR regularization if requested
            if use_sosr:
                loss_sosr = compute_sosr_loss(coeff_val, lora1, lora2)
                return loss_sl + sosr_weight * loss_sosr
            else:
                return loss_sl
            
        # Optimization update step
        if use_sam:
            # SATA-TTA Sharpness-Aware Step
            # 1. Compute original loss and gradients
            optimizer.zero_grad()
            loss = get_loss(coefficients, head1_adapted, head2_adapted)
            loss.backward()
            
            # 2. Compute perturbation and apply
            with torch.no_grad():
                # Compute global gradient norm of all adapted parameters
                grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in adapted_params if p.grad is not None) + 1e-12)
                perturbations = {}
                for p in adapted_params:
                    if p.grad is not None:
                        perturbations[p] = rho * p.grad / grad_norm
                        p.add_(perturbations[p])
            
            # 3. Compute perturbed loss and gradients
            optimizer.zero_grad()
            loss_perturbed = get_loss(coefficients, head1_adapted, head2_adapted)
            loss_perturbed.backward()
            
            # 4. Restore original parameters and update using perturbed gradients
            with torch.no_grad():
                for p in adapted_params:
                    if p in perturbations:
                        p.sub_(perturbations[p]) # restore
            
            # Step the optimizer
            optimizer.step()
            
        else:
            # Standard TTA Gradient Descent Step
            optimizer.zero_grad()
            loss = get_loss(coefficients, head1_adapted, head2_adapted)
            loss.backward()
            optimizer.step()
            
        if (step + 1) % 10 == 0:
            print(f"TTA Step {step+1}/{steps} | Loss: {loss.item():.4f} | Avg Coeff: {coefficients.mean().item():.3f}")
            
    # Return the final adapted coefficients and classification heads
    return coefficients.detach(), head1_adapted, head2_adapted

def main():
    parser = argparse.ArgumentParser(description="SATA-TTA: Spectral-Regularized Test-Time Adaptation for LoRA Merging")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate_static", "evaluate_tta"], help="Execution mode")
    parser.add_argument("--subset_size", type=int, default=10000, help="Number of train samples for experts")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Train epochs for experts")
    parser.add_argument("--lr_train", type=float, default=5e-4, help="Training learning rate")
    parser.add_argument("--lr_tta", type=float, default=0.01, help="TTA learning rate")
    parser.add_argument("--steps_tta", type=int, default=50, help="Number of TTA optimization steps")
    parser.add_argument("--rho_sam", type=float, default=0.05, help="SAM perturbation radius")
    parser.add_argument("--sosr_weight", type=float, default=0.1, help="SOSR regularization weight")
    parser.add_argument("--stream_type", type=str, default="alternating", choices=["alternating", "sequential", "skewed"], help="TTA stream type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cifar_train_loader, cifar_test_loader, svhn_train_loader, svhn_test_loader = get_dataloaders(
        subset_size=args.subset_size, batch_size=args.batch_size, seed=args.seed
    )
    
    if args.mode == "train":
        # Train CIFAR-10 expert
        train_expert("cifar10", cifar_train_loader, cifar_test_loader, epochs=args.epochs, lr=args.lr_train, device=device)
        # Train SVHN expert
        train_expert("svhn", svhn_train_loader, svhn_test_loader, epochs=args.epochs, lr=args.lr_train, device=device)
        
    elif args.mode == "evaluate_static":
        # Load the experts
        if not os.path.exists("cifar10_best_model.pt") or not os.path.exists("svhn_best_model.pt"):
            print("Error: Trained expert models not found! Run with '--mode train' first.")
            return
            
        print("\n=== Loading Expert Weights ===")
        cifar_state = torch.load("cifar10_best_model.pt", map_location="cpu")
        svhn_state = torch.load("svhn_best_model.pt", map_location="cpu")
        
        lora1, head1 = extract_lora_and_head(cifar_state)
        lora2, head2 = extract_lora_and_head(svhn_state)
        
        # Load base model
        base_model, base_params, base_buffers = get_base_model_and_params(device=device)
        
        # Evaluate Single Task Experts (Self check)
        print("\n=== Evaluating Individual Expert Performance (Direct Parameter Merging @ task-specific coefficients) ===")
        
        # 1. CIFAR-10 Expert on CIFAR-10 test
        c10_coeff = torch.tensor([1.0] * 24, device=device)
        c10_acc = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, c10_coeff, cifar_test_loader, task_idx=1, device=device)
        print(f"CIFAR-10 Expert Accuracy on CIFAR-10 test: {c10_acc:.2f}%")
        
        # 2. SVHN Expert on SVHN test
        svhn_coeff = torch.tensor([0.0] * 24, device=device)
        svhn_acc = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, svhn_coeff, svhn_test_loader, task_idx=2, device=device)
        print(f"SVHN Expert Accuracy on SVHN test: {svhn_acc:.2f}%")
        
        # Evaluate Statically Merged Models
        print("\n=== Evaluating Static Linear Merging (DPM) ===")
        for lam_val in [0.3, 0.5, 0.7]:
            coeff = torch.tensor([lam_val] * 24, device=device)
            acc1 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, coeff, cifar_test_loader, task_idx=1, device=device)
            acc2 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, coeff, svhn_test_loader, task_idx=2, device=device)
            avg_acc = (acc1 + acc2) / 2.0
            print(f"Lambda = {lam_val:.1f} | CIFAR-10 Acc: {acc1:.2f}% | SVHN Acc: {acc2:.2f}% | Multi-Task Avg: {avg_acc:.2f}%")
            
    elif args.mode == "evaluate_tta":
        # Load the experts
        if not os.path.exists("cifar10_best_model.pt") or not os.path.exists("svhn_best_model.pt"):
            print("Error: Trained expert models not found! Run with '--mode train' first.")
            return
            
        cifar_state = torch.load("cifar10_best_model.pt", map_location="cpu")
        svhn_state = torch.load("svhn_best_model.pt", map_location="cpu")
        
        lora1, head1 = extract_lora_and_head(cifar_state)
        lora2, head2 = extract_lora_and_head(svhn_state)
        
        # Load base model
        base_model, base_params, base_buffers = get_base_model_and_params(device=device)
        
        # 1. Static Base Merging (for direct comparison)
        static_coeff = torch.tensor([0.5] * 24, device=device)
        static_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, static_coeff, cifar_test_loader, task_idx=1, device=device)
        static_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, head1, lora2, head2, static_coeff, svhn_test_loader, task_idx=2, device=device)
        static_avg = (static_c10 + static_svhn) / 2.0
        print("\n=== Static Baseline (No Adaptation) ===")
        print(f"CIFAR-10 Acc: {static_c10:.2f}% | SVHN Acc: {static_svhn:.2f}% | Multi-Task Avg: {static_avg:.2f}%")
        
        # 2. Standard TTA (unsupervised prediction entropy minimization, no SAM, no SOSR)
        std_coeff, std_h1, std_h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=args.steps_tta, batch_size=args.batch_size, lr=args.lr_tta,
            use_sam=False, use_sosr=False, stream_type=args.stream_type, device=device
        )
        std_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, std_h1, lora2, std_h2, std_coeff, cifar_test_loader, task_idx=1, device=device)
        std_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, std_h1, lora2, std_h2, std_coeff, svhn_test_loader, task_idx=2, device=device)
        std_avg = (std_c10 + std_svhn) / 2.0
        print("\n=== Standard TTA Results ===")
        print(f"CIFAR-10 Acc: {std_c10:.2f}% | SVHN Acc: {std_svhn:.2f}% | Multi-Task Avg: {std_avg:.2f}%")
        
        # 3. SAM-Only TTA (Our proposed method: prediction entropy minimization + SAM, no SOSR)
        sam_coeff, sam_h1, sam_h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=args.steps_tta, batch_size=args.batch_size, lr=args.lr_tta,
            rho=args.rho_sam, use_sam=True, use_sosr=False, stream_type=args.stream_type, device=device
        )
        sam_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sam_h1, lora2, sam_h2, sam_coeff, cifar_test_loader, task_idx=1, device=device)
        sam_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sam_h1, lora2, sam_h2, sam_coeff, svhn_test_loader, task_idx=2, device=device)
        sam_avg = (sam_c10 + sam_svhn) / 2.0
        print("\n=== SAM-Only TTA Results ===")
        print(f"CIFAR-10 Acc: {sam_c10:.2f}% | SVHN Acc: {sam_svhn:.2f}% | Multi-Task Avg: {sam_avg:.2f}%")
        
        # 4. SATA-TTA (prediction entropy minimization + SAM + SOSR)
        sata_coeff, sata_h1, sata_h2 = run_tta(
            base_model, base_params, base_buffers, lora1, head1, lora2, head2,
            cifar_test_loader, svhn_test_loader,
            steps=args.steps_tta, batch_size=args.batch_size, lr=args.lr_tta,
            rho=args.rho_sam, use_sam=True, use_sosr=True, sosr_weight=args.sosr_weight, stream_type=args.stream_type, device=device
        )
        sata_c10 = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sata_h1, lora2, sata_h2, sata_coeff, cifar_test_loader, task_idx=1, device=device)
        sata_svhn = evaluate_merged_model(base_model, base_params, base_buffers, lora1, sata_h1, lora2, sata_h2, sata_coeff, svhn_test_loader, task_idx=2, device=device)
        sata_avg = (sata_c10 + sata_svhn) / 2.0
        print("\n=== SATA-TTA Results ===")
        print(f"CIFAR-10 Acc: {sata_c10:.2f}% | SVHN Acc: {sata_svhn:.2f}% | Multi-Task Avg: {sata_avg:.2f}%")
        
        # Print comparison table
        print("\n=== Final Multi-Task Merging Comparison ===")
        print(f"Method          | CIFAR-10 Acc | SVHN Acc | Multi-Task Avg |")
        print(f"----------------|--------------|----------|----------------|")
        print(f"Static Merged   | {static_c10:12.2f}% | {static_svhn:8.2f}% | {static_avg:14.2f}% |")
        print(f"Standard TTA    | {std_c10:12.2f}% | {std_svhn:8.2f}% | {std_avg:14.2f}% |")
        print(f"SAM-Only TTA    | {sam_c10:12.2f}% | {sam_svhn:8.2f}% | {sam_avg:14.2f}% |")
        print(f"SATA-TTA (Ours) | {sata_c10:12.2f}% | {sata_svhn:8.2f}% | {sata_avg:14.2f}% |")

if __name__ == "__main__":
    main()
