import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn_2_cuda'] = None

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model

# SAM implementation
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.e_w_dict = {}

    @torch.no_grad()
    def first_step(self):
        # Calculate grad norm
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in self.params if p.grad is not None])
        )
        # Apply perturbation
        for p in self.params:
            if p.grad is None:
                continue
            e_w = p.grad * (self.rho / (grad_norm + 1e-12))
            p.add_(e_w)  # Store perturbation
            self.e_w_dict[p] = e_w

    @torch.no_grad()
    def second_step(self):
        # Remove perturbation and restore original parameters
        for p in self.params:
            if p.grad is None:
                continue
            if p in self.e_w_dict:
                p.sub_(self.e_w_dict[p])
        self.base_optimizer.step()
        self.e_w_dict.clear()

# Isotropic Spectral Regularization (ISR) helper via Factor Orthogonalization
def compute_isr_loss(model, r=8):
    isr_loss = 0.0
    count = 0
    
    # Iterate through named parameters to find matching lora_A and lora_B weights
    params = dict(model.named_parameters())
    for name, param in params.items():
        if 'lora_B' in name and name.endswith('weight'):
            # Find the corresponding lora_A
            name_A = name.replace('lora_B', 'lora_A')
            if name_A in params:
                B = param         # Shape: [d_out, r]
                A = params[name_A] # Shape: [r, d_in]
                
                try:
                    # BtB and AAt should be close to scaled identity matrices
                    # This enforces that different rank-1 updates are orthogonal and have similar energy,
                    # resulting in a perfectly flat singular value spectrum for B @ A.
                    BtB = B.t() @ B # [r, r]
                    AAt = A @ A.t() # [r, r]
                    
                    scale_B = BtB.diag().mean()
                    target_B = scale_B * torch.eye(r, device=B.device)
                    loss_B = torch.sum((BtB - target_B) ** 2)
                    
                    scale_A = AAt.diag().mean()
                    target_A = scale_A * torch.eye(r, device=A.device)
                    loss_A = torch.sum((AAt - target_A) ** 2)
                    
                    # Normalize the loss relative to the scale to prevent numerical explosion
                    loss_B_norm = loss_B / (scale_B ** 2 + 1e-6)
                    loss_A_norm = loss_A / (scale_A ** 2 + 1e-6)
                    
                    isr_loss += (loss_B_norm + loss_A_norm)
                    count += 1
                except Exception as e:
                    continue
                    
    if count > 0:
        return isr_loss / count
    return torch.tensor(0.0, device=model.device)

# Soft Orthogonality Spectral Regularization (SOSR) helper
def compute_sosr_loss(model, r=8):
    sosr_loss = 0.0
    count = 0
    params = dict(model.named_parameters())
    for name, param in params.items():
        if 'lora_B' in name and name.endswith('weight'):
            name_A = name.replace('lora_B', 'lora_A')
            if name_A in params:
                B = param         # Shape: [d_out, r]
                A = params[name_A] # Shape: [r, d_in]
                
                try:
                    BtB = B.t() @ B # [r, r]
                    AAt = A @ A.t() # [r, r]
                    
                    # Compute off-diagonal elements
                    diag_B = torch.diag(BtB)
                    off_diag_B = BtB - torch.diag_embed(diag_B)
                    loss_B = torch.sum(off_diag_B ** 2)
                    norm_B = torch.sum(diag_B ** 2)
                    loss_B_norm = loss_B / (norm_B + 1e-6)
                    
                    diag_A = torch.diag(AAt)
                    off_diag_A = AAt - torch.diag_embed(diag_A)
                    loss_A = torch.sum(off_diag_A ** 2)
                    norm_A = torch.sum(diag_A ** 2)
                    loss_A_norm = loss_A / (norm_A + 1e-6)
                    
                    sosr_loss += (loss_B_norm + loss_A_norm)
                    count += 1
                except Exception as e:
                    continue
                    
    if count > 0:
        return sosr_loss / count
    return torch.tensor(0.0, device=model.device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT using LoRA with SATA-LR")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "svhn"])
    parser.add_argument("--mode", type=str, required=True, choices=["standard", "sam", "isr", "sata_lr", "isr_soft", "sata_lr_soft"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_isr", type=float, default=0.1)
    parser.add_argument("--rho_sam", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # Image preprocessing
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # Load dataset
    print(f"Loading {args.dataset.upper()}...")
    if args.dataset == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_labels = 10
    elif args.dataset == "svhn":
        train_set = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform)
        num_labels = 10
        
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize Pre-trained Vision Transformer
    print("Loading pre-trained ViT...")
    base_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # Apply LoRA via PEFT
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.print_trainable_parameters()
    
    # Base Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # SAM wrapper if needed
    if args.mode in ["sam", "sata_lr", "sata_lr_soft"]:
        sam_opt = SAM(trainable_params, base_optimizer, rho=args.rho_sam)
    else:
        sam_opt = None
        
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print(f"Starting training in {args.mode.upper()} mode...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            if args.mode in ["standard", "isr", "isr_soft"]:
                # Normal forward-backward
                base_optimizer.zero_grad()
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                # Add spectral regularization if ISR
                if args.mode == "isr":
                    isr_loss = compute_isr_loss(model, r=args.r)
                    loss = loss + args.lambda_isr * isr_loss
                elif args.mode == "isr_soft":
                    isr_loss = compute_sosr_loss(model, r=args.r)
                    loss = loss + args.lambda_isr * isr_loss
                    
                loss.backward()
                base_optimizer.step()
                
            elif args.mode in ["sam", "sata_lr", "sata_lr_soft"]:
                # SAM First Step
                base_optimizer.zero_grad()
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                if args.mode == "sata_lr":
                    isr_loss = compute_isr_loss(model, r=args.r)
                    loss = loss + args.lambda_isr * isr_loss
                elif args.mode == "sata_lr_soft":
                    isr_loss = compute_sosr_loss(model, r=args.r)
                    loss = loss + args.lambda_isr * isr_loss
                loss.backward()
                sam_opt.first_step()
                
                # SAM Second Step
                base_optimizer.zero_grad()
                outputs_perturbed = model(images).logits
                loss_perturbed = criterion(outputs_perturbed, labels)
                if args.mode == "sata_lr":
                    isr_loss_perturbed = compute_isr_loss(model, r=args.r)
                    loss_perturbed = loss_perturbed + args.lambda_isr * isr_loss_perturbed
                elif args.mode == "sata_lr_soft":
                    isr_loss_perturbed = compute_sosr_loss(model, r=args.r)
                    loss_perturbed = loss_perturbed + args.lambda_isr * isr_loss_perturbed
                loss_perturbed.backward()
                sam_opt.second_step()
                loss = loss_perturbed # Use perturbed loss for tracking
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i+1) % 50 == 0 or (i+1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
                
        # Test Evaluation at end of epoch
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
        test_acc = 100. * test_correct / test_total
        print(f"Epoch [{epoch+1}/{args.epochs}] Test Accuracy: {test_acc:.2f}%")
        
    # Save the adapter
    mode_dir = f"{args.dataset}_{args.mode}"
    save_path = os.path.join(args.output_dir, mode_dir)
    print(f"Saving adapter to {save_path}...")
    model.save_pretrained(save_path)
    
    # Save the task-specific classifier head weights
    classifier_path = os.path.join(save_path, "classifier_head.bin")
    torch.save(model.base_model.model.classifier.state_dict(), classifier_path)
    print(f"Saved classifier head to {classifier_path}")
    print("Training finished!")

if __name__ == "__main__":
    main()
