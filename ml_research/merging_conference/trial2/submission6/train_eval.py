import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn_2_cuda'] = None
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# Custom LoRA Linear layer to wrap nn.Linear
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze original linear layer weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        # Define lora trainable parameters
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Forward pass through base linear layer
        base_out = self.original_linear(x)
        # Forward pass through LoRA path
        # x shape: [batch, seq_len, in_features] or [batch, in_features]
        # lora_A shape: [r, in_features]
        # lora_B shape: [out_features, r]
        # lora_out = x @ A.t() @ B.t()
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

def apply_lora(model, r=8, lora_alpha=16):
    # Apply LoRA specifically to query and value projection layers in all 12 self-attention blocks
    for i in range(12):
        attn = model.vit.encoder.layer[i].attention.attention
        attn.query = LoRALinear(attn.query, r=r, lora_alpha=lora_alpha)
        attn.value = LoRALinear(attn.value, r=r, lora_alpha=lora_alpha)

# Compute Soft-Orthogonality Spectral Regularization (SOSR) penalty
def compute_sosr(model, eps=1e-6):
    sosr_loss = 0.0
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            B = module.lora_B  # [out_features, r]
            A = module.lora_A  # [r, in_features]
            
            # B^T B (shape: [r, r])
            BtB = torch.matmul(B.t(), B)
            diag_BtB = torch.diag(BtB)
            off_diag_BtB = BtB - torch.diag(diag_BtB)
            numerator_B = torch.norm(off_diag_BtB, p='fro')**2
            denominator_B = torch.norm(diag_BtB, p=2)**2 + eps
            
            # A A^T (shape: [r, r])
            AAt = torch.matmul(A, A.t())
            diag_AAt = torch.diag(AAt)
            off_diag_AAt = AAt - torch.diag(diag_AAt)
            numerator_A = torch.norm(off_diag_AAt, p='fro')**2
            denominator_A = torch.norm(diag_AAt, p=2)**2 + eps
            
            sosr_loss += (numerator_B / denominator_B) + (numerator_A / denominator_A)
            count += 1
            
    if count > 0:
        return sosr_loss / count
    return torch.tensor(0.0)

# SAM Double-pass optimizer wrapper
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.params = list(params)
        self.base_optimizer = base_optimizer(self.params, **kwargs)
        self.rho = rho
        self.state = {}

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for p in self.params:
            if p.grad is None: continue
            scale = self.rho / (grad_norm + 1e-12)
            e_w = p.grad * scale
            p.add_(e_w)  # perturb weight
            self.state[p] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            if p.grad is None: continue
            if p in self.state:
                p.sub_(self.state[p])  # restore weight
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.params[0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for p in self.params if p.grad is not None
            ]),
            p=2
        )
        return norm

def train(model, dataloader, device, optimizer, method, lambda_sosr, epoch, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if method in ['sam', 'sata']:
            # First pass
            outputs = model(inputs).logits
            loss_task = criterion(outputs, targets)
            loss_total = loss_task
            
            if method == 'sata':
                loss_reg = compute_sosr(model).to(device)
                loss_total = loss_total + lambda_sosr * loss_reg
                
            loss_total.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second pass
            outputs_perturbed = model(inputs).logits
            loss_task_perturbed = criterion(outputs_perturbed, targets)
            loss_total_perturbed = loss_task_perturbed
            
            if method == 'sata':
                loss_reg_perturbed = compute_sosr(model).to(device)
                loss_total_perturbed = loss_total_perturbed + lambda_sosr * loss_reg_perturbed
                
            loss_total_perturbed.backward()
            optimizer.second_step(zero_grad=True)
            
            # Stats logging using standard outputs
            total_loss += loss_task.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        else:
            # Standard/SOSR optimization
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss_task = criterion(outputs, targets)
            loss_total = loss_task
            
            if method == 'sosr':
                loss_reg = compute_sosr(model).to(device)
                loss_total = loss_total + lambda_sosr * loss_reg
                
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_task.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {total_loss / (batch_idx + 1):.4f} | Acc: {100. * correct / total:.2f}%")

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).logits
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def main():
    # Disable cuDNN to bypass cuDNN initialization errors on certain GPU cluster configurations
    torch.backends.cudnn.enabled = False
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning on CIFAR-10 or SVHN")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "svhn"])
    parser.add_argument("--method", type=str, required=True, choices=["standard", "sam", "sosr", "sata"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--lambda_sosr", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./models")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | Method: {args.method}")
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load data
    os.makedirs("./data", exist_ok=True)
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        train_set = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        test_set = datasets.SVHN(root="./data", split="test", download=True, transform=transform)
        
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load pre-trained model
    print("Loading pre-trained ViT-B/16...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier head for 10 classes (requires_grad is True by default)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    
    # Apply custom LoRA adapters
    apply_lora(model)
    model.to(device)
    
    # Print trainable parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"Trainable: {name} | Shape: {param.shape}")
            
    # Optimizer setup
    if args.method in ["sam", "sata"]:
        optimizer = SAM(trainable_params, optim.AdamW, rho=args.rho, lr=args.lr)
    else:
        optimizer = optim.AdamW(trainable_params, lr=args.lr)
        
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, device, optimizer, args.method, args.lambda_sosr, epoch, args.epochs)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch}/{args.epochs}] Test Accuracy: {test_acc:.2f}%")
        
    # Save trained weights
    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = f"{args.dataset}_{args.method}.pt"
    save_path = os.path.join(args.save_dir, save_filename)
    
    state_dict_to_save = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "classifier" in name:
            state_dict_to_save[name] = param.cpu().data
            
    torch.save(state_dict_to_save, save_path)
    print(f"Successfully saved adapter weights to {save_path}")

if __name__ == "__main__":
    main()
