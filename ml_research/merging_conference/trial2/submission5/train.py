import argparse
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Disable cuDNN to avoid initialization errors on cluster
torch.backends.cudnn.enabled = False

# Custom SAM Optimizer
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w) # climb to local maximum
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"]) # restore original weights
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# Dataloader for Split CIFAR-10
class SplitCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, classes=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        if classes is not None:
            indices = [i for i, (_, label) in enumerate(self.dataset) if label in classes]
            self.data = [self.dataset.data[i] for i in indices]
            self.targets = [self.dataset.targets[i] for i in indices]
        else:
            self.data = self.dataset.data
            self.targets = self.dataset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# Helper to compute SPOR loss
def compute_spor_loss(model, base_model):
    spor_loss = 0.0
    count = 0
    # Flatten weights of each Conv2d layer and penalize deviation from orthogonal rotation of W0
    for (name, module), (base_name, base_module) in zip(model.named_modules(), base_model.named_modules()):
        if isinstance(module, nn.Conv2d):
            W = module.weight
            W0 = base_module.weight
            
            # Flatten to 2D matrix
            W_flat = W.view(W.size(0), -1)
            W0_flat = W0.view(W0.size(0), -1)
            
            # Row normalization
            W_norm = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-8)
            W0_norm = W0_flat / (W0_flat.norm(dim=1, keepdim=True) + 1e-8)
            
            # M = W_norm * W0_norm^T
            M = torch.mm(W_norm, W0_norm.t()) # [C_out, C_out]
            I = torch.eye(M.size(0), device=M.device)
            
            # Deviate of M M^T from Identity
            layer_loss = torch.mean((torch.mm(M, M.t()) - I) ** 2)
            spor_loss += layer_loss
            count += 1
            
    return spor_loss / max(count, 1)

def train(args):
    print(f"=== Training Expert {args.expert} with mode {args.mode} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize base model
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    base_model = torchvision.models.resnet18(weights=weights)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    
    # Save the base model if not already saved
    base_path = os.path.join(args.save_dir, "base_model.pt")
    if not os.path.exists(base_path):
        torch.save(base_model.state_dict(), base_path)
        print(f"Saved base model to {base_path}")
        
    # We load base_model on device for SPOR reference and model initialization
    base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
        
    # Model to train
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(base_path))
    model = model.to(device)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Task classes selection
    classes = list(range(0, 5)) if args.expert == "A" else list(range(5, 10))
    
    # Dataset and DataLoader
    train_dataset = SplitCIFAR10(root=args.data_dir, train=True, transform=train_transform, classes=classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Loss and optimizer setup
    criterion = nn.CrossEntropyLoss()
    
    if args.mode == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else: # sam or sam_spor
        optimizer = SAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer if args.mode == "sgd" else optimizer.base_optimizer, 
        T_max=args.epochs
    )
    
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_spor = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            if args.mode == "sgd":
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            elif args.mode == "sam":
                # First pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second pass
                outputs_perturbed = model(images)
                loss_perturbed = criterion(outputs_perturbed, targets)
                loss_perturbed.backward()
                optimizer.second_step(zero_grad=True)
                running_loss += loss.item() * images.size(0)
            elif args.mode == "sam_spor":
                # First pass with SPOR
                optimizer.zero_grad()
                outputs = model(images)
                task_loss = criterion(outputs, targets)
                spor_loss = compute_spor_loss(model, base_model)
                loss = task_loss + args.beta * spor_loss
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second pass with SPOR
                outputs_perturbed = model(images)
                task_loss_perturbed = criterion(outputs_perturbed, targets)
                spor_loss_perturbed = compute_spor_loss(model, base_model)
                loss_perturbed = task_loss_perturbed + args.beta * spor_loss_perturbed
                loss_perturbed.backward()
                optimizer.second_step(zero_grad=True)
                
                running_loss += task_loss.item() * images.size(0)
                running_spor += spor_loss.item() * images.size(0)
                
            # Accuracy metric on the training batch
            if args.mode == "sam_spor":
                outputs = model(images) # re-eval after update
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_spor = running_spor / total if args.mode == "sam_spor" else 0.0
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {epoch_loss:.4f} | SPOR: {epoch_spor:.4f} | Acc: {epoch_acc:.2f}% | Time: {time.time()-start_time:.1f}s")
        
    # Save the trained model checkpoint
    if args.mode == "sam_spor":
        save_path = os.path.join(args.save_dir, f"expert_{args.expert}_{args.mode}_beta_{args.beta}.pt")
    else:
        save_path = os.path.join(args.save_dir, f"expert_{args.expert}_{args.mode}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 experts on split CIFAR-10")
    parser.add_argument("--mode", type=str, default="sgd", choices=["sgd", "sam", "sam_spor"], help="Training mode")
    parser.add_argument("--expert", type=str, default="A", choices=["A", "B"], help="Which expert (A: classes 0-4, B: classes 5-9)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--rho", type=float, default=0.05, help="SAM perturbation radius")
    parser.add_argument("--beta", type=float, default=0.01, help="SPOR regularization coefficient")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    train(args)
