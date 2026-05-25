import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on some H100 nodes
torch.backends.cudnn.enabled = False

class SplitCIFAR10(Dataset):
    def __init__(self, root, train=True, task="A", transform=None, download=False):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        if task == "A":
            self.indices = [i for i, label in enumerate(self.dataset.targets) if label in [0, 1, 2, 3, 4]]
        elif task == "B":
            self.indices = [i for i, label in enumerate(self.dataset.targets) if label in [5, 6, 7, 8, 9]]
        else:
            self.indices = list(range(len(self.dataset)))

    def __getitem__(self, index):
        actual_index = self.indices[index]
        return self.dataset[actual_index]

    def __len__(self):
        return len(self.indices)

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
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to original weights
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

def get_spor_loss(model, base_model, beta, running_fisher=None, fg_mode=None):
    loss = 0.0
    count = 0
    eps_eps = 1e-8
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            base_module = dict(base_model.named_modules())[name]
            
            W = module.weight
            W0 = base_module.weight
            
            C_out = W.shape[0]
            W_flat = W.view(C_out, -1)
            W0_flat = W0.view(C_out, -1)
            
            W_norm = W_flat / (torch.norm(W_flat, p=2, dim=1, keepdim=True) + eps_eps)
            W0_norm = W0_flat / (torch.norm(W0_flat, p=2, dim=1, keepdim=True) + eps_eps)
            
            M = torch.mm(W_norm, W0_norm.t())
            MMt = torch.mm(M, M.t())
            I = torch.eye(C_out, device=W.device)
            
            if fg_mode is not None and running_fisher is not None and f"{name}.weight" in running_fisher:
                fisher = running_fisher[f"{name}.weight"]
                mean_fisher = fisher.mean()
                if mean_fisher > 0:
                    f = fisher / (mean_fisher + 1e-12)
                else:
                    f = torch.ones_like(fisher)
                
                if fg_mode == "direct":
                    v = f
                elif fg_mode == "inverse":
                    inv_f = 1.0 / (f + 0.1)
                    mean_inv_f = inv_f.mean()
                    v = inv_f / (mean_inv_f + 1e-12)
                else:
                    v = torch.ones_like(fisher)
                
                v_col = v.view(-1, 1)
                v_row = v.view(1, -1)
                V_matrix = torch.mm(v_col, v_row)
                
                diff = V_matrix * (MMt - I)
                layer_loss = torch.sum(diff ** 2) / (C_out ** 2)
            else:
                diff = MMt - I
                layer_loss = torch.sum(diff ** 2) / (C_out ** 2)
                
            loss += layer_loss
            count += 1
            
    return beta * loss

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 Experts on split CIFAR-10")
    parser.add_argument("--task", type=str, default="A", choices=["A", "B"], help="Split task")
    parser.add_argument("--config", type=str, default="sgd", choices=["sgd", "sam", "spor", "fg_spor_direct", "fg_spor_inverse"], help="Optimization config")
    parser.add_argument("--beta", type=float, default=0.05, help="SPOR regularization coefficient")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = SplitCIFAR10(root="./data", train=True, task=args.task, transform=transform_train, download=True)
    test_set = SplitCIFAR10(root="./data", train=False, task=args.task, transform=transform_test, download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load pre-trained ResNet-18
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model = base_model.to(device)
    base_model.eval()

    # Active model being trained
    model = copy.deepcopy(base_model)
    model = model.to(device)

    # Freeze base model weights for reference
    for p in base_model.parameters():
        p.requires_grad = False

    # Optimizers
    if args.config == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        # SAM-based configs
        optimizer = SAM(model.parameters(), optim.SGD, rho=0.05, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer if args.config != "sgd" else optimizer, 
        T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Tracking running Fisher Information
    running_fisher = {}
    fisher_alpha = 0.1

    print(f"Training Expert for Task {args.task} with config {args.config} (beta={args.beta})")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if args.config == "sgd":
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            else:
                # SAM-based training
                # First step: unperturbed pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Add SPOR if applicable
                fg_mode = None
                if args.config == "fg_spor_direct": fg_mode = "direct"
                elif args.config == "fg_spor_inverse": fg_mode = "inverse"
                
                if args.config in ["spor", "fg_spor_direct", "fg_spor_inverse"]:
                    spor_loss = get_spor_loss(model, base_model, args.beta, running_fisher, fg_mode)
                    total_loss = loss + spor_loss
                else:
                    total_loss = loss
                    
                total_loss.backward()

                # Update running Fisher information using unperturbed gradients before taking SAM step
                if args.config in ["fg_spor_direct", "fg_spor_inverse"]:
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            if "weight" in name and len(p.shape) == 4 and p.grad is not None:
                                grad_sq = p.grad.pow(2).sum(dim=[1, 2, 3])
                                if name not in running_fisher:
                                    running_fisher[name] = grad_sq.clone()
                                else:
                                    running_fisher[name] = (1 - fisher_alpha) * running_fisher[name] + fisher_alpha * grad_sq

                optimizer.first_step(zero_grad=True)

                # Second step: perturbed pass
                outputs_p = model(inputs)
                loss_p = criterion(outputs_p, targets)
                if args.config in ["spor", "fg_spor_direct", "fg_spor_inverse"]:
                    spor_loss_p = get_spor_loss(model, base_model, args.beta, running_fisher, fg_mode)
                    total_loss_p = loss_p + spor_loss_p
                else:
                    total_loss_p = loss_p
                    
                total_loss_p.backward()
                optimizer.second_step(zero_grad=True)

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        epoch_loss = train_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    # Evaluation on task test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100.0 * correct / total
    print(f"Task {args.task} Test Acc: {test_acc:.2f}%")

    # Save model checkpoint
    os.makedirs("./checkpoints", exist_ok=True)
    save_path = f"./checkpoints/expert_{args.task}_{args.config}_beta_{args.beta}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

if __name__ == "__main__":
    main()
