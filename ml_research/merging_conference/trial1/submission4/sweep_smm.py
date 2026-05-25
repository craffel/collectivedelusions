import os
import argparse
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

# Define SAM Optimizer
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.params = list(params)
        self.optimizer = base_optimizer(self.params, **kwargs)
        self.rho = rho
        self.e_ws = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for p in self.params:
            if p.grad is None: continue
            scale = self.rho / (grad_norm + 1e-12)
            e_w = p.grad * scale
            p.add_(e_w) # perturb
            self.e_ws[p] = e_w
        if zero_grad:
            self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            if p in self.e_ws:
                p.sub_(self.e_ws[p]) # restore original
        self.e_ws.clear()
        self.optimizer.step()
        if zero_grad:
            self.optimizer.zero_grad()

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

    def zero_grad(self):
        self.optimizer.zero_grad()

# Block-diagonal Orthogonal Layer (OFT)
class OFTLayer(nn.Module):
    def __init__(self, original_layer, block_size=16):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
        self.block_size = block_size
        self.out_features = original_layer.weight.shape[0]
        
        # Adjust block_size to be a divisor of out_features
        for b in range(self.block_size, 0, -1):
            if self.out_features % b == 0:
                self.block_size = b
                break
        
        self.num_blocks = self.out_features // self.block_size
        self.tri_size = self.block_size * (self.block_size - 1) // 2
        
        if self.tri_size > 0:
            self.q_params = nn.Parameter(torch.zeros(self.num_blocks, self.tri_size))
            nn.init.normal_(self.q_params, std=1e-4)
        else:
            self.q_params = None

    def get_R(self):
        device = self.original_layer.weight.device
        if self.q_params is None:
            return None
            
        Q = torch.zeros(self.num_blocks, self.block_size, self.block_size, device=device)
        tri_indices = torch.triu_indices(self.block_size, self.block_size, offset=1, device=device)
        
        Q[:, tri_indices[0], tri_indices[1]] = self.q_params
        Q = Q - Q.transpose(-1, -2) # skew-symmetry
        
        I = torch.eye(self.block_size, device=device).unsqueeze(0).expand(self.num_blocks, -1, -1)
        # Cayley transform
        R_blocks = torch.matmul(I + Q, torch.inverse(I - Q))
        return R_blocks

    def forward(self, x):
        weight_0 = self.original_layer.weight
        bias = self.original_layer.bias
        C_out = weight_0.shape[0]
        weight_flat = weight_0.view(C_out, -1)
        
        R_blocks = self.get_R()
        if R_blocks is not None:
            weight_reshaped = weight_flat.view(self.num_blocks, self.block_size, -1)
            weight_oft = torch.matmul(R_blocks, weight_reshaped)
            weight_oft = weight_oft.view(C_out, -1)
        else:
            weight_oft = weight_flat
            
        weight = weight_oft.view_as(weight_0)
        
        if isinstance(self.original_layer, nn.Conv2d):
            return F.conv2d(x, weight, bias, 
                           stride=self.original_layer.stride,
                           padding=self.original_layer.padding,
                           dilation=self.original_layer.dilation,
                           groups=self.original_layer.groups)
        elif isinstance(self.original_layer, nn.Linear):
            return F.linear(x, weight, bias)
        else:
            return self.original_layer(x)

def apply_oft(model, block_size=16):
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name == 'fc':
                continue
            setattr(model, name, OFTLayer(module, block_size))
        else:
            apply_oft(module, block_size)

# Load data
def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_targets = torch.tensor(trainset.targets)
    test_targets = torch.tensor(testset.targets)

    task1_train_idx = torch.where(train_targets < 5)[0]
    task2_train_idx = torch.where(train_targets >= 5)[0]

    task1_test_idx = torch.where(test_targets < 5)[0]
    task2_test_idx = torch.where(test_targets >= 5)[0]

    train_loader1 = DataLoader(Subset(trainset, task1_train_idx), batch_size=128, shuffle=True, num_workers=2)
    train_loader2 = DataLoader(Subset(trainset, task2_train_idx), batch_size=128, shuffle=True, num_workers=2)

    test_loader1 = DataLoader(Subset(testset, task1_test_idx), batch_size=128, shuffle=False, num_workers=2)
    test_loader2 = DataLoader(Subset(testset, task2_test_idx), batch_size=128, shuffle=False, num_workers=2)

    return train_loader1, train_loader2, test_loader1, test_loader2

# Train standard or joint SAM
def train_joint(model, loader, optimizer, epochs, device, sam_opt=None, is_task2=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_task2:
                targets = targets - 5
                
            if sam_opt is not None:
                # First step of SAM
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                sam_opt.first_step(zero_grad=True)
                
                # Second step of SAM
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                sam_opt.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# Train decoupled SAM (SAM only on Q, standard Adam on FC)
def train_decoupled(model, loader, sam_q, optimizer_fc, epochs, device, is_task2=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_task2:
                targets = targets - 5
                
            # First step of SAM (only on Q)
            optimizer_fc.zero_grad()
            sam_q.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Perturb q params
            sam_q.first_step(zero_grad=True)
            optimizer_fc.zero_grad() # clear FC gradients so we don't double count
            
            # Second step of SAM
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Update both
            sam_q.second_step(zero_grad=True)
            optimizer_fc.step()
            optimizer_fc.zero_grad()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# Evaluate function
def evaluate(model, loader, device, is_task2=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_task2:
                targets = targets - 5
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['orthomerge', 'smm_joint', 'smm_decoupled'], required=True)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sweep on: {device} | Mode: {args.mode} | rho: {args.rho} | lr: {args.lr} | block_size: {args.block_size}")
    
    train_loader1, train_loader2, test_loader1, test_loader2 = get_data()
    
    # Task 1
    model1 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    apply_oft(model1, block_size=args.block_size)
    model1.fc = nn.Linear(model1.fc.in_features, 5)
    model1 = model1.to(device)
    
    # Task 2
    model2 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    apply_oft(model2, block_size=args.block_size)
    model2.fc = nn.Linear(model2.fc.in_features, 5)
    model2 = model2.to(device)

    # Set up optimizers based on mode
    if args.mode == 'orthomerge':
        oft_params1 = [p for n, p in model1.named_parameters() if 'q_params' in n or 'fc' in n]
        optimizer1 = torch.optim.Adam(oft_params1, lr=args.lr)
        print("Training Task 1 with standard Adam...")
        train_joint(model1, train_loader1, optimizer1, args.epochs, device)

        oft_params2 = [p for n, p in model2.named_parameters() if 'q_params' in n or 'fc' in n]
        optimizer2 = torch.optim.Adam(oft_params2, lr=args.lr)
        print("Training Task 2 with standard Adam...")
        train_joint(model2, train_loader2, optimizer2, args.epochs, device, is_task2=True)

    elif args.mode == 'smm_joint':
        smm_params1 = [p for n, p in model1.named_parameters() if 'q_params' in n or 'fc' in n]
        sam_opt1 = SAM(smm_params1, torch.optim.Adam, rho=args.rho, lr=args.lr)
        print("Training Task 1 with Joint SMM (SAM on Q & FC)...")
        train_joint(model1, train_loader1, None, args.epochs, device, sam_opt=sam_opt1)

        smm_params2 = [p for n, p in model2.named_parameters() if 'q_params' in n or 'fc' in n]
        sam_opt2 = SAM(smm_params2, torch.optim.Adam, rho=args.rho, lr=args.lr)
        print("Training Task 2 with Joint SMM (SAM on Q & FC)...")
        train_joint(model2, train_loader2, None, args.epochs, device, sam_opt=sam_opt2, is_task2=True)

    elif args.mode == 'smm_decoupled':
        q_params1 = [p for n, p in model1.named_parameters() if 'q_params' in n]
        fc_params1 = [p for n, p in model1.named_parameters() if 'fc' in n]
        sam_q1 = SAM(q_params1, torch.optim.Adam, rho=args.rho, lr=args.lr)
        optimizer_fc1 = torch.optim.Adam(fc_params1, lr=args.lr)
        print("Training Task 1 with Decoupled SMM (SAM only on Q, Adam on FC)...")
        train_decoupled(model1, train_loader1, sam_q1, optimizer_fc1, args.epochs, device)

        q_params2 = [p for n, p in model2.named_parameters() if 'q_params' in n]
        fc_params2 = [p for n, p in model2.named_parameters() if 'fc' in n]
        sam_q2 = SAM(q_params2, torch.optim.Adam, rho=args.rho, lr=args.lr)
        optimizer_fc2 = torch.optim.Adam(fc_params2, lr=args.lr)
        print("Training Task 2 with Decoupled SMM (SAM only on Q, Adam on FC)...")
        train_decoupled(model2, train_loader2, sam_q2, optimizer_fc2, args.epochs, device, is_task2=True)

    # Save heads
    fc1_state = copy.deepcopy(model1.fc.state_dict())
    fc2_state = copy.deepcopy(model2.fc.state_dict())
    
    # Merge Q parameters via magnitude-corrected average
    m1_sd = model1.state_dict()
    m2_sd = model2.state_dict()
    merged_state_dict = {}
    
    for key in m1_sd.keys():
        if 'fc' in key:
            continue
        if 'q_params' in key:
            q1 = m1_sd[key]
            q2 = m2_sd[key]
            # Magnitude-corrected average
            norm_sum = torch.norm(q1) + torch.norm(q2)
            q_avg = 0.5 * (q1 + q2)
            norm_avg = torch.norm(q_avg)
            c = norm_sum / (2.0 * norm_avg + 1e-12)
            merged_state_dict[key] = c * q_avg
        else:
            merged_state_dict[key] = m1_sd[key]
            
    # Evaluate Merged model
    merged_model = torchvision.models.resnet18()
    apply_oft(merged_model, block_size=args.block_size)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 5)
    merged_model = merged_model.to(device)
    
    merged_model.load_state_dict(merged_state_dict, strict=False)
    merged_model.fc.load_state_dict(fc1_state)
    acc1 = evaluate(merged_model, test_loader1, device)
    
    merged_model.fc.load_state_dict(fc2_state)
    acc2 = evaluate(merged_model, test_loader2, device, is_task2=True)
    
    avg_acc = (acc1 + acc2) / 2.0
    print(f"Results -> Task 1 Acc: {acc1:.2f}%, Task 2 Acc: {acc2:.2f}%, Avg Acc: {avg_acc:.2f}%")
    
    results = {
        'mode': args.mode,
        'rho': args.rho,
        'lr': args.lr,
        'block_size': args.block_size,
        'task1_acc': acc1,
        'task2_acc': acc2,
        'avg_acc': avg_acc
    }
    
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()
