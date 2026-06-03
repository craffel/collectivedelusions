import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors.")

def get_dataloaders(batch_size=64):
    # ImageNet normalization transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Datasets
    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    
    train_fmnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
    test_fmnist = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)
    
    train_cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    test_cifar = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    
    # Dataloaders
    loaders = {
        'mnist': {
            'train': DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'fmnist': {
            'train': DataLoader(train_fmnist, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'cifar': {
            'train': DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

def get_resnet18_model():
    # Load pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Modify the classifier head for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_model(model, train_loader, epochs=5, lr=1e-4):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
    return model

def evaluate_model(model, test_loader):
    model = model.to(DEVICE)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    loss = running_loss / len(test_loader.dataset)
    acc = 100.0 * correct / total
    return loss, acc

def merge_weights_wa(models_dict):
    # Perform Weight Averaging of the backbone (all layers except fc)
    merged_model = get_resnet18_model()
    merged_state = merged_model.state_dict()
    
    keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    for key in keys:
        temp = torch.zeros_like(merged_state[key], dtype=torch.float32)
        for name, m in models_dict.items():
            temp += m.state_dict()[key].cpu().float()
        merged_state[key].copy_(temp / len(models_dict))
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_weights_ta(models_dict, progenitor_model, lam=0.5):
    # Perform Task Arithmetic
    # W_merged = W_init + lam * sum(W_i - W_init)
    merged_model = get_resnet18_model()
    merged_state = merged_model.state_dict()
    prog_state = progenitor_model.state_dict()
    
    keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    for key in keys:
        init_val = prog_state[key].float().cpu()
        update_sum = torch.zeros_like(init_val)
        for name, m in models_dict.items():
            update_sum += (m.state_dict()[key].float().cpu() - init_val)
        merged_state[key].copy_(init_val + lam * update_sum)
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def apply_hns(merged_model, expert_model, progenitor_model):
    # Holographic Norm Scaling (HNS) in parameter space applied to task vectors (updates)
    merged_state = merged_model.state_dict()
    expert_state = expert_model.state_dict()
    prog_state = progenitor_model.state_dict()
    
    with torch.no_grad():
        for key in merged_state.keys():
            if 'fc' in key or 'classifier' in key:
                continue
            param_m = merged_state[key]
            param_e = expert_state[key]
            param_p = prog_state[key]
            
            if len(param_m.shape) >= 2:
                # It's a weight tensor (Conv or Linear)
                device = param_m.device
                param_e_dev = param_e.to(device).float()
                param_p_dev = param_p.to(device).float()
                param_m_float = param_m.float()
                
                # Compute task vectors relative to progenitor initialization
                tv_e = param_e_dev - param_p_dev
                tv_m = param_m_float - param_p_dev
                
                dim = tuple(range(1, len(param_m.shape)))
                norm_e = torch.norm(tv_e, p=2, dim=dim, keepdim=True)
                norm_m = torch.norm(tv_m, p=2, dim=dim, keepdim=True)
                
                scale = norm_e / (norm_m + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                
                # Apply scaling specifically along output channels
                tv_m_scaled = tv_m * scale.view(-1, *([1]*(len(param_m.shape)-1)))
                param_m.copy_((param_p_dev + tv_m_scaled).to(param_m.dtype))

def copy_bn_and_fc(target_model, source_model):
    # Copy both BatchNorm parameters/buffers and FC classification head from source to target
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()
    
    for key in target_state.keys():
        if 'fc' in key or 'bn' in key or 'downsample.1' in key:
            target_state[key].copy_(source_state[key].to(target_state[key].device))
            
    target_model.load_state_dict(target_state)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the expert models from scratch')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    # Ensure save directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Loading dataloaders...")
    loaders = get_dataloaders()
    
    # Save progenitor checkpoint
    progenitor_path = 'checkpoints/progenitor.pt'
    if not os.path.exists(progenitor_path) or args.train:
        print("Saving ImageNet-pretrained progenitor model...")
        progenitor = get_resnet18_model()
        torch.save(progenitor.state_dict(), progenitor_path)
    else:
        print("Progenitor model already exists.")
        progenitor = get_resnet18_model()
        progenitor.load_state_dict(torch.load(progenitor_path, map_location=DEVICE))
        
    experts = {}
    tasks = ['mnist', 'fmnist', 'cifar']
    
    for task in tasks:
        checkpoint_path = f'checkpoints/{task}_expert.pt'
        model = get_resnet18_model()
        if not os.path.exists(checkpoint_path) or args.train:
            print(f"\n--- Training Expert Model on {task.upper()} ---")
            # Initialize with progenitor weights
            model.load_state_dict(torch.load(progenitor_path))
            model = train_model(model, loaders[task]['train'], epochs=args.epochs, lr=args.lr)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            print(f"Loading pretrained expert model for {task}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        experts[task] = model
        
    # Evaluate individual expert oracles
    print("\n=== Evaluating Expert Oracles ===")
    oracle_accs = {}
    for task in tasks:
        _, acc = evaluate_model(experts[task], loaders[task]['test'])
        oracle_accs[task] = acc
        print(f"Expert Oracle Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"Average Oracle Accuracy: {sum(oracle_accs.values())/3:.2f}%")
    
    # 1. Weight Averaging (WA) Baseline
    print("\n=== Merging via Weight Averaging (WA) Baseline ===")
    wa_model = merge_weights_wa(experts)
    
    wa_accs = {}
    for task in tasks:
        # For evaluation, copy the task-specific BN and FC head
        temp_model = copy.deepcopy(wa_model)
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        wa_accs[task] = acc
        print(f"WA Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"Average WA Accuracy: {sum(wa_accs.values())/3:.2f}%")
    
    # 2. Task Arithmetic (TA) Baseline
    print("\n=== Merging via Task Arithmetic (TA) Baseline ===")
    for lam in [0.3, 0.5, 0.7]:
        print(f"\nTesting TA (lambda = {lam}):")
        ta_model = merge_weights_ta(experts, progenitor, lam=lam)
        ta_accs = {}
        for task in tasks:
            temp_model = copy.deepcopy(ta_model)
            copy_bn_and_fc(temp_model, experts[task])
            _, acc = evaluate_model(temp_model, loaders[task]['test'])
            ta_accs[task] = acc
            print(f"  TA ({lam}) Accuracy on {task.upper()}: {acc:.2f}%")
        print(f"  Average TA ({lam}) Accuracy: {sum(ta_accs.values())/3:.2f}%")
        
    # 3. Proposed Method: Holographic Norm Scaling (HNS) Merging
    print("\n=== Merging via Proposed Holographic Norm Scaling (HNS) ===")
    hns_accs_wa = {}
    print("\nEvaluating HNS on top of Weight Averaging (HNS-WA):")
    for task in tasks:
        # Create a deepcopy of the merged WA backbone
        temp_model = copy.deepcopy(wa_model)
        # Apply HNS specifically scaled to the current expert's weights
        apply_hns(temp_model, experts[task], progenitor)
        # Copy the expert's BN running statistics and classification head
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        hns_accs_wa[task] = acc
        print(f"  HNS-WA Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"  Average HNS-WA Accuracy: {sum(hns_accs_wa.values())/3:.2f}%")
    
    print("\nEvaluating HNS on top of Task Arithmetic (HNS-TA, lambda = 0.5):")
    ta_model = merge_weights_ta(experts, progenitor, lam=0.5)
    hns_accs_ta = {}
    for task in tasks:
        temp_model = copy.deepcopy(ta_model)
        apply_hns(temp_model, experts[task], progenitor)
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        hns_accs_ta[task] = acc
        print(f"  HNS-TA Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"  Average HNS-TA Accuracy: {sum(hns_accs_ta.values())/3:.2f}%")

if __name__ == '__main__':
    main()
