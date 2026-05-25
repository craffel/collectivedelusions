import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_transforms(dataset_name):
    if dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset_name == "svhn":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
    elif dataset_name == "fmnist":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    return None

def build_test_stream(batch_size=64, num_batches_per_domain=30):
    """
    Builds a non-stationary test stream consisting of:
    - 30 batches of CIFAR-10
    - 30 batches of SVHN
    - 30 batches of FashionMNIST
    Total = 90 batches.
    """
    data_dir = "./data"
    
    # Load test datasets
    cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=get_transforms("cifar10"))
    svhn_test = torchvision.datasets.SVHN(root=data_dir, split="test", download=False, transform=get_transforms("svhn"))
    fmnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=get_transforms("fmnist"))
    
    # Take slices for the stream
    cifar_subset = Subset(cifar_test, range(num_batches_per_domain * batch_size))
    svhn_subset = Subset(svhn_test, range(num_batches_per_domain * batch_size))
    fmnist_subset = Subset(fmnist_test, range(num_batches_per_domain * batch_size))
    
    # Create sequential stream of batches
    cifar_loader = DataLoader(cifar_subset, batch_size=batch_size, shuffle=False)
    svhn_loader = DataLoader(svhn_subset, batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(fmnist_subset, batch_size=batch_size, shuffle=False)
    
    stream = []
    # Add Task A: CIFAR-10 (batches 1-30)
    for inputs, targets in cifar_loader:
        stream.append((inputs, targets, "cifar10"))
        
    # Add Task B: SVHN (batches 31-60)
    for inputs, targets in svhn_loader:
        stream.append((inputs, targets, "svhn"))
        
    # Add Task C: FashionMNIST (batches 61-90, novel domain)
    for inputs, targets in fmnist_loader:
        stream.append((inputs, targets, "fmnist"))
        
    return stream

def get_calibration_loader(dataset_name, num_samples=256, batch_size=32):
    """
    Returns a small calibration loader from the training split of the dataset.
    """
    data_dir = "./data"
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=get_transforms("cifar10"))
    elif dataset_name == "svhn":
        train_dataset = torchvision.datasets.SVHN(root=data_dir, split="train", download=False, transform=get_transforms("svhn"))
    elif dataset_name == "fmnist":
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=get_transforms("fmnist"))
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    subset = Subset(train_dataset, range(min(num_samples, len(train_dataset))))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

def compute_expert_prototypes(expert, calibration_loader, device):
    """
    Computes class-specific prototypes for an expert model on its calibration set.
    Returns: A tensor of shape [10, 512] representing class prototypes.
    """
    expert.eval()
    expert.to(device)
    
    features_by_class = {c: [] for c in range(10)}
    
    with torch.no_grad():
        for inputs, targets in calibration_loader:
            inputs = inputs.to(device)
            # Extract features (before fc layer)
            feats = expert.extract_features(inputs) # Shape: [B, 512]
            for feat, target in zip(feats, targets):
                features_by_class[target.item()].append(feat.cpu())
                
    prototypes = []
    for c in range(10):
        if len(features_by_class[c]) > 0:
            class_feats = torch.stack(features_by_class[c])
            mean_feat = class_feats.mean(dim=0)
            prototypes.append(mean_feat)
        else:
            prototypes.append(torch.zeros(512))
            
    prototypes = torch.stack(prototypes).to(device)
    # Center prototypes to match test-time Isotropic Feature Centering (IFC)
    prototypes_mean = prototypes.mean(dim=0, keepdim=True)
    prototypes_centered = prototypes - prototypes_mean
    # L2-normalize the centered prototypes
    prototypes_normalized = prototypes_centered / (prototypes_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
    
    return prototypes_normalized

def compute_diagonal_fisher(expert, calibration_loader, device):
    """
    Computes diagonal Fisher Information for an expert model.
    Returns: A dict mapping parameter names to their diagonal Fisher Information tensors.
    """
    expert.eval()
    expert.to(device)
    
    fisher = {name: torch.zeros_like(param) for name, param in expert.named_parameters() if param.requires_grad}
    criterion = nn.CrossEntropyLoss()
    total_samples = 0
    
    for inputs, targets in calibration_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Compute gradients sample-by-sample for exact Fisher
        for i in range(batch_size):
            expert.zero_grad()
            input_single = inputs[i:i+1]
            target_single = targets[i:i+1]
            
            output = expert(input_single)
            loss = criterion(output, target_single)
            loss.backward()
            
            for name, param in expert.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
                    
            total_samples += 1
            
    # Normalize by total samples
    for name in fisher.keys():
        fisher[name] = fisher[name] / total_samples
        
    return fisher
