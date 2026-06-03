import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import os

def get_transforms():
    # Standard transform for color images (CIFAR-10)
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Grayscale to 3-channel RGB transform for MNIST and FashionMNIST
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform_color, transform_gray

_dataset_cache = {}
_dataloader_cache = {}

def get_dataset(name, train=True, root='./data'):
    key = (name.lower(), train, root)
    if key in _dataset_cache:
        return _dataset_cache[key]
        
    transform_color, transform_gray = get_transforms()
    if name.lower() == 'mnist':
        ds = datasets.MNIST(root=root, train=train, download=True, transform=transform_gray)
    elif name.lower() == 'fmnist':
        ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform_gray)
    elif name.lower() == 'cifar10':
        ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    _dataset_cache[key] = ds
    return ds

def get_dataloader(name, batch_size=256, train=True, root='./data', num_workers=0):
    key = (name.lower(), batch_size, train, root, num_workers)
    if key in _dataloader_cache:
        return _dataloader_cache[key]
        
    dataset = get_dataset(name, train=train, root=root)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    _dataloader_cache[key] = dl
    return dl

_class_indices_cache = {}

def get_calibration_subset(name, num_samples=64, root='./data', seed=42):
    """
    Get a balanced subset of calibration samples for data-efficient calibration.
    """
    dataset = get_dataset(name, train=True, root=root)
    # Group indices by class
    key = name.lower()
    if key in _class_indices_cache:
        class_indices = _class_indices_cache[key]
    else:
        class_indices = {}
        if hasattr(dataset, 'targets') and dataset.targets is not None:
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            for idx, label in enumerate(targets):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        else:
            for idx in range(len(dataset)):
                # dataset[idx] returns (img, label)
                _, label = dataset[idx]
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
                
                # Stop scanning early to save time if we have enough per class
                # Assuming 10 classes and we want num_samples in total
                min_required = max(10, num_samples // 10 + 5)
                if all(len(indices) >= min_required for indices in class_indices.values()) and len(class_indices) == 10:
                    break
        _class_indices_cache[key] = class_indices
            
    # Sample equally from each class
    sampled_indices = []
    samples_per_class = num_samples // 10
    remainder = num_samples % 10
    
    import random
    random.seed(seed)
    for label, indices in sorted(class_indices.items()):
        k = samples_per_class + (1 if label < remainder else 0)
        sampled_indices.extend(random.sample(indices, min(k, len(indices))))
        
    subset = Subset(dataset, sampled_indices)
    return DataLoader(subset, batch_size=num_samples, shuffle=False)
