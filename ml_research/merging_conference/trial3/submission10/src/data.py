import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_transforms(img_size=32, corruption='clean'):
    # Base transforms: grayscale to RGB and resize to img_size
    base_transform = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    if corruption == 'clean':
        pass
    elif corruption == 'noise':
        # Add Gaussian noise with sigma = 0.4
        base_transform.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.4))
    elif corruption == 'blur':
        # Gaussian blur with kernel size 5 and sigma 1.5
        base_transform.append(transforms.GaussianBlur(kernel_size=5, sigma=1.5))
    elif corruption == 'contrast':
        # Scale contrast by 0.25
        base_transform.append(transforms.ColorJitter(contrast=(0.25, 0.25)))
    elif corruption == 'rotation':
        # Rotate by 30 degrees
        base_transform.append(transforms.RandomRotation((30, 30)))
        
    return transforms.Compose(base_transform)

def get_expert_dataloaders(img_size=32, batch_size=128, num_train_samples=10000):
    transform = get_transforms(img_size, 'clean')
    
    # Load MNIST
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if num_train_samples < len(mnist_train):
        mnist_train = Subset(mnist_train, list(range(num_train_samples)))
    mnist_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load FashionMNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    if num_train_samples < len(fmnist_train):
        fmnist_train = Subset(fmnist_train, list(range(num_train_samples)))
    fmnist_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load KMNIST
    kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    if num_train_samples < len(kmnist_train):
        kmnist_train = Subset(kmnist_train, list(range(num_train_samples)))
    kmnist_loader = DataLoader(kmnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return mnist_loader, fmnist_loader, kmnist_loader

def get_tta_streams(img_size=32, corruption='clean', num_samples_per_task=512, batch_size=32, stream_type='sequential'):
    """
    Constructs TTA streams:
    - Balanced (alternating): batches alternate task k
    - Sequential (non-stationary): task 0 batches, then task 1, then task 2
    """
    transform = get_transforms(img_size, corruption)
    
    # Test sets
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Subsets of num_samples_per_task
    mnist_sub = Subset(mnist_test, list(range(num_samples_per_task)))
    fmnist_sub = Subset(fmnist_test, list(range(num_samples_per_task)))
    kmnist_sub = Subset(kmnist_test, list(range(num_samples_per_task)))
    
    # Full uncorrupted evaluation sets for measuring final generalization
    eval_transform = get_transforms(img_size, 'clean')
    mnist_full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=eval_transform)
    fmnist_full_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=eval_transform)
    kmnist_full_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=eval_transform)
    
    eval_loaders = {
        'mnist': DataLoader(mnist_full_test, batch_size=128, shuffle=False, num_workers=2),
        'fmnist': DataLoader(fmnist_full_test, batch_size=128, shuffle=False, num_workers=2),
        'kmnist': DataLoader(kmnist_full_test, batch_size=128, shuffle=False, num_workers=2)
    }
    
    # We construct batches
    mnist_loader = DataLoader(mnist_sub, batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=batch_size, shuffle=False)
    
    batches = []
    # Store batches as (images, labels, task_id)
    mnist_batches = [(imgs, lbls, 0) for imgs, lbls in mnist_loader]
    fmnist_batches = [(imgs, lbls, 1) for imgs, lbls in fmnist_loader]
    kmnist_batches = [(imgs, lbls, 2) for imgs, lbls in kmnist_loader]
    
    num_batches = len(mnist_batches) # should be 512 / 32 = 16
    
    if stream_type == 'sequential':
        batches = mnist_batches + fmnist_batches + kmnist_batches
    elif stream_type == 'balanced':
        for i in range(num_batches):
            batches.append(mnist_batches[i])
            batches.append(fmnist_batches[i])
            batches.append(kmnist_batches[i])
            
    return batches, eval_loaders
