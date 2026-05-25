import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_datasets():
    # Transform: to tensor, then normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download datasets
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    return mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test

def get_train_subsets(mnist_train, fmnist_train, num_samples=10000, seed=42):
    # Get subsets of 10,000 samples for expert pre-training
    np.random.seed(seed)
    
    mnist_indices = np.random.choice(len(mnist_train), num_samples, replace=False)
    fmnist_indices = np.random.choice(len(fmnist_train), num_samples, replace=False)
    
    mnist_sub = Subset(mnist_train, mnist_indices)
    fmnist_sub = Subset(fmnist_train, fmnist_indices)
    
    return mnist_sub, fmnist_sub

def get_calibration_samples(mnist_train, fmnist_train, num_samples=256, seed=42):
    # Precompute prototypes using 256 clean calibration samples per expert
    np.random.seed(seed)
    
    # We want class-wise calibration samples: 256 total samples, which is ~25-26 samples per class
    # To keep it simple, we can just take a random subset of 256 samples
    mnist_indices = np.random.choice(len(mnist_train), num_samples, replace=False)
    fmnist_indices = np.random.choice(len(fmnist_train), num_samples, replace=False)
    
    mnist_cal = Subset(mnist_train, mnist_indices)
    fmnist_cal = Subset(fmnist_train, fmnist_indices)
    
    return mnist_cal, fmnist_cal

def create_non_stationary_stream(mnist_test, fmnist_test, kmnist_test, batch_size=64, noise_std=0.6, seed=42):
    # 50 sequential batches (size B = 64)
    # Phase 0: Clean MNIST (batches 0-9)
    # Phase 1: Noisy MNIST (batches 10-19)
    # Phase 2: Clean FashionMNIST (batches 20-29)
    # Phase 3: Noisy FashionMNIST (batches 30-39)
    # Phase 4: Novel KMNIST (batches 40-49)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # We need 20 batches of MNIST, 20 batches of FashionMNIST, and 10 batches of KMNIST.
    # Total samples: MNIST (20 * 64 = 1280), FMNIST (20 * 64 = 1280), KMNIST (10 * 64 = 640)
    
    mnist_indices = np.random.choice(len(mnist_test), 1280, replace=False)
    fmnist_indices = np.random.choice(len(fmnist_test), 1280, replace=False)
    kmnist_indices = np.random.choice(len(kmnist_test), 640, replace=False)
    
    mnist_sub = Subset(mnist_test, mnist_indices)
    fmnist_sub = Subset(fmnist_test, fmnist_indices)
    kmnist_sub = Subset(kmnist_test, kmnist_indices)
    
    mnist_loader = DataLoader(mnist_sub, batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=batch_size, shuffle=False)
    
    stream_batches = []
    
    # Batches 0-9: Clean MNIST
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append((x, y, 0)) # Task index 0 (MNIST)
        
    # Batches 10-19: Noisy MNIST
    for _ in range(10):
        x, y = next(mnist_iter)
        # Add noise
        x_noisy = x + torch.randn_like(x) * noise_std
        stream_batches.append((x_noisy, y, 0))
        
    # Batches 20-29: Clean FashionMNIST
    fmnist_iter = iter(fmnist_loader)
    for _ in range(10):
        x, y = next(fmnist_iter)
        stream_batches.append((x, y, 1)) # Task index 1 (FashionMNIST)
        
    # Batches 30-39: Noisy FashionMNIST
    for _ in range(10):
        x, y = next(fmnist_iter)
        x_noisy = x + torch.randn_like(x) * noise_std
        stream_batches.append((x_noisy, y, 1))
        
    # Batches 40-49: Novel KMNIST (unseen)
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append((x, y, 2)) # Task index 2 (KMNIST)
        
    return stream_batches
