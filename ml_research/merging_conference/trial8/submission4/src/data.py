import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_transforms():
    # Grayscale normalization
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def download_datasets(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    transform = get_transforms()
    
    # Download MNIST
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Download KMNIST
    kmnist_train = datasets.KMNIST(data_dir, train=True, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(data_dir, train=False, download=True, transform=transform)
    
    # Download FashionMNIST
    fmnist_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    return {
        "mnist": (mnist_train, mnist_test),
        "kmnist": (kmnist_train, kmnist_test),
        "fmnist": (fmnist_train, fmnist_test)
    }

def get_expert_loaders(dataset_dict, num_train_samples=10000, batch_size=64):
    loaders = {}
    for name, (train_dataset, test_dataset) in dataset_dict.items():
        # Subset of train_dataset
        indices = list(range(min(num_train_samples, len(train_dataset))))
        subset_train = Subset(train_dataset, indices)
        
        train_loader = DataLoader(subset_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        loaders[name] = {
            "train": train_loader,
            "test": test_loader,
            "raw_test": test_dataset
        }
    return loaders

def add_gaussian_noise(x, std=0.6):
    noise = torch.randn_like(x) * std
    return x + noise

def build_streams(expert_loaders, batch_size=64, noise_std=0.6):
    mnist_test = expert_loaders["mnist"]["raw_test"]
    kmnist_test = expert_loaders["kmnist"]["raw_test"]
    fmnist_test = expert_loaders["fmnist"]["raw_test"]
    
    # We will draw batches deterministically for reproducibility
    mnist_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False)
    
    # Convert loaders to lists of batches
    mnist_batches = list(mnist_loader)
    kmnist_batches = list(kmnist_loader)
    fmnist_batches = list(fmnist_loader)
    
    # Stream 1: Closed Sequential (20 batches MNIST, 20 batches KMNIST)
    stream_seq = []
    for i in range(20):
        stream_seq.append((mnist_batches[i][0], mnist_batches[i][1], "mnist"))
    for i in range(20):
        stream_seq.append((kmnist_batches[i][0], kmnist_batches[i][1], "kmnist"))
        
    # Stream 2: Closed Alternating (40 batches alternating between MNIST and KMNIST)
    stream_alt = []
    for i in range(20):
        stream_alt.append((mnist_batches[i][0], mnist_batches[i][1], "mnist"))
        stream_alt.append((kmnist_batches[i][0], kmnist_batches[i][1], "kmnist"))
        
    # Stream 3: Open-World (15 MNIST, 15 KMNIST, 15 FashionMNIST as Novel)
    stream_ow = []
    for i in range(15):
        stream_ow.append((mnist_batches[i][0], mnist_batches[i][1], "mnist"))
    for i in range(15):
        stream_ow.append((kmnist_batches[i][0], kmnist_batches[i][1], "kmnist"))
    for i in range(15):
        stream_ow.append((fmnist_batches[i][0], fmnist_batches[i][1], "fmnist"))
        
    # Stream 4: Noisy Open-World
    # 10 Clean MNIST, 10 Noisy MNIST, 10 Clean KMNIST, 10 Noisy KMNIST, 10 Novel (FashionMNIST)
    stream_noisy_ow = []
    for i in range(10):
        stream_noisy_ow.append((mnist_batches[i][0], mnist_batches[i][1], "mnist"))
    for i in range(10):
        x_noisy = add_gaussian_noise(mnist_batches[10+i][0], noise_std)
        stream_noisy_ow.append((x_noisy, mnist_batches[10+i][1], "mnist_noisy"))
    for i in range(10):
        stream_noisy_ow.append((kmnist_batches[i][0], kmnist_batches[i][1], "kmnist"))
    for i in range(10):
        x_noisy = add_gaussian_noise(kmnist_batches[10+i][0], noise_std)
        stream_noisy_ow.append((x_noisy, kmnist_batches[10+i][1], "kmnist_noisy"))
    for i in range(10):
        stream_noisy_ow.append((fmnist_batches[i][0], fmnist_batches[i][1], "fmnist"))
        
    return {
        "Closed Sequential": stream_seq,
        "Closed Alternating": stream_alt,
        "Open-World": stream_ow,
        "Noisy Open-World": stream_noisy_ow
    }
