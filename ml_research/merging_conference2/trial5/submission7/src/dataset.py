import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

def get_transforms():
    # Grayscale datasets: resize to 32x32, convert to 3 channels, and normalize to [-1, 1]
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Color datasets: resize to 32x32, normalize to [-1, 1]
    color_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return gray_transform, color_transform

def get_datasets(data_dir="./data", calib_size=128, calib_seed=None):
    gray_tr, color_tr = get_transforms()
    
    # Download raw datasets
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=gray_tr)
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=gray_tr)
    
    fmnist_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=gray_tr)
    fmnist_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=gray_tr)
    
    cifar10_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=color_tr)
    cifar10_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=color_tr)
    
    # Splits:
    # Fine-Tuning: First 5,000 samples (we use a subset of 3,000 for training as in papers)
    train_indices = list(range(3000))
    if calib_seed is not None:
        import random
        rng = random.Random(calib_seed)
        calib_indices = rng.sample(range(5000, 10000), calib_size)
    else:
        calib_indices = list(range(5000, 5000 + calib_size))
    
    splits = {
        'mnist': {
            'train': Subset(mnist_train, train_indices),
            'calib': Subset(mnist_train, calib_indices),
            'test': mnist_test
        },
        'fmnist': {
            'train': Subset(fmnist_train, train_indices),
            'calib': Subset(fmnist_train, calib_indices),
            'test': fmnist_test
        },
        'cifar10': {
            'train': Subset(cifar10_train, train_indices),
            'calib': Subset(cifar10_train, calib_indices),
            'test': cifar10_test
        }
    }
    
    return splits

def get_dataloaders(splits, batch_size=128):
    loaders = {}
    for task, datasets_dict in splits.items():
        loaders[task] = {
            'train': DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True, num_workers=2),
            'calib': DataLoader(datasets_dict['calib'], batch_size=batch_size, shuffle=False, num_workers=2),
            'test': DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False, num_workers=2)
        }
    return loaders
