import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

def get_transforms(is_grayscale=False):
    if is_grayscale:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_datasets(data_dir="./data"):
    # MNIST
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=get_transforms(True))
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=get_transforms(True))
    
    # Fashion-MNIST
    fmnist_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=get_transforms(True))
    fmnist_test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=get_transforms(True))
    
    # CIFAR-10
    cifar_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=get_transforms(False))
    cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_transforms(False))
    
    return {
        "mnist": {"train": mnist_train, "test": mnist_test},
        "fmnist": {"train": fmnist_train, "test": fmnist_test},
        "cifar10": {"train": cifar_train, "test": cifar_test}
    }

def get_subsets(all_datasets, train_size=5000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    subsets = {}
    for name, data_dict in all_datasets.items():
        train_dataset = data_dict["train"]
        test_dataset = data_dict["test"]
        
        # Get deterministic indices for train subset
        num_samples = len(train_dataset)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        
        subsets[name] = {
            "train": Subset(train_dataset, train_indices),
            "test": test_dataset
        }
    return subsets

def get_calibration_loader(train_subset, n_samples, batch_size, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(train_subset))
    np.random.shuffle(indices)
    calib_indices = indices[:n_samples]
    calib_subset = Subset(train_subset, calib_indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=False)
