import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

def get_transforms(dataset_name):
    """
    Returns the appropriate transform for the dataset.
    Grayscale images (MNIST, Fashion-MNIST) are resized to 32x32 and replicated to 3 channels.
    CIFAR-10 is resized to 32x32 (or left as is since it is already 32x32).
    Standard ImageNet normalization is used to match pre-trained ResNet-18 expects.
    """
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if dataset_name in ['mnist', 'fmnist']:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # Replicate 1 channel to 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            imagenet_normalize
        ])
    elif dataset_name == 'cifar10':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            imagenet_normalize
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_multi_task_datasets(data_dir='./data', seed=42):
    """
    Downloads and returns the 5,000-sample training subsets and full 10,000-sample test sets
    for MNIST, Fashion-MNIST, and CIFAR-10.
    """
    # Set seed for reproducibility of subset selection
    rng = np.random.default_rng(seed)
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    train_datasets = {}
    test_datasets = {}
    
    # MNIST
    mnist_train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=get_transforms('mnist'))
    mnist_test_full = datasets.MNIST(root=data_dir, train=False, download=True, transform=get_transforms('mnist'))
    mnist_train_indices = rng.choice(len(mnist_train_full), 5000, replace=False)
    train_datasets['mnist'] = Subset(mnist_train_full, mnist_train_indices)
    test_datasets['mnist'] = mnist_test_full
    
    # Fashion-MNIST
    fmnist_train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=get_transforms('fmnist'))
    fmnist_test_full = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=get_transforms('fmnist'))
    fmnist_train_indices = rng.choice(len(fmnist_train_full), 5000, replace=False)
    train_datasets['fmnist'] = Subset(fmnist_train_full, fmnist_train_indices)
    test_datasets['fmnist'] = fmnist_test_full
    
    # CIFAR-10
    cifar10_train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms('cifar10'))
    cifar10_test_full = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms('cifar10'))
    cifar10_train_indices = rng.choice(len(cifar10_train_full), 5000, replace=False)
    train_datasets['cifar10'] = Subset(cifar10_train_full, cifar10_train_indices)
    test_datasets['cifar10'] = cifar10_test_full
    
    return train_datasets, test_datasets

def get_calibration_subset(train_dataset, n_samples, seed=42):
    """
    Returns a small calibration subset of size n_samples from the training dataset.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_dataset), n_samples, replace=False)
    return Subset(train_dataset, indices)
