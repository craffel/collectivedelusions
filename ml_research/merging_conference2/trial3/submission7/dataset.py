import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np

class MultiChannelDataset(Dataset):
    """
    Wrapper to ensure all images are 3-channel and normalized.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        # Ensure 3-channel (just a safeguard)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, label

def get_transforms(augment=False):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # We use a Lambda to convert any PIL Image to RGB before applying other transforms.
    to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
    
    if augment:
        train_transform = transforms.Compose([
            to_rgb,
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        train_transform = transforms.Compose([
            to_rgb,
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        
    test_transform = transforms.Compose([
        to_rgb,
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return train_transform, test_transform

def load_datasets(data_dir="./data", num_train_samples=2000, seed=42):
    """
    Loads MNIST, Fashion-MNIST, and CIFAR-10.
    Returns:
        dict: {
            'mnist': (train_loader, test_loader),
            'fmnist': (train_loader, test_loader),
            'cifar10': (train_loader, test_loader)
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_transform, test_transform = get_transforms(augment=False)
    # Augment versions can be fetched if needed, but keeping it standard is safer for fast training.
    train_transform_aug, _ = get_transforms(augment=True)
    
    # Download raw datasets
    raw_mnist_train = datasets.MNIST(data_dir, train=True, download=True)
    raw_mnist_test = datasets.MNIST(data_dir, train=False, download=True)
    
    raw_fmnist_train = datasets.FashionMNIST(data_dir, train=True, download=True)
    raw_fmnist_test = datasets.FashionMNIST(data_dir, train=False, download=True)
    
    raw_cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True)
    raw_cifar10_test = datasets.CIFAR10(data_dir, train=False, download=True)
    
    # Wrap in MultiChannelDataset
    mnist_train = MultiChannelDataset(raw_mnist_train, transform=train_transform_aug)
    mnist_test = MultiChannelDataset(raw_mnist_test, transform=test_transform)
    
    fmnist_train = MultiChannelDataset(raw_fmnist_train, transform=train_transform_aug)
    fmnist_test = MultiChannelDataset(raw_fmnist_test, transform=test_transform)
    
    cifar10_train = MultiChannelDataset(raw_cifar10_train, transform=train_transform_aug)
    cifar10_test = MultiChannelDataset(raw_cifar10_test, transform=test_transform)
    
    # Select random subsets for expert training to speed up
    def get_subset(dataset, num_samples):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        return Subset(dataset, indices)
    
    mnist_train_sub = get_subset(mnist_train, num_train_samples)
    fmnist_train_sub = get_subset(fmnist_train, num_train_samples)
    cifar10_train_sub = get_subset(cifar10_train, num_train_samples)
    
    # Create Dataloaders
    batch_size = 128
    dataloaders = {
        'mnist': (
            DataLoader(mnist_train_sub, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=2)
        ),
        'fmnist': (
            DataLoader(fmnist_train_sub, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(fmnist_test, batch_size=256, shuffle=False, num_workers=2)
        ),
        'cifar10': (
            DataLoader(cifar10_train_sub, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(cifar10_test, batch_size=256, shuffle=False, num_workers=2)
        )
    }
    
    return dataloaders

def get_calibration_set(data_dir="./data", N=128, seed=42):
    """
    Constructs a joint calibration set containing an equal mix of N samples from each of the 3 tasks.
    Also returns task-specific calibration sets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, test_transform = get_transforms(augment=False)
    
    # Use training sets of raw datasets to draw calibration samples, wrapped with eval transform
    raw_mnist = datasets.MNIST(data_dir, train=True, download=True)
    raw_fmnist = datasets.FashionMNIST(data_dir, train=True, download=True)
    raw_cifar10 = datasets.CIFAR10(data_dir, train=True, download=True)
    
    mnist_cal = MultiChannelDataset(raw_mnist, transform=test_transform)
    fmnist_cal = MultiChannelDataset(raw_fmnist, transform=test_transform)
    cifar10_cal = MultiChannelDataset(raw_cifar10, transform=test_transform)
    
    mnist_idx = np.random.choice(len(mnist_cal), N, replace=False)
    fmnist_idx = np.random.choice(len(fmnist_cal), N, replace=False)
    cifar10_idx = np.random.choice(len(cifar10_cal), N, replace=False)
    
    mnist_sub = Subset(mnist_cal, mnist_idx)
    fmnist_sub = Subset(fmnist_cal, fmnist_idx)
    cifar10_sub = Subset(cifar10_cal, cifar10_idx)
    
    # Combine into joint dataset
    joint_samples = []
    # Store indices and mappings for joint dataset
    for dataset, task_id in [(mnist_sub, 0), (fmnist_sub, 1), (cifar10_sub, 2)]:
        for i in range(len(dataset)):
            img, label = dataset[i]
            joint_samples.append((img, label, task_id))
            
    class JointCalibrationDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            # return img, label, task_id
            return self.samples[idx]
            
    return JointCalibrationDataset(joint_samples), {
        'mnist': mnist_sub,
        'fmnist': fmnist_sub,
        'cifar10': cifar10_sub
    }
