import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset(name, train=True, batch_size=256, download=True, root='./data'):
    # Define common transform for resizing
    if name in ['mnist', 'fmnist']:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # Duplicate grayscale to 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if name == 'mnist':
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)
    elif name == 'fmnist':
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader
