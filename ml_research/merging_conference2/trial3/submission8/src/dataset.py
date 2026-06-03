import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

class GrayscaleToRGB:
    """Convert a 1-channel grayscale tensor to a 3-channel RGB tensor."""
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        return tensor

def get_transforms():
    # Common transformations: resize to 32x32, convert to tensor, normalize
    # We use ImageNet normalization stats or standard [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        GrayscaleToRGB(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform

_dataset_cache = {}

def get_dataset(name, root='./data', train=True, subset_size=None, seed=42):
    transform = get_transforms()
    
    cache_key = (name, train)
    if cache_key not in _dataset_cache:
        if name == 'mnist':
            dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        elif name == 'fashion':
            dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
        elif name == 'cifar':
            dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        _dataset_cache[cache_key] = dataset
    else:
        dataset = _dataset_cache[cache_key]
    
    if subset_size is not None and train:
        # Get a deterministic subset of 5000 images per task for training
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:subset_size].tolist()
        dataset = Subset(dataset, indices)
        
    return dataset

def get_dataloaders(batch_size=128, subset_size=5000, seed=42):
    tasks = ['mnist', 'fashion', 'cifar']
    train_loaders = {}
    test_loaders = {}
    
    for task in tasks:
        train_ds = get_dataset(task, train=True, subset_size=subset_size, seed=seed)
        test_ds = get_dataset(task, train=False)
        
        train_loaders[task] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loaders[task] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
    return train_loaders, test_loaders
