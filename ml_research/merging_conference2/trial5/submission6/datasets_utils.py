import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

class ReplicateChannels(object):
    def __call__(self, img):
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img

def get_transforms(is_mnist=False):
    transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
    if is_mnist:
        transform_list.append(ReplicateChannels())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

def get_datasets(root='./data'):
    # MNIST
    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=get_transforms(is_mnist=True))
    mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=get_transforms(is_mnist=True))
    
    # FashionMNIST
    fmnist_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=get_transforms(is_mnist=True))
    fmnist_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=get_transforms(is_mnist=True))
    
    # CIFAR10
    cifar_train = datasets.CIFAR10(root=root, train=True, download=True, transform=get_transforms(is_mnist=False))
    cifar_test = datasets.CIFAR10(root=root, train=False, download=True, transform=get_transforms(is_mnist=False))
    
    return {
        'mnist': (mnist_train, mnist_test),
        'fashion': (fmnist_train, fmnist_test),
        'cifar': (cifar_train, cifar_test)
    }

def get_dataloaders(root='./data', batch_size=128):
    all_data = get_datasets(root)
    loaders = {}
    
    for task, (train_ds, test_ds) in all_data.items():
        # Fine-Tuning set (indices 0 to 3000)
        ft_subset = Subset(train_ds, list(range(3000)))
        # Calibration set (indices 3000 to 3128)
        cal_subset = Subset(train_ds, list(range(3000, 3128)))
        
        ft_loader = DataLoader(ft_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        cal_loader = DataLoader(cal_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        loaders[task] = {
            'ft': ft_loader,
            'cal': cal_loader,
            'test': test_loader
        }
        
    return loaders
