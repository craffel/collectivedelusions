import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

def get_dataset(name, root='./data', split='train'):
    transform_list = []
    if name in ['mnist', 'fashion_mnist']:
        transform_list.append(GrayscaleToRGB())
    
    transform_list.extend([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose(transform_list)

    if name == 'mnist':
        ds = torchvision.datasets.MNIST(root=root, train=(split != 'test'), download=True, transform=transform)
    elif name == 'fashion_mnist':
        ds = torchvision.datasets.FashionMNIST(root=root, train=(split != 'test'), download=True, transform=transform)
    elif name == 'cifar10':
        ds = torchvision.datasets.CIFAR10(root=root, train=(split != 'test'), download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds

_cache = {}

def get_splits(name, root='./data'):
    if name not in _cache:
        train_ds = get_dataset(name, root=root, split='train')
        test_ds = get_dataset(name, root=root, split='test')
        
        # Fine-Tuning Set (first 5,000 samples)
        ft_ds = Subset(train_ds, list(range(5000)))
        
        # Calibration Set (next 128 samples: 5,000 to 5,128)
        cal_ds = Subset(train_ds, list(range(5000, 5128)))
        
        _cache[name] = (ft_ds, cal_ds, test_ds)
    return _cache[name]

if __name__ == '__main__':
    print("Testing data loading and splits...")
    for name in ['mnist', 'fashion_mnist', 'cifar10']:
        ft, cal, test = get_splits(name)
        print(f"{name}: Fine-tuning size={len(ft)}, Calibration size={len(cal)}, Test size={len(test)}")
        # Check shapes
        loader = DataLoader(cal, batch_size=4, shuffle=False)
        x, y = next(iter(loader))
        print(f"Sample batch shape: {x.shape}, labels: {y}")
