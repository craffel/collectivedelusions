import torchvision
import os

print("Downloading datasets...")
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# 1. MNIST
print("Downloading MNIST...")
torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
torchvision.datasets.MNIST(root=data_dir, train=False, download=True)

# 2. FashionMNIST
print("Downloading FashionMNIST...")
torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True)
torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True)

# 3. CIFAR10
print("Downloading CIFAR10...")
torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

# 4. SVHN
print("Downloading SVHN...")
torchvision.datasets.SVHN(root=data_dir, split='train', download=True)
torchvision.datasets.SVHN(root=data_dir, split='test', download=True)

print("All datasets downloaded and cached successfully!")
