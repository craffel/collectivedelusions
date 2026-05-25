import torchvision
import torchvision.transforms as transforms
import os

print("Downloading datasets...")
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

try:
    print("Downloading CIFAR10...")
    cifar10 = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    print("CIFAR10 downloaded successfully.")
except Exception as e:
    print(f"Error downloading CIFAR10: {e}")

try:
    print("Downloading SVHN...")
    svhn = torchvision.datasets.SVHN(root=data_dir, split="train", download=True)
    svhn_test = torchvision.datasets.SVHN(root=data_dir, split="test", download=True)
    print("SVHN downloaded successfully.")
except Exception as e:
    print(f"Error downloading SVHN: {e}")

try:
    print("Downloading FashionMNIST...")
    fmnist = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True)
    fmnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True)
    print("FashionMNIST downloaded successfully.")
except Exception as e:
    print(f"Error downloading FashionMNIST: {e}")

print("Dataset download check finished.")
