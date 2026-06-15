import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

try:
    print("Downloading MNIST...")
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    print("Downloading FashionMNIST...")
    fmnist = datasets.FashionMNIST(root='./data', train=True, download=True)
    print("Downloading CIFAR10...")
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True)
    print("Downloading SVHN...")
    svhn = datasets.SVHN(root='./data', split='train', download=True)
    print("All datasets successfully downloaded!")
except Exception as e:
    print(f"Error: {e}")
