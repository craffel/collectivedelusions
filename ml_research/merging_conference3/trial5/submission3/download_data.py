import torchvision
import torchvision.transforms as transforms
import os

def download_datasets():
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root='./data', train=True, download=True)
    torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    print("Downloading FashionMNIST...")
    torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    
    print("Downloading CIFAR10...")
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    print("Downloading SVHN...")
    torchvision.datasets.SVHN(root='./data', split='train', download=True)
    torchvision.datasets.SVHN(root='./data', split='test', download=True)
    print("All datasets downloaded successfully!")

if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)
    download_datasets()
