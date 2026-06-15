import torchvision
import torchvision.transforms as transforms
import torch

try:
    # 1. MNIST
    mnist_train = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    )
    print("MNIST Loaded successfully, size:", len(mnist_train))
except Exception as e:
    print("MNIST failed:", e)

try:
    # 2. FashionMNIST
    fmnist_train = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    )
    print("FashionMNIST Loaded successfully, size:", len(fmnist_train))
except Exception as e:
    print("FashionMNIST failed:", e)

try:
    # 3. CIFAR10
    cifar_train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    print("CIFAR-10 Loaded successfully, size:", len(cifar_train))
except Exception as e:
    print("CIFAR-10 failed:", e)

try:
    # 4. SVHN
    svhn_train = torchvision.datasets.SVHN(
        root='./data', 
        split='train', 
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    print("SVHN Loaded successfully, size:", len(svhn_train))
except Exception as e:
    print("SVHN failed:", e)
