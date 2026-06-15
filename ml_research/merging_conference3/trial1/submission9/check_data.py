import torchvision
import torch

try:
    print("Testing MNIST download...")
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    print("MNIST success! Size:", len(train_mnist))
    
    print("Testing FashionMNIST download...")
    train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    print("FashionMNIST success! Size:", len(train_fashion))
    
    print("Testing KMNIST download...")
    train_kmnist = torchvision.datasets.KMNIST(root='./data', train=True, download=True)
    print("KMNIST success! Size:", len(train_kmnist))
    
    print("All datasets downloaded successfully!")
except Exception as e:
    print("Error during dataset download:", e)
