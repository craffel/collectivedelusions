import torchvision
import torchvision.transforms as transforms
import os

print("Starting dataset download check...")
os.makedirs("data", exist_ok=True)

try:
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root="./data", train=True, download=True)
    torchvision.datasets.MNIST(root="./data", train=False, download=True)
    print("MNIST downloaded.")
except Exception as e:
    print("Error downloading MNIST:", e)

try:
    print("Downloading FashionMNIST...")
    torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    print("FashionMNIST downloaded.")
except Exception as e:
    print("Error downloading FashionMNIST:", e)

try:
    print("Downloading CIFAR10...")
    torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    print("CIFAR10 downloaded.")
except Exception as e:
    print("Error downloading CIFAR10:", e)

print("Dataset check completed.")
