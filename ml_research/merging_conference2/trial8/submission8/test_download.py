import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import os

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

# Try downloading datasets to a local folder 'data'
os.makedirs("data", exist_ok=True)
try:
    print("Downloading MNIST...")
    mnist = MNIST(root="./data", train=True, download=True)
    print("MNIST downloaded. Size:", len(mnist))
    
    print("Downloading FashionMNIST...")
    fmnist = FashionMNIST(root="./data", train=True, download=True)
    print("FashionMNIST downloaded. Size:", len(fmnist))
    
    print("Downloading CIFAR10...")
    cifar = CIFAR10(root="./data", train=True, download=True)
    print("CIFAR10 downloaded. Size:", len(cifar))
except Exception as e:
    print("Error downloading datasets:", e)

# Try loading ResNet-18
try:
    print("Loading pretrained ResNet-18...")
    model = torchvision.models.resnet18(pretrained=True)
    print("Successfully loaded pretrained ResNet-18!")
except Exception as e:
    print("Error loading ResNet-18:", e)
