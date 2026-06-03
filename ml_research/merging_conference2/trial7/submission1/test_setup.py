import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

try:
    # Test downloading datasets
    print("Testing MNIST download...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    print("Testing FashionMNIST download...")
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    print("Testing CIFAR10 download...")
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    print("Datasets downloaded successfully!")
    
    # Test model loading
    print("Loading pretrained ResNet-18...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("Pretrained ResNet-18 loaded successfully!")
except Exception as e:
    print("Error during setup test:", e)
