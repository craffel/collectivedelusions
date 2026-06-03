import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

print("Downloading pretrained ResNet-18 weights...")
try:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    print("Successfully downloaded ResNet-18 weights.")
except Exception as e:
    print(f"Error downloading weights: {e}")

print("Downloading datasets...")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

try:
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print("Successfully downloaded MNIST.")
except Exception as e:
    print(f"Error downloading MNIST: {e}")

try:
    print("Downloading Fashion-MNIST...")
    torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    print("Successfully downloaded Fashion-MNIST.")
except Exception as e:
    print(f"Error downloading Fashion-MNIST: {e}")

try:
    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("Successfully downloaded CIFAR-10.")
except Exception as e:
    print(f"Error downloading CIFAR-10: {e}")

print("All downloads complete.")
