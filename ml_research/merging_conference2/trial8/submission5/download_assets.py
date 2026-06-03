import os
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

print("Downloading pre-trained ResNet18 weights...")
# Force downloading of resnet18 weights
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
print("ResNet18 weights downloaded successfully!")

print("Downloading MNIST...")
torchvision.datasets.MNIST(root='./data', train=True, download=True)
torchvision.datasets.MNIST(root='./data', train=False, download=True)

print("Downloading Fashion-MNIST...")
torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)

print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

print("All downloads completed!")
