import time
print("1. Starting diagnostic...")

print("2. Importing torch...")
import torch
print("Imported torch.")

print("3. Importing torchvision...")
import torchvision
import torchvision.transforms as transforms
print("Imported torchvision.")

print("4. Importing timm...")
import timm
print("Imported timm.")

print("5. Configuring matplotlib Agg backend and importing...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("Imported matplotlib.")

print("6. Loading MNIST...")
transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
print("Loaded MNIST.")

print("7. Loading FashionMNIST...")
train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
print("Loaded FashionMNIST.")

print("8. Loading CIFAR-10...")
transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
print("Loaded CIFAR-10.")

print("9. Loading SVHN...")
train_svhn = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_color)
print("Loaded SVHN.")

print("10. Instantiating timm model...")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
print("Instantiated model successfully!")
