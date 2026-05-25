import torchvision
import os

print("Pre-downloading MNIST, FashionMNIST, and KMNIST datasets...")
os.makedirs("data", exist_ok=True)

torchvision.datasets.MNIST(root="data", train=True, download=True)
torchvision.datasets.MNIST(root="data", train=False, download=True)

torchvision.datasets.FashionMNIST(root="data", train=True, download=True)
torchvision.datasets.FashionMNIST(root="data", train=False, download=True)

torchvision.datasets.KMNIST(root="data", train=True, download=True)
torchvision.datasets.KMNIST(root="data", train=False, download=True)

print("Pre-downloading finished successfully!")
