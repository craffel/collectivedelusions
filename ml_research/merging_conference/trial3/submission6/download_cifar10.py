import torchvision
import os

print("Downloading CIFAR-10 dataset...")
os.makedirs("./data", exist_ok=True)
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print("CIFAR-10 downloaded successfully!")
