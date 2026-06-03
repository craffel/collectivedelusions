import torchvision
import torchvision.transforms as T

print("Downloading CIFAR10...")
torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

print("Downloading SVHN...")
torchvision.datasets.SVHN(root="./data", split="train", download=True)
torchvision.datasets.SVHN(root="./data", split="test", download=True)

print("Downloading FashionMNIST...")
torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)

print("Datasets downloaded successfully!")
