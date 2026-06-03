import torchvision
import torchvision.transforms as transforms
import torch

def check_datasets():
    print("Checking datasets...")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    try:
        print("Downloading MNIST...")
        mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        print(f"MNIST download successful! Size: {len(mnist)}")
    except Exception as e:
        print(f"MNIST error: {e}")

    try:
        print("Downloading FashionMNIST...")
        fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        print(f"FashionMNIST download successful! Size: {len(fmnist)}")
    except Exception as e:
        print(f"FashionMNIST error: {e}")

    try:
        print("Downloading CIFAR10...")
        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        print(f"CIFAR10 download successful! Size: {len(cifar10)}")
    except Exception as e:
        print(f"CIFAR10 error: {e}")

if __name__ == "__main__":
    check_datasets()
