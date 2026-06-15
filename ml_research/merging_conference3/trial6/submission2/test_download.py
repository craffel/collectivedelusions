import torchvision
import torchvision.transforms as transforms
import torch

def test_downloads():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Testing MNIST...")
    try:
        mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        print(f"MNIST Success: {len(mnist)} samples")
    except Exception as e:
        print(f"MNIST Failed: {e}")
        
    print("Testing FashionMNIST...")
    try:
        fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        print(f"FashionMNIST Success: {len(fmnist)} samples")
    except Exception as e:
        print(f"FashionMNIST Failed: {e}")
        
    print("Testing CIFAR-10...")
    try:
        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        print(f"CIFAR-10 Success: {len(cifar10)} samples")
    except Exception as e:
        print(f"CIFAR-10 Failed: {e}")
        
    print("Testing SVHN...")
    try:
        svhn = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        print(f"SVHN Success: {len(svhn)} samples")
    except Exception as e:
        print(f"SVHN Failed: {e}")

if __name__ == "__main__":
    test_downloads()
