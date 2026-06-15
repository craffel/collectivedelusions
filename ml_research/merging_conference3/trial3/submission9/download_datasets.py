import torchvision
import torchvision.transforms as transforms

def main():
    print("Pre-downloading datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # MNIST
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # FashionMNIST
    print("Downloading FashionMNIST...")
    torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # CIFAR10
    print("Downloading CIFAR10...")
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # SVHN
    print("Downloading SVHN...")
    torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    print("Datasets successfully downloaded and cached.")

if __name__ == '__main__':
    main()
