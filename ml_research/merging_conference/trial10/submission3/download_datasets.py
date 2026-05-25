import torchvision
import torchvision.transforms as transforms
import os

def download():
    print("Downloading datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    os.makedirs("data", exist_ok=True)
    
    print("Downloading MNIST...")
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    print(f"MNIST Train size: {len(mnist_train)}, Test size: {len(mnist_test)}")
    
    print("Downloading FashionMNIST...")
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    print(f"FashionMNIST Train size: {len(fmnist_train)}, Test size: {len(fmnist_test)}")
    
    print("Downloading KMNIST...")
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    print(f"KMNIST Test size: {len(kmnist_test)}")
    print("All datasets downloaded and verified successfully!")

if __name__ == "__main__":
    download()
