import torchvision
import torchvision.transforms as transforms
import torch

def check():
    print("Checking dataset downloads...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        print(f"MNIST train size: {len(mnist_train)}, test size: {len(mnist_test)}")
        
        fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        print(f"FashionMNIST train size: {len(fmnist_train)}, test size: {len(fmnist_test)}")
        
        kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        print(f"KMNIST train size: {len(kmnist_train)}, test size: {len(kmnist_test)}")
        
        print("All datasets downloaded successfully!")
    except Exception as e:
        print(f"Error downloading datasets: {e}")

if __name__ == "__main__":
    check()
