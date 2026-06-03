import torchvision

def download():
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root='./data', train=True, download=True)
    torchvision.datasets.MNIST(root='./data', train=False, download=True)
    print("Downloading FashionMNIST...")
    torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    print("Downloading CIFAR10...")
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    print("Done downloading datasets!")

if __name__ == "__main__":
    download()
