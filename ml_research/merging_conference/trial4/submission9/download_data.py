import torchvision

try:
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root="./data", train=True, download=True)
    torchvision.datasets.MNIST(root="./data", train=False, download=True)
    print("Downloading FashionMNIST...")
    torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    print("Downloading KMNIST...")
    torchvision.datasets.KMNIST(root="./data", train=True, download=True)
    torchvision.datasets.KMNIST(root="./data", train=False, download=True)
    print("All datasets downloaded successfully!")
except Exception as e:
    print(f"Error downloading datasets: {e}")
