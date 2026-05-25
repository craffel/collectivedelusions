import torchvision
print("Downloading CIFAR10 train dataset...")
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
print("Downloading CIFAR10 test dataset...")
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print("Download complete!")
