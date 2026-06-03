import torchvision

print("Downloading CIFAR-100...")
torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
print("CIFAR-100 downloaded successfully!")
