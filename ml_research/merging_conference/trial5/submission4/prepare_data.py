import torchvision
import torchvision.transforms as transforms

print("Downloading CIFAR-10...")
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

print("Downloading SVHN...")
svhn_train = torchvision.datasets.SVHN(root='./data', split='train', download=True)
svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True)

print("All datasets downloaded successfully!")
