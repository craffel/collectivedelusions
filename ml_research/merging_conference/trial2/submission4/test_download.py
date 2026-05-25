import torchvision
import torchvision.transforms as transforms

print("Attempting to load CIFAR-10...")
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
print("CIFAR-10 loaded. Size:", len(cifar_train))

print("Attempting to load SVHN...")
svhn_train = torchvision.datasets.SVHN(root='./data', split='train', download=True)
print("SVHN loaded. Size:", len(svhn_train))
