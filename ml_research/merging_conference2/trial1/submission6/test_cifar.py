import torchvision.datasets as datasets
try:
    print("Loading CIFAR-10...")
    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True)
    print("CIFAR-10 loaded. Size:", len(cifar10))
except Exception as e:
    print("CIFAR-10 failed:", e)

try:
    print("Loading CIFAR-100...")
    cifar100 = datasets.CIFAR100(root='./data', train=False, download=True)
    print("CIFAR-100 loaded. Size:", len(cifar100))
except Exception as e:
    print("CIFAR-100 failed:", e)
