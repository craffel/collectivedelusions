import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasets import load_dataset
import os

print("Testing dataset loading...")

# 1. MNIST
try:
    print("Loading MNIST via torchvision...")
    mnist_train = datasets.MNIST(root='./data', train=False, download=True)
    print("MNIST loaded. Size:", len(mnist_train))
except Exception as e:
    print("MNIST failed:", e)

# 2. SVHN
try:
    print("Loading SVHN via torchvision...")
    svhn_test = datasets.SVHN(root='./data', split='test', download=True)
    print("SVHN loaded. Size:", len(svhn_test))
except Exception as e:
    print("SVHN failed:", e)

# 3. EuroSAT (Torchvision has EuroSAT, but download can be tricky or requires ssl)
try:
    print("Loading EuroSAT via HuggingFace...")
    eurosat_test = load_dataset("graphelier/eurosat", split="train[:10%]") # EuroSAT only has train split in some places
    print("EuroSAT loaded via HF. Size:", len(eurosat_test))
except Exception as e:
    print("EuroSAT failed:", e)

# 4. DTD
try:
    print("Loading DTD via torchvision...")
    dtd_test = datasets.DTD(root='./data', split='test', download=True)
    print("DTD loaded. Size:", len(dtd_test))
except Exception as e:
    print("DTD failed:", e)
