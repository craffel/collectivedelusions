"""Dataset utilities: load torchvision datasets + CLIP-style text class prompts."""
import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import random

DATA_ROOT = os.environ.get("DATA_ROOT", "/fsx/craffel/collectivedelusions/ml_research/testclaude/data")

# Templates (CLIP paper)
TEMPLATE = "a photo of a {}"

# Mapping task name -> (torchvision dataset factory returning (train, test), class_names)
def _mnist(transform_train, transform_test):
    train = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform_test)
    classes = [f"{d}" for d in range(10)]
    classes = [f"the number {c}" for c in classes]
    return train, test, classes

def _cifar10(transform_train, transform_test):
    train = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform_test)
    classes = train.classes
    return train, test, classes

def _cifar100(transform_train, transform_test):
    train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=transform_test)
    classes = train.classes
    return train, test, classes

def _svhn(transform_train, transform_test):
    train = torchvision.datasets.SVHN(DATA_ROOT, split="train", download=True, transform=transform_train)
    test = torchvision.datasets.SVHN(DATA_ROOT, split="test", download=True, transform=transform_test)
    classes = [f"the street number {d}" for d in range(10)]
    return train, test, classes

def _stl10(transform_train, transform_test):
    train = torchvision.datasets.STL10(DATA_ROOT, split="train", download=True, transform=transform_train)
    test = torchvision.datasets.STL10(DATA_ROOT, split="test", download=True, transform=transform_test)
    classes = train.classes
    return train, test, classes

def _eurosat(transform_train, transform_test):
    train_full = torchvision.datasets.EuroSAT(DATA_ROOT, download=True, transform=transform_train)
    # 80/20 split deterministic
    n = len(train_full)
    idx = list(range(n)); random.Random(42).shuffle(idx)
    ntr = int(0.8 * n)
    tr_idx, te_idx = idx[:ntr], idx[ntr:]
    train = Subset(train_full, tr_idx)
    test_full = torchvision.datasets.EuroSAT(DATA_ROOT, download=True, transform=transform_test)
    test = Subset(test_full, te_idx)
    classes = train_full.classes
    return train, test, classes

GTSRB_CLASSES = [
    "speed limit 20 km/h traffic sign", "speed limit 30 km/h traffic sign",
    "speed limit 50 km/h traffic sign", "speed limit 60 km/h traffic sign",
    "speed limit 70 km/h traffic sign", "speed limit 80 km/h traffic sign",
    "end of speed limit 80 km/h traffic sign", "speed limit 100 km/h traffic sign",
    "speed limit 120 km/h traffic sign", "no passing traffic sign",
    "no passing for vehicles over 3.5 metric tons traffic sign",
    "right-of-way at the next intersection traffic sign",
    "priority road traffic sign", "yield traffic sign", "stop traffic sign",
    "no vehicles traffic sign", "vehicles over 3.5 metric tons prohibited traffic sign",
    "no entry traffic sign", "general caution traffic sign",
    "dangerous curve to the left traffic sign", "dangerous curve to the right traffic sign",
    "double curve traffic sign", "bumpy road traffic sign", "slippery road traffic sign",
    "road narrows on the right traffic sign", "road work traffic sign",
    "traffic signals traffic sign", "pedestrians traffic sign",
    "children crossing traffic sign", "bicycles crossing traffic sign",
    "beware of ice or snow traffic sign", "wild animals crossing traffic sign",
    "end of all speed and passing limits traffic sign", "turn right ahead traffic sign",
    "turn left ahead traffic sign", "ahead only traffic sign",
    "go straight or right traffic sign", "go straight or left traffic sign",
    "keep right traffic sign", "keep left traffic sign",
    "roundabout mandatory traffic sign", "end of no passing traffic sign",
    "end of no passing by vehicles over 3.5 metric tons traffic sign",
]

def _gtsrb(transform_train, transform_test):
    train = torchvision.datasets.GTSRB(DATA_ROOT, split="train", download=True, transform=transform_train)
    test = torchvision.datasets.GTSRB(DATA_ROOT, split="test", download=True, transform=transform_test)
    return train, test, GTSRB_CLASSES

def _dtd(transform_train, transform_test):
    train = torchvision.datasets.DTD(DATA_ROOT, split="train", download=True, transform=transform_train)
    test = torchvision.datasets.DTD(DATA_ROOT, split="test", download=True, transform=transform_test)
    classes = train.classes
    return train, test, classes

def _fashion_mnist(transform_train, transform_test):
    train = torchvision.datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform_test)
    classes = train.classes
    return train, test, classes

def _pets(transform_train, transform_test):
    train = torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="trainval", download=True, transform=transform_train)
    test = torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="test", download=True, transform=transform_test)
    classes = train.classes
    return train, test, classes

DATASETS = {
    "MNIST": _mnist,
    "CIFAR10": _cifar10,
    "CIFAR100": _cifar100,
    "SVHN": _svhn,
    "STL10": _stl10,
    "EuroSAT": _eurosat,
    "GTSRB": _gtsrb,
    "DTD": _dtd,
    "FashionMNIST": _fashion_mnist,
    "OxfordPets": _pets,
}

def get_dataset(name, transform_train, transform_test):
    return DATASETS[name](transform_train, transform_test)

def build_label_map(dataset):
    """Return a function that maps a raw dataset sample to (img, label_int)."""
    return dataset
