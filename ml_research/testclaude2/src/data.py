"""Image classification dataset loaders for model merging benchmark.

Returns torch Datasets of (PIL.Image RGB, int label) for compatibility with
HF CLIPImageProcessor (we run the processor inside the DataLoader collate).
"""
from __future__ import annotations
import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset


def _to_rgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


class HFImageDataset(Dataset):
    """Wraps a Hugging Face datasets.Dataset with (image_key, label_key)."""

    def __init__(self, hf_ds, image_key: str, label_key: str):
        self.ds = hf_ds
        self.ik = image_key
        self.lk = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = _to_rgb(row[self.ik])
        return img, int(row[self.lk])


def _torchvision_to_dataset(tv_ds):
    """Wrap a torchvision dataset so __getitem__ returns (PIL.RGB, int)."""

    class _W(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            return _to_rgb(img), int(label)

    return _W(tv_ds)


# Each entry: name -> (loader_fn, num_classes, class_names_or_none)
TASKS = ["cifar10", "cifar100", "mnist", "svhn", "fashionmnist", "eurosat", "gtsrb", "dtd"]


def load_task(name: str, root: str) -> Tuple[Dataset, Dataset, int, List[str]]:
    """Return (train_ds, val_ds, num_classes, class_names)."""
    os.makedirs(root, exist_ok=True)
    if name == "cifar10":
        from torchvision.datasets import CIFAR10
        tr = CIFAR10(root, train=True, download=True)
        te = CIFAR10(root, train=False, download=True)
        cn = tr.classes
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 10, cn
    if name == "cifar100":
        from torchvision.datasets import CIFAR100
        tr = CIFAR100(root, train=True, download=True)
        te = CIFAR100(root, train=False, download=True)
        cn = tr.classes
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 100, cn
    if name == "mnist":
        from torchvision.datasets import MNIST
        tr = MNIST(root, train=True, download=True)
        te = MNIST(root, train=False, download=True)
        cn = [str(i) for i in range(10)]
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 10, cn
    if name == "svhn":
        from torchvision.datasets import SVHN
        tr = SVHN(root, split="train", download=True)
        te = SVHN(root, split="test", download=True)
        cn = [str(i) for i in range(10)]
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 10, cn
    if name == "fashionmnist":
        from torchvision.datasets import FashionMNIST
        tr = FashionMNIST(root, train=True, download=True)
        te = FashionMNIST(root, train=False, download=True)
        cn = tr.classes
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 10, cn
    if name == "eurosat":
        from torchvision.datasets import EuroSAT
        from torch.utils.data import Subset
        ds = EuroSAT(root, download=True)
        n = len(ds)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=g).tolist()
        cut = int(0.8 * n)
        tr = Subset(ds, perm[:cut]); te = Subset(ds, perm[cut:])
        cn = ds.classes
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 10, cn
    if name == "gtsrb":
        from torchvision.datasets import GTSRB
        tr = GTSRB(root, split="train", download=True)
        te = GTSRB(root, split="test", download=True)
        cn = [f"class_{i}" for i in range(43)]
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), 43, cn
    if name == "dtd":
        from torchvision.datasets import DTD
        tr = DTD(root, split="train", download=True)
        te = DTD(root, split="test", download=True)
        cn = tr.classes
        return _torchvision_to_dataset(tr), _torchvision_to_dataset(te), len(cn), cn
    raise ValueError(f"unknown task {name}")
