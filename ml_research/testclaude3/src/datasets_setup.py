"""Dataset utilities for CLIP-based multi-task vision experiments.

We use small/medium torchvision datasets to keep fine-tuning under a 6-hour budget.
Each task returns: (train_loader, val_loader, classnames, prompt_template).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tvd
from torchvision import transforms as T

DATA_ROOT = "/fsx/craffel/collectivedelusions/ml_research/testclaude3/data"

# Use a CLIP-style preprocessing: resize + center-crop to 224, normalize with CLIP stats.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

train_tf = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])
test_tf = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

# Per-dataset transform overrides (greyscale -> RGB for MNIST)
def to_rgb(x):
    return x.convert("RGB")

train_tf_gray = T.Compose([T.Lambda(to_rgb)] + train_tf.transforms)
test_tf_gray = T.Compose([T.Lambda(to_rgb)] + test_tf.transforms)


@dataclass
class TaskBundle:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader  # we use this for eval
    classnames: list[str]
    prompt_template: str  # e.g. "a photo of a {}."


# Compact prompt templates (single-template variants used in CLIP)
PROMPTS = {
    "MNIST": "a photo of the number: \"{}\".",
    "SVHN": "a photo of the number: \"{}\".",
    "CIFAR10": "a photo of a {}.",
    "CIFAR100": "a photo of a {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "GTSRB": "a zoomed in photo of a \"{}\" traffic sign.",
    "DTD": "a photo of a {} texture.",
}

# Human-readable classnames for non-trivial datasets.
CIFAR100_CLASSES = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear",
    "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman",
    "worm",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

EUROSAT_CLASSES = [
    "annual crop land", "forest", "brushland or shrubland", "highway or road",
    "industrial buildings or commercial buildings", "pasture land",
    "permanent crop land", "residential buildings or homes or apartments",
    "river", "lake or sea",
]

GTSRB_CLASSES_43 = [
    "red and white circle 20 kph speed limit",
    "red and white circle 30 kph speed limit",
    "red and white circle 50 kph speed limit",
    "red and white circle 60 kph speed limit",
    "red and white circle 70 kph speed limit",
    "red and white circle 80 kph speed limit",
    "end / 80 kph speed limit",
    "red and white circle 100 kph speed limit",
    "red and white circle 120 kph speed limit",
    "red and white circle red car and black car no passing",
    "red and white circle red truck and black car no passing",
    "red and white triangle road intersection warning",
    "white and yellow diamond priority road",
    "red and white upside down triangle yield right-of-way",
    "stop",
    "empty red and white circle",
    "red and white circle no truck entry",
    "red circle with white horizonal stripe no entry",
    "red and white triangle with exclamation mark warning",
    "red and white triangle with black left curve approaching warning",
    "red and white triangle with black right curve approaching warning",
    "red and white triangle with black double curve approaching warning",
    "red and white triangle rough / bumpy road warning",
    "red and white triangle car skidding / slipping warning",
    "red and white triangle with merging / narrow lanes warning",
    "red and white triangle with person digging / construction / road work warning",
    "red and white triangle with traffic light approaching warning",
    "red and white triangle with person walking warning",
    "red and white triangle with child and person walking warning",
    "red and white triangle with bicyle warning",
    "red and white triangle with snowflake / ice warning",
    "red and white triangle with deer warning",
    "white circle with gray strike bar no speed limit",
    "blue circle with white right turn arrow mandatory",
    "blue circle with white left turn arrow mandatory",
    "blue circle with white forward arrow mandatory",
    "blue circle with white forward or right turn arrow mandatory",
    "blue circle with white forward or left turn arrow mandatory",
    "blue circle with white keep right arrow mandatory",
    "blue circle with white keep left arrow mandatory",
    "blue circle with white arrows indicating a traffic circle",
    "white circle with gray strike bar indicating no passing for cars has ended",
    "white circle with gray strike bar indicating no passing for trucks has ended",
]


def _maybe_subset(ds, max_samples: int | None, seed: int = 0):
    if max_samples is None or len(ds) <= max_samples:
        return ds
    g = random.Random(seed)
    idxs = list(range(len(ds)))
    g.shuffle(idxs)
    return Subset(ds, idxs[:max_samples])


def build_task(name: str,
               batch_size: int = 128,
               num_workers: int = 4,
               max_train: int | None = None,
               max_test: int | None = None) -> TaskBundle:
    """Build a TaskBundle for the named task.

    The first call downloads the dataset to DATA_ROOT. Failures (e.g., no internet)
    propagate to the caller; the agent skips missing datasets.
    """
    name = name.upper()
    if name == "MNIST":
        train = tvd.MNIST(DATA_ROOT, train=True, download=True, transform=train_tf_gray)
        test = tvd.MNIST(DATA_ROOT, train=False, download=True, transform=test_tf_gray)
        classnames = [str(i) for i in range(10)]
        prompt = PROMPTS["MNIST"]
        key = "MNIST"
    elif name == "SVHN":
        train = tvd.SVHN(DATA_ROOT, split="train", download=True, transform=train_tf)
        test = tvd.SVHN(DATA_ROOT, split="test", download=True, transform=test_tf)
        classnames = [str(i) for i in range(10)]
        prompt = PROMPTS["SVHN"]
        key = "SVHN"
    elif name == "CIFAR10":
        train = tvd.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
        test = tvd.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
        classnames = CIFAR10_CLASSES
        prompt = PROMPTS["CIFAR10"]
        key = "CIFAR10"
    elif name == "CIFAR100":
        train = tvd.CIFAR100(DATA_ROOT, train=True, download=True, transform=train_tf)
        test = tvd.CIFAR100(DATA_ROOT, train=False, download=True, transform=test_tf)
        classnames = CIFAR100_CLASSES
        prompt = PROMPTS["CIFAR100"]
        key = "CIFAR100"
    elif name == "EUROSAT":
        # 27k images, 10 classes
        ds_full = tvd.EuroSAT(DATA_ROOT, download=True, transform=train_tf)
        # No official train/test split: use a fixed 80/20 split
        N = len(ds_full)
        idx = list(range(N))
        rng = random.Random(0)
        rng.shuffle(idx)
        n_train = int(0.8 * N)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        train_full = tvd.EuroSAT(DATA_ROOT, download=False, transform=train_tf)
        test_full = tvd.EuroSAT(DATA_ROOT, download=False, transform=test_tf)
        train = Subset(train_full, train_idx)
        test = Subset(test_full, test_idx)
        classnames = EUROSAT_CLASSES
        prompt = PROMPTS["EuroSAT"]
        key = "EuroSAT"
    elif name == "GTSRB":
        train = tvd.GTSRB(DATA_ROOT, split="train", download=True, transform=train_tf)
        test = tvd.GTSRB(DATA_ROOT, split="test", download=True, transform=test_tf)
        classnames = GTSRB_CLASSES_43
        prompt = PROMPTS["GTSRB"]
        key = "GTSRB"
    elif name == "DTD":
        train = tvd.DTD(DATA_ROOT, split="train", download=True, transform=train_tf)
        test = tvd.DTD(DATA_ROOT, split="test", download=True, transform=test_tf)
        classnames = train.classes
        # Replace underscores in classnames with spaces
        classnames = [c.replace("_", " ").replace("-", " ") for c in classnames]
        prompt = PROMPTS["DTD"]
        key = "DTD"
    else:
        raise ValueError(f"Unknown task: {name}")

    train = _maybe_subset(train, max_train)
    test_ds = _maybe_subset(test, max_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    test_loader = val_loader  # same for our purposes

    return TaskBundle(name=key, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, classnames=classnames,
                      prompt_template=prompt)


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]
