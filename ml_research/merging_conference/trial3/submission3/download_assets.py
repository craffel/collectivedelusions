import os
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights

def main():
    print("Downloading datasets...")
    # CIFAR10
    datasets.CIFAR10(root="./data", train=True, download=True)
    datasets.CIFAR10(root="./data", train=False, download=True)
    print("CIFAR10 downloaded successfully.")

    # SVHN
    datasets.SVHN(root="./data", split="train", download=True)
    datasets.SVHN(root="./data", split="test", download=True)
    print("SVHN downloaded successfully.")

    print("Downloading pretrained weights...")
    os.environ["TORCH_HOME"] = "./torch_cache"
    # ResNet18
    resnet18(weights=ResNet18_Weights.DEFAULT)
    print("ResNet18 weights downloaded successfully.")

    # ViT-B/16
    vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    print("ViT-B/16 weights downloaded successfully.")

if __name__ == "__main__":
    main()
