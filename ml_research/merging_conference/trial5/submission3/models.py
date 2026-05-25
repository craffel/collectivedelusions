import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    """
    ResNet-18 Encoder extracting 512-dimensional feature embeddings.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet-18 with default ImageNet weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace classification head with Identity to output embeddings
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

class ClassificationHead(nn.Module):
    """
    Task-specific classification head mapping 512-dimensional embeddings to 10 classes.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(x)
