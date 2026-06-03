import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Expert(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        if pretrained:
            # Load IMAGENET1K_V1 pre-trained weights
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace the original fc layer with an Identity mapping
        self.backbone.fc = nn.Identity()
        # Create the task-specific classification head
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

class MLPExpert(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return self.fc(x)

def get_model(arch, num_classes=10, pretrained=True):
    if arch == 'resnet18':
        return ResNet18Expert(num_classes=num_classes, pretrained=pretrained)
    elif arch == 'mlp':
        return MLPExpert(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
