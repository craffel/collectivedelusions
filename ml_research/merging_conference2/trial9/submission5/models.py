import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initialize a standard ResNet-18
        self.model = models.resnet18(weights=None)
        # Adapt conv1 for 32x32 images: change kernel size from 7x7 to 3x3, stride to 1, padding to 1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Replace maxpool with identity to preserve spatial resolution
        self.model.maxpool = nn.Identity()
        # Adapt fc layer for the target number of classes
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

class MLPCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Sanity checks
    resnet = ResNet18CIFAR(10)
    mlp = MLPCIFAR(10)
    x = torch.randn(2, 3, 32, 32)
    out_resnet = resnet(x)
    out_mlp = mlp(x)
    print(f"ResNet-18 output shape: {out_resnet.shape}")
    print(f"MLP output shape: {out_mlp.shape}")
