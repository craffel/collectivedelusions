import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-channel input (28x28 grayscale)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return x

class TaskHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, x, task_name):
        features = self.backbone(x)
        return self.heads[task_name](features)
