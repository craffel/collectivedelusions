import torch
import torch.nn as nn

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.relu on conv2 (no pool)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(3136, 128)
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool2(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.linear(x)

class MergedModel(nn.Module):
    def __init__(self, encoder, heads):
        super(MergedModel, self).__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads)
        
    def forward(self, x, task_idx):
        features = self.encoder(x)
        return self.heads[task_idx](features)
