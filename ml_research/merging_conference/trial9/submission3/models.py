import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLinear(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFaceLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, y=None):
        # Normalize weights and features
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        
        if y is not None and self.training:
            # Apply margin
            # cos(theta) - m
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, y.view(-1, 1).long(), 1.0)
            output = self.s * (cosine - one_hot * self.m)
        else:
            output = self.s * cosine
            
        return output

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, s=30.0, m=0.35):
        super(SimpleCNN, self).__init__()
        self.use_cosface = use_cosface
        
        # Layer 1 & 2 & 3 & 4 & 5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Layer 6 & 7 & 8 & 9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 11
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.25)
        
        # Layer 14
        if use_cosface:
            self.classifier = CosFaceLinear(128, 10, s=s, m=m)
        else:
            self.classifier = nn.Linear(128, 10)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x, y=None):
        features = self.get_features(x)
        features_relu = F.relu(features)
        features_drop = self.dropout(features_relu)
        
        if self.use_cosface:
            logits = self.classifier(features_drop, y)
        else:
            logits = self.classifier(features_drop)
            
        return logits
