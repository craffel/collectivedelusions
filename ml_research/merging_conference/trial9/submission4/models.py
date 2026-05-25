import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLinear(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        # Normalize weights and features to spherical unit domain
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(x_norm, weight_norm) # output shape: [batch_size, out_features]
        
        if self.training and label is not None:
            # Apply additive angular margin (cos(theta) - m)
            phi = cosine - self.m
            
            # One-hot representation of labels
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            
            # Combine: phi for target class, original cosine for other classes
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            return output
        else:
            return cosine * self.s

class SimpleCNN(nn.Module):
    def __init__(self, is_cosface=False, s=30.0, m=0.35):
        super().__init__()
        self.is_cosface = is_cosface
        
        # Layer 2-5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 6-9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 11
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # Layer 13
        self.dropout = nn.Dropout(p=0.25)
        
        # Layer 14
        if is_cosface:
            self.classifier = CosFaceLinear(128, 10, s=s, m=m)
        else:
            self.classifier = nn.Linear(128, 10)

    def get_features(self, x):
        # Extract features (the 128-dimensional penultimate representation)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x, label=None):
        features = self.get_features(x)
        features = self.dropout(features)
        if self.is_cosface:
            return self.classifier(features, label)
        else:
            return self.classifier(features)
