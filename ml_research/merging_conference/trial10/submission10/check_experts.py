import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np

# Same architectures and setup
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
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(x_norm, w_norm)
        if label is None:
            return self.s * cos_theta
        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = self.s * (cos_theta - one_hot * self.m)
        return output

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.use_cosface = use_cosface
        if use_cosface:
            self.classifier = CosFaceLinear(128, 10)
        else:
            self.classifier = nn.Linear(128, 10)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

    def forward(self, x, label=None):
        features = self.extract_features(x)
        if self.use_cosface:
            return self.classifier(features, label)
        else:
            return self.classifier(features)

# Load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
mnist_subset = torch.utils.data.Subset(mnist_train, list(range(10000)))
fashion_subset = torch.utils.data.Subset(fashion_train, list(range(10000)))

class JointDataset(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
    def __len__(self):
        return len(self.ds1) + len(self.ds2)
    def __getitem__(self, idx):
        if idx < len(self.ds1):
            return self.ds1[idx]
        else:
            return self.ds2[idx - len(self.ds1)]

joint_train = JointDataset(mnist_subset, fashion_subset)
joint_loader = torch.utils.data.DataLoader(joint_train, batch_size=64, shuffle=True)
mnist_loader = torch.utils.data.DataLoader(mnist_subset, batch_size=64, shuffle=True)
fashion_loader = torch.utils.data.DataLoader(fashion_subset, batch_size=64, shuffle=True)

def train_model(model, dataloader, epochs, lr, weight_decay, use_cosface=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            if use_cosface:
                outputs = model(images, labels)
            else:
                outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

# Train standard experts
print("Training standard experts...")
base_std = SimpleCNN(use_cosface=False)
train_model(base_std, joint_loader, epochs=1, lr=1e-3, weight_decay=1e-4)
expert0_std = SimpleCNN(use_cosface=False)
expert0_std.load_state_dict(base_std.state_dict())
train_model(expert0_std, mnist_loader, epochs=1, lr=2e-4, weight_decay=1e-5)
expert1_std = SimpleCNN(use_cosface=False)
expert1_std.load_state_dict(base_std.state_dict())
train_model(expert1_std, fashion_loader, epochs=1, lr=2e-4, weight_decay=1e-5)

# Train CosFace experts
print("Training CosFace experts...")
base_cos = SimpleCNN(use_cosface=True)
train_model(base_cos, joint_loader, epochs=1, lr=1e-3, weight_decay=1e-4, use_cosface=True)
expert0_cos = SimpleCNN(use_cosface=True)
expert0_cos.load_state_dict(base_cos.state_dict())
train_model(expert0_cos, mnist_loader, epochs=1, lr=2e-4, weight_decay=1e-5, use_cosface=True)
expert1_cos = SimpleCNN(use_cosface=True)
expert1_cos.load_state_dict(base_cos.state_dict())
train_model(expert1_cos, fashion_loader, epochs=1, lr=2e-4, weight_decay=1e-5, use_cosface=True)

# Test datasets
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

# Evaluate function
def eval_expert(expert, dataset, name, noisy=False):
    expert.eval()
    correct = 0
    total = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for img, lbl in loader:
            if noisy:
                noise = torch.randn_like(img) * 0.6
                img = torch.clamp(img + noise, -1.0, 1.0)
            outputs = expert(img)
            _, pred = torch.max(outputs, 1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
    print(f"{name} accuracy (noisy={noisy}): {correct / total * 100:.2f}%")

eval_expert(expert0_std, mnist_test, "Expert0 (Std, MNIST)", noisy=False)
eval_expert(expert0_std, mnist_test, "Expert0 (Std, MNIST)", noisy=True)
eval_expert(expert1_std, fashion_test, "Expert1 (Std, Fashion)", noisy=False)
eval_expert(expert1_std, fashion_test, "Expert1 (Std, Fashion)", noisy=True)

eval_expert(expert0_cos, mnist_test, "Expert0 (Cos, MNIST)", noisy=False)
eval_expert(expert0_cos, mnist_test, "Expert0 (Cos, MNIST)", noisy=True)
eval_expert(expert1_cos, fashion_test, "Expert1 (Cos, Fashion)", noisy=False)
eval_expert(expert1_cos, fashion_test, "Expert1 (Cos, Fashion)", noisy=True)
