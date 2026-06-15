import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Tuning Expert Heads on ResNet-18 features...")

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity()
resnet18 = resnet18.to(device)
resnet18.eval()

# Let's test 128x128 resize for speed
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading datasets...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Let's extract 500 samples for training
targets = np.array(mnist_train.targets)
train_idx = []
for c in range(10):
    c_idx = np.where(targets == c)[0]
    train_idx.extend(c_idx[:50]) # 500 total

test_targets = np.array(mnist_test.targets)
test_idx = []
for c in range(10):
    c_idx = np.where(test_targets == c)[0]
    test_idx.extend(c_idx[:100]) # 1000 total

def extract(dataset, indices):
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(resnet18(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats).to(device), torch.cat(labels).to(device)

print("Extracting features...")
train_feats, train_labels = extract(mnist_train, train_idx)
test_feats, test_labels = extract(mnist_test, test_idx)

# Tune LR and WD
for lr in [0.01, 0.001, 0.0005]:
    for wd in [0.0, 1e-4, 1e-3]:
        head = nn.Linear(512, 10, bias=True).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        
        for epoch in range(100):
            opt.zero_grad()
            loss = crit(head(train_feats), train_labels)
            loss.backward()
            opt.step()
            
        with torch.no_grad():
            preds = torch.argmax(head(test_feats), dim=-1)
            acc = (preds == test_labels).float().mean().item() * 100
            print(f"lr={lr}, wd={wd} -> Test Acc: {acc:.2f}%")
