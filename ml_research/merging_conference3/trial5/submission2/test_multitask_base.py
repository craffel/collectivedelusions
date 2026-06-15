import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Deep12LayerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.ModuleList([
            ConvBlock(3, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

device = torch.device("cpu")

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
transform_color = transforms.Compose([
    transforms.ToTensor(),
])

print("Loading datasets...")
datasets = {
    0: (torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray),
        torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)),
    1: (torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray),
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)),
    2: (torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_color),
        torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color)),
    3: (torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform_color),
        torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_color))
}

print("Constructing subsets...")
sub_datasets = {}
for k in range(4):
    train_ds, test_ds = datasets[k]
    sub_datasets[k] = (Subset(train_ds, list(range(1000))), Subset(test_ds, list(range(100))))

# Pre-train base model on mixed dataset
print("\n--- Pre-training Base Model ---")
base_model = Deep12LayerCNN().to(device)
base_optimizer = optim.Adam(base_model.parameters(), lr=1e-3)

mixed_X = []
mixed_y = []
for k in range(4):
    train_sub, _ = sub_datasets[k]
    loader = DataLoader(train_sub, batch_size=1000, shuffle=False)
    imgs, labels = next(iter(loader))
    mixed_X.append(imgs)
    mixed_y.append(labels)
    
mixed_dataset = TensorDataset(torch.cat(mixed_X), torch.cat(mixed_y))
mixed_loader = DataLoader(mixed_dataset, batch_size=64, shuffle=True)

start_time = time.time()
base_model.train()
for epoch in range(5):
    total_loss = 0.0
    for imgs, labels in mixed_loader:
        base_optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(base_model(imgs), labels)
        loss.backward()
        base_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(mixed_loader):.4f}")
print(f"Base model pre-training took {time.time() - start_time:.1f}s.")

# Evaluate base model on tasks (using its shared head - wait, since they are different tasks, the shared head is evaluated)
print("\nEvaluating Base Model on Tasks:")
base_model.eval()
for k in range(4):
    _, test_sub = sub_datasets[k]
    loader = DataLoader(test_sub, batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = base_model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Base Model Task {k} Accuracy: {correct/total*100.0:.2f}%")

# Fine-tune expert models
expert_models = []
for k in range(4):
    print(f"\nFine-tuning Expert {k} on Task {k}...")
    expert = Deep12LayerCNN().to(device)
    expert.load_state_dict(base_model.state_dict())
    expert_optimizer = optim.Adam(expert.parameters(), lr=1e-3)
    
    train_sub, test_sub = sub_datasets[k]
    loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    
    expert.train()
    for epoch in range(2):
        for imgs, labels in loader:
            expert_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(expert(imgs), labels)
            loss.backward()
            expert_optimizer.step()
            
    expert_models.append(expert)
    
    # Evaluate expert
    expert.eval()
    test_loader = DataLoader(test_sub, batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = expert(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Expert {k} Task {k} Accuracy: {correct/total*100.0:.2f}%")

# Test Uniform ensembling
print("\n--- Evaluating Static Uniform Merging ---")
merged_model = Deep12LayerCNN().to(device)

def set_merged_weights_for_task(model, base_model, expert_models, alphas, task_idx):
    with torch.no_grad():
        for l in range(11):
            target_w = model.features[l].conv.weight
            target_b = model.features[l].conv.bias
            base_w = base_model.features[l].conv.weight
            base_b = base_model.features[l].conv.bias
            expert_ws = [m.features[l].conv.weight for m in expert_models]
            expert_bs = [m.features[l].conv.bias for m in expert_models]
            
            merged_w = base_w.clone()
            for k in range(4):
                merged_w += alphas[k][l] * (expert_ws[k] - base_w)
            target_w.copy_(merged_w)
            
            if target_b is not None:
                merged_b = base_b.clone()
                for k in range(4):
                    merged_b += alphas[k][l] * (expert_bs[k] - base_b)
                target_b.copy_(merged_b)
                
            # BatchNorm
            target_bn_w = model.features[l].bn.weight
            target_bn_b = model.features[l].bn.bias
            base_bn_w = base_model.features[l].bn.weight
            base_bn_b = base_model.features[l].bn.bias
            expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
            expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
            
            merged_bn_w = base_bn_w.clone()
            for k in range(4):
                merged_bn_w += alphas[k][l] * (expert_bn_ws[k] - base_bn_w)
            target_bn_w.copy_(merged_bn_w)
            
            if target_bn_b is not None:
                merged_bn_b = base_bn_b.clone()
                for k in range(4):
                    merged_bn_b += alphas[k][l] * (expert_bn_bs[k] - base_bn_b)
                target_bn_b.copy_(merged_bn_b)
                
        # Head
        model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
        model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)

alphas_uniform = torch.full((4, 12), 0.25)
for k in range(4):
    _, test_sub = sub_datasets[k]
    loader = DataLoader(test_sub, batch_size=32, shuffle=False)
    
    set_merged_weights_for_task(merged_model, base_model, expert_models, alphas_uniform, k)
    merged_model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = merged_model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Uniform Merged Task {k} Accuracy: {correct/total*100.0:.2f}%")
