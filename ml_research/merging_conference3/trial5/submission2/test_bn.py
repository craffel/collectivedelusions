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

# Load datasets
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
    sub_datasets[k] = (Subset(train_ds, list(range(1000))), Subset(test_ds, list(range(500))))

print("\n--- Pre-training Base Model ---")
base_model = Deep12LayerCNN().to(device)
base_optimizer = optim.Adam(base_model.parameters(), lr=2e-3)

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

base_model.train()
for epoch in range(2):
    for imgs, labels in mixed_loader:
        base_optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(base_model(imgs), labels)
        loss.backward()
        base_optimizer.step()

expert_models = []
for k in range(4):
    print(f"Fine-tuning Expert {k} on Task {k}...")
    expert = Deep12LayerCNN().to(device)
    expert.load_state_dict(base_model.state_dict())
    expert_optimizer = optim.Adam(expert.parameters(), lr=1e-3)
    
    train_sub, _ = sub_datasets[k]
    loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    
    expert.train()
    for epoch in range(5):
        for imgs, labels in loader:
            expert_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(expert(imgs), labels)
            loss.backward()
            expert_optimizer.step()
    expert_models.append(expert)

# Helper to copy and merge weights statically layer-wise for a specific task
def set_merged_weights_for_task(model, base_model, expert_models, alphas, task_idx):
    alphas_clamped = torch.clamp(alphas, 0.0, 1.0)
    
    with torch.no_grad():
        # Layers 0 to 10: Merged Shared Backbone
        for l in range(11):
            # Target model parameters
            target_w = model.features[l].conv.weight
            target_b = model.features[l].conv.bias
            base_w = base_model.features[l].conv.weight
            base_b = base_model.features[l].conv.bias
            expert_ws = [m.features[l].conv.weight for m in expert_models]
            expert_bs = [m.features[l].conv.bias for m in expert_models]
            
            # Merged Conv Weight
            merged_w = base_w.clone()
            for k in range(4):
                merged_w += alphas_clamped[k][l] * (expert_ws[k] - base_w)
            target_w.copy_(merged_w)
            
            if target_b is not None:
                merged_b = base_b.clone()
                for k in range(4):
                    merged_b += alphas_clamped[k][l] * (expert_bs[k] - base_b)
                target_b.copy_(merged_b)
            
            # BatchNorm layers ensembling (weight & bias)
            target_bn_w = model.features[l].bn.weight
            target_bn_b = model.features[l].bn.bias
            base_bn_w = base_model.features[l].bn.weight
            base_bn_b = base_model.features[l].bn.bias
            expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
            expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
            
            merged_bn_w = base_bn_w.clone()
            for k in range(4):
                merged_bn_w += alphas_clamped[k][l] * (expert_bn_ws[k] - base_bn_w)
            target_bn_w.copy_(merged_bn_w)
            
            if target_bn_b is not None:
                merged_bn_b = base_bn_b.clone()
                for k in range(4):
                    merged_bn_b += alphas_clamped[k][l] * (expert_bn_bs[k] - base_bn_b)
                target_bn_b.copy_(merged_bn_b)
            
            # BatchNorm running mean and running var
            target_bn_rm = model.features[l].bn.running_mean
            target_bn_rv = model.features[l].bn.running_var
            base_bn_rm = base_model.features[l].bn.running_mean
            base_bn_rv = base_model.features[l].bn.running_var
            expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
            expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
            
            merged_bn_rm = base_bn_rm.clone()
            merged_bn_rv = base_bn_rv.clone()
            for k in range(4):
                merged_bn_rm += alphas_clamped[k][l] * (expert_bn_rms[k] - base_bn_rm)
                merged_bn_rv += alphas_clamped[k][l] * (expert_bn_rvs[k] - base_bn_rv)
            target_bn_rm.copy_(merged_bn_rm)
            target_bn_rv.copy_(merged_bn_rv)
            
        # Layer 11: Task-Specific Classifier Head (Preserved!)
        model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
        model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)

def evaluate_model(model, sub_datasets, base_model, expert_models, alphas, bn_mode, device):
    accuracies = []
    
    for task_idx in range(4):
        _, test_dataset = sub_datasets[task_idx]
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Merge shared backbone, and apply task-specific classification head
        set_merged_weights_for_task(model, base_model, expert_models, alphas, task_idx)
        
        model.eval()
        if bn_mode == 'train':
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accuracies.append(acc)
    return accuracies

merged_model = Deep12LayerCNN().to(device)
alphas_uniform = torch.full((4, 12), 0.25).to(device)

print("\n--- Evaluating Static Uniform Merging with BatchNorm in EVAL mode ---")
accs_eval = evaluate_model(merged_model, sub_datasets, base_model, expert_models, alphas_uniform, 'eval', device)
print(f"Uniform Merging Accs (EVAL): {accs_eval}, Average: {np.mean(accs_eval):.2f}%")

print("\n--- Evaluating Static Uniform Merging with BatchNorm in TRAIN mode ---")
accs_train = evaluate_model(merged_model, sub_datasets, base_model, expert_models, alphas_uniform, 'train', device)
print(f"Uniform Merging Accs (TRAIN): {accs_train}, Average: {np.mean(accs_train):.2f}%")
