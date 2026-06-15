import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc(x))
        return x

class ClassifierHead(nn.Module):
    def __init__(self):
        super(ClassifierHead, self).__init__()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.fc2(x)

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, head):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

subset_train_size = 5000
subset_val_size = 1000
subset_test_size = 1000

def get_subset_loader(dataset, size, batch_size, shuffle=True):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:size]
    subset = Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

def get_val_test_loaders_disjoint(dataset, val_size, test_size, batch_size):
    indices = list(range(len(dataset)))
    rng = random.Random(42)
    rng.shuffle(indices)
    val_indices = indices[:val_size]
    test_indices = indices[val_size : val_size + test_size]
    return (DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False),
            DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False))

mnist_train_loader = get_subset_loader(mnist_train, subset_train_size, batch_size=64)
fashion_train_loader = get_subset_loader(fashion_train, subset_train_size, batch_size=64)
kmnist_train_loader = get_subset_loader(kmnist_train, subset_train_size, batch_size=64)

mnist_val_loader, mnist_test_loader = get_val_test_loaders_disjoint(mnist_test, subset_val_size, subset_test_size, 128)
fashion_val_loader, fashion_test_loader = get_val_test_loaders_disjoint(fashion_test, subset_val_size, subset_test_size, 128)
kmnist_val_loader, kmnist_test_loader = get_val_test_loaders_disjoint(kmnist_test, subset_val_size, subset_test_size, 128)

def train_model_custom(model, loader, epochs, lr):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Train experts
pretrained_backbone = SimpleCNN().to(device)
combined_train = torch.utils.data.ConcatDataset([
    Subset(mnist_train, list(range(1000))),
    Subset(fashion_train, list(range(1000))),
    Subset(kmnist_train, list(range(1000)))
])
combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True)
mixed_head = ClassifierHead().to(device)
mixed_model = MultiTaskModel(pretrained_backbone, mixed_head)
train_model_custom(mixed_model, combined_loader, epochs=1, lr=0.005)
pretrained_backbone_state = copy.deepcopy(pretrained_backbone.state_dict())

mnist_backbone = SimpleCNN().to(device)
mnist_backbone.load_state_dict(copy.deepcopy(pretrained_backbone_state))
mnist_head = ClassifierHead().to(device)
train_model_custom(MultiTaskModel(mnist_backbone, mnist_head), mnist_train_loader, epochs=3, lr=0.001)
mnist_backbone_state = copy.deepcopy(mnist_backbone.state_dict())

fashion_backbone = SimpleCNN().to(device)
fashion_backbone.load_state_dict(copy.deepcopy(pretrained_backbone_state))
fashion_head = ClassifierHead().to(device)
train_model_custom(MultiTaskModel(fashion_backbone, fashion_head), fashion_train_loader, epochs=2, lr=0.003)
fashion_backbone_state = copy.deepcopy(fashion_backbone.state_dict())

kmnist_backbone = SimpleCNN().to(device)
kmnist_backbone.load_state_dict(copy.deepcopy(pretrained_backbone_state))
kmnist_head = ClassifierHead().to(device)
train_model_custom(MultiTaskModel(kmnist_backbone, kmnist_head), kmnist_train_loader, epochs=1, lr=0.002)
kmnist_backbone_state = copy.deepcopy(kmnist_backbone.state_dict())

def evaluate_model(backbone, head, loader):
    backbone.eval()
    head.eval()
    backbone.to(device)
    head.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            features = backbone(data)
            output = head(features)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total

def evaluate_merged_val(backbone_state):
    backbone = SimpleCNN().to(device)
    backbone.load_state_dict(backbone_state)
    acc_m = evaluate_model(backbone, mnist_head, mnist_val_loader)
    acc_f = evaluate_model(backbone, fashion_head, fashion_val_loader)
    acc_k = evaluate_model(backbone, kmnist_head, kmnist_val_loader)
    return (acc_m + acc_f + acc_k) / 3

def evaluate_merged_test(backbone_state):
    backbone = SimpleCNN().to(device)
    backbone.load_state_dict(backbone_state)
    acc_m = evaluate_model(backbone, mnist_head, mnist_test_loader)
    acc_f = evaluate_model(backbone, fashion_head, fashion_test_loader)
    acc_k = evaluate_model(backbone, kmnist_head, kmnist_test_loader)
    return acc_m, acc_f, acc_k, (acc_m + acc_f + acc_k) / 3

task_states = [mnist_backbone_state, fashion_backbone_state, kmnist_backbone_state]
task_vectors = []
for state in task_states:
    vec = {}
    for key in pretrained_backbone_state:
        vec[key] = state[key] - pretrained_backbone_state[key]
    task_vectors.append(vec)

# Use the exact same fine-grained lambdas grid [0.3, 1.5] with step 0.05
lambdas_grid = np.arange(0.3, 1.51, 0.05)

print("--- Variant 1: RMS-Norm Only (no scale calibration) ---")
best_norm_val = 0.0
best_norm_lam = 0.0
for lam in lambdas_grid:
    rms_vec = {}
    for key in pretrained_backbone_state:
        merged_tensor_list = []
        for vec in task_vectors:
            v = vec[key]
            rms = torch.sqrt(torch.mean(v ** 2) + 1e-8).item()
            merged_tensor_list.append(v / rms)
        avg_normalized_direction = sum(merged_tensor_list) / len(merged_tensor_list)
        rms_vec[key] = lam * avg_normalized_direction
    
    state = {}
    for key in pretrained_backbone_state:
        state[key] = pretrained_backbone_state[key] + rms_vec[key]
    val_acc = evaluate_merged_val(state)
    if val_acc > best_norm_val:
        best_norm_val = val_acc
        best_norm_lam = lam

opt_state = {}
for key in pretrained_backbone_state:
    merged_tensor_list = []
    for vec in task_vectors:
        v = vec[key]
        rms = torch.sqrt(torch.mean(v ** 2) + 1e-8).item()
        merged_tensor_list.append(v / rms)
    avg_normalized_direction = sum(merged_tensor_list) / len(merged_tensor_list)
    opt_state[key] = pretrained_backbone_state[key] + best_norm_lam * avg_normalized_direction
test_res = evaluate_merged_test(opt_state)
print(f"Norm Only -> Best lam: {best_norm_lam:.2f} | Val Avg: {best_norm_val*100:.2f}% | Test MNIST: {test_res[0]*100:.2f}%, Fashion: {test_res[1]*100:.2f}%, KMNIST: {test_res[2]*100:.2f}%, Test Avg: {test_res[3]*100:.2f}%")


print("\n--- Variant 2: RMS-Calib Only (no individual task normalization) ---")
best_calib_val = 0.0
best_calib_lam = 0.0
for lam in lambdas_grid:
    rms_vec = {}
    for key in pretrained_backbone_state:
        v_list = [vec[key] for vec in task_vectors]
        rmss = []
        for v in v_list:
            rms = torch.sqrt(torch.mean(v ** 2) + 1e-8).item()
            rmss.append(rms)
        mean_rms = sum(rmss) / len(rmss)
        avg_vector = sum(v_list) / len(v_list)
        rms_vec[key] = lam * mean_rms * avg_vector
        
    state = {}
    for key in pretrained_backbone_state:
        state[key] = pretrained_backbone_state[key] + rms_vec[key]
    val_acc = evaluate_merged_val(state)
    if val_acc > best_calib_val:
        best_calib_val = val_acc
        best_calib_lam = lam

opt_state = {}
for key in pretrained_backbone_state:
    v_list = [vec[key] for vec in task_vectors]
    rmss = []
    for v in v_list:
        rms = torch.sqrt(torch.mean(v ** 2) + 1e-8).item()
        rmss.append(rms)
    mean_rms = sum(rmss) / len(rmss)
    avg_vector = sum(v_list) / len(v_list)
    opt_state[key] = pretrained_backbone_state[key] + best_calib_lam * mean_rms * avg_vector
test_res = evaluate_merged_test(opt_state)
print(f"Calib Only -> Best lam: {best_calib_lam:.2f} | Val Avg: {best_calib_val*100:.2f}% | Test MNIST: {test_res[0]*100:.2f}%, Fashion: {test_res[1]*100:.2f}%, KMNIST: {test_res[2]*100:.2f}%, Test Avg: {test_res[3]*100:.2f}%")
