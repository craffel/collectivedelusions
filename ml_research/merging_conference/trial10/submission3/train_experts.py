import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, s=30.0, m=0.35):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)
        
        self.use_cosface = use_cosface
        if self.use_cosface:
            self.classifier_weight = nn.Parameter(torch.randn(10, 128))
            nn.init.xavier_uniform_(self.classifier_weight)
            self.s = s
            self.m = m
        else:
            self.classifier = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        
        feat = self.dropout(F.relu(self.bn3(self.fc1(x))))
        
        if return_features:
            return feat
            
        if self.use_cosface:
            feat_norm = F.normalize(feat, p=2, dim=1)
            weight_norm = F.normalize(self.classifier_weight, p=2, dim=1)
            cos_theta = F.linear(feat_norm, weight_norm)
            
            if self.training:
                return cos_theta
            else:
                return self.s * cos_theta
        else:
            logits = self.classifier(feat)
            return logits

def train_one_expert(dataset_name, use_cosface=False, seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining expert for {dataset_name} (CosFace: {use_cosface}) on {device}...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
        
    # Subset of 10,000 samples for training
    train_subset = Subset(train_dataset, list(range(10000)))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model = SimpleCNN(use_cosface=use_cosface).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if use_cosface:
                cos_theta = model(inputs)
                # Apply additive margin to target class logits
                # cos_theta has shape [B, 10]
                target_logit = cos_theta[range(inputs.size(0)), targets]
                target_logit_margin = target_logit - model.m
                
                # Reconstruct logits
                logits = cos_theta.clone()
                logits[range(inputs.size(0)), targets] = target_logit_margin
                logits = logits * model.s
            else:
                logits = model(inputs)
                
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
        print(f"Epoch {epoch+1}/2 | Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}%")
        
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
    test_acc = correct / total * 100
    print(f"Test Accuracy on {dataset_name}: {test_acc:.2f}%")
    
    os.makedirs("checkpoints", exist_ok=True)
    suffix = "cosface" if use_cosface else "standard"
    name = f"expert_{0 if dataset_name == 'MNIST' else 1}_{suffix}.pt"
    save_path = os.path.join("checkpoints", name)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")
    return test_acc

if __name__ == "__main__":
    train_one_expert("MNIST", use_cosface=False)
    train_one_expert("FashionMNIST", use_cosface=False)
    train_one_expert("MNIST", use_cosface=True)
    train_one_expert("FashionMNIST", use_cosface=True)
