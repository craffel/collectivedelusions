import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.enabled = False
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

class ResNetBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_dataset(name, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if name == 'mnist':
        return torchvision.datasets.MNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'fashion':
        return torchvision.datasets.FashionMNIST(root='./data', train=train, download=False, transform=transform)
    elif name == 'kmnist':
        return torchvision.datasets.KMNIST(root='./data', train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")

def train_expert(task_idx, dataset_name, epochs, batch_size, device):
    print(f"\n--- Training Expert for Task {task_idx}: {dataset_name.upper()} ---")
    
    # Load dataset
    train_set = get_dataset(dataset_name, train=True)
    test_set = get_dataset(dataset_name, train=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize model with ImageNet pre-trained weights
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = ResNetBackbone(base_model).to(device)
    head = nn.Linear(512, 10).to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), 
        lr=1e-4, 
        betas=(0.9, 0.999)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        backbone.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            features = backbone(inputs)
            outputs = head(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%")
                
        # Evaluate after epoch
        backbone.eval()
        head.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                features = backbone(inputs)
                outputs = head(features)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        print(f"Epoch {epoch+1} Test Accuracy: {100.0 * test_correct / test_total:.2f}%")
        
    # Save the standalone models
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'head_state_dict': head.state_dict(),
        'test_accuracy': 100.0 * test_correct / test_total
    }, f'checkpoints/expert_{task_idx}.pt')
    print(f"Saved expert_{task_idx}.pt")
    
    # Compute clean diagonal FIM prior for classification head
    print(f"Computing diagonal FIM prior for classification head of Task {task_idx}...")
    backbone.eval()
    head.eval()
    
    # Select N = 200 samples for FIM precomputation
    fim_subset = Subset(train_set, list(range(200)))
    fim_loader = DataLoader(fim_subset, batch_size=1, shuffle=False)
    
    # Initialize Fisher parameters same shape as head parameters with zeros
    fisher_dict = {}
    for name, param in head.named_parameters():
        fisher_dict[name] = torch.zeros_like(param)
        
    for inputs, targets in fim_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients of head
        head.zero_grad()
        with torch.no_grad():
            features = backbone(inputs)
        outputs = head(features)
        
        # Log-likelihood loss for single sample
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in head.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += (param.grad.data ** 2) / 200.0
                
    # Save Fisher Information Matrix
    torch.save(fisher_dict, f'checkpoints/fim_{task_idx}.pt')
    print(f"Saved checkpoints/fim_{task_idx}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.main_args = parser.parse_known_args()[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    datasets = ['mnist', 'fashion', 'kmnist']
    for idx, name in enumerate(datasets):
        train_expert(idx, name, epochs=3, batch_size=128, device=device)
    print("\nTraining of all experts complete!")
