import os
import torch
import torch.nn as nn

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on cluster nodes
torch.backends.cudnn.enabled = False
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

def get_dataset(name, train=True, transform=None):
    if name == 'mnist':
        return datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif name == 'fashion':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == 'kmnist':
        return datasets.KMNIST(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def train_expert(dataset_name, save_path):
    print(f"=== Training Expert for {dataset_name.upper()} ===")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load data
    train_dataset = get_dataset(dataset_name, train=True, transform=transform)
    test_dataset = get_dataset(dataset_name, train=False, transform=transform)
    
    # Subset train to 10,000 samples
    torch.manual_seed(42)
    train_indices = torch.randperm(len(train_dataset))[:10000]
    train_subset = Subset(train_dataset, train_indices)
    
    # Subset test to 5,000 samples
    test_indices = torch.randperm(len(test_dataset))[:5000]
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=4)
    
    # Load pre-trained ResNet-18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Modify conv1 to accept 1 channel (grayscale)
    pretrained_conv1 = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        # Copy mean of channels to initialize 1-channel conv1
        model.conv1.weight.copy_(pretrained_conv1.mean(dim=1, keepdim=True))
        
    # Modify fc layer to output 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(4):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/4"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_subset)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Test Accuracy on {dataset_name.upper()}: {accuracy:.2f}%")
    
    # Save expert state dict
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}\n")

if __name__ == '__main__':
    os.makedirs('experts', exist_ok=True)
    train_expert('mnist', 'experts/mnist_expert.pt')
    train_expert('fashion', 'experts/fashion_expert.pt')
    train_expert('kmnist', 'experts/kmnist_expert.pt')
