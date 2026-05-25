import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("anchors", exist_ok=True)

# Transforms: Resize grayscale images to 32x32, convert to RGB, and normalize
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datasets_config = {
    "mnist": torchvision.datasets.MNIST,
    "fashion": torchvision.datasets.FashionMNIST,
    "kmnist": torchvision.datasets.KMNIST
}

def get_dataloader(name, train=True, batch_size=128):
    dataset_cls = datasets_config[name]
    dataset = dataset_cls(root="./data", train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader

# Custom ResNet wrapper that allows feature extraction
class ResNetExpert(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetExpert, self).__init__()
        # Load pretrained ResNet-18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # We will extract features from the layer before fc
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1) # Shape: [B, 512]
        out = self.fc(features)
        if return_features:
            return out, features
        return out

def train_expert(name, epochs=2):
    print(f"\n--- Training Expert for {name.upper()} ---")
    train_loader = get_dataloader(name, train=True)
    test_loader = get_dataloader(name, train=False)
    
    model = ResNetExpert(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc*100:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    print(f"Test Accuracy for {name.upper()}: {test_correct/test_total*100:.2f}%")
    
    # Save expert
    torch.save(model.state_dict(), f"models/expert_{name}.pth")
    print(f"Saved expert to models/expert_{name}.pth")
    
    # Compute anchors (class centroids) using a subset of train data
    print(f"Computing anchors for {name.upper()}...")
    anchors = torch.zeros(10, 512).to(device)
    class_counts = torch.zeros(10).to(device)
    
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, features = model(inputs, return_features=True) # Shape: [B, 512]
            for c in range(10):
                mask = (targets == c)
                if mask.sum() > 0:
                    anchors[c] += features[mask].sum(0)
                    class_counts[c] += mask.sum()
                    
    for c in range(10):
        if class_counts[c] > 0:
            anchors[c] /= class_counts[c]
            
    # Normalize anchors
    anchors_norm = nn.functional.normalize(anchors, p=2, dim=1)
    np.save(f"anchors/anchors_{name}.npy", anchors_norm.cpu().numpy())
    print(f"Saved normalized anchors to anchors/anchors_{name}.npy")

if __name__ == "__main__":
    for name in ["mnist", "fashion", "kmnist"]:
        train_expert(name, epochs=2)
