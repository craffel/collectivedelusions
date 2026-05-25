import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

def get_modified_resnet18():
    # Load pretrained resnet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Modify first conv layer to accept 1 channel (grayscale) instead of 3 channels
    conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=(conv1.bias is not None)
    )
    # Sum pre-trained weights along input channel dimension (dim 1)
    with torch.no_grad():
        new_conv1.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
        if conv1.bias is not None:
            new_conv1.bias.copy_(conv1.bias)
    model.conv1 = new_conv1
    
    # Modify classifier head to output 10 logits
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

def get_dataloader(dataset_name, batch_size=256, train=True):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "mnist":
        dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "kmnist":
        dataset = datasets.KMNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "fashionmnist":
        dataset = datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2 if train else 1)
    return loader

def train_expert(dataset_name, epochs=3):
    print(f"\n--- Training Expert on {dataset_name.upper()} ---")
    model = get_modified_resnet18()
    train_loader = get_dataloader(dataset_name, batch_size=256, train=True)
    test_loader = get_dataloader(dataset_name, batch_size=256, train=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy on {dataset_name.upper()}: {test_acc:.2f}%")
    
    # Save model state dict
    save_path = f"expert_{dataset_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert to {save_path}")
    return test_acc

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    accs = {}
    for ds in ["mnist", "kmnist", "fashionmnist"]:
        accs[ds] = train_expert(ds, epochs=3)
    print("\n--- Summary of Expert stand-alone accuracies ---")
    for ds, acc in accs.items():
        print(f"{ds.upper()}: {acc:.2f}%")
