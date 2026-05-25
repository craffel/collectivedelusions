import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

class ChannelReductionResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load standard ResNet-18 pretrained weights
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Surgery on conv1 to reduce from 3 channels to 1 channel
        old_conv1 = self.resnet.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Sum the pretrained weights across the channel dimension
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.sum(dim=1, keepdim=True))
        self.resnet.conv1 = new_conv1
        
        # Replace classification head
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)

def train_model(dataset_name, save_path, epochs=2, batch_size=128, lr=0.01):
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {dataset_name} using {device} (cuDNN disabled)...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset_class = getattr(torchvision.datasets, dataset_name)
    train_dataset = dataset_class("./data", train=True, download=True, transform=transform)
    val_dataset = dataset_class("./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = ChannelReductionResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
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
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return val_acc

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model("MNIST", "models/expert_mnist.pt")
    train_model("FashionMNIST", "models/expert_fashion.pt")
