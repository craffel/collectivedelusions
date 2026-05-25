import os
import torch
# Disable cuDNN to bypass recurrent initialization errors on the cluster environment
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Define Expert Model
class ExpertModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ExpertModel, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        # Modify final layer to output 10 classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        if return_features:
            return self.get_features(x)
        return self.resnet(x)

    def get_features(self, x):
        # Extract features right before the fc layer (512-dim)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def train_expert(dataset_name, save_path, epochs=2, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training expert for {dataset_name} on {device} ---")

    # Set up transforms
    if dataset_name == "FashionMNIST":
        # Convert grayscale (1 channel) to RGB (3 channels) by repeating
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load dataset
    if dataset_name == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "SVHN":
        train_set = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Use a subset of 15,000 train samples and 2,000 test samples for fast, high-quality pretraining
    train_subset = Subset(train_set, list(range(min(len(train_set), 15000))))
    test_subset = Subset(test_set, list(range(min(len(test_set), 2000))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ExpertModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")

    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"Validation Accuracy: {test_acc:.2f}%")

    # Save weights
    torch.save(model.state_dict(), save_path)
    print(f"Expert model saved successfully to {save_path}")

if __name__ == "__main__":
    train_expert("CIFAR10", "expert_cifar10.pth", epochs=2)
    train_expert("SVHN", "expert_svhn.pth", epochs=2)
    train_expert("FashionMNIST", "expert_fmnist.pth", epochs=2)
