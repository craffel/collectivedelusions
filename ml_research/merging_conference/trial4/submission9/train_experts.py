import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define the shared CNN base encoder
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.pool2(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc(x))
        return x

# Define the task-specific classification head
class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

# Complete model combining encoder and head
class ExpertModel(nn.Module):
    def __init__(self, encoder, head):
        super(ExpertModel, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

def train_expert(dataset_name, save_path):
    print(f"\n--- Training Expert for {dataset_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
    elif dataset_name == "KMNIST":
        train_dataset = datasets.KMNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.KMNIST(root="./data", train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # Initialize models
    # We want a shared base encoder initialization. To do this, we'll set a manual seed
    # when initializing the encoder so that all experts start from the SAME base weights.
    torch.manual_seed(42)
    encoder = CNNEncoder()
    head = ClassificationHead()
    
    model = ExpertModel(encoder, head).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train for 5 epochs
    for epoch in range(5):
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
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc*100:.2f}%")

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_acc = test_correct / test_total
    print(f"Finished training {dataset_name}. Test Acc: {test_acc*100:.2f}%")

    # Save checkpoint
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'head_state_dict': head.state_dict(),
        'test_acc': test_acc
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_expert("MNIST", "checkpoints/mnist_expert.pt")
    train_expert("FashionMNIST", "checkpoints/fashion_expert.pt")
    train_expert("KMNIST", "checkpoints/kmnist_expert.pt")
