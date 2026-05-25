import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Define the custom CNN model as described in the papers
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Conv Layer 1: Conv2d(1, 32, 3, padding=1), ReLU, MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: Conv2d(32, 64, 3, padding=1), ReLU (no pool)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Conv Layer 3: Conv2d(64, 64, 3, padding=1), ReLU, MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Linear projection to 128
        self.fc = nn.Linear(3136, 128)
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool2(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc(x))
        return x

class ExpertModel(nn.Module):
    def __init__(self, base_encoder):
        super(ExpertModel, self).__init__()
        self.base_encoder = base_encoder
        self.head = nn.Linear(128, 10)
        
    def forward(self, x):
        features = self.base_encoder(x)
        out = self.head(features)
        return out

def train_expert(dataset_name, train_dataset, base_encoder, device, epochs=5, lr=0.001, batch_size=64, wd=1e-4):
    print(f"\n--- Training Expert for {dataset_name} ---")
    
    # Instantiate the expert model sharing the base encoder
    model = ExpertModel(base_encoder).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize both encoder and head
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    return model

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Seeding
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load Datasets
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    fashion_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST('./data', train=True, download=True, transform=transform)
    
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST('./data', train=False, download=True, transform=transform)
    
    # Save test datasets for evaluation stream
    os.makedirs("./data/processed", exist_ok=True)
    torch.save(mnist_test, "./data/processed/mnist_test.pt")
    torch.save(fashion_test, "./data/processed/fashion_test.pt")
    torch.save(kmnist_test, "./data/processed/kmnist_test.pt")
    
    # Sibling training loaders to train independent experts starting from a shared base encoder initialization
    # We first initialize a single shared base_encoder
    shared_base_encoder = CustomCNN().to(device)
    
    # We save the initial state of the shared base encoder
    os.makedirs("./experts", exist_ok=True)
    torch.save(shared_base_encoder.state_dict(), "./experts/base_encoder_init.pt")
    
    # For each expert, we start with a copy of the shared_base_encoder state
    # This matches: "We train 3 independent expert models starting from a shared base encoder initialization."
    
    # 1. MNIST Expert
    base_mnist = CustomCNN().to(device)
    base_mnist.load_state_dict(torch.load("./experts/base_encoder_init.pt"))
    model_mnist = train_expert("MNIST", mnist_train, base_mnist, device)
    torch.save(model_mnist.state_dict(), "./experts/expert_mnist.pt")
    
    # 2. FashionMNIST Expert
    base_fashion = CustomCNN().to(device)
    base_fashion.load_state_dict(torch.load("./experts/base_encoder_init.pt"))
    model_fashion = train_expert("FashionMNIST", fashion_train, base_fashion, device)
    torch.save(model_fashion.state_dict(), "./experts/expert_fashion.pt")
    
    # 3. KMNIST Expert
    base_kmnist = CustomCNN().to(device)
    base_kmnist.load_state_dict(torch.load("./experts/base_encoder_init.pt"))
    model_kmnist = train_expert("KMNIST", kmnist_train, base_kmnist, device)
    torch.save(model_kmnist.state_dict(), "./experts/expert_kmnist.pt")
    
    print("\nAll 3 experts trained and saved successfully.")

if __name__ == "__main__":
    main()
