import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

# Model architecture definitions
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(x)

def train_one_expert(task_name, dataset_class, base_encoder, device, epochs=5, lr=0.01, batch_size=64):
    print(f"\n--- Training Expert for Task: {task_name} ---")
    
    # Initialize encoder and head
    # We copy the base encoder's state to start from a shared initialization
    encoder = Encoder().to(device)
    encoder.load_state_dict(base_encoder.state_dict())
    head = Head().to(device)
    
    # Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        encoder.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = encoder(x)
            outputs = head(features)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
        # Test eval
        encoder.eval()
        head.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = head(encoder(x))
                _, predicted = outputs.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
        test_acc = 100.0 * test_correct / test_total
        print(f"Test Acc: {test_acc:.2f}%")
        
    # Save the trained expert states
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), f"checkpoints/expert_{task_name}_encoder.pt")
    torch.save(head.state_dict(), f"checkpoints/expert_{task_name}_head.pt")
    print(f"Saved expert_{task_name} weights.")

if __name__ == "__main__":
    import torch.backends.cudnn as cudnn
    cudnn.enabled = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define and save base encoder
    base_encoder = Encoder().to(device)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(base_encoder.state_dict(), "checkpoints/base_encoder.pt")
    print("Saved base_encoder.pt.")
    
    # Train experts for MNIST, FashionMNIST, KMNIST
    train_one_expert("mnist", datasets.MNIST, base_encoder, device, epochs=5, lr=0.001)
    train_one_expert("fashion", datasets.FashionMNIST, base_encoder, device, epochs=5, lr=0.001)
    train_one_expert("kmnist", datasets.KMNIST, base_encoder, device, epochs=5, lr=0.001)
    
    print("\nAll experts successfully trained and saved!")
