import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Workaround for cuDNN initialization errors on some GPU nodes
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("cuDNN disabled as a workaround for cluster compatibility.")

# Define directories
os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Define dataset transforms (Grayscale tasks converted to 3-channel, 32x32)
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_datasets():
    print("Loading datasets...")
    mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform_gray)
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform_gray)
    
    fmnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_gray)
    fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_gray)
    
    cifar_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_color)
    cifar_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_color)
    
    return {
        "mnist": (mnist_train, mnist_test),
        "fmnist": (fmnist_train, fmnist_test),
        "cifar10": (cifar_train, cifar_test)
    }

class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() # Replace FC with Identity to get 512 features
        
        # Heads will be separate linear layers
        self.heads = nn.ModuleDict({
            "mnist": nn.Linear(512, 10),
            "fmnist": nn.Linear(512, 10),
            "cifar10": nn.Linear(512, 10)
        })
        
    def forward(self, x, task_name):
        feats = self.backbone(x)
        return self.heads[task_name](feats)

def train_expert(task_name, train_dataset, test_dataset, epochs=5, batch_size=256):
    print(f"\n--- Training Expert for Task: {task_name.upper()} ---")
    
    # Initialize fresh model from ImageNet weights
    model = MultiTaskResNet18().to(device)
    
    # Checkpoint progenitor if not already saved
    progenitor_path = "checkpoints/progenitor.pt"
    if not os.path.exists(progenitor_path):
        print("Saving progenitor baseline...")
        torch.save(model.state_dict(), progenitor_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimize only the backbone and the task head
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, task_name)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total:.4f} - Train Acc: {train_acc*100:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_name)
            preds = logits.argmax(dim=1)
            test_correct += (preds == y).sum().item()
            test_total += x.size(0)
            
    test_acc = test_correct / test_total
    print(f"Finished {task_name.upper()} - Test Accuracy: {test_acc*100:.2f}%")
    
    # Save expert checkpoint
    checkpoint_path = f"checkpoints/{task_name}_expert.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return test_acc

if __name__ == "__main__":
    task_data = get_datasets()
    accuracies = {}
    for task_name, (train_ds, test_ds) in task_data.items():
        accuracies[task_name] = train_expert(task_name, train_ds, test_ds)
        
    print("\n--- Summary of Expert Performances ---")
    for k, v in accuracies.items():
        print(f"Task {k.upper()}: {v*100:.2f}%")
