import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def get_base_model():
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify first conv to handle 1-channel grayscale 28x28 images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify final fc layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_expert(name, dataset_class, save_path, base_model_path, epochs=2, lr=1e-3, batch_size=256, device="cuda"):
    print(f"\n--- Training Expert: {name} ---")
    
    # Load base model checkpoint to ensure shared starting weights
    model = get_base_model()
    model.load_state_dict(torch.load(base_model_path, weights_only=True))
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = dataset_class(root="./data", train=True, download=True, transform=transform)
    test_dataset = dataset_class(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Expert {name} Test Accuracy: {test_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Saved {name} expert to {save_path}")
    return test_acc

def main():
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs("./models", exist_ok=True)
    
    # Save base model
    base_model_path = "./models/base_model.pt"
    if not os.path.exists(base_model_path):
        base_model = get_base_model()
        torch.save(base_model.state_dict(), base_model_path)
        print(f"Saved initial base model to {base_model_path}")
    
    # Train experts
    train_expert("MNIST", torchvision.datasets.MNIST, "./models/expert_mnist.pt", base_model_path, epochs=3, device=device)
    train_expert("FashionMNIST", torchvision.datasets.FashionMNIST, "./models/expert_fashion.pt", base_model_path, epochs=3, device=device)
    train_expert("KMNIST", torchvision.datasets.KMNIST, "./models/expert_kmnist.pt", base_model_path, epochs=3, device=device)

if __name__ == "__main__":
    main()
