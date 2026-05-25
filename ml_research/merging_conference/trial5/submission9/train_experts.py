import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import CNNEncoder, ClassifierHead

def get_dataloader(dataset_name, batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "mnist":
        ds = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "fmnist":
        ds = datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "kmnist":
        ds = datasets.KMNIST(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader

def train_expert(dataset_name, base_encoder_path, device):
    print(f"\n--- Training Expert for {dataset_name.upper()} ---")
    
    # Load base encoder weights
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(base_encoder_path, map_location=device))
    
    head = ClassifierHead().to(device)
    
    # Setup training
    train_loader = get_dataloader(dataset_name, batch_size=64, train=True)
    test_loader = get_dataloader(dataset_name, batch_size=64, train=False)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        encoder.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            features = encoder(images)
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        train_loss = total_loss / total
        
        # Eval
        encoder.eval()
        head.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                outputs = head(features)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}/5 - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")
        
    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), f"checkpoints/{dataset_name}_encoder.pth")
    torch.save(head.state_dict(), f"checkpoints/{dataset_name}_head.pth")
    print(f"Saved {dataset_name} expert to checkpoints/")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.enabled = False  # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED
        
    # Initialize and save pre-trained base model
    os.makedirs("checkpoints", exist_ok=True)
    base_encoder = CNNEncoder()
    # Explicit initialization
    for m in base_encoder.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
                
    base_encoder_path = "checkpoints/base_encoder.pth"
    torch.save(base_encoder.state_dict(), base_encoder_path)
    print(f"Initialized base encoder and saved to {base_encoder_path}")
    
    # Train experts
    for dataset in ["mnist", "fmnist", "kmnist"]:
        train_expert(dataset, base_encoder_path, device)

if __name__ == "__main__":
    main()
