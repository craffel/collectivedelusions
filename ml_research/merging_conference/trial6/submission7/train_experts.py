import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import get_resnet18_model

def train_one_expert(name, dataset_cls, save_path, device):
    print(f"\n--- Training Expert: {name} ---")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download and load train dataset
    full_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
    # Use first 10,000 samples
    train_subset = Subset(full_dataset, range(min(10000, len(full_dataset))))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    
    # Instantiate model fine-tuned from ImageNet pretrained model
    model = get_resnet18_model(num_classes=10, pretrained=True)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(4):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/4 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert model checkpoint to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN to avoid initialization errors.")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # First, save base model (un-fine-tuned but with classification head)
    # This represents the shared starting point of the expert models.
    base_model_path = "checkpoints/base_model.pt"
    if not os.path.exists(base_model_path):
        print("Saving base model checkpoint...")
        base_model = get_resnet18_model(num_classes=10, pretrained=True)
        torch.save(base_model.state_dict(), base_model_path)
        print(f"Saved base model checkpoint to {base_model_path}")
    
    # Train experts
    experts = [
        ("MNIST", datasets.MNIST, "checkpoints/mnist_expert.pt"),
        ("FashionMNIST", datasets.FashionMNIST, "checkpoints/fashion_expert.pt"),
        ("KMNIST", datasets.KMNIST, "checkpoints/kmnist_expert.pt")
    ]
    
    for name, dataset_cls, save_path in experts:
        if not os.path.exists(save_path):
            train_one_expert(name, dataset_cls, save_path, device)
        else:
            print(f"Expert {name} checkpoint already exists at {save_path}, skipping training.")

if __name__ == "__main__":
    main()
