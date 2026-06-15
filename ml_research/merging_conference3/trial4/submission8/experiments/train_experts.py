import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import timm
from tqdm import tqdm

# Setup directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transforms
def get_transforms(is_grayscale=False):
    t_list = []
    if is_grayscale:
        t_list.append(transforms.Grayscale(num_output_channels=3))
    t_list.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(t_list)

# Datasets dict
datasets_config = {
    "mnist": (torchvision.datasets.MNIST, True),
    "fashionmnist": (torchvision.datasets.FashionMNIST, True),
    "cifar10": (torchvision.datasets.CIFAR10, False),
    "svhn": (torchvision.datasets.SVHN, False)
}

def load_data(name, subset_size=2000):
    dataset_cls, is_grayscale = datasets_config[name]
    transform = get_transforms(is_grayscale)
    
    if name == "svhn":
        train_set = dataset_cls(root="./data", split="train", download=True, transform=transform)
        test_set = dataset_cls(root="./data", split="test", download=True, transform=transform)
    else:
        train_set = dataset_cls(root="./data", train=True, download=True, transform=transform)
        test_set = dataset_cls(root="./data", train=False, download=True, transform=transform)
        
    # Take subsets to make training and evaluation extremely fast but still highly representative
    # 2000 train samples, 1000 test samples
    train_indices = list(range(min(subset_size, len(train_set))))
    test_indices = list(range(min(1000, len(test_set))))
    
    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)
    
    return train_subset, test_subset

def train_expert(dataset_name, epochs=5, batch_size=64, lr=5e-5, weight_decay=0.01):
    print(f"\n--- Training Expert for {dataset_name.upper()} ---")
    train_set, test_set = load_data(dataset_name)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Load ImageNet pre-trained vit_tiny model
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
            
        train_loss = total_loss / total
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
    test_acc = correct / total
    print(f"Test Accuracy on {dataset_name.upper()}: {test_acc*100:.2f}%")
    
    # Save the expert checkpoint (only the state_dict to keep it lightweight)
    checkpoint_path = f"checkpoints/vit_tiny_{dataset_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    return test_acc

if __name__ == "__main__":
    # Also save the pre-trained model state dict (which serves as the shared initialization / pretrained backbone)
    print("Saving pre-trained ImageNet baseline model...")
    pretrained_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
    torch.save(pretrained_model.state_dict(), "checkpoints/vit_tiny_pretrained.pt")
    print("Pre-trained model saved.")
    
    results = {}
    for dataset in ["mnist", "fashionmnist", "cifar10", "svhn"]:
        results[dataset] = train_expert(dataset)
        
    print("\nTraining complete! Results summary:")
    for d, acc in results.items():
        print(f"{d.upper()}: {acc*100:.2f}%")
