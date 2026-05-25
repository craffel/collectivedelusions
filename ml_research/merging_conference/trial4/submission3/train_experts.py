import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to avoid initialization errors
torch.backends.cudnn.enabled = False

# Set device with fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define directories
SAVE_DIR = "/fsx/craffel/collectivedelusions/ml_research/merging_conference/trial4/submission3"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define standard transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
def get_datasets(task_name):
    if task_name == "mnist":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif task_name == "fashion":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    elif task_name == "kmnist":
        train_dataset = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    return train_dataset, test_dataset

class ExpertModel(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        # Load ImageNet pre-trained ResNet-18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace fully connected layer with Identity to get feature extractor
        self.backbone.fc = nn.Identity()
        # Dropout layer for Monte Carlo variational estimation
        self.dropout = nn.Dropout(p=dropout_prob)
        # Custom classification head (512 features -> 10 classes)
        self.head = nn.Linear(512, 10)
        
    def forward(self, x, force_dropout=False):
        features = self.backbone(x)
        if force_dropout:
            features = nn.functional.dropout(features, p=self.dropout.p, training=True)
        else:
            features = self.dropout(features)
        logits = self.head(features)
        return logits

def train_expert(task_name, epochs=3, batch_size=128, lr=1e-4):
    global device
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    train_dataset, test_dataset = get_datasets(task_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
    
    model = ExpertModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Simple training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDA" in str(e):
                    print(f"CUDA Error encountered: {e}. Falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.to(device)
                    images, labels = images.to(device), labels.to(device)
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    raise e
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    # Final Evaluation
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
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Final standalone test accuracy on {task_name.upper()}: {test_acc:.2f}%")
    
    # Save expert state dict
    save_path = os.path.join(SAVE_DIR, f"{task_name}_expert.pt")
    torch.save({
        'backbone_state_dict': model.backbone.state_dict(),
        'head_state_dict': model.head.state_dict(),
        'test_accuracy': test_acc
    }, save_path)
    print(f"Saved expert to {save_path}")

if __name__ == "__main__":
    try:
        train_expert("mnist", epochs=3, lr=1e-4)
        train_expert("fashion", epochs=3, lr=1e-4)
        train_expert("kmnist", epochs=5, lr=1e-4)
    except Exception as e:
        print(f"Critical error during training: {e}")
        # If any other issue, fall back and run on CPU
        device = torch.device("cpu")
        print("Switching entirely to CPU and retrying...")
        train_expert("mnist", epochs=3, lr=1e-4)
        train_expert("fashion", epochs=3, lr=1e-4)
        train_expert("kmnist", epochs=5, lr=1e-4)
