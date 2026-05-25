import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

# Define CNN Base Encoder
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Block 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Latent projection
        self.fc = nn.Linear(3136, 128)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        # Block 2
        x = F.relu(self.conv2(x))
        # Block 3
        x = self.pool2(F.relu(self.conv3(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # Latent projection
        x = F.relu(self.fc(x))
        return x

# Full model with task-specific head
class TaskExpert(nn.Module):
    def __init__(self, encoder):
        super(TaskExpert, self).__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features)
        return out

def get_dataloader(task_name, train=True, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    os.makedirs("./data", exist_ok=True)
    if task_name == "MNIST":
        dataset = datasets.MNIST("./data", train=train, download=True, transform=transform)
    elif task_name == "FashionMNIST":
        dataset = datasets.FashionMNIST("./data", train=train, download=True, transform=transform)
    elif task_name == "KMNIST":
        dataset = datasets.KMNIST("./data", train=train, download=True, transform=transform)
    else:
        raise ValueError("Unknown task")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=not train)

def compute_fim(model, dataloader, device, num_samples=500):
    model.eval()
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)
            
    count = 0
    with torch.enable_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Process sample by sample to get individual gradients
            for i in range(x.size(0)):
                xi = x[i:i+1]
                yi = y[i:i+1]
                out = model(xi)
                loss = F.cross_entropy(out, yi)
                model.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fim[name] += param.grad.data ** 2
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
            
    with torch.no_grad():
        for name in fim:
            fim[name] /= count
            
    return fim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Initialize the shared base encoder
    shared_encoder = CNNEncoder()
    # Save the initial encoder state (base initialization)
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(shared_encoder.state_dict(), "./checkpoints/shared_encoder_init.pt")
    
    tasks = ["MNIST", "FashionMNIST", "KMNIST"]
    experts = {}
    
    for task in tasks:
        print(f"\n--- Training Expert for {task} ---")
        train_loader = get_dataloader(task, train=True, batch_size=64)
        test_loader = get_dataloader(task, train=False, batch_size=64)
        
        # Instantiate expert with its own copy of the shared encoder
        encoder = CNNEncoder()
        encoder.load_state_dict(shared_encoder.state_dict())
        model = TaskExpert(encoder).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Train for 5 epochs
        for epoch in range(5):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        test_loss /= total
        test_acc = 100. * correct / total
        print(f"Test Evaluation for {task} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"./checkpoints/expert_{task}.pt")
        
        # Step 2: Compute Fisher Information Matrix (FIM)
        print(f"Computing Fisher Information Matrix for {task}...")
        fim_loader = get_dataloader(task, train=True, batch_size=1)
        fim = compute_fim(model, fim_loader, device, num_samples=500)
        torch.save(fim, f"./checkpoints/fim_{task}.pt")
        print(f"FIM saved for {task}.")

if __name__ == "__main__":
    main()
