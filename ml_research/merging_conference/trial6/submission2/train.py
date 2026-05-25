import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import os
import copy
from model import SimpleCNN

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster nodes
torch.backends.cudnn.enabled = False

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
BASE_EPOCHS = 4
FT_EPOCHS = 6
CALIBRATION_SIZE = 1000

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    return {
        'mnist': (mnist_train, mnist_test),
        'fmnist': (fmnist_train, fmnist_test),
        'kmnist': (kmnist_train, kmnist_test)
    }

def train_model(model, train_loader, epochs, lr=0.01, task_name=""):
    print(f"Training model on {task_name} for {epochs} epochs...")
    model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    return model

def evaluate(model, test_loader, task_name=""):
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out, _ = model(x)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    acc = 100. * correct / total
    print(f"Accuracy of model on {task_name}: {acc:.2f}%")
    return acc

def compute_diagonal_fisher(model, dataset, num_samples=500):
    print("Computing diagonal Fisher Information...")
    model.to(DEVICE)
    model.eval()
    
    # Initialize Fisher dictionary
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            
    # Sample a calibration set
    indices = torch.randperm(len(dataset))[:num_samples]
    sub_dataset = Subset(dataset, indices)
    loader = DataLoader(sub_dataset, batch_size=1, shuffle=False)
    
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        out, _ = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.data ** 2) / num_samples
                
    return fisher

def extract_class_prototypes(model, dataset, num_samples=1000):
    print("Extracting class prototypes...")
    model.to(DEVICE)
    model.eval()
    
    indices = torch.randperm(len(dataset))[:num_samples]
    sub_dataset = Subset(dataset, indices)
    loader = DataLoader(sub_dataset, batch_size=64, shuffle=False)
    
    features_by_class = {c: [] for c in range(10)}
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            _, feat = model(x) # feat shape: (B, 128)
            for i in range(x.size(0)):
                label = y[i].item()
                features_by_class[label].append(feat[i].cpu())
                
    prototypes = torch.zeros(10, 128)
    for c in range(10):
        if len(features_by_class[c]) > 0:
            stacked = torch.stack(features_by_class[c])
            mean_feat = stacked.mean(dim=0)
            # Normalize to unit sphere
            prototypes[c] = mean_feat / (mean_feat.norm(p=2) + 1e-8)
        else:
            # Fallback if class is missing in random sample
            prototypes[c] = torch.zeros(128)
            
    return prototypes

def main():
    os.makedirs("checkpoints", exist_ok=True)
    datasets = get_datasets()
    
    mnist_train, mnist_test = datasets['mnist']
    fmnist_train, fmnist_test = datasets['fmnist']
    kmnist_train, kmnist_test = datasets['kmnist']
    
    # Create combined dataset for base pre-training
    # Take 10,000 samples from each training set to keep it balanced and fast
    size_per_dataset = 10000
    mnist_sub = Subset(mnist_train, torch.randperm(len(mnist_train))[:size_per_dataset])
    fmnist_sub = Subset(fmnist_train, torch.randperm(len(fmnist_train))[:size_per_dataset])
    kmnist_sub = Subset(kmnist_train, torch.randperm(len(kmnist_train))[:size_per_dataset])
    base_dataset = ConcatDataset([mnist_sub, fmnist_sub, kmnist_sub])
    base_loader = DataLoader(base_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train Base Model
    base_model = SimpleCNN()
    base_model = train_model(base_model, base_loader, BASE_EPOCHS, lr=0.001, task_name="Multi-Task Base")
    torch.save(base_model.state_dict(), "checkpoints/base.pt")
    
    # Verify Base Model performance
    test_loaders = {
        'mnist': DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False),
        'fmnist': DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False),
        'kmnist': DataLoader(kmnist_test, batch_size=BATCH_SIZE, shuffle=False)
    }
    
    print("\n--- Base Model Evaluation ---")
    evaluate(base_model, test_loaders['mnist'], "MNIST")
    evaluate(base_model, test_loaders['fmnist'], "FashionMNIST")
    evaluate(base_model, test_loaders['kmnist'], "KMNIST")
    print("------------------------------\n")
    
    # Train Experts by fine-tuning from Base
    experts = {}
    loaders = {
        'mnist': DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True),
        'fmnist': DataLoader(fmnist_train, batch_size=BATCH_SIZE, shuffle=True),
        'kmnist': DataLoader(kmnist_train, batch_size=BATCH_SIZE, shuffle=True)
    }
    
    for name in ['mnist', 'fmnist', 'kmnist']:
        print(f"\nTraining Expert for {name}...")
        expert = SimpleCNN()
        expert.load_state_dict(torch.load("checkpoints/base.pt"))
        expert = train_model(expert, loaders[name], FT_EPOCHS, lr=0.0005, task_name=name)
        torch.save(expert.state_dict(), f"checkpoints/expert_{name}.pt")
        evaluate(expert, test_loaders[name], f"Expert {name}")
        experts[name] = expert
        
    # Compute and save Fisher information and Prototypes for each expert
    fisher_infos = {}
    prototypes = {}
    
    train_sets = {
        'mnist': mnist_train,
        'fmnist': fmnist_train,
        'kmnist': kmnist_train
    }
    
    for name in ['mnist', 'fmnist', 'kmnist']:
        print(f"\nProcessing Expert {name} metadata...")
        expert = experts[name]
        fisher = compute_diagonal_fisher(expert, train_sets[name], num_samples=CALIBRATION_SIZE)
        fisher_infos[name] = fisher
        
        proto = extract_class_prototypes(expert, train_sets[name], num_samples=CALIBRATION_SIZE)
        prototypes[name] = proto
        
    torch.save(fisher_infos, "checkpoints/fisher_infos.pt")
    torch.save(prototypes, "checkpoints/prototypes.pt")
    print("\nAll checkpoints and metadata successfully saved to 'checkpoints/' directory.")

if __name__ == "__main__":
    main()
