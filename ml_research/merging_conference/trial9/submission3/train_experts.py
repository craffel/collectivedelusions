import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import SimpleCNN
import data
import os

def train_model(model, train_loader, epochs=2, lr=1e-3, weight_decay=1e-4, device='cpu'):
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if model.use_cosface:
                logits = model(x, y)
            else:
                logits = model(x)
                
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    return model

def compute_prototypes(model, cal_dataset, device='cpu'):
    model = model.to(device)
    model.eval()
    
    class_features = {c: [] for c in range(10)}
    
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for x, y in cal_loader:
            x = x.to(device)
            features = model.get_features(x)
            for i in range(x.size(0)):
                c = y[i].item()
                class_features[c].append(features[i])
                
    l2_prototypes = {}
    spherical_prototypes = {}
    
    for c in range(10):
        feats = class_features[c]
        if len(feats) > 0:
            feats = torch.stack(feats)
            l2_prototypes[c] = feats.mean(dim=0)
            
            feats_norm = feats / torch.norm(feats, p=2, dim=1, keepdim=True)
            avg_feat = feats_norm.mean(dim=0)
            spherical_prototypes[c] = avg_feat / torch.norm(avg_feat, p=2)
        else:
            l2_prototypes[c] = torch.zeros(128, device=device)
            spherical_prototypes[c] = torch.zeros(128, device=device)
            
    l2_tensor = torch.stack([l2_prototypes[c] for c in range(10)])
    sph_tensor = torch.stack([spherical_prototypes[c] for c in range(10)])
    
    return l2_tensor, sph_tensor

def compute_fisher_sensitivities(model, cal_dataset, device='cpu'):
    model = model.to(device)
    model.eval()
    
    # We want to compute empirical Fisher for each parameter tensor.
    # Initialize Fisher accumulators to zero
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    total_samples = 0
    for x, y in cal_loader:
        x, y = x.to(device), y.to(device)
        total_samples += x.size(0)
        
        # For each sample in batch, compute gradients independently
        for i in range(x.size(0)):
            model.zero_grad()
            xi = x[i].unsqueeze(0)
            yi = y[i].unsqueeze(0)
            
            if model.use_cosface:
                logits = model(xi, yi)
            else:
                logits = model(xi)
                
            loss = criterion(logits, yi)
            loss.backward()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                        
    # Average Fisher values
    for name in fisher:
        fisher[name] /= total_samples
        
    # Layer-wise average sensitivities (scalar per parameter tensor)
    sensitivities = {}
    for name, f_val in fisher.items():
        sensitivities[name] = f_val.mean().item()
        
    return sensitivities

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    mnist_train, _, fmnist_train, _, _ = data.get_datasets()
    mnist_sub, fmnist_sub = data.get_train_subsets(mnist_train, fmnist_train)
    mnist_cal, fmnist_cal = data.get_calibration_samples(mnist_train, fmnist_train)
    
    mnist_loader = DataLoader(mnist_sub, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=64, shuffle=True)
    
    # Train Standard Experts
    print("\n--- Training Standard MNIST Expert ---")
    std_mnist = SimpleCNN(use_cosface=False)
    std_mnist = train_model(std_mnist, mnist_loader, device=device)
    
    print("\n--- Training Standard FashionMNIST Expert ---")
    std_fmnist = SimpleCNN(use_cosface=False)
    std_fmnist = train_model(std_fmnist, fmnist_loader, device=device)
    
    # Train CosFace Experts
    print("\n--- Training CosFace MNIST Expert ---")
    cos_mnist = SimpleCNN(use_cosface=True)
    cos_mnist = train_model(cos_mnist, mnist_loader, device=device)
    
    print("\n--- Training CosFace FashionMNIST Expert ---")
    cos_fmnist = SimpleCNN(use_cosface=True)
    cos_fmnist = train_model(cos_fmnist, fmnist_loader, device=device)
    
    # Save model checkpoints
    torch.save(std_mnist.state_dict(), 'checkpoints/std_mnist.pt')
    torch.save(std_fmnist.state_dict(), 'checkpoints/std_fmnist.pt')
    torch.save(cos_mnist.state_dict(), 'checkpoints/cos_mnist.pt')
    torch.save(cos_fmnist.state_dict(), 'checkpoints/cos_fmnist.pt')
    
    # Compute and save prototypes
    print("\n--- Computing Prototypes ---")
    std_mnist_l2, std_mnist_sph = compute_prototypes(std_mnist, mnist_cal, device=device)
    std_fmnist_l2, std_fmnist_sph = compute_prototypes(std_fmnist, fmnist_cal, device=device)
    cos_mnist_l2, cos_mnist_sph = compute_prototypes(cos_mnist, mnist_cal, device=device)
    cos_fmnist_l2, cos_fmnist_sph = compute_prototypes(cos_fmnist, fmnist_cal, device=device)
    
    # Compute Fisher sensitivities
    print("\n--- Computing Fisher Sensitivities ---")
    std_mnist_fish = compute_fisher_sensitivities(std_mnist, mnist_cal, device=device)
    std_fmnist_fish = compute_fisher_sensitivities(std_fmnist, fmnist_cal, device=device)
    cos_mnist_fish = compute_fisher_sensitivities(cos_mnist, mnist_cal, device=device)
    cos_fmnist_fish = compute_fisher_sensitivities(cos_fmnist, fmnist_cal, device=device)
    
    prototypes = {
        'std_mnist_l2': std_mnist_l2.cpu(),
        'std_mnist_sph': std_mnist_sph.cpu(),
        'std_fmnist_l2': std_fmnist_l2.cpu(),
        'std_fmnist_sph': std_fmnist_sph.cpu(),
        'cos_mnist_l2': cos_mnist_l2.cpu(),
        'cos_mnist_sph': cos_mnist_sph.cpu(),
        'cos_fmnist_l2': cos_fmnist_l2.cpu(),
        'cos_fmnist_sph': cos_fmnist_sph.cpu(),
        'std_mnist_fish': std_mnist_fish,
        'std_fmnist_fish': std_fmnist_fish,
        'cos_mnist_fish': cos_mnist_fish,
        'cos_fmnist_fish': cos_fmnist_fish,
    }
    torch.save(prototypes, 'checkpoints/prototypes.pt')
    print("Pre-training, prototype, and Fisher precomputation complete!")

if __name__ == '__main__':
    main()
