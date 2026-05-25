import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from train_experts import CustomCNN, ExpertModel

def compute_diagonal_fisher(model, dataset, device, num_samples=200):
    model.eval()
    loader = DataLoader(Subset(dataset, range(min(num_samples, len(dataset)))), batch_size=1, shuffle=False)
    
    # Initialize dictionary for storing diagonal Fisher
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    criterion = nn.CrossEntropyLoss()
    
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # We compute gradient of log-likelihood for single sample
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
        
        count += 1
        
    # Average across samples
    for name in fisher_dict:
        fisher_dict[name] /= count
        
    return fisher_dict

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load raw datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    fashion_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST('./data', train=True, download=True, transform=transform)
    
    # Initialize base encoder
    base_encoder = CustomCNN().to(device)
    
    # Load 3 expert models
    expert_mnist = ExpertModel(base_encoder).to(device)
    expert_mnist.load_state_dict(torch.load("./experts/expert_mnist.pt", map_location=device))
    
    expert_fashion = ExpertModel(base_encoder).to(device)
    expert_fashion.load_state_dict(torch.load("./experts/expert_fashion.pt", map_location=device))
    
    expert_kmnist = ExpertModel(base_encoder).to(device)
    expert_kmnist.load_state_dict(torch.load("./experts/expert_kmnist.pt", map_location=device))
    
    print("Computing diagonal Fisher Information for MNIST Expert...")
    fisher_mnist = compute_diagonal_fisher(expert_mnist, mnist_train, device)
    
    print("Computing diagonal Fisher Information for FashionMNIST Expert...")
    fisher_fashion = compute_diagonal_fisher(expert_fashion, fashion_train, device)
    
    print("Computing diagonal Fisher Information for KMNIST Expert...")
    fisher_kmnist = compute_diagonal_fisher(expert_kmnist, kmnist_train, device)
    
    # Save Fisher matrices
    os.makedirs("./fisher", exist_ok=True)
    torch.save(fisher_mnist, "./fisher/fisher_mnist.pt")
    torch.save(fisher_fashion, "./fisher/fisher_fashion.pt")
    torch.save(fisher_kmnist, "./fisher/fisher_kmnist.pt")
    
    print("Fisher computation complete and saved.")

if __name__ == "__main__":
    main()
