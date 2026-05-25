import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models import MultiTaskCNN, merge_backbone

def check():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load raw datasets and subsets
    mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(Subset(mnist_test, list(range(2000))), batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(Subset(fmnist_test, list(range(2000))), batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(Subset(kmnist_test, list(range(2000))), batch_size=64, shuffle=False)
    
    # Load experts
    experts = []
    for k in range(3):
        model = MultiTaskCNN(num_tasks=3, num_classes=10)
        model.load_state_dict(torch.load(f"./checkpoints/expert_{k}.pt", map_location=device))
        model.eval()
        experts.append(model)
        
    print("Evaluating individual experts directly on their respective tasks:")
    
    # 1. Expert 0 on MNIST
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in mnist_loader:
            logits, _ = experts[0](imgs, 0)
            _, preds = logits.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)
    print(f"Expert 0 on MNIST directly: {correct/total*100:.2f}%")
    
    # 2. Merged model with lambda = [1.0, 0.0, 0.0] on MNIST
    merged_model = MultiTaskCNN(num_tasks=3, num_classes=10)
    merge_backbone(merged_model, experts, torch.tensor([1.0, 0.0, 0.0]))
    merged_model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in mnist_loader:
            logits, _ = merged_model(imgs, 0)
            _, preds = logits.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)
    print(f"Merged [1, 0, 0] on MNIST: {correct/total*100:.2f}%")
    
    # 3. Merged model with lambda = [0.0, 1.0, 0.0] on FashionMNIST
    merge_backbone(merged_model, experts, torch.tensor([0.0, 1.0, 0.0]))
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in fmnist_loader:
            logits, _ = merged_model(imgs, 1)
            _, preds = logits.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)
    print(f"Merged [0, 1, 0] on FashionMNIST: {correct/total*100:.2f}%")

    # 4. Merged model with lambda = [0.0, 0.0, 1.0] on KMNIST
    merge_backbone(merged_model, experts, torch.tensor([0.0, 0.0, 1.0]))
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in kmnist_loader:
            logits, _ = merged_model(imgs, 2)
            _, preds = logits.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)
    print(f"Merged [0, 0, 1] on KMNIST: {correct/total*100:.2f}%")

if __name__ == "__main__":
    check()
