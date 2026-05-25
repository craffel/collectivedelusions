import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from models import MultiTaskCNN

def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.enabled = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple transform for grayscale 28x28 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 1. Load datasets
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    
    kmnist_train = torchvision.datasets.KMNIST("./data", train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    # 2. Extract subsets (10k train, 2k test)
    # Using deterministic subsets
    train_indices = list(range(10000))
    test_indices = list(range(2000))
    
    datasets = {
        0: {"train": Subset(mnist_train, train_indices), "test": Subset(mnist_test, test_indices), "name": "MNIST"},
        1: {"train": Subset(fmnist_train, train_indices), "test": Subset(fmnist_test, test_indices), "name": "FashionMNIST"},
        2: {"train": Subset(kmnist_train, train_indices), "test": Subset(kmnist_test, test_indices), "name": "KMNIST"}
    }
    
    # 3. Train Experts
    # Create and save a shared initial checkpoint so all experts start from the exact same weights
    shared_init_model = MultiTaskCNN(num_tasks=3, num_classes=10).to(device)
    shared_init_path = "./checkpoints/shared_init.pt"
    torch.save(shared_init_model.state_dict(), shared_init_path)
    print(f"Saved shared initialization to {shared_init_path}")
    
    experts = []
    prototypes = torch.zeros(3, 10, 128) # [num_tasks, num_classes, embed_dim]
    
    for task_idx in range(3):
        name = datasets[task_idx]["name"]
        print(f"\n--- Training Expert for {name} (Task {task_idx}) ---")
        
        train_loader = DataLoader(datasets[task_idx]["train"], batch_size=64, shuffle=True)
        test_loader = DataLoader(datasets[task_idx]["test"], batch_size=64, shuffle=False)
        
        # Initialize model with the shared initialization
        model = MultiTaskCNN(num_tasks=3, num_classes=10).to(device)
        model.load_state_dict(torch.load(shared_init_path, map_location=device))
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(5):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits, _ = model(images, task_idx)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            epoch_loss = running_loss / total
            epoch_acc = correct / total * 100
            print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
            
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images, task_idx)
                _, predicted = logits.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = test_correct / test_total * 100
        print(f"Finished {name} - Test Acc: {test_acc:.2f}%")
        
        # Save expert weights
        ckpt_path = f"./checkpoints/expert_{task_idx}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved expert weights to {ckpt_path}")
        experts.append(model)
        
        # 4. Compute class prototypes
        print(f"Computing prototypes for {name}...")
        proto_loader = DataLoader(datasets[task_idx]["train"], batch_size=256, shuffle=False)
        class_embeddings = {c: [] for c in range(10)}
        
        with torch.no_grad():
            for images, labels in proto_loader:
                images = images.to(device)
                # Extract embeddings via model's backbone
                embeddings = model.backbone(images) # [B, 128]
                for i, label in enumerate(labels):
                    class_embeddings[label.item()].append(embeddings[i].cpu())
                    
        for c in range(10):
            if len(class_embeddings[c]) > 0:
                class_tensor = torch.stack(class_embeddings[c]) # [num_samples_of_c, 128]
                mean_embedding = class_tensor.mean(dim=0)
                # L2 normalization
                mean_embedding = mean_embedding / torch.norm(mean_embedding, p=2)
                prototypes[task_idx, c] = mean_embedding
            else:
                print(f"Warning: No samples found for class {c} in task {task_idx}")
                
    # Save the prototypes
    torch.save(prototypes, "./checkpoints/prototypes.pt")
    print("\nSuccessfully computed and saved prototypes of shape:", prototypes.shape)
    print("Saved prototypes to ./checkpoints/prototypes.pt")

if __name__ == "__main__":
    main()
