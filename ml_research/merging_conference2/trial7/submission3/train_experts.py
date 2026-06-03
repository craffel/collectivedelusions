import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from src.utils import get_datasets, get_resnet18_progenitor
import os

def train_one_expert(task_name, dataloader, device, num_epochs=5):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    
    # Load progenitor
    model = get_resnet18_progenitor(pretrained=True)
    
    # Modify the final linear layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save dir
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save progenitor first
    print("Saving progenitor model...")
    prog_model = get_resnet18_progenitor(pretrained=True)
    # Give progenitor a dummy 10-class fc head as well so we can easily copy-paste/merge later
    prog_model.fc = nn.Linear(prog_model.fc.in_features, 10)
    torch.save(prog_model.state_dict(), 'checkpoints/progenitor.pt')
    
    # Load loaders
    loaders = get_datasets(batch_size=128)
    
    # Train MNIST Expert (5 epochs)
    mnist_model = train_one_expert('mnist', loaders['mnist']['train'], device, num_epochs=5)
    torch.save(mnist_model.state_dict(), 'checkpoints/expert_mnist.pt')
    
    # Train FashionMNIST Expert (5 epochs)
    fashion_model = train_one_expert('fashion', loaders['fashion']['train'], device, num_epochs=5)
    torch.save(fashion_model.state_dict(), 'checkpoints/expert_fashion.pt')
    
    # Train CIFAR-10 Expert (10 epochs for better convergence)
    cifar_model = train_one_expert('cifar10', loaders['cifar10']['train'], device, num_epochs=10)
    torch.save(cifar_model.state_dict(), 'checkpoints/expert_cifar10.pt')
    
    print("\nAll experts trained and saved successfully.")

if __name__ == '__main__':
    main()
