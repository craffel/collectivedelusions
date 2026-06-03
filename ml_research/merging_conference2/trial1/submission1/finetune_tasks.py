import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN as a fallback configuration.")

# Define transforms
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataloaders():
    # 1. CIFAR-10
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    # 2. SVHN
    svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=transform_rgb)
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
    
    # 3. FashionMNIST
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    loaders = {
        'cifar10': {
            'train': DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2),
            'test': DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)
        },
        'svhn': {
            'train': DataLoader(svhn_train, batch_size=128, shuffle=True, num_workers=2),
            'test': DataLoader(svhn_test, batch_size=128, shuffle=False, num_workers=2)
        },
        'fmnist': {
            'train': DataLoader(fmnist_train, batch_size=128, shuffle=True, num_workers=2),
            'test': DataLoader(fmnist_test, batch_size=128, shuffle=False, num_workers=2)
        }
    }
    return loaders

class ModelWrapper(nn.Module):
    def __init__(self, encoder, classifier):
        super(ModelWrapper, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        
    def forward(self, x):
        features = self.encoder(x)
        # Flatten the features for the classifier
        features = torch.flatten(features, 1)
        return self.classifier(features)

def train_task(task_name, train_loader, test_loader, epochs=2):
    print(f"\n--- Fine-tuning ResNet-18 on {task_name} ---")
    
    # Load pretrained ResNet-18
    # Use weights=models.ResNet18_Weights.IMAGENET1K_V1
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Extract the encoder (all layers except the final fc layer)
    encoder = nn.Sequential(*(list(resnet.children())[:-1]))
    encoder = encoder.to(device)
    
    # Classifier head
    classifier = nn.Linear(512, 10).to(device)
    
    model = ModelWrapper(encoder, classifier).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = train_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Eval
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
        
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    return encoder, classifier

def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    loaders = get_dataloaders()
    
    # Save the initial pre-trained base encoder
    resnet_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_encoder = nn.Sequential(*(list(resnet_base.children())[:-1]))
    torch.save(base_encoder.state_dict(), 'checkpoints/base_encoder.pt')
    print("Saved initial pre-trained base encoder.")
    
    # Fine-tune on each task
    for task_name in ['cifar10', 'svhn', 'fmnist']:
        train_loader = loaders[task_name]['train']
        test_loader = loaders[task_name]['test']
        
        encoder, classifier = train_task(task_name, train_loader, test_loader, epochs=2)
        
        # Save checkpoints
        torch.save(encoder.state_dict(), f'checkpoints/{task_name}_encoder.pt')
        torch.save(classifier.state_dict(), f'checkpoints/{task_name}_classifier.pt')
        print(f"Saved fine-tuned checkpoints for {task_name}.")

if __name__ == "__main__":
    main()
