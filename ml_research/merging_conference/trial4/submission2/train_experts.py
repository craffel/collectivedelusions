import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

# Define CNN Base Encoder
class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Projection layer
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc(x))
        return x

# Define Classification Head
class ClassHead(nn.Module):
    def __init__(self):
        super(ClassHead, self).__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

# Expert Model consisting of Encoder and Task Head
class ExpertModel(nn.Module):
    def __init__(self, encoder, head):
        super(ExpertModel, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

def get_dataloader(dataset_name, batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'fashion':
        dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'kmnist':
        dataset = datasets.KMNIST(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)

def train_expert(dataset_name, shared_encoder, save_path):
    print(f"\n--- Training Expert on {dataset_name.upper()} ---")
    
    # Initialize task head
    head = ClassHead()
    model = ExpertModel(shared_encoder, head)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_loader = get_dataloader(dataset_name, batch_size=64, train=True)
    test_loader = get_dataloader(dataset_name, batch_size=64, train=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * correct / total
    print(f"Test Accuracy on {dataset_name.upper()}: {test_acc:.2f}%")
    
    # Save checkpoints
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': shared_encoder.state_dict(),
        'head_state_dict': head.state_dict(),
        'test_acc': test_acc
    }, save_path)
    print(f"Saved expert model checkpoint to {save_path}")

if __name__ == '__main__':
    # Initialize shared encoder to start from the same base initialization
    shared_encoder = BaseEncoder()
    
    # Save the initial shared encoder weights (O_pre)
    os.makedirs('./experts', exist_ok=True)
    torch.save(shared_encoder.state_dict(), './experts/base_encoder_init.pt')
    print("Saved base encoder initialization weights.")
    
    # Train each expert starting from the shared base encoder initialization
    # We create a copy of the base encoder state dict to initialize each training
    datasets_to_train = ['mnist', 'fashion', 'kmnist']
    for ds in datasets_to_train:
        # Load the initial base encoder weights to start from the same shared init
        encoder = BaseEncoder()
        encoder.load_state_dict(torch.load('./experts/base_encoder_init.pt', weights_only=True))
        train_expert(ds, encoder, f'./experts/{ds}_expert.pt')
