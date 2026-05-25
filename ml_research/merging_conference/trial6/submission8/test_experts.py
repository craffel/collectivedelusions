import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

class ResNetWithFeatures(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base_model = resnet18()
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.base_model.fc(features)
        return features, logits

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    loaders = {
        'expert1_cifar': DataLoader(Subset(cifar_test, list(range(500))), batch_size=64, shuffle=False),
        'expert2_svhn': DataLoader(Subset(svhn_test, list(range(500))), batch_size=64, shuffle=False),
        'expert3_fmnist': DataLoader(Subset(fmnist_test, list(range(500))), batch_size=64, shuffle=False)
    }
    
    for name, loader in loaders.items():
        model = ResNetWithFeatures()
        model.base_model.load_state_dict(torch.load(f"experts/{name}.pt", map_location=device))
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, logits = model(x)
                _, preds = logits.max(1)
                correct += preds.eq(y).sum().item()
                total += len(y)
        print(f"Accuracy for {name}: {100.0 * correct / total:.2f}%")

if __name__ == '__main__':
    main()
