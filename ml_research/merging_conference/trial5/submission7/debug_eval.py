import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

# Disable cuDNN
torch.backends.cudnn.enabled = False

def load_base_model():
    from torchvision.models import ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    pretrained_conv1 = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(pretrained_conv1.mean(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def debug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load expert
    experts = {}
    experts['mnist'] = torch.load('experts/mnist_expert.pt', map_location='cpu')
    experts['fashion'] = torch.load('experts/fashion_expert.pt', map_location='cpu')
    experts['kmnist'] = torch.load('experts/kmnist_expert.pt', map_location='cpu')
    
    # Load clean MNIST test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    subset = Subset(mnist_test, list(range(500)))
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    merged_model = load_base_model().to(device)
    base_weights = load_base_model().state_dict()
    
    # Pre-move weights to device
    base_weights_dev = {k: v.to(device) for k, v in base_weights.items()}
    experts_dev = {
        'mnist': {k: v.to(device) for k, v in experts['mnist'].items()},
        'fashion': {k: v.to(device) for k, v in experts['fashion'].items()},
        'kmnist': {k: v.to(device) for k, v in experts['kmnist'].items()}
    }
    backbone_layers = [name for name, param in merged_model.state_dict().items() if 'fc' not in name and param.is_floating_point()]

    # Helper to blend
    def blend_weights_local(lambdas):
        with torch.no_grad():
            merged_state = merged_model.state_dict()
            for name, param in merged_state.items():
                if name in lambdas and param.is_floating_point():
                    l = lambdas[name]
                    v1 = experts_dev['mnist'][name] - base_weights_dev[name]
                    v2 = experts_dev['fashion'][name] - base_weights_dev[name]
                    v3 = experts_dev['kmnist'][name] - base_weights_dev[name]
                    param.copy_(base_weights_dev[name] + l[0] * v1 + l[1] * v2 + l[2] * v3)
            merged_model.load_state_dict(merged_state)
                    
    # Test 1: l = [1, 0, 0] (should be MNIST expert itself)
    print("Testing l = [1, 0, 0] (Pure MNIST expert)...")
    lambdas = {name: torch.tensor([1.0, 0.0, 0.0], device=device) for name in backbone_layers}
    blend_weights_local(lambdas)
    
    with torch.no_grad():
        merged_model.fc.weight.copy_(experts_dev['mnist']['fc.weight'])
        merged_model.fc.bias.copy_(experts_dev['mnist']['fc.bias'])
    merged_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = merged_model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")
    
    # Test 2: l = [1/3, 1/3, 1/3] (Uniform merge)
    print("\nTesting l = [1/3, 1/3, 1/3] (Uniform merge)...")
    lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in backbone_layers}
    blend_weights_local(lambdas)
    
    with torch.no_grad():
        merged_model.fc.weight.copy_(experts_dev['mnist']['fc.weight'])
        merged_model.fc.bias.copy_(experts_dev['mnist']['fc.bias'])
    merged_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = merged_model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    debug()
