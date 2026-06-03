import torch
import torch.nn as nn
from models import ResNet18Backbone, MLPBackbone, CompleteModel
import merging
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def get_loaders():
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
    mnist_subset = Subset(mnist_test, range(200)) # tiny subset for super fast run
    return DataLoader(mnist_subset, batch_size=64, shuffle=False)

# Load progenitor and experts
progenitor = ResNet18Backbone().to(device)
progenitor.load_state_dict(torch.load('checkpoints/resnet18_progenitor.pt', map_location=device))

experts = []
heads = {}
for task in ['mnist']:
    exp = ResNet18Backbone().to(device)
    exp.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_backbone.pt', map_location=device))
    experts.append(exp)
    
    head = nn.Linear(512, 10).to(device)
    head.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_head.pt', map_location=device))
    heads[task] = head

# Merge
print("Merging using Weight Averaging...")
merged_wa = merging.weight_averaging(experts)

print("Merging using TIES...")
merged_ties = merging.ties_merging(experts, progenitor, fraction=0.2)

print("Merging using WCPR...")
merged_wcpr = merging.wcpr_merging(experts, progenitor)

print("Merging using QR-SP-WCPR...")
merged_qr_sp_wcpr = merging.qr_sp_wcpr_merging(experts, progenitor, sign_merger='ties', fraction=0.2, gamma=2.0, scale_compensation=True)

test_loader = get_loaders()

for name, state in [("WA", merged_wa), ("TIES", merged_ties), ("WCPR", merged_wcpr), ("QR-SP-WCPR", merged_qr_sp_wcpr)]:
    bb = ResNet18Backbone().to(device)
    bb.load_state_dict(state)
    
    # Optional: Apply DE-BN calibration
    # let's see what happens without DE-BN
    model = CompleteModel(bb, heads['mnist'])
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, pred = outputs.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    print(f"{name} MNIST Accuracy (No BN Calibration): {100.0 * correct / total:.2f}%")
