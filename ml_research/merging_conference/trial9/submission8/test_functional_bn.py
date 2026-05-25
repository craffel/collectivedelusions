import torch
import torch.nn as nn
from torch.func import functional_call
import torchvision
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(64 * 3 * 3, 10)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        feat = self.pool3(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        if return_features:
            return out, feat
        return out

def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Load dataset
    print("Loading datasets...")
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
    
    # Load Expert 0 (MNIST)
    expert0 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load("expert0.pt", map_location=device))
    
    # Setup base merged model
    merged_model = SimpleCNN().to(device)
    
    # Get a noisy batch
    mnist_iter = iter(mnist_loader)
    clean_inputs, targets = next(mnist_iter)
    noisy_inputs = clean_inputs + 0.6 * torch.randn_like(clean_inputs)
    noisy_inputs = torch.clamp(noisy_inputs, -1.0, 1.0)
    
    # Case 1: BN in eval mode in merged_model
    merged_model.eval()
    state0 = expert0.state_dict()
    out1 = functional_call(merged_model, state0, args=(noisy_inputs,))
    acc1 = 100.0 * out1.max(1)[1].eq(targets).sum().item() / targets.size(0)
    print(f"Noisy MNIST with functional_call (BN eval): {acc1:.2f}%")
    
    # Case 2: BN in train mode in merged_model
    merged_model.eval()
    for m in merged_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            
    out2 = functional_call(merged_model, state0, args=(noisy_inputs,))
    acc2 = 100.0 * out2.max(1)[1].eq(targets).sum().item() / targets.size(0)
    print(f"Noisy MNIST with functional_call (BN train/adapt): {acc2:.2f}%")

if __name__ == "__main__":
    main()
