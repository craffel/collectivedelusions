import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

# Load datasets and models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_mnist = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_kmnist = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=False)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

def get_grayscale_resnet18(num_classes=10):
    resnet = models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

static_model = get_grayscale_resnet18()
# Just load one of the saved expert weights to instantiate
static_model.load_state_dict(torch.load('models/mnist_expert.pt', map_location=device))
static_model.to(device)
static_model.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)

static_feat_extractor = FeatureExtractor(static_model)
static_feat_extractor.eval()

# Let's get 1 batch of each
mnist_loader = torch.utils.data.DataLoader(test_mnist, batch_size=64, shuffle=False)
kmnist_loader = torch.utils.data.DataLoader(test_kmnist, batch_size=64, shuffle=False)
fashion_loader = torch.utils.data.DataLoader(test_fashion, batch_size=64, shuffle=False)

mnist_x, _ = next(iter(mnist_loader))
kmnist_x, _ = next(iter(kmnist_loader))
fashion_x, _ = next(iter(fashion_loader))

mnist_x = mnist_x.to(device)
kmnist_x = kmnist_x.to(device)
fashion_x = fashion_x.to(device)

with torch.no_grad():
    f_mnist = static_feat_extractor(mnist_x)
    f_kmnist = static_feat_extractor(kmnist_x)
    f_fashion = static_feat_extractor(fashion_x)

print("MNIST features norm mean:", f_mnist.norm(dim=1).mean().item())
print("KMNIST features norm mean:", f_kmnist.norm(dim=1).mean().item())
print("FashionMNIST features norm mean:", f_fashion.norm(dim=1).mean().item())

# Let's see some cosine similarities between fashion features and mnist/kmnist features
# Let's check how they correlate
sims_mf = F.cosine_similarity(f_mnist[0:1], f_fashion, dim=1)
print("Cosine sim MNIST[0] vs FashionMNIST batches:", sims_mf.mean().item())
