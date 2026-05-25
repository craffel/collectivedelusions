import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    model = resnet18(num_classes=10)
    print("ResNet18 loaded successfully!")
except Exception as e:
    print("Error loading ResNet18:", e)
