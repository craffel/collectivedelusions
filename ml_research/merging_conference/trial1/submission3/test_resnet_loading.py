import torch
import torchvision.models as models

print("Testing ResNet-18 loading with default weights...")
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print("Pre-trained ResNet-18 loaded successfully!")
    print(model.fc)
except Exception as e:
    print("Error loading pre-trained ResNet-18:", e)
