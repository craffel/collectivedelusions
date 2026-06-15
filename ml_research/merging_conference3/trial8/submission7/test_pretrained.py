import torch
import torchvision.models as models

try:
    print("Loading pre-trained ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    print("Successfully loaded pre-trained ResNet-18!")
    
    # Test a forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("Forward pass successful, output shape:", out.shape)
except Exception as e:
    print("Error:", e)
