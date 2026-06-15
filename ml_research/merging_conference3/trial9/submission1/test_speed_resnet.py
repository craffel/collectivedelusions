import torch
import torchvision.models as models
import time

print("Loading ResNet-18...")
model = models.resnet18(pretrained=True)
model.eval()

images = torch.randn(64, 3, 224, 224)

print("Starting forward pass on CPU for 64 images...")
start = time.time()
with torch.no_grad():
    x = model(images)
end = time.time()
print(f"64 images took: {end - start:.2f} seconds")
