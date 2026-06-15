import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

# Create a tiny training subset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR10 subset...")
dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
subset = Subset(dataset, range(64)) # just 64 images for testing speed
loader = DataLoader(subset, batch_size=16, shuffle=True)

print("Creating model...")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)

# We can freeze early layers to speed up training
# Let's freeze blocks 0 to 8, and keep blocks 9 to 11 trainable, plus head
print("Freezing early layers...")
for name, param in model.named_parameters():
    if "blocks" in name:
        block_idx = int(name.split("blocks.")[1].split(".")[0])
        if block_idx < 9:
            param.requires_grad = False
    elif "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Running 1 epoch...")
start_time = time.time()
model.train()
for batch_idx, (images, targets) in enumerate(loader):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Batch {batch_idx+1}/{len(loader)} loss: {loss.item():.4f}")

end_time = time.time()
print(f"Epoch finished in {end_time - start_time:.2f} seconds!")
