import sys
sys.path.insert(0, "./env_packages")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

t_list = [
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
transform = transforms.Compose(t_list)

train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_subset = Subset(train_set, list(range(2000)))
test_subset = Subset(test_set, list(range(100)))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
model = model.to(device)

# Use a learning rate of 5e-5 (much better for pre-trained ViT)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=-1)
        correct += preds.eq(y).sum().item()
        total += x.size(0)
    print(f"Epoch {epoch+1} Train Acc: {correct/total*100:.2f}%")
