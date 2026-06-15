import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import timm
import time

device = torch.device("cpu")
print("Setting up datasets...")
transform_train_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform_train_gray)
subset = Subset(mnist_train, list(range(128)))
loader = DataLoader(subset, batch_size=64, shuffle=True)

model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.reset_classifier(num_classes=10)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
start_time = time.time()
model.train()
for epoch in range(5):
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
end_time = time.time()
print(f"Time taken for 128 images, 5 epochs: {end_time - start_time:.2f} seconds")
