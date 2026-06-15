import torch
import torchvision
import torchvision.transforms as transforms
import timm
import time

print("Starting test...")
start_time = time.time()

# Dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download a small slice of MNIST
try:
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(10))
    loader = torch.utils.data.DataLoader(subset, batch_size=2, shuffle=True)
    print("Dataset downloaded and loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)

# Model
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Single step
x, y = next(iter(loader))
print("Input shape:", x.shape)
print("Target shape:", y.shape)

out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()

print(f"Forward/backward step successful. Loss: {loss.item():.4f}")
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
