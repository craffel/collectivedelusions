import torch
import torchvision
import torchvision.transforms as transforms
import timm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training SVHN expert on:", device)

transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_svhn = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_color)
test_svhn = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_color)

# Set seed for reproducibility
torch.manual_seed(42)

train_indices = torch.randperm(len(train_svhn))[:1500]
train_subset = torch.utils.data.Subset(train_svhn, train_indices)
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)

test_indices = torch.randperm(len(test_svhn))[:200]
test_subset = torch.utils.data.Subset(test_svhn, test_indices)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10).to(device)
model.load_state_dict(torch.load('./checkpoints/svhn_expert.pt', map_location=device))

optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

def evaluate(m, loader):
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = m(x.to(device))
            preds = out.argmax(dim=-1)
            correct += (preds == y.to(device)).sum().item()
            total += len(y)
    return correct / total * 100.0

print(f"Before fine-tuning, test accuracy on 200 samples: {evaluate(model, test_loader):.2f}%")

model.train()
for epoch in range(5):
    loss_sum = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x.to(device))
        loss = criterion(out, y.to(device))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch} loss: {loss_sum / len(train_loader):.4f}")

accuracy = evaluate(model, test_loader)
print(f"After fine-tuning, test accuracy on 200 samples: {accuracy:.2f}%")

if accuracy > 65.0:
    print("Saving the updated SVHN expert to disk...")
    torch.save(model.state_dict(), './checkpoints/svhn_expert.pt')
else:
    print("Warning: Accuracy did not meet threshold, not saving!")
