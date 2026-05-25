import torch
import torch.nn as nn
import torch.optim as optim
from train_experts import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyTaskClassifier(nn.Module):
    def __init__(self):
        super(TinyTaskClassifier, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2) # 8x14x14
        self.pool = nn.MaxPool2d(2, 2) # 8x7x7
        self.fc = nn.Linear(8 * 7 * 7, 3)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_classifier():
    classifier = TinyTaskClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Load small training subsets
    loaders = {
        0: get_dataloader('mnist', batch_size=64, train=True),
        1: get_dataloader('fashion', batch_size=64, train=True),
        2: get_dataloader('kmnist', batch_size=64, train=True)
    }
    
    print("Training tiny task classifier...")
    classifier.train()
    
    # Train for just 1 epoch on a fraction of data to make it extremely fast
    for epoch in range(1):
        correct = 0
        total = 0
        
        # Interleave batches from the three loaders
        iters = {k: iter(v) for k, v in loaders.items()}
        for step in range(200): # 200 steps is plenty
            for task_id in range(3):
                try:
                    images, _ = next(iters[task_id])
                except StopIteration:
                    iters[task_id] = iter(loaders[task_id])
                    images, _ = next(iters[task_id])
                    
                images = images.to(device)
                labels = torch.full((images.size(0),), task_id, dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        print(f"Epoch {epoch+1} - Training Accuracy: {100.0 * correct / total:.2f}%")
        
    # Evaluate on test loaders
    test_loaders = {
        0: get_dataloader('mnist', batch_size=64, train=False),
        1: get_dataloader('fashion', batch_size=64, train=False),
        2: get_dataloader('kmnist', batch_size=64, train=False)
    }
    
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for task_id in range(3):
            iterator = iter(test_loaders[task_id])
            for _ in range(50): # evaluate on 50 batches
                try:
                    images, _ = next(iterator)
                except StopIteration:
                    break
                images = images.to(device)
                labels = torch.full((images.size(0),), task_id, dtype=torch.long, device=device)
                outputs = classifier(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
    print(f"Test Accuracy of Tiny Task Classifier: {100.0 * correct / total:.2f}%")
    
    # Save the model
    torch.save(classifier.state_dict(), './experts/tiny_task_classifier.pt')
    print("Saved tiny task classifier checkpoint to ./experts/tiny_task_classifier.pt")

if __name__ == '__main__':
    train_classifier()
