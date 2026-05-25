import torch
import torch.nn as nn
import torch.optim as optim
from train_experts import get_dataloader
import torchvision.transforms.functional as TF
import random

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

# Unnormalize and normalize helper functions
def unnormalize(tensor):
    return tensor * 0.3081 + 0.1307

def normalize(tensor):
    return (tensor - 0.1307) / 0.3081

# Image corruptions
def apply_gaussian_noise(x, sigma=0.4):
    unnorm_x = unnormalize(x)
    noise = torch.randn_like(unnorm_x) * sigma
    corrupted_x = torch.clamp(unnorm_x + noise, 0.0, 1.0)
    return normalize(corrupted_x)

def apply_gaussian_blur(x, sigma=2.0):
    unnorm_x = unnormalize(x)
    corrupted_x = TF.gaussian_blur(unnorm_x, kernel_size=[5, 5], sigma=[sigma, sigma])
    return normalize(corrupted_x)

def apply_contrast_reduction(x, alpha=0.15):
    unnorm_x = unnormalize(x)
    corrupted_x = torch.clamp(0.5 + alpha * (unnorm_x - 0.5), 0.0, 1.0)
    return normalize(corrupted_x)

def augment_batch_randomly(images):
    augmented_images = []
    for i in range(images.size(0)):
        img = images[i:i+1]
        aug_type = random.choice(['clean', 'noise', 'blur', 'contrast'])
        if aug_type == 'noise':
            sig = random.uniform(0.1, 0.4)
            img = apply_gaussian_noise(img, sigma=sig)
        elif aug_type == 'blur':
            sig = random.uniform(0.5, 2.0)
            img = apply_gaussian_blur(img, sigma=sig)
        elif aug_type == 'contrast':
            alp = random.uniform(0.15, 0.5)
            img = apply_contrast_reduction(img, alpha=alp)
        augmented_images.append(img)
    return torch.cat(augmented_images, dim=0)

def train_robust_classifier():
    classifier = TinyTaskClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Load training sets
    loaders = {
        0: get_dataloader('mnist', batch_size=64, train=True),
        1: get_dataloader('fashion', batch_size=64, train=True),
        2: get_dataloader('kmnist', batch_size=64, train=True)
    }
    
    print("Training robust tiny task classifier with data augmentation...")
    classifier.train()
    
    # Train for 2 epochs to handle the augmented distribution
    for epoch in range(2):
        correct = 0
        total = 0
        
        iters = {k: iter(v) for k, v in loaders.items()}
        for step in range(300):
            for task_id in range(3):
                try:
                    images, _ = next(iters[task_id])
                except StopIteration:
                    iters[task_id] = iter(loaders[task_id])
                    images, _ = next(iters[task_id])
                    
                images = images.to(device)
                images_aug = augment_batch_randomly(images)
                labels = torch.full((images.size(0),), task_id, dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                outputs = classifier(images_aug)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        print(f"Epoch {epoch+1}/2 - Training Accuracy: {100.0 * correct / total:.2f}%")
        
    # Evaluate on test loaders with corruptions
    test_loaders = {
        0: get_dataloader('mnist', batch_size=64, train=False),
        1: get_dataloader('fashion', batch_size=64, train=False),
        2: get_dataloader('kmnist', batch_size=64, train=False)
    }
    
    classifier.eval()
    corruptions = ['clean', 'gaussian_noise', 'gaussian_blur', 'contrast']
    
    for corr in corruptions:
        correct = 0
        total = 0
        with torch.no_grad():
            for task_id in range(3):
                iterator = iter(test_loaders[task_id])
                for _ in range(30):
                    try:
                        images, _ = next(iterator)
                    except StopIteration:
                        break
                    images = images.to(device)
                    if corr == 'gaussian_noise':
                        images = apply_gaussian_noise(images, sigma=0.4)
                    elif corr == 'gaussian_blur':
                        images = apply_gaussian_blur(images, sigma=2.0)
                    elif corr == 'contrast':
                        images = apply_contrast_reduction(images, alpha=0.15)
                        
                    labels = torch.full((images.size(0),), task_id, dtype=torch.long, device=device)
                    outputs = classifier(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                    
        print(f"Corruption: {corr.upper():15s} | Tiny Robust Classifier Accuracy: {100.0 * correct / total:.2f}%")
        
    # Save the model
    torch.save(classifier.state_dict(), './experts/tiny_task_classifier_robust.pt')
    print("Saved robust tiny task classifier to ./experts/tiny_task_classifier_robust.pt")

if __name__ == '__main__':
    train_robust_classifier()
