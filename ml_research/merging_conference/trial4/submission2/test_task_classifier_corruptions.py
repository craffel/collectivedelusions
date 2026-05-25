import torch
import torch.nn as nn
from train_experts import get_dataloader
import torchvision.transforms.functional as TF
from train_task_classifier import TinyTaskClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def evaluate_classifier_corruptions():
    classifier = TinyTaskClassifier().to(device)
    classifier.load_state_dict(torch.load('./experts/tiny_task_classifier_robust.pt', map_location=device))
    classifier.eval()
    
    test_loaders = {
        0: get_dataloader('mnist', batch_size=64, train=False),
        1: get_dataloader('fashion', batch_size=64, train=False),
        2: get_dataloader('kmnist', batch_size=64, train=False)
    }
    
    corruptions = ['clean', 'gaussian_noise', 'gaussian_blur', 'contrast']
    
    for corr in corruptions:
        correct = 0
        total = 0
        with torch.no_grad():
            for task_id in range(3):
                iterator = iter(test_loaders[task_id])
                for _ in range(50):
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
                    
        print(f"Corruption: {corr.upper():15s} | Tiny Classifier Accuracy: {100.0 * correct / total:.2f}%")

if __name__ == '__main__':
    evaluate_classifier_corruptions()
