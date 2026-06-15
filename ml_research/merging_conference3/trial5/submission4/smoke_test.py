import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, Subset

# Directories
CHECKPOINT_DIR = "./checkpoints_test"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

device = torch.device("cpu")
print("Running smoke test on CPU...")

# Transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)), # small resize for smoke test speed
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

tasks = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

def get_dataset(task_name, train=True):
    if task_name == "MNIST":
        return torchvision.datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform_gray)
    elif task_name == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform_gray)
    elif task_name == "CIFAR10":
        return torchvision.datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=transform_rgb)
    elif task_name == "SVHN":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root=DATA_DIR, split=split, download=True, transform=transform_rgb)

def run_smoke_test():
    print("Step 1: Training 4 experts for 1 batch step...")
    for task in tasks:
        print(f"  Training {task}...")
        dataset = get_dataset(task, train=True)
        subset = Subset(dataset, list(range(2))) # only 2 samples
        loader = DataLoader(subset, batch_size=2, shuffle=False)
        
        # We can use vit_tiny_patch16_224 but we can override resize_img to False to speed up
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for images, labels in loader:
            # We must resize images to 224x224 since vit_tiny_patch16_224 expects 224x224
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break # only 1 step
            
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"expert_{task.lower()}.pth"))
        print(f"  Saved {task} checkpoint.")

    print("\nStep 2: Checking merge_and_eval compatibility...")
    import merge_and_eval
    # Override checkpoints and dataset directories
    merge_and_eval.CHECKPOINT_DIR = CHECKPOINT_DIR
    merge_and_eval.device = device
    
    # We will run a mini version of run_experiments()
    print("Running mini merge and eval...")
    merge_and_eval.run_experiments()

if __name__ == "__main__":
    run_smoke_test()
