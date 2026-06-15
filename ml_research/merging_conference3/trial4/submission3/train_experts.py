import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def get_transforms(grayscale=False):
    if grayscale:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_one_expert(task_name, dataset_class, grayscale, epochs=3, batch_size=256, subset_size=10000):
    print(f"\n--- Training Expert for {task_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    transform = get_transforms(grayscale=grayscale)
    if task_name == "SVHN":
        train_dataset = dataset_class(root="./data", split="train", download=True, transform=transform)
        test_dataset = dataset_class(root="./data", split="test", download=True, transform=transform)
    else:
        train_dataset = dataset_class(root="./data", train=True, download=True, transform=transform)
        test_dataset = dataset_class(root="./data", train=False, download=True, transform=transform)

    # Subsampling to speed up training while preserving accuracy
    if subset_size and len(train_dataset) > subset_size:
        print(f"Subsampling training set to {subset_size} samples...")
        indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, indices)
    
    # Subsample test set for faster evaluation
    test_subset_size = min(2000, len(test_dataset))
    print(f"Subsampling test set to {test_subset_size} samples...")
    test_indices = torch.randperm(len(test_dataset))[:test_subset_size].tolist()
    test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create base model
    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    
    # Freeze backbone
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Replace head
    base_model.head = nn.Linear(192, 10)
    
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["qkv"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Wrap in PEFT
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(device)
    
    # Head requires grad
    for param in peft_model.base_model.model.head.parameters():
        param.requires_grad = True

    peft_model.print_trainable_parameters()

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=2e-3,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(epochs):
        peft_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = peft_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # Evaluation
    peft_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = peft_model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save
    os.makedirs(f"checkpoints/{task_name.lower()}_lora", exist_ok=True)
    peft_model.save_pretrained(f"checkpoints/{task_name.lower()}_lora")
    torch.save(peft_model.base_model.model.head.state_dict(), f"checkpoints/{task_name.lower()}_lora/head.pt")
    print(f"Saved {task_name} adapter and head to checkpoints/{task_name.lower()}_lora/")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Train MNIST
    # train_one_expert("MNIST", datasets.MNIST, grayscale=True, epochs=3, subset_size=15000)
    
    # Train FashionMNIST
    # train_one_expert("FashionMNIST", datasets.FashionMNIST, grayscale=True, epochs=3, subset_size=15000)
    
    # Train CIFAR-10
    # train_one_expert("CIFAR10", datasets.CIFAR10, grayscale=False, epochs=4, subset_size=20000)
    
    # Train SVHN
    train_one_expert("SVHN", datasets.SVHN, grayscale=False, epochs=4, subset_size=20000)
