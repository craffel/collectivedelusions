import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_dataloaders(dataset_name, batch_size=128):
    # Normalize with standard ViT values (from ImageNet-1K)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip() if dataset_name == "cifar10" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == "svhn":
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # As specified in the papers, use fast-converging subsets of 10,000 training images per task
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(trainset), generator=g)[:10000].tolist()
    train_subset = Subset(trainset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size)
    
    print("Loading base ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )
    
    print("Applying LoRA config...")
    model = get_peft_model(model, peft_config)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)
            
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).logits
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}: Train Loss: {running_loss/total:.4f} | Train Acc: {100.*correct/total:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"Saving best model checkpoint for {args.dataset}...")
            os.makedirs("checkpoints", exist_ok=True)
            model.save_pretrained(f"checkpoints/{args.dataset}_best")
            
    print(f"Training completed. Best Test Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA expert on CIFAR-10 or SVHN")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "svhn"], help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()
    train(args)
