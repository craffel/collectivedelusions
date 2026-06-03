import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import argparse

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid initialization errors

def get_model(arch, num_classes=10):
    if arch == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, "fc"
    elif arch == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model, "heads.head"
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def get_dataloader(task, train=True, batch_size=128, image_size=32):
    # Transforms: CIFAR10, SVHN, and FMNIST
    # FMNIST is grayscale, so we convert it to RGB (3 channels) and resize to image_size to match CIFAR10/SVHN
    if task == 'cifar10':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)
    elif task == 'svhn':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        split = "train" if train else "test"
        dataset = torchvision.datasets.SVHN(root="./data", split=split, download=False, transform=transform)
    elif task == 'fmnist':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),  # Convert to 3 channels
            T.ToTensor(),
            T.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
        ])
        dataset = torchvision.datasets.FashionMNIST(root="./data", train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_loss = running_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.argument_default = argparse.SUPPRESS
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vit_b_16'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, required=True, choices=['cifar10', 'svhn', 'fmnist'])
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Save directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Instantiate base model and save its initial pre-trained backbone if not already saved
    # The backbone includes all parameters except the classifier head (e.g. fc or heads.head)
    base_model, head_name = get_model(args.arch)
    
    # Extract backbone state dict
    def get_backbone_state(model, head_name):
        full_sd = model.state_dict()
        backbone_sd = {k: v for k, v in full_sd.items() if not k.startswith(head_name)}
        return backbone_sd
    
    base_backbone_path = f"checkpoints/{args.arch}_base_backbone.pt"
    if not os.path.exists(base_backbone_path):
        print(f"Saving base pre-trained backbone to {base_backbone_path}...")
        torch.save(get_backbone_state(base_model, head_name), base_backbone_path)
    
    # 2. Train the model on the task starting from base pre-trained weights
    # We load the base backbone weights into our model, and initialize a fresh head
    print(f"Loading base pre-trained backbone for training on {args.task}...")
    base_sd = torch.load(base_backbone_path, map_location='cpu')
    base_model.load_state_dict({**base_sd, **{k: v for k, v in base_model.state_dict().items() if k.startswith(head_name)}})
    
    base_model = base_model.to(device)
    
    img_size = 224 if args.arch == 'vit_b_16' else 32
    train_loader = get_dataloader(args.task, train=True, batch_size=args.batch_size, image_size=img_size)
    test_loader = get_dataloader(args.task, train=False, batch_size=args.batch_size, image_size=img_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"Starting training on {args.task} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(base_model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(base_model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {loss:.4f}, Train Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
    # Save the fine-tuned backbone and task head
    backbone_path = f"checkpoints/{args.arch}_{args.task}_backbone.pt"
    head_path = f"checkpoints/{args.arch}_{args.task}_head.pt"
    
    print(f"Saving fine-tuned backbone to {backbone_path}...")
    torch.save(get_backbone_state(base_model, head_name), backbone_path)
    
    print(f"Saving fine-tuned head to {head_path}...")
    head_sd = {k: v for k, v in base_model.state_dict().items() if k.startswith(head_name)}
    torch.save(head_sd, head_path)
    
    print(f"Task {args.task} completed and checkpoints saved successfully!\n")

if __name__ == "__main__":
    main()
