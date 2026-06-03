import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

# Define dataset-specific classes and prompts
DATASET_INFO = {
    "cifar10": {
        "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "prompt": "a photo of a {}"
    },
    "svhn": {
        "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        "prompt": "a photo of the digit {}"
    },
    "mnist": {
        "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        "prompt": "a photo of the handwritten digit {}"
    }
}

def get_dataset(task, split="train", limit=None):
    # Setup interpolation
    interpolation = T.InterpolationMode.BICUBIC
    
    # Standard CLIP normalization
    norm_transform = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    if task == "mnist":
        transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((224, 224), interpolation=interpolation),
            T.ToTensor(),
            norm_transform
        ])
    else:
        transform = T.Compose([
            T.Resize((224, 224), interpolation=interpolation),
            T.ToTensor(),
            norm_transform
        ])
        
    if task == "cifar10":
        ds = torchvision.datasets.CIFAR10(root="./data", train=(split=="train"), download=True, transform=transform)
    elif task == "svhn":
        ds = torchvision.datasets.SVHN(root="./data", split=split, download=True, transform=transform)
    elif task == "mnist":
        ds = torchvision.datasets.MNIST(root="./data", train=(split=="train"), download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")
        
    if limit is not None and limit < len(ds):
        # Deterministic subset selection
        indices = list(range(limit))
        ds = Subset(ds, indices)
        
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["cifar10", "svhn", "mnist"])
    parser.add_argument("--batch_size", type=str, default="128") # string to easily override if needed
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit_train", type=int, default=10000) # limit to speed up training
    parser.add_argument("--limit_test", type=int, default=2000)
    args = parser.parse_args()
    
    batch_size = int(args.batch_size)
    task = args.task
    
    print(f"=== Training Expert Model for Task: {task} ===")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model and tokenizer
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    # Freeze Text Encoder & Logit Scale
    for param in model.text_model.parameters():
        param.requires_grad = False
    model.text_projection.requires_grad = False
    model.logit_scale.requires_grad = False
    
    # Only keep Vision Encoder parameters active
    for name, param in model.named_parameters():
        if "vision_model" in name or "visual_projection" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Compile text features (classification head)
    classes = DATASET_INFO[task]["classes"]
    prompt_template = DATASET_INFO[task]["prompt"]
    prompts = [prompt_template.format(c) for c in classes]
    
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        # Normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    # Get Datasets
    train_dataset = get_dataset(task, "train", limit=args.limit_train)
    test_dataset = get_dataset(task, "test" if task != "svhn" else "test", limit=args.limit_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Check zero-shot accuracy first
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.t())
            
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    zero_shot_acc = 100.0 * correct / total
    print(f"Initial Zero-Shot Accuracy on {task}: {zero_shot_acc:.2f}%")
    
    # Fine-tuning loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.t())
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)
            
        epoch_loss = train_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                image_features = model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.t())
                
                preds = logits.argmax(dim=-1)
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)
                
        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1} Results - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            
    # Save the fine-tuned vision encoder weights
    os.makedirs("checkpoints", exist_ok=True)
    save_dict = {
        "vision_model": model.vision_model.state_dict(),
        "visual_projection": model.visual_projection.state_dict(),
        "zero_shot_acc": zero_shot_acc,
        "fine_tune_acc": best_acc
    }
    save_path = f"checkpoints/{task}_expert.pt"
    torch.save(save_dict, save_path)
    print(f"Saved {task} expert model to {save_path} (Best Acc: {best_acc:.2f}%)")

if __name__ == "__main__":
    main()
