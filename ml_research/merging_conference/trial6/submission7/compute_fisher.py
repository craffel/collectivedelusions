import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import get_resnet18_model

def compute_fisher_for_expert(name, dataset_cls, checkpoint_path, save_path, device):
    print(f"\n--- Computing Fisher for {name} ---")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download and load train dataset (for calibration, use indices 10000 to 10500 to avoid train overlap)
    full_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
    calib_subset = Subset(full_dataset, range(10000, 10500))
    calib_loader = DataLoader(calib_subset, batch_size=32, shuffle=False)
    
    # Load model
    model = get_resnet18_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Fisher dictionary
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    total_samples = 0
    # For each sample, we compute gradient. Standard empirical Fisher computes gradient of CE loss.
    # To compute diagonal Fisher, we can compute sample-wise gradients or estimate using batch average gradients.
    # Estimating sample-wise gradients is exact but slow if done sample-by-sample.
    # Standard practice for diagonal Fisher is empirical Fisher: we average over batches but square the gradients.
    # Let's do batch-by-batch accumulation (which is extremely standard and computationally efficient):
    for inputs, targets in calib_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # We want to accumulate squared gradients. 
        # Standard approach: run backward for each sample individually in the batch to get true sample-wise squared gradients.
        for i in range(batch_size):
            model.zero_grad()
            single_input = inputs[i:i+1]
            single_target = targets[i:i+1]
            output = model(single_input)
            loss = criterion(output, single_target)
            loss.backward()
            
            for p_name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[p_name] += (param.grad.data ** 2)
            
            total_samples += 1
            
    # Normalize by total samples
    for p_name in fisher:
        fisher[p_name] /= total_samples
        # Add a tiny epsilon to avoid division by zero
        fisher[p_name] += 1e-8
        
    torch.save(fisher, save_path)
    print(f"Computed Fisher for {total_samples} samples. Saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN to avoid initialization errors.")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    experts = [
        ("MNIST", datasets.MNIST, "checkpoints/mnist_expert.pt", "checkpoints/mnist_fisher.pt"),
        ("FashionMNIST", datasets.FashionMNIST, "checkpoints/fashion_expert.pt", "checkpoints/fashion_fisher.pt"),
        ("KMNIST", datasets.KMNIST, "checkpoints/kmnist_expert.pt", "checkpoints/kmnist_fisher.pt")
    ]
    
    for name, dataset_cls, checkpoint_path, save_path in experts:
        if not os.path.exists(save_path):
            compute_fisher_for_expert(name, dataset_cls, checkpoint_path, save_path, device)
        else:
            print(f"Fisher for {name} already exists at {save_path}, skipping computation.")

if __name__ == "__main__":
    main()
