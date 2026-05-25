import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import os

torch.backends.cudnn.enabled = False

def compute_layer_fisher(model, calibration_dataset, device):
    model.to(device)
    model.eval()
    
    # We will accumulate squared gradients for each encoder parameter
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if not name.startswith("fc.")}
    
    loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # We perform backprop for each individual sample or per batch. 
        # For standard empirical Fisher, backprop per sample is mathematically correct, 
        # but batch-wise backprop is a highly robust and standard approximation in practice.
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not name.startswith("fc.") and param.grad is not None:
                    fisher_dict[name] += param.grad.data.clone().pow(2) * inputs.size(0)
        
        count += inputs.size(0)
        
    # Average the Fisher values over the dataset size
    for name in fisher_dict:
        fisher_dict[name] /= count
        
    # For each named parameter, calculate the average Fisher value across all its elements
    layer_sensitivity = {}
    for name, fisher in fisher_dict.items():
        layer_sensitivity[name] = fisher.mean().item()
        
    return layer_sensitivity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load Expert 1
    print("Loading Expert 1 (CIFAR-10)...")
    expert1 = resnet18()
    expert1.fc = nn.Linear(expert1.fc.in_features, 10)
    expert1.load_state_dict(torch.load("checkpoints/expert_cifar10.pt", map_location=device))
    
    # Load Expert 2
    print("Loading Expert 2 (SVHN)...")
    expert2 = resnet18()
    expert2.fc = nn.Linear(expert2.fc.in_features, 10)
    expert2.load_state_dict(torch.load("checkpoints/expert_svhn.pt", map_location=device))
    
    # Load calibration datasets
    cifar_calib = torch.load("checkpoints/cifar_calib.pt", map_location="cpu")
    svhn_calib = torch.load("checkpoints/svhn_calib.pt", map_location="cpu")
    
    print("Computing Fisher sensitivities for Expert 1...")
    sens1 = compute_layer_fisher(expert1, cifar_calib, device)
    
    print("Computing Fisher sensitivities for Expert 2...")
    sens2 = compute_layer_fisher(expert2, svhn_calib, device)
    
    # Combine sensitivities (average of both experts)
    joint_sensitivity = {}
    print("\nJoint parameter sensitivities (first 10 layers shown):")
    shown = 0
    for name in sens1:
        joint_sensitivity[name] = 0.5 * (sens1[name] + sens2[name])
        if shown < 10:
            print(f"  {name}: {joint_sensitivity[name]:.4e}")
            shown += 1
            
    # Save joint sensitivities
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(joint_sensitivity, "checkpoints/layer_fisher.pt")
    print("\nSaved joint layer-wise Fisher sensitivities to checkpoints/layer_fisher.pt")

if __name__ == "__main__":
    main()
