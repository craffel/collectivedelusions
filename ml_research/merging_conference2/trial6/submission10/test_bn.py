import torch
import torch.nn as nn
from torchvision import models
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    print("Creating ResNet-18 model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    # Let's inspect one BatchNorm layer's running stats
    bn = model.layer1[0].bn1
    print("Initial running mean (first 5 elements):", bn.running_mean[:5])
    print("Initial running var (first 5 elements):", bn.running_var[:5])
    print("Initial num_batches_tracked:", bn.num_batches_tracked)
    
    # Reset stats
    print("\nResetting BN stats...")
    bn.reset_running_stats()
    print("After reset running mean:", bn.running_mean[:5])
    print("After reset running var:", bn.running_var[:5])
    print("After reset num_batches_tracked:", bn.num_batches_tracked)
    
    # Run calibration
    print("\nRunning calibration forward passes...")
    model.train()
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
        
    for _ in range(5):
        inputs = torch.randn(64, 3, 32, 32).to(device)
        _ = model(inputs)
        
    print("\nAfter calibration running mean:", bn.running_mean[:5])
    print("After calibration running var:", bn.running_var[:5])
    print("After calibration num_batches_tracked:", bn.num_batches_tracked)

if __name__ == "__main__":
    test()
