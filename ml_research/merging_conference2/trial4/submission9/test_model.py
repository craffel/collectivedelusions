import torch
import torch.nn as nn
from run_experiments import create_base_resnet, ExpertModel, run_srac_evaluation

def test_srac_shapes():
    print("Testing model structure and shapes...")
    device = torch.device("cpu")
    
    # Create fake backbone and heads
    backbone = create_base_resnet().to(device)
    heads = {
        'mnist': nn.Linear(512, 10).to(device),
        'fmnist': nn.Linear(512, 10).to(device),
        'cifar': nn.Linear(512, 10).to(device)
    }
    
    # Verify backbone layers
    print("Backbone layers:")
    for name, module in backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d) and ('layer3' in name or 'layer4' in name):
            print(f"BatchNorm layer to calibrate: {name}")
            
    # Run dummy inputs
    x = torch.randn(4, 3, 32, 32).to(device)
    features = backbone(x)
    print(f"Input shape: {x.shape}, Output feature shape: {features.shape}")
    assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"
    print("Shapes are correct!")

if __name__ == "__main__":
    test_srac_shapes()
