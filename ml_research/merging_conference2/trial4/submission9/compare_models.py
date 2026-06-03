import torch
import torch.nn as nn
from run_experiments import get_datasets, create_base_resnet, merge_backbones

def main():
    device = torch.device("cpu")
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    # Method A: merged_backbone in evaluate_dbpr.py
    merged_a = merge_backbones(expert_paths)
    
    # Method B: merged_backbone in run_experiments.py
    # Let's inspect weights of some layers
    weight_a = merged_a.conv1.weight.clone()
    print(f"Weight A sum: {weight_a.sum().item():.6f}")
    
    # Let's load the expert backbones and print their weight sums
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        print(f"Expert {name.upper()} conv1 weight sum: {backbone.conv1.weight.sum().item():.6f}")

if __name__ == "__main__":
    main()
