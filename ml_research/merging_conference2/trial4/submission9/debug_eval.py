import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones, compute_calibration_data, evaluate_merged_config

def debug():
    device = torch.device("cpu")
    subsets = get_datasets()
    
    expert_names = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in expert_names]
    
    # Load experts
    experts = {}
    for name in expert_names:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = ExpertModel(backbone, head).to(device)
        
    heads = {name: experts[name].head for name in expert_names}
    merged_backbone = merge_backbones(expert_paths)
    
    # Compute calibration data
    cal_data = compute_calibration_data(expert_paths, merged_backbone, subsets)
    
    # Inspect a single layer's stats
    layer_name = 'layer1.0.bn1'
    print(f"\nStats for {layer_name}:")
    for task in expert_names:
        t_mean = cal_data['experts'][task][layer_name]['mean']
        t_std = cal_data['experts'][task][layer_name]['std']
        m_mean = cal_data['merged'][task][layer_name]['mean']
        m_std = cal_data['merged'][task][layer_name]['std']
        print(f"Task: {task.upper()}")
        print(f"  Expert mean: min={t_mean.min().item():.4f}, max={t_mean.max().item():.4f}, mean={t_mean.mean().item():.4f}")
        print(f"  Expert std:  min={t_std.min().item():.4f}, max={t_std.max().item():.4f}, mean={t_std.mean().item():.4f}")
        print(f"  Merged mean: min={m_mean.min().item():.4f}, max={m_mean.max().item():.4f}, mean={m_mean.mean().item():.4f}")
        print(f"  Merged std:  min={m_std.min().item():.4f}, max={m_std.max().item():.4f}, mean={m_std.mean().item():.4f}")

if __name__ == "__main__":
    debug()
