import torch
import torch.nn as nn
from run_experiments import get_datasets, create_base_resnet, merge_backbones

def verify():
    device = torch.device("cpu")
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    # Let's load Experts
    experts = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = backbone # Storing backbone as expert
        
    merged_backbone = merge_backbones(expert_paths)
    merged_backbone.eval()
    
    test_loader = DataLoader = torch.utils.data.DataLoader(subsets['mnist']['test'], batch_size=5, shuffle=False)
    for x, y in test_loader:
        x = x.to(device)
        with torch.no_grad():
            features = merged_backbone(x)
        print("Features mean:", features.mean().item())
        print("Features var:", features.var().item())
        print("Features shape:", features.shape)
        break

if __name__ == "__main__":
    verify()
