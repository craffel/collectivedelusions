import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import get_datasets, create_base_resnet, merge_backbones, evaluate_merged_config

def main():
    device = torch.device("cpu")
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    experts = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = backbone # Wait! In run_experiments.py, experts[name] is an ExpertModel(backbone, head)
        # Ah! Let's check how experts are constructed in run_experiments.py!
        # expert = ExpertModel(backbone, head)
        # heads = {name: experts[name].head for name in expert_names}
        # Wait, inside evaluate_dbpr.py we did the same.
        
    # Let's load experts exactly as run_experiments.py does:
    from run_experiments import ExpertModel
    experts_real = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts_real[name] = ExpertModel(backbone, head).to(device)
        
    heads_real = {name: experts_real[name].head for name in sorted_tasks}
    
    merged_backbone = merge_backbones(expert_paths)
    
    # Let's run evaluate_merged_config from run_experiments.py
    print("\nEvaluating with evaluate_merged_config from run_experiments.py:")
    res = evaluate_merged_config(merged_backbone, heads_real, subsets, "none")
    print(res)
    
    # Let's run manual evaluation from evaluate_dbpr.py
    print("\nEvaluating with manual evaluation loop:")
    correct_base = {name: 0 for name in sorted_tasks}
    total_base = {name: 0 for name in sorted_tasks}
    for name in sorted_tasks:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        merged_backbone.eval()
        for h in heads_real.values():
            h.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features = merged_backbone(x)
                logits = heads_real[name](features)
                correct_base[name] += (logits.argmax(dim=1) == y).sum().item()
                total_base[name] += y.size(0)
    for name in sorted_tasks:
        print(f"  Manual evaluation on {name.upper()}: {(correct_base[name]/total_base[name])*100:.2f}%")

if __name__ == "__main__":
    main()
