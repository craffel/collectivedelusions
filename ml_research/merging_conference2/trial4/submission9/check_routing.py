import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones

device = torch.device("cpu")

def check():
    subsets = get_datasets()
    expert_names = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in expert_names]
    
    # Load backbone
    backbone = merge_backbones(expert_paths)
    backbone.eval()
    
    # 1. Extract task prototypes at layer2
    prototypes = {}
    anchor_act = None
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    hook_handle = backbone.layer2.register_forward_hook(anchor_hook)
    
    for name in expert_names:
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                backbone(x)
                break
        pooled = anchor_act.mean(dim=[2, 3]) # [B, 128]
        proto = pooled.mean(dim=0)
        proto = proto / (proto.norm(p=2) + 1e-8)
        prototypes[name] = proto
        
    hook_handle.remove()
    
    # 2. Check routing on test sets for different betas
    for beta in [1.0, 5.0, 15.0, 30.0]:
        print(f"\nEvaluating Routing with Beta={beta}:")
        
        # We need a new hook to collect routing weights during test time forward passes
        routing_weights = None
        def test_anchor_hook(module, input, output):
            nonlocal routing_weights
            pooled = output.mean(dim=[2, 3]) # [B, 128]
            pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8)
            sims = []
            for name in expert_names:
                proto = prototypes[name]
                sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1)
                sims.append(sim)
            sims = torch.stack(sims, dim=1) # [B, 3]
            routing_weights = torch.softmax(beta * sims, dim=1)
            
        h_test = backbone.layer2.register_forward_hook(test_anchor_hook)
        
        for name in expert_names:
            test_loader = DataLoader(subsets[name]['test'], batch_size=128, shuffle=False)
            all_weights = []
            with torch.no_grad():
                for i, (x, _) in enumerate(test_loader):
                    backbone(x)
                    all_weights.append(routing_weights.cpu())
                    if i >= 10:  # Sample first ~1400 images
                        break
            avg_weights = torch.cat(all_weights, dim=0).mean(dim=0)
            print(f"  Test Dataset: {name.upper():<6} | Routed Probs: MNIST={avg_weights[0].item()*100:.1f}%, F-MNIST={avg_weights[1].item()*100:.1f}%, CIFAR={avg_weights[2].item()*100:.1f}%")
            
        h_test.remove()

if __name__ == "__main__":
    check()
