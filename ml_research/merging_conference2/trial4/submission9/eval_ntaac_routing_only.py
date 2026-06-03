import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones, apply_n_taac, eval_model_simple

device = torch.device("cpu")

def main():
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
    
    # Reconstruct N-TAAC backbone
    merged_backbone = merge_backbones(expert_paths)
    n_taac_backbone = apply_n_taac(merged_backbone, subsets)
    n_taac_backbone.eval()
    
    # 1. Extract task prototypes at layer2
    prototypes = {}
    anchor_act = None
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    hook_handle = n_taac_backbone.layer2.register_forward_hook(anchor_hook)
    
    for name in expert_names:
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                n_taac_backbone(x)
                break
        pooled = anchor_act.mean(dim=[2, 3]) # [B, 128]
        proto = pooled.mean(dim=0)
        proto = proto / (proto.norm(p=2) + 1e-8)
        prototypes[name] = proto
        
    hook_handle.remove()
    
    # Evaluate for different betas
    for beta in [5.0, 15.0, 30.0, 50.0]:
        print(f"\nEvaluating Head Routing ONLY with Beta={beta}:")
        
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
            
        h_test = n_taac_backbone.layer2.register_forward_hook(test_anchor_hook)
        
        accuracies = {}
        for name in expert_names:
            test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    features = n_taac_backbone(x)
                    
                    logits_mnist = heads['mnist'](features)
                    logits_fmnist = heads['fmnist'](features)
                    logits_cifar = heads['cifar'](features)
                    logits_all = torch.stack([logits_mnist, logits_fmnist, logits_cifar], dim=1)
                    
                    logits = torch.sum(routing_weights.unsqueeze(-1) * logits_all, dim=1)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            accuracies[name] = (correct / total) * 100.0
            
        avg_acc = (accuracies['mnist'] + accuracies['fmnist'] + accuracies['cifar']) / 3.0
        print(f"  Avg Acc: {avg_acc:.2f}% (MNIST: {accuracies['mnist']:.2f}%, F-MNIST: {accuracies['fmnist']:.2f}%, CIFAR: {accuracies['cifar']:.2f}%)")
        h_test.remove()

if __name__ == "__main__":
    main()
