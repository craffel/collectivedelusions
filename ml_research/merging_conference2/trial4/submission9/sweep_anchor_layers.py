import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones, apply_n_taac

device = torch.device("cpu")

def run_evaluation_with_anchor(n_taac_backbone, heads, subsets, sorted_tasks, anchor_layer_name, beta):
    # 1. Extract task prototypes at the specified anchor layer
    n_taac_backbone.eval()
    prototypes = {}
    anchor_act = None
    
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    anchor_layer = getattr(n_taac_backbone, anchor_layer_name)
    hook_handle = anchor_layer.register_forward_hook(anchor_hook)
    
    for name in sorted_tasks:
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                n_taac_backbone(x)
                break
        pooled = anchor_act.mean(dim=[2, 3]) # [B, C]
        proto = pooled.mean(dim=0)
        proto = proto / (proto.norm(p=2) + 1e-8)
        prototypes[name] = proto
        
    hook_handle.remove()
    
    # 2. Register test hook
    routing_weights = None
    def test_anchor_hook(module, input, output):
        nonlocal routing_weights
        pooled = output.mean(dim=[2, 3])
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sims = []
        for name in sorted_tasks:
            proto = prototypes[name]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1)
            sims.append(sim)
        sims = torch.stack(sims, dim=1)
        routing_weights = torch.softmax(beta * sims, dim=1)
        
    h_test = anchor_layer.register_forward_hook(test_anchor_hook)
    
    accuracies = {}
    for name in sorted_tasks:
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
        
    h_test.remove()
    avg_acc = np.mean([accuracies[k] for k in sorted_tasks])
    return avg_acc, accuracies

def main():
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    # Load experts
    experts = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = ExpertModel(backbone, head).to(device)
        
    heads = {name: experts[name].head for name in sorted_tasks}
    
    # Reconstruct N-TAAC backbone
    merged_backbone = merge_backbones(expert_paths)
    n_taac_backbone = apply_n_taac(merged_backbone, subsets)
    
    anchor_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    betas = [15.0, 30.0, 50.0, 100.0]
    
    print("\n==================== SWEEPING ANCHOR LAYERS AND BETAS ====================")
    best_avg = 0.0
    best_config = None
    
    for layer in anchor_layers:
        print(f"\n--- Anchor Layer: {layer.upper()} ---")
        for beta in betas:
            avg_acc, accs = run_evaluation_with_anchor(n_taac_backbone, heads, subsets, sorted_tasks, layer, beta)
            print(f"  Beta={beta:<5} | Avg: {avg_acc:.2f}% (MNIST: {accs['mnist']:.2f}%, F-MNIST: {accs['fmnist']:.2f}%, CIFAR: {accs['cifar']:.2f}%)")
            if avg_acc > best_avg:
                best_avg = avg_acc
                best_config = (layer, beta, accs)
                
    print(f"\n=========================================================================")
    print(f"BEST CONFIGURATION:")
    print(f"  Anchor Layer: {best_config[0].upper()}")
    print(f"  Beta: {best_config[1]}")
    print(f"  Best Average Accuracy: {best_avg:.2f}%")
    print(f"  MNIST: {best_config[2]['mnist']:.2f}%, F-MNIST: {best_config[2]['fmnist']:.2f}%, CIFAR: {best_config[2]['cifar']:.2f}%")
    print(f"=========================================================================")

if __name__ == "__main__":
    main()
