import torch
import torch.nn as nn
from run_experiments import create_base_resnet, ExpertModel, merge_backbones, compute_calibration_data, register_tcac_hooks, get_datasets, evaluate_merged_config

def test():
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
        
    merged_backbone = merge_backbones(expert_paths)
    cal_data = compute_calibration_data(expert_paths, merged_backbone, subsets)
    
    # Evaluate a small subset of MNIST with and without TCAC hooks
    mnist_loader = torch.utils.data.DataLoader(subsets['mnist']['test'], batch_size=16, shuffle=False)
    
    # Get first batch
    x, y = next(iter(mnist_loader))
    x, y = x.to(device), y.to(device)
    
    # Evaluate with merged model (no hooks)
    merged_backbone.eval()
    with torch.no_grad():
        out_no_hooks = merged_backbone(x)
        logits_no_hooks = experts['mnist'].head(out_no_hooks)
        preds_no_hooks = logits_no_hooks.argmax(dim=1)
        
    print(f"Predictions with NO hooks: {preds_no_hooks.tolist()}")
    print(f"True labels:               {y.tolist()}")
    
    # Register hooks
    hooks = register_tcac_hooks(merged_backbone, 'mnist', cal_data)
    
    with torch.no_grad():
        out_with_hooks = merged_backbone(x)
        logits_with_hooks = experts['mnist'].head(out_with_hooks)
        preds_with_hooks = logits_with_hooks.argmax(dim=1)
        
    print(f"Predictions WITH TCAC hooks: {preds_with_hooks.tolist()}")
    
    # Remove hooks
    for h in hooks:
        h.remove()
        
if __name__ == "__main__":
    test()
