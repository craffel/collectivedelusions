import torch
import torchvision.models as models
import torch.nn as nn
import copy

def get_layer_wise_procrustes(checkpoint_path, base_model):
    device = torch.device("cpu")
    model = copy.deepcopy(base_model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    eps_eps = 1e-8
    layer_norms = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            base_module = dict(base_model.named_modules())[name]
            
            W = module.weight
            W0 = base_module.weight
            
            C_out = W.shape[0]
            W_flat = W.view(C_out, -1)
            W0_flat = W0.view(C_out, -1)
            
            W_norm = W_flat / (torch.norm(W_flat, p=2, dim=1, keepdim=True) + eps_eps)
            W0_norm = W0_flat / (torch.norm(W0_flat, p=2, dim=1, keepdim=True) + eps_eps)
            
            U, _, Vt = torch.linalg.svd(torch.mm(W_norm, W0_norm.t()))
            R = torch.mm(U, Vt)
            
            residual = W_norm - torch.mm(R, W0_norm)
            norm = torch.norm(residual, p='fro').item()
            layer_norms[name] = norm
            
    return layer_norms

def main():
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    
    configs = {
        "SGD": "checkpoints/expert_A_sgd_beta_0.0.pth",
        "SAM": "checkpoints/expert_A_sam_beta_0.0.pth",
        "SPOR (0.2)": "checkpoints/expert_A_spor_beta_0.2.pth",
        "FG-SPOR-Inv (0.2)": "checkpoints/expert_A_fg_spor_inverse_beta_0.2.pth"
    }
    
    results = {}
    for name, path in configs.items():
        try:
            results[name] = get_layer_wise_procrustes(path, base_model)
        except Exception as e:
            print(f"Failed for {name}: {e}")
            
    # Print layer-wise table
    layers = list(results["SGD"].keys())
    print(f"{"Layer Name":<25} | {"SGD":<10} | {"SAM":<10} | {"SPOR (0.2)":<10} | {"FG-SPOR-Inv":<10}")
    print("-" * 75)
    for layer in layers:
        row = f"{layer:<25}"
        for name in configs.keys():
            val = results[name][layer]
            row += f" | {val:<10.6f}"
        print(row)

if __name__ == "__main__":
    main()
