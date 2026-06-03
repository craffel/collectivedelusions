import torch
import numpy as np
from run_benchmark import get_resnet18_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_svd():
    print("Loading checkpoints...")
    progenitor = get_resnet18_model().to(DEVICE)
    progenitor.load_state_dict(torch.load('checkpoints/progenitor.pt', map_location=DEVICE))
    
    tasks = ['mnist', 'fmnist', 'cifar']
    experts = {}
    for task in tasks:
        model = get_resnet18_model().to(DEVICE)
        model.load_state_dict(torch.load(f'checkpoints/{task}_expert.pt', map_location=DEVICE))
        experts[task] = model
        
    # Standard Weight Averaging of the backbone
    merged_state = get_resnet18_model().state_dict()
    keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    for key in keys:
        temp = torch.zeros_like(merged_state[key], dtype=torch.float32)
        for name, m in experts.items():
            temp += m.state_dict()[key].cpu().float()
        merged_state[key].copy_(temp / 3.0)
        
    # Let's pick a representative deep layer to analyze.
    # We want a layer that is highly task-specialized and has large dimensionality.
    # layer4.1.conv2.weight is a good candidate (shape: [512, 512, 3, 3] or similar).
    target_key = 'layer4.1.conv2.weight'
    if target_key not in keys:
        # If not found, let's list some 4D convolutional layers.
        candidates = [k for k in keys if len(merged_state[k].shape) == 4]
        target_key = candidates[-1]  # Pick the last (deepest) conv layer
        
    print(f"\nAnalyzing SVD on Layer: {target_key}")
    shape = merged_state[target_key].shape
    print(f"Shape: {shape}")
    
    # We flatten the 4D tensor of shape [C_out, C_in, H, W] to 2D matrix [C_out, C_in * H * W]
    # to compute standard 2D matrix singular values.
    def get_2d_matrix(tensor):
        return tensor.view(tensor.shape[0], -1)

    prog_w = progenitor.state_dict()[target_key].to(DEVICE).float()
    
    # Compute updates
    updates = {}
    for task in tasks:
        exp_w = experts[task].state_dict()[target_key].to(DEVICE).float()
        updates[task] = exp_w - prog_w
        
    merged_w = merged_state[target_key].to(DEVICE).float()
    updates['merged'] = merged_w - prog_w
    
    # Reconstruct HNS update
    # HNS scale calculation
    tv_e = updates['mnist']  # Let's reconstruct for MNIST
    tv_m = updates['merged']
    
    dim = tuple(range(1, len(shape)))
    norm_e = torch.norm(tv_e, p=2, dim=dim, keepdim=True)
    norm_m = torch.norm(tv_m, p=2, dim=dim, keepdim=True)
    scale = norm_e / (norm_m + 1e-8)
    scale = torch.clamp(scale, min=0.1, max=10.0)
    
    hns_update = tv_m * scale.view(-1, *([1]*(len(shape)-1)))
    updates['hns_mnist'] = hns_update
    
    # Run SVD
    for name, update in updates.items():
        mat = get_2d_matrix(update)
        # Compute SVD
        U, S, V = torch.linalg.svd(mat, full_matrices=False)
        S_np = S.detach().cpu().numpy()
        
        frob_norm = torch.norm(mat, p='fro').item()
        nuc_norm = S.sum().item()
        
        print(f"\n--- Update SVD Profile for {name.upper()} ---")
        print(f"Frobenius Norm: {frob_norm:.4f}")
        print(f"Nuclear Norm (sum of singular values): {nuc_norm:.4f}")
        print(f"Top 5 Singular Values: {S_np[:5]}")
        print(f"Bottom 5 Singular Values: {S_np[-5:]}")
        print(f"Condition Number (S_max / S_min): {S_np[0] / (S_np[-1] + 1e-8):.4f}")

if __name__ == '__main__':
    analyze_svd()
