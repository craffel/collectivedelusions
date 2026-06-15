import os
import torch
import torch.nn as nn
import timm

def main():
    device = torch.device("cpu")
    tasks = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
    
    # 1. Load base model
    print("Loading pre-trained base model...")
    base_model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True).to(device)
    base_params = {n: p.clone().detach() for n, p in base_model.named_parameters()}
    
    # 2. Load experts and extract parameters
    expert_params = {task: {} for task in tasks}
    for task in tasks:
        path = f"experts/{task.lower()}_expert.pt"
        if os.path.exists(path):
            print(f"Loading expert {task} checkpoint from {path}...")
            model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
            model.head = nn.Linear(192, 10)
            model.load_state_dict(torch.load(path, map_location=device))
            expert_params[task] = {
                n: p.clone().detach() for n, p in model.named_parameters() if "head" not in n
            }
        else:
            print(f"Warning: expert {task} not found!")
            return

    # 3. Identify target layers
    target_layers = []
    for name, p in base_model.named_parameters():
        if "blocks" in name and "weight" in name and len(p.shape) == 2:
            target_layers.append(name)
            
    # 4. Analyze representative layers
    representative_layers = {
        "Layer 0 Attention QKV Projection": "blocks.0.attn.qkv.weight",
        "Layer 5 Attention Out Projection": "blocks.5.attn.proj.weight",
        "Layer 11 MLP Expansion (fc1)": "blocks.11.mlp.fc1.weight",
        "Layer 11 MLP Contraction (fc2)": "blocks.11.mlp.fc2.weight"
    }
    
    print("\n" + "=" * 80)
    print("SINGULAR VALUE DECAY AND ENERGY SPECTRUM ANALYSIS")
    print("=" * 80)
    
    for layer_name, param_key in representative_layers.items():
        if param_key not in target_layers:
            continue
            
        # Construct joint update matrix: M^(l) of shape (d_out, K * d_in)
        # Horizontally concatenate task vectors
        task_updates = []
        for task in tasks:
            update = expert_params[task][param_key] - base_params[param_key]
            task_updates.append(update)
            
        M = torch.cat(task_updates, dim=1) # Shape: (d_out, K * d_in)
        
        # SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        
        # Normalize singular values
        S_norm = S / S[0]
        
        # Total energy (sum of squared singular values)
        total_energy = torch.sum(S ** 2).item()
        
        print(f"\n{layer_name} (Shape: {M.shape[0]} x {M.shape[1]})")
        print("-" * 50)
        print("Top 10 Normalized Singular Values:")
        print("  " + ", ".join([f"{S_norm[i].item():.4f}" for i in range(min(10, len(S_norm)))]))
        
        print("\nCumulative Energy Captured at Different Ranks (γ):")
        for gamma in [0.1, 0.2, 0.3, 0.5]:
            r = int(gamma * M.shape[0])
            energy_r = torch.sum(S[:r] ** 2).item()
            percentage = (energy_r / total_energy) * 100.0
            print(f"  γ = {gamma:.1f} (r = {r:3d}): {percentage:6.2f}% energy")

if __name__ == "__main__":
    main()
