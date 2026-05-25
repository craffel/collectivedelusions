import os
import time
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Disable CUDNN and force standard execution
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.backends.cudnn.enabled = False

# OrthoMerge Core Weights Function
def orthomerge_weights(w_a, w_b, w_0):
    orig_shape = w_a.shape
    
    # Flatten to 2D
    if len(orig_shape) > 2:
        d_out = orig_shape[0]
        d_in = torch.numel(w_a) // d_out
        W_a_2d = w_a.view(d_out, d_in)
        W_b_2d = w_b.view(d_out, d_in)
        W_0_2d = w_0.view(d_out, d_in)
    elif len(orig_shape) == 2:
        W_a_2d = w_a
        W_b_2d = w_b
        W_0_2d = w_0
    else:
        return 0.5 * (w_a + w_b)

    device = w_a.device
    W_0_2d = W_0_2d.to(device)
    I = torch.eye(W_a_2d.shape[0], device=device)
    
    # Task A
    M_a = W_a_2d @ W_0_2d.T
    U_a, S_a, V_a_T = torch.linalg.svd(M_a)
    R_a = U_a @ V_a_T
    
    # Task B
    M_b = W_b_2d @ W_0_2d.T
    U_b, S_b, V_b_T = torch.linalg.svd(M_b)
    R_b = U_b @ V_b_T
    
    # Inverse Cayley Transform
    R_a_stab = R_a + I + 1e-6 * I
    R_b_stab = R_b + I + 1e-6 * I
    Q_a = torch.linalg.solve(R_a_stab, R_a - I)
    Q_b = torch.linalg.solve(R_b_stab, R_b - I)
    
    # Magnitude-Corrected Merging
    Q_sum = Q_a + Q_b
    norm_sum = torch.linalg.norm(Q_sum, ord='fro')
    sum_norm = torch.linalg.norm(Q_a, ord='fro') + torch.linalg.norm(Q_b, ord='fro')
    
    c = sum_norm / (norm_sum + 1e-8)
    Q_merged = c * (0.5 * Q_sum)
    
    # Map back to Orthogonal Group via Cayley
    R_merged = torch.linalg.solve(I - Q_merged, I + Q_merged)
    
    # Residuals
    rho_a = W_a_2d - R_a @ W_0_2d
    rho_b = W_b_2d - R_b @ W_0_2d
    rho_merged = 0.5 * (rho_a + rho_b)
    
    W_final_2d = R_merged @ W_0_2d + rho_merged
    return W_final_2d.view(orig_shape)


# Selective OrthoMerge function
def perform_selective_orthomerge(model_a, model_b, model_base, mode="all"):
    """
    mode can be:
    - "all": OrthoMerge on all weights with dim >= 2 (Conv + Linear)
    - "linear_only": OrthoMerge ONLY on 2D linear weights (e.g. fc.weight)
    - "linear_and_1x1": OrthoMerge on linear weights + 1x1 conv layers
    - "none": Task Arithmetic (standard Euclidean average) on all layers
    """
    if mode == "none":
        # Task Arithmetic on all
        model_merged = copy.deepcopy(model_base)
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        state_m = model_merged.state_dict()
        for name in state_m.keys():
            val_a = state_a[name]
            val_b = state_b[name]
            if val_a.is_floating_point():
                state_m[name].copy_(0.5 * (val_a + val_b))
            else:
                state_m[name].copy_(val_a)
        model_merged.load_state_dict(state_m)
        return model_merged, 0.0

    model_merged = copy.deepcopy(model_base)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_0 = model_base.state_dict()
    state_m = model_merged.state_dict()
    
    total_layers_merged = 0
    total_residual_norm_a = 0.0
    total_residual_norm_b = 0.0
    
    for name, p_0 in model_base.named_parameters():
        if not p_0.requires_grad:
            continue
            
        p_a = state_a[name]
        p_b = state_b[name]
        
        is_weight = 'weight' in name and p_0.dim() >= 2
        apply_om = False
        
        if is_weight:
            if mode == "all":
                apply_om = True
            elif mode == "linear_only":
                apply_om = (p_0.dim() == 2) # shape [out_features, in_features]
            elif mode == "linear_and_1x1":
                is_linear = (p_0.dim() == 2)
                is_1x1 = (p_0.dim() == 4 and p_0.shape[2] == 1 and p_0.shape[3] == 1)
                apply_om = is_linear or is_1x1
                
        if apply_om:
            w_merged = orthomerge_weights(p_a, p_b, p_0)
            state_m[name].copy_(w_merged)
            
            # Log metrics
            orig_shape = p_a.shape
            d_out = orig_shape[0]
            d_in = torch.numel(p_a) // d_out
            W_a_2d = p_a.view(d_out, d_in)
            W_b_2d = p_b.view(d_out, d_in)
            W_0_2d = p_0.view(d_out, d_in).to(p_a.device)
            
            with torch.no_grad():
                M_a = W_a_2d @ W_0_2d.T
                U_a, _, V_a_T = torch.linalg.svd(M_a)
                R_a = U_a @ V_a_T
                res_a = W_a_2d - R_a @ W_0_2d
                
                M_b = W_b_2d @ W_0_2d.T
                U_b, _, V_b_T = torch.linalg.svd(M_b)
                R_b = U_b @ V_b_T
                res_b = W_b_2d - R_b @ W_0_2d
                
                total_residual_norm_a += torch.linalg.norm(res_a).item()
                total_residual_norm_b += torch.linalg.norm(res_b).item()
                total_layers_merged += 1
        else:
            w_merged = 0.5 * (p_a + p_b)
            state_m[name].copy_(w_merged)
            
    for name, buf_0 in model_base.named_buffers():
        buf_a = state_a[name]
        buf_b = state_b[name]
        if buf_0.is_floating_point():
            state_m[name].copy_(0.5 * (buf_a + buf_b))
        else:
            state_m[name].copy_(buf_a)
            
    model_merged.load_state_dict(state_m)
    avg_res_norm = (total_residual_norm_a + total_residual_norm_b) / (2 * total_layers_merged + 1e-8) if total_layers_merged > 0 else 0.0
    return model_merged, avg_res_norm


# Evaluation Function
def evaluate_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Preparation
    print("Preparing CIFAR-10 Datasets...")
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_labels = full_testset.targets
    
    task_a_test_indices = [i for i, label in enumerate(test_labels) if label < 5]
    task_b_test_indices = [i for i, label in enumerate(test_labels) if label >= 5]
    
    test_a_set = Subset(full_testset, task_a_test_indices)
    test_b_set = Subset(full_testset, task_b_test_indices)
    
    batch_size = 128
    loader_test_a = DataLoader(test_a_set, batch_size=batch_size, shuffle=False, num_workers=4)
    loader_test_b = DataLoader(test_b_set, batch_size=batch_size, shuffle=False, num_workers=4)
    loader_test_full = DataLoader(full_testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. Initialize Models
    print("Initializing Pretrained ResNet-18 Base Model...")
    base_model = models.resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model.load_state_dict(torch.load("checkpoints/model_base.pt", map_location=device))
    
    # Load Standard experts
    model_a_std = models.resnet18(weights=None)
    model_a_std.fc = nn.Linear(model_a_std.fc.in_features, 10)
    model_a_std.load_state_dict(torch.load("checkpoints/model_a_standard.pt", map_location=device))
    
    model_b_std = models.resnet18(weights=None)
    model_b_std.fc = nn.Linear(model_b_std.fc.in_features, 10)
    model_b_std.load_state_dict(torch.load("checkpoints/model_b_standard.pt", map_location=device))
    
    # Load SAM experts
    model_a_sam = models.resnet18(weights=None)
    model_a_sam.fc = nn.Linear(model_a_sam.fc.in_features, 10)
    model_a_sam.load_state_dict(torch.load("checkpoints/model_a_sam.pt", map_location=device))
    
    model_b_sam = models.resnet18(weights=None)
    model_b_sam.fc = nn.Linear(model_b_sam.fc.in_features, 10)
    model_b_sam.load_state_dict(torch.load("checkpoints/model_b_sam.pt", map_location=device))
    
    print("\nRunning Ablation Experiments...")
    modes = ["none", "linear_only", "linear_and_1x1", "all"]
    mode_names = {
        "none": "Task Arithmetic (TA)",
        "linear_only": "OM-Linear (FC Only)",
        "linear_and_1x1": "OM-Linear+1x1 Conv",
        "all": "OM-All (Conv+Linear)"
    }
    
    # Track results
    results = {
        "standard": {},
        "sam": {}
    }
    
    # Run Standard Experts
    print("\n>>> Standard SGD Experts Merging Ablations:")
    for mode in modes:
        print(f"Running mode: {mode_names[mode]}")
        model_merged, res_norm = perform_selective_orthomerge(model_a_std, model_b_std, base_model, mode)
        acc_a = evaluate_model(model_merged, loader_test_a, device)
        acc_b = evaluate_model(model_merged, loader_test_b, device)
        acc_full = evaluate_model(model_merged, loader_test_full, device)
        
        results["standard"][mode] = {
            "acc_a": acc_a,
            "acc_b": acc_b,
            "acc_full": acc_full,
            "res_norm": res_norm
        }
        print(f"  Task A: {acc_a:.2f}% | Task B: {acc_b:.2f}% | Full: {acc_full:.2f}% | Res Norm: {res_norm:.6f}")
        
    # Run SAM Experts
    print("\n>>> SAM Experts Merging Ablations:")
    for mode in modes:
        print(f"Running mode: {mode_names[mode]}")
        model_merged, res_norm = perform_selective_orthomerge(model_a_sam, model_b_sam, base_model, mode)
        acc_a = evaluate_model(model_merged, loader_test_a, device)
        acc_b = evaluate_model(model_merged, loader_test_b, device)
        acc_full = evaluate_model(model_merged, loader_test_full, device)
        
        results["sam"][mode] = {
            "acc_a": acc_a,
            "acc_b": acc_b,
            "acc_full": acc_full,
            "res_norm": res_norm
        }
        print(f"  Task A: {acc_a:.2f}% | Task B: {acc_b:.2f}% | Full: {acc_full:.2f}% | Res Norm: {res_norm:.6f}")
        
    # Plot results
    # We will save a combined plot under plots/
    print("\nSaving plots...")
    import matplotlib.pyplot as plt
    
    # Plotting Full accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    x_indices = range(len(modes))
    width = 0.35
    
    std_fulls = [results["standard"][m]["acc_full"] for m in modes]
    sam_fulls = [results["sam"][m]["acc_full"] for m in modes]
    
    rects1 = ax.bar([x - width/2 for x in x_indices], std_fulls, width, label='Standard SGD Experts', color='#1f77b4')
    rects2 = ax.bar([x + width/2 for x in x_indices], sam_fulls, width, label='SAM Experts', color='#2ca02c')
    
    ax.set_ylabel('Full CIFAR-10 Accuracy (%)')
    ax.set_title('Ablation Study: selective OrthoMerge vs. Task Arithmetic')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([mode_names[m] for m in modes])
    ax.set_ylim(60, 90)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Attach labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/selective_orthomerge_ablation.png", dpi=300)
    plt.close()
    
    # Save the text results to a file
    with open("ablation_results.txt", "w") as f:
        f.write("Ablation Study Results:\n")
        f.write("="*80 + "\n")
        for mode in modes:
            f.write(f"Mode: {mode_names[mode]}\n")
            f.write(f"  Standard Experts | Task A: {results['standard'][mode]['acc_a']:.2f}% | Task B: {results['standard'][mode]['acc_b']:.2f}% | Full: {results['standard'][mode]['acc_full']:.2f}% | Res Norm: {results['standard'][mode]['res_norm']:.6f}\n")
            f.write(f"  SAM Experts      | Task A: {results['sam'][mode]['acc_a']:.2f}% | Task B: {results['sam'][mode]['acc_b']:.2f}% | Full: {results['sam'][mode]['acc_full']:.2f}% | Res Norm: {results['sam'][mode]['res_norm']:.6f}\n")
            f.write("-" * 80 + "\n")

    print("Ablation results saved to ablation_results.txt")

if __name__ == "__main__":
    main()
