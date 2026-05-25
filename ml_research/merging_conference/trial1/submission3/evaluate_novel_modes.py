import os
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

# Disable CUDNN to ensure CPU compatibility and standard execution
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.backends.cudnn.enabled = False

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

def perform_selective_merge(model_a, model_b, model_base, mode="all"):
    """
    Applies OrthoMerge to selected layers based on the mode, and Task Arithmetic to the rest.
    
    Modes:
    - "ta": Pure Task Arithmetic (all layers)
    - "om-all": Standard OrthoMerge on all layers (dim >= 2)
    - "om-fc-only": OrthoMerge on FC weight only, TA on the rest
    - "om-conv-only": OrthoMerge on all convolutional layers, TA on FC
    - "om-conv-no-downsample": OrthoMerge on 3x3 conv layers, TA on FC and 1x1 downsample convs
    - "om-early-layers": OrthoMerge on early conv layers (conv1 + layer1 + layer2), TA on layer3, layer4, and FC
    - "om-late-layers": OrthoMerge on late conv layers (layer3 + layer4), TA on early layers and FC
    """
    model_merged = copy.deepcopy(model_base)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_0 = model_base.state_dict()
    state_m = model_merged.state_dict()
    
    for name in state_m.keys():
        val_a = state_a[name]
        val_b = state_b[name]
        val_0 = state_0[name]
        
        if not val_a.is_floating_point():
            state_m[name].copy_(val_a)
            continue
            
        # Check if this parameter is a weight matrix with dimension >= 2 to apply OrthoMerge
        is_weight = 'weight' in name and val_0.dim() >= 2
        apply_om = False
        
        if is_weight:
            if mode == "om-all":
                apply_om = True
            elif mode == "om-fc-only":
                apply_om = ("fc.weight" in name)
            elif mode == "om-conv-only":
                apply_om = ("fc.weight" not in name)
            elif mode == "om-conv-no-downsample":
                is_fc = "fc.weight" in name
                is_downsample = "downsample" in name
                is_1x1 = (val_0.dim() == 4 and val_0.shape[2] == 1 and val_0.shape[3] == 1)
                apply_om = (not is_fc) and (not is_downsample) and (not is_1x1)
            elif mode == "om-early-layers":
                is_fc = "fc.weight" in name
                is_late = "layer3" in name or "layer4" in name
                apply_om = (not is_fc) and (not is_late)
            elif mode == "om-late-layers":
                is_fc = "fc.weight" in name
                is_late = "layer3" in name or "layer4" in name
                apply_om = (not is_fc) and is_late
                
        if apply_om:
            w_merged = orthomerge_weights(val_a, val_b, val_0)
            state_m[name].copy_(w_merged)
        else:
            # Task Arithmetic / standard average
            state_m[name].copy_(0.5 * (val_a + val_b))
            
    model_merged.load_state_dict(state_m)
    return model_merged

def evaluate_model(model, dataloader, device, task_name=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    print(f"  {task_name} Accuracy: {acc:.2f}%")
    return acc

def main():
    device = torch.device("cpu") # Force CPU for local evaluation
    print(f"Running evaluation on: {device}")
    
    # Data Preparation
    print("Preparing CIFAR-10 Test Datasets...")
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_labels = testset.targets
    
    task_a_indices = [i for i, label in enumerate(test_labels) if label < 5]
    task_b_indices = [i for i, label in enumerate(test_labels) if label >= 5]
    
    test_a_loader = DataLoader(Subset(testset, task_a_indices), batch_size=128, shuffle=False)
    test_b_loader = DataLoader(Subset(testset, task_b_indices), batch_size=128, shuffle=False)
    test_full_loader = DataLoader(testset, batch_size=128, shuffle=False)
    
    # Initialize base model structure
    base_model = models.resnet18()
    base_model.fc = nn.Linear(512, 10)
    base_model.load_state_dict(torch.load("checkpoints/model_base.pt", map_location="cpu"))
    
    # Load Standard SGD Checkpoints
    print("\nLoading Standard SGD expert checkpoints...")
    model_a_std = copy.deepcopy(base_model)
    model_a_std.load_state_dict(torch.load("checkpoints/model_a_standard.pt", map_location="cpu"))
    
    model_b_std = copy.deepcopy(base_model)
    model_b_std.load_state_dict(torch.load("checkpoints/model_b_standard.pt", map_location="cpu"))
    
    # Load SAM Checkpoints
    print("Loading SAM expert checkpoints...")
    model_a_sam = copy.deepcopy(base_model)
    model_a_sam.load_state_dict(torch.load("checkpoints/model_a_sam.pt", map_location="cpu"))
    
    model_b_sam = copy.deepcopy(base_model)
    model_b_sam.load_state_dict(torch.load("checkpoints/model_b_sam.pt", map_location="cpu"))
    
    modes = [
        "ta",
        "om-all",
        "om-fc-only",
        "om-conv-only",
        "om-conv-no-downsample",
        "om-early-layers",
        "om-late-layers"
    ]
    
    results = {}
    
    for cond, (m_a, m_b) in [("Standard SGD", (model_a_std, model_b_std)), ("SAM", (model_a_sam, model_b_sam))]:
        print(f"\n================== {cond} EXPERTS ==================")
        results[cond] = {}
        for mode in modes:
            print(f"\nMerging mode: {mode}")
            m_merged = perform_selective_merge(m_a, m_b, base_model, mode=mode)
            acc_a = evaluate_model(m_merged, test_a_loader, device, "Task A")
            acc_b = evaluate_model(m_merged, test_b_loader, device, "Task B")
            acc_full = evaluate_model(m_merged, test_full_loader, device, "Full CIFAR-10")
            results[cond][mode] = (acc_a, acc_b, acc_full)
            
    # Print a beautiful final table
    print("\n" + "="*80)
    print("                      DETAILED MERGING RESULTS COMPARISON")
    print("="*80)
    for cond in results:
        print(f"\n--- {cond} Experts ---")
        print(f"{'Merging Mode':<25} | {'Task A Acc':<12} | {'Task B Acc':<12} | {'Full Acc':<12}")
        print("-"*80)
        for mode in modes:
            acc_a, acc_b, acc_full = results[cond][mode]
            print(f"{mode:<25} | {acc_a:>10.2f}% | {acc_b:>10.2f}% | {acc_full:>10.2f}%")
            
    # Write summary to file
    with open("selective_merge_results.txt", "w") as f:
        f.write("Selective Model Merging Detailed Analysis\n")
        f.write("="*50 + "\n")
        for cond in results:
            f.write(f"\n--- {cond} Experts ---\n")
            f.write(f"{'Merging Mode':<25} | {'Task A Acc':<12} | {'Task B Acc':<12} | {'Full Acc':<12}\n")
            f.write("-"*60 + "\n")
            for mode in modes:
                acc_a, acc_b, acc_full = results[cond][mode]
                f.write(f"{mode:<25} | {acc_a:>10.2f}% | {acc_b:>10.2f}% | {acc_full:>10.2f}%\n")

if __name__ == "__main__":
    main()
