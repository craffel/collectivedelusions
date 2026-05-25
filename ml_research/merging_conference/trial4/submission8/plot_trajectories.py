import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

def entropy_loss(logits):
    probs = torch.softmax(logits, dim=-1)
    return - (probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()

def run_tta_track_trajectories(base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head, 
                               test_stream, layer_fisher, lr, alpha, epsilon=1e-8, optimizer_type="adam", device="cpu"):
    
    # Identify names of parameters we want to merge (excluding batchnorm buffers)
    merge_names = [
        name for name in base_params.keys()
        if base_params[name].dtype == torch.float32 
        and "running_mean" not in name 
        and "running_var" not in name 
        and "num_batches_tracked" not in name
    ]
    
    # Initialize merging coefficients
    lam1 = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in merge_names}
    lam2 = {name: torch.tensor(0.5, requires_grad=True, device=device) for name in merge_names}
    
    # Set up optimizer with layer-wise learning rates (parameter groups)
    param_groups = []
    for name in merge_names:
        fisher_val = layer_fisher.get(name, 0.0)
        # Scale learning rate inversely by Fisher sensitivity
        lr_w = lr / ((fisher_val + epsilon) ** alpha)
        
        param_groups.append({"params": [lam1[name]], "lr": lr_w})
        param_groups.append({"params": [lam2[name]], "lr": lr_w})
        
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Representative layers to track
    tracked_layers = ["conv1.weight", "layer2.0.conv1.weight", "layer4.1.conv2.weight"]
    trajectories = {name: [] for name in tracked_layers}
    
    for batch_idx, (inputs, labels, task_id) in enumerate(test_stream):
        inputs, labels = inputs.to(device), labels.to(device)
        task_id = task_id[0].item() # 0 for CIFAR-10, 1 for SVHN
        
        # Decide which head to use
        head_weight, head_bias = (expert1_head if task_id == 0 else expert2_head)
        
        # Record pre-update coefficient for this batch
        for name in tracked_layers:
            trajectories[name].append(lam1[name].item())
            
        # 1. Test-Time Adaptation step
        # Construct merged parameters
        merged_params = {}
        for name in base_params:
            if name in merge_names:
                tau1 = expert1_params[name].to(device) - base_params[name].to(device)
                tau2 = expert2_params[name].to(device) - base_params[name].to(device)
                merged_params[name] = base_params[name].to(device) + lam1[name] * tau1 + lam2[name] * tau2
            else:
                merged_params[name] = (0.5 * (expert1_params[name].to(device) + expert2_params[name].to(device))).detach()
        
        optimizer.zero_grad()
        features = torch.func.functional_call(base_encoder, merged_params, inputs)
        logits = torch.matmul(features, head_weight.t()) + head_bias
        loss = entropy_loss(logits)
        loss.backward()
        optimizer.step()
        
        # Project coefficients to [0, 1]
        with torch.no_grad():
            for name in merge_names:
                lam1[name].clamp_(0.0, 1.0)
                lam2[name].clamp_(0.0, 1.0)
                
    # Record the very last values as well
    for name in tracked_layers:
        trajectories[name].append(lam1[name].item())
        
    return trajectories

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load base encoder and experts
    print("Loading checkpoints...")
    base_params = torch.load("checkpoints/base_encoder.pt", map_location="cpu")
    expert1_state = torch.load("checkpoints/expert_cifar10.pt", map_location="cpu")
    expert2_state = torch.load("checkpoints/expert_svhn.pt", map_location="cpu")
    
    expert1_params = {k: v for k, v in expert1_state.items() if not k.startswith("fc.")}
    expert1_head = (expert1_state["fc.weight"].to(device), expert1_state["fc.bias"].to(device))
    
    expert2_params = {k: v for k, v in expert2_state.items() if not k.startswith("fc.")}
    expert2_head = (expert2_state["fc.weight"].to(device), expert2_state["fc.bias"].to(device))
    
    layer_fisher = torch.load("checkpoints/layer_fisher.pt", map_location="cpu")
    
    cifar_test = torch.load("checkpoints/cifar_test.pt", map_location="cpu")
    svhn_test = torch.load("checkpoints/svhn_test.pt", map_location="cpu")
    
    cifar_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)
    svhn_loader = DataLoader(svhn_test, batch_size=64, shuffle=False)
    
    # Create alternating test stream
    test_stream_alt = []
    cifar_iter = iter(cifar_loader)
    svhn_iter = iter(svhn_loader)
    while True:
        try:
            inputs, labels = next(cifar_iter)
            test_stream_alt.append((inputs, labels, torch.zeros(inputs.size(0), dtype=torch.long)))
        except StopIteration:
            break
            
        try:
            inputs, labels = next(svhn_iter)
            test_stream_alt.append((inputs, labels, torch.ones(inputs.size(0), dtype=torch.long)))
        except StopIteration:
            break
            
    base_encoder = resnet18()
    base_encoder.fc = nn.Identity()
    base_encoder.eval().to(device)
    
    print("Running TTA with Standard TTA...")
    standard_traj = run_tta_track_trajectories(
        base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
        test_stream_alt, layer_fisher, lr=0.1, alpha=0.0, optimizer_type="adam", device=device
    )
    
    print("Running TTA with LFWA...")
    lfwa_traj = run_tta_track_trajectories(
        base_encoder, base_params, expert1_params, expert2_params, expert1_head, expert2_head,
        test_stream_alt, layer_fisher, lr=0.001, alpha=0.5, optimizer_type="adam", device=device
    )
    
    # Plotting trajectories
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    
    colors = {
        "conv1.weight": "#e74c3c",          # Red for early layer (high Fisher)
        "layer2.0.conv1.weight": "#f39c12", # Orange for middle layer (medium Fisher)
        "layer4.1.conv2.weight": "#2ecc71"  # Green for late layer (low Fisher)
    }
    
    labels = {
        "conv1.weight": r"Early conv (conv1, $F_w \approx 0.0235$)",
        "layer2.0.conv1.weight": r"Middle conv (layer2.0.conv1, $F_w \approx 0.00036$)",
        "layer4.1.conv2.weight": r"Late conv (layer4.1.conv2, $F_w \approx 10^{-6}$)"
    }
    
    batches = np.arange(33) # 32 batches + final evaluation
    
    # Standard TTA Plot
    ax_std = axes[0]
    for name in standard_traj:
        ax_std.plot(batches, standard_traj[name], label=labels[name], color=colors[name], linewidth=2.5, marker='o', markersize=4)
    ax_std.set_title("Standard TTA (Uniform LR: " + r"$\eta = 0.1, \alpha = 0.0$)", fontsize=12, fontweight="bold")
    ax_std.set_xlabel("Adaptation Batch Index", fontsize=11)
    ax_std.set_ylabel("Merging Coefficient for CIFAR-10 (" + r"$\lambda_{1,w}$" + ")", fontsize=11)
    ax_std.set_ylim(-0.05, 1.05)
    ax_std.grid(True, linestyle="--", alpha=0.6)
    
    # LFWA Plot
    ax_lfwa = axes[1]
    for name in lfwa_traj:
        ax_lfwa.plot(batches, lfwa_traj[name], label=labels[name], color=colors[name], linewidth=2.5, marker='s', markersize=4)
    ax_lfwa.set_title("LFWA (Fisher Preconditioned: " + r"$\eta = 0.001, \alpha = 0.5$)", fontsize=12, fontweight="bold")
    ax_lfwa.set_xlabel("Adaptation Batch Index", fontsize=11)
    ax_lfwa.grid(True, linestyle="--", alpha=0.6)
    
    # Add a unified legend at the bottom
    handles, plot_labels = ax_std.get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    # Adjust layout to make room for legend at the bottom
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig("coefficient_trajectories.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("coefficient_trajectories.png", bbox_inches="tight", dpi=300)
    print("Saved coefficient trajectory plots to coefficient_trajectories.pdf and coefficient_trajectories.png")

if __name__ == "__main__":
    main()
