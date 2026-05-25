import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.nn.utils.stateless import functional_call
import numpy as np
import copy
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2026)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Transforms: Resize grayscale images to 32x32, convert to RGB, and normalize
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model wrapper
class ResNetExpert(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetExpert, self).__init__()
        self.resnet = resnet18()
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        if return_features:
            return out, features
        return out

def get_layer_group(param_name):
    # Group parameters into 5 layer groups
    if "conv1" in param_name or "bn1" in param_name:
        return 0
    elif "layer1" in param_name:
        return 1
    elif "layer2" in param_name:
        return 2
    elif "layer3" in param_name:
        return 3
    elif "layer4" in param_name:
        return 4
    else:
        return 0

# Apply OOD Corruption: Gaussian noise + Average Blur
def apply_corruption(inputs):
    # Add Gaussian noise
    noise = torch.randn_like(inputs) * 0.25
    corrupted = inputs + noise
    # Apply average box blur
    import torch.nn.functional as F
    corrupted = F.avg_pool2d(corrupted, kernel_size=3, stride=1, padding=1)
    return corrupted

def make_test_stream(batch_size=64):
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="./data", train=False, download=False, transform=transform)
    
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_test, batch_size=batch_size, shuffle=True)
    kmnist_loader = torch.utils.data.DataLoader(kmnist_test, batch_size=batch_size, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Block-wise sequential stream
    # Each block has 10 batches. Total 60 batches.
    blocks = [
        ("mnist", mnist_iter, 0),
        ("fashion", fashion_iter, 1),
        ("kmnist", kmnist_iter, 2),
        ("mnist", mnist_iter, 0),
        ("fashion", fashion_iter, 1),
        ("kmnist", kmnist_iter, 2)
    ]
    
    stream = []
    for name, iterator, task_id in blocks:
        for _ in range(10):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                if name == "mnist":
                    mnist_iter = iter(mnist_loader)
                    inputs, targets = next(mnist_iter)
                elif name == "fashion":
                    fashion_iter = iter(fashion_loader)
                    inputs, targets = next(fashion_iter)
                else:
                    kmnist_iter = iter(kmnist_loader)
                    inputs, targets = next(kmnist_iter)
            stream.append((inputs, targets, task_id))
    return stream

def run_experiment(method, stream, experts, original_heads, anchors=None, lr=0.01):
    print(f"\n>>> Running Experiment for Method: {method.upper()} <<<")
    set_seed(2026)
    
    # Initialize merging logits 'alphas' for 5 layer groups and 3 experts
    alphas = torch.zeros(5, 3, device=device, requires_grad=(method != "static"))
    
    # Initialize heads for each task
    heads = [copy.deepcopy(h).to(device) for h in original_heads]
    
    # Set require grads for heads
    for idx, h in enumerate(heads):
        for p in h.parameters():
            if method in ["unconstrained", "caba", "caba_no_distill", "caba_no_align"]:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
    # Define optimizer
    params_to_opt = []
    if method != "static":
        lr_alpha = 0.005 if method.startswith("caba") else lr
        params_to_opt.append({"params": [alphas], "lr": lr_alpha})
    if method in ["unconstrained", "caba", "caba_no_distill", "caba_no_align"]:
        lr_head = 0.001 if method.startswith("caba") else lr
        for h in heads:
            params_to_opt.append({"params": h.parameters(), "lr": lr_head})
            
    optimizer = optim.Adam(params_to_opt) if params_to_opt else None
    
    # Base feature extractor template
    base_model = ResNetExpert().to(device)
    base_fe = base_model.resnet
    base_fe.fc = nn.Identity()
    
    # Extract original expert feature extractor params
    expert_params = []
    for exp in experts:
        # Separate fe parameters
        sd = exp.state_dict()
        fe_sd = {k[7:]: v for k, v in sd.items() if k.startswith("resnet.") and not k.startswith("resnet.fc")}
        expert_params.append(fe_sd)
        
    fe_param_names = list(expert_params[0].keys())
    
    def merge_weights(lambdas):
        merged = {}
        for name in fe_param_names:
            g = get_layer_group(name)
            c = lambdas[g]
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                c = c.detach()
            merged[name] = (
                c[0] * expert_params[0][name] +
                c[1] * expert_params[1][name] +
                c[2] * expert_params[2][name]
            )
        return merged
    
    # Track accuracy over time
    step_accuracies = []
    running_correct = 0
    running_total = 0
    
    for step, (inputs, targets, true_task_id) in enumerate(stream):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply OOD corruptions
        corrupted_inputs = apply_corruption(inputs)
        
        # Phase 1: Determine active task
        # For sequential streams, the active task domain is provided/known sequentially
        inferred_task_id = true_task_id
            
        # Phase 2: Adaptation step (if not static)
        if method != "static" and optimizer is not None:
            optimizer.zero_grad()
            
            # Forward pass with grads enabled
            lambdas = torch.softmax(alphas, dim=1)
            merged_params = merge_weights(lambdas)
                
            features_raw = functional_call(base_fe, merged_params, corrupted_inputs)
            features = torch.flatten(features_raw, 1)
            
            logits = heads[inferred_task_id](features)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 1. Numerically stable Entropy Loss
            loss_ent = -torch.sum(probs * log_probs, dim=-1).mean()
            
            # 2. Numerically stable Consistency Loss
            augmented_inputs = corrupted_inputs + torch.randn_like(corrupted_inputs) * 0.1
            features_aug_raw = functional_call(base_fe, merged_params, augmented_inputs)
            features_aug = torch.flatten(features_aug_raw, 1)
            logits_aug = heads[inferred_task_id](features_aug)
            log_probs_aug = torch.log_softmax(logits_aug, dim=-1)
            loss_const = nn.functional.kl_div(log_probs_aug, probs.detach(), reduction="batchmean")
            
            # 3. Anchor-Alignment Loss (CAbA-Merge and variants)
            loss_align = 0.0
            loss_distill = 0.0
            if method.startswith("caba") and anchors is not None:
                task_anchors = torch.tensor(anchors[inferred_task_id], device=device) # Shape: [10, 512]
                features_norm = nn.functional.normalize(features, p=2, dim=1) # Shape: [B, 512]
                
                # KL Divergence with stable anchor-based class distributions
                sims = torch.mm(features_norm, task_anchors.t()) # Shape: [B, 10]
                temp = 0.1
                p_anc = torch.softmax(sims / temp, dim=-1)
                loss_align = nn.functional.kl_div(log_probs, p_anc.detach(), reduction="batchmean")
                
                # Self-distillation to prevent head collapse
                with torch.no_grad():
                    orig_logits = original_heads[inferred_task_id].to(device)(features)
                    orig_probs = torch.softmax(orig_logits, dim=-1)
                loss_distill = nn.functional.kl_div(log_probs, orig_probs.detach(), reduction="batchmean")
                
            # Combine losses
            gamma = 5.0 if method in ["caba", "caba_no_distill"] else 0.0
            distill_weight = 5.0 if method in ["caba", "caba_no_align"] else 0.0
            beta = 5.0
            loss = loss_ent + beta * loss_const + gamma * loss_align + distill_weight * loss_distill
            
            loss.backward()
            optimizer.step()
            
        # Phase 3: Evaluation on current batch using the inferred active head
        with torch.no_grad():
            lambdas = torch.softmax(alphas, dim=1) if method != "static" else torch.ones(5, 3, device=device) / 3.0
            merged_params = merge_weights(lambdas)
            # Forward pass to classify
            features_raw = functional_call(base_fe, merged_params, corrupted_inputs)
            features = torch.flatten(features_raw, 1)
            # Since we evaluate overall multi-task accuracy, we use the inferred head
            logits = heads[inferred_task_id](features)
            _, predicted = logits.max(1)
            
            # Check accuracy against targets
            correct = predicted.eq(targets).sum().item()
            acc = correct / targets.size(0)
            step_accuracies.append(acc)
            
            running_correct += correct
            running_total += targets.size(0)
            
            if (step + 1) % 10 == 0:
                print(f"Batch {step+1}/60 - Inferred Task: {inferred_task_id} (True: {true_task_id}) - Acc: {acc*100:.2f}%")
                if method != "static":
                    # Print lambdas of first few layer groups
                    print(f"  lambdas(g0): {lambdas[0].cpu().numpy().round(3)}, lambdas(g4): {lambdas[4].cpu().numpy().round(3)}")
                    
    total_acc = running_correct / running_total
    print(f"Overall Multi-Task Stream Accuracy: {total_acc*100:.2f}%")
    return step_accuracies, total_acc

def main():
    print("Loading experts...")
    experts = []
    original_heads = []
    anchors = []
    
    names = ["mnist", "fashion", "kmnist"]
    for name in names:
        # Load expert model
        model = ResNetExpert(num_classes=10).to(device)
        model.load_state_dict(torch.load(f"models/expert_{name}.pth", map_location=device, weights_only=True))
        model.eval()
        experts.append(model)
        original_heads.append(copy.deepcopy(model.fc))
        
        # Load anchors
        anc = np.load(f"anchors/anchors_{name}.npy")
        anchors.append(anc)
        
    print("Preparing test stream...")
    set_seed(2026)
    stream = make_test_stream(batch_size=64)
    print(f"Created stream of {len(stream)} batches.")
    
    # Run all methods (including ablations)
    results = {}
    
    # 1. Static Merging
    accs_static, overall_static = run_experiment("static", stream, experts, original_heads)
    results["static"] = (accs_static, overall_static)
    
    # 2. S2C-Merge (Frozen Heads TTA)
    accs_s2c, overall_s2c = run_experiment("s2c", stream, experts, original_heads, lr=0.01)
    results["s2c-merge"] = (accs_s2c, overall_s2c)
    
    # 3. Unconstrained TTA
    accs_unconstrained, overall_unconstrained = run_experiment("unconstrained", stream, experts, original_heads, lr=0.01)
    results["unconstrained-tta"] = (accs_unconstrained, overall_unconstrained)
    
    # 4. CAbA-Merge (Proposed)
    accs_caba, overall_caba = run_experiment("caba", stream, experts, original_heads, anchors=anchors, lr=0.01)
    results["caba-merge"] = (accs_caba, overall_caba)

    # 5. Ablation: CAbA-Merge without Distillation Loss
    accs_no_distill, overall_no_distill = run_experiment("caba_no_distill", stream, experts, original_heads, anchors=anchors, lr=0.01)
    results["caba-no-distill"] = (accs_no_distill, overall_no_distill)

    # 6. Ablation: CAbA-Merge without Anchor Alignment
    accs_no_align, overall_no_align = run_experiment("caba_no_align", stream, experts, original_heads, anchors=anchors, lr=0.01)
    results["caba-no-align"] = (accs_no_align, overall_no_align)
    
    # Generate Comparison Plot
    plt.figure(figsize=(10, 6))
    for name, (accs, overall) in results.items():
        # Compute rolling average to smooth the plot curves
        rolling_accs = [np.mean(accs[max(0, i-4):i+1]) for i in range(len(accs))]
        plt.plot(rolling_accs, label=f"{name} (Avg: {overall*100:.2f}%)", linewidth=2.5)
        
    plt.title("Test-Time Model Merging Adaptation on Corrupted Alternating Stream", fontsize=14, fontweight='bold')
    plt.xlabel("Test Stream Batch Steps", fontsize=12)
    plt.ylabel("Accuracy (5-Batch Rolling Average)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=300)
    print("\nSaved comparison plot to results_plot.png")
    
    # Save text results
    with open("experiments_summary.txt", "w") as f:
        f.write("=== TEST-TIME MODEL MERGING COMPARISON SUMMARY ===\n")
        for name, (accs, overall) in results.items():
            f.write(f"{name}: {overall*100:.2f}%\n")
            
if __name__ == "__main__":
    main()
