import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms.functional import gaussian_blur
from torch.func import functional_call
import copy
import numpy as np
import os

from models import SharedEncoder, ClassificationHead

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False

# ----------------- Environment Corruptions -----------------
def apply_corruption(x, corruption_type):
    # x has shape (B, 1, 28, 28), values in [-1, 1] after normalization.
    # Note: the input is normalized to [-1, 1] by transforms.Normalize((0.5,), (0.5,)).
    # We should convert it to [0, 1] to apply noise/blur/contrast, then normalize back to [-1, 1] if needed,
    # or apply it directly to [0, 1] and then normalize.
    # Let's convert to [0, 1]:
    x_unnorm = x * 0.5 + 0.5
    
    if corruption_type == "clean":
        return x
    elif corruption_type == "noise":
        # Gaussian Noise with sigma = 0.4
        noise = torch.randn_like(x_unnorm) * 0.4
        x_corrupted = torch.clamp(x_unnorm + noise, 0.0, 1.0)
    elif corruption_type == "blur":
        # Gaussian Blur with size 5x5, sigma = 2.0
        x_corrupted = gaussian_blur(x_unnorm, kernel_size=[5, 5], sigma=[2.0, 2.0])
    elif corruption_type == "contrast":
        # Contrast compression with alpha = 0.15
        x_corrupted = torch.clamp(0.5 + 0.15 * (x_unnorm - 0.5), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
        
    # Normalize back to [-1, 1]
    return (x_corrupted - 0.5) / 0.5

# ----------------- Task-Aware Augmentations (S2C / DURGP) -----------------
def augment_batch(x, task_idx):
    # Convert to [0, 1] for augmentation, then back
    x_unnorm = x * 0.5 + 0.5
    
    # 1. Random Translation (within 2 pixels)
    shift_x = torch.randint(-2, 3, (1,)).item()
    shift_y = torch.randint(-2, 3, (1,)).item()
    x_aug = torch.roll(x_unnorm, shifts=(shift_y, shift_x), dims=(2, 3))
    
    # 2. Random Horizontal Flip (FashionMNIST only, task_idx=1)
    if task_idx == 1:
        if torch.rand(1).item() > 0.5:
            x_aug = torch.flip(x_aug, dims=[3])
            
    return (x_aug - 0.5) / 0.5

# ----------------- Load Test Data & Create Streams -----------------
def get_test_streams(batch_size=64, num_batches_per_task=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load raw test sets
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    total_samples = batch_size * num_batches_per_task
    
    # Ensure we don't exceed dataset size
    mnist_subset = Subset(mnist_test, list(range(min(total_samples, len(mnist_test)))))
    fmnist_subset = Subset(fmnist_test, list(range(min(total_samples, len(fmnist_test)))))
    kmnist_subset = Subset(kmnist_test, list(range(min(total_samples, len(kmnist_test)))))
    
    mnist_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
    fmnist_loader = DataLoader(fmnist_subset, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_subset, batch_size=batch_size, shuffle=False)
    
    # Extract batches
    mnist_batches = list(mnist_loader)
    fmnist_batches = list(fmnist_loader)
    kmnist_batches = list(kmnist_loader)
    
    # We want exactly num_batches_per_task from each
    mnist_batches = mnist_batches[:num_batches_per_task]
    fmnist_batches = fmnist_batches[:num_batches_per_task]
    kmnist_batches = kmnist_batches[:num_batches_per_task]
    
    # Store with task index metadata: (images, labels, task_idx)
    mnist_meta = [(imgs, lbls, 0) for imgs, lbls in mnist_batches]
    fmnist_meta = [(imgs, lbls, 1) for imgs, lbls in fmnist_batches]
    kmnist_meta = [(imgs, lbls, 2) for imgs, lbls in kmnist_batches]
    
    # 1. Sequential Stream: MNIST -> FMNIST -> KMNIST
    seq_stream = mnist_meta + fmnist_meta + kmnist_meta
    
    # 2. Alternating Stream: interleaved batches
    alt_stream = []
    for i in range(num_batches_per_task):
        alt_stream.append(mnist_meta[i])
        alt_stream.append(fmnist_meta[i])
        alt_stream.append(kmnist_meta[i])
        
    return seq_stream, alt_stream

# ----------------- Precompute Fisher Information Matrices (for EWC) -----------------
def precompute_fisher_priors(device):
    print("Precomputing Fisher Information Matrices for classification heads...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Standard validation/training subsets to compute FIM (e.g., 200 samples)
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    fmnist_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST("./data", train=True, download=True, transform=transform)
    
    # Create tiny subsets of size N=200
    subsets = {
        0: Subset(mnist_train, list(range(200))),
        1: Subset(fmnist_train, list(range(200))),
        2: Subset(kmnist_train, list(range(200)))
    }
    
    # Load expert models and heads
    heads = {
        0: ClassificationHead().to(device),
        1: ClassificationHead().to(device),
        2: ClassificationHead().to(device)
    }
    heads[0].load_state_dict(torch.load("head_mnist.pth", map_location=device, weights_only=True))
    heads[1].load_state_dict(torch.load("head_fmnist.pth", map_location=device, weights_only=True))
    heads[2].load_state_dict(torch.load("head_kmnist.pth", map_location=device, weights_only=True))
    
    encoders = {
        0: SharedEncoder().to(device),
        1: SharedEncoder().to(device),
        2: SharedEncoder().to(device)
    }
    encoders[0].load_state_dict(torch.load("encoder_mnist.pth", map_location=device, weights_only=True))
    encoders[1].load_state_dict(torch.load("encoder_fmnist.pth", map_location=device, weights_only=True))
    encoders[2].load_state_dict(torch.load("encoder_kmnist.pth", map_location=device, weights_only=True))
    
    fisher_priors = {}
    
    for task_idx in [0, 1, 2]:
        head = heads[task_idx]
        encoder = encoders[task_idx]
        loader = DataLoader(subsets[task_idx], batch_size=32, shuffle=False)
        
        # Calculate Fisher elements for head.linear.weight and head.linear.bias
        fisher_w = torch.zeros_like(head.linear.weight)
        fisher_b = torch.zeros_like(head.linear.bias)
        
        encoder.eval()
        head.eval()
        
        total_samples = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            features = encoder(x)
            outputs = head(features)
            log_probs = F.log_softmax(outputs, dim=-1)
            
            for i in range(x.size(0)):
                target_prob = log_probs[i, y[i]]
                # Backprop to compute gradients for this sample
                head.zero_grad()
                target_prob.backward(retain_graph=True)
                
                fisher_w += head.linear.weight.grad ** 2
                fisher_b += head.linear.bias.grad ** 2
                total_samples += 1
                
        fisher_w = fisher_w / total_samples + 1e-8
        fisher_b = fisher_b / total_samples + 1e-8
        
        fisher_priors[task_idx] = {
            "weight": fisher_w.detach(),
            "bias": fisher_b.detach(),
            "init_weight": head.linear.weight.clone().detach(),
            "init_bias": head.linear.bias.clone().detach()
        }
        
    print("Fisher priors computed successfully!")
    return fisher_priors

# ----------------- Evaluation Function for a given TTA method -----------------
def run_eval(method_name, stream_type, corruption_type, stream, fisher_priors, device):
    set_seed(42) # Ensure identical initialization
    
    # 1. Load expert weights
    encoder_mnist = torch.load("encoder_mnist.pth", map_location=device, weights_only=True)
    encoder_fmnist = torch.load("encoder_fmnist.pth", map_location=device, weights_only=True)
    encoder_kmnist = torch.load("encoder_kmnist.pth", map_location=device, weights_only=True)
    
    expert_encoders = [encoder_mnist, encoder_fmnist, encoder_kmnist]
    param_names = list(encoder_mnist.keys())
    
    # Initial classification heads
    heads = {
        0: ClassificationHead().to(device),
        1: ClassificationHead().to(device),
        2: ClassificationHead().to(device)
    }
    heads[0].load_state_dict(torch.load("head_mnist.pth", map_location=device, weights_only=True))
    heads[1].load_state_dict(torch.load("head_fmnist.pth", map_location=device, weights_only=True))
    heads[2].load_state_dict(torch.load("head_kmnist.pth", map_location=device, weights_only=True))
    
    # Helper instance for stateless functional_call
    base_encoder = SharedEncoder().to(device)
    
    # Initialize merging logits: shape (num_layers, num_experts) -> (8, 3)
    num_layers = len(param_names)
    merging_logits = torch.zeros((num_layers, 3), device=device, requires_grad=True)
    
    # Task-specific merging logits for our proposed method TD-ATMM
    merging_logits_dict = None
    optimizers_dict = None
    if method_name == "td_atmm":
        merging_logits_dict = {
            0: torch.zeros((num_layers, 3), device=device, requires_grad=True),
            1: torch.zeros((num_layers, 3), device=device, requires_grad=True),
            2: torch.zeros((num_layers, 3), device=device, requires_grad=True)
        }
        optimizers_dict = {
            0: torch.optim.Adam([merging_logits_dict[0]], lr=0.01),
            1: torch.optim.Adam([merging_logits_dict[1]], lr=0.01),
            2: torch.optim.Adam([merging_logits_dict[2]], lr=0.01)
        }
    
    # Define optimizer for logits and active classification heads
    # standard hyperparameters from benchmark paper: lr_logits = 0.005, lr_heads = 0.05
    # (except for Static Merged which has no adaptation)
    
    optimizer = None
    if method_name != "static" and method_name != "td_atmm":
        params_to_opt = []
        if method_name in ["standard", "ewc", "durgp"]:
            # Optimizes both logits and heads
            params_to_opt.append({"params": [merging_logits], "lr": 0.005})
            for k in [0, 1, 2]:
                params_to_opt.append({"params": heads[k].parameters(), "lr": 0.05})
        elif method_name == "s2c":
            # S2C-Merge freezes heads and only optimizes merging coefficients
            params_to_opt.append({"params": [merging_logits], "lr": 0.005})
            
        optimizer = torch.optim.Adam(params_to_opt)
        
    total_correct = 0
    total_samples = 0
    
    # To track class accuracies or print progress
    for step, (x, y, task_idx) in enumerate(stream):
        x, y = x.to(device), y.to(device)
        
        # Apply environmental domain corruption
        x_corrupted = apply_corruption(x, corruption_type)
        
        # Active head
        head = heads[task_idx]
        
        # --- Merging & Forward Pass ---
        # 1. Softmax on merging logits (using task-specific logits if td_atmm)
        if method_name == "td_atmm":
            weights = torch.softmax(merging_logits_dict[task_idx], dim=1)
        else:
            weights = torch.softmax(merging_logits, dim=1) # shape (8, 3)
        
        # 2. Reconstruct merged encoder parameters
        merged_params = {}
        for l_idx, name in enumerate(param_names):
            merged_params[name] = (
                weights[l_idx, 0] * expert_encoders[0][name].to(device) +
                weights[l_idx, 1] * expert_encoders[1][name].to(device) +
                weights[l_idx, 2] * expert_encoders[2][name].to(device)
            )
            
        # 3. Stateless functional_call for encoder forward
        base_encoder.train() if method_name != "static" else base_encoder.eval()
        head.train() if method_name != "static" else head.eval()
        
        features = functional_call(base_encoder, merged_params, x_corrupted)
        outputs = head(features)
        probs = F.softmax(outputs, dim=-1)
        
        # Record accuracy for this step BEFORE adaptation (standard TTA evaluation protocol)
        with torch.no_grad():
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += y.size(0)
            
        # --- Adaptation (Backward Pass) ---
        if method_name != "static":
            if method_name == "td_atmm":
                optimizers_dict[task_idx].zero_grad()
            else:
                optimizer.zero_grad()
            
            # Loss computation
            loss = 0.0
            
            # A. Entropy Minimization
            # Lent = -E[sum(p log p)]
            ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
            loss += ent_loss
            
            # B. Augmentation Consistency (for S2C, proposed DURGP, and td_atmm)
            if method_name in ["s2c", "durgp", "td_atmm"]:
                x_aug = augment_batch(x_corrupted, task_idx)
                features_aug = functional_call(base_encoder, merged_params, x_aug)
                outputs_aug = head(features_aug)
                probs_aug = F.softmax(outputs_aug, dim=-1)
                
                # KL Divergence consistency
                # D_KL(P_aug || sg(P))
                kl_loss = F.kl_div(torch.log(probs_aug + 1e-12), probs.detach(), reduction="batchmean")
                loss += 1.0 * kl_loss
                
            # C. Regularization for Classification Heads
            if method_name == "ewc":
                # EWC style penalty: 1/2 * sum(F_p * (theta - theta_init)^2)
                # scale penalty by gamma = 100.0 (from EWC-TTA paper)
                ewc_penalty = 0.0
                prior = fisher_priors[task_idx]
                ewc_penalty += torch.sum(prior["weight"] * (head.linear.weight - prior["init_weight"]) ** 2)
                ewc_penalty += torch.sum(prior["bias"] * (head.linear.bias - prior["init_bias"]) ** 2)
                loss += 100.0 * 0.5 * ewc_penalty
                
            elif method_name == "durgp":
                # Proposed: Dynamic Uncertainty-Weighted Relative Geometry Preservation
                # 1. Normalise head rows
                w_norm = F.normalize(head.linear.weight, p=2, dim=1) # shape (10, 128)
                w0_norm = F.normalize(fisher_priors[task_idx]["init_weight"], p=2, dim=1)
                
                # 2. Gram matrices (class similarities)
                G = torch.mm(w_norm, w_norm.t()) # shape (10, 10)
                G0 = torch.mm(w0_norm, w0_norm.t())
                
                # 3. Class-wise confidence weights q_c = mean predicted prob in current batch
                q = torch.mean(probs, dim=0) # shape (10,)
                
                # Outer product to create pairwise confidence weighting matrix Q
                Q = torch.outer(q, q) # shape (10, 10)
                
                # Element-wise weighted L2 loss on Gram matrix difference
                durgp_loss = torch.sum(Q * (G - G0) ** 2) / 100.0
                
                # Dynamic weight gamma_k = gamma * max(0, 1 - H_k / log(10))
                # with base scaling gamma = 0.1
                H = ent_loss.item()
                gamma_k = 0.1 * max(0, 1.0 - H / np.log(10.0))
                
                loss += gamma_k * durgp_loss
                
            elif method_name == "td_atmm":
                # L2 penalty on task-specific logits to anchor them to the robust equal-weight split
                loss += 0.05 * torch.sum(merging_logits_dict[task_idx] ** 2)
                
            # Backward and Optimize
            loss.backward()
            if method_name == "td_atmm":
                optimizers_dict[task_idx].step()
            else:
                optimizer.step()
            
    avg_accuracy = 100.0 * total_correct / total_samples
    return avg_accuracy

# ----------------- Main Experiment Harness -----------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # 1. Get streams
    seq_stream, alt_stream = get_test_streams(batch_size=64, num_batches_per_task=50)
    
    # 2. Compute Fisher priors
    fisher_priors = precompute_fisher_priors(device)
    
    methods = ["static", "standard", "s2c", "ewc", "durgp", "td_atmm"]
    corruptions = ["clean", "noise", "blur", "contrast"]
    streams = {
        "sequential": seq_stream,
        "alternating": alt_stream
    }
    
    results = {}
    
    for s_name, stream in streams.items():
        results[s_name] = {}
        print(f"\n==========================================")
        print(f"Evaluating on {s_name.upper()} Stream")
        print(f"==========================================")
        
        for m_name in methods:
            results[s_name][m_name] = {}
            print(f"\n--- Method: {m_name.upper()} ---")
            
            for c_name in corruptions:
                acc = run_eval(m_name, s_name, c_name, stream, fisher_priors, device)
                results[s_name][m_name][c_name] = acc
                print(f"Corruption: {c_name:8s} | Accuracy: {acc:.2f}%")
                
    # 3. Print a beautiful table of all results
    print("\n\n" + "="*80)
    print("FINAL TEST-TIME ADAPTATION RESULTS (Multi-task Average Accuracy %)")
    print("="*80)
    
    for s_name in ["sequential", "alternating"]:
        print(f"\nStream Type: {s_name.upper()}")
        print("-"*80)
        print(f"{'Method':12s} | {'Clean':8s} | {'Noise':8s} | {'Blur':8s} | {'Contrast':8s} | {'Average':8s}")
        print("-"*80)
        for m_name in methods:
            row_accs = [results[s_name][m_name][c] for c in corruptions]
            avg_acc = np.mean(row_accs)
            print(f"{m_name.upper():12s} | {row_accs[0]:7.2f}% | {row_accs[1]:7.2f}% | {row_accs[2]:7.2f}% | {row_accs[3]:7.2f}% | {avg_acc:7.2f}%")
        print("-"*80)

if __name__ == "__main__":
    main()
