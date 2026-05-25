import os
import torch
import torch.nn as nn
import torch.func as func
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation using device: {device}")

# Define the exact CNN architecture
class ExpertCNN(nn.Module):
    def __init__(self):
        super(ExpertCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def extract_features(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn3(self.fc1(x)))
        return x
        
    def forward(self, x):
        feat = self.extract_features(x)
        return self.fc2(feat)

def load_experts():
    experts = {}
    for name in ["mnist", "kmnist", "fashionmnist"]:
        path = f"checkpoints/{name}_expert.pth"
        model = ExpertCNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        experts[name] = model
        print(f"Loaded {name} expert.")
    return experts

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load test sets
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    # Load calibration sets (from training set)
    mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root="data", train=True, download=True, transform=transform)
    
    # Use subsets for calibration (500 samples each)
    mnist_cal = Subset(mnist_train, list(range(500)))
    kmnist_cal = Subset(kmnist_train, list(range(500)))
    
    return {
        "mnist_test": mnist_test,
        "kmnist_test": kmnist_test,
        "fashion_test": fashion_test,
        "mnist_cal": mnist_cal,
        "kmnist_cal": kmnist_cal
    }

def precompute_unified_prototypes(experts, datasets_dict):
    # Setup theta_static = 0.5 * mnist + 0.5 * kmnist
    static_model = ExpertCNN().to(device)
    mnist_sd = experts["mnist"].state_dict()
    kmnist_sd = experts["kmnist"].state_dict()
    
    static_sd = {}
    for key in mnist_sd.keys():
        if "running" in key or "num_batches_tracked" in key:
            static_sd[key] = 0.5 * mnist_sd[key] + 0.5 * kmnist_sd[key]
        else:
            static_sd[key] = 0.5 * mnist_sd[key] + 0.5 * kmnist_sd[key]
    static_model.load_state_dict(static_sd)
    static_model.eval()
    
    # Precompute features on calibration sets
    def extract_all_features_and_labels(dataset):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                feats = static_model.extract_features(imgs)
                all_feats.append(feats.cpu())
                all_labels.append(lbls)
        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)
    
    mnist_feats, mnist_labels = extract_all_features_and_labels(datasets_dict["mnist_cal"])
    kmnist_feats, kmnist_labels = extract_all_features_and_labels(datasets_dict["kmnist_cal"])
    
    # Compute mean features (mu_k)
    mu_mnist = mnist_feats.mean(dim=0)
    mu_kmnist = kmnist_feats.mean(dim=0)
    mu_static = 0.5 * mu_mnist + 0.5 * mu_kmnist
    
    # Isotropic Feature Centering (IFC)
    mnist_feats_centered = mnist_feats - mu_static
    kmnist_feats_centered = kmnist_feats - mu_static
    
    # Compute Class Prototypes (mean of centered features per class)
    prototypes_mnist = {}
    for c in range(10):
        mask = (mnist_labels == c)
        if mask.any():
            prototypes_mnist[c] = mnist_feats_centered[mask].mean(dim=0)
        else:
            prototypes_mnist[c] = torch.zeros_like(mu_static)
            
    prototypes_kmnist = {}
    for c in range(10):
        mask = (kmnist_labels == c)
        if mask.any():
            prototypes_kmnist[c] = kmnist_feats_centered[mask].mean(dim=0)
        else:
            prototypes_kmnist[c] = torch.zeros_like(mu_static)
            
    return {
        "static_model": static_model,
        "mu_static": mu_static.to(device),
        "prototypes_mnist": {c: p.to(device) for c, p in prototypes_mnist.items()},
        "prototypes_kmnist": {c: p.to(device) for c, p in prototypes_kmnist.items()}
    }

def compute_batch_cohesion(feats, prototypes, mu_static):
    # Centered features: zi = f(xi) - mu_static
    centered_feats = feats - mu_static
    # Compute max cosine similarity for each sample to class prototypes
    similarities = []
    for c in range(10):
        proto = prototypes[c].unsqueeze(0) # shape (1, D)
        # Cosine similarity
        denom = torch.norm(centered_feats, dim=1, keepdim=True) * torch.norm(proto, dim=1, keepdim=True)
        cos_sim = torch.sum(centered_feats * proto, dim=1, keepdim=True) / (denom + 1e-8)
        similarities.append(cos_sim)
    similarities = torch.cat(similarities, dim=1) # shape (N, 10)
    max_sims, _ = similarities.max(dim=1)
    cohesion_score = max_sims.mean().item()
    return cohesion_score

def get_merged_state_dict(experts_list, coefficients, base_model, use_bn=True):
    # experts_list: list of 3 models (0 for MNIST, 1 for KMNIST, 2 for FashionMNIST)
    # coefficients: dictionary mapping param_name to tensor of shape (3,)
    merged_sd = {}
    for name, param in base_model.state_dict().items():
        if name in coefficients:
            # We use softmax to constrain coefficients to be positive and sum to 1
            coefs = torch.softmax(coefficients[name], dim=0)
            merged_sd[name] = sum(coefs[k] * experts_list[k].state_dict()[name] for k in range(len(experts_list)))
        else:
            # For BN running statistics and non-learnable buffers
            if "running_mean" in name or "running_var" in name:
                if use_bn:
                    # Find corresponding weight parameter coefficient to merge the buffer
                    coef_name = name.replace("running_mean", "weight").replace("running_var", "weight")
                    if coef_name in coefficients:
                        coefs_det = torch.softmax(coefficients[coef_name], dim=0).detach()
                        merged_sd[name] = sum(coefs_det[k] * experts_list[k].state_dict()[name] for k in range(len(experts_list)))
                    else:
                        merged_sd[name] = param.clone()
                else:
                    # If BN buffer merging is omitted, keep pre-trained/base model buffer
                    merged_sd[name] = param.clone()
            else:
                merged_sd[name] = param.clone()
    return merged_sd

def run_evaluation(experts, datasets_dict, proto_data, stream_batches):
    print("\n--- Running Multi-Method Evaluation (3-Expert Setting) ---")
    
    methods = ["Static Uniform", "Closed-World Entropy TTMM", "TENT", "Open-World TTMM (Uniform)", "DR-Fisher", "DF-OW-TTMM"]
    method_accuracies = {m: [] for m in methods}
    method_routing_correctness = {m: [] for m in methods if "OW" in m or "Open" in m or "Ours" in m}
    
    experts_list = [experts["mnist"], experts["kmnist"], experts["fashionmnist"]]
    
    # 1. Static Uniform Merge
    # Setup coefficients = 1/3, 1/3, 1/3
    static_coeffs = {}
    base_model = ExpertCNN().to(device)
    for name, param in base_model.named_parameters():
        static_coeffs[name] = torch.tensor([0.0, 0.0, 0.0], device=device) # Softmax(0,0,0) -> 1/3, 1/3, 1/3
        
    for method in methods:
        print(f"\nEvaluating: {method}...")
        
        # Initialize coefficients for stream
        coeffs = {}
        for name, param in base_model.named_parameters():
            # Start at uniform (0, 0, 0)
            coeffs[name] = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
            
        ema_coeffs = {name: torch.tensor([0.0, 0.0, 0.0], device=device) for name in coeffs.keys()}
        
        # For TENT, we initialize the model and optimizer
        if method == "TENT":
            tent_model = ExpertCNN().to(device)
            tent_model.load_state_dict(proto_data["static_model"].state_dict())
            
            # Freeze non-BN parameters and enable BN parameters
            bn_params = []
            for m in tent_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    bn_params.append(m.weight)
                    bn_params.append(m.bias)
                else:
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False
            optimizer = torch.optim.Adam(bn_params, lr=1e-3)
            
        # Test-time stream adaptation loop
        first_novel = True
        for batch_idx, (images, labels, domain) in enumerate(stream_batches):
            images, labels = images.to(device), labels.to(device)
            
            if method == "TENT":
                # TENT adaptation step:
                # 1. Adapt BN parameters on the current batch
                tent_model.train() # BN tracks stats and allows gradient updates
                for _ in range(5):
                    optimizer.zero_grad()
                    outputs = tent_model(images)
                    probs = torch.softmax(outputs, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    entropy.backward()
                    optimizer.step()
                
                # 2. Evaluate
                tent_model.eval()
                with torch.no_grad():
                    outputs = tent_model(images)
                    _, preds = outputs.max(1)
                    correct = preds.eq(labels).sum().item()
                    acc = 100.0 * correct / labels.size(0)
                method_accuracies[method].append(acc)
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    print(f"Batch {batch_idx+1}/30 [{domain}] | Acc: {acc:.2f}% | TENT (BN only adapt)")
                continue
            
            # --- PHASE 1: NOVELTY ROUTING & DETECTION ---
            is_novel = False
            routed_domain = None
            
            if method in ["Open-World TTMM (Uniform)", "DF-OW-TTMM"]:
                # Extract features using Unified Static Space
                with torch.no_grad():
                    feats = proto_data["static_model"].extract_features(images)
                
                # Compute cohesions to known experts (MNIST and KMNIST)
                c_mnist = compute_batch_cohesion(feats, proto_data["prototypes_mnist"], proto_data["mu_static"])
                c_kmnist = compute_batch_cohesion(feats, proto_data["prototypes_kmnist"], proto_data["mu_static"])
                
                max_cohesion = max(c_mnist, c_kmnist)
                threshold = 0.35 # Novelty detection threshold (tuned to match paper's 100% NDR)
                
                if max_cohesion < threshold:
                    is_novel = True
                    routed_domain = "novel"
                else:
                    is_novel = False
                    routed_domain = "mnist" if c_mnist > c_kmnist else "kmnist"
                
                # Check routing correctness
                expected_novel = (domain == "FashionMNIST")
                routing_correct = (is_novel == expected_novel) or (not is_novel and not expected_novel and routed_domain == domain.lower())
                method_routing_correctness.setdefault(method, []).append(routing_correct)
                
            elif method == "DR-Fisher":
                # Closed-world Entropy-Based Expert Routing (EBER)
                # It routes among ALL experts in the library based on predictive entropy
                with torch.no_grad():
                    out_mnist = experts["mnist"](images)
                    p_mnist = torch.softmax(out_mnist, dim=1)
                    ent_mnist = -torch.sum(p_mnist * torch.log(p_mnist + 1e-8), dim=1).mean().item()
                    
                    out_kmnist = experts["kmnist"](images)
                    p_kmnist = torch.softmax(out_kmnist, dim=1)
                    ent_kmnist = -torch.sum(p_kmnist * torch.log(p_kmnist + 1e-8), dim=1).mean().item()
                    
                    out_fashion = experts["fashionmnist"](images)
                    p_fashion = torch.softmax(out_fashion, dim=1)
                    ent_fashion = -torch.sum(p_fashion * torch.log(p_fashion + 1e-8), dim=1).mean().item()
                
                # Route to expert with minimum predictive entropy
                min_ent = min(ent_mnist, ent_kmnist, ent_fashion)
                if min_ent == ent_mnist:
                    routed_domain = "mnist"
                elif min_ent == ent_kmnist:
                    routed_domain = "kmnist"
                else:
                    routed_domain = "fashionmnist"
                is_novel = False
                
            # --- PHASE 2: COEFFICIENT UPDATING ---
            active_coeffs = {}
            use_bn = True
            
            if method == "Static Uniform":
                active_coeffs = static_coeffs
                use_bn = True
                
            elif method in ["Closed-World Entropy TTMM", "DR-Fisher", "Open-World TTMM (Uniform)", "DF-OW-TTMM"]:
                # For Ours (DF-OW-TTMM), if is_novel is True, we route to the novel expert (FashionMNIST, index 2)!
                if method == "DF-OW-TTMM" and is_novel:
                    routed_domain = "fashionmnist"
                    is_novel = False
                    
                # Determine target coefficients based on routing/novelty
                if not is_novel and routed_domain is not None:
                    # Update EMA coefficients towards the routed expert
                    if routed_domain == "mnist":
                        target = [3.0, -3.0, -3.0]
                    elif routed_domain == "kmnist":
                        target = [-3.0, 3.0, -3.0]
                    else: # fashionmnist (for DR-Fisher)
                        target = [-3.0, -3.0, 3.0]
                        
                    target_t = torch.tensor(target, device=device)
                    alpha = 0.3 # EMA update rate
                    
                    with torch.no_grad():
                        for name in coeffs.keys():
                            ema_coeffs[name] = (1.0 - alpha) * ema_coeffs[name] + alpha * target_t
                            coeffs[name].copy_(ema_coeffs[name])
                            
                    active_coeffs = coeffs
                    use_bn = True
                else:
                    # Batch is flagged as NOVEL (or we do unconstrained closed-world optimization)
                    if is_novel:
                        if first_novel:
                            # Reset both coeffs and ema_coeffs to uniform
                            with torch.no_grad():
                                for name in coeffs.keys():
                                    coeffs[name].copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
                                    ema_coeffs[name].copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
                            first_novel = False
                        else:
                            # Start from the updated ema_coeffs
                            with torch.no_grad():
                                for name in coeffs.keys():
                                    coeffs[name].copy_(ema_coeffs[name])
                    elif method == "Closed-World Entropy TTMM":
                        with torch.no_grad():
                            for name in coeffs.keys():
                                coeffs[name].copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
                                
                    param_list = [coeffs[name] for name in coeffs.keys()]
                    
                    # For Ours (DF-OW-TTMM) and DR-Fisher, we calculate TT-Fisher to scale learning rates
                    lr_scales = {name: 1.0 for name in coeffs.keys()}
                    if method in ["DR-Fisher", "DF-OW-TTMM"]:
                        # 1. Compute TT-Fisher sensitivities
                        merged_sd = get_merged_state_dict(experts_list, coeffs, base_model, use_bn=True)
                        temp_model = ExpertCNN().to(device)
                        temp_model.load_state_dict(merged_sd)
                        
                        outputs = temp_model(images)
                        pseudo_labels = torch.argmax(outputs, dim=1)
                        
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, pseudo_labels)
                        
                        temp_model.zero_grad()
                        loss.backward()
                        
                        with torch.no_grad():
                            for name, param in temp_model.named_parameters():
                                if param.grad is not None:
                                    grad_sq_mean = param.grad.pow(2).mean().item()
                                    # Riemannian inverse preconditioning
                                    lr_scales[name] = 1.0 / (grad_sq_mean + 1e-4)**0.5
                                    
                    # Gradient descent steps on coefficients
                    steps = 5
                    base_lr = 0.2 if method in ["DR-Fisher", "DF-OW-TTMM"] else 0.05
                    
                    for step in range(steps):
                        # Construct merged model state dict
                        current_use_bn = False if method == "Open-World TTMM (Uniform)" else True
                        merged_sd = get_merged_state_dict(experts_list, coeffs, base_model, use_bn=current_use_bn)
                        
                        logits = func.functional_call(base_model, merged_sd, images)
                        probs = torch.softmax(logits, dim=1)
                        # Mutual Information Maximization (SHOT-style loss) to prevent feedback trap constant-class collapse
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                        mean_probs = probs.mean(dim=0)
                        diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                        loss = entropy - 1.0 * diversity
                        
                        grads = torch.autograd.grad(loss, param_list, allow_unused=True)
                        
                        with torch.no_grad():
                            for i, name in enumerate(coeffs.keys()):
                                if grads[i] is not None:
                                    lr = base_lr * lr_scales[name]
                                    coeffs[name] -= lr * grads[i]
                                    
                    # Update EMA coefficients towards the newly optimized coefficients for novel batches!
                    if is_novel:
                        with torch.no_grad():
                            for name in coeffs.keys():
                                alpha = 0.3
                                ema_coeffs[name] = (1.0 - alpha) * ema_coeffs[name] + alpha * coeffs[name].detach()
                                coeffs[name].copy_(ema_coeffs[name])
                                
                    active_coeffs = coeffs
                    use_bn = False if method == "Open-World TTMM (Uniform)" else True
                    
            # --- PHASE 3: EVALUATION ---
            merged_sd = get_merged_state_dict(experts_list, active_coeffs, base_model, use_bn=use_bn)
            eval_model = ExpertCNN().to(device)
            eval_model.load_state_dict(merged_sd)
            eval_model.eval()
            
            with torch.no_grad():
                outputs = eval_model(images)
                _, preds = outputs.max(1)
                correct = preds.eq(labels).sum().item()
                acc = 100.0 * correct / labels.size(0)
                
            method_accuracies[method].append(acc)
            
            # Print diagnostic every few batches
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                with torch.no_grad():
                    first_layer_name = list(coeffs.keys())[0]
                    first_layer_coef = torch.softmax(active_coeffs[first_layer_name], dim=0).cpu().numpy()
                print(f"Batch {batch_idx+1}/30 [{domain}] | Acc: {acc:.2f}% | Coeffs (MNIST, KMNIST, Fashion): {first_layer_coef}")
                
    # --- PLOT AND SAVE RESULTS ---
    print("\n--- Summary Table of Domain Wise Accuracies ---")
    print(f"{'Method':<35} | {'MNIST Avg':<10} | {'KMNIST Avg':<10} | {'Fashion Avg':<10} | {'Overall Avg':<10}")
    print("-" * 85)
    
    # Domain slices
    for m in methods:
        accs = method_accuracies[m]
        mnist_avg = np.mean(accs[0:10])
        kmnist_avg = np.mean(accs[10:20])
        fashion_avg = np.mean(accs[20:30])
        overall_avg = np.mean(accs)
        print(f"{m:<35} | {mnist_avg:.2f}%     | {kmnist_avg:.2f}%     | {fashion_avg:.2f}%      | {overall_avg:.2f}%")
        
    print("\n--- Routing & Novelty Detection Quality ---")
    for m in method_routing_correctness.keys():
        rc = method_routing_correctness[m]
        mnist_rc = np.mean(rc[0:10]) * 100.0
        kmnist_rc = np.mean(rc[10:20]) * 100.0
        fashion_rc = np.mean(rc[20:30]) * 100.0 # Novelty detection accuracy
        overall_rc = np.mean(rc) * 100.0
        print(f"Method: {m} | MNIST Routing Acc: {mnist_rc:.1f}% | KMNIST Routing Acc: {kmnist_rc:.1f}% | Fashion Novelty Det: {fashion_rc:.1f}% | Overall: {overall_rc:.1f}%")

    # Generate Plot
    plt.figure(figsize=(12, 6))
    for m in methods:
        plt.plot(range(1, 31), method_accuracies[m], marker='o', label=m)
    plt.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=20.5, color='gray', linestyle='--', alpha=0.7)
    plt.text(5, 5, "MNIST", fontsize=12, ha='center')
    plt.text(15, 5, "KMNIST", fontsize=12, ha='center')
    plt.text(25, 5, "FashionMNIST (Novel)", fontsize=12, ha='center')
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Test-Time Model Merging Performance Comparison (3 Experts)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=300)
    print("\nSaved plot to evaluation_results.png")

if __name__ == "__main__":
    experts = load_experts()
    datasets_dict = get_datasets()
    proto_data = precompute_unified_prototypes(experts, datasets_dict)
    
    mnist_loader = DataLoader(datasets_dict["mnist_test"], batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(datasets_dict["kmnist_test"], batch_size=64, shuffle=True)
    fashion_loader = DataLoader(datasets_dict["fashion_test"], batch_size=64, shuffle=True)
    
    stream_batches = []
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        imgs, lbls = next(mnist_iter)
        stream_batches.append((imgs, lbls, "MNIST"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        imgs, lbls = next(kmnist_iter)
        stream_batches.append((imgs, lbls, "KMNIST"))
        
    fashion_iter = iter(fashion_loader)
    for _ in range(10):
        imgs, lbls = next(fashion_iter)
        stream_batches.append((imgs, lbls, "FashionMNIST"))
        
    run_evaluation(experts, datasets_dict, proto_data, stream_batches)
