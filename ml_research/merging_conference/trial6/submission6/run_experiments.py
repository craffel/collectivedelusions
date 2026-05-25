import os
import copy
import numpy as np
import torch
# Disable cuDNN to bypass recurrent initialization errors on the cluster environment
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Import ExpertModel from pretrain
from pretrain import ExpertModel, train_expert

# Simplex projection helper
def project_simplex(v):
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(n_features, device=v.device, dtype=v.dtype) + 1.0
    cond = u - cssv / ind > 0
    rho = torch.max(ind * cond) - 1
    theta = cssv[int(rho.item())] / (rho + 1.0)
    w = torch.clamp(v - theta, min=0.0)
    return w

# Diagonal Fisher helper
def compute_diagonal_fisher(model, dataset, num_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
    with torch.no_grad():
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
    return fisher_dict

# Offline prototypes helper
def compute_offline_prototypes(model, dataset, device, num_samples_per_class=100):
    model.eval()
    model.to(device)
    class_features = {c: [] for c in range(10)}
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            for f, l in zip(features, labels):
                l_item = l.item()
                if len(class_features[l_item]) < num_samples_per_class:
                    class_features[l_item].append(f.cpu())
            if all(len(class_features[c]) >= num_samples_per_class for c in range(10)):
                break
    prototypes = {}
    for c in range(10):
        if len(class_features[c]) > 0:
            prototypes[c] = torch.stack(class_features[c]).mean(dim=0)
        else:
            prototypes[c] = torch.zeros(512)
    return prototypes

# Corruption function
def add_corruption(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x_raw = x * std + mean
    noise = torch.randn_like(x_raw) * 0.2
    x_corrupted = torch.clamp(x_raw + noise, 0.0, 1.0)
    x_normalized = (x_corrupted - mean) / std
    return x_normalized

# Compute contrastive loss for online adaptation
def compute_contrastive_loss(features, logits, prototypes, tau_p, temperature=0.1):
    probs = torch.softmax(logits, dim=1)
    max_probs, pseudo_labels = probs.max(dim=1)
    
    # Filter by confidence threshold
    mask = max_probs > tau_p
    if not mask.any():
        return None
        
    masked_features = features[mask]
    masked_labels = pseudo_labels[mask]
    
    # Stack all prototypes (Known Expert 1, Known Expert 2, Online Novel)
    all_protos = []
    # Expert 1 prototypes
    for c in range(10):
        all_protos.append(prototypes["expert1"][c].to(features.device))
    # Expert 2 prototypes
    for c in range(10):
        all_protos.append(prototypes["expert2"][c].to(features.device))
    # Novel prototypes
    for c in range(10):
        all_protos.append(prototypes["novel"][c].to(features.device))
        
    all_protos = torch.stack(all_protos) # (30, 512)
    
    # Cosine similarity (B_masked, 30)
    masked_features_norm = masked_features / (masked_features.norm(dim=1, keepdim=True) + 1e-8)
    all_protos_norm = all_protos / (all_protos.norm(dim=1, keepdim=True) + 1e-8)
    
    sim_matrix = torch.matmul(masked_features_norm, all_protos_norm.T) / temperature
    
    # Targets correspond to novel prototypes (index 20 to 29)
    targets = 20 + masked_labels
    
    loss = nn.CrossEntropyLoss()(sim_matrix, targets)
    return loss

# Main experimentation suite
def run_evaluation(use_corruption=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"RUNNING EXPERIMENTS (Corruption: {use_corruption})")
    print(f"==========================================\n")

    # 1. Train experts if not already available
    if not os.path.exists("expert_cifar10.pth"):
        print("Expert checkpoints not found. Training experts first...")
        train_expert("CIFAR10", "expert_cifar10.pth")
    if not os.path.exists("expert_svhn.pth"):
        train_expert("SVHN", "expert_svhn.pth")
    if not os.path.exists("expert_fmnist.pth"):
        train_expert("FashionMNIST", "expert_fmnist.pth")

    # 2. Load experts
    print("Loading experts...")
    expert1 = ExpertModel()
    expert1.load_state_dict(torch.load("expert_cifar10.pth", map_location=device))
    expert1.to(device).eval()

    expert2 = ExpertModel()
    expert2.load_state_dict(torch.load("expert_svhn.pth", map_location=device))
    expert2.to(device).eval()

    expert3 = ExpertModel()
    expert3.load_state_dict(torch.load("expert_fmnist.pth", map_location=device))
    expert3.to(device).eval()

    experts = [expert1, expert2, expert3]

    # Create dummy base model to hold merged weights functionally
    base_model = ExpertModel().to(device).eval()

    # 3. Pre-compute Fisher, Class Prototypes, and Centroids
    transform_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_fmnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_clean)
    svhn_train = torchvision.datasets.SVHN(root="./data", split="train", download=False, transform=transform_clean)
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform_fmnist)

    print("Pre-computing elements...")
    cifar_cal = Subset(cifar_train, list(range(500)))
    svhn_cal = Subset(svhn_train, list(range(500)))
    fmnist_cal = Subset(fmnist_train, list(range(500)))

    # Fisher information
    fisher1 = compute_diagonal_fisher(expert1, cifar_cal, num_samples=500)
    fisher2 = compute_diagonal_fisher(expert2, svhn_cal, num_samples=500)
    fisher3 = compute_diagonal_fisher(expert3, fmnist_cal, num_samples=500)

    joint_fisher = {}
    for name in fisher1:
        f_val = (fisher1[name] + fisher2[name] + fisher3[name]) / 3.0
        joint_fisher[name] = f_val.mean().item()

    # Offline prototypes
    prototypes1 = compute_offline_prototypes(expert1, cifar_train, device)
    prototypes2 = compute_offline_prototypes(expert2, svhn_train, device)

    # Centroids
    centroid1 = torch.stack(list(prototypes1.values())).mean(dim=0).to(device)
    centroid2 = torch.stack(list(prototypes2.values())).mean(dim=0).to(device)
    
    # Pre-center the offline prototypes
    for c in range(10):
        prototypes1[c] = prototypes1[c] - centroid1.cpu()
        prototypes2[c] = prototypes2[c] - centroid2.cpu()
    
    # Expert 3 Centroid (clean)
    loader_fmnist_cent = DataLoader(Subset(fmnist_train, list(range(1000))), batch_size=64, shuffle=False)
    feats_fmnist = []
    with torch.no_grad():
        for inputs, _ in loader_fmnist_cent:
            inputs = inputs.to(device)
            f = expert3.get_features(inputs)
            feats_fmnist.append(f)
    centroid3 = torch.cat(feats_fmnist).mean(dim=0).to(device)

    centroids = [centroid1, centroid2, centroid3]

    # Pre-computed prototypes dict
    pre_prototypes = {
        "expert1": prototypes1,
        "expert2": prototypes2
    }

    # Helper function to dynamically merge parameters and buffers
    def merge_model_state(base_model, experts, lambda_dict, global_lambda, method, device):
        merged_state = {}
        for name, val in base_model.state_dict().items():
            if method in ["Static", "PROTO-TTMM", "FP-OW (Ours)"]:
                if name in lambda_dict:
                    coeffs = proj_simplex_torch(lambda_dict[name])
                else:
                    found_coeffs = None
                    for suffix in ["running_mean", "running_var", "num_batches_tracked"]:
                        if name.endswith(suffix):
                            base_key = name[:-len(suffix)]
                            for param_suffix in ["weight", "bias"]:
                                test_key = base_key + param_suffix
                                if test_key in lambda_dict:
                                    found_coeffs = proj_simplex_torch(lambda_dict[test_key])
                                    break
                            if found_coeffs is not None:
                                break
                    if found_coeffs is not None:
                        coeffs = found_coeffs
                    else:
                        coeffs = global_lambda
            else:
                coeffs = global_lambda
                
            expert_vals = [exp.state_dict()[name].to(device) for exp in experts]
            if val.is_floating_point() or val.is_complex():
                val_merged = sum(coeffs[k] * expert_vals[k] for k in range(3))
                if name not in lambda_dict:
                    val_merged = val_merged.detach()
                merged_state[name] = val_merged
            else:
                best_k = torch.argmax(coeffs).item()
                merged_state[name] = expert_vals[best_k].detach()
        return merged_state

    # 4. Prepare Test Stream
    test_set_cifar = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_clean)
    test_set_svhn = torchvision.datasets.SVHN(root="./data", split="test", download=False, transform=transform_clean)
    test_set_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform_fmnist)

    subset_cifar = Subset(test_set_cifar, list(range(1920)))
    subset_svhn = Subset(test_set_svhn, list(range(1920)))
    subset_fmnist = Subset(test_set_fmnist, list(range(1920)))

    test_batches = []
    
    loader_cifar = DataLoader(subset_cifar, batch_size=64, shuffle=False)
    for x, y in loader_cifar:
        test_batches.append((x, y, "CIFAR10"))

    loader_svhn = DataLoader(subset_svhn, batch_size=64, shuffle=False)
    for x, y in loader_svhn:
        test_batches.append((x, y, "SVHN"))

    loader_fmnist = DataLoader(subset_fmnist, batch_size=64, shuffle=False)
    for x, y in loader_fmnist:
        test_batches.append((x, y, "FashionMNIST"))

    # Define Methods
    methods = ["Static", "TENT", "PC-Merge", "CPA-Merge", "PROTO-TTMM", "FP-OW (Ours)"]
    results = {m: {"accs": [], "ndr": 0, "fpr": 0, "task_accs": {"CIFAR10": [], "SVHN": [], "FashionMNIST": []}} for m in methods}

    # Simplex projection helper for numpy / torch
    def proj_simplex_torch(v):
        return project_simplex(v)

    # Let's run evaluation for each method
    for method in methods:
        print(f"\n--- Evaluating Method: {method} ---")
        
        # Create deep copies of experts to prevent any running stats updates from bleeding between methods
        expert1_m = copy.deepcopy(expert1)
        expert2_m = copy.deepcopy(expert2)
        expert3_m = copy.deepcopy(expert3)
        experts_m = [expert1_m, expert2_m, expert3_m]

        # Initialize layer-wise merging coefficients
        # Unconstrained logits representation for simplex weights
        lambda_dict = {}
        for name, param in base_model.named_parameters():
            # Initialized to uniform weight [1/3, 1/3, 1/3]
            lambda_dict[name] = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device, requires_grad=True)

        # Global coefficient EMA for routing
        global_lambda = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device)

        # Online novel prototypes
        online_novel_prototypes = {c: None for c in range(10)}
        novel_detected_flag = False

        # If TENT, prepare BN optimizer
        tent_model = copy.deepcopy(base_model)
        # Set to average weights initially
        with torch.no_grad():
            for name, param in tent_model.named_parameters():
                expert_params = [exp.state_dict()[name].to(device) for exp in experts_m]
                merged = sum(1.0/3.0 * expert_params[k] for k in range(3))
                param.copy_(merged)
        
        bn_params = []
        for name, param in tent_model.named_parameters():
            if "bn" in name and ("weight" in name or "bias" in name):
                param.requires_grad = True
                bn_params.append(param)
            else:
                param.requires_grad = False
        tent_optimizer = optim.SGD(bn_params, lr=0.001)

        # Metrics trackers
        novel_batches_correct = 0
        novel_batches_total = 0
        known_batches_false = 0
        known_batches_total = 0

        for b_idx, (inputs, labels, task_name) in enumerate(test_batches):
            inputs, labels = inputs.to(device), labels.to(device)
            if use_corruption:
                inputs = add_corruption(inputs)

            # Step 1: Compute Centered Features and Cohesion Scores for Routing using AdaBN (model.train())
            with torch.no_grad():
                # Extract features from Expert 1 and Expert 2 individually for robust, unbiased routing
                expert1_m.train()
                features1 = expert1_m.get_features(inputs)
                features_cent1 = features1 - centroids[0]
                features_norm1 = features_cent1 / (features_cent1.norm(dim=1, keepdim=True) + 1e-8)
                sim_exp1 = []
                for c in range(10):
                    p_norm = pre_prototypes["expert1"][c].to(device)
                    p_norm = p_norm / (p_norm.norm() + 1e-8)
                    sim_exp1.append(torch.matmul(features_norm1, p_norm))
                sim_exp1 = torch.stack(sim_exp1, dim=1) # (B, 10)
                cohesion_exp1 = sim_exp1.max(dim=1)[0].mean().item()

                expert2_m.train()
                features2 = expert2_m.get_features(inputs)
                features_cent2 = features2 - centroids[1]
                features_norm2 = features_cent2 / (features_cent2.norm(dim=1, keepdim=True) + 1e-8)
                sim_exp2 = []
                for c in range(10):
                    p_norm = pre_prototypes["expert2"][c].to(device)
                    p_norm = p_norm / (p_norm.norm() + 1e-8)
                    sim_exp2.append(torch.matmul(features_norm2, p_norm))
                sim_exp2 = torch.stack(sim_exp2, dim=1) # (B, 10)
                cohesion_exp2 = sim_exp2.max(dim=1)[0].mean().item()

            # Novelty Detection
            max_cohesion = max(cohesion_exp1, cohesion_exp2)
            is_novel_detected = max_cohesion < 0.50

            # Track Novelty Detection Rates
            if task_name == "FashionMNIST":
                novel_batches_total += 1
                if is_novel_detected:
                    novel_batches_correct += 1
            else:
                known_batches_total += 1
                if is_novel_detected:
                    known_batches_false += 1

            # Step 2: Adaptation depending on the method
            if method == "Static":
                # Always use 1/3 global lambda, no adaptation
                pass

            elif method == "TENT":
                # Adapt BN parameters functionally via entropy minimization
                tent_model.train()
                tent_optimizer.zero_grad()
                outputs = tent_model(inputs)
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
                entropy.backward()
                tent_optimizer.step()
                
                # For testing, run in eval mode
                tent_model.eval()
                with torch.no_grad():
                    outputs = tent_model(inputs)
                _, predicted = outputs.max(1)
                acc = predicted.eq(labels).sum().item() / inputs.size(0)
                results[method]["accs"].append(acc)
                results[method]["task_accs"][task_name].append(acc)
                continue

            elif method == "PC-Merge":
                # Route completely to the best expert, no novelty adaptation
                k_star = np.argmax([cohesion_exp1, cohesion_exp2])
                global_lambda = torch.zeros(3, device=device)
                global_lambda[k_star] = 1.0

            elif method == "CPA-Merge":
                # Entropy-based routing and EMA update
                with torch.no_grad():
                    # Compute prediction entropy of the current merged model
                    merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, method, device)
                    base_model.train()
                    logits = torch.func.functional_call(base_model, merged_params, (inputs, False))
                    probs = torch.softmax(logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean().item()
                    
                    # CPA-Merge selects the expert that minimizes entropy. Let's see which expert has higher confidence
                    # (Lower entropy). We can evaluate entropy of Expert 1 vs Expert 2 on this batch
                    expert1_m.train()
                    outputs_exp1 = expert1_m(inputs)
                    entropy_exp1 = -torch.sum(torch.softmax(outputs_exp1, dim=1) * torch.log(torch.softmax(outputs_exp1, dim=1) + 1e-6), dim=1).mean().item()
                    expert2_m.train()
                    outputs_exp2 = expert2_m(inputs)
                    entropy_exp2 = -torch.sum(torch.softmax(outputs_exp2, dim=1) * torch.log(torch.softmax(outputs_exp2, dim=1) + 1e-6), dim=1).mean().item()
                    
                    k_star = 0 if entropy_exp1 < entropy_exp2 else 1
                    # EMA update
                    alpha = 0.99
                    global_lambda = (1.0 - alpha) * global_lambda + alpha * torch.eye(3, device=device)[k_star]

            elif method in ["PROTO-TTMM", "FP-OW (Ours)"]:
                if not is_novel_detected:
                    # Case 1: Known Domain. Standard EMA routing.
                    k_star = np.argmax([cohesion_exp1, cohesion_exp2])
                    alpha = 0.99
                    global_lambda = (1.0 - alpha) * global_lambda + alpha * torch.eye(3, device=device)[k_star]
                    
                    # Update all layer-wise merging coefficients to match global coefficients
                    with torch.no_grad():
                        for name in lambda_dict:
                            lambda_dict[name].copy_(global_lambda)
                else:
                    # Case 2: Novel Domain. Online Prototype adaptation!
                    novel_detected_flag = True
                    
                    # Initialize online prototypes for FashionMNIST on first novel detection
                    if online_novel_prototypes[0] is None:
                        # Reset global_lambda and lambda_dict to uniform to bootstrap novel domain adaptation
                        with torch.no_grad():
                            global_lambda = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device)
                            for name in lambda_dict:
                                lambda_dict[name].copy_(global_lambda)
                                
                        # Get centered features
                        with torch.no_grad():
                            merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, method, device)
                            base_model.train()
                            features = torch.func.functional_call(base_model, merged_params, (inputs, True))
                            features_cent = features - sum(global_lambda[k] * centroids[k] for k in range(3))
                            logits = torch.func.functional_call(base_model, merged_params, (inputs, False))
                            probs = torch.softmax(logits, dim=1)
                            max_probs, preds = probs.max(dim=1)
                        
                        # Initialize novel prototypes class-by-class
                        for c in range(10):
                            c_mask = (preds == c) & (max_probs > 0.40)
                            if c_mask.any():
                                online_novel_prototypes[c] = features_cent[c_mask].mean(dim=0).cpu()
                            else:
                                online_novel_prototypes[c] = features_cent.mean(dim=0).cpu() # fallback
                    else:
                        # Update online prototypes with EMA (gamma = 0.1)
                        with torch.no_grad():
                            merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, method, device)
                            base_model.train()
                            features = torch.func.functional_call(base_model, merged_params, (inputs, True))
                            features_cent = features - sum(global_lambda[k] * centroids[k] for k in range(3))
                            logits = torch.func.functional_call(base_model, merged_params, (inputs, False))
                            probs = torch.softmax(logits, dim=1)
                            max_probs, preds = probs.max(dim=1)
                        
                        for c in range(10):
                            c_mask = (preds == c) & (max_probs > 0.40)
                            if c_mask.any():
                                b_mean = features_cent[c_mask].mean(dim=0).cpu()
                                if online_novel_prototypes[c] is None:
                                    online_novel_prototypes[c] = b_mean
                                else:
                                    online_novel_prototypes[c] = 0.90 * online_novel_prototypes[c] + 0.10 * b_mean

                    # Optimize merging coefficients
                    # Assemble prototypes dictionary
                    prototypes_dict = {
                        "expert1": pre_prototypes["expert1"],
                        "expert2": pre_prototypes["expert2"],
                        "novel": online_novel_prototypes
                    }

                    # Number of test-time optimization steps per batch
                    num_steps = 5
                    for step in range(num_steps):
                        # Construct merged model functionally using both parameters and buffers
                        merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, method, device)

                        base_model.train()
                        # Forward pass to get features and logits
                        features = torch.func.functional_call(base_model, merged_params, (inputs, True))
                        # Centering features using global_lambda centroid for stability during gradient step
                        features_centered = features - sum(global_lambda[k] * centroids[k] for k in range(3))

                        logits = torch.func.functional_call(base_model, merged_params, (inputs, False))

                        # Loss
                        loss = compute_contrastive_loss(features_centered, logits, prototypes_dict, tau_p=0.40)
                        
                        if loss is not None:
                            # Backpropagation to compute gradients on lambda_dict
                            grads = torch.autograd.grad(loss, list(lambda_dict.values()), allow_unused=True)
                            
                            # Update coefficients
                            with torch.no_grad():
                                for i, (name, param) in enumerate(lambda_dict.items()):
                                    grad = grads[i]
                                    if grad is not None:
                                        if method == "PROTO-TTMM":
                                            # Uniform learning rate (0.001)
                                            lr = 0.001
                                        else:
                                            # FP-OW (Ours): Fisher preconditioned learning rate!
                                            # η_w = η * (F_w + ϵ_scale)^(-α)
                                            lr = 0.001 * ((joint_fisher[name] + 1e-6) ** -1.0)
                                            # Clip learning rate to avoid extreme gradient steps
                                            lr = min(lr, 0.1)
                                            
                                        param.copy_(param - lr * grad)
                                        # Project back onto probability simplex
                                        param.copy_(proj_simplex_torch(param))

                    # Update global_lambda based on the mean of layer-wise coefficients for routing center
                    with torch.no_grad():
                        layer4_coeffs = []
                        for name in lambda_dict:
                            if "resnet.layer4" in name:
                                layer4_coeffs.append(proj_simplex_torch(lambda_dict[name]))
                        if len(layer4_coeffs) > 0:
                            global_lambda = torch.stack(layer4_coeffs).mean(dim=0)
                        else:
                            global_lambda = torch.stack([proj_simplex_torch(v) for v in lambda_dict.values()]).mean(dim=0)

            # Step 3: Run Forward Pass of current Merged Model to evaluate accuracy
            with torch.no_grad():
                merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, method, device)
                base_model.train()
                outputs = torch.func.functional_call(base_model, merged_params, (inputs, False))
                _, predicted = outputs.max(1)
                acc = predicted.eq(labels).sum().item() / inputs.size(0)

            results[method]["accs"].append(acc)
            results[method]["task_accs"][task_name].append(acc)

        # Record NDR and FPR
        ndr = 100. * novel_batches_correct / novel_batches_total if novel_batches_total > 0 else 0.0
        fpr = 100. * known_batches_false / known_batches_total if known_batches_total > 0 else 0.0
        results[method]["ndr"] = ndr
        results[method]["fpr"] = fpr

        # Print results
        cifar_acc = 100. * np.mean(results[method]["task_accs"]["CIFAR10"])
        svhn_acc = 100. * np.mean(results[method]["task_accs"]["SVHN"])
        fmnist_acc = 100. * np.mean(results[method]["task_accs"]["FashionMNIST"])
        overall_acc = 100. * np.mean(results[method]["accs"])
        
        print(f"Results for {method}:")
        print(f"  CIFAR-10 Acc:     {cifar_acc:.2f}%")
        print(f"  SVHN Acc:         {svhn_acc:.2f}%")
        print(f"  FashionMNIST Acc: {fmnist_acc:.2f}%")
        print(f"  Overall Acc:      {overall_acc:.2f}%")
        print(f"  Novelty Detection Rate (NDR): {ndr:.2f}%")
        print(f"  False Positive Rate (FPR):    {fpr:.2f}%")

    return results

if __name__ == "__main__":
    # Run clean experiments
    clean_results = run_evaluation(use_corruption=False)
    
    # Run corrupted experiments
    corrupted_results = run_evaluation(use_corruption=True)

    # Save results as npz for paper writing and plotting
    np.savez("experiment_results.npz", clean=clean_results, corrupted=corrupted_results)
    print("\nAll experiments completed and results saved to experiment_results.npz!")
