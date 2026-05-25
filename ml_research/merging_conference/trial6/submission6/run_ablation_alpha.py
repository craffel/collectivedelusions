import os
import copy
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pretrain import ExpertModel

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

# Compute contrastive loss
def compute_contrastive_loss(features, logits, prototypes, tau_p, temperature=0.1):
    probs = torch.softmax(logits, dim=1)
    max_probs, pseudo_labels = probs.max(dim=1)
    
    mask = max_probs > tau_p
    if not mask.any():
        return None
        
    masked_features = features[mask]
    masked_labels = pseudo_labels[mask]
    
    all_protos = []
    for c in range(10):
        all_protos.append(prototypes["expert1"][c].to(features.device))
    for c in range(10):
        all_protos.append(prototypes["expert2"][c].to(features.device))
    for c in range(10):
        all_protos.append(prototypes["novel"][c].to(features.device))
        
    all_protos = torch.stack(all_protos)
    
    masked_features_norm = masked_features / (masked_features.norm(dim=1, keepdim=True) + 1e-8)
    all_protos_norm = all_protos / (all_protos.norm(dim=1, keepdim=True) + 1e-8)
    
    sim_matrix = torch.matmul(masked_features_norm, all_protos_norm.T) / temperature
    targets = 20 + masked_labels
    
    loss = nn.CrossEntropyLoss()(sim_matrix, targets)
    return loss

def run_ablation(alpha_val, use_corruption=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experts
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
    base_model = ExpertModel().to(device).eval()

    # Pre-compute elements
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

    cifar_cal = Subset(cifar_train, list(range(500)))
    svhn_cal = Subset(svhn_train, list(range(500)))
    fmnist_cal = Subset(fmnist_train, list(range(500)))

    fisher1 = compute_diagonal_fisher(expert1, cifar_cal, num_samples=500)
    fisher2 = compute_diagonal_fisher(expert2, svhn_cal, num_samples=500)
    fisher3 = compute_diagonal_fisher(expert3, fmnist_cal, num_samples=500)

    joint_fisher = {}
    for name in fisher1:
        f_val = (fisher1[name] + fisher2[name] + fisher3[name]) / 3.0
        joint_fisher[name] = f_val.mean().item()

    prototypes1 = compute_offline_prototypes(expert1, cifar_train, device)
    prototypes2 = compute_offline_prototypes(expert2, svhn_train, device)

    centroid1 = torch.stack(list(prototypes1.values())).mean(dim=0).to(device)
    centroid2 = torch.stack(list(prototypes2.values())).mean(dim=0).to(device)

    for c in range(10):
        prototypes1[c] = prototypes1[c] - centroid1.cpu()
        prototypes2[c] = prototypes2[c] - centroid2.cpu()
    
    loader_fmnist_cent = DataLoader(Subset(fmnist_train, list(range(1000))), batch_size=64, shuffle=False)
    feats_fmnist = []
    with torch.no_grad():
        for inputs, _ in loader_fmnist_cent:
            inputs = inputs.to(device)
            f = expert3.get_features(inputs)
            feats_fmnist.append(f)
    centroid3 = torch.cat(feats_fmnist).mean(dim=0).to(device)

    centroids = [centroid1, centroid2, centroid3]
    pre_prototypes = {"expert1": prototypes1, "expert2": prototypes2}

    def merge_model_state(base_model, experts, lambda_dict, global_lambda, device):
        merged_state = {}
        for name, val in base_model.state_dict().items():
            if name in lambda_dict:
                coeffs = project_simplex(lambda_dict[name])
            else:
                found_coeffs = None
                for suffix in ["running_mean", "running_var", "num_batches_tracked"]:
                    if name.endswith(suffix):
                        base_key = name[:-len(suffix)]
                        for param_suffix in ["weight", "bias"]:
                            test_key = base_key + param_suffix
                            if test_key in lambda_dict:
                                found_coeffs = project_simplex(lambda_dict[test_key])
                                break
                        if found_coeffs is not None:
                            break
                if found_coeffs is not None:
                    coeffs = found_coeffs
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

    # Prepare streams
    test_set_cifar = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_clean)
    test_set_svhn = torchvision.datasets.SVHN(root="./data", split="test", download=False, transform=transform_clean)
    test_set_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform_fmnist)

    subset_cifar = Subset(test_set_cifar, list(range(1920)))
    subset_svhn = Subset(test_set_svhn, list(range(1920)))
    subset_fmnist = Subset(test_set_fmnist, list(range(1920)))

    test_batches = []
    for x, y in DataLoader(subset_cifar, batch_size=64, shuffle=False):
        test_batches.append((x, y, "CIFAR10"))
    for x, y in DataLoader(subset_svhn, batch_size=64, shuffle=False):
        test_batches.append((x, y, "SVHN"))
    for x, y in DataLoader(subset_fmnist, batch_size=64, shuffle=False):
        test_batches.append((x, y, "FashionMNIST"))

    # Initializations
    expert1_m = copy.deepcopy(expert1)
    expert2_m = copy.deepcopy(expert2)
    expert3_m = copy.deepcopy(expert3)
    experts_m = [expert1_m, expert2_m, expert3_m]

    lambda_dict = {}
    for name, param in base_model.named_parameters():
        lambda_dict[name] = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device, requires_grad=True)

    global_lambda = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device)
    online_novel_prototypes = {c: None for c in range(10)}

    accs = []
    task_accs = {"CIFAR10": [], "SVHN": [], "FashionMNIST": []}

    for b_idx, (inputs, labels, task_name) in enumerate(test_batches):
        inputs, labels = inputs.to(device), labels.to(device)
        if use_corruption:
            inputs = add_corruption(inputs)

        with torch.no_grad():
            expert1_m.train()
            features1 = expert1_m.get_features(inputs)
            features_cent1 = features1 - centroids[0]
            features_norm1 = features_cent1 / (features_cent1.norm(dim=1, keepdim=True) + 1e-8)
            sim_exp1 = []
            for c in range(10):
                p_norm = pre_prototypes["expert1"][c].to(device)
                p_norm = p_norm / (p_norm.norm() + 1e-8)
                sim_exp1.append(torch.matmul(features_norm1, p_norm))
            sim_exp1 = torch.stack(sim_exp1, dim=1)
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
            sim_exp2 = torch.stack(sim_exp2, dim=1)
            cohesion_exp2 = sim_exp2.max(dim=1)[0].mean().item()

        max_cohesion = max(cohesion_exp1, cohesion_exp2)
        is_novel_detected = max_cohesion < 0.50

        if not is_novel_detected:
            k_star = np.argmax([cohesion_exp1, cohesion_exp2])
            alpha_ema = 0.99
            global_lambda = (1.0 - alpha_ema) * global_lambda + alpha_ema * torch.eye(3, device=device)[k_star]
            with torch.no_grad():
                for name in lambda_dict:
                    lambda_dict[name].copy_(global_lambda)
        else:
            if online_novel_prototypes[0] is None:
                with torch.no_grad():
                    global_lambda = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0], device=device)
                    for name in lambda_dict:
                        lambda_dict[name].copy_(global_lambda)
                        
                with torch.no_grad():
                    merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, device)
                    base_model.train()
                    features = torch.func.functional_call(base_model, merged_params, (inputs, True))
                    features_cent = features - sum(global_lambda[k] * centroids[k] for k in range(3))
                    logits = torch.func.functional_call(base_model, merged_params, (inputs, False))
                    probs = torch.softmax(logits, dim=1)
                    max_probs, preds = probs.max(dim=1)
                
                for c in range(10):
                    c_mask = (preds == c) & (max_probs > 0.40)
                    if c_mask.any():
                        online_novel_prototypes[c] = features_cent[c_mask].mean(dim=0).cpu()
                    else:
                        online_novel_prototypes[c] = features_cent.mean(dim=0).cpu()
            else:
                with torch.no_grad():
                    merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, device)
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

            prototypes_dict = {
                "expert1": pre_prototypes["expert1"],
                "expert2": pre_prototypes["expert2"],
                "novel": online_novel_prototypes
            }

            num_steps = 5
            for step in range(num_steps):
                merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, device)
                base_model.train()
                features = torch.func.functional_call(base_model, merged_params, (inputs, True))
                features_centered = features - sum(global_lambda[k] * centroids[k] for k in range(3))
                logits = torch.func.functional_call(base_model, merged_params, (inputs, False))
                loss = compute_contrastive_loss(features_centered, logits, prototypes_dict, tau_p=0.40)
                
                if loss is not None:
                    grads = torch.autograd.grad(loss, list(lambda_dict.values()), allow_unused=True)
                    with torch.no_grad():
                        for i, (name, param) in enumerate(lambda_dict.items()):
                            grad = grads[i]
                            if grad is not None:
                                # Preconditioned learning rate with variable alpha
                                lr = 0.001 * ((joint_fisher[name] + 1e-6) ** -alpha_val)
                                lr = min(lr, 0.1)
                                param.copy_(param - lr * grad)
                                param.copy_(project_simplex(param))

            with torch.no_grad():
                layer4_coeffs = []
                for name in lambda_dict:
                    if "resnet.layer4" in name:
                        layer4_coeffs.append(project_simplex(lambda_dict[name]))
                if len(layer4_coeffs) > 0:
                    global_lambda = torch.stack(layer4_coeffs).mean(dim=0)
                else:
                    global_lambda = torch.stack([project_simplex(v) for v in lambda_dict.values()]).mean(dim=0)

        with torch.no_grad():
            merged_params = merge_model_state(base_model, experts_m, lambda_dict, global_lambda, device)
            base_model.train()
            outputs = torch.func.functional_call(base_model, merged_params, (inputs, False))
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels).sum().item() / inputs.size(0)

        accs.append(acc)
        task_accs[task_name].append(acc)

    cifar_acc = 100. * np.mean(task_accs["CIFAR10"])
    svhn_acc = 100. * np.mean(task_accs["SVHN"])
    fmnist_acc = 100. * np.mean(task_accs["FashionMNIST"])
    overall_acc = 100. * np.mean(accs)
    
    return cifar_acc, svhn_acc, fmnist_acc, overall_acc

if __name__ == "__main__":
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    
    results_clean = []
    results_corrupted = []
    
    print("=== Running Ablation Sweep over Alpha (Clean scenario) ===")
    for a in alphas:
        c_acc, s_acc, f_acc, o_acc = run_ablation(a, use_corruption=False)
        print(f"Alpha: {a:.2f} | CIFAR10: {c_acc:.2f}% | SVHN: {s_acc:.2f}% | F-MNIST (Novel): {f_acc:.2f}% | Overall: {o_acc:.2f}%")
        results_clean.append((c_acc, s_acc, f_acc, o_acc))
        
    print("\n=== Running Ablation Sweep over Alpha (Corrupted scenario) ===")
    for a in alphas:
        c_acc, s_acc, f_acc, o_acc = run_ablation(a, use_corruption=True)
        print(f"Alpha: {a:.2f} | CIFAR10: {c_acc:.2f}% | SVHN: {s_acc:.2f}% | F-MNIST (Novel): {f_acc:.2f}% | Overall: {o_acc:.2f}%")
        results_corrupted.append((c_acc, s_acc, f_acc, o_acc))
        
    results_clean = np.array(results_clean)
    results_corrupted = np.array(results_corrupted)
    
    np.savez("ablation_results.npz", alphas=alphas, clean=results_clean, corrupted=results_corrupted)
    print("\nAblation sweep completed! Saved to ablation_results.npz")
    
    # Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Clean
    axes[0].plot(alphas, results_clean[:, 2], marker='o', color='royalblue', label='Novel Domain (FashionMNIST)')
    axes[0].plot(alphas, results_clean[:, 3], marker='s', color='darkorange', label='Overall Accuracy')
    axes[0].set_title("Clean Test Stream")
    axes[0].set_xlabel("Sensitivity Damping Exponent ($\\alpha$)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Right: Corrupted
    axes[1].plot(alphas, results_corrupted[:, 2], marker='o', color='crimson', label='Novel Domain (FashionMNIST)')
    axes[1].plot(alphas, results_corrupted[:, 3], marker='s', color='purple', label='Overall Accuracy')
    axes[1].set_title("Corrupted Test Stream")
    axes[1].set_xlabel("Sensitivity Damping Exponent ($\\alpha$)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("ablation_alpha_plot.png", dpi=300)
    print("Ablation plot saved to ablation_alpha_plot.png")
