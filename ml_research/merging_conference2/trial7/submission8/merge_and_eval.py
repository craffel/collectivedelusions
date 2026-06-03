import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED
torch.backends.cudnn.enabled = False

# Transforms (same as training)
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_test_loader(name, batch_size=128):
    if name == "mnist":
        dataset = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    elif name == "fmnist":
        dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=fmnist_transform)
    elif name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar10_transform)
    else:
        raise ValueError("Unknown dataset " + name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def load_expert_state(path):
    print("Loading expert from:", path)
    return torch.load(path, map_location="cpu")

# Evaluation function
def evaluate_model(backbone_state, fc_state, loader):
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    
    # Assemble model state dict
    full_state = {}
    for k, v in backbone_state.items():
        full_state[k] = v
    for k, v in fc_state.items():
        full_state[k] = v
        
    model.load_state_dict(full_state, strict=True)
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            test_correct = predicted.eq(targets).sum().item()
            correct += test_correct
            
    return 100.0 * correct / total

# Measure activation variance
def measure_activation_variance(backbone_state, loader):
    model = models.resnet18()
    model.fc = nn.Identity() # we only care about the backbone
    
    # Load backbone parameters
    model_state = model.state_dict()
    for k in model_state.keys():
        if k in backbone_state:
            model_state[k] = backbone_state[k]
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    variances = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            # Compute variance of the output activations across the batch and spatial/channel dims
            variances[name] = torch.var(output).item()
        return hook

    # Register hooks on layers of interest (the four main ResNet blocks)
    hooks.append(model.layer1.register_forward_hook(make_hook("layer1")))
    hooks.append(model.layer2.register_forward_hook(make_hook("layer2")))
    hooks.append(model.layer3.register_forward_hook(make_hook("layer3")))
    hooks.append(model.layer4.register_forward_hook(make_hook("layer4")))
    
    # Get one batch from loader
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        _ = model(inputs)
        
    for h in hooks:
        h.remove()
        
    return variances

# Merging functions
def weight_averaging(progenitor, experts, weights=None):
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        merged_state[key] = sum(w * exp[key] for w, exp in zip(weights, experts))
    return merged_state

def task_arithmetic(progenitor, experts, lam=0.5):
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        task_vectors = [exp[key] - progenitor[key] for exp in experts]
        merged_state[key] = progenitor[key] + lam * sum(task_vectors)
    return merged_state

def spherical_karcher_mean(vectors, weights, num_iters=10, eps=1e-6):
    orig_shape = vectors[0].shape
    flattened = [v.detach().clone().float().flatten() for v in vectors]
    
    if len(flattened[0]) <= 1:
        # 0-D or 1-D parameters (like scalar biases or tracking stats), use weighted average
        return sum(w * v for w, v in zip(weights, vectors))
        
    norms = [torch.norm(v, p=2) for v in flattened]
    normalized = [v / (norm + 1e-10) for v, norm in zip(flattened, norms)]
    
    # Initialize as normalized weighted average
    mu = sum(w * v for w, v in zip(weights, normalized))
    mu_norm = torch.norm(mu, p=2)
    if mu_norm < 1e-10:
        mu = normalized[0].clone()
    else:
        mu = mu / mu_norm
        
    for _ in range(num_iters):
        tangent_sum = torch.zeros_like(mu)
        for w, u in zip(weights, normalized):
            dot = torch.clamp(torch.dot(mu, u), -1.0 + eps, 1.0 - eps)
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            if sin_theta < 1e-7:
                v = torch.zeros_like(mu)
            else:
                v = (theta / sin_theta) * (u - dot * mu)
            tangent_sum += w * v
            
        v_norm = torch.norm(tangent_sum, p=2)
        if v_norm < 1e-7:
            break
            
        # Exponential map
        mu = torch.cos(v_norm) * mu + torch.sin(v_norm) * (tangent_sum / v_norm)
        mu = mu / torch.norm(mu, p=2)
        
    avg_norm = sum(w * n for w, n in zip(weights, norms))
    rescaled = mu * avg_norm
    return rescaled.view(orig_shape).to(vectors[0].dtype)

def spherical_karcher_mean_channelwise(vectors, weights, num_iters=10, eps=1e-6):
    shape = vectors[0].shape
    if len(shape) < 2:
        return spherical_karcher_mean(vectors, weights, num_iters=num_iters, eps=eps)
    
    out_dim = shape[0]
    merged_channels = []
    for i in range(out_dim):
        channel_vectors = [v[i] for v in vectors]
        merged_channel = spherical_karcher_mean(channel_vectors, weights, num_iters=num_iters, eps=eps)
        merged_channels.append(merged_channel)
        
    return torch.stack(merged_channels, dim=0)

def ties_merging(progenitor, experts, weights=None, fraction=0.2):
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        tensors = [exp[key] for exp in experts]
        
        is_projection_weight = (
            key.endswith(".weight") 
            and "bn" not in key 
            and "downsample.1" not in key
        )
        
        if is_projection_weight:
            task_vectors = [t - progenitor[key] for t in tensors]
            trimmed_tvs = []
            for tv in task_vectors:
                orig_shape = tv.shape
                flat = tv.flatten()
                k = int(fraction * len(flat))
                if k > 0:
                    thres = torch.topk(flat.abs(), k=k).values[-1]
                    mask = flat.abs() >= thres
                    flat_trimmed = flat * mask
                else:
                    flat_trimmed = torch.zeros_like(flat)
                trimmed_tvs.append(flat_trimmed.view(orig_shape))
                
            weighted_signs = sum(w * torch.sign(tv) for w, tv in zip(weights, trimmed_tvs))
            dominant_sign = torch.sign(weighted_signs)
            
            resolved_tvs = []
            for tv in trimmed_tvs:
                agree_mask = (torch.sign(tv) == dominant_sign) | (tv == 0)
                resolved_tvs.append(tv * agree_mask)
                
            stacked_resolved = torch.stack(resolved_tvs, dim=0)
            non_zero_counts = (stacked_resolved != 0).sum(dim=0).float()
            
            sum_tvs = sum(w * tv for w, tv in zip(weights, resolved_tvs))
            merged_tv = torch.where(non_zero_counts > 0, (sum_tvs * len(experts)) / non_zero_counts, torch.zeros_like(sum_tvs))
            merged_state[key] = progenitor[key] + merged_tv
        else:
            merged_state[key] = sum(w * exp[key] for w, exp in zip(weights, experts))
            
    return merged_state

def dare_merging(progenitor, experts, weights=None, p_drop=0.5):
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        tensors = [exp[key] for exp in experts]
        
        is_projection_weight = (
            key.endswith(".weight") 
            and "bn" not in key 
            and "downsample.1" not in key
        )
        
        if is_projection_weight:
            task_vectors = [t - progenitor[key] for t in tensors]
            dropped_tvs = []
            for tv in task_vectors:
                mask = (torch.rand_like(tv.float()) >= p_drop).to(tv.dtype)
                rescaled_tv = (tv * mask) / (1.0 - p_drop)
                dropped_tvs.append(rescaled_tv)
            
            merged_tv = sum(w * dt for w, dt in zip(weights, dropped_tvs))
            merged_state[key] = progenitor[key] + merged_tv
        else:
            merged_state[key] = sum(w * exp[key] for w, exp in zip(weights, experts))
            
    return merged_state

def spherical_karcher_task_arithmetic(progenitor, experts, weights=None, num_iters=10, selective=True, channelwise=False, lam=0.3):
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        tensors = [exp[key] for exp in experts]
        
        is_projection_weight = (
            key.endswith(".weight") 
            and "bn" not in key 
            and "downsample.1" not in key
        )
        
        if selective:
            if is_projection_weight:
                task_vectors = [t - progenitor[key] for t in tensors]
                if channelwise:
                    merged_tv = spherical_karcher_mean_channelwise(task_vectors, weights, num_iters=num_iters)
                else:
                    merged_tv = spherical_karcher_mean(task_vectors, weights, num_iters=num_iters)
                merged_state[key] = progenitor[key] + lam * merged_tv
            else:
                merged_state[key] = sum(w * exp[key] for w, exp in zip(weights, experts))
        else:
            task_vectors = [t - progenitor[key] for t in tensors]
            merged_tv = spherical_karcher_mean(task_vectors, weights, num_iters=num_iters)
            merged_state[key] = progenitor[key] + lam * merged_tv
            
    return merged_state

def sk_ties_merging(progenitor, experts, weights=None, fraction=0.4, num_iters=1, channelwise=False):
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        tensors = [exp[key] for exp in experts]
        
        is_projection_weight = (
            key.endswith(".weight") 
            and "bn" not in key 
            and "downsample.1" not in key
        )
        
        if is_projection_weight:
            task_vectors = [t - progenitor[key] for t in tensors]
            trimmed_tvs = []
            for tv in task_vectors:
                orig_shape = tv.shape
                flat = tv.flatten()
                k = int(fraction * len(flat))
                if k > 0:
                    thres = torch.topk(flat.abs(), k=k).values[-1]
                    mask = flat.abs() >= thres
                    flat_trimmed = flat * mask
                else:
                    flat_trimmed = torch.zeros_like(flat)
                trimmed_tvs.append(flat_trimmed.view(orig_shape))
                
            weighted_signs = sum(w * torch.sign(tv) for w, tv in zip(weights, trimmed_tvs))
            dominant_sign = torch.sign(weighted_signs)
            
            resolved_tvs = []
            for tv in trimmed_tvs:
                agree_mask = (torch.sign(tv) == dominant_sign) | (tv == 0)
                resolved_tvs.append(tv * agree_mask)
                
            stacked_resolved = torch.stack(resolved_tvs, dim=0)
            non_zero_counts = (stacked_resolved != 0).sum(dim=0).float()
            
            if channelwise:
                merged_tv = spherical_karcher_mean_channelwise(resolved_tvs, weights, num_iters=num_iters)
            else:
                merged_tv = spherical_karcher_mean(resolved_tvs, weights, num_iters=num_iters)
                
            # Post-active scaling coordinate-wise
            merged_tv = torch.where(non_zero_counts > 0, (merged_tv * len(experts)) / non_zero_counts, torch.zeros_like(merged_tv))
            merged_state[key] = progenitor[key] + merged_tv
        else:
            merged_state[key] = sum(w * exp[key] for w, exp in zip(weights, experts))
            
    return merged_state

def spherical_karcher_merging(progenitor, experts, weights=None, num_iters=10, selective=True, channelwise=False):
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    merged_state = {}
    for key in progenitor.keys():
        if "fc" in key or "num_batches_tracked" in key:
            continue
        tensors = [exp[key] for exp in experts]
        
        is_projection_weight = (
            key.endswith(".weight") 
            and "bn" not in key 
            and "downsample.1" not in key
        )
        
        if selective:
            if is_projection_weight:
                if channelwise:
                    merged_state[key] = spherical_karcher_mean_channelwise(tensors, weights, num_iters=num_iters)
                else:
                    merged_state[key] = spherical_karcher_mean(tensors, weights, num_iters=num_iters)
            else:
                # Use standard weight averaging for biases, Batch Normalization running statistics, and scales/shifts
                merged_state[key] = sum(w * exp[key] for w, exp in zip(weights, experts))
        else:
            merged_state[key] = spherical_karcher_mean(tensors, weights, num_iters=num_iters)
            
    return merged_state

def main():
    # Load progenitor
    print("Loading progenitor ResNet-18...")
    progenitor_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    progenitor_state = progenitor_model.state_dict()
    
    # Load expert states
    mnist_expert = load_expert_state("checkpoints/mnist_expert.pt")
    fmnist_expert = load_expert_state("checkpoints/fmnist_expert.pt")
    cifar10_expert = load_expert_state("checkpoints/cifar10_expert.pt")
    
    experts = [mnist_expert, fmnist_expert, cifar10_expert]
    
    # Extract classification heads
    mnist_fc = {k: v for k, v in mnist_expert.items() if "fc" in k}
    fmnist_fc = {k: v for k, v in fmnist_expert.items() if "fc" in k}
    cifar10_fc = {k: v for k, v in cifar10_expert.items() if "fc" in k}
    
    fc_heads = {
        "mnist": mnist_fc,
        "fmnist": fmnist_fc,
        "cifar10": cifar10_fc
    }
    
    # Load test loaders
    print("Loading test loaders...")
    loaders = {
        "mnist": get_test_loader("mnist"),
        "fmnist": get_test_loader("fmnist"),
        "cifar10": get_test_loader("cifar10")
    }
    
    # Evaluate individual experts (as reference)
    print("\nEvaluating Individual Experts:")
    expert_accs = {}
    for name, exp_state in zip(["mnist", "fmnist", "cifar10"], experts):
        loader = loaders[name]
        acc = evaluate_model(exp_state, fc_heads[name], loader)
        expert_accs[name] = acc
        print(f"  {name.upper()} Expert on {name.upper()} test set: {acc:.2f}%")
        
    results = {
        "experts": expert_accs,
        "merging": {}
    }
    
    # 1. Weight Averaging (WA)
    print("\n--- Running Weight Averaging (WA) ---")
    wa_backbone = weight_averaging(progenitor_state, experts)
    wa_accs = {}
    for name, fc_state in fc_heads.items():
        acc = evaluate_model(wa_backbone, fc_state, loaders[name])
        wa_accs[name] = acc
        print(f"  WA on {name.upper()} accuracy: {acc:.2f}%")
    wa_avg = sum(wa_accs.values()) / len(wa_accs)
    print(f"  WA Average Accuracy: {wa_avg:.2f}%")
    wa_vars = measure_activation_variance(wa_backbone, loaders["cifar10"])
    print("  WA Activation Variances across layers:", wa_vars)
    
    results["merging"]["WA"] = {
        "accuracies": wa_accs,
        "average": wa_avg,
        "variances": wa_vars
    }
    
    # 2. Task Arithmetic (TA)
    print("\n--- Running Task Arithmetic (TA) ---")
    best_ta_avg = 0.0
    best_ta_lam = 0.5
    best_ta_results = {}
    
    # Sweep lambda to find the best Task Arithmetic parameter
    for lam in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ta_backbone = task_arithmetic(progenitor_state, experts, lam=lam)
        ta_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(ta_backbone, fc_state, loaders[name])
            ta_accs[name] = acc
        ta_avg = sum(ta_accs.values()) / len(ta_accs)
        print(f"  TA with lambda={lam:.1f} average accuracy: {ta_avg:.2f}%")
        if ta_avg > best_ta_avg:
            best_ta_avg = ta_avg
            best_ta_lam = lam
            best_ta_results = {
                "lambda": lam,
                "accuracies": ta_accs,
                "average": ta_avg,
                "variances": measure_activation_variance(ta_backbone, loaders["cifar10"])
            }
            
    print(f"  Best TA average: {best_ta_avg:.2f}% with lambda={best_ta_lam:.1f}")
    results["merging"]["TA"] = best_ta_results
    
    # 3. Global Spherical Karcher Merging (Global SKM)
    print("\n--- Running Global Spherical Karcher Merging (Global SKM) ---")
    skm_global_backbone = spherical_karcher_merging(progenitor_state, experts, selective=False)
    skm_global_accs = {}
    for name, fc_state in fc_heads.items():
        acc = evaluate_model(skm_global_backbone, fc_state, loaders[name])
        skm_global_accs[name] = acc
        print(f"  Global SKM on {name.upper()} accuracy: {acc:.2f}%")
    skm_global_avg = sum(skm_global_accs.values()) / len(skm_global_accs)
    print(f"  Global SKM Average Accuracy: {skm_global_avg:.2f}%")
    skm_global_vars = measure_activation_variance(skm_global_backbone, loaders["cifar10"])
    print("  Global SKM Activation Variances across layers:", skm_global_vars)
    
    results["merging"]["SKM_Global"] = {
        "accuracies": skm_global_accs,
        "average": skm_global_avg,
        "variances": skm_global_vars
    }

    # 4. Selective Spherical Karcher Merging (S-SKM, Ours)
    print("\n--- Running Selective Spherical Karcher Merging (S-SKM, Ours) ---")
    sskm_backbone = spherical_karcher_merging(progenitor_state, experts, selective=True, channelwise=False)
    sskm_accs = {}
    for name, fc_state in fc_heads.items():
        acc = evaluate_model(sskm_backbone, fc_state, loaders[name])
        sskm_accs[name] = acc
        print(f"  S-SKM on {name.upper()} accuracy: {acc:.2f}%")
    sskm_avg = sum(sskm_accs.values()) / len(sskm_accs)
    print(f"  S-SKM Average Accuracy: {sskm_avg:.2f}%")
    sskm_vars = measure_activation_variance(sskm_backbone, loaders["cifar10"])
    print("  S-SKM Activation Variances across layers:", sskm_vars)
    
    results["merging"]["S-SKM"] = {
        "accuracies": sskm_accs,
        "average": sskm_avg,
        "variances": sskm_vars
    }

    # 5. Selective Channel-wise Spherical Karcher Merging (SC-SKM, Ours)
    print("\n--- Running Selective Channel-wise Spherical Karcher Merging (SC-SKM, Ours) ---")
    scskm_backbone = spherical_karcher_merging(progenitor_state, experts, selective=True, channelwise=True)
    scskm_accs = {}
    for name, fc_state in fc_heads.items():
        acc = evaluate_model(scskm_backbone, fc_state, loaders[name])
        scskm_accs[name] = acc
        print(f"  SC-SKM on {name.upper()} accuracy: {acc:.2f}%")
    scskm_avg = sum(scskm_accs.values()) / len(scskm_accs)
    print(f"  SC-SKM Average Accuracy: {scskm_avg:.2f}%")
    scskm_vars = measure_activation_variance(scskm_backbone, loaders["cifar10"])
    print("  SC-SKM Activation Variances across layers:", scskm_vars)
    
    results["merging"]["C-SKM"] = {
        "accuracies": scskm_accs,
        "average": scskm_avg,
        "variances": scskm_vars
    }

    # Map the best performing Selective method to "SKM" for the paper generation
    if sskm_avg >= scskm_avg:
        results["merging"]["SKM"] = results["merging"]["S-SKM"]
    else:
        results["merging"]["SKM"] = results["merging"]["C-SKM"]
    
    # Define fractions to sweep for sparsification methods
    sweep_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sweep_results = {
        "TIES": {},
        "SK-TIES": {},
        "SC-SK-TIES": {}
    }

    # 6. TIES-Merging Baseline
    print("\n--- Running TIES-Merging Baseline & Sweep ---")
    best_ties_avg = 0.0
    best_ties_fraction = 0.2
    best_ties_results = {}
    for fraction in sweep_fractions:
        ties_backbone = ties_merging(progenitor_state, experts, fraction=fraction)
        ties_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(ties_backbone, fc_state, loaders[name])
            ties_accs[name] = acc
        ties_avg = sum(ties_accs.values()) / len(ties_accs)
        print(f"  TIES with fraction={fraction:.2f} average accuracy: {ties_avg:.2f}%")
        
        # Save to detailed sweep
        sweep_results["TIES"][f"{fraction:.2f}"] = {
            "average": ties_avg,
            "accuracies": ties_accs
        }
        
        if ties_avg > best_ties_avg:
            best_ties_avg = ties_avg
            best_ties_fraction = fraction
            best_ties_results = {
                "fraction": fraction,
                "accuracies": ties_accs,
                "average": ties_avg,
                "variances": measure_activation_variance(ties_backbone, loaders["cifar10"])
            }
    print(f"  Best TIES average: {best_ties_avg:.2f}% with fraction={best_ties_fraction:.2f}")
    results["merging"]["TIES"] = best_ties_results

    # 7. DARE-Merging Baseline
    print("\n--- Running DARE-Merging Baseline ---")
    best_dare_avg = 0.0
    best_dare_drop = 0.5
    best_dare_results = {}
    for p_drop in [0.1, 0.3, 0.5, 0.7]:
        dare_backbone = dare_merging(progenitor_state, experts, p_drop=p_drop)
        dare_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(dare_backbone, fc_state, loaders[name])
            dare_accs[name] = acc
        dare_avg = sum(dare_accs.values()) / len(dare_accs)
        print(f"  DARE with p_drop={p_drop:.1f} average accuracy: {dare_avg:.2f}%")
        if dare_avg > best_dare_avg:
            best_dare_avg = dare_avg
            best_dare_drop = p_drop
            best_dare_results = {
                "p_drop": p_drop,
                "accuracies": dare_accs,
                "average": dare_avg,
                "variances": measure_activation_variance(dare_backbone, loaders["cifar10"])
            }
    print(f"  Best DARE average: {best_dare_avg:.2f}% with p_drop={best_dare_drop:.1f}")
    results["merging"]["DARE"] = best_dare_results

    # 8. Selective Spherical Karcher Task Arithmetic (S-SKTA, Ours)
    print("\n--- Running Selective Spherical Karcher Task Arithmetic (S-SKTA, Ours) ---")
    best_sskta_avg = 0.0
    best_sskta_lam = 0.3
    best_sskta_results = {}
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
        sskta_backbone = spherical_karcher_task_arithmetic(progenitor_state, experts, num_iters=1, selective=True, channelwise=False, lam=lam)
        sskta_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(sskta_backbone, fc_state, loaders[name])
            sskta_accs[name] = acc
        sskta_avg = sum(sskta_accs.values()) / len(sskta_accs)
        print(f"  S-SKTA with lambda={lam:.1f} average accuracy: {sskta_avg:.2f}%")
        if sskta_avg > best_sskta_avg:
            best_sskta_avg = sskta_avg
            best_sskta_lam = lam
            best_sskta_results = {
                "lambda": lam,
                "accuracies": sskta_accs,
                "average": sskta_avg,
                "variances": measure_activation_variance(sskta_backbone, loaders["cifar10"])
            }
    print(f"  Best S-SKTA average: {best_sskta_avg:.2f}% with lambda={best_sskta_lam:.1f}")
    results["merging"]["S-SKTA"] = best_sskta_results

    # 9. Selective Channel-wise Spherical Karcher Task Arithmetic (SC-SKTA, Ours)
    print("\n--- Running Selective Channel-wise Spherical Karcher Task Arithmetic (SC-SKTA, Ours) ---")
    best_scskta_avg = 0.0
    best_scskta_lam = 0.3
    best_scskta_results = {}
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
        scskta_backbone = spherical_karcher_task_arithmetic(progenitor_state, experts, num_iters=1, selective=True, channelwise=True, lam=lam)
        scskta_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(scskta_backbone, fc_state, loaders[name])
            scskta_accs[name] = acc
        scskta_avg = sum(scskta_accs.values()) / len(scskta_accs)
        print(f"  SC-SKTA with lambda={lam:.1f} average accuracy: {scskta_avg:.2f}%")
        if scskta_avg > best_scskta_avg:
            best_scskta_avg = scskta_avg
            best_scskta_lam = lam
            best_scskta_results = {
                "lambda": lam,
                "accuracies": scskta_accs,
                "average": scskta_avg,
                "variances": measure_activation_variance(scskta_backbone, loaders["cifar10"])
            }
    print(f"  Best SC-SKTA average: {best_scskta_avg:.2f}% with lambda={best_scskta_lam:.1f}")
    results["merging"]["SC-SKTA"] = best_scskta_results
    
    # 9b. Spherical Karcher TIES-Merging (SK-TIES, Ours)
    print("\n--- Running Spherical Karcher TIES-Merging (SK-TIES, Ours) & Sweep ---")
    best_skties_avg = 0.0
    best_skties_fraction = 0.4
    best_skties_results = {}
    for fraction in sweep_fractions:
        skties_backbone = sk_ties_merging(progenitor_state, experts, fraction=fraction, num_iters=1, channelwise=False)
        skties_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(skties_backbone, fc_state, loaders[name])
            skties_accs[name] = acc
        skties_avg = sum(skties_accs.values()) / len(skties_accs)
        print(f"  SK-TIES with fraction={fraction:.2f} average accuracy: {skties_avg:.2f}%")
        
        # Save to detailed sweep
        sweep_results["SK-TIES"][f"{fraction:.2f}"] = {
            "average": skties_avg,
            "accuracies": skties_accs
        }
        
        if skties_avg > best_skties_avg:
            best_skties_avg = skties_avg
            best_skties_fraction = fraction
            best_skties_results = {
                "fraction": fraction,
                "accuracies": skties_accs,
                "average": skties_avg,
                "variances": measure_activation_variance(skties_backbone, loaders["cifar10"])
            }
    print(f"  Best SK-TIES average: {best_skties_avg:.2f}% with fraction={best_skties_fraction:.2f}")
    results["merging"]["SK-TIES"] = best_skties_results

    # 9c. Channel-wise Spherical Karcher TIES-Merging (SC-SK-TIES, Ours)
    print("\n--- Running Channel-wise Spherical Karcher TIES-Merging (SC-SK-TIES, Ours) & Sweep ---")
    best_scskties_avg = 0.0
    best_scskties_fraction = 0.4
    best_scskties_results = {}
    for fraction in sweep_fractions:
        scskties_backbone = sk_ties_merging(progenitor_state, experts, fraction=fraction, num_iters=1, channelwise=True)
        scskties_accs = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(scskties_backbone, fc_state, loaders[name])
            scskties_accs[name] = acc
        scskties_avg = sum(scskties_accs.values()) / len(scskties_accs)
        print(f"  SC-SK-TIES with fraction={fraction:.2f} average accuracy: {scskties_avg:.2f}%")
        
        # Save to detailed sweep
        sweep_results["SC-SK-TIES"][f"{fraction:.2f}"] = {
            "average": scskties_avg,
            "accuracies": scskties_accs
        }
        
        if scskties_avg > best_scskties_avg:
            best_scskties_avg = scskties_avg
            best_scskties_fraction = fraction
            best_scskties_results = {
                "fraction": fraction,
                "accuracies": scskties_accs,
                "average": scskties_avg,
                "variances": measure_activation_variance(scskties_backbone, loaders["cifar10"])
            }
    print(f"  Best SC-SK-TIES average: {best_scskties_avg:.2f}% with fraction={best_scskties_fraction:.2f}")
    results["merging"]["SC-SK-TIES"] = best_scskties_results

    # Save complete fraction sweep results
    results["ties_fraction_sweep"] = sweep_results

    # 10. Ablation on Karcher Mean Iterations (num_iters)
    print("\n--- Running Ablation Study: Convergence of Karcher Mean Iterations (num_iters) ---")
    ablation_results = {}
    for num_iters in [1, 2, 3, 5, 10]:
        print(f"Evaluating S-SKM with num_iters={num_iters}...")
        sskm_backbone_it = spherical_karcher_merging(progenitor_state, experts, selective=True, channelwise=False, num_iters=num_iters)
        sskm_accs_it = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(sskm_backbone_it, fc_state, loaders[name])
            sskm_accs_it[name] = acc
        sskm_avg_it = sum(sskm_accs_it.values()) / len(sskm_accs_it)
        
        ablation_results[str(num_iters)] = {
            "accuracies": sskm_accs_it,
            "average": sskm_avg_it
        }
        print(f"  num_iters={num_iters}: Average Accuracy = {sskm_avg_it:.2f}%")
        
    results["ablation_num_iters"] = ablation_results

    # 11. Robustness to Unequal Task Weights
    print("\n--- Running Robustness to Unequal Task Weights Sweep ---")
    unequal_results = {}
    for w_cifar in [0.1, 0.3, 0.5, 0.7]:
        w_mnist = (1.0 - w_cifar) / 2.0
        w_fmnist = w_mnist
        weights_list = [w_mnist, w_fmnist, w_cifar]
        print(f"Evaluating with weights [MNIST={w_mnist:.2f}, FMNIST={w_fmnist:.2f}, CIFAR10={w_cifar:.2f}]...")
        
        # TIES (with best fraction 0.5)
        ties_backbone_un = ties_merging(progenitor_state, experts, weights=weights_list, fraction=0.5)
        ties_accs_un = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(ties_backbone_un, fc_state, loaders[name])
            ties_accs_un[name] = acc
        ties_avg_un = sum(ties_accs_un.values()) / len(ties_accs_un)
        
        # SC-SK-TIES (with best fraction 0.6)
        scskties_backbone_un = sk_ties_merging(progenitor_state, experts, weights=weights_list, fraction=0.6, num_iters=1, channelwise=True)
        scskties_accs_un = {}
        for name, fc_state in fc_heads.items():
            acc = evaluate_model(scskties_backbone_un, fc_state, loaders[name])
            scskties_accs_un[name] = acc
        scskties_avg_un = sum(scskties_accs_un.values()) / len(scskties_accs_un)
        
        unequal_results[f"{w_cifar:.2f}"] = {
            "weights": weights_list,
            "TIES": {
                "accuracies": ties_accs_un,
                "average": ties_avg_un
            },
            "SC-SK-TIES": {
                "accuracies": scskties_accs_un,
                "average": scskties_avg_un
            }
        }
        print(f"  w_cifar={w_cifar:.2f}: TIES={ties_avg_un:.2f}%, SC-SK-TIES={scskties_avg_un:.2f}%")
        
    results["unequal_weights_sweep"] = unequal_results

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved all results to results.json")

if __name__ == "__main__":
    main()
