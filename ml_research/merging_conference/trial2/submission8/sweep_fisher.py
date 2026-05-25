import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.nn.utils.stateless import functional_call
import copy
import torchvision.transforms.functional as TF

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
torch.backends.cudnn.enabled = False

def get_resnet18_model():
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def apply_corruption(images, corruption_type):
    if corruption_type == "clean":
        return images
    elif corruption_type == "noise":
        noise = 0.4 * torch.randn_like(images)
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == "blur":
        return TF.gaussian_blur(images, kernel_size=[5, 5], sigma=[1.5, 1.5])
    elif corruption_type == "contrast":
        return images * 0.25
    elif corruption_type == "rotation":
        return TF.rotate(images, angle=30)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")

def reconstruct_merged_state(base_state, tau_mnist, tau_fashion, tau_kmnist, lambdas, task_head):
    params_dict = {}
    for k in base_state.keys():
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            with torch.no_grad():
                params_dict[k] = (base_state[k] + lambdas[0] * tau_mnist[k] + lambdas[1] * tau_fashion[k] + lambdas[2] * tau_kmnist[k]).detach()
        else:
            params_dict[k] = base_state[k] + lambdas[0] * tau_mnist[k] + lambdas[1] * tau_fashion[k] + lambdas[2] * tau_kmnist[k]
    params_dict["fc.weight"] = task_head["weight"]
    params_dict["fc.bias"] = task_head["bias"]
    return params_dict

def evaluate_all(lambdas, heads, test_loaders, corruption_type, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist):
    base_model.eval()
    accuracies = {}
    
    with torch.no_grad():
        for task_name, loader in test_loaders.items():
            correct = 0
            total = 0
            
            params_dict = reconstruct_merged_state(base_state, tau_mnist, tau_fashion, tau_kmnist, lambdas, heads[task_name])
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                images = apply_corruption(images, corruption_type)
                
                outputs = functional_call(base_model, params_dict, images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            accuracies[task_name] = 100.0 * correct / total
            
    accuracies["avg"] = np.mean([accuracies["mnist"], accuracies["fashionmnist"], accuracies["kmnist"]])
    return accuracies

def run_variant(variant_cfg, base_model, base_state, tau_mnist, tau_fashion, tau_kmnist, heads, expert_state_dicts, test_loaders, mnist_tta, fashion_tta, kmnist_tta, device):
    env_results = {}
    environments = ["clean", "noise", "blur", "contrast", "rotation"]
    
    rho = variant_cfg["rho"]
    mode = variant_cfg["mode"]  # "original_sbf", "true_fisher", "scale_invariant_exp", "asam_like_exp", "power_law"
    norm_scope = variant_cfg["norm_scope"]  # "global" or "tensor"
    param_cfg = variant_cfg["param_cfg"]  # value for eta, gamma, or alpha
    
    for env in environments:
        # Re-initialize TTA loaders
        tta_loaders = {
            "mnist": DataLoader(mnist_tta, batch_size=32, shuffle=False),
            "fashionmnist": DataLoader(fashion_tta, batch_size=32, shuffle=False),
            "kmnist": DataLoader(kmnist_tta, batch_size=32, shuffle=False)
        }
        iterators = {task: iter(loader) for task, loader in tta_loaders.items()}
        
        lambdas = torch.tensor([0.3, 0.3, 0.3], device=device, requires_grad=True)
        adapted_heads = {
            "mnist": {
                "weight": heads["mnist"]["weight"].clone().detach().requires_grad_(True),
                "bias": heads["mnist"]["bias"].clone().detach().requires_grad_(True)
            },
            "fashionmnist": {
                "weight": heads["fashionmnist"]["weight"].clone().detach().requires_grad_(True),
                "bias": heads["fashionmnist"]["bias"].clone().detach().requires_grad_(True)
            },
            "kmnist": {
                "weight": heads["kmnist"]["weight"].clone().detach().requires_grad_(True),
                "bias": heads["kmnist"]["bias"].clone().detach().requires_grad_(True)
            }
        }
        
        optimizer = optim.Adam([
            {"params": [lambdas], "lr": 0.001},
            {"params": [adapted_heads["mnist"]["weight"], adapted_heads["mnist"]["bias"],
                       adapted_heads["fashionmnist"]["weight"], adapted_heads["fashionmnist"]["bias"],
                       adapted_heads["kmnist"]["weight"], adapted_heads["kmnist"]["bias"]], "lr": 0.01}
        ])
        
        active_params = [lambdas,
                         adapted_heads["mnist"]["weight"], adapted_heads["mnist"]["bias"],
                         adapted_heads["fashionmnist"]["weight"], adapted_heads["fashionmnist"]["bias"],
                         adapted_heads["kmnist"]["weight"], adapted_heads["kmnist"]["bias"]]
                         
        fisher_estimates = None
        
        for step in range(10):
            optimizer.zero_grad()
            batches = {}
            for task_name in tta_loaders.keys():
                images, _ = next(iterators[task_name])
                images = apply_corruption(images, env).to(device)
                batches[task_name] = images
                
            def compute_loss_on_batches(curr_lambdas, curr_heads):
                total_kl = 0.0
                for task_name in tta_loaders.keys():
                    images = batches[task_name]
                    with torch.no_grad():
                        expert_outputs = functional_call(base_model, expert_state_dicts[task_name], images)
                        p_expert = torch.softmax(expert_outputs, dim=-1)
                        
                    params_dict = reconstruct_merged_state(base_state, tau_mnist, tau_fashion, tau_kmnist, curr_lambdas, curr_heads[task_name])
                    outputs = functional_call(base_model, params_dict, images)
                    p_merged = torch.softmax(outputs, dim=-1)
                    kl = torch.sum(p_expert * (torch.log(p_expert + 1e-8) - torch.log(p_merged + 1e-8)), dim=-1).mean()
                    total_kl += kl
                return total_kl / len(tta_loaders)
                
            loss = compute_loss_on_batches(lambdas, adapted_heads)
            loss.backward()
            
            grads = [p.grad.clone() if p.grad is not None else None for p in active_params]
            
            if fisher_estimates is None:
                fisher_estimates = [g.pow(2) if g is not None else None for g in grads]
            else:
                for idx, g in enumerate(grads):
                    if g is not None:
                        fisher_estimates[idx] = 0.9 * fisher_estimates[idx] + 0.1 * g.pow(2)
                        
            # Compute mean Fisher based on scope
            if norm_scope == "global":
                all_f = []
                for f in fisher_estimates:
                    if f is not None:
                        all_f.append(f.view(-1))
                mean_f = torch.cat(all_f).mean().item() if len(all_f) > 0 else 0.0
                mean_f_list = [mean_f] * len(active_params)
            else:
                # Per-tensor normalization
                mean_f_list = [f.mean().item() if f is not None else 0.0 for f in fisher_estimates]
                
            # Compute scaling multipliers (d or t)
            scalers = []
            for f, m_f in zip(fisher_estimates, mean_f_list):
                if f is None:
                    scalers.append(None)
                    continue
                norm_f = f / (m_f + 1e-8)
                
                if mode == "original_sbf":
                    t = torch.exp(-norm_f)
                    scalers.append(t)
                elif mode == "true_fisher":
                    eta = param_cfg
                    d = 1.0 / (norm_f + eta)
                    scalers.append(d)
                elif mode == "scale_invariant_exp":
                    gamma = param_cfg
                    d = torch.exp(-gamma * norm_f)
                    scalers.append(d)
                elif mode == "asam_like_exp":
                    gamma = param_cfg
                    d = torch.exp(-gamma * norm_f)
                    scalers.append(d)
                elif mode == "power_law":
                    alpha = param_cfg
                    d = 1.0 / ((norm_f + 0.01) ** alpha)
                    scalers.append(d)
                    
            # Compute denominator and apply perturbation
            denom = 0.0
            for g, s in zip(grads, scalers):
                if g is not None and s is not None:
                    if mode == "original_sbf" or mode == "asam_like_exp":
                        denom += (s.pow(2) * g.pow(2)).sum()
                    else:
                        denom += (s.pow(2) * g.pow(2)).sum()
            denom = torch.sqrt(denom)
            
            for p, g, s in zip(active_params, grads, scalers):
                if g is not None and s is not None:
                    if mode == "original_sbf" or mode == "asam_like_exp":
                        epsilon = rho * s.pow(2) * g / (denom + 1e-8)
                    else:
                        # True Fisher constraint (s * g / denom)
                        epsilon = rho * s * g / (denom + 1e-8)
                    p.data.add_(epsilon)
                    
            optimizer.zero_grad()
            loss_perturbed = compute_loss_on_batches(lambdas, adapted_heads)
            loss_perturbed.backward()
            
            # Restore
            for p, g, s in zip(active_params, grads, scalers):
                if g is not None and s is not None:
                    if mode == "original_sbf" or mode == "asam_like_exp":
                        epsilon = rho * s.pow(2) * g / (denom + 1e-8)
                    else:
                        epsilon = rho * s * g / (denom + 1e-8)
                    p.data.sub_(epsilon)
                    
            optimizer.step()
            
        accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
        env_results[env] = accuracies["avg"]
        
    env_results["ood_avg"] = np.mean([env_results["noise"], env_results["blur"], env_results["contrast"], env_results["rotation"]])
    return env_results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_model = get_resnet18_model().to(device)
    base_state = torch.load("models/pretrained_base.pth", map_location=device)
    
    mnist_state = torch.load("models/mnist_expert.pth", map_location=device)
    fashion_state = torch.load("models/fashionmnist_expert.pth", map_location=device)
    kmnist_state = torch.load("models/kmnist_expert.pth", map_location=device)
    
    tau_mnist = {}
    tau_fashion = {}
    tau_kmnist = {}
    for k in base_state.keys():
        tau_mnist[k] = mnist_state[k] - base_state[k]
        tau_fashion[k] = fashion_state[k] - base_state[k]
        tau_kmnist[k] = kmnist_state[k] - base_state[k]
        
    heads = {
        "mnist": {
            "weight": mnist_state["fc.weight"].clone(),
            "bias": mnist_state["fc.bias"].clone()
        },
        "fashionmnist": {
            "weight": fashion_state["fc.weight"].clone(),
            "bias": fashion_state["fc.bias"].clone()
        },
        "kmnist": {
            "weight": kmnist_state["fc.weight"].clone(),
            "bias": kmnist_state["fc.bias"].clone()
        }
    }
    
    expert_state_dicts = {
        "mnist": mnist_state,
        "fashionmnist": fashion_state,
        "kmnist": kmnist_state
    }
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    mnist_tta = Subset(mnist_test, list(range(512)))
    fashion_tta = Subset(fashion_test, list(range(512)))
    kmnist_tta = Subset(kmnist_test, list(range(512)))
    
    test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=4),
        "fashionmnist": DataLoader(fashion_test, batch_size=128, shuffle=False, num_workers=4),
        "kmnist": DataLoader(kmnist_test, batch_size=128, shuffle=False, num_workers=4)
    }
    
    variants = [
        {"name": "Original SBF", "rho": 0.05, "mode": "original_sbf", "norm_scope": "global", "param_cfg": None},
        {"name": "Original SBF Per-Tensor", "rho": 0.05, "mode": "original_sbf", "norm_scope": "tensor", "param_cfg": None},
        
        # True Fisher with eta
        {"name": "True Fisher (eta=0.01) Global", "rho": 0.02, "mode": "true_fisher", "norm_scope": "global", "param_cfg": 0.01},
        {"name": "True Fisher (eta=0.1) Global", "rho": 0.02, "mode": "true_fisher", "norm_scope": "global", "param_cfg": 0.1},
        {"name": "True Fisher (eta=1.0) Global", "rho": 0.02, "mode": "true_fisher", "norm_scope": "global", "param_cfg": 1.0},
        {"name": "True Fisher (eta=10.0) Global", "rho": 0.02, "mode": "true_fisher", "norm_scope": "global", "param_cfg": 10.0},
        
        {"name": "True Fisher (eta=0.01) Tensor", "rho": 0.02, "mode": "true_fisher", "norm_scope": "tensor", "param_cfg": 0.01},
        {"name": "True Fisher (eta=0.1) Tensor", "rho": 0.02, "mode": "true_fisher", "norm_scope": "tensor", "param_cfg": 0.1},
        {"name": "True Fisher (eta=1.0) Tensor", "rho": 0.02, "mode": "true_fisher", "norm_scope": "tensor", "param_cfg": 1.0},
        {"name": "True Fisher (eta=10.0) Tensor", "rho": 0.02, "mode": "true_fisher", "norm_scope": "tensor", "param_cfg": 10.0},
        
        # Scale Invariant Exp with gamma
        {"name": "SI-Exp (gamma=0.1) Tensor", "rho": 0.02, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 0.1},
        {"name": "SI-Exp (gamma=0.5) Tensor", "rho": 0.02, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 0.5},
        {"name": "SI-Exp (gamma=1.0) Tensor", "rho": 0.02, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 1.0},
        {"name": "SI-Exp (gamma=2.0) Tensor", "rho": 0.02, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 2.0},
        
        {"name": "SI-Exp (gamma=0.1) Tensor (rho=0.05)", "rho": 0.05, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 0.1},
        {"name": "SI-Exp (gamma=0.5) Tensor (rho=0.05)", "rho": 0.05, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 0.5},
        {"name": "SI-Exp (gamma=1.0) Tensor (rho=0.05)", "rho": 0.05, "mode": "scale_invariant_exp", "norm_scope": "tensor", "param_cfg": 1.0},
        
        # Power Law with alpha
        {"name": "Power Law (alpha=0.1) Tensor", "rho": 0.02, "mode": "power_law", "norm_scope": "tensor", "param_cfg": 0.1},
        {"name": "Power Law (alpha=0.25) Tensor", "rho": 0.02, "mode": "power_law", "norm_scope": "tensor", "param_cfg": 0.25},
        {"name": "Power Law (alpha=0.5) Tensor", "rho": 0.02, "mode": "power_law", "norm_scope": "tensor", "param_cfg": 0.5},
    ]
    
    print("Starting hyperparameter and formulation sweep...")
    print(f"{'Variant Name':<40} | {'Clean':<7} | {'Noise':<7} | {'Blur':<7} | {'Contrast':<7} | {'Rotation':<7} | {'OOD Avg':<7}")
    print("-" * 105)
    
    for v in variants:
        res = run_variant(v, base_model, base_state, tau_mnist, tau_fashion, tau_kmnist, heads, expert_state_dicts, test_loaders, mnist_tta, fashion_tta, kmnist_tta, device)
        print(f"{v['name']:<40} | {res['clean']:.2f}% | {res['noise']:.2f}% | {res['blur']:.2f}% | {res['contrast']:.2f}% | {res['rotation']:.2f}% | {res['ood_avg']:.2f}%")

if __name__ == "__main__":
    main()
