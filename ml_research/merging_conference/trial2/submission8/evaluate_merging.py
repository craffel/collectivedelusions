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
        # Gaussian Noise: adding random Gaussian noise (sigma=0.4) to pixel tensors
        noise = 0.4 * torch.randn_like(images)
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == "blur":
        # Gaussian Blur: spatial blur with kernel size of 5x5 and standard deviation 1.5
        return TF.gaussian_blur(images, kernel_size=[5, 5], sigma=[1.5, 1.5])
    elif corruption_type == "contrast":
        # Contrast Reduction: scaling pixel contrast by factor of 0.25
        return images * 0.25
    elif corruption_type == "rotation":
        # Image Rotation: rotating images by 30 degrees
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
            
            # Reconstruct the merged model parameter dictionary for this task
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load original pretrained base model
    print("Loading base model...")
    base_model = get_resnet18_model().to(device)
    base_state = torch.load("models/pretrained_base.pth", map_location=device)
    
    # Load expert weights
    print("Loading experts...")
    mnist_state = torch.load("models/mnist_expert.pth", map_location=device)
    fashion_state = torch.load("models/fashionmnist_expert.pth", map_location=device)
    kmnist_state = torch.load("models/kmnist_expert.pth", map_location=device)
    
    # Compute task vectors
    tau_mnist = {}
    tau_fashion = {}
    tau_kmnist = {}
    for k in base_state.keys():
        tau_mnist[k] = mnist_state[k] - base_state[k]
        tau_fashion[k] = fashion_state[k] - base_state[k]
        tau_kmnist[k] = kmnist_state[k] - base_state[k]
        
    # Original classification heads
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
    
    # Expert state dicts for soft labels
    expert_state_dicts = {
        "mnist": mnist_state,
        "fashionmnist": fashion_state,
        "kmnist": kmnist_state
    }
    
    # Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    print("Loading test datasets...")
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    # Create TTA subsets (first 512 images)
    mnist_tta = Subset(mnist_test, list(range(512)))
    fashion_tta = Subset(fashion_test, list(range(512)))
    kmnist_tta = Subset(kmnist_test, list(range(512)))
    
    # Create full evaluation sets (remaining or full)
    test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=4),
        "fashionmnist": DataLoader(fashion_test, batch_size=128, shuffle=False, num_workers=4),
        "kmnist": DataLoader(kmnist_test, batch_size=128, shuffle=False, num_workers=4)
    }
    
    environments = ["clean", "noise", "blur", "contrast", "rotation"]
    methods = ["ta", "adamerging", "symerge", "sat-symerge", "asam-symerge", "fisher-sat-symerge"]
    
    results = {m: {} for m in methods}
    
    for env in environments:
        print(f"\n================ ENVIRONMENT: {env.upper()} ================")
        
        for method in methods:
            print(f"Running TTA method: {method.upper()} on {env}")
            
            # Re-initialize TTA loaders with batch_size=32 to draw batches step-by-step
            tta_loaders = {
                "mnist": DataLoader(mnist_tta, batch_size=32, shuffle=False),
                "fashionmnist": DataLoader(fashion_tta, batch_size=32, shuffle=False),
                "kmnist": DataLoader(kmnist_tta, batch_size=32, shuffle=False)
            }
            
            # Setup iterators
            iterators = {task: iter(loader) for task, loader in tta_loaders.items()}
            
            # Initialize merging coefficients and heads
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
            
            if method == "ta":
                # Static merging - no adaptation
                accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
                results[method][env] = accuracies
                print(f"Accuracy: {accuracies['avg']:.2f}% (MNIST: {accuracies['mnist']:.2f}%, Fashion: {accuracies['fashionmnist']:.2f}%, KMNIST: {accuracies['kmnist']:.2f}%)")
                continue
                
            elif method == "adamerging":
                optimizer = optim.Adam([lambdas], lr=0.001)
                for step in range(10):
                    optimizer.zero_grad()
                    loss = 0.0
                    for task_name in tta_loaders.keys():
                        images, _ = next(iterators[task_name])
                        images = apply_corruption(images, env).to(device)
                        
                        # Reconstruct model state
                        params_dict = reconstruct_merged_state(base_state, tau_mnist, tau_fashion, tau_kmnist, lambdas, adapted_heads[task_name])
                        
                        outputs = functional_call(base_model, params_dict, images)
                        probs = torch.softmax(outputs, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                        loss += entropy
                    
                    loss = loss / len(tta_loaders)
                    loss.backward()
                    optimizer.step()
                    
                accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
                results[method][env] = accuracies
                print(f"Accuracy: {accuracies['avg']:.2f}% (MNIST: {accuracies['mnist']:.2f}%, Fashion: {accuracies['fashionmnist']:.2f}%, KMNIST: {accuracies['kmnist']:.2f}%)")
                
            elif method == "symerge":
                optimizer = optim.Adam([
                    {"params": [lambdas], "lr": 0.001},
                    {"params": [adapted_heads["mnist"]["weight"], adapted_heads["mnist"]["bias"],
                               adapted_heads["fashionmnist"]["weight"], adapted_heads["fashionmnist"]["bias"],
                               adapted_heads["kmnist"]["weight"], adapted_heads["kmnist"]["bias"]], "lr": 0.01}
                ])
                
                for step in range(10):
                    optimizer.zero_grad()
                    loss = 0.0
                    for task_name in tta_loaders.keys():
                        images, _ = next(iterators[task_name])
                        images = apply_corruption(images, env).to(device)
                        
                        with torch.no_grad():
                            expert_outputs = functional_call(base_model, expert_state_dicts[task_name], images)
                            p_expert = torch.softmax(expert_outputs, dim=-1)
                            
                        params_dict = reconstruct_merged_state(base_state, tau_mnist, tau_fashion, tau_kmnist, lambdas, adapted_heads[task_name])
                        
                        outputs = functional_call(base_model, params_dict, images)
                        p_merged = torch.softmax(outputs, dim=-1)
                        kl = torch.sum(p_expert * (torch.log(p_expert + 1e-8) - torch.log(p_merged + 1e-8)), dim=-1).mean()
                        loss += kl
                        
                    loss = loss / len(tta_loaders)
                    loss.backward()
                    optimizer.step()
                    
                accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
                results[method][env] = accuracies
                print(f"Accuracy: {accuracies['avg']:.2f}% (MNIST: {accuracies['mnist']:.2f}%, Fashion: {accuracies['fashionmnist']:.2f}%, KMNIST: {accuracies['kmnist']:.2f}%)")
                
            elif method == "sat-symerge":
                # Sharpness-Aware Test-Time Synergistic Merging
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
                                 
                rho = 0.02 # Tensor-Wise optimal perturbation scale from Submission 1
                
                for step in range(10):
                    optimizer.zero_grad()
                    
                    # Fetch batches once to keep aligned
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
                        
                    # First forward-backward pass
                    loss = compute_loss_on_batches(lambdas, adapted_heads)
                    loss.backward()
                    
                    # Save gradients and compute norm
                    grads = [p.grad.clone() if p.grad is not None else None for p in active_params]
                    g_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
                    
                    # Apply perturbation
                    for p, g in zip(active_params, grads):
                        if g is not None:
                            p.data.add_(rho * g / (g_norm + 1e-8))
                            
                    # Second forward-backward pass
                    optimizer.zero_grad()
                    loss_perturbed = compute_loss_on_batches(lambdas, adapted_heads)
                    loss_perturbed.backward()
                    
                    # Restore original parameters
                    for p, g in zip(active_params, grads):
                        if g is not None:
                            p.data.sub_(rho * g / (g_norm + 1e-8))
                            
                    optimizer.step()
                    
                accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
                results[method][env] = accuracies
                print(f"Accuracy: {accuracies['avg']:.2f}% (MNIST: {accuracies['mnist']:.2f}%, Fashion: {accuracies['fashionmnist']:.2f}%, KMNIST: {accuracies['kmnist']:.2f}%)")
                
            elif method == "asam-symerge":
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
                                 
                rho = 0.05
                eta = 0.01
                
                for step in range(10):
                    optimizer.zero_grad()
                    
                    # Fetch batches once to keep aligned
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
                    
                    # ASAM norm
                    denom = 0.0
                    for p, g in zip(active_params, grads):
                        if g is not None:
                            denom += ((p.data.abs() + eta).pow(2) * g.pow(2)).sum()
                    denom = torch.sqrt(denom)
                    
                    # Apply ASAM perturbation
                    for p, g in zip(active_params, grads):
                        if g is not None:
                            epsilon = rho * (p.data.abs() + eta).pow(2) * g / (denom + 1e-8)
                            p.data.add_(epsilon)
                            
                    optimizer.zero_grad()
                    loss_perturbed = compute_loss_on_batches(lambdas, adapted_heads)
                    loss_perturbed.backward()
                    
                    # Restore original parameters
                    for p, g in zip(active_params, grads):
                        if g is not None:
                            epsilon = rho * (p.data.abs() + eta).pow(2) * g / (denom + 1e-8)
                            p.data.sub_(epsilon)
                            
                    optimizer.step()
                    
                accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
                results[method][env] = accuracies
                print(f"Accuracy: {accuracies['avg']:.2f}% (MNIST: {accuracies['mnist']:.2f}%, Fashion: {accuracies['fashionmnist']:.2f}%, KMNIST: {accuracies['kmnist']:.2f}%)")
                
            elif method == "fisher-sat-symerge":
                # Our Proposed Method: Soft-Bounded Fisher-Guided SAT-SyMerge (SBF-SAT-SyMerge)
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
                                 
                rho = 0.05
                fisher_estimates = None
                
                for step in range(10):
                    optimizer.zero_grad()
                    
                    # Fetch batches once to keep aligned
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
                    
                    # Update running Fisher estimates
                    if fisher_estimates is None:
                        fisher_estimates = [g.pow(2) if g is not None else None for g in grads]
                    else:
                        for idx, g in enumerate(grads):
                            if g is not None:
                                fisher_estimates[idx] = 0.9 * fisher_estimates[idx] + 0.1 * g.pow(2)
                                
                    # Compute average running Fisher across all active parameters
                    all_f = []
                    for f in fisher_estimates:
                        if f is not None:
                            all_f.append(f.view(-1))
                    mean_f = torch.cat(all_f).mean().item() if len(all_f) > 0 else 0.0
                    
                    # Soft-Bounded Fisher scale: t = exp(- f / (mean_f + 1e-8))
                    denom = 0.0
                    for g, f in zip(grads, fisher_estimates):
                        if g is not None and f is not None:
                            t = torch.exp(-f / (mean_f + 1e-8))
                            denom += (t.pow(2) * g.pow(2)).sum()
                    denom = torch.sqrt(denom)
                    
                    # Apply Soft-Bounded Fisher perturbation
                    for p, g, f in zip(active_params, grads, fisher_estimates):
                        if g is not None and f is not None:
                            t = torch.exp(-f / (mean_f + 1e-8))
                            epsilon = rho * t.pow(2) * g / (denom + 1e-8)
                            p.data.add_(epsilon)
                            
                    optimizer.zero_grad()
                    loss_perturbed = compute_loss_on_batches(lambdas, adapted_heads)
                    loss_perturbed.backward()
                    
                    # Restore original parameters (subtract epsilon)
                    for p, g, f in zip(active_params, grads, fisher_estimates):
                        if g is not None and f is not None:
                            t = torch.exp(-f / (mean_f + 1e-8))
                            epsilon = rho * t.pow(2) * g / (denom + 1e-8)
                            p.data.sub_(epsilon)
                            
                    optimizer.step()
                    
                accuracies = evaluate_all(lambdas, adapted_heads, test_loaders, env, base_model, device, base_state, tau_mnist, tau_fashion, tau_kmnist)
                results[method][env] = accuracies
                print(f"Accuracy: {accuracies['avg']:.2f}% (MNIST: {accuracies['mnist']:.2f}%, Fashion: {accuracies['fashionmnist']:.2f}%, KMNIST: {accuracies['kmnist']:.2f}%)")
                
    print("\n\n================ FINAL RESULTS SUMMARY ================")
    # Print beautiful markdown table of results
    print("| Method | Clean | Noise | Blur | Contrast | Rotation | OOD Average |")
    print("|---|---|---|---|---|---|---|")
    for method in methods:
        clean_acc = results[method]["clean"]["avg"]
        noise_acc = results[method]["noise"]["avg"]
        blur_acc = results[method]["blur"]["avg"]
        contrast_acc = results[method]["contrast"]["avg"]
        rotation_acc = results[method]["rotation"]["avg"]
        
        ood_avg = np.mean([noise_acc, blur_acc, contrast_acc, rotation_acc])
        print(f"| {method.upper()} | {clean_acc:.2f}% | {noise_acc:.2f}% | {blur_acc:.2f}% | {contrast_acc:.2f}% | {rotation_acc:.2f}% | {ood_avg:.2f}% |")

if __name__ == "__main__":
    main()
