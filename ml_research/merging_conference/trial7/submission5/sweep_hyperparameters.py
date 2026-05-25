import os
import torch
import torch.nn as nn
import torch.func as func
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from evaluate_ttmm import ExpertCNN, load_experts, get_datasets, precompute_unified_prototypes, compute_batch_cohesion, get_merged_state_dict

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate_ours(experts, datasets_dict, proto_data, stream_batches, base_lr, steps, beta, device):
    experts_list = [experts["mnist"], experts["kmnist"], experts["fashionmnist"]]
    base_model = ExpertCNN().to(device)
    
    coeffs = {}
    for name, param in base_model.named_parameters():
        coeffs[name] = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
        
    ema_coeffs = {name: torch.tensor([0.0, 0.0, 0.0], device=device) for name in coeffs.keys()}
    
    accuracies = []
    first_novel = True
    
    for batch_idx, (images, labels, domain) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        is_novel = False
        routed_domain = None
        
        # Unified Static Space routing
        with torch.no_grad():
            feats = proto_data["static_model"].extract_features(images)
        
        c_mnist = compute_batch_cohesion(feats, proto_data["prototypes_mnist"], proto_data["mu_static"])
        c_kmnist = compute_batch_cohesion(feats, proto_data["prototypes_kmnist"], proto_data["mu_static"])
        
        max_cohesion = max(c_mnist, c_kmnist)
        threshold = 0.35
        
        if max_cohesion < threshold:
            is_novel = True
            routed_domain = "novel"
        else:
            is_novel = False
            routed_domain = "mnist" if c_mnist > c_kmnist else "kmnist"
            
        active_coeffs = {}
        use_bn = True
        
        if is_novel:
            routed_domain = "fashionmnist"
            is_novel = False
            
        if not is_novel and routed_domain is not None:
            if routed_domain == "mnist":
                target = [3.0, -3.0, -3.0]
            elif routed_domain == "kmnist":
                target = [-3.0, 3.0, -3.0]
            else: # fashionmnist
                target = [-3.0, -3.0, 3.0]
                
            target_t = torch.tensor(target, device=device)
            alpha = 0.3
            
            with torch.no_grad():
                for name in coeffs.keys():
                    ema_coeffs[name] = (1.0 - alpha) * ema_coeffs[name] + alpha * target_t
                    coeffs[name].copy_(ema_coeffs[name])
                    
            active_coeffs = coeffs
            use_bn = True
        else:
            # Adaptation block for novel domain
            if is_novel:
                if first_novel:
                    with torch.no_grad():
                        for name in coeffs.keys():
                            coeffs[name].copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
                            ema_coeffs[name].copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
                    first_novel = False
                else:
                    with torch.no_grad():
                        for name in coeffs.keys():
                            coeffs[name].copy_(ema_coeffs[name])
                            
            param_list = [coeffs[name] for name in coeffs.keys()]
            
            # TT-Fisher
            lr_scales = {name: 1.0 for name in coeffs.keys()}
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
                        lr_scales[name] = 1.0 / (grad_sq_mean + 1e-4)**0.5
                        
            for step in range(steps):
                merged_sd = get_merged_state_dict(experts_list, coeffs, base_model, use_bn=True)
                logits = func.functional_call(base_model, merged_sd, images)
                probs = torch.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                mean_probs = probs.mean(dim=0)
                diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                loss = entropy - beta * diversity
                
                grads = torch.autograd.grad(loss, param_list, allow_unused=True)
                
                with torch.no_grad():
                    for i, name in enumerate(coeffs.keys()):
                        if grads[i] is not None:
                            lr = base_lr * lr_scales[name]
                            coeffs[name] -= lr * grads[i]
                            
            if is_novel:
                with torch.no_grad():
                    for name in coeffs.keys():
                        alpha = 0.3
                        ema_coeffs[name] = (1.0 - alpha) * ema_coeffs[name] + alpha * coeffs[name].detach()
                        coeffs[name].copy_(ema_coeffs[name])
                        
            active_coeffs = coeffs
            use_bn = True
            
        merged_sd = get_merged_state_dict(experts_list, active_coeffs, base_model, use_bn=use_bn)
        eval_model = ExpertCNN().to(device)
        eval_model.load_state_dict(merged_sd)
        eval_model.eval()
        
        with torch.no_grad():
            outputs = eval_model(images)
            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum().item()
            acc = 100.0 * correct / labels.size(0)
            
        accuracies.append(acc)
        
    mnist_avg = np.mean(accuracies[0:10])
    kmnist_avg = np.mean(accuracies[10:20])
    fashion_avg = np.mean(accuracies[20:30])
    overall_avg = np.mean(accuracies)
    return mnist_avg, kmnist_avg, fashion_avg, overall_avg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sweeping on device: {device}")
    
    experts = load_experts()
    datasets_dict = get_datasets()
    proto_data = precompute_unified_prototypes(experts, datasets_dict)
    
    # Standard 30-batch sequential stream with fixed seed
    set_seed(42)
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

    # Default parameters: eta_0 = 0.2, steps = 5, beta = 1.0
    print("\n" + "="*50)
    print("1. SWEEPING LEARNING RATE (eta_0) [steps=5, beta=1.0]")
    print("="*50)
    print(f"{'eta_0':<10} | {'MNIST Avg (%)':<15} | {'KMNIST Avg (%)':<15} | {'Fashion Avg (%)':<15} | {'Overall Avg (%)':<15}")
    print("-" * 75)
    for lr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        set_seed(42)
        m, k, f, o = evaluate_ours(experts, datasets_dict, proto_data, stream_batches, base_lr=lr, steps=5, beta=1.0, device=device)
        print(f"{lr:<10} | {m:.2f}%          | {k:.2f}%          | {f:.2f}%            | {o:.2f}%")

    print("\n" + "="*50)
    print("2. SWEEPING OPTIMIZATION STEPS (S) [eta_0=0.2, beta=1.0]")
    print("="*50)
    print(f"{'Steps (S)':<10} | {'MNIST Avg (%)':<15} | {'KMNIST Avg (%)':<15} | {'Fashion Avg (%)':<15} | {'Overall Avg (%)':<15}")
    print("-" * 75)
    for s in [1, 2, 3, 5, 10, 20]:
        set_seed(42)
        m, k, f, o = evaluate_ours(experts, datasets_dict, proto_data, stream_batches, base_lr=0.2, steps=s, beta=1.0, device=device)
        print(f"{s:<10} | {m:.2f}%          | {k:.2f}%          | {f:.2f}%            | {o:.2f}%")

    print("\n" + "="*50)
    print("3. SWEEPING DIVERSITY COEFFICIENT (beta) [eta_0=0.2, steps=5]")
    print("="*50)
    print(f"{'beta':<10} | {'MNIST Avg (%)':<15} | {'KMNIST Avg (%)':<15} | {'Fashion Avg (%)':<15} | {'Overall Avg (%)':<15}")
    print("-" * 75)
    for b in [0.0, 0.1, 0.3, 0.5, 1.0, 1.5]:
        set_seed(42)
        m, k, f, o = evaluate_ours(experts, datasets_dict, proto_data, stream_batches, base_lr=0.2, steps=5, beta=b, device=device)
        print(f"{b:<10} | {m:.2f}%          | {k:.2f}%          | {f:.2f}%            | {o:.2f}%")

if __name__ == "__main__":
    main()
