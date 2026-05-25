import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from run_experiments import LoRALinear, inject_lora, get_dataloaders, get_base_vit, ModelMerger

def add_gaussian_noise_with_sigma(x, sigma):
    if sigma == 0.0:
        return x
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, -2.5, 2.5)

def run_tta_with_noise(loaders, expert_cifar_state, expert_svhn_state, method, sigma):
    # Set random seed for reproducibility of noise across different methods and sigmas
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize base model
    base_model = get_base_vit()
    inject_lora(base_model)
    
    # Extract expert LoRA weights
    cifar_lora = {k: v.to(DEVICE) for k, v in expert_cifar_state.items() if "lora" in k}
    svhn_lora = {k: v.to(DEVICE) for k, v in expert_svhn_state.items() if "lora" in k}
    
    # Heads
    cifar_head = nn.Linear(768, 10).to(DEVICE)
    cifar_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_cifar_state.items() if "heads.head" in k})
    
    svhn_head = nn.Linear(768, 10).to(DEVICE)
    svhn_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_svhn_state.items() if "heads.head" in k})
    
    # Initialize merger
    merger = ModelMerger(base_model, cifar_lora, svhn_lora, cifar_head, svhn_head).to(DEVICE)
    merger.apply_merge()
    
    # Decide what to optimize
    trainable_params = []
    if method == "static":
        pass
    elif method == "symerge" or method == "sat_tta" or method == "cas_merge":
        # Optimize merging coefficients AND heads
        merger.coeffs.requires_grad = True
        merger.cifar_head.weight.requires_grad = True
        merger.cifar_head.bias.requires_grad = True
        merger.svhn_head.weight.requires_grad = True
        merger.svhn_head.bias.requires_grad = True
        trainable_params = [merger.coeffs, merger.cifar_head.weight, merger.cifar_head.bias, merger.svhn_head.weight, merger.svhn_head.bias]
    
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3) if trainable_params else None
    
    tasks = [
        {"name": "cifar10", "loader": loaders["cifar_test"], "head": merger.cifar_head, "expert_state": expert_cifar_state},
        {"name": "svhn", "loader": loaders["svhn_test"], "head": merger.svhn_head, "expert_state": expert_svhn_state}
    ]
    
    task_accuracies = {}
    
    for task in tasks:
        task_name = task["name"]
        loader = task["loader"]
        head = task["head"]
        
        # Expert guide model
        expert_model = get_base_vit()
        inject_lora(expert_model)
        expert_model.heads.head = nn.Linear(768, 10)
        expert_model.load_state_dict(task["expert_state"])
        expert_model = expert_model.to(DEVICE)
        expert_model.eval()
        
        with torch.no_grad():
            merger.coeffs.data.fill_(0.5)
            merger.apply_merge()
            
        correct = 0
        total = 0
        
        # Enable activation tracking for CAS-Merge
        if method == "cas_merge":
            for m in merger.base_model.modules():
                if isinstance(m, LoRALinear):
                    m.track_activations = True
                    
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = add_gaussian_noise_with_sigma(x, sigma)
                
            # 1. Evaluate current step model
            merger.base_model.eval()
            head.eval()
            merger.base_model.heads.head = head
            with torch.no_grad():
                logits_eval = merger.base_model(x)
                preds = logits_eval.argmax(dim=-1)
                correct += preds.eq(y).sum().item()
                total += x.size(0)
                
            # 2. Adaptation Step
            if method != "static" and optimizer is not None:
                merger.base_model.train()
                head.train()
                merger.base_model.heads.head = head
                
                with torch.no_grad():
                    expert_logits = expert_model(x)
                    expert_probs = F.softmax(expert_logits, dim=-1)
                
                def compute_loss():
                    logits = merger.base_model(x)
                    loss_distill = F.kl_div(F.log_softmax(logits, dim=-1), expert_probs, reduction="batchmean")
                    
                    loss_calp = 0.0
                    if method == "cas_merge":
                        beta = 1.0  # Use the optimal beta found in ablation
                        for m_name, m in merger.base_model.named_modules():
                            if isinstance(m, LoRALinear) and m.current_activations is not None:
                                act = m.current_activations
                                if act.dim() == 3:
                                    act_flat = act.view(-1, act.size(-1))
                                else:
                                    act_flat = act
                                var = act_flat.var(dim=0, unbiased=False)
                                
                                key_a = m_name + ".lora_a"
                                key_b = m_name + ".lora_b"
                                w_a_cifar = merger.expert_cifar_lora[key_a]
                                w_a_svhn = merger.expert_svhn_lora[key_a]
                                w_b_cifar = merger.expert_cifar_lora[key_b]
                                w_b_svhn = merger.expert_svhn_lora[key_b]
                                
                                coeff_idx = 12
                                for i in range(12):
                                    if f"encoder_layer_{i}" in m_name:
                                        coeff_idx = i
                                        break
                                lmbda = torch.sigmoid(merger.coeffs[coeff_idx])
                                
                                dyn_lora_a = lmbda * w_a_cifar + (1.0 - lmbda) * w_a_svhn
                                dyn_lora_b = lmbda * w_b_cifar + (1.0 - lmbda) * w_b_svhn
                                merged_W = dyn_lora_b @ dyn_lora_a * m.scaling
                                target_W = (lmbda * (w_b_cifar @ w_a_cifar) + (1.0 - lmbda) * (w_b_svhn @ w_a_svhn)) * m.scaling
                                
                                diff = merged_W - target_W
                                loss_calp += torch.sum((diff ** 2) * (var.unsqueeze(0) + 1e-5))
                                
                        loss_distill += beta * loss_calp
                        
                    return loss_distill
                
                # Apply optimizer steps
                if method == "sat_tta" or method == "cas_merge":
                    # SAM 2-step update
                    rho = 0.05
                    optimizer.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    
                    with torch.no_grad():
                        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in trainable_params if p.grad is not None]), 2)
                        if grad_norm > 1e-5:
                            for p in trainable_params:
                                if p.grad is not None:
                                    eps = p.grad * (rho / (grad_norm + 1e-12))
                                    p.add_(eps)
                                    p.eps = eps
                                    
                    optimizer.zero_grad()
                    loss_perturbed = compute_loss()
                    loss_perturbed.backward()
                    
                    with torch.no_grad():
                        for p in trainable_params:
                            if hasattr(p, "eps"):
                                p.sub_(p.eps)
                                del p.eps
                    optimizer.step()
                else:
                    # Standard gradient descent
                    optimizer.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    merger.apply_merge()
                    
        acc = (correct / total) * 100
        task_accuracies[task_name] = acc
        
    task_accuracies["average"] = (task_accuracies["cifar10"] + task_accuracies["svhn"]) / 2.0
    return task_accuracies

def main():
    print("=========================================")
    print("   CAS-Merge: Noise Robustness Sweep     ")
    print("=========================================")
    
    loaders = get_dataloaders()
    expert_cifar_state = torch.load("expert_cifar10.pt", map_location=DEVICE)
    expert_svhn_state = torch.load("expert_svhn.pt", map_location=DEVICE)
    
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    methods = ["static", "symerge", "sat_tta", "cas_merge"]
    
    results = {m: [] for m in methods}
    
    for sigma in sigmas:
        print(f"\n--- Evaluating Noise Sigma: {sigma} ---")
        for method in methods:
            res = run_tta_with_noise(loaders, expert_cifar_state, expert_svhn_state, method, sigma)
            avg_acc = res["average"]
            print(f"Method: {method:<10} | CIFAR10: {res['cifar10']:.2f}% | SVHN: {res['svhn']:.2f}% | Avg: {avg_acc:.2f}%")
            results[method].append(avg_acc)
            
    # Save results to JSON
    with open("noise_results.json", "w") as f:
        json.dump({"sigmas": sigmas, "results": results}, f, indent=4)
        
    # Plot results
    plt.figure(figsize=(8, 5))
    styles = {
        "static": {"color": "gray", "marker": "o", "linestyle": "--", "label": "Static Merging"},
        "symerge": {"color": "blue", "marker": "s", "linestyle": "-.", "label": "SyMerge"},
        "sat_tta": {"color": "orange", "marker": "^", "linestyle": ":", "label": "SAT-TTA (SAM)"},
        "cas_merge": {"color": "red", "marker": "D", "linestyle": "-", "linewidth": 2, "label": "CAS-Merge (Ours)"}
    }
    
    for m in methods:
        plt.plot(sigmas, results[m], **styles[m])
        
    plt.xlabel("Gaussian Noise Standard Deviation ($\\sigma$)", fontsize=12)
    plt.ylabel("Multi-Task Average Accuracy (%)", fontsize=12)
    plt.title("OOD Noise Robustness on CIFAR-10 / SVHN", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig("noise_robustness.png", dpi=300)
    print("\nNoise sweep completed and 'noise_robustness.png' saved!")

if __name__ == "__main__":
    main()
