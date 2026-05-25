import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from run_experiments import LoRALinear, inject_lora, get_dataloaders, get_base_vit, ModelMerger, add_gaussian_noise, evaluate

def run_tta_updated(loaders, expert_cifar_state, expert_svhn_state, method="static", use_sam=False, use_calp=False, beta=1.0, rho=0.05, corrupt=False):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    base_model = get_base_vit()
    inject_lora(base_model)
    
    cifar_lora = {k: v.to(DEVICE) for k, v in expert_cifar_state.items() if "lora" in k}
    svhn_lora = {k: v.to(DEVICE) for k, v in expert_svhn_state.items() if "lora" in k}
    
    cifar_head = nn.Linear(768, 10).to(DEVICE)
    cifar_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_cifar_state.items() if "heads.head" in k})
    
    svhn_head = nn.Linear(768, 10).to(DEVICE)
    svhn_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_svhn_state.items() if "heads.head" in k})
    
    merger = ModelMerger(base_model, cifar_lora, svhn_lora, cifar_head, svhn_head).to(DEVICE)
    
    # Initialize coefficients to 0.0 -> Sigmoid(0.0) = 0.5 (exact even merge)
    with torch.no_grad():
        merger.coeffs.data.fill_(0.0)
        merger.apply_merge()
        
    trainable_params = []
    if method == "static":
        pass
    elif method == "adamerging":
        merger.coeffs.requires_grad = True
        trainable_params = [merger.coeffs]
    elif method == "tta":
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
        
        expert_model = get_base_vit()
        inject_lora(expert_model)
        expert_model.heads.head = nn.Linear(768, 10)
        expert_model.load_state_dict(task["expert_state"])
        expert_model = expert_model.to(DEVICE)
        expert_model.eval()
        
        with torch.no_grad():
            merger.coeffs.data.fill_(0.0)
            merger.apply_merge()
            
        correct = 0
        total = 0
        
        if use_calp:
            for m in merger.base_model.modules():
                if isinstance(m, LoRALinear):
                    m.track_activations = True
                    
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if corrupt:
                x = add_gaussian_noise(x)
                
            merger.base_model.eval()
            head.eval()
            merger.base_model.heads.head = head
            with torch.no_grad():
                logits_eval = merger.base_model(x)
                preds = logits_eval.argmax(dim=-1)
                correct += preds.eq(y).sum().item()
                total += x.size(0)
                
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
                    if use_calp:
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
                                
                                # Use relative loss since it is scale-invariant
                                loss_calp += torch.sum((diff ** 2) * (var.unsqueeze(0) + 1e-5)) / (torch.sum(target_W ** 2) + 1e-5)
                                
                        loss_distill += beta * loss_calp
                        
                    return loss_distill
                
                if use_sam:
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
    loaders = get_dataloaders()
    expert_cifar_state = torch.load("expert_cifar10.pt", map_location=DEVICE)
    expert_svhn_state = torch.load("expert_svhn.pt", map_location=DEVICE)
    
    # We will evaluate 6 distinct configurations:
    # 1. Static Merging (even merge, lambda=0.5)
    # 2. AdaMerging (even merge, lambda=0.5, coefficient-only TTA)
    # 3. SyMerge (even merge, lambda=0.5, joint head/coeff TTA via GD)
    # 4. SAT-TTA (even merge, lambda=0.5, joint head/coeff TTA via SAM)
    # 5. CAS-Merge (GD) (ours, even merge, joint head/coeff TTA via GD + CALP relative beta=1.0)
    # 6. CAS-Merge (SAM) (ours, even merge, joint head/coeff TTA via SAM + CALP relative beta=1.0, rho=0.05)
    
    configs = [
        {"name": "static", "method": "static", "use_sam": False, "use_calp": False, "beta": 0.0},
        {"name": "adamerging", "method": "adamerging", "use_sam": False, "use_calp": False, "beta": 0.0},
        {"name": "symerge", "method": "tta", "use_sam": False, "use_calp": False, "beta": 0.0},
        {"name": "sat_tta", "method": "tta", "use_sam": True, "use_calp": False, "beta": 0.0, "rho": 0.05},
        {"name": "cas_merge_gd", "method": "tta", "use_sam": False, "use_calp": True, "beta": 1.0},
        {"name": "cas_merge_sam", "method": "tta", "use_sam": True, "use_calp": True, "beta": 1.0, "rho": 0.05}
    ]
    
    results_clean = {}
    results_corrupt = {}
    
    print("Evaluating Clean Streams...")
    for config in configs:
        res = run_tta_updated(loaders, expert_cifar_state, expert_svhn_state, 
                             method=config["method"], use_sam=config["use_sam"], 
                             use_calp=config["use_calp"], beta=config["beta"], 
                             rho=config.get("rho", 0.05), corrupt=False)
        results_clean[config["name"]] = res
        print(f"{config['name']:<15} | CIFAR: {res['cifar10']:.2f}% | SVHN: {res['svhn']:.2f}% | Average: {res['average']:.2f}%")
        
    print("\nEvaluating Corrupted Streams (OOD)...")
    for config in configs:
        res = run_tta_updated(loaders, expert_cifar_state, expert_svhn_state, 
                             method=config["method"], use_sam=config["use_sam"], 
                             use_calp=config["use_calp"], beta=config["beta"], 
                             rho=config.get("rho", 0.05), corrupt=True)
        results_corrupt[config["name"]] = res
        print(f"{config['name']:<15} | CIFAR: {res['cifar10']:.2f}% | SVHN: {res['svhn']:.2f}% | Average: {res['average']:.2f}%")
        
    # Write to a JSON file
    with open("results_updated.json", "w") as f:
        json.dump({"clean": results_clean, "corrupt": results_corrupt}, f, indent=4)
    print("\nSaved updated results to 'results_updated.json' successfully!")

if __name__ == "__main__":
    main()
