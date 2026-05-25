import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from run_experiments import get_dataloaders, get_base_vit, inject_lora, ModelMerger, add_gaussian_noise, LoRALinear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inspect():
    loaders = get_dataloaders()
    expert_cifar_path = "expert_cifar10.pt"
    expert_svhn_path = "expert_svhn.pt"
    
    if not os.path.exists(expert_cifar_path) or not os.path.exists(expert_svhn_path):
        print("Pre-trained experts not found.")
        return
        
    expert_cifar_state = torch.load(expert_cifar_path, map_location=DEVICE)
    expert_svhn_state = torch.load(expert_svhn_path, map_location=DEVICE)
    
    # Run TTA for SyMerge and CAS-Merge (Ours with beta=1.0) on corrupted stream
    methods = ["symerge", "sat_tta", "cas_merge"]
    for method in methods:
        print(f"\n--- Running TTA for {method} ---")
        base_model = get_base_vit()
        inject_lora(base_model)
        
        cifar_lora = {k: v.to(DEVICE) for k, v in expert_cifar_state.items() if "lora" in k}
        svhn_lora = {k: v.to(DEVICE) for k, v in expert_svhn_state.items() if "lora" in k}
        
        cifar_head = nn.Linear(768, 10).to(DEVICE)
        cifar_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_cifar_state.items() if "heads.head" in k})
        
        svhn_head = nn.Linear(768, 10).to(DEVICE)
        svhn_head.load_state_dict({k.replace("heads.head.", ""): v for k, v in expert_svhn_state.items() if "heads.head" in k})
        
        merger = ModelMerger(base_model, cifar_lora, svhn_lora, cifar_head, svhn_head).to(DEVICE)
        merger.apply_merge()
        
        trainable_params = []
        merger.coeffs.requires_grad = True
        merger.cifar_head.weight.requires_grad = True
        merger.cifar_head.bias.requires_grad = True
        merger.svhn_head.weight.requires_grad = True
        merger.svhn_head.bias.requires_grad = True
        trainable_params = [merger.coeffs, merger.cifar_head.weight, merger.cifar_head.bias, merger.svhn_head.weight, merger.svhn_head.bias]
        
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        
        tasks = [
            {"name": "cifar10", "loader": loaders["cifar_test"], "head": merger.cifar_head, "expert_state": expert_cifar_state},
            {"name": "svhn", "loader": loaders["svhn_test"], "head": merger.svhn_head, "expert_state": expert_svhn_state}
        ]
        
        for task in tasks:
            task_name = task["name"]
            loader = task["loader"]
            head = task["head"]
            
            # Reset coefficients
            with torch.no_grad():
                merger.coeffs.data.fill_(0.5)
                merger.apply_merge()
                
            # Expert model guide
            expert_model = get_base_vit()
            inject_lora(expert_model)
            expert_model.heads.head = nn.Linear(768, 10)
            expert_model.load_state_dict(task["expert_state"])
            expert_model = expert_model.to(DEVICE)
            expert_model.eval()
            
            if method == "cas_merge":
                for m in merger.base_model.modules():
                    if isinstance(m, LoRALinear):
                        m.track_activations = True
                        
            correct = 0
            total = 0
            
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = add_gaussian_noise(x, severity=0.15)
                
                # Eval
                merger.base_model.eval()
                head.eval()
                merger.base_model.heads.head = head
                with torch.no_grad():
                    logits = merger.base_model(x)
                    preds = logits.argmax(dim=-1)
                    correct += preds.eq(y).sum().item()
                    total += x.size(0)
                    
                # Train step
                merger.base_model.train()
                head.train()
                
                with torch.no_grad():
                    expert_logits = expert_model(x)
                    expert_probs = F.softmax(expert_logits, dim=-1)
                    
                def compute_loss():
                    logits = merger.base_model(x)
                    loss_distill = F.kl_div(F.log_softmax(logits, dim=-1), expert_probs, reduction="batchmean")
                    
                    loss_calp = 0.0
                    if method == "cas_merge":
                        beta = 1.0
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
                
                if method in ["sat_tta", "cas_merge"]:
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
                    optimizer.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    optimizer.step()
                    
                with torch.no_grad():
                    merger.apply_merge()
            
            print(f"Task: {task_name} | Final Coeffs (sigmoids):")
            with torch.no_grad():
                coeffs_sig = torch.sigmoid(merger.coeffs).cpu().numpy()
                print(" ".join([f"{c:.4f}" for c in coeffs_sig]))
            print(f"Accuracy: {correct/total*100:.2f}%")

if __name__ == "__main__":
    inspect()
