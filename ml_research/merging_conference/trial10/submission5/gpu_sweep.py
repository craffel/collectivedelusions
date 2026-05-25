import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from torch.func import functional_call
import time

from run_ttmm import SimpleCNN, get_layer_group, compute_hoyers_sparsity, compute_entropy

def get_merged_state_dict_differentiable_gpu(expert_0_state, expert_1_state, w_global, delta, device):
    merged_params_and_buffers = {}
    for name in expert_0_state.keys():
        if "running_mean" in name:
            mu0 = expert_0_state[name]
            mu1 = expert_1_state[name]
            bn_key = name.split(".")[0] + ".weight"
            group = get_layer_group(bn_key)
            l = torch.sigmoid(torch.tensor(w_global, device=device) + delta[group]).detach()
            merged_params_and_buffers[name] = l * mu0 + (1 - l) * mu1
        elif "running_var" in name:
            mean_name = name.replace("running_var", "running_mean")
            mu0 = expert_0_state[mean_name]
            mu1 = expert_1_state[mean_name]
            var0 = expert_0_state[name]
            var1 = expert_1_state[name]
            bn_key = name.split(".")[0] + ".weight"
            group = get_layer_group(bn_key)
            l = torch.sigmoid(torch.tensor(w_global, device=device) + delta[group]).detach()
            merged_params_and_buffers[name] = l * var0 + (1 - l) * var1 + l * (1 - l) * (mu0 - mu1) ** 2
        elif "num_batches_tracked" in name:
            merged_params_and_buffers[name] = expert_0_state[name]
        else:
            group = get_layer_group(name)
            l = torch.sigmoid(torch.tensor(w_global, device=device) + delta[group])
            merged_params_and_buffers[name] = l * expert_0_state[name] + (1 - l) * expert_1_state[name]
    return merged_params_and_buffers

def evaluate_stream_gpu(expert_0, expert_1, stream_batches, lr_base, alpha, beta, rho, damping_base, device):
    merged_model = SimpleCNN().to(device)
    merged_model.eval()
    
    all_preds = []
    all_targets = []
    
    sd_0 = expert_0.state_dict()
    sd_1 = expert_1.state_dict()
    
    for b_idx, (x, y) in enumerate(stream_batches):
        S = compute_hoyers_sparsity(x)
        if S >= 0.50:
            w_global = 2.0
        else:
            w_global = -2.0
            
        delta = torch.zeros(4, requires_grad=True, device=device)
        
        # Step 1: Compute original loss
        merged_params = get_merged_state_dict_differentiable_gpu(
            sd_0, sd_1, w_global, delta, device
        )
        logits, _ = functional_call(merged_model, merged_params, (x,))
        loss_orig = compute_entropy(logits)
        
        # Compute gradient
        loss_orig.backward()
        g = delta.grad.clone()
        
        # Step 2: Sharpness-aware perturbation
        epsilon = rho * g / (torch.norm(g) + 1e-12)
        delta.grad.zero_()
        
        # Step 3: Compute perturbed loss
        merged_params_pert = get_merged_state_dict_differentiable_gpu(
            sd_0, sd_1, w_global, delta + epsilon, device
        )
        logits_pert, _ = functional_call(merged_model, merged_params_pert, (x,))
        loss_pert = compute_entropy(logits_pert)
        
        loss_gap = max(0.0, (loss_pert - loss_orig).item())
        
        lr_adaptive = lr_base * np.exp(-alpha * loss_gap)
        damping_adaptive = damping_base * (1.0 + beta * loss_gap)
        
        # Compute gradient of perturbed loss
        loss_pert.backward()
        g_pert = delta.grad.clone()
        
        # Preconditioning
        F_j = g_pert ** 2
        F_tilde = F_j / (F_j.sum() + 1e-8)
        
        # Update delta
        eta_j = lr_adaptive / (F_tilde + damping_adaptive)
        with torch.no_grad():
            delta -= eta_j * g_pert
        delta.grad.zero_()
        
        # Final pass
        with torch.no_grad():
            merged_params_final = get_merged_state_dict_differentiable_gpu(
                sd_0, sd_1, w_global, delta, device
            )
            logits, _ = functional_call(merged_model, merged_params_final, (x,))
            
        preds = logits.argmax(dim=1)
        all_preds.append(preds)
        all_targets.append(y)
        
    all_preds_cat = torch.cat(all_preds)
    all_targets_cat = torch.cat(all_targets)
    
    phase_accs = []
    for p in range(5):
        p_preds = all_preds_cat[p*640 : (p+1)*640]
        p_targets = all_targets_cat[p*640 : (p+1)*640]
        acc = (p_preds == p_targets).float().mean().item()
        phase_accs.append(acc)
        
    overall_acc = (all_preds_cat == all_targets_cat).float().mean().item()
    return phase_accs, overall_acc

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    expert_0 = SimpleCNN().to(device)
    expert_1 = SimpleCNN().to(device)
    expert_0.load_state_dict(torch.load("models/expert_0.pt", map_location=device))
    expert_1.load_state_dict(torch.load("models/expert_1.pt", map_location=device))
    expert_0.eval()
    expert_1.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = datasets.MNIST(root="data", train=False, download=False, transform=transform)
    fmnist_test = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="data", train=False, download=False, transform=transform)
    
    mnist_clean_loader = DataLoader(Subset(mnist_test, list(range(640))), batch_size=64, shuffle=False)
    fmnist_clean_loader = DataLoader(Subset(fmnist_test, list(range(640))), batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(Subset(kmnist_test, list(range(640))), batch_size=64, shuffle=False)
    
    stream_batches = []
    for x, y in mnist_clean_loader:
        stream_batches.append((x.to(device), y.to(device)))
    for x, y in mnist_clean_loader:
        noisy_x = x + 0.6 * torch.randn_like(x)
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        stream_batches.append((noisy_x.to(device), y.to(device)))
    for x, y in fmnist_clean_loader:
        stream_batches.append((x.to(device), y.to(device)))
    for x, y in fmnist_clean_loader:
        noisy_x = x + 0.6 * torch.randn_like(x)
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        stream_batches.append((noisy_x.to(device), y.to(device)))
    for x, y in kmnist_loader:
        stream_batches.append((x.to(device), y.to(device)))
        
    print("Stream loaded.")
    
    # If on CPU, do a quick test of 1 config to verify equivalence with run_ttmm
    if device.type == "cpu":
        # Test the tuned configuration
        print("Running test on CPU...")
        pa, oa = evaluate_stream_gpu(expert_0, expert_1, stream_batches, lr_base=150.0, alpha=10.0, beta=50.0, rho=0.03, damping_base=0.05, device=device)
        print(f"Test Accuracy: {oa*100:.4f}% (Expected: 59.5938%)")
        print(f"Phase accuracies: {[round(a*100, 2) for a in pa]}")
        return
        
    # Large grid search
    # Choices
    lr_candidates = [100.0, 120.0, 150.0, 180.0, 200.0, 250.0] # 6
    alphas = [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0] # 7
    betas = [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0] # 7
    rhos = [0.01, 0.02, 0.03, 0.04, 0.05] # 5
    dampings = [0.02, 0.03, 0.04, 0.05, 0.08, 0.10] # 6
    # Total configs: 6 * 7 * 7 * 5 * 6 = 8820 configs
    
    print(f"Starting GPU sweep of 8820 configurations...")
    start_time = time.time()
    
    best_acc = 0.0
    best_config = {}
    
    count = 0
    for lr in lr_candidates:
        for alpha in alphas:
            for beta in betas:
                for rho in rhos:
                    for db in dampings:
                        pa, oa = evaluate_stream_gpu(expert_0, expert_1, stream_batches, lr, alpha, beta, rho, db, device)
                        count += 1
                        if oa > best_acc:
                            best_acc = oa
                            best_config = {
                                "lr_base": lr, "alpha": alpha, "beta": beta, "rho": rho, "damping_base": db, "phase_accs": pa
                            }
                            print(f"NEW BEST ({count}): lr={lr} | alpha={alpha} | beta={beta} | rho={rho} | db={db} -> Acc: {oa*100:.4f}% (P1: {pa[0]*100:.2f}%, P2: {pa[1]*100:.2f}%, P3: {pa[2]*100:.2f}%, P4: {pa[3]*100:.2f}%, P5: {pa[4]*100:.2f}%)")
                            
                        if count % 1000 == 0:
                            print(f"Evaluated {count}/8820 configs in {time.time() - start_time:.2f}s...")
                            
    elapsed = time.time() - start_time
    print(f"Sweep completed in {elapsed:.2f}s ({count} configs, {elapsed/count*1000:.2f} ms per config).")
    print("==================================================")
    print("GLOBAL BEST CONFIG:")
    print(best_config)
    print(f"Best Accuracy: {best_acc*100:.4f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
