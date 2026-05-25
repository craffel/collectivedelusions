import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os

from evaluate_ttmm import SimpleCNN, precompute_prototypes, compute_routing_priors, differentiable_forward, merge_models, compute_entropy

def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load experts
    model0 = SimpleCNN()
    model1 = SimpleCNN()
    model0.load_state_dict(torch.load('checkpoints/expert0.pth', map_location=device))
    model1.load_state_dict(torch.load('checkpoints/expert1.pth', map_location=device))
    model0.to(device)
    model1.to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_cal = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_cal = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_cal_loader = DataLoader(Subset(mnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    fmnist_cal_loader = DataLoader(Subset(fmnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    
    proto0 = precompute_prototypes(model0, mnist_cal_loader, device)
    proto1 = precompute_prototypes(model1, fmnist_cal_loader, device)
    
    # Re-generate the exact 50-batch stream with a fixed seed to be consistent
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    stream_batches = []
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Phase 0: Clean MNIST
    for _ in range(10):
        stream_batches.append(next(mnist_iter))
    # Phase 1: Noisy MNIST
    for _ in range(10):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((images_noisy, labels))
    # Phase 2: Clean FashionMNIST
    for _ in range(10):
        stream_batches.append(next(fmnist_iter))
    # Phase 3: Noisy FashionMNIST
    for _ in range(10):
        images, labels = next(fmnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((images_noisy, labels))
    # Phase 4: Novel KMNIST
    for _ in range(10):
        stream_batches.append(next(kmnist_iter))
        
    layer_groups = ['conv1', 'bn1', 'conv2', 'bn2', 'fc1', 'fc2']
    
    # We will test single-step vs multi-step adaptation, and different learning rates.
    lrs = [0.005, 0.01, 0.05, 0.1]
    steps_options = [1, 5]
    
    for steps in steps_options:
        for lr_sweep in lrs:
            print(f"\n==================== CONFIG: STEPS = {steps}, LR = {lr_sweep} ====================")
            for method in ["sam_ttmm", "sw_sam_ttmm"]:
                w_global = 0.0
                deltas = {g: 0.0 for g in layer_groups}
                
                accuracies = []
                
                for b_idx, (batch_images, batch_labels) in enumerate(stream_batches):
                    phase_idx = b_idx // 10
                    batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                    w0, w1, h_batch, H_avg = compute_routing_priors(batch_images, model0, model1, proto0, proto1, device)
                    
                    if b_idx % 10 == 0:
                        w_global = np.log(w1 / (w0 + 1e-8) + 1e-8)
                        deltas = {g: 0.0 for g in layer_groups}
                        
                    # Outer loop for adaptation steps
                    for step_idx in range(steps):
                        if method == "sam_ttmm":
                            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
                            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
                            
                            rho = 0.05
                            eta = lr_sweep
                            epsilon_stab = 0.1
                            
                            lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
                            logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
                            
                            lambda_bar = torch.stack(list(lambdas.values())).mean()
                            q = torch.stack([lambda_bar, 1.0 - lambda_bar])
                            p = torch.tensor([w1, w0], device=device)
                            loss_kl = F.kl_div(torch.log(q + 1e-8), p, reduction='sum')
                            loss = compute_entropy(logits) + 0.01 * loss_kl
                            loss.backward()
                            
                            F_sens = {}
                            for g in layer_groups:
                                grad_val = deltas_t[g].grad
                                F_sens[g] = torch.mean(grad_val**2).item() if grad_val is not None else 1e-8
                            total_F = sum(F_sens.values()) + 1e-8
                            F_sens_norm = {g: F_sens[g] / total_F for g in layer_groups}
                            
                            d_w = w_global_t.grad.item() if w_global_t.grad is not None else 0.0
                            d_deltas = {}
                            for g in layer_groups:
                                g_val = deltas_t[g].grad.item() if deltas_t[g].grad is not None else 0.0
                                d_deltas[g] = g_val / (F_sens_norm[g] + epsilon_stab)
                                
                            D_norm = np.sqrt(d_w**2 + sum(val**2 for val in d_deltas.values()) + epsilon_stab)
                            epsilon_w = rho * d_w / D_norm
                            epsilon_deltas = {g: rho * d_deltas[g] / D_norm for g in layer_groups}
                            
                            w_global_pert = w_global_t.item() + epsilon_w
                            deltas_pert = {g: deltas_t[g].item() + epsilon_deltas[g] for g in layer_groups}
                            
                            w_global_pert_t = torch.tensor(w_global_pert, requires_grad=True, device=device)
                            deltas_pert_t = {g: torch.tensor(deltas_pert[g], requires_grad=True, device=device) for g in layer_groups}
                            lambdas_p = {g: torch.sigmoid(w_global_pert_t + deltas_pert_t[g]) for g in layer_groups}
                            logits_p = differentiable_forward(batch_images, model0, model1, lambdas_p, device)
                            loss_p = compute_entropy(logits_p)
                            loss_p.backward()
                            
                            # Update
                            w_grad = w_global_pert_t.grad.item() if w_global_pert_t.grad is not None else 0.0
                            w_global = w_global_t.item() - eta * w_grad
                            for g in layer_groups:
                                g_grad = deltas_pert_t[g].grad.item() if deltas_pert_t[g].grad is not None else 0.0
                                deltas[g] = deltas_t[g].item() - eta * (1.0 / (F_sens_norm[g] + epsilon_stab)) * g_grad
                                
                        elif method == "sw_sam_ttmm":
                            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
                            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
                            
                            rho = 0.05
                            if h_batch >= 0.50:
                                rho_adaptive = rho * (1.0 - h_batch)
                            else:
                                rho_adaptive = rho
                                
                            gamma_ealr = 5.0
                            eta = lr_sweep
                            eta_t = eta / (1.0 + gamma_ealr * H_avg)
                            epsilon_stab = 0.1
                            
                            lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
                            logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
                            
                            lambda_bar = torch.stack(list(lambdas.values())).mean()
                            q = torch.stack([lambda_bar, 1.0 - lambda_bar])
                            p = torch.tensor([w1, w0], device=device)
                            loss_kl = F.kl_div(torch.log(q + 1e-8), p, reduction='sum')
                            loss = compute_entropy(logits) + 0.01 * loss_kl
                            loss.backward()
                            
                            F_sens = {}
                            for g in layer_groups:
                                grad_val = deltas_t[g].grad
                                F_sens[g] = torch.mean(grad_val**2).item() if grad_val is not None else 1e-8
                            total_F = sum(F_sens.values()) + 1e-8
                            F_sens_norm = {g: F_sens[g] / total_F for g in layer_groups}
                            
                            d_w = w_global_t.grad.item() if w_global_t.grad is not None else 0.0
                            d_deltas = {}
                            for g in layer_groups:
                                g_val = deltas_t[g].grad.item() if deltas_t[g].grad is not None else 0.0
                                d_deltas[g] = g_val / (F_sens_norm[g] + epsilon_stab)
                                
                            D_norm = np.sqrt(d_w**2 + sum(val**2 for val in d_deltas.values()) + epsilon_stab)
                            epsilon_w = rho_adaptive * d_w / D_norm
                            epsilon_deltas = {g: rho_adaptive * d_deltas[g] / D_norm for g in layer_groups}
                            
                            w_global_pert = w_global_t.item() + epsilon_w
                            deltas_pert = {g: deltas_t[g].item() + epsilon_deltas[g] for g in layer_groups}
                            
                            w_global_pert_t = torch.tensor(w_global_pert, requires_grad=True, device=device)
                            deltas_pert_t = {g: torch.tensor(deltas_pert[g], requires_grad=True, device=device) for g in layer_groups}
                            lambdas_p = {g: torch.sigmoid(w_global_pert_t + deltas_pert_t[g]) for g in layer_groups}
                            logits_p = differentiable_forward(batch_images, model0, model1, lambdas_p, device)
                            loss_p = compute_entropy(logits_p)
                            loss_p.backward()
                            
                            # Update using EALR
                            w_grad = w_global_pert_t.grad.item() if w_global_pert_t.grad is not None else 0.0
                            w_global = w_global_t.item() - eta_t * w_grad
                            for g in layer_groups:
                                g_grad = deltas_pert_t[g].grad.item() if deltas_pert_t[g].grad is not None else 0.0
                                deltas[g] = deltas_t[g].item() - eta_t * (1.0 / (F_sens_norm[g] + epsilon_stab)) * g_grad
                    
                    # Evaluate final model on this batch
                    lambdas_final = {g: torch.sigmoid(torch.tensor(w_global + deltas[g])).item() for g in layer_groups}
                    merged = merge_models(model0, model1, lambdas_final).to(device)
                    merged.eval()
                    with torch.no_grad():
                        logits, _ = merged(batch_images)
                        _, preds = logits.max(1)
                        acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                    accuracies.append(acc)
                print(f"  {method.upper()} Overall Accuracy: {np.mean(accuracies)*100:.2f}%")

if __name__ == "__main__":
    run_diagnostic()
