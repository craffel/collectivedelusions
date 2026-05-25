import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import os
import numpy as np
import time

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

from evaluate_ttmm import (
    get_resnet18_1channel,
    precompute_offline_prototypes,
    get_merged_state_dict,
    extract_features,
    get_cohesion_score,
    project_simplex
)

def benchmark_method(method_name, sd0, sd1, test_stream, mu_static, class_prototypes0, class_prototypes1, device="cuda",
                     tau_entropy=0.70, alpha_ema=0.1, lr=0.005, beta_damping=0.5, gamma=5.0, anchor_layers=None):
    print(f"Benchmarking latency for: {method_name}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    adapt_layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    coefs = {prefix: torch.tensor([0.5, 0.5], device=device, requires_grad=True) for prefix in adapt_layers}
    
    model0 = get_resnet18_1channel().to(device)
    model0.load_state_dict(sd0)
    model0.eval()
    
    model1 = get_resnet18_1channel().to(device)
    model1.load_state_dict(sd1)
    model1.eval()
    
    merged_model = get_resnet18_1channel().to(device)
    
    dynamic_prototypes = {0: {}, 1: {}}
    
    activations = {}
    gradients = {}
    forward_hooks = []
    backward_hooks = []
    
    def register_hooks(model):
        nonlocal forward_hooks, backward_hooks
        for h in forward_hooks: h.remove()
        for h in backward_hooks: h.remove()
        forward_hooks.clear()
        backward_hooks.clear()
        
        for name, module in model.named_modules():
            if name in adapt_layers or any(name == prefix for prefix in adapt_layers):
                def get_f_hook(n):
                    def hook(m, inp, out):
                        x = inp[0].detach()
                        activations[n] = (x ** 2).sum().item() / (x.numel() / x.shape[1] if len(x.shape) > 1 else 1)
                    return hook
                def get_b_hook(n):
                    def hook(m, g_inp, g_out):
                        g = g_out[0].detach()
                        gradients[n] = (g ** 2).sum().item() / (g.numel() / g.shape[1] if len(g.shape) > 1 else 1)
                    return hook
                
                forward_hooks.append(module.register_forward_hook(get_f_hook(name)))
                backward_hooks.append(module.register_backward_hook(get_b_hook(name)))

    # Warmup GPU
    if device == "cuda":
        dummy_in = torch.randn(64, 1, 28, 28, device=device)
        for _ in range(10):
            _ = model0(dummy_in)
        torch.cuda.synchronize()

    # Measure latency
    start_time = time.time()
    
    for batch_idx, (images, labels, source_task) in enumerate(test_stream):
        images, labels = images.to(device), labels.to(device)
        
        # 1. MERGING & PREDICTION
        with torch.no_grad():
            if method_name == "Static":
                merged_sd = get_merged_state_dict(sd0, sd1, {k: torch.tensor([0.5, 0.5], device=device) for k in coefs})
            elif method_name == "PROTO-TTMM" or method_name == "KT-Fisher":
                merged_sd = get_merged_state_dict(sd0, sd1, {k: coefs[k].detach() for k in coefs})
            elif method_name == "DF-Bayes-TTMM":
                # Compute expert prediction entropies for Soft BN Buffer Fusion
                out0 = model0(images)
                p0 = torch.softmax(out0, dim=-1)
                ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                
                out1 = model1(images)
                p1 = torch.softmax(out1, dim=-1)
                ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                
                avg_entropy = 0.5 * (ent0 + ent1)
                is_novel = (avg_entropy > tau_entropy)
                
                if not is_novel:
                    w0 = np.exp(-gamma * ent0)
                    w1 = np.exp(-gamma * ent1)
                    w_bn = torch.tensor([w0 / (w0 + w1), w1 / (w0 + w1)], device=device)
                else:
                    w_bn = torch.tensor([0.5, 0.5], device=device) # default flat
                
                merged_sd = get_merged_state_dict(sd0, sd1, {k: coefs[k].detach() for k in coefs}, w_bn=w_bn)
            elif method_name == "FDF-DPA":
                # Compute expert prediction entropies for Soft BN Buffer Fusion
                out0 = model0(images)
                p0 = torch.softmax(out0, dim=-1)
                ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                
                out1 = model1(images)
                p1 = torch.softmax(out1, dim=-1)
                ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                
                avg_entropy = 0.5 * (ent0 + ent1)
                is_novel = (avg_entropy > tau_entropy)
                
                if not is_novel:
                    w0 = np.exp(-gamma * ent0)
                    w1 = np.exp(-gamma * ent1)
                    w_bn = torch.tensor([w0 / (w0 + w1), w1 / (w0 + w1)], device=device)
                else:
                    w_bn = torch.tensor([0.5, 0.5], device=device) # default flat
                
                merged_sd = get_merged_state_dict(sd0, sd1, {k: coefs[k].detach() for k in coefs}, w_bn=w_bn)
                
            merged_model.load_state_dict(merged_sd)
            merged_model.eval()
            
            # Predict
            outputs = merged_model(images)
            _, preds = outputs.max(1)
            
        # 2. ADAPTATION (Compute update for next step)
        if method_name == "Static":
            pass
            
        elif method_name == "PROTO-TTMM" or method_name == "KT-Fisher":
            with torch.no_grad():
                feats = extract_features(merged_model, images)
                c0 = get_cohesion_score(feats, mu_static, class_prototypes0, device)
                c1 = get_cohesion_score(feats, mu_static, class_prototypes1, device)
                max_cohesion = max(c0, c1)
                
            tau_N = 0.58
            is_novel = (max_cohesion < tau_N)
            
            if not is_novel:
                k_star = 0 if c0 > c1 else 1
                target_vector = torch.zeros(2, device=device)
                target_vector[k_star] = 1.0
                
                alpha_ema = 0.9
                for k in coefs:
                    coefs[k] = alpha_ema * coefs[k].detach() + (1.0 - alpha_ema) * target_vector
            else:
                with torch.no_grad():
                    out0 = model0(images)
                    p0 = torch.softmax(out0, dim=-1)
                    ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                    
                    out1 = model1(images)
                    p1 = torch.softmax(out1, dim=-1)
                    ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                    
                target_k = 0 if ent0 < ent1 else 1
                Y_t = torch.zeros(2, device=device)
                Y_t[target_k] = 1.0
                
                if method_name == "PROTO-TTMM":
                    lr = 0.005
                    for k in coefs:
                        new_val = coefs[k].detach() - lr * (coefs[k].detach() - Y_t)
                        coefs[k] = project_simplex(new_val)
                else:
                    register_hooks(merged_model)
                    optimizer = torch.optim.SGD(merged_model.parameters(), lr=0.0)
                    optimizer.zero_grad()
                    
                    merged_out = merged_model(images)
                    probs = torch.softmax(merged_out, dim=-1)
                    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                    entropy_loss.backward()
                    
                    lr = 0.005
                    beta_damping = 0.5
                    eps_scale = 1e-5
                    
                    for k in coefs:
                        name = k
                        tr_A = activations.get(name, 1.0)
                        tr_G = gradients.get(name, 1.0)
                        
                        param_size = 1.0
                        for name_p, p in merged_model.named_parameters():
                            if name_p.startswith(k) and "weight" in name_p:
                                param_size = p.numel()
                                break
                                
                        F_w = (tr_G * tr_A) / param_size
                        lr_w = lr * ((F_w + eps_scale) ** (-beta_damping))
                        new_val = coefs[k].detach() - lr_w * (coefs[k].detach() - Y_t)
                        coefs[k] = project_simplex(new_val)
                        
        elif method_name == "DF-Bayes-TTMM":
            # DF-Bayes-TTMM Baseline (Data-Free Bayesian TTMM)
            with torch.no_grad():
                out0 = model0(images)
                p0 = torch.softmax(out0, dim=-1)
                ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                
                out1 = model1(images)
                p1 = torch.softmax(out1, dim=-1)
                ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                
            avg_entropy = 0.5 * (ent0 + ent1)
            is_novel = (avg_entropy > tau_entropy)
            
            if not is_novel:
                w0 = np.exp(-gamma * ent0)
                w1 = np.exp(-gamma * ent1)
                w = torch.tensor([w0 / (w0 + w1), w1 / (w0 + w1)], device=device)
                for k in coefs:
                    coefs[k] = w.clone()
            else:
                target_k = 0 if ent0 < ent1 else 1
                Y_t = torch.zeros(2, device=device)
                Y_t[target_k] = 1.0
                
                register_hooks(merged_model)
                optimizer = torch.optim.SGD(merged_model.parameters(), lr=0.0)
                optimizer.zero_grad()
                
                merged_out = merged_model(images)
                probs = torch.softmax(merged_out, dim=-1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                entropy_loss.backward()
                
                eps_scale = 1e-5
                for k in coefs:
                    tr_A = activations.get(k, 1.0)
                    tr_G = gradients.get(k, 1.0)
                    
                    param_size = 1.0
                    for name_p, p in merged_model.named_parameters():
                        if name_p.startswith(k) and "weight" in name_p:
                            param_size = p.numel()
                            break
                    F_w = (tr_G * tr_A) / param_size
                    lr_w = lr * ((F_w + eps_scale) ** (-beta_damping))
                    
                    new_val = coefs[k].detach() - lr_w * (coefs[k].detach() - Y_t)
                    coefs[k] = project_simplex(new_val)
                    
        elif method_name == "FDF-DPA":
            with torch.no_grad():
                out0 = model0(images)
                p0 = torch.softmax(out0, dim=-1)
                ent0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=-1).mean().item()
                
                out1 = model1(images)
                p1 = torch.softmax(out1, dim=-1)
                ent1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=-1).mean().item()
                
            avg_entropy = 0.5 * (ent0 + ent1)
            is_novel = (avg_entropy > tau_entropy)
            
            if not is_novel:
                k_star = 0 if ent0 < ent1 else 1
                target_vector = torch.zeros(2, device=device)
                target_vector[k_star] = 1.0
                
                for k in coefs:
                    coefs[k] = alpha_ema * coefs[k].detach() + (1.0 - alpha_ema) * target_vector
                    
                with torch.no_grad():
                    feats = extract_features(merged_model, images)
                    active_outputs = out0 if k_star == 0 else out1
                    active_probs = torch.softmax(active_outputs, dim=-1)
                    max_probs, pred_classes = active_probs.max(dim=1)
                    
                    for i in range(images.size(0)):
                        if max_probs[i] > 0.95:
                            c = pred_classes[i].item()
                            feat = feats[i].detach()
                            if c not in dynamic_prototypes[k_star]:
                                dynamic_prototypes[k_star][c] = feat.clone()
                            else:
                                alpha_proto = 0.95
                                dynamic_prototypes[k_star][c] = alpha_proto * dynamic_prototypes[k_star][c] + (1.0 - alpha_proto) * feat
                                
            else:
                target_k = 0 if ent0 < ent1 else 1
                Y_t = torch.zeros(2, device=device)
                Y_t[target_k] = 1.0
                
                register_hooks(merged_model)
                optimizer = torch.optim.SGD(merged_model.parameters(), lr=0.0)
                optimizer.zero_grad()
                
                merged_out = merged_model(images)
                probs = torch.softmax(merged_out, dim=-1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                entropy_loss.backward()
                
                eps_scale = 1e-5
                if anchor_layers is None:
                    anchor_layers = ["conv1", "bn1", "layer1", "layer2"]
                
                for k in coefs:
                    if k in anchor_layers:
                        lr_w = 0.0
                    else:
                        tr_A = activations.get(k, 1.0)
                        tr_G = gradients.get(k, 1.0)
                        
                        param_size = 1.0
                        for name_p, p in merged_model.named_parameters():
                            if name_p.startswith(k) and "weight" in name_p:
                                param_size = p.numel()
                                break
                        F_w = (tr_G * tr_A) / param_size
                        lr_w = lr * ((F_w + eps_scale) ** (-beta_damping))
                        
                    new_val = coefs[k].detach() - lr_w * (coefs[k].detach() - Y_t)
                    coefs[k] = project_simplex(new_val)
                    
    if device == "cuda":
        torch.cuda.synchronize()
        
    total_time = time.time() - start_time
    # Clean up hooks
    for h in forward_hooks: h.remove()
    for h in backward_hooks: h.remove()
    
    avg_latency = (total_time / len(test_stream)) * 1000.0 # in ms
    print(f"Total time: {total_time:.4f}s | Avg Latency per Batch: {avg_latency:.2f} ms")
    return avg_latency

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Disable cuDNN for stability
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN for stability")
        
    # Load expert model weights
    mnist_path = "expert_mnist.pth"
    kmnist_path = "expert_kmnist.pth"
    
    sd_mnist = torch.load(mnist_path, map_location=device)
    sd_kmnist = torch.load(kmnist_path, map_location=device)
    
    # Prepare test stream dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root=".", train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root=".", train=False, download=False, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root=".", train=False, download=False, transform=transform)
    
    # Create non-stationary test stream: 90 sequential batches
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    fashion_loader = DataLoader(fashion_test, batch_size=64, shuffle=True)
    
    stream_batches = []
    
    # Get 30 batches of each
    mnist_iter = iter(mnist_loader)
    for _ in range(30):
        imgs, lbls = next(mnist_iter)
        stream_batches.append((imgs, lbls, "MNIST"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(30):
        imgs, lbls = next(kmnist_iter)
        stream_batches.append((imgs, lbls, "KMNIST"))
        
    fashion_iter = iter(fashion_loader)
    for _ in range(30):
        imgs, lbls = next(fashion_iter)
        stream_batches.append((imgs, lbls, "FashionMNIST"))
        
    print(f"Built test stream with {len(stream_batches)} batches.")
    
    mu_static, class_prototypes0, class_prototypes1 = precompute_offline_prototypes(sd_mnist, sd_kmnist, device)
    
    latency_results = {}
    methods = ["Static", "PROTO-TTMM", "DF-Bayes-TTMM", "KT-Fisher", "FDF-DPA"]
    
    for m in methods:
        latency_results[m] = benchmark_method(m, sd_mnist, sd_kmnist, stream_batches, mu_static, class_prototypes0, class_prototypes1, device)
        
    print("\n" + "="*50)
    print("LATENCY BENCHMARK RESULTS")
    print("="*50)
    print(f"{'Method':<15} | {'Avg Latency per Batch (ms)':<30}")
    print("-"*50)
    for m in methods:
        print(f"{m:<15} | {latency_results[m]:<30.2f}")
    print("="*50)
    
    with open("latency_results.json", "w") as f:
        json.dump(latency_results, f, indent=4)
    print("Latency results saved successfully.")

if __name__ == "__main__":
    main()
