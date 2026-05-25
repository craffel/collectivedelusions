import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from run_experiments import SimpleCNN, get_datasets, compute_fisher_and_prototypes, hoyer_sparsity
import numpy as np
import math

def test_routing_variant(variant_name, routing_fn):
    device = torch.device("cpu")
    train_mnist, test_mnist, train_fashion, test_fashion, test_kmnist = get_datasets()
    
    expert_mnist_cos = SimpleCNN(use_cosface=True).to(device)
    expert_mnist_cos.load_state_dict(torch.load("./models/mnist_cosface.pt", map_location=device))
    expert_fashion_cos = SimpleCNN(use_cosface=True).to(device)
    expert_fashion_cos.load_state_dict(torch.load("./models/fashion_cosface.pt", map_location=device))
    
    prototypes_cos0, fisher_cos0 = compute_fisher_and_prototypes(expert_mnist_cos, test_mnist, device, is_cosface=True)
    prototypes_cos1, fisher_cos1 = compute_fisher_and_prototypes(expert_fashion_cos, test_fashion, device, is_cosface=True)
    
    # Generate stream
    mnist_loader = DataLoader(test_mnist, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(test_fashion, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(test_kmnist, batch_size=64, shuffle=False)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream_batches = []
    for _ in range(10):
        images, labels = next(mnist_iter)
        stream_batches.append((images, labels, "Clean MNIST"))
    for _ in range(10):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((noisy_images, labels, "Noisy MNIST"))
    for _ in range(10):
        images, labels = next(fashion_iter)
        stream_batches.append((images, labels, "Clean Fashion"))
    for _ in range(10):
        images, labels = next(fashion_iter)
        noise = torch.randn_like(images) * 0.6
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((noisy_images, labels, "Noisy Fashion"))
    for _ in range(10):
        images, labels = next(kmnist_iter)
        stream_batches.append((images, labels, "Novel KMNIST"))
        
    # Run evaluation
    batch_accuracies = []
    eval_model = SimpleCNN(use_cosface=True).to(device)
    
    param_names = [name for name, _ in expert_mnist_cos.named_parameters()]
    state0 = expert_mnist_cos.state_dict()
    state1 = expert_fashion_cos.state_dict()
    
    bn_stats0 = {}
    for name, module in expert_mnist_cos.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats0[name] = {'running_mean': module.running_mean, 'running_var': module.running_var}
            
    bn_stats1 = {}
    for name, module in expert_fashion_cos.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats1[name] = {'running_mean': module.running_mean, 'running_var': module.running_var}
            
    for batch_idx, (X_t, Y_t, domain) in enumerate(stream_batches):
        X_t, Y_t = X_t.to(device), Y_t.to(device)
        
        # Call the custom routing function to get w0, w1
        w0, w1 = routing_fn(X_t, expert_mnist_cos, expert_fashion_cos, prototypes_cos0, prototypes_cos1)
        
        # Fixed TTA static merge (same as Method A but with our custom weights)
        with torch.no_grad():
            merged_state = {}
            for name in param_names:
                merged_state[name] = w0 * state0[name] + w1 * state1[name]
            eval_model.load_state_dict(merged_state, strict=False)
            
            for name, module in eval_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    mean0, var0 = bn_stats0[name]['running_mean'], bn_stats0[name]['running_var']
                    mean1, var1 = bn_stats1[name]['running_mean'], bn_stats1[name]['running_var']
                    mean_f = w0 * mean0 + w1 * mean1
                    var_f = w0 * (var0 + (mean0 - mean_f)**2) + w1 * (var1 + (mean1 - mean_f)**2)
                    module.running_mean.copy_(mean_f)
                    module.running_var.copy_(var_f)
                    
            preds = eval_model(X_t)
            acc = (preds.max(1)[1] == Y_t).float().mean().item() * 100.0
            batch_accuracies.append(acc)
            
    # Print results
    print(f"=== Variant: {variant_name} ===")
    print(f"  Clean MNIST:    {np.mean(batch_accuracies[0:10]):.2f}%")
    print(f"  Noisy MNIST:    {np.mean(batch_accuracies[10:20]):.2f}%")
    print(f"  Clean Fashion:  {np.mean(batch_accuracies[20:30]):.2f}%")
    print(f"  Noisy Fashion:  {np.mean(batch_accuracies[30:40]):.2f}%")
    print(f"  Novel KMNIST:   {np.mean(batch_accuracies[40:50]):.2f}%")
    print(f"  Overall:        {np.mean(batch_accuracies):.2f}%")
    return np.mean(batch_accuracies)

# Let's define different routing functions

def routing_baseline_angular(X_t, exp0, exp1, prot0, prot1):
    # This is standard Method D angular routing
    with torch.no_grad():
        f0 = exp0.get_features(X_t)
        f1 = exp1.get_features(X_t)
        
        dist0_list = []
        dist1_list = []
        for i in range(X_t.size(0)):
            f0_i_norm = F.normalize(f0[i], p=2, dim=0)
            f1_i_norm = F.normalize(f1[i], p=2, dim=0)
            d0 = min((1.0 - torch.dot(f0_i_norm, prot0[c])).item() for c in range(10))
            d1 = min((1.0 - torch.dot(f1_i_norm, prot1[c])).item() for c in range(10))
            dist0_list.append(d0)
            dist1_list.append(d1)
            
        D0_bar = np.mean(dist0_list)
        D1_bar = np.mean(dist1_list)
        gap = abs(D1_bar - D0_bar)
        tau = (gap / 3.0) + 0.04
        w0 = math.exp(-D0_bar / tau)
        w1 = math.exp(-D1_bar / tau)
        sum_w = w0 + w1
        return w0/sum_w, w1/sum_w

def routing_decisive_under_noise_l2(X_t, exp0, exp1, prot0, prot1):
    with torch.no_grad():
        out0 = exp0(X_t)
        out1 = exp1(X_t)
        p0 = F.softmax(out0, dim=1)
        p1 = F.softmax(out1, dim=1)
        h0 = -torch.mean(torch.sum(p0 * torch.log(p0 + 1e-8), dim=1)).item()
        h1 = -torch.mean(torch.sum(p1 * torch.log(p1 + 1e-8), dim=1)).item()
        h_avg = 0.5 * (h0 + h1)
        
        f0 = exp0.get_features(X_t)
        f1 = exp1.get_features(X_t)
        
        dist0_list = []
        dist1_list = []
        for i in range(X_t.size(0)):
            f0_i_norm = F.normalize(f0[i], p=2, dim=0)
            f1_i_norm = F.normalize(f1[i], p=2, dim=0)
            d0 = min(torch.sum((f0_i_norm - prot0[c])**2).item() for c in range(10))
            d1 = min(torch.sum((f1_i_norm - prot1[c])**2).item() for c in range(10))
            dist0_list.append(d0)
            dist1_list.append(d1)
            
        D0_bar = np.mean(dist0_list)
        D1_bar = np.mean(dist1_list)
        gap = abs(D1_bar - D0_bar)
        
        # Scale stability factor downwards under high uncertainty
        eps_stab = 0.08 / (1.0 + 2.0 * h_avg)
        tau = (gap / 3.0) + eps_stab
        
        w0 = math.exp(-D0_bar / tau)
        w1 = math.exp(-D1_bar / tau)
        sum_w = w0 + w1
        return w0/sum_w, w1/sum_w

if __name__ == "__main__":
    test_routing_variant("Baseline Angular (Method D)", routing_baseline_angular)
    test_routing_variant("Decisive Under Noise (L2)", routing_decisive_under_noise_l2)
