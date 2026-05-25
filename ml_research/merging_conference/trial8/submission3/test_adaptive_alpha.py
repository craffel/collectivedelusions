import torch
import numpy as np
import os
from run_experiment import SimpleCNN, AdaKLBatchNorm2D, convert_to_adakl_bn, precompute_prototypes, compute_expert_distances, compute_entropy, apply_soft_bn_stats, merge_weights, merge_layer_weights_adaptive
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def compute_kl_divergence(batch_mean, batch_var, fused_mean, fused_var, eps=1e-5):
    # batch_mean: [1, C, 1, 1], batch_var: [1, C, 1, 1]
    # fused_mean: [C], fused_var: [C]
    bm = batch_mean.view(-1)
    bv = batch_var.view(-1)
    fm = fused_mean.view(-1)
    fv = fused_var.view(-1)
    
    # KL(N(bm, bv) || N(fm, fv))
    kl = 0.5 * (torch.log(fv / (bv + eps)) + (bv + (bm - fm)**2) / (fv + eps) - 1.0)
    return kl.mean().item()

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    expert0_state = torch.load("mnist_expert.pth", map_location="cpu")
    expert1_state = torch.load("fashion_expert.pth", map_location="cpu")
    
    base_model = SimpleCNN()
    convert_to_adakl_bn(base_model)
    
    for name, m in base_model.named_modules():
        if isinstance(m, AdaKLBatchNorm2D):
            m0_mean = expert0_state[name + '.running_mean']
            m0_var = expert0_state[name + '.running_var']
            m1_mean = expert1_state[name + '.running_mean']
            m1_var = expert1_state[name + '.running_var']
            m.set_expert_stats(m0_mean, m0_var, m1_mean, m1_var)
            
    # Stream construction
    stream_batches = []
    mnist_iter = iter(mnist_loader)
    for _ in range(5):
        x, y = next(mnist_iter)
        stream_batches.append((x, "Clean MNIST"))
    for _ in range(5):
        x, y = next(mnist_iter)
        x_raw = x * 0.3081 + 0.1307
        noise = torch.randn_like(x_raw) * 0.15
        x_noisy = (torch.clamp(x_raw + noise, 0.0, 1.0) - 0.1307) / 0.3081
        stream_batches.append((x_noisy, "Noisy MNIST"))
        
    fmnist_iter = iter(fmnist_loader)
    for _ in range(5):
        x, y = next(fmnist_iter)
        stream_batches.append((x, "Clean Fashion"))
    for _ in range(5):
        x, y = next(fmnist_iter)
        x_raw = x * 0.3081 + 0.1307
        noise = torch.randn_like(x_raw) * 0.15
        x_noisy = (torch.clamp(x_raw + noise, 0.0, 1.0) - 0.1307) / 0.3081
        stream_batches.append((x_noisy, "Noisy Fashion"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(5):
        x, y = next(kmnist_iter)
        stream_batches.append((x, "Novel KMNIST"))
        
    print("Tracking KL Divergences for each phase...")
    for x, phase in stream_batches:
        # Compute fused stats at routing weights w0=0.5, w1=0.5 for comparison
        w0, w1 = 0.5, 0.5
        kl_list = []
        for name, m in base_model.named_modules():
            if isinstance(m, AdaKLBatchNorm2D):
                # Calculate fused stats
                mu_fused = w0 * m.expert0_mean + w1 * m.expert1_mean
                var_fused = w0 * (m.expert0_var + (m.expert0_mean - mu_fused)**2) + \
                            w1 * (m.expert1_var + (m.expert1_mean - mu_fused)**2)
                # Compute batch statistics of x passing through up to this layer
                # For simplicity, let's just pass x through the model and catch intermediate activations
                pass
                
        # Let's extract the activations of the first Conv layer to see
        # We can add a hook to conv1 and conv2 to get activations
        acts = {}
        def get_act(name):
            def hook(mod, inp, out):
                acts[name] = inp[0].detach()
            return hook
            
        h1 = base_model.conv1.register_forward_hook(get_act('conv1'))
        h2 = base_model.conv2.register_forward_hook(get_act('conv2'))
        
        base_model(x)
        
        h1.remove()
        h2.remove()
        
        # Calculate KL for conv1 input (first layer, base activations)
        bn1 = base_model.bn1
        batch_mean = acts['conv1'].mean(dim=(0, 2, 3), keepdim=True)
        batch_var = acts['conv1'].var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        
        mu_fused = w0 * bn1.expert0_mean + w1 * bn1.expert1_mean
        var_fused = w0 * (bn1.expert0_var + (bn1.expert0_mean - mu_fused)**2) + \
                    w1 * (bn1.expert1_var + (bn1.expert1_mean - mu_fused)**2)
                    
        kl = compute_kl_divergence(batch_mean, batch_var, mu_fused, var_fused)
        print(f"[{phase}] First BN layer KL: {kl:.4f}")

if __name__ == "__main__":
    main()
