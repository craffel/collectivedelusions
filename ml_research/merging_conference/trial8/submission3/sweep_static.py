import torch
import numpy as np
import os
from run_experiment import SimpleCNN, AdaKLBatchNorm2D, convert_to_adakl_bn, precompute_prototypes, compute_expert_distances, compute_entropy, apply_soft_bn_stats, merge_weights, merge_layer_weights_adaptive
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def run_evaluation_static(alpha):
    torch.manual_seed(42)
    np.random.seed(42)
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
            
            # Monkey-patch forward with static alpha
            def custom_forward(self, x_in):
                if self.use_adakl and self.w is not None:
                    w0, w1 = self.w[0], self.w[1]
                    mu_fused = w0 * self.expert0_mean + w1 * self.expert1_mean
                    var_fused = w0 * (self.expert0_var + (self.expert0_mean - mu_fused)**2) + \
                                w1 * (self.expert1_var + (self.expert1_mean - mu_fused)**2)
                    
                    batch_mean = x_in.mean(dim=(0, 2, 3), keepdim=True)
                    batch_var = x_in.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
                    
                    mu_fused_bc = mu_fused.view(1, -1, 1, 1)
                    var_fused_bc = var_fused.view(1, -1, 1, 1)
                    
                    # Static alpha
                    mean_adapted = alpha * batch_mean + (1.0 - alpha) * mu_fused_bc
                    var_adapted = alpha * batch_var + (1.0 - alpha) * var_fused_bc
                    
                    x_scaled = (x_in - mean_adapted) / torch.sqrt(var_adapted + self.eps)
                    if self.affine:
                        return x_scaled * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
                    return x_scaled
                else:
                    mean = self.running_mean.view(1, -1, 1, 1)
                    var = self.running_var.view(1, -1, 1, 1)
                    x_norm = (x_in - mean) / torch.sqrt(var + self.eps)
                    if self.affine:
                        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
                    return x_norm
            
            import types
            m.forward = types.MethodType(custom_forward, m)
            
    # Re-instantiate expert models for prototype precomputation
    expert0_m = SimpleCNN()
    expert0_m.load_state_dict(expert0_state)
    expert1_m = SimpleCNN()
    expert1_m.load_state_dict(expert1_state)
    prototypes0 = precompute_prototypes(expert0_m, mnist_loader, num_samples=256)
    prototypes1 = precompute_prototypes(expert1_m, fmnist_loader, num_samples=256)
    
    # Task stream
    stream_batches = []
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append((x, y, "Clean MNIST"))
    for _ in range(10):
        x, y = next(mnist_iter)
        x_raw = x * 0.3081 + 0.1307
        noise = torch.randn_like(x_raw) * 0.15
        x_noisy = (torch.clamp(x_raw + noise, 0.0, 1.0) - 0.1307) / 0.3081
        stream_batches.append((x_noisy, y, "Noisy MNIST"))
        
    fmnist_iter = iter(fmnist_loader)
    for _ in range(10):
        x, y = next(fmnist_iter)
        stream_batches.append((x, y, "Clean Fashion"))
    for _ in range(10):
        x, y = next(fmnist_iter)
        x_raw = x * 0.3081 + 0.1307
        noise = torch.randn_like(x_raw) * 0.15
        x_noisy = (torch.clamp(x_raw + noise, 0.0, 1.0) - 0.1307) / 0.3081
        stream_batches.append((x_noisy, y, "Noisy Fashion"))
        
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append((x, y, "Novel KMNIST"))
        
    correct_by_phase = {p: 0 for p in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]}
    total_by_phase = {p: 0 for p in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]}
    
    for b_idx, (x, y, phase) in enumerate(stream_batches):
        for m in base_model.modules():
            if isinstance(m, AdaKLBatchNorm2D):
                m.use_adakl = True
                m.w = None
                
        D0, D1 = compute_expert_distances(x, base_model, expert0_state, expert1_state, prototypes0, prototypes1)
        delta = abs(D0 - D1)
        tau_self = (delta / 5.0) + 0.01
        w0 = 1.0 / (1.0 + np.exp((D0 - D1) / tau_self))
        w1 = 1.0 - w0
        w_prior = torch.tensor([w0, w1])
        
        base_model.load_state_dict(expert0_state, strict=False)
        with torch.no_grad():
            H0 = compute_entropy(base_model(x)).item()
        base_model.load_state_dict(expert1_state, strict=False)
        with torch.no_grad():
            H1 = compute_entropy(base_model(x)).item()
        H_bar = 0.5 * (H0 + H1)
        
        if H_bar > 1.2:
            w0, w1 = 0.5, 0.5
            w_prior = torch.tensor([0.5, 0.5])
            
        for m in base_model.modules():
            if isinstance(m, AdaKLBatchNorm2D):
                m.use_adakl = True
                m.w = w_prior
                
        p_init = np.clip(w0, 1e-4, 1.0 - 1e-4)
        w_param = torch.tensor([np.log(p_init / (1.0 - p_init))] * 4, requires_grad=True)
        
        merge_weights(base_model, expert0_state, expert1_state, w0, w1)
        hooks = []
        from run_experiment import get_forward_hook, get_backward_hook, compute_layer_sensitivities
        hooks.append(base_model.conv1.register_forward_hook(get_forward_hook('conv1')))
        hooks.append(base_model.conv1.register_full_backward_hook(get_backward_hook('conv1')))
        hooks.append(base_model.conv2.register_forward_hook(get_forward_hook('conv2')))
        hooks.append(base_model.conv2.register_full_backward_hook(get_backward_hook('conv2')))
        hooks.append(base_model.fc1.register_forward_hook(get_forward_hook('fc1')))
        hooks.append(base_model.fc1.register_full_backward_hook(get_backward_hook('fc1')))
        hooks.append(base_model.fc2.register_forward_hook(get_forward_hook('fc2')))
        hooks.append(base_model.fc2.register_full_backward_hook(get_backward_hook('fc2')))
        
        logits = base_model(x)
        loss_ent = compute_entropy(logits)
        base_model.zero_grad()
        loss_ent.backward()
        
        sensitivities = compute_layer_sensitivities(base_model)
        sens_vals = list(sensitivities.values())
        if len(sens_vals) > 0 and sum(sens_vals) > 0:
            mean_sens = sum(sens_vals) / len(sens_vals)
            for k in sensitivities:
                sensitivities[k] /= mean_sens
                
        for h in hooks:
            h.remove()
            
        eta = 0.05
        beta_reg = 1.5
        layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
        for step in range(5):
            lambdas = torch.sigmoid(w_param)
            merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas)
            logits = base_model(x)
            loss_ent = compute_entropy(logits)
            loss_kl = 0.0
            for l_idx in range(4):
                lam = lambdas[l_idx]
                loss_kl += w_prior[0] * torch.log(w_prior[0] / (lam + 1e-10) + 1e-10) + \
                          w_prior[1] * torch.log(w_prior[1] / (1.0 - lam + 1e-10) + 1e-10)
            loss_total = loss_ent + beta_reg * loss_kl
            
            if w_param.grad is not None:
                w_param.grad.zero_()
            loss_total.backward()
            
            with torch.no_grad():
                grad = w_param.grad.clone()
                scale_eps = 1e-5
                for l_idx, l_name in enumerate(layer_names):
                    sens = sensitivities.get(l_name, 1.0)
                    grad[l_idx] /= (sens + scale_eps)
                grad = torch.clamp(grad, -1.0, 1.0)
                w_param -= eta * grad
                
        with torch.no_grad():
            lambdas = torch.sigmoid(w_param)
            merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas)
            
        base_model.eval()
        with torch.no_grad():
            logits = base_model(x)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            
        correct_by_phase[phase] += correct
        total_by_phase[phase] += x.size(0)
        
    results = []
    for phase in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]:
        acc = correct_by_phase[phase] / total_by_phase[phase] * 100
        results.append(acc)
    return results

def main():
    print("--- SWEEPING STATIC ALPHA VALUES ---")
    alphas = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50]
    for alpha in alphas:
        res = run_evaluation_static(alpha)
        print(f"Alpha: {alpha:.2f} | CleanM: {res[0]:.2f}% | NoisyM: {res[1]:.2f}% | CleanF: {res[2]:.2f}% | NoisyF: {res[3]:.2f}% | KMNIST: {res[4]:.2f}%")

if __name__ == "__main__":
    main()
