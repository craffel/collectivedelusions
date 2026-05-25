import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call
import copy

# Import SimpleCNN and other utilities from evaluate_ttmm
from evaluate_ttmm import (
    SimpleCNN, hoyer_sparsity, compute_entropy, get_distances,
    stable_softmax, set_bn_mode, fuse_bn_buffers, estimate_sensitivities
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_tta_custom(merged_model, params_mnist, params_fashion, initial_w_global, initial_deltas, 
                   x_batch, w1, w0, preconditioning_sensitivities=None, use_coherence=True, lr=0.01, 
                   use_sam=False, rho=0.05, optimizer_type="Adam"):
    
    # Set up learnable parameters
    w_global = torch.tensor(initial_w_global, requires_grad=True, device=x_batch.device)
    deltas = {}
    for name, delta_init in initial_deltas.items():
        deltas[name] = torch.tensor(delta_init, requires_grad=True, device=x_batch.device)
        
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam([w_global] + list(deltas.values()), lr=lr)
    else:
        optimizer = torch.optim.SGD([w_global] + list(deltas.values()), lr=lr)
        
    # Run 5 steps of TTA
    for step in range(5):
        if optimizer_type in ["Adam", "SGD"]:
            optimizer.zero_grad()
        
        # Helper to compute loss for given parameters
        def compute_loss(w_g, d_dict):
            merged_params = {}
            for name in params_mnist.keys():
                p0 = params_mnist[name]
                p1 = params_fashion[name]
                delta_j = d_dict[name] if name in d_dict else 0.0
                lambda_j = torch.sigmoid(w_g + delta_j)
                merged_params[name] = lambda_j * p1 + (1.0 - lambda_j) * p0
                
            out = functional_call(merged_model, merged_params, x_batch)
            L_entropy = compute_entropy(out)
            
            lambda_global = torch.sigmoid(w_g)
            L_kl = w1 * torch.log(w1 / (lambda_global + 1e-9) + 1e-9) + w0 * torch.log(w0 / (1.0 - lambda_global + 1e-9) + 1e-9)
            
            L_coherence = 0.0
            if use_coherence and preconditioning_sensitivities is not None:
                for name, delta_j in d_dict.items():
                    sens = preconditioning_sensitivities.get(name, 0.0)
                    L_coherence += sens * torch.sum(delta_j ** 2)
                    
            return L_entropy + 1.5 * L_kl + 0.02 * L_coherence

        if not use_sam:
            loss = compute_loss(w_global, deltas)
            loss.backward()
            optimizer.step()
        else:
            if optimizer_type == "Preconditioned_SGD":
                # Manual preconditioned SGD
                loss = compute_loss(w_global, deltas)
                loss.backward()
                
                with torch.no_grad():
                    dw = w_global.grad.clone() if w_global.grad is not None else torch.zeros_like(w_global)
                    d_deltas = {}
                    sq_sum = dw.item() ** 2
                    
                    for name, delta_param in deltas.items():
                        if delta_param.grad is not None:
                            sens = preconditioning_sensitivities.get(name, 0.0) if preconditioning_sensitivities is not None else 0.0
                            d_deltas[name] = delta_param.grad / (sens + 1e-4)
                            sq_sum += torch.sum(d_deltas[name] ** 2).item()
                        else:
                            d_deltas[name] = torch.zeros_like(delta_param)
                    
                    norm_D = np.sqrt(sq_sum + 1e-4)
                    eps_w = rho * dw / norm_D
                    eps_deltas = {name: rho * d / norm_D for name, d in d_deltas.items()}
                    
                    # Compute perturbed coordinates
                    w_g_perturbed = (w_global + eps_w).requires_grad_(True)
                    deltas_perturbed = {name: (param + eps_deltas.get(name, 0.0)).requires_grad_(True) for name, param in deltas.items()}
                    
                # Compute loss at perturbed coordinates
                loss_p = compute_loss(w_g_perturbed, deltas_perturbed)
                loss_p.backward()
                
                # Update original parameters using perturbed gradients with preconditioned learning rates
                with torch.no_grad():
                    if w_g_perturbed.grad is not None:
                        w_global -= lr * w_g_perturbed.grad
                    for name, delta_param in deltas.items():
                        p_grad = deltas_perturbed[name].grad
                        if p_grad is not None:
                            sens = preconditioning_sensitivities.get(name, 0.0) if preconditioning_sensitivities is not None else 0.0
                            # Preconditioned update: step is scaled down by sensitivities
                            delta_param -= lr * p_grad / (sens + 1e-4)
                            
                # Clear gradients manually
                if w_global.grad is not None: w_global.grad.zero_()
                for delta_param in deltas.values():
                    if delta_param.grad is not None: delta_param.grad.zero_()
            else:
                # Adam or SGD with SAM
                loss = compute_loss(w_global, deltas)
                loss.backward()
                
                with torch.no_grad():
                    dw = w_global.grad.clone() if w_global.grad is not None else torch.zeros_like(w_global)
                    d_deltas = {}
                    sq_sum = dw.item() ** 2
                    
                    for name, delta_param in deltas.items():
                        if delta_param.grad is not None:
                            sens = preconditioning_sensitivities.get(name, 0.0) if preconditioning_sensitivities is not None else 0.0
                            d_deltas[name] = delta_param.grad / (sens + 1e-4)
                            sq_sum += torch.sum(d_deltas[name] ** 2).item()
                        else:
                            d_deltas[name] = torch.zeros_like(delta_param)
                    
                    norm_D = np.sqrt(sq_sum + 1e-4)
                    eps_w = rho * dw / norm_D
                    eps_deltas = {name: rho * d / norm_D for name, d in d_deltas.items()}
                    
                with torch.no_grad():
                    w_global_perturbed = w_global + eps_w
                    deltas_perturbed = {}
                    for name, delta_param in deltas.items():
                        deltas_perturbed[name] = delta_param + eps_deltas.get(name, 0.0)
                        
                optimizer.zero_grad()
                w_global_p_tensor = w_global_perturbed.clone().requires_grad_(True)
                deltas_p_tensors = {name: val.clone().requires_grad_(True) for name, val in deltas_perturbed.items()}
                
                loss_p = compute_loss(w_global_p_tensor, deltas_p_tensors)
                loss_p.backward()
                
                with torch.no_grad():
                    if w_global.grad is not None and w_global_p_tensor.grad is not None:
                        w_global.grad.copy_(w_global_p_tensor.grad)
                    for name, delta_param in deltas.items():
                        if delta_param.grad is not None and deltas_p_tensors[name].grad is not None:
                            delta_param.grad.copy_(deltas_p_tensors[name].grad)
                            
                optimizer.step()
        
    final_params = {}
    with torch.no_grad():
        for name in params_mnist.keys():
            p0 = params_mnist[name]
            p1 = params_fashion[name]
            delta_j = deltas[name].detach() if name in deltas else 0.0
            lambda_j = torch.sigmoid(w_global.detach() + delta_j)
            final_params[name] = lambda_j * p1 + (1.0 - lambda_j) * p0
            
    return final_params

def evaluate_custom(test_batches, model_mnist, model_fashion, protos, device, 
                    use_sam=True, rho=0.05, optimizer_type="Adam"):
    
    params_mnist = {k: v.to(device) for k, v in model_mnist.named_parameters()}
    params_fashion = {k: v.to(device) for k, v in model_fashion.named_parameters()}
    
    proto_mnist_norm = protos["mnist_norm"].to(device)
    proto_fashion_norm = protos["fashion_norm"].to(device)
    
    phase_correct = [0] * 5
    phase_total = [0] * 5
    
    merged_model = SimpleCNN().to(device)
    
    for b_idx, (x_batch, y_batch) in enumerate(test_batches):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        phase_idx = b_idx // 10
        
        # Sparsity
        x_pos = (x_batch + 1.0) / 2.0
        x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
        S_hoyer = hoyer_sparsity(x_denoised)
        
        model_mnist.eval()
        model_fashion.eval()
        set_bn_mode(model_mnist, train=True)
        set_bn_mode(model_fashion, train=True)
        
        with torch.no_grad():
            feats_mnist = model_mnist.forward_features(x_batch)
            feats_fashion = model_fashion.forward_features(x_batch)
            logits_mnist = model_mnist(x_batch)
            logits_fashion = model_fashion(x_batch)
            H_mnist = compute_entropy(logits_mnist).item()
            H_fashion = compute_entropy(logits_fashion).item()
            H_avg = 0.5 * (H_mnist + H_fashion)
            
        # Euclidean
        feats_mnist_norm = feats_mnist / (feats_mnist.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        feats_fashion_norm = feats_fashion / (feats_fashion.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        d0_E = get_distances(feats_mnist_norm, proto_mnist_norm, "Euclidean")
        d1_E = get_distances(feats_fashion_norm, proto_fashion_norm, "Euclidean")
        D0_E, D1_E = d0_E.mean().item(), d1_E.mean().item()
        gap_E = abs(D0_E - D1_E)
        eps_stab_E = 0.08 / (1.0 + 5.0 * H_avg)
        tau_E = (gap_E / 3.0) + eps_stab_E
        w1_E, _ = stable_softmax(D0_E, D1_E, tau_E)
        
        # Angular
        d0_A = get_distances(feats_mnist, proto_mnist_norm, "Angular")
        d1_A = get_distances(feats_fashion, proto_fashion_norm, "Angular")
        D0_A, D1_A = d0_A.mean().item(), d1_A.mean().item()
        gap_A = abs(D0_A - D1_A)
        eps_stab_A = 0.04 / (1.0 + 5.0 * H_avg)
        tau_A = (gap_A / 3.0) + eps_stab_A
        w1_A, _ = stable_softmax(D0_A, D1_A, tau_A)
        
        lambda_blend = 1.0 / (1.0 + np.exp(-50.0 * (S_hoyer - 0.50)))
        w1 = lambda_blend * w1_E + (1.0 - lambda_blend) * w1_A
        w0 = 1.0 - w1
        
        # Enable TTBN
        set_bn_mode(merged_model, train=True)
        set_bn_mode(model_mnist, train=True)
        set_bn_mode(model_fashion, train=True)
        
        fused_buffers = fuse_bn_buffers(model_mnist, model_fashion, w1, w0)
        merged_model.load_state_dict(model_mnist.state_dict(), strict=False)
        for k, v in fused_buffers.items():
            merged_model.state_dict()[k].copy_(v)
            
        sensitivities = estimate_sensitivities(merged_model, params_mnist, params_fashion, w1, w0, x_batch)
        
        initial_w_global = np.log(w1 / (w0 + 1e-9) + 1e-9)
        initial_deltas = {name: 0.0 for name in params_mnist.keys()}
        
        lr_base = 0.05
        lr_t = lr_base / (1.0 + 5.0 * H_avg)
        
        final_params = run_tta_custom(merged_model, params_mnist, params_fashion, initial_w_global, initial_deltas,
                                     x_batch, w1, w0, preconditioning_sensitivities=sensitivities, 
                                     use_coherence=True, lr=lr_t, use_sam=use_sam, rho=rho, optimizer_type=optimizer_type)
        
        set_bn_mode(merged_model, train=True)
        with torch.no_grad():
            out = functional_call(merged_model, final_params, x_batch)
            _, predicted = out.max(1)
            correct = predicted.eq(y_batch).sum().item()
            
        phase_correct[phase_idx] += correct
        phase_total[phase_idx] += x_batch.size(0)
        
    accuracies = []
    for i in range(5):
        acc = 100.0 * phase_correct[i] / phase_total[i]
        accuracies.append(acc)
    overall_acc = np.mean(accuracies)
    return accuracies, overall_acc

def get_test_batches(seed, mnist_test, fmnist_test, kmnist_test):
    set_seed(seed)
    test_batches = []
    
    loader_mnist_clean = DataLoader(Subset(mnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_mnist_clean:
        test_batches.append((x, y))
        
    loader_mnist_noisy = DataLoader(Subset(mnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_mnist_noisy:
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    loader_fashion_clean = DataLoader(Subset(fmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_clean:
        test_batches.append((x, y))
        
    loader_fashion_noisy = DataLoader(Subset(fmnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_noisy:
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    loader_kmnist = DataLoader(Subset(kmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_kmnist:
        test_batches.append((x, y))
        
    return test_batches

def main():
    device = torch.device("cpu") # use cpu since it is extremely fast and robust
    
    # Load expert models
    model_mnist = SimpleCNN().to(device)
    model_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device))
    model_mnist.eval()
    
    model_fashion = SimpleCNN().to(device)
    model_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device))
    model_fashion.eval()
    
    # Load prototypes
    protos = torch.load("prototypes.pth", map_location=device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)
    
    seeds = [42, 43, 44, 45, 46]
    
    # Grid of configurations to sweep
    configs = [
        {"use_sam": True, "rho": 0.005, "optimizer_type": "Adam"},
        {"use_sam": True, "rho": 0.01, "optimizer_type": "Adam"},
        {"use_sam": True, "rho": 0.02, "optimizer_type": "Adam"},
        {"use_sam": True, "rho": 0.05, "optimizer_type": "Adam"}, # baseline
        {"use_sam": True, "rho": 0.1, "optimizer_type": "Adam"},
        {"use_sam": True, "rho": 0.01, "optimizer_type": "SGD"},
        {"use_sam": True, "rho": 0.02, "optimizer_type": "SGD"},
        {"use_sam": True, "rho": 0.05, "optimizer_type": "SGD"},
        {"use_sam": True, "rho": 0.01, "optimizer_type": "Preconditioned_SGD"},
        {"use_sam": True, "rho": 0.02, "optimizer_type": "Preconditioned_SGD"},
        {"use_sam": True, "rho": 0.05, "optimizer_type": "Preconditioned_SGD"},
        {"use_sam": True, "rho": 0.1, "optimizer_type": "Preconditioned_SGD"},
    ]
    
    print("Starting Sweep over SAM configurations...")
    for config in configs:
        overalls = []
        for seed in seeds:
            test_batches = get_test_batches(seed, mnist_test, fmnist_test, kmnist_test)
            _, overall = evaluate_custom(test_batches, model_mnist, model_fashion, protos, device, 
                                         use_sam=config["use_sam"], rho=config["rho"], optimizer_type=config["optimizer_type"])
            overalls.append(overall)
            
        mean_ov = np.mean(overalls)
        std_ov = np.std(overalls)
        print(f"Config: rho={config['rho']}, opt={config['optimizer_type']} => Mean Overall Accuracy = {mean_ov:.4f}% ± {std_ov:.4f}%")

if __name__ == "__main__":
    main()
