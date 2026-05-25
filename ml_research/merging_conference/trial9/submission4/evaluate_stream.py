import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call
import numpy as np
import os
from models import SimpleCNN

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation running on device: {device}")

# 1. Load Checkpoints
print("Loading checkpoints...")
checkpoint_std_mnist = torch.load('checkpoints/standard_mnist.pt', map_location=device, weights_only=False)
checkpoint_std_fashion = torch.load('checkpoints/standard_fashion.pt', map_location=device, weights_only=False)
checkpoint_cos_mnist = torch.load('checkpoints/cosface_mnist.pt', map_location=device, weights_only=False)
checkpoint_cos_fashion = torch.load('checkpoints/cosface_fashion.pt', map_location=device, weights_only=False)

# 2. Reconstruct Stream
def build_stream():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = datasets.MNIST(root="data", train=False, download=False, transform=transform)
    fashion_test = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform)
    kmnist_test = datasets.KMNIST(root="data", train=False, download=False, transform=transform)
    
    loader_mnist = DataLoader(mnist_test, batch_size=64, shuffle=False)
    loader_fashion = DataLoader(fashion_test, batch_size=64, shuffle=False)
    loader_kmnist = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    batches = []
    
    # Batches 0-9: Clean MNIST
    mnist_iter = iter(loader_mnist)
    for _ in range(10):
        x, y = next(mnist_iter)
        batches.append((x.clone(), y.clone(), "Clean MNIST"))
        
    # Batches 10-19: Noisy MNIST
    for _ in range(10):
        x, y = next(mnist_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        batches.append((x_noisy, y.clone(), "Noisy MNIST"))
        
    # Batches 20-29: Clean FashionMNIST
    fashion_iter = iter(loader_fashion)
    for _ in range(10):
        x, y = next(fashion_iter)
        batches.append((x.clone(), y.clone(), "Clean Fashion"))
        
    # Batches 30-39: Noisy FashionMNIST
    for _ in range(10):
        x, y = next(fashion_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        batches.append((x_noisy, y.clone(), "Noisy Fashion"))
        
    # Batches 40-49: Novel KMNIST
    kmnist_iter = iter(loader_kmnist)
    for _ in range(10):
        x, y = next(kmnist_iter)
        batches.append((x.clone(), y.clone(), "Novel KMNIST"))
        
    return batches

# 3. Helpers for Distance Computation
def compute_euclidean_distances(features, prototypes):
    proto_tensor = torch.tensor(np.array([prototypes[c] for c in range(10)]), dtype=torch.float32, device=features.device)
    feat_sq = torch.sum(features**2, dim=1, keepdim=True)
    proto_sq = torch.sum(proto_tensor**2, dim=1, keepdim=True).t()
    cross = torch.matmul(features, proto_tensor.t())
    dist_matrix = feat_sq + proto_sq - 2.0 * cross
    min_d, _ = torch.min(dist_matrix, dim=1)
    return min_d

def compute_angular_distances(features, prototypes):
    proto_tensor = torch.tensor(np.array([prototypes[c] for c in range(10)]), dtype=torch.float32, device=features.device)
    proto_norm = F.normalize(proto_tensor, p=2, dim=1)
    feat_norm = F.normalize(features, p=2, dim=1)
    cosine_sim = torch.matmul(feat_norm, proto_norm.t())
    dist_matrix = 1.0 - cosine_sim
    min_d, _ = torch.min(dist_matrix, dim=1)
    return min_d

def get_hoyer_sparsity(x):
    x_flat = x.view(x.size(0), -1)
    x_pos = (x_flat + 1.0) / 2.0
    x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
    norm1 = torch.norm(x_denoised, p=1, dim=1)
    norm2 = torch.norm(x_denoised, p=2, dim=1)
    d = x_denoised.size(1)
    sparsity = (np.sqrt(d) - (norm1 / (norm2 + 1e-8))) / (np.sqrt(d) - 1.0)
    return sparsity.mean().item()

def set_bn_training(model, train_mode=True):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if train_mode:
                m.train()
            else:
                m.eval()

# Precompute offline sensitivities for Methods B, C, D using calibration samples
def compute_offline_sensitivities(model1, model2, calibration_mnist, calibration_fashion):
    model1.eval()
    model2.eval()
    
    sens1 = {}
    sens2 = {}
    
    # Compute for MNIST
    loader1 = DataLoader(Subset(calibration_mnist, list(range(256))), batch_size=256, shuffle=False)
    for x, y in loader1:
        x, y = x.to(device), y.to(device)
        # Enable grad tracking on parameters of model1
        for p in model1.parameters():
            p.requires_grad = True
        outputs = model1(x)
        loss = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
        # Compute gradients
        grads = torch.autograd.grad(loss, model1.parameters(), retain_graph=True, allow_unused=True)
        for name, param in model1.named_parameters():
            idx = list(model1.named_parameters()).index((name, param))
            if grads[idx] is not None:
                sens1[name] = torch.mean(grads[idx]**2).item()
            else:
                sens1[name] = 1e-4
                
    # Compute for FashionMNIST
    loader2 = DataLoader(Subset(calibration_fashion, list(range(256))), batch_size=256, shuffle=False)
    for x, y in loader2:
        x, y = x.to(device), y.to(device)
        for p in model2.parameters():
            p.requires_grad = True
        outputs = model2(x)
        loss = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
        grads = torch.autograd.grad(loss, model2.parameters(), retain_graph=True, allow_unused=True)
        for name, param in model2.named_parameters():
            idx = list(model2.named_parameters()).index((name, param))
            if grads[idx] is not None:
                sens2[name] = torch.mean(grads[idx]**2).item()
            else:
                sens2[name] = 1e-4
                
    # Reset grad requirements
    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False
        
    # Average and normalize joint sensitivities
    sens_joint = {}
    total_sens = 0.0
    for name in sens1.keys():
        sens_joint[name] = (sens1[name] + sens2[name]) / 2.0
        total_sens += sens_joint[name]
        
    for name in sens_joint.keys():
        sens_joint[name] /= (total_sens + 1e-8)
        
    return sens_joint

def evaluate_method(method_name, stream_batches):
    print(f"\nEvaluating: {method_name}...")
    
    # Initialize models and configs based on the method
    is_cosface = (method_name in ["Method D", "Method E"])
    
    model1 = SimpleCNN(is_cosface=is_cosface).to(device)
    model2 = SimpleCNN(is_cosface=is_cosface).to(device)
    
    if is_cosface:
        ckpt1 = checkpoint_cos_mnist
        ckpt2 = checkpoint_cos_fashion
    else:
        ckpt1 = checkpoint_std_mnist
        ckpt2 = checkpoint_std_fashion
        
    model1.load_state_dict(ckpt1['state_dict'])
    model2.load_state_dict(ckpt2['state_dict'])
    
    model1.eval()
    model2.eval()
    
    l2_proto1, sph_proto1 = ckpt1['l2_proto'], ckpt1['sph_proto']
    l2_proto2, sph_proto2 = ckpt2['l2_proto'], ckpt2['sph_proto']
    
    # Precomputed offline sensitivities (only computed once)
    offline_sens = None
    if method_name in ["Method B", "Method C", "Method D"]:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train = datasets.MNIST(root="data", train=True, download=False, transform=transform)
        fashion_train = datasets.FashionMNIST(root="data", train=True, download=False, transform=transform)
        offline_sens = compute_offline_sensitivities(model1, model2, mnist_train, fashion_train)
        
    batch_accuracies = []
    merging_coefficients = []
    
    for b_idx, (x, y, seg_name) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # Reset expert models to eval mode
        model1.eval()
        model2.eval()
        if method_name == "Method E":
            set_bn_training(model1, True)
            set_bn_training(model2, True)
            
        # 1. Routing Prior calculation
        with torch.no_grad():
            feat1 = model1.get_features(x)
            feat2 = model2.get_features(x)
        
        # Sparsity-Aware Gating (Method E only)
        if method_name == "Method E":
            sparsity = get_hoyer_sparsity(x)
            is_sparse = (sparsity >= 0.50)
        else:
            is_sparse = False
            
        # Determine SCTS type and active models for Method E
        active_model1 = model1
        active_model2 = model2
        active_l2_proto1, active_sph_proto1 = l2_proto1, sph_proto1
        active_l2_proto2, active_sph_proto2 = l2_proto2, sph_proto2
        
        if method_name == "Method E":
            if is_sparse:
                # Use standard models for sparse tasks
                active_model1 = SimpleCNN(is_cosface=False).to(device)
                active_model2 = SimpleCNN(is_cosface=False).to(device)
                active_model1.load_state_dict(checkpoint_std_mnist['state_dict'])
                active_model2.load_state_dict(checkpoint_std_fashion['state_dict'])
                active_model1.eval()
                active_model2.eval()
                set_bn_training(active_model1, True)
                set_bn_training(active_model2, True)
                active_l2_proto1 = checkpoint_std_mnist['l2_proto']
                active_l2_proto2 = checkpoint_std_fashion['l2_proto']
                
                # SCTS L2
                with torch.no_grad():
                    f1 = active_model1.get_features(x)
                    f2 = active_model2.get_features(x)
                d1 = compute_euclidean_distances(f1, active_l2_proto1).mean()
                d2 = compute_euclidean_distances(f2, active_l2_proto2).mean()
                temp = torch.abs(d1 - d2) / 3.0 + 150.0
                w1 = torch.exp(-d1 / temp) / (torch.exp(-d1 / temp) + torch.exp(-d2 / temp))
            else:
                # Use CosFace models and Angular routing for dense tasks
                d1 = compute_angular_distances(feat1, sph_proto1).mean()
                d2 = compute_angular_distances(feat2, sph_proto2).mean()
                temp = torch.abs(d1 - d2) / 3.0 + 0.04
                w1 = torch.exp(-d1 / temp) / (torch.exp(-d1 / temp) + torch.exp(-d2 / temp))
        else:
            if method_name in ["Method C", "Method D"]:
                d1 = compute_angular_distances(feat1, sph_proto1).mean()
                d2 = compute_angular_distances(feat2, sph_proto2).mean()
                temp = torch.abs(d1 - d2) / 3.0 + 0.04
                w1 = torch.exp(-d1 / temp) / (torch.exp(-d1 / temp) + torch.exp(-d2 / temp))
            else:
                d1 = compute_euclidean_distances(feat1, l2_proto1).mean()
                d2 = compute_euclidean_distances(feat2, l2_proto2).mean()
                temp = torch.abs(d1 - d2) / 3.0 + 150.0
                w1 = torch.exp(-d1 / temp) / (torch.exp(-d1 / temp) + torch.exp(-d2 / temp))
                
        w2 = 1.0 - w1
        p = w1.item()
        
        # 2. Initialize merging logit
        p_clamped = np.clip(p, 1e-4, 1.0 - 1e-4)
        w_global = torch.tensor(np.log(p_clamped / (1.0 - p_clamped)), requires_grad=True, device=device)
        
        # Layer offsets
        offsets = {}
        for name, param in active_model1.named_parameters():
            offsets[name] = torch.tensor(0.0, requires_grad=True, device=device)
                
        # 3. Create temporary merged model for test-time adaptation
        merged_model = SimpleCNN(is_cosface=(method_name == "Method D" or (method_name == "Method E" and not is_sparse))).to(device)
        merged_model.eval()
        if method_name == "Method E":
            set_bn_training(merged_model, True)
        
        # For Method E, compute sensitivities on-the-fly differentiably
        on_the_fly_sens = {}
        if method_name == "Method E":
            init_state = {}
            for name, param in active_model1.named_parameters():
                init_state[name] = w1 * param + w2 * dict(active_model2.named_parameters())[name]
                init_state[name].requires_grad_ = True
                
            w1_det = w1.clone().detach().to(device)
            w2_det = 1.0 - w1_det
            init_buffers = {}
            for name, buf in active_model1.named_buffers():
                if "running_mean" in name or "running_var" in name:
                    var_name = name.replace("running_mean", "running_var") if "running_mean" in name else name
                    mean_name = name.replace("running_var", "running_mean") if "running_var" in name else name
                    mu1, mu2 = dict(active_model1.named_buffers())[mean_name], dict(active_model2.named_buffers())[mean_name]
                    sig1, sig2 = dict(active_model1.named_buffers())[var_name], dict(active_model2.named_buffers())[var_name]
                    mu_fused = w1_det * mu1 + w2_det * mu2
                    sig_fused = w1_det * (sig1 + (mu1 - mu_fused)**2) + w2_det * (sig2 + (mu2 - mu_fused)**2)
                    if "running_mean" in name:
                        init_buffers[name] = mu_fused
                    else:
                        init_buffers[name] = sig_fused
                else:
                    init_buffers[name] = buf
            
            outputs = functional_call(merged_model, {**init_state, **init_buffers}, x)
            loss_sens = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
            
            # Compute gradients directly on parameter tensors in the dictionary
            grads_sens = torch.autograd.grad(loss_sens, init_state.values(), allow_unused=True)
            
            total_sens = 0.0
            for (name, _), grad in zip(init_state.items(), grads_sens):
                if grad is not None:
                    sens_val = torch.mean(grad**2).item()
                    on_the_fly_sens[name] = sens_val
                    total_sens += sens_val
                else:
                    on_the_fly_sens[name] = 1e-4
                    total_sens += 1e-4
                    
            for name in on_the_fly_sens.keys():
                on_the_fly_sens[name] /= (total_sens + 1e-8)
                
            merged_model.zero_grad()
            
        # TTA Steps
        N_steps = 5
        lr = 0.05
        beta = 1.5
        gamma = 0.02
        
        for step in range(N_steps):
            # Reconstruct merged weights differentiably
            merged_params = {}
            for name, param in active_model1.named_parameters():
                if name in offsets:
                    lambda_j = torch.sigmoid(w_global + offsets[name])
                    param2 = dict(active_model2.named_parameters())[name]
                    merged_params[name] = lambda_j * param + (1.0 - lambda_j) * param2
                else:
                    merged_params[name] = param
                    
            # Reconstruct merged buffers (non-differentiable)
            merged_buffers = {}
            lambda_global_detached = torch.sigmoid(w_global).detach()
            for name, buf in active_model1.named_buffers():
                if "running_mean" in name or "running_var" in name:
                    if method_name == "Method E":
                        # Soft BN Buffer Fusion using moment matching
                        var_name = name.replace("running_mean", "running_var") if "running_mean" in name else name
                        mean_name = name.replace("running_var", "running_mean") if "running_var" in name else name
                        
                        mu1, mu2 = dict(active_model1.named_buffers())[mean_name], dict(active_model2.named_buffers())[mean_name]
                        sig1, sig2 = dict(active_model1.named_buffers())[var_name], dict(active_model2.named_buffers())[var_name]
                        
                        mu_fused = lambda_global_detached * mu1 + (1.0 - lambda_global_detached) * mu2
                        sig_fused = lambda_global_detached * (sig1 + (mu1 - mu_fused)**2) + (1.0 - lambda_global_detached) * (sig2 + (mu2 - mu_fused)**2)
                        
                        if "running_mean" in name:
                            merged_buffers[name] = mu_fused
                        else:
                            merged_buffers[name] = sig_fused
                    else:
                        # Linear blend
                        buf2 = dict(active_model2.named_buffers())[name]
                        merged_buffers[name] = lambda_global_detached * buf + (1.0 - lambda_global_detached) * buf2
                else:
                    merged_buffers[name] = buf
                    
            # Forward pass via functional_call
            outputs = functional_call(merged_model, {**merged_params, **merged_buffers}, x)
            
            # Entropy loss
            L_ent = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
            
            # Prior regularization loss (KL divergence)
            lambda_global = torch.sigmoid(w_global)
            L_prior = beta * (w1 * torch.log(w1 / (lambda_global + 1e-8) + 1e-8) + w2 * torch.log(w2 / (1.0 - lambda_global + 1e-8) + 1e-8))
            
            # Coherence penalty
            L_coh = 0.0
            if method_name == "Method A":
                pass
            elif method_name in ["Method B", "Method C", "Method D"]:
                for name in offsets.keys():
                    sens_j = offline_sens.get(name, 1e-4)
                    L_coh += gamma * sens_j * torch.sum(offsets[name]**2)
            elif method_name == "Method E":
                for name in offsets.keys():
                    sens_j = on_the_fly_sens.get(name, 1e-4)
                    L_coh += gamma * sens_j * torch.sum(offsets[name]**2)
                    
            loss = L_ent + L_prior + L_coh
            
            # Compute gradients
            params_to_opt = [w_global] + list(offsets.values()) if method_name != "Method A" else [w_global]
            grads = torch.autograd.grad(loss, params_to_opt, allow_unused=True)
            
            # Update parameters with learning rate preconditioning
            with torch.no_grad():
                # Update w_global
                if grads[0] is not None:
                    w_global.copy_(w_global - lr * grads[0])
                
                if method_name != "Method A":
                    for name_idx, name in enumerate(offsets.keys()):
                        grad_val = grads[name_idx + 1]
                        if grad_val is not None:
                            if method_name in ["Method B", "Method C", "Method D"]:
                                sens_j = offline_sens.get(name, 1e-4)
                                offsets[name].copy_(offsets[name] - lr * (1.0 / (sens_j + 10**-2)) * grad_val)
                            else:
                                sens_j = on_the_fly_sens.get(name, 1e-4)
                                offsets[name].copy_(offsets[name] - lr * (1.0 / (sens_j + 0.01)) * grad_val)
                            
        # 4. Evaluation of the optimized merged model on this batch
        with torch.no_grad():
            # Final merged params reconstruction
            merged_params = {}
            for name, param in active_model1.named_parameters():
                if name in offsets:
                    lambda_j = torch.sigmoid(w_global + offsets[name])
                    param2 = dict(active_model2.named_parameters())[name]
                    merged_params[name] = lambda_j * param + (1.0 - lambda_j) * param2
                else:
                    merged_params[name] = param
                    
            # Final merged buffers reconstruction
            merged_buffers = {}
            lambda_global_detached = torch.sigmoid(w_global).detach()
            for name, buf in active_model1.named_buffers():
                if "running_mean" in name or "running_var" in name:
                    if method_name == "Method E":
                        var_name = name.replace("running_mean", "running_var") if "running_mean" in name else name
                        mean_name = name.replace("running_var", "running_mean") if "running_var" in name else name
                        
                        mu1, mu2 = dict(active_model1.named_buffers())[mean_name], dict(active_model2.named_buffers())[mean_name]
                        sig1, sig2 = dict(active_model1.named_buffers())[var_name], dict(active_model2.named_buffers())[var_name]
                        
                        mu_fused = lambda_global_detached * mu1 + (1.0 - lambda_global_detached) * mu2
                        sig_fused = lambda_global_detached * (sig1 + (mu1 - mu_fused)**2) + (1.0 - lambda_global_detached) * (sig2 + (mu2 - mu_fused)**2)
                        
                        if "running_mean" in name:
                            merged_buffers[name] = mu_fused
                        else:
                            merged_buffers[name] = sig_fused
                    else:
                        buf2 = dict(active_model2.named_buffers())[name]
                        merged_buffers[name] = lambda_global_detached * buf + (1.0 - lambda_global_detached) * buf2
                else:
                    merged_buffers[name] = buf
                    
            outputs = functional_call(merged_model, {**merged_params, **merged_buffers}, x)
            _, preds = outputs.max(1)
            correct = preds.eq(y).sum().item()
            acc = correct / x.size(0) * 100.0
            
        batch_accuracies.append(acc)
        merging_coefficients.append(torch.sigmoid(w_global).item())
        
        if (b_idx + 1) % 10 == 0:
            print(f"Batch {b_idx+1}/50 ({seg_name}) - Acc: {acc:.2f}% - Lambda: {torch.sigmoid(w_global).item():.4f}")
            
    mean_acc = np.mean(batch_accuracies)
    print(f"--> {method_name} Overall Mean Accuracy: {mean_acc:.2f}%")
    return batch_accuracies, merging_coefficients

# Load the stream batches
stream_batches = build_stream()

# Evaluate each method
results = {}
methods = [
    "Method A", # Fixed TTA + Reset
    "Method B", # CL W-Fisher + SCTS (L2)
    "Method C", # CL W-Fisher + A-SCTS
    "Method D", # CP-AM (Ours in paper 1)
    "Method E"  # BK-AHR (Ours)
]

for m in methods:
    accs, lambdas = evaluate_method(m, stream_batches)
    results[m] = {
        'accuracies': accs,
        'lambdas': lambdas
    }
    
# Print Summary Table
print("\n" + "="*80)
print(f"{'Method':<35} | {'MNIST Clean':<12} | {'MNIST Noisy':<12} | {'Fashion Clean':<14} | {'Fashion Noisy':<14} | {'KMNIST Novel':<12} | {'Overall':<8}")
print("="*80)

for m in methods:
    accs = results[m]['accuracies']
    c_mnist = np.mean(accs[0:10])
    n_mnist = np.mean(accs[10:20])
    c_fashion = np.mean(accs[20:30])
    n_fashion = np.mean(accs[30:40])
    n_kmnist = np.mean(accs[40:50])
    overall = np.mean(accs)
    print(f"{m:<35} | {c_mnist:11.2f}% | {n_mnist:11.2f}% | {c_fashion:13.2f}% | {n_fashion:13.2f}% | {n_kmnist:11.2f}% | {overall:7.2f}%")
print("="*80)

# Save results for plotting
torch.save(results, 'checkpoints/stream_results.pt')
print("Stream evaluation and results saving complete!")
