import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call
import numpy as np
from models import SimpleCNN

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Checkpoints
checkpoint_std_mnist = torch.load('checkpoints/standard_mnist.pt', map_location=device, weights_only=False)
checkpoint_std_fashion = torch.load('checkpoints/standard_fashion.pt', map_location=device, weights_only=False)
checkpoint_cos_mnist = torch.load('checkpoints/cosface_mnist.pt', map_location=device, weights_only=False)
checkpoint_cos_fashion = torch.load('checkpoints/cosface_fashion.pt', map_location=device, weights_only=False)

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
    
    mnist_iter = iter(loader_mnist)
    for _ in range(10):
        x, y = next(mnist_iter)
        batches.append((x.clone(), y.clone(), "Clean MNIST"))
        
    for _ in range(10):
        x, y = next(mnist_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        batches.append((x_noisy, y.clone(), "Noisy MNIST"))
        
    fashion_iter = iter(loader_fashion)
    for _ in range(10):
        x, y = next(fashion_iter)
        batches.append((x.clone(), y.clone(), "Clean Fashion"))
        
    for _ in range(10):
        x, y = next(fashion_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        batches.append((x_noisy, y.clone(), "Noisy Fashion"))
        
    kmnist_iter = iter(loader_kmnist)
    for _ in range(10):
        x, y = next(kmnist_iter)
        batches.append((x.clone(), y.clone(), "Novel KMNIST"))
        
    return batches

stream_batches = build_stream()

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

def evaluate_config_ttbn(use_ttbn_experts=False, use_ttbn_merged=False, damping=0.01):
    model1 = SimpleCNN(is_cosface=True).to(device)
    model2 = SimpleCNN(is_cosface=True).to(device)
    model1.load_state_dict(checkpoint_cos_mnist['state_dict'])
    model2.load_state_dict(checkpoint_cos_fashion['state_dict'])
    model1.eval()
    model2.eval()
    
    if use_ttbn_experts:
        set_bn_training(model1, True)
        set_bn_training(model2, True)
        
    l2_proto1, sph_proto1 = checkpoint_cos_mnist['l2_proto'], checkpoint_cos_mnist['sph_proto']
    l2_proto2, sph_proto2 = checkpoint_cos_fashion['l2_proto'], checkpoint_cos_fashion['sph_proto']
    
    batch_accuracies = []
    
    for b_idx, (x, y, seg_name) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            feat1 = model1.get_features(x)
            feat2 = model2.get_features(x)
        
        sparsity = get_hoyer_sparsity(x)
        is_sparse = (sparsity >= 0.50)
            
        active_model1 = model1
        active_model2 = model2
        active_l2_proto1, active_sph_proto1 = l2_proto1, sph_proto1
        active_l2_proto2, active_sph_proto2 = l2_proto2, sph_proto2
        
        if is_sparse:
            active_model1 = SimpleCNN(is_cosface=False).to(device)
            active_model2 = SimpleCNN(is_cosface=False).to(device)
            active_model1.load_state_dict(checkpoint_std_mnist['state_dict'])
            active_model2.load_state_dict(checkpoint_std_fashion['state_dict'])
            active_model1.eval()
            active_model2.eval()
            if use_ttbn_experts:
                set_bn_training(active_model1, True)
                set_bn_training(active_model2, True)
            active_l2_proto1 = checkpoint_std_mnist['l2_proto']
            active_l2_proto2 = checkpoint_std_fashion['l2_proto']
            
            with torch.no_grad():
                f1 = active_model1.get_features(x)
                f2 = active_model2.get_features(x)
            d1 = compute_euclidean_distances(f1, active_l2_proto1).mean()
            d2 = compute_euclidean_distances(f2, active_l2_proto2).mean()
            temp = torch.abs(d1 - d2) / 3.0 + 150.0
            w1 = torch.exp(-d1 / temp) / (torch.exp(-d1 / temp) + torch.exp(-d2 / temp))
        else:
            d1 = compute_angular_distances(feat1, sph_proto1).mean()
            d2 = compute_angular_distances(feat2, sph_proto2).mean()
            temp = torch.abs(d1 - d2) / 3.0 + 0.04
            w1 = torch.exp(-d1 / temp) / (torch.exp(-d1 / temp) + torch.exp(-d2 / temp))
            
        w2 = 1.0 - w1
        p = w1.item()
        
        p_clamped = np.clip(p, 1e-4, 1.0 - 1e-4)
        w_global = torch.tensor(np.log(p_clamped / (1.0 - p_clamped)), requires_grad=True, device=device)
        
        offsets = {}
        for name, param in active_model1.named_parameters():
            offsets[name] = torch.tensor(0.0, requires_grad=True, device=device)
                
        merged_model = SimpleCNN(is_cosface=(not is_sparse)).to(device)
        merged_model.eval()
        if use_ttbn_merged:
            set_bn_training(merged_model, True)
            
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
        grads_sens = torch.autograd.grad(loss_sens, init_state.values(), allow_unused=True)
        
        on_the_fly_sens = {}
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
        
        N_steps = 5
        lr = 0.05
        beta = 1.5
        gamma = 0.02
        
        for step in range(N_steps):
            merged_params = {}
            for name, param in active_model1.named_parameters():
                if name in offsets:
                    lambda_j = torch.sigmoid(w_global + offsets[name])
                    param2 = dict(active_model2.named_parameters())[name]
                    merged_params[name] = lambda_j * param + (1.0 - lambda_j) * param2
                else:
                    merged_params[name] = param
                    
            merged_buffers = {}
            lambda_global_detached = torch.sigmoid(w_global).detach()
            for name, buf in active_model1.named_buffers():
                if "running_mean" in name or "running_var" in name:
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
                    merged_buffers[name] = buf
                    
            outputs = functional_call(merged_model, {**merged_params, **merged_buffers}, x)
            L_ent = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
            
            lambda_global = torch.sigmoid(w_global)
            L_prior = beta * (w1 * torch.log(w1 / (lambda_global + 1e-8) + 1e-8) + w2 * torch.log(w2 / (1.0 - lambda_global + 1e-8) + 1e-8))
            
            L_coh = 0.0
            for name in offsets.keys():
                sens_j = on_the_fly_sens.get(name, 1e-4)
                L_coh += gamma * sens_j * torch.sum(offsets[name]**2)
                
            loss = L_ent + L_prior + L_coh
            
            params_to_opt = [w_global] + list(offsets.values())
            grads = torch.autograd.grad(loss, params_to_opt, allow_unused=True)
            
            with torch.no_grad():
                if grads[0] is not None:
                    w_global.copy_(w_global - lr * grads[0])
                
                for name_idx, name in enumerate(offsets.keys()):
                    grad_val = grads[name_idx + 1]
                    if grad_val is not None:
                        sens_j = on_the_fly_sens.get(name, 1e-4)
                        offsets[name].copy_(offsets[name] - lr * (1.0 / (sens_j + damping)) * grad_val)
                            
        with torch.no_grad():
            merged_params = {}
            for name, param in active_model1.named_parameters():
                if name in offsets:
                    lambda_j = torch.sigmoid(w_global + offsets[name])
                    param2 = dict(active_model2.named_parameters())[name]
                    merged_params[name] = lambda_j * param + (1.0 - lambda_j) * param2
                else:
                    merged_params[name] = param
                    
            merged_buffers = {}
            lambda_global_detached = torch.sigmoid(w_global).detach()
            for name, buf in active_model1.named_buffers():
                if "running_mean" in name or "running_var" in name:
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
                    merged_buffers[name] = buf
                    
            outputs = functional_call(merged_model, {**merged_params, **merged_buffers}, x)
            _, preds = outputs.max(1)
            correct = preds.eq(y).sum().item()
            acc = correct / x.size(0) * 100.0
            
        batch_accuracies.append(acc)
        
    return batch_accuracies

print("Starting Test-Time Batch Normalization (TTBN) Damping Sweep...")
dampings = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
for d in dampings:
    accs = evaluate_config_ttbn(use_ttbn_experts=True, use_ttbn_merged=True, damping=d)
    c_mnist = np.mean(accs[0:10])
    n_mnist = np.mean(accs[10:20])
    c_fashion = np.mean(accs[20:30])
    n_fashion = np.mean(accs[30:40])
    n_kmnist = np.mean(accs[40:50])
    overall = np.mean(accs)
    print(f"Damping={d:<6} | MNIST_C: {c_mnist:.2f}% | MNIST_N: {n_mnist:.2f}% | Fash_C: {c_fashion:.2f}% | Fash_N: {n_fashion:.2f}% | KMNIST: {n_kmnist:.2f}% | OVERALL: {overall:.2f}%")
