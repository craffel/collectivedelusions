import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import copy
import json

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, s=30.0, m=0.35):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)
        
        self.use_cosface = use_cosface
        if self.use_cosface:
            self.classifier_weight = nn.Parameter(torch.randn(10, 128))
            nn.init.xavier_uniform_(self.classifier_weight)
            self.s = s
            self.m = m
        else:
            self.classifier = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        
        feat = self.dropout(F.relu(self.bn3(self.fc1(x))))
        
        if return_features:
            return feat
            
        if self.use_cosface:
            feat_norm = F.normalize(feat, p=2, dim=1)
            weight_norm = F.normalize(self.classifier_weight, p=2, dim=1)
            cos_theta = F.linear(feat_norm, weight_norm)
            
            if self.training:
                return cos_theta
            else:
                return self.s * cos_theta
        else:
            logits = self.classifier(feat)
            return logits

def get_stream(mnist_test, fmnist_test, kmnist_test, batch_size=64):
    torch.manual_seed(42)  # Ensure reproducible noise addition
    stream_batches = []
    
    # Phase 0: Batches 0-9 (Clean MNIST)
    for b in range(10):
        imgs = []
        lbls = []
        for i in range(batch_size):
            img, lbl = mnist_test[b * batch_size + i]
            imgs.append(img)
            lbls.append(lbl)
        stream_batches.append((torch.stack(imgs), torch.tensor(lbls), "MNIST_Clean"))
        
    # Phase 1: Batches 10-19 (Noisy MNIST)
    for b in range(10):
        imgs = []
        lbls = []
        for i in range(batch_size):
            img, lbl = mnist_test[(10 + b) * batch_size + i]
            noise = torch.randn_like(img) * 0.6
            img_noisy = torch.clamp(img + noise, -1.0, 1.0)
            imgs.append(img_noisy)
            lbls.append(lbl)
        stream_batches.append((torch.stack(imgs), torch.tensor(lbls), "MNIST_Noisy"))
        
    # Phase 2: Batches 20-29 (Clean FashionMNIST)
    for b in range(10):
        imgs = []
        lbls = []
        for i in range(batch_size):
            img, lbl = fmnist_test[b * batch_size + i]
            imgs.append(img)
            lbls.append(lbl)
        stream_batches.append((torch.stack(imgs), torch.tensor(lbls), "Fashion_Clean"))
        
    # Phase 3: Batches 30-39 (Noisy FashionMNIST)
    for b in range(10):
        imgs = []
        lbls = []
        for i in range(batch_size):
            img, lbl = fmnist_test[(10 + b) * batch_size + i]
            noise = torch.randn_like(img) * 0.6
            img_noisy = torch.clamp(img + noise, -1.0, 1.0)
            imgs.append(img_noisy)
            lbls.append(lbl)
        stream_batches.append((torch.stack(imgs), torch.tensor(lbls), "Fashion_Noisy"))
        
    # Phase 4: Batches 40-49 (Novel KMNIST)
    for b in range(10):
        imgs = []
        lbls = []
        for i in range(batch_size):
            img, lbl = kmnist_test[b * batch_size + i]
            imgs.append(img)
            lbls.append(lbl)
        stream_batches.append((torch.stack(imgs), torch.tensor(lbls), "KMNIST_Novel"))
        
    return stream_batches

def compute_hoyer_sparsity(x):
    # Map normalized x in [-1, 1] to positive [0, 1]
    x_pos = (x + 1.0) / 2.0
    # Apply thresholding
    x_denoised = (x_pos > 0.35).float() * x_pos
    # Flatten
    x_flat = x_denoised.view(x_denoised.size(0), -1)
    d = x_flat.size(1)
    l1 = torch.norm(x_flat, p=1, dim=1)
    l2 = torch.norm(x_flat, p=2, dim=1) + 1e-8
    
    sparsity = (np.sqrt(d) - (l1 / l2)) / (np.sqrt(d) - 1.0)
    return sparsity.mean().item()

def precompute_prototypes(expert_model, train_dataset, device):
    expert_model.eval()
    expert_model.to(device)
    
    class_features = {c: [] for c in range(10)}
    counts = {c: 0 for c in range(10)}
    for i in range(len(train_dataset)):
        img, lbl = train_dataset[i]
        if counts[lbl] < 50:
            class_features[lbl].append(img)
            counts[lbl] += 1
        if all(c >= 50 for c in counts.values()):
            break
            
    prototypes = []
    with torch.no_grad():
        for c in range(10):
            imgs = torch.stack(class_features[c]).to(device)
            feats = expert_model(imgs, return_features=True)
            feats_norm = F.normalize(feats, p=2, dim=1)
            mean_feat = feats_norm.mean(dim=0)
            prototype = F.normalize(mean_feat, p=2, dim=0)
            prototypes.append(prototype)
            
    return torch.stack(prototypes)

def get_fused_bn_buffers(expert_0_dict, expert_1_dict, w_global, buffer_names):
    fused_dict = {}
    lambda_det = torch.sigmoid(w_global).item()
    for name in buffer_names:
        if name.endswith("running_mean"):
            var_name = name.replace("running_mean", "running_var")
            mean_0 = expert_0_dict[name]
            mean_1 = expert_1_dict[name]
            var_0 = expert_0_dict[var_name]
            var_1 = expert_1_dict[var_name]
            
            mean_fused = (1.0 - lambda_det) * mean_0 + lambda_det * mean_1
            var_fused = (1.0 - lambda_det) * (var_0 + (mean_0 - mean_fused)**2) + lambda_det * (var_1 + (mean_1 - mean_fused)**2)
            
            fused_dict[name] = mean_fused
            fused_dict[var_name] = var_fused
        elif name.endswith("num_batches_tracked"):
            fused_dict[name] = expert_0_dict[name]
    return fused_dict

def evaluate_hyperparameters(method_name, stream_batches, mnist_train, fmnist_train, device,
                               rho=0.05, eps_precond=0.1, beta=1.0, gamma_c=1.0, eta=0.05):
    torch.manual_seed(42)
    
    expert_0_std = SimpleCNN(use_cosface=False).to(device)
    expert_1_std = SimpleCNN(use_cosface=False).to(device)
    expert_0_cos = SimpleCNN(use_cosface=True).to(device)
    expert_1_cos = SimpleCNN(use_cosface=True).to(device)
    
    expert_0_std.load_state_dict(torch.load("checkpoints/expert_0_standard.pt", map_location=device))
    expert_1_std.load_state_dict(torch.load("checkpoints/expert_1_standard.pt", map_location=device))
    expert_0_cos.load_state_dict(torch.load("checkpoints/expert_0_cosface.pt", map_location=device))
    expert_1_cos.load_state_dict(torch.load("checkpoints/expert_1_cosface.pt", map_location=device))
    
    prototypes_std_0 = precompute_prototypes(expert_0_std, mnist_train, device)
    prototypes_std_1 = precompute_prototypes(expert_1_std, fmnist_train, device)
    prototypes_cos_0 = precompute_prototypes(expert_0_cos, mnist_train, device)
    prototypes_cos_1 = precompute_prototypes(expert_1_cos, fmnist_train, device)
    
    base_model_std = SimpleCNN(use_cosface=False).to(device)
    base_model_cos = SimpleCNN(use_cosface=True).to(device)
    
    param_names_std = [name for name, _ in base_model_std.named_parameters()]
    buffer_names_std = [name for name, _ in base_model_std.named_buffers()]
    param_names_cos = [name for name, _ in base_model_cos.named_parameters()]
    buffer_names_cos = [name for name, _ in base_model_cos.named_buffers()]
    
    batch_accuracies = []
    
    for t, (X_t, Y_t, phase_name) in enumerate(stream_batches):
        X_t, Y_t = X_t.to(device), Y_t.to(device)
        
        sparsity = compute_hoyer_sparsity(X_t)
        
        if sparsity >= 0.50:
            routing_type = 0 # Euclidean
            active_expert_0 = expert_0_std
            active_expert_1 = expert_1_std
            active_prototypes_0 = prototypes_std_0
            active_prototypes_1 = prototypes_std_1
            base_model = base_model_std
            param_names = param_names_std
            buffer_names = buffer_names_std
        else:
            routing_type = 1 # Angular
            active_expert_0 = expert_0_cos
            active_expert_1 = expert_1_cos
            active_prototypes_0 = prototypes_cos_0
            active_prototypes_1 = prototypes_cos_1
            base_model = base_model_cos
            param_names = param_names_cos
            buffer_names = buffer_names_cos
                
        active_expert_0.eval()
        active_expert_1.eval()
        with torch.no_grad():
            feat_0 = active_expert_0(X_t, return_features=True)
            feat_1 = active_expert_1(X_t, return_features=True)
            
            out_0 = active_expert_0(X_t)
            out_1 = active_expert_1(X_t)
            p_0 = F.softmax(out_0, dim=1)
            p_1 = F.softmax(out_1, dim=1)
            H_0 = -torch.sum(p_0 * torch.log(p_0 + 1e-8), dim=1).mean()
            H_1 = -torch.sum(p_1 * torch.log(p_1 + 1e-8), dim=1).mean()
            Havg = 0.5 * (H_0 + H_1)
            
        d0_all, d1_all = [], []
        feat_0_norm = F.normalize(feat_0, p=2, dim=1)
        feat_1_norm = F.normalize(feat_1, p=2, dim=1)
        
        for i in range(X_t.size(0)):
            if routing_type == 0:
                d0_i = torch.min(torch.norm(feat_0_norm[i] - active_prototypes_0, p=2, dim=1) ** 2)
                d1_i = torch.min(torch.norm(feat_1_norm[i] - active_prototypes_1, p=2, dim=1) ** 2)
            else:
                d0_i = torch.min(1.0 - torch.mv(active_prototypes_0, feat_0_norm[i]))
                d1_i = torch.min(1.0 - torch.mv(active_prototypes_1, feat_1_norm[i]))
            d0_all.append(d0_i)
            d1_all.append(d1_i)
            
        D0 = torch.stack(d0_all).mean()
        D1 = torch.stack(d1_all).mean()
        gap = torch.abs(D0 - D1)
        
        gamma_dun = 2.0
        if routing_type == 0:
            eps_stab_scts = 0.08 / (1.0 + gamma_dun * Havg.item())
        else:
            eps_stab_scts = 0.04 / (1.0 + gamma_dun * Havg.item())
            
        tau = (gap.item() / 3.0) + eps_stab_scts
        w1_prior = np.exp(-D1.item() / tau) / (np.exp(-D0.item() / tau) + np.exp(-D1.item() / tau) + 1e-8)
        w0_prior = 1.0 - w1_prior
        
        w_global = torch.tensor(np.log(w1_prior / (w0_prior + 1e-8)), requires_grad=True, device=device)
        delta = {name: torch.tensor(0.0, requires_grad=True, device=device) for name in param_names}
        
        opt_params = [w_global] + list(delta.values())
        
        exp0_params = dict(active_expert_0.named_parameters())
        exp1_params = dict(active_expert_1.named_parameters())
        exp_buffers_0 = dict(active_expert_0.named_buffers())
        exp_buffers_1 = dict(active_expert_1.named_buffers())
        
        # Determine Preconditioning Sensitivities
        if method_name == "Method F (Weight-SAM)":
            # Estimate on-the-fly preconditioning sensitivity wrt initial weights
            init_w_global = w_global.detach()
            init_params_dict = {}
            for name in param_names:
                lambda_j_init = torch.sigmoid(init_w_global)
                p0 = exp0_params[name]
                p1 = exp1_params[name]
                init_params_dict[name] = ((1.0 - lambda_j_init) * p0 + lambda_j_init * p1).detach().clone().requires_grad_(True)
                
            fused_buffers_init = get_fused_bn_buffers(exp_buffers_0, exp_buffers_1, init_w_global, buffer_names)
            logits_init = torch.func.functional_call(base_model, (init_params_dict, fused_buffers_init), X_t)
            probs_init = F.softmax(logits_init, dim=1)
            L_entropy_init = -torch.sum(probs_init * torch.log(probs_init + 1e-8), dim=1).mean()
            grad_init_weights = torch.autograd.grad(L_entropy_init, init_params_dict.values())
            
            F_sens = {}
            F_sum = 0.0
            for name, g in zip(param_names, grad_init_weights):
                F_j = torch.mean(g ** 2).item()
                F_sens[name] = F_j
                F_sum += F_j
            F_norm = {name: F_sens[name] / (F_sum + 1e-8) for name in param_names}
            
        elif method_name == "Method G (DPS-SAM)":
            # Direct Parameter Sensitivity: Estimate sensitivity wrt delta directly
            init_w_global = w_global.detach()
            init_params_dict = {}
            for name in param_names:
                lambda_j_init = torch.sigmoid(init_w_global + delta[name])
                p0 = exp0_params[name]
                p1 = exp1_params[name]
                init_params_dict[name] = (1.0 - lambda_j_init) * p0 + lambda_j_init * p1
                
            fused_buffers_init = get_fused_bn_buffers(exp_buffers_0, exp_buffers_1, init_w_global, buffer_names)
            logits_init = torch.func.functional_call(base_model, (init_params_dict, fused_buffers_init), X_t)
            probs_init = F.softmax(logits_init, dim=1)
            L_entropy_init = -torch.sum(probs_init * torch.log(probs_init + 1e-8), dim=1).mean()
            grad_init = torch.autograd.grad(L_entropy_init, list(delta.values()))
            
            F_sens = {}
            F_sum = 0.0
            for name, g in zip(param_names, grad_init):
                F_j = (g.item()) ** 2
                F_sens[name] = F_j
                F_sum += F_j
            F_norm = {name: F_sens[name] / (F_sum + 1e-8) for name in param_names}
            
        # Adaptive learning rate (EALR)
        gamma_ealr = 5.0
        eta_t = eta / (1.0 + gamma_ealr * Havg.item())
        
        Nstep = 5
        
        for step in range(Nstep):
            # ----------------- FIRST PASS -----------------
            params_dict = {}
            for name in param_names:
                lambda_j = torch.sigmoid(w_global + delta[name])
                params_dict[name] = (1.0 - lambda_j) * exp0_params[name] + lambda_j * exp1_params[name]
                
            fused_buffers = get_fused_bn_buffers(exp_buffers_0, exp_buffers_1, w_global, buffer_names)
            
            logits = torch.func.functional_call(base_model, (params_dict, fused_buffers), X_t)
            probs = F.softmax(logits, dim=1)
            L_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            mean_lambda = torch.stack([torch.sigmoid(w_global + delta[name]) for name in param_names]).mean()
            L_KL = mean_lambda * torch.log(mean_lambda / (w1_prior + 1e-8)) + (1.0 - mean_lambda) * torch.log((1.0 - mean_lambda) / (w0_prior + 1e-8))
            
            L_coherence = gamma_c * sum(F_norm[name] * (delta[name] ** 2) for name in param_names)
            
            L_total = L_entropy + beta * L_KL + L_coherence
            
            grads = torch.autograd.grad(L_total, opt_params)
            g_w = grads[0]
            g_delta = {name: grads[i+1] for i, name in enumerate(param_names)}
            
            # ----------------- PERTURBATION STEP -----------------
            d_w = g_w
            d_delta = {name: g_delta[name] / (F_norm[name] + eps_precond) for name in param_names}
            
            D_norm_sq = d_w ** 2 + sum(torch.sum(d_delta[name] ** 2) for name in param_names) + eps_precond
            D_norm = torch.sqrt(D_norm_sq)
            
            epsilon_w = rho * d_w / D_norm
            epsilon_delta = {name: rho * d_delta[name] / D_norm for name in param_names}
            
            w_perturbed = w_global + epsilon_w
            delta_perturbed = {name: delta[name] + epsilon_delta[name] for name in param_names}
            
            # ----------------- SECOND PASS -----------------
            params_dict_perturbed = {}
            for name in param_names:
                lambda_j_perturbed = torch.sigmoid(w_perturbed + delta_perturbed[name])
                params_dict_perturbed[name] = (1.0 - lambda_j_perturbed) * exp0_params[name] + lambda_j_perturbed * exp1_params[name]
                
            fused_buffers_perturbed = get_fused_bn_buffers(exp_buffers_0, exp_buffers_1, w_perturbed, buffer_names)
            
            logits_perturbed = torch.func.functional_call(base_model, (params_dict_perturbed, fused_buffers_perturbed), X_t)
            probs_perturbed = F.softmax(logits_perturbed, dim=1)
            L_entropy_perturbed = -torch.sum(probs_perturbed * torch.log(probs_perturbed + 1e-8), dim=1).mean()
            
            mean_lambda_perturbed = torch.stack([torch.sigmoid(w_perturbed + delta_perturbed[name]) for name in param_names]).mean()
            L_KL_perturbed = mean_lambda_perturbed * torch.log(mean_lambda_perturbed / (w1_prior + 1e-8)) + (1.0 - mean_lambda_perturbed) * torch.log((1.0 - mean_lambda_perturbed) / (w0_prior + 1e-8))
            
            L_coherence_perturbed = gamma_c * sum(F_norm[name] * (delta_perturbed[name] ** 2) for name in param_names)
            
            L_total_perturbed = L_entropy_perturbed + beta * L_KL_perturbed + L_coherence_perturbed
            
            grads_perturbed = torch.autograd.grad(L_total_perturbed, opt_params)
            grad_w = grads_perturbed[0]
            grad_delta = {name: grads_perturbed[i+1] for i, name in enumerate(param_names)}
            
            with torch.no_grad():
                w_global.copy_(w_global - eta_t * grad_w)
                for name in param_names:
                    delta[name].copy_(delta[name] - eta_t * grad_delta[name] / (F_norm[name] + eps_precond))
                    
        # ----------------- EVALUATE -----------------
        final_params = {}
        with torch.no_grad():
            for name in param_names:
                lambda_j = torch.sigmoid(w_global + delta[name])
                final_params[name] = (1.0 - lambda_j) * exp0_params[name] + lambda_j * exp1_params[name]
                
            final_buffers = get_fused_bn_buffers(exp_buffers_0, exp_buffers_1, w_global, buffer_names)
            final_logits = torch.func.functional_call(base_model, (final_params, final_buffers), X_t)
            preds = final_logits.argmax(dim=1)
            correct = preds.eq(Y_t).sum().item()
            acc = correct / X_t.size(0) * 100.0
            batch_accuracies.append(acc)
            
    # Calculate Phase-wise accuracies
    acc_clean_mnist = np.mean(batch_accuracies[0:10])
    acc_noisy_mnist = np.mean(batch_accuracies[10:20])
    acc_clean_fashion = np.mean(batch_accuracies[20:30])
    acc_noisy_fashion = np.mean(batch_accuracies[30:40])
    acc_novel_kmnist = np.mean(batch_accuracies[40:50])
    overall_acc = np.mean(batch_accuracies)
    
    return {
        "Clean MNIST": acc_clean_mnist,
        "Noisy MNIST": acc_noisy_mnist,
        "Clean Fashion": acc_clean_fashion,
        "Noisy Fashion": acc_noisy_fashion,
        "Novel KMNIST": acc_novel_kmnist,
        "Overall": overall_acc,
        "trajectories": batch_accuracies
    }

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sweep on {device}...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="data", train=False, download=False, transform=transform)
    
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, download=False, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, download=False, transform=transform)
    
    stream_batches = get_stream(mnist_test, fmnist_test, kmnist_test)
    
    # Grid of hyperparameters to search
    rho_vals = [0.01, 0.02, 0.05, 0.1, 0.2]
    eps_vals = [0.01, 0.05, 0.1, 0.2]
    methods = ["Method F (Weight-SAM)", "Method G (DPS-SAM)"]
    
    sweep_results = []
    
    for method in methods:
        for rho in rho_vals:
            for eps_precond in eps_vals:
                print(f"Evaluating {method} with rho={rho}, eps_precond={eps_precond}...")
                res = evaluate_hyperparameters(method, stream_batches, mnist_train, fmnist_train, device,
                                               rho=rho, eps_precond=eps_precond)
                record = {
                    "method": method,
                    "rho": rho,
                    "eps_precond": eps_precond,
                    "Clean MNIST": res["Clean MNIST"],
                    "Noisy MNIST": res["Noisy MNIST"],
                    "Clean Fashion": res["Clean Fashion"],
                    "Noisy Fashion": res["Noisy Fashion"],
                    "Novel KMNIST": res["Novel KMNIST"],
                    "Overall": res["Overall"]
                }
                print(f"  Overall: {res['Overall']:.4f}%")
                sweep_results.append(record)
                
    # Save sweep results
    with open("sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=4)
        
    print("Sweep completed! Results saved to sweep_results.json")
