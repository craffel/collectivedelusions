import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Set reproducibility seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ResNet18Custom(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.base_model.fc(feat)
        return feat, logits

def get_resnet18_1channel():
    model = resnet18()
    conv1_new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = conv1_new
    model.fc = nn.Linear(512, 10)
    return model

# Fast batch Fisher approximation
def compute_batch_fisher(expert_model, batch_X, device):
    fisher = {name: torch.zeros_like(p) for name, p in expert_model.named_parameters() if p.requires_grad}
    expert_model.eval()
    
    _, logits = expert_model(batch_X)
    pseudo_labels = torch.argmax(logits, dim=-1)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, pseudo_labels)
    expert_model.zero_grad()
    loss.backward()
    
    for name, param in expert_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher[name] = param.grad.data ** 2
            
    return fisher

def get_joint_fisher(expert_models, batch_X, device):
    joint_fisher = {}
    K = len(expert_models)
    for k in range(K):
        fisher_k = compute_batch_fisher(expert_models[k], batch_X, device)
        for name, val in fisher_k.items():
            clean_name = name.replace('base_model.', '')
            tensor_avg = val.mean().item()
            if clean_name not in joint_fisher:
                joint_fisher[clean_name] = 0.0
            joint_fisher[clean_name] += tensor_avg / K
    return joint_fisher

def compute_preconditioned_lrs(joint_fisher, base_lr=1e-3, eps=1e-6, alpha=1.0):
    sensitivities = list(joint_fisher.values())
    mean_sens = sum(sensitivities) / len(sensitivities)
    
    preconditioned_lrs = {}
    for name, sens in joint_fisher.items():
        norm_sens = sens / (mean_sens + 1e-12)
        lr_mult = (norm_sens + eps) ** (-alpha)
        lr_mult = torch.clamp(torch.tensor(lr_mult), 0.01, 10.0).item()
        preconditioned_lrs[name] = base_lr * lr_mult
    return preconditioned_lrs

def project_simplex(v):
    v_sorted, _ = torch.sort(v, descending=True)
    j = torch.arange(1, len(v) + 1, device=v.device)
    cumsum = torch.cumsum(v_sorted, dim=0)
    rho_cond = v_sorted - (cumsum - 1.0) / j > 0
    rho = torch.max(torch.where(rho_cond)[0]) + 1
    theta = (cumsum[rho - 1] - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running ablation on device:", device)
    
    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Load expert models
    expert_paths = {
        'mnist': 'mnist_expert.pt',
        'kmnist': 'kmnist_expert.pt',
        'fashionmnist': 'fashionmnist_expert.pt'
    }
    experts = {}
    for name, path in expert_paths.items():
        model = get_resnet18_1channel()
        model.load_state_dict(torch.load(path, map_location=device))
        model = ResNet18Custom(model).to(device)
        model.eval()
        experts[name] = model
        
    expert_list = [experts['mnist'], experts['kmnist'], experts['fashionmnist']]
    expert_names = ['mnist', 'kmnist', 'fashionmnist']
    K = len(expert_list)
    
    # Static anchor model
    static_model = get_resnet18_1channel()
    static_state_dict = {}
    for name in static_model.state_dict().keys():
        w_sum = sum(experts[e_name].base_model.state_dict()[name].float() for e_name in expert_names)
        static_state_dict[name] = w_sum / K
    static_model.load_state_dict(static_state_dict)
    static_model = ResNet18Custom(static_model).to(device)
    static_model.eval()
    
    # Precompute dataset mean feature vector (mu_k) on calibration sets (500 samples)
    calib_size = 500
    calib_subsets = {
        'mnist': Subset(mnist_train, list(range(calib_size))),
        'kmnist': Subset(kmnist_train, list(range(calib_size))),
        'fashionmnist': Subset(fmnist_train, list(range(calib_size)))
    }
    
    mu_static_domain = {}
    for e_name in expert_names:
        loader = DataLoader(calib_subsets[e_name], batch_size=100, shuffle=False)
        feats_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                feat, _ = static_model(x)
                feats_list.append(feat)
        mu_static_domain[e_name] = torch.cat(feats_list, dim=0).mean(dim=0)
    
    # Precompute class prototypes for known tasks
    prototypes = {0: {}, 1: {}}
    known_datasets = ['mnist', 'kmnist']
    for k, e_name in enumerate(known_datasets):
        loader = DataLoader(calib_subsets[e_name], batch_size=100, shuffle=False)
        feats_all = []
        labels_all = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feat, _ = static_model(x)
                feats_all.append(feat)
                labels_all.append(y.to(device))
        feats_all = torch.cat(feats_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        
        # Center features with domain mean
        z_all = feats_all - mu_static_domain[e_name]
        
        for c in range(10):
            mask = (labels_all == c)
            if mask.sum() > 0:
                pi_c = z_all[mask].mean(dim=0)
                pi_c = pi_c / (pi_c.norm(p=2) + 1e-12)
                prototypes[k][c] = pi_c
            else:
                prototypes[k][c] = torch.zeros_like(mu_static_domain[e_name])
                
    # Source Fisher (S-Fisher)
    sf_subset_mnist = Subset(mnist_train, list(range(250)))
    sf_subset_kmnist = Subset(kmnist_train, list(range(250)))
    sf_loader = DataLoader(Subset(mnist_train, list(range(250))), batch_size=250, shuffle=False)
    for x_m, _ in sf_loader:
        sf_x_mnist = x_m
    sf_loader = DataLoader(Subset(kmnist_train, list(range(250))), batch_size=250, shuffle=False)
    for x_k, _ in sf_loader:
        sf_x_kmnist = x_k
    sf_X = torch.cat([sf_x_mnist, sf_x_kmnist], dim=0).to(device)
    s_fisher = get_joint_fisher(expert_list, sf_X, device)
    
    # Test streams
    test_size_per_task = 30 * 64
    mnist_test_subset = Subset(mnist_test, list(range(test_size_per_task)))
    kmnist_test_subset = Subset(kmnist_test, list(range(test_size_per_task)))
    fmnist_test_subset = Subset(fmnist_test, list(range(test_size_per_task)))
    
    mnist_loader = DataLoader(mnist_test_subset, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test_subset, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test_subset, batch_size=64, shuffle=False)
    
    mnist_batches = [b for b in mnist_loader]
    kmnist_batches = [b for b in kmnist_loader]
    fmnist_batches = [b for b in fmnist_loader]
    
    seq_stream_clean = mnist_batches[:30] + kmnist_batches[:30] + fmnist_batches[:30]
    seq_domains = [0]*30 + [1]*30 + [2]*30
    
    def apply_noise(batch_X):
        return torch.clamp(batch_X + torch.randn_like(batch_X) * 0.2, -1.0, 1.0)
        
    def apply_contrast(batch_X):
        return torch.clamp(batch_X * 0.3, -1.0, 1.0)
        
    streams = {
        'Sequential_Clean': (seq_stream_clean, seq_domains),
        'Sequential_Noise': ([(apply_noise(x), y) for x, y in seq_stream_clean], seq_domains),
        'Sequential_Contrast': ([(apply_contrast(x), y) for x, y in seq_stream_clean], seq_domains)
    }
    
    base_model = get_resnet18_1channel().to(device)
    parameter_names = [name for name, p in base_model.named_parameters() if p.requires_grad]
    buffer_names = [name for name, b in base_model.named_buffers()]
    
    expert_params = []
    expert_buffers = []
    for k in range(K):
        expert_params.append({name: p.clone().detach() for name, p in expert_list[k].base_model.named_parameters()})
        expert_buffers.append({name: b.clone().detach() for name, b in expert_list[k].base_model.named_buffers()})
        
    base_buffers_dict = {name: b for name, b in base_model.named_buffers()}
    
    # 2x2 Ablation methods:
    # 1. DR-Fisher-Static (original DR-Fisher)
    # 2. DR-Fisher-DFC (using DFC but frozen Fisher)
    # 3. D-TT-Fisher-Static (using dynamic Fisher but static centering)
    # 4. D-TT-Fisher-DFC (proposed method, using both)
    
    ablation_methods = {
        'DR-Fisher-Static': {'use_dfc': False, 'use_dynamic_fisher': False, 'tau': 0.65},
        'DR-Fisher-DFC': {'use_dfc': True, 'use_dynamic_fisher': False, 'tau': 0.51},
        'D-TT-Fisher-Static': {'use_dfc': False, 'use_dynamic_fisher': True, 'tau': 0.65},
        'D-TT-Fisher-DFC': {'use_dfc': True, 'use_dynamic_fisher': True, 'tau': 0.51}
    }
    
    results = {}
    for stream_name, (stream, domains) in streams.items():
        print(f"\nEvaluating stream: {stream_name}...")
        results[stream_name] = {}
        
        for name, config in ablation_methods.items():
            use_dfc = config['use_dfc']
            use_dynamic_fisher = config['use_dynamic_fisher']
            tau_N = config['tau']
            
            # Reset coefficients
            coefficients = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in parameter_names}
            
            correct_total = 0
            samples_total = 0
            novel_detected = 0
            novel_actual = 0
            known_detected_novel = 0
            known_actual = 0
            
            running_fisher = {name: torch.tensor(s_fisher[name], device=device) for name in parameter_names}
            tt_fisher_frozen = None
            
            for t, (batch_X, batch_y) in enumerate(stream):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                true_dom = domains[t]
                is_novel_actual = (true_dom == 2)
                
                # --- Cohesion & Novelty Prediction ---
                with torch.no_grad():
                    anchor_feats, _ = static_model(batch_X)
                
                cohesions = []
                for k in range(2):
                    e_name = known_datasets[k]
                    if use_dfc:
                        # center with active batch mean
                        z_anchor_k = anchor_feats - anchor_feats.mean(dim=0)
                    else:
                        # center with offline clean domain mean
                        z_anchor_k = anchor_feats - mu_static_domain[e_name]
                        
                    max_sims = []
                    for i in range(batch_X.size(0)):
                        z_i = z_anchor_k[i]
                        z_i_norm = z_i / (z_i.norm(p=2) + 1e-12)
                        max_sim = max(torch.dot(z_i_norm, prototypes[k][c]) for c in range(10))
                        max_sims.append(max_sim)
                    cohesion_k = sum(max_sims) / len(max_sims)
                    cohesions.append(cohesion_k.item())
                    
                max_cohesion = max(cohesions)
                is_novel_pred = (max_cohesion < tau_N)
                
                if is_novel_actual:
                    novel_actual += 1
                    if is_novel_pred:
                        novel_detected += 1
                else:
                    known_actual += 1
                    if is_novel_pred:
                        known_detected_novel += 1
                        
                # --- Entropy-based routing ---
                entropies = []
                with torch.no_grad():
                    for k in range(K):
                        _, logits_k = expert_list[k](batch_X)
                        probs_k = torch.softmax(logits_k, dim=-1)
                        ent_k = -torch.mean(torch.sum(probs_k * torch.log(probs_k + 1e-12), dim=-1)).item()
                        entropies.append(ent_k)
                        
                k_star = np.argmin(entropies)
                
                # Reset to prior
                Lambda_prior = torch.tensor([0.005, 0.005, 0.005], device=device)
                Lambda_prior[k_star] = 0.99
                
                with torch.no_grad():
                    for param_n in parameter_names:
                        coefficients[param_n].copy_(Lambda_prior)
                        
                for param_n in parameter_names:
                    coefficients[param_n].requires_grad_(True)
                    if coefficients[param_n].grad is not None:
                        coefficients[param_n].grad.zero_()
                        
                # Merge weights and compute gradient
                params_dict = {}
                for param_n in parameter_names:
                    coeff = coefficients[param_n]
                    params_dict[param_n] = coeff[0]*expert_params[0][param_n] + coeff[1]*expert_params[1][param_n] + coeff[2]*expert_params[2][param_n]
                    
                logits = torch.func.functional_call(base_model, params_dict, (batch_X,))
                probs = torch.softmax(logits, dim=-1)
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
                loss.backward()
                
                # Preconditioned LR and update
                if not use_dynamic_fisher:
                    if t < 15:
                        batch_fisher = get_joint_fisher(expert_list, batch_X, device)
                        if tt_fisher_frozen is None:
                            tt_fisher_frozen = {param_n: batch_fisher[param_n] for param_n in parameter_names}
                        else:
                            for param_n in parameter_names:
                                tt_fisher_frozen[param_n] = (tt_fisher_frozen[param_n] * t + batch_fisher[param_n]) / (t + 1)
                    lrs = compute_preconditioned_lrs(tt_fisher_frozen if tt_fisher_frozen is not None else s_fisher, base_lr=1e-3, alpha=1.0)
                else:
                    batch_fisher = get_joint_fisher(expert_list, batch_X, device)
                    for param_n in parameter_names:
                        running_fisher[param_n] = (1 - 0.1) * running_fisher[param_n] + 0.1 * batch_fisher[param_n]
                    lrs = compute_preconditioned_lrs({param_n: val.item() for param_n, val in running_fisher.items()}, base_lr=1e-3, alpha=1.0)
                    
                with torch.no_grad():
                    for param_n in parameter_names:
                        grad = coefficients[param_n].grad
                        if grad is not None:
                            updated = coefficients[param_n] - lrs[param_n] * grad
                            coefficients[param_n].copy_(project_simplex(updated))
                            
                # Sync BN buffers
                with torch.no_grad():
                    coeff_sum = torch.zeros(K, device=device)
                    for param_n in parameter_names:
                        coeff_sum += coefficients[param_n]
                    coeff_avg = coeff_sum / len(parameter_names)
                    
                    for buf_name in buffer_names:
                        b_val = 0.0
                        for k in range(K):
                            b_val += coeff_avg[k].item() * expert_buffers[k][buf_name]
                        base_buffers_dict[buf_name].copy_(b_val)
                        
                # Final forward pass and evaluation
                params_dict_eval = {}
                for param_n in parameter_names:
                    coeff = coefficients[param_n]
                    params_dict_eval[param_n] = coeff[0]*expert_params[0][param_n] + coeff[1]*expert_params[1][param_n] + coeff[2]*expert_params[2][param_n]
                    
                with torch.no_grad():
                    logits = torch.func.functional_call(base_model, params_dict_eval, (batch_X,))
                    _, preds = logits.max(1)
                    correct_total += preds.eq(batch_y).sum().item()
                    samples_total += batch_X.size(0)
                    
            acc = (correct_total / samples_total) * 100.0
            ndr = (novel_detected / novel_actual) * 100.0 if novel_actual > 0 else 100.0
            fpr = (known_detected_novel / known_actual) * 100.0 if known_actual > 0 else 0.0
            
            results[stream_name][name] = {'Acc': acc, 'NDR': ndr, 'FPR': fpr}
            print(f"  [{name}] Acc: {acc:.2f}%, NDR: {ndr:.2f}%, FPR: {fpr:.2f}%")

    print("\n--- Summary of 2x2 Ablation ---")
    for stream_name, res in results.items():
        print(f"\nStream: {stream_name}")
        print("-" * 60)
        print(f"{'Method':<25} | {'Acc (%)':<10} | {'NDR (%)':<10} | {'FPR (%)':<10}")
        print("-" * 60)
        for name in ablation_methods.keys():
            metrics = res[name]
            print(f"{name:<25} | {metrics['Acc']:<10.2f} | {metrics['NDR']:<10.2f} | {metrics['FPR']:<10.2f}")

if __name__ == '__main__':
    main()
