import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. DESIGN MATRIX GENERATORS
# ---------------------------------------------------------

def get_monomial_design_matrix(L, degree):
    l_indices = torch.linspace(0.0, 1.0, L)
    V = []
    for j in range(degree + 1):
        V.append(l_indices ** j)
    return torch.stack(V, dim=1) # Shape: (L, degree + 1)

def get_chebyshev_design_matrix(L, degree):
    l_indices = torch.linspace(0, L - 1, L)
    x = 2.0 * l_indices / (L - 1) - 1.0
    C = []
    C.append(torch.ones_like(x)) # T_0(x) = 1
    if degree >= 1:
        C.append(x)              # T_1(x) = x
    for j in range(2, degree + 1):
        C.append(2.0 * x * C[-1] - C[-2]) # T_j(x) = 2x * T_{j-1}(x) - T_{j-2}(x)
    return torch.stack(C, dim=1) # Shape: (L, degree + 1)

# ---------------------------------------------------------
# 2. SIMULATION SETUP & CONSTANTS
# ---------------------------------------------------------

L = 12
num_tasks = 4
num_seeds = 30
seeds = list(range(42, 42 + num_seeds))
num_steps = 500
lr = 1e-2

# Normalized layer depth
l_indices = torch.linspace(0, L - 1, L)
l_bar = l_indices / (L - 1)

# Target optimal layer importance profiles
lambda_star = torch.zeros(num_tasks, L)
lambda_star[0] = 0.5 - 0.25 * l_bar  # MNIST
lambda_star[1] = 0.2 + 0.35 * torch.sin(torch.pi * l_bar)  # FashionMNIST
lambda_star[2] = 0.1 + 0.45 * (l_bar ** 2)  # CIFAR-10
lambda_star[3] = 0.4 - 0.35 * ((l_bar - 0.5) ** 2)  # SVHN

# Calibrated baseline and sensitivity parameters from CLIP literature
baselines = torch.tensor([0.9271, 0.8164, 0.9017, 0.7324])
deltas = torch.tensor([0.015, 0.040, 0.025, 0.055])

# Model II layer sensitivity and coupling
s = torch.zeros(L)
early = L // 3
mid = L // 3
late = L - early - mid
s[:early] = 0.6
s[early:early+mid] = 1.0
s[early+mid:] = 1.6

Sigma = torch.zeros(L, L)
for i in range(L):
    for j in range(L):
        Sigma[i, j] = torch.sqrt(s[i] * s[j]) * (0.5 ** abs(i - j))
Sigma_inv = torch.inverse(Sigma)

# ---------------------------------------------------------
# 3. NOISE SAMPLING
# ---------------------------------------------------------

def sample_noise_model1(seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    z_k = torch.normal(0.0, 0.12, size=(num_tasks, 1))
    alt_noise = z_k * ((-1.0) ** torch.arange(L).unsqueeze(0))
    return alt_noise

def sample_noise_model2(seed):
    torch.manual_seed(seed)
    # Alternating noise component
    z_k = torch.normal(0.0, 0.12, size=(num_tasks, 1))
    alt_noise = z_k * ((-1.0) ** torch.arange(L).unsqueeze(0))
    
    # White noise component
    white_noise = torch.normal(0.0, 0.08, size=(num_tasks, L))
    
    # Brownian noise component
    brown_noise = torch.zeros(num_tasks, L)
    eps = torch.normal(0.0, 0.08, size=(num_tasks, L))
    brown_noise[:, 0] = eps[:, 0]
    for l in range(1, L):
        brown_noise[:, l] = brown_noise[:, l-1] + eps[:, l]
        
    eta = 0.5 * alt_noise + 0.3 * white_noise + 0.2 * brown_noise
    return eta

# ---------------------------------------------------------
# 4. GENERALIZATION ACCURACY EVALUATION
# ---------------------------------------------------------

def evaluate_accuracy_model1(lambdas):
    # lambdas shape: (num_tasks, L)
    accuracies = []
    for k in range(num_tasks):
        # Mean squared distance
        d_lambda = torch.mean((lambdas[k] - lambda_star[k]) ** 2)
        d_03 = torch.mean((0.3 - lambda_star[k]) ** 2)
        acc_k = baselines[k] + deltas[k] * (1.0 - d_lambda / d_03)
        accuracies.append(acc_k.item())
    return accuracies

def evaluate_accuracy_model2(lambdas):
    accuracies = []
    for k in range(num_tasks):
        d_k = lambdas[k] - lambda_star[k]
        d_0_k = 0.3 - lambda_star[k]
        quad_d = torch.matmul(torch.matmul(d_k.unsqueeze(0), Sigma_inv), d_k.unsqueeze(1)).squeeze()
        quad_d0 = torch.matmul(torch.matmul(d_0_k.unsqueeze(0), Sigma_inv), d_0_k.unsqueeze(1)).squeeze()
        acc_k = baselines[k] + deltas[k] * (1.0 - quad_d / quad_d0)
        accuracies.append(acc_k.item())
    return accuracies

# ---------------------------------------------------------
# 5. CORE OPTIMIZATION EXPERIMENTS
# ---------------------------------------------------------

def run_optimization(model_type, method_name, seed, **kwargs):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get noise
    if model_type == 'Model I':
        eta = sample_noise_model1(seed)
        t = lambda_star + eta
    else:
        eta = sample_noise_model2(seed)
        t = lambda_star + eta
        
    # Initialize parameters
    if method_name == 'Task Arithmetic':
        lambdas = torch.full((num_tasks, L), 0.3)
        if kwargs.get('save_trajectory', False):
            # Evaluate losses
            losses = []
            for step in range(num_steps):
                if model_type == 'Model I':
                    loss = torch.sum(0.5 + (5.0 / L) * torch.sum((lambdas - t) ** 2, dim=1))
                else:
                    loss_val = 0.0
                    for k in range(num_tasks):
                        e_k = lambdas[k] - t[k]
                        quad_form = torch.matmul(torch.matmul(e_k.unsqueeze(0), Sigma_inv), e_k.unsqueeze(1)).squeeze()
                        rastrigin = torch.sum(1.0 - torch.cos(10.0 * torch.pi * e_k))
                        loss_val += 0.5 + 1.5 * quad_form + 0.03 * rastrigin
                    loss = loss_val
                losses.append(loss.item())
            return lambdas.detach(), losses
        return lambdas.detach()
        
    elif method_name == 'Unconstrained':
        lambdas_param = nn.Parameter(torch.full((num_tasks, L), 0.3))
        params = [lambdas_param]
        
    elif method_name == 'TV Regularized':
        lambdas_param = nn.Parameter(torch.full((num_tasks, L), 0.3))
        params = [lambdas_param]
        beta = kwargs.get('beta', 20.0)
        
    elif method_name == 'L2 Regularized':
        lambdas_param = nn.Parameter(torch.full((num_tasks, L), 0.3))
        params = [lambdas_param]
        mu = kwargs.get('mu', 5.0)
        
    elif method_name.startswith('PolyMerge'):
        degree = kwargs.get('degree', 2)
        alpha_param = nn.Parameter(torch.zeros(num_tasks, degree + 1))
        with torch.no_grad():
            alpha_param[:, 0] = 0.3
        params = [alpha_param]
        V = get_monomial_design_matrix(L, degree)
        
    elif method_name.startswith('ChebyMerge-CSD'):
        degree = kwargs.get('degree', 2)
        alpha_params = []
        for j in range(degree + 1):
            p = nn.Parameter(torch.zeros(num_tasks, 1))
            if j == 0:
                with torch.no_grad():
                    p.fill_(0.3)
            alpha_params.append(p)
        C = get_chebyshev_design_matrix(L, degree)
        params = alpha_params
        
    elif method_name.startswith('ChebyMerge'):
        degree = kwargs.get('degree', 2)
        alpha_param = nn.Parameter(torch.zeros(num_tasks, degree + 1))
        with torch.no_grad():
            alpha_param[:, 0] = 0.3
        params = [alpha_param]
        C = get_chebyshev_design_matrix(L, degree)
        
    if method_name.startswith('ChebyMerge-CSD'):
        decay_factor = kwargs.get('decay_factor', 0.2)
        method_lr = kwargs.get('lr', 2e-2)
        param_groups = []
        for j in range(degree + 1):
            param_groups.append({
                'params': [alpha_params[j]],
                'lr': method_lr * (decay_factor ** j)
            })
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
        
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Synthesize lambdas
        if method_name == 'Unconstrained' or method_name == 'TV Regularized' or method_name == 'L2 Regularized':
            lambdas = lambdas_param
        elif method_name.startswith('PolyMerge'):
            lambdas = torch.matmul(alpha_param, V.t())
        elif method_name.startswith('ChebyMerge-CSD'):
            lambdas = torch.zeros(num_tasks, L)
            for j in range(degree + 1):
                lambdas += alpha_params[j] * C[:, j].unsqueeze(0)
        elif method_name.startswith('ChebyMerge'):
            lambdas = torch.matmul(alpha_param, C.t())
            
        # Compute loss
        if model_type == 'Model I':
            loss = torch.sum(0.5 + (5.0 / L) * torch.sum((lambdas - t) ** 2, dim=1))
            if method_name == 'TV Regularized':
                tv_loss = torch.sum((lambdas[:, 1:] - lambdas[:, :-1]) ** 2)
                loss += beta * tv_loss
            elif method_name == 'L2 Regularized':
                l2_loss = torch.sum((lambdas - 0.3) ** 2)
                loss += mu * l2_loss
        else:
            loss_val = 0.0
            for k in range(num_tasks):
                e_k = lambdas[k] - t[k]
                quad_form = torch.matmul(torch.matmul(e_k.unsqueeze(0), Sigma_inv), e_k.unsqueeze(1)).squeeze()
                rastrigin = torch.sum(1.0 - torch.cos(10.0 * torch.pi * e_k))
                loss_val += 0.5 + 1.5 * quad_form + 0.03 * rastrigin
            loss = loss_val
            if method_name == 'TV Regularized':
                tv_loss = torch.sum((lambdas[:, 1:] - lambdas[:, :-1]) ** 2)
                loss += beta * tv_loss
                
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    if kwargs.get('save_trajectory', False):
        return lambdas.detach(), losses
    return lambdas.detach()

# ---------------------------------------------------------
# 6. RUN ALL EXPERIMENTS
# ---------------------------------------------------------

results = {
    'Model I': {},
    'Model II': {}
}

# Methods to run for Model I
methods_model1 = {
    'Task Arithmetic': {},
    'Unconstrained': {},
    'TV Regularized': {'beta': 20.0},
    'L2 Regularized': {'mu': 5.0},
    'PolyMerge (d=0)': {'degree': 0},
    'PolyMerge (d=1)': {'degree': 1},
    'PolyMerge (d=2)': {'degree': 2},
    'PolyMerge (d=3)': {'degree': 3},
    'ChebyMerge (d=0)': {'degree': 0},
    'ChebyMerge (d=1)': {'degree': 1},
    'ChebyMerge (d=2)': {'degree': 2},
    'ChebyMerge (d=3)': {'degree': 3},
    'ChebyMerge-CSD (d=2)': {'degree': 2, 'decay_factor': 0.2, 'lr': 2e-2},
    'ChebyMerge-CSD (d=3)': {'degree': 3, 'decay_factor': 0.1, 'lr': 2e-2}
}

# Methods to run for Model II
methods_model2 = {
    'Task Arithmetic': {},
    'Unconstrained': {},
    'TV Regularized': {'beta': 50.0},
    'PolyMerge (d=0)': {'degree': 0},
    'PolyMerge (d=1)': {'degree': 1},
    'PolyMerge (d=2)': {'degree': 2},
    'PolyMerge (d=3)': {'degree': 3},
    'ChebyMerge (d=0)': {'degree': 0},
    'ChebyMerge (d=1)': {'degree': 1},
    'ChebyMerge (d=2)': {'degree': 2},
    'ChebyMerge (d=3)': {'degree': 3},
    'ChebyMerge-CSD (d=2)': {'degree': 2, 'decay_factor': 0.2, 'lr': 2e-2},
    'ChebyMerge-CSD (d=3)': {'degree': 3, 'decay_factor': 0.1, 'lr': 2e-2}
}

print("Running Model I Experiments...")
for method, args in methods_model1.items():
    print(f"  Method: {method}")
    all_accs = []
    for s_val in seeds:
        final_lambdas = run_optimization('Model I', method, s_val, **args)
        accs = evaluate_accuracy_model1(final_lambdas)
        all_accs.append(accs)
    all_accs = np.array(all_accs) # Shape: (30, 4)
    results['Model I'][method] = {
        'MNIST': {'mean': np.mean(all_accs[:, 0]), 'std': np.std(all_accs[:, 0])},
        'FashionMNIST': {'mean': np.mean(all_accs[:, 1]), 'std': np.std(all_accs[:, 1])},
        'CIFAR10': {'mean': np.mean(all_accs[:, 2]), 'std': np.std(all_accs[:, 2])},
        'SVHN': {'mean': np.mean(all_accs[:, 3]), 'std': np.std(all_accs[:, 3])},
        'Average': {'mean': np.mean(np.mean(all_accs, axis=1)), 'std': np.std(np.mean(all_accs, axis=1))}
    }

print("Running Model II Experiments...")
for method, args in methods_model2.items():
    print(f"  Method: {method}")
    all_accs = []
    for s_val in seeds:
        final_lambdas = run_optimization('Model II', method, s_val, **args)
        accs = evaluate_accuracy_model2(final_lambdas)
        all_accs.append(accs)
    all_accs = np.array(all_accs) # Shape: (30, 4)
    results['Model II'][method] = {
        'MNIST': {'mean': np.mean(all_accs[:, 0]), 'std': np.std(all_accs[:, 0])},
        'FashionMNIST': {'mean': np.mean(all_accs[:, 1]), 'std': np.std(all_accs[:, 1])},
        'CIFAR10': {'mean': np.mean(all_accs[:, 2]), 'std': np.std(all_accs[:, 2])},
        'SVHN': {'mean': np.mean(all_accs[:, 3]), 'std': np.std(all_accs[:, 3])},
        'Average': {'mean': np.mean(np.mean(all_accs, axis=1)), 'std': np.std(np.mean(all_accs, axis=1))}
    }

# ---------------------------------------------------------
# 7. GENERATE CONDITION NUMBERS
# ---------------------------------------------------------

cond_numbers = {
    'Monomial': {},
    'Chebyshev': {}
}

for d in [1, 2, 3]:
    V = get_monomial_design_matrix(L, d).detach().cpu().numpy()
    C = get_chebyshev_design_matrix(L, d).detach().cpu().numpy()
    
    cond_numbers['Monomial'][f'd={d}'] = np.linalg.cond(V.T @ V)
    cond_numbers['Chebyshev'][f'd={d}'] = np.linalg.cond(C.T @ C)

print("\nGram Matrix Condition Numbers (X^T X):")
for d in [1, 2, 3]:
    print(f"  degree {d}: Monomial = {cond_numbers['Monomial'][f'd={d}']:.4f}, Chebyshev = {cond_numbers['Chebyshev'][f'd={d}']:.4f}")

# ---------------------------------------------------------
# 8. GENERATE PLOTS (SEED 42)
# ---------------------------------------------------------

os.makedirs('results', exist_ok=True)

# Plot 1: Optimization Trajectories (Model II, Seed 42)
plt.figure(figsize=(10, 6))
methods_to_plot = {
    'Task Arithmetic': {},
    'Unconstrained': {},
    'TV Regularized': {'beta': 50.0},
    'PolyMerge (d=2)': {'degree': 2},
    'ChebyMerge (d=2)': {'degree': 2},
    'ChebyMerge-CSD (d=2)': {'degree': 2, 'decay_factor': 0.2, 'lr': 2e-2}
}

for method, args in methods_to_plot.items():
    _, traj_losses = run_optimization('Model II', method, 42, save_trajectory=True, **args)
    plt.plot(traj_losses, label=method, linewidth=2)

plt.xlabel('Optimization Steps', fontsize=12)
plt.ylabel('Unsupervised simulated TTA Loss', fontsize=12)
plt.title('Optimization Trajectory (Model II, Seed 42)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/fig1_trajectory.png', dpi=300)
plt.close()

# Plot 2: Layer Coefficient Profiles for SVHN (Model II, Seed 42)
plt.figure(figsize=(10, 6))

# True Target profile for SVHN is lambda_star[3]
plt.plot(l_indices.numpy(), lambda_star[3].numpy(), 'k--', label='Optimal Target Profile (SVHN)', linewidth=2.5)

profiles_to_plot = {
    'Unconstrained': {},
    'TV Regularized': {'beta': 50.0},
    'PolyMerge (d=2)': {'degree': 2},
    'ChebyMerge (d=2)': {'degree': 2},
    'ChebyMerge-CSD (d=2)': {'degree': 2, 'decay_factor': 0.2, 'lr': 2e-2}
}

for method, args in profiles_to_plot.items():
    final_lambdas = run_optimization('Model II', method, 42, **args)
    plt.plot(l_indices.numpy(), final_lambdas[3].numpy(), label=method, marker='o', linewidth=2)

plt.xlabel('Layer Index (l)', fontsize=12)
plt.ylabel('Merging Coefficient (lambda)', fontsize=12)
plt.title('SVHN Merging Coefficient Profiles (Model II, Seed 42)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/fig2_profiles.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 9. SAVE METRICS AND OUTPUT MARKDOWN TABLES
# ---------------------------------------------------------

def convert_to_serializable(val):
    if isinstance(val, dict):
        return {k: convert_to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [convert_to_serializable(v) for v in val]
    elif isinstance(val, (np.float32, np.float64, np.floating)):
        return float(val)
    elif isinstance(val, (np.int32, np.int64, np.integer)):
        return int(val)
    elif isinstance(val, torch.Tensor):
        return val.item()
    else:
        return val

output_data = convert_to_serializable({
    'results': results,
    'condition_numbers': cond_numbers
})

with open('results/metrics.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\nAll experiments run successfully. Saved metrics to results/metrics.json.")
