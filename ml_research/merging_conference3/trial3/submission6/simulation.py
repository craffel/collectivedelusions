import torch
import numpy as np
import pandas as pd
import math

# Set device
device = torch.device("cpu")

# Simulation Parameters
L = 12
K = 4  # MNIST, FashionMNIST, CIFAR-10, SVHN
num_seeds = 30

# Calibrated Baseline and Sensitivity parameters
datasets = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
Base = torch.tensor([92.71, 81.64, 90.17, 73.24], device=device)
Delta = torch.tensor([1.5, 4.0, 2.5, 5.5], device=device)

# Normalized layer depth
l_bar = torch.linspace(0.0, 1.0, L, device=device)

# Define optimal targets \lambda^*_k
def get_optimal_targets():
    targets = torch.zeros(K, L, device=device)
    # MNIST
    targets[0] = 0.5 - 0.25 * l_bar
    # FashionMNIST
    targets[1] = 0.2 + 0.35 * torch.sin(math.pi * l_bar)
    # CIFAR-10
    targets[2] = 0.1 + 0.45 * (l_bar ** 2)
    # SVHN
    targets[3] = 0.4 - 0.35 * ((l_bar - 0.5) ** 2)
    return targets

# Covariance matrix for Model II
s = torch.zeros(L, device=device)
s[0:4] = 0.6
s[4:8] = 1.0
s[8:12] = 1.6

Sigma = torch.zeros(L, L, device=device)
for i in range(L):
    for j in range(L):
        Sigma[i, j] = math.sqrt(s[i] * s[j]) * (0.5 ** abs(i - j))

Sigma_inv = torch.inverse(Sigma)

# Distance metrics
def dist_MSD(u, v):
    return torch.mean((u - v) ** 2, dim=-1)

def dist_Mahalanobis(u, v):
    # u and v are of shape (K, L) or (L,)
    diff = u - v
    if diff.dim() == 1:
        return torch.dot(diff, torch.matmul(Sigma_inv, diff))
    else:
        # batch of K tasks
        res = torch.zeros(diff.shape[0], device=device)
        for k in range(diff.shape[0]):
            res[k] = torch.dot(diff[k], torch.matmul(Sigma_inv, diff[k]))
        return res

# Accuracy functions
def get_accuracy_model1(lambdas, targets):
    d_current = dist_MSD(lambdas, targets)
    d_initial = dist_MSD(torch.full_like(targets, 0.3), targets)
    accs = Base + Delta * (1.0 - d_current / d_initial)
    return accs

def get_accuracy_model2(lambdas, targets):
    d_current = dist_Mahalanobis(lambdas, targets)
    d_initial = dist_Mahalanobis(torch.full_like(targets, 0.3), targets)
    accs = Base + Delta * (1.0 - d_current / d_initial)
    return accs

# Loss functions
def get_loss_model1(lambdas, noisy_targets):
    # shape of lambdas: (K, L)
    # shape of noisy_targets: (K, L)
    term1 = 0.5
    term2 = (5.0 / L) * torch.sum((lambdas - noisy_targets) ** 2, dim=-1)
    return torch.sum(term1 + term2)

def get_loss_model2(lambdas, noisy_targets, calibrated_weights=None):
    # shape of lambdas: (K, L)
    # shape of noisy_targets: (K, L)
    # calibrated_weights: SNEW weights for each task
    e = lambdas - noisy_targets
    loss_val = 0.0
    for k in range(K):
        term1 = 0.5
        term2 = 1.5 * torch.dot(e[k], torch.matmul(Sigma_inv, e[k]))
        term3 = 0.03 * torch.sum(1.0 - torch.cos(10 * math.pi * e[k]))
        task_loss = term1 + term2 + term3
        if calibrated_weights is not None:
            task_loss = task_loss * calibrated_weights[k]
        loss_val += task_loss
    return loss_val

# Noise generation
def generate_noise(seed, model_type):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Alternating noise
    z = torch.randn(K, device=device) * 0.12
    alt_noise = torch.zeros(K, L, device=device)
    for l in range(L):
        alt_noise[:, l] = z * ((-1) ** l)
        
    if model_type == 1:
        return alt_noise
    else:
        # White noise
        white_noise = torch.randn(K, L, device=device) * 0.08
        
        # Brownian noise
        brown_noise = torch.zeros(K, L, device=device)
        for k in range(K):
            current = 0.0
            for l in range(L):
                step = torch.randn(1, device=device).item() * 0.04
                current += step
                brown_noise[k, l] = current
                
        # Combine
        noise = 0.5 * alt_noise + 0.3 * white_noise + 0.2 * brown_noise
        return noise

# Baselines implementation
def run_task_arithmetic(targets, model_type):
    lambdas = torch.full((K, L), 0.3, device=device)
    if model_type == 1:
        accs = get_accuracy_model1(lambdas, targets)
    else:
        accs = get_accuracy_model2(lambdas, targets)
    return accs

def run_adamerging(targets, noisy_targets, model_type):
    # Initialize coefficients to 0.3
    lambdas = torch.full((K, L), 0.3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([lambdas], lr=0.02)
    
    for step in range(500):
        optimizer.zero_grad()
        if model_type == 1:
            loss = get_loss_model1(lambdas, noisy_targets)
        else:
            loss = get_loss_model2(lambdas, noisy_targets)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        if model_type == 1:
            accs = get_accuracy_model1(lambdas, targets)
        else:
            accs = get_accuracy_model2(lambdas, targets)
    return accs

def run_regcalmerge(targets, noisy_targets, model_type):
    # Compute SNEW weights at step 0 (initialization = 0.3)
    lambdas_init = torch.full((K, L), 0.3, device=device)
    e_init = lambdas_init - noisy_targets
    
    # Baseline entropies (measured as distance component of TTA loss at step 0)
    baseline_losses = torch.zeros(K, device=device)
    if model_type == 1:
        for k in range(K):
            baseline_losses[k] = 0.5 + (5.0 / L) * torch.sum(e_init[k] ** 2)
    else:
        for k in range(K):
            term1 = 0.5
            term2 = 1.5 * torch.dot(e_init[k], torch.matmul(Sigma_inv, e_init[k]))
            term3 = 0.03 * torch.sum(1.0 - torch.cos(10 * math.pi * e_init[k]))
            baseline_losses[k] = term1 + term2 + term3
            
    # SNEW weights: inverse of baseline loss
    snew_weights = 1.0 / baseline_losses
    snew_weights = snew_weights / torch.mean(snew_weights)  # normalized
    
    # Initialize coefficients to 0.3
    lambdas = torch.full((K, L), 0.3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([lambdas], lr=0.02)
    
    beta = 1.0  # Proximity penalty strength
    gamma = 1.0  # Spatial deviation penalty strength
    
    for step in range(500):
        optimizer.zero_grad()
        
        # Base TTA loss with SNEW calibration
        if model_type == 1:
            loss = 0.0
            for k in range(K):
                task_loss = 0.5 + (5.0 / L) * torch.sum((lambdas[k] - noisy_targets[k]) ** 2)
                loss += task_loss * snew_weights[k]
        else:
            loss = get_loss_model2(lambdas, noisy_targets, snew_weights)
            
        # ESR Regularization
        proximity_penalty = beta * torch.sum((lambdas - 0.3) ** 2)
        spatial_penalty = gamma * torch.sum((lambdas[:, 1:] - lambdas[:, :-1]) ** 2)
        
        total_loss = loss + proximity_penalty + spatial_penalty
        total_loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        if model_type == 1:
            accs = get_accuracy_model1(lambdas, targets)
        else:
            accs = get_accuracy_model2(lambdas, targets)
    return accs

def run_polymerge(targets, noisy_targets, model_type):
    # Parameterize coefficients as a quadratic polynomial (degree d=2)
    degree = 2
    alpha = torch.zeros(K, degree + 1, device=device, requires_grad=True)
    with torch.no_grad():
        alpha[:, 0] = 0.3  # Initialize to uniform 0.3
        
    # Vandermonde matrix
    normalized_depth = torch.linspace(0.0, 1.0, L, device=device)
    V_matrix = torch.stack([normalized_depth ** j for j in range(degree + 1)], dim=1) # (L, degree+1)
    
    optimizer = torch.optim.Adam([alpha], lr=0.02)
    
    for step in range(500):
        optimizer.zero_grad()
        # Synthesize lambdas
        lambdas = torch.matmul(alpha, V_matrix.t())  # (K, L)
        if model_type == 1:
            loss = get_loss_model1(lambdas, noisy_targets)
        else:
            loss = get_loss_model2(lambdas, noisy_targets)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        lambdas = torch.matmul(alpha, V_matrix.t())
        if model_type == 1:
            accs = get_accuracy_model1(lambdas, targets)
        else:
            accs = get_accuracy_model2(lambdas, targets)
    return accs

def run_acm(targets, noisy_targets, model_type):
    # ACM: Direct, training-free analytical curvature-aware solver
    # We parameterize coefficients as a quadratic polynomial (degree d=2)
    # and solve for the optimal polynomial parameters alpha analytically!
    degree = 2
    
    # Vandermonde matrix
    normalized_depth = torch.linspace(0.0, 1.0, L, device=device)
    V_matrix = torch.stack([normalized_depth ** j for j in range(degree + 1)], dim=1) # (L, degree+1)
    
    alpha_init = torch.zeros(degree + 1, device=device)
    alpha_init[0] = 0.3  # Initialize to uniform 0.3
    
    lambdas = torch.zeros(K, L, device=device)
    
    if model_type == 1:
        # H_k = (10.0 / L) * I
        H_val = 10.0 / L
        H = H_val * torch.eye(L, device=device)
        gamma_acm = 0.05  # Ridge regularization strength
    else:
        # H_k = 3.0 * Sigma_inv
        H = 3.0 * Sigma_inv
        gamma_acm = 0.1  # Ridge regularization strength
        
    for k in range(K):
        # A_matrix = V^T H V
        A_matrix = torch.matmul(V_matrix.t(), torch.matmul(H, V_matrix))
        # Add Ridge regularization
        A_reg = A_matrix + gamma_acm * torch.eye(degree + 1, device=device)
        # b_vector = V^T H t_k + gamma * alpha_init
        b_vector = torch.matmul(V_matrix.t(), torch.matmul(H, noisy_targets[k])) + gamma_acm * alpha_init
        
        # Solve
        alpha_sol = torch.matmul(torch.inverse(A_reg), b_vector)
        # Synthesize lambdas
        lambdas[k] = torch.matmul(V_matrix, alpha_sol)
            
    if model_type == 1:
        accs = get_accuracy_model1(lambdas, targets)
    else:
        accs = get_accuracy_model2(lambdas, targets)
    return accs

# Run evaluation over all seeds
def run_evaluation():
    targets = get_optimal_targets()
    
    results = {
        1: {m: [] for m in ["Task Arithmetic", "AdaMerging", "RegCalMerge", "PolyMerge", "ACM"]},
        2: {m: [] for m in ["Task Arithmetic", "AdaMerging", "RegCalMerge", "PolyMerge", "ACM"]}
    }
    
    print("Starting simulation sweeps...")
    for model_type in [1, 2]:
        print(f"\n--- Running Model {model_type} ---")
        for seed in range(1, num_seeds + 1):
            noise = generate_noise(seed, model_type)
            noisy_targets = targets + noise
            
            # 1. Task Arithmetic
            ta_acc = run_task_arithmetic(targets, model_type)
            results[model_type]["Task Arithmetic"].append(ta_acc.cpu().numpy())
            
            # 2. AdaMerging
            ada_acc = run_adamerging(targets, noisy_targets, model_type)
            results[model_type]["AdaMerging"].append(ada_acc.cpu().numpy())
            
            # 3. RegCalMerge
            rcm_acc = run_regcalmerge(targets, noisy_targets, model_type)
            results[model_type]["RegCalMerge"].append(rcm_acc.cpu().numpy())
            
            # 4. PolyMerge
            pm_acc = run_polymerge(targets, noisy_targets, model_type)
            results[model_type]["PolyMerge"].append(pm_acc.cpu().numpy())
            
            # 5. ACM (Proposed)
            acm_acc = run_acm(targets, noisy_targets, model_type)
            results[model_type]["ACM"].append(acm_acc.cpu().numpy())
            
            if seed % 5 == 0:
                print(f"Seed {seed}/30 complete...")
                
    # Aggregate and print results
    for model_type in [1, 2]:
        print(f"\n=========================================")
        print(f"RESULTS FOR MODEL {model_type}")
        print(f"=========================================")
        
        summary_rows = []
        for method_name in ["Task Arithmetic", "AdaMerging", "RegCalMerge", "PolyMerge", "ACM"]:
            data = np.array(results[model_type][method_name])  # (num_seeds, K)
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            
            joint_mean = np.mean(data)
            joint_std = np.std(np.mean(data, axis=1))  # standard deviation of the joint means across seeds
            
            summary_rows.append({
                "Method": method_name,
                "MNIST": f"{means[0]:.2f}% ± {stds[0]:.2f}%",
                "FashionMNIST": f"{means[1]:.2f}% ± {stds[1]:.2f}%",
                "CIFAR-10": f"{means[2]:.2f}% ± {stds[2]:.2f}%",
                "SVHN": f"{means[3]:.2f}% ± {stds[3]:.2f}%",
                "Average": f"{joint_mean:.2f}% ± {joint_std:.2f}%",
                "RawAverage": joint_mean
            })
            
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))
        
        # Save to csv for persistence
        df.to_csv(f"simulation_model{model_type}_results.csv", index=False)

if __name__ == "__main__":
    run_evaluation()
