import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# System configurations
L = 14          # number of layers
D = 192         # representation dimension
K = 4           # number of expert tasks (MNIST, FashionMNIST, CIFAR-10, SVHN)
d = D // K      # task block size (48)
C = 10          # number of classes per task
C_tasks = [10, 10, 10, 4] # task-specific class sizes (asymmetric)

# Noise levels calibrated to match expert ceilings:
# MNIST (0.05 -> ~95%), F-MNIST (0.15 -> ~85%), CIFAR-10 (0.45 -> ~50%), SVHN (0.85 -> ~32%)
task_noises = [0.05, 0.15, 0.45, 0.85]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_sandbox_data(seed, rho=0.33):
    """
    Generates the Isolating Coordinate Sandbox representation spaces.
    Creates orthogonal prototypes mixed with a shared component to model subspace overlap rho.
    Generates calibration (64 samples) and test (1000 samples) splits.
    """
    set_seed(seed)
    
    # 1. Generate base orthogonal prototypes and shared prototypes
    # For each task, we have 10 class prototype vectors of size d
    W_orthogonal = torch.randn(K, C, d) # [K, C, d]
    Phi_shared = torch.randn(C, d)      # [C, d] shared across tasks
    
    C_prototypes = torch.zeros(K, C, d)
    for k in range(K):
        for c in range(C):
            # Mix orthogonal and shared prototypes
            proto = torch.sqrt(torch.tensor(1.0 - rho)) * W_orthogonal[k, c] + torch.sqrt(torch.tensor(rho)) * Phi_shared[c]
            C_prototypes[k, c] = proto / torch.norm(proto) # normalize to unit norm
            
    # 2. Generate Calibration Split (16 samples per task = 64 total)
    cal_features = []
    cal_labels = []
    cal_tasks = []
    
    for k in range(K):
        for _ in range(16):
            # Pick a class label from task-specific class size
            c = torch.randint(0, C_tasks[k], (1,)).item()
            
            # Base representation blocks
            z_base = torch.randn(K, d) * 0.1
            
            # Target task expert block with anisotropic noise (even clean, odd noisy)
            noise_scale = task_noises[k] * torch.tensor([0.1 if j % 2 == 0 else 1.9 for j in range(d)])
            z_expert = C_prototypes[k, c] + noise_scale * torch.randn(d)
            
            # Assemble representation vector
            z = torch.zeros(K, d)
            for j in range(K):
                if j == k:
                    z[j] = z_expert
                else:
                    # Interference vector from other tasks
                    z[j] = z_base[j] + torch.randn(d) * 0.5
                    
            cal_features.append(z.view(-1))
            cal_labels.append(c)
            cal_tasks.append(k)
            
    cal_features = torch.stack(cal_features).to(device)
    cal_labels = torch.tensor(cal_labels, dtype=torch.long).to(device)
    cal_tasks = torch.tensor(cal_tasks, dtype=torch.long).to(device)
    
    # 3. Generate Test Split (250 samples per task = 1000 total)
    test_features = []
    test_labels = []
    test_tasks = []
    
    for k in range(K):
        for _ in range(250):
            c = torch.randint(0, C_tasks[k], (1,)).item()
            z_base = torch.randn(K, d) * 0.1
            
            # Anisotropic noise scaling (even clean, odd noisy)
            noise_scale = task_noises[k] * torch.tensor([0.1 if j % 2 == 0 else 1.9 for j in range(d)])
            z_expert = C_prototypes[k, c] + noise_scale * torch.randn(d)
            
            z = torch.zeros(K, d)
            for j in range(K):
                if j == k:
                    z[j] = z_expert
                else:
                    z[j] = z_base[j] + torch.randn(d) * 0.5
                    
            test_features.append(z.view(-1))
            test_labels.append(c)
            test_tasks.append(k)
            
    test_features = torch.stack(test_features).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
    test_tasks = torch.tensor(test_tasks, dtype=torch.long).to(device)
    
    # Send prototypes to device
    C_prototypes = C_prototypes.to(device)
    
    return C_prototypes, cal_features, cal_labels, cal_tasks, test_features, test_labels, test_tasks

def estimate_diagonal_fisher(C_prototypes, cal_features, cal_labels, cal_tasks):
    """
    Computes representation-space diagonal Fisher Information Matrix (dFIM).
    This measures the inverse coordinate variance (1/sigma^2) of the activations,
    representing the information-geometric sensitivity of the coordinates.
    Outputs normalized Fisher vectors of shape [K, C, d].
    """
    F = torch.zeros(K, C, d).to(device)

    for k in range(K):
        # Extract calibration samples of task k
        mask = (cal_tasks == k)
        feat_k = cal_features[mask] # [16, D]

        # Expert block features
        z_k = feat_k.view(-1, K, d)[:, k, :] # [16, d]

        # Compute pooled within-class coordinate variance to isolate noise and prevent centroid conflation
        labels_k = cal_labels[mask]
        sq_deviations = torch.zeros(d).to(device)
        df = 0  # degrees of freedom
        for c in range(C_tasks[k]):
            class_mask = (labels_k == c)
            class_samples = z_k[class_mask]
            Nc = len(class_samples)
            if Nc > 1:
                class_mean = class_samples.mean(dim=0)
                sq_deviations += torch.sum((class_samples - class_mean) ** 2, dim=0)
                df += (Nc - 1)

        if df > 0:
            var_k = sq_deviations / df + 1e-5
        else:
            var_k = torch.var(z_k, dim=0, unbiased=True) + 1e-5

        # Fisher Information of Gaussian mean parameter is inverse variance
        F_k_raw = 1.0 / var_k # [d]

        # Smooth and normalize using the regularized power-scaling formula
        beta = 0.5
        gamma = 0.7
        smoothed = (F_k_raw + beta) ** gamma
        F_k_norm = smoothed / torch.sum(smoothed) # [d]

        # Broadcast across classes of task k
        for c in range(C_tasks[k]):
            F[k, c] = F_k_norm

    return F

def fisher_weighted_cosine_similarity(z_k, W_k, F_k):
    """
    Fisher-Weighted Cosine Similarity (Riemannian local metric).
    z_k: [B, d]
    W_k: [C, d]
    F_k: [C, d]
    Returns similarity matrix of shape [B, C].
    """
    z_k_expanded = z_k.unsqueeze(1) # [B, 1, d]
    W_k_expanded = W_k.unsqueeze(0) # [1, C, d]
    F_k_expanded = F_k.unsqueeze(0) # [1, C, d]
    
    num = torch.sum(F_k_expanded * W_k_expanded * z_k_expanded, dim=-1) # [B, C]
    den1 = torch.sqrt(torch.sum(F_k_expanded * (W_k_expanded ** 2), dim=-1)) # [1, C]
    den2 = torch.sqrt(torch.sum(F_k_expanded * (z_k_expanded ** 2), dim=-1)) # [B, C]
    
    sim = num / (den1 * den2 + 1e-8) # [B, C]
    return sim

def standard_cosine_similarity(z_k, W_k):
    """
    Standard flat Cosine Similarity.
    z_k: [B, d]
    W_k: [C, d]
    Returns similarity matrix of shape [B, C].
    """
    z_k_norm = z_k / (torch.norm(z_k, dim=-1, keepdim=True) + 1e-8)
    W_k_norm = W_k / (torch.norm(W_k, dim=-1, keepdim=True) + 1e-8)
    sim = torch.mm(z_k_norm, W_k_norm.t()) # [B, C]
    return sim

# Class-size scaling denominator constants (CSC) per task
csc_denom = torch.tensor([np.sqrt(2 * np.log(C_tasks[k]) / d) for k in range(K)]).to(device)

def compute_merged_logits(z, alpha, C_prototypes, tasks):
    """
    Simulates weight ensembling forward pass.
    z: [B, D] representations
    alpha: [B, K] ensembling coefficients
    C_prototypes: [K, C, d] expert classifier heads
    tasks: [B] target task IDs
    """
    B_size = len(z)
    logits = torch.zeros(B_size, C).to(device)
    
    # Reshape representations into expert blocks [B, K, d]
    z_blocks = z.view(B_size, K, d)
    
    for b in range(B_size):
        k = tasks[b].item()
        
        # Ensembled representation block k
        z_base_k = torch.randn(d).to(device) * 0.1
        z_expert_k = z_blocks[b, k] # is already pre-built with expert signal + noise
        
        # Simulated interference vectors from other experts
        interference = torch.zeros(d).to(device)
        for i in range(K):
            if i != k:
                interference += alpha[b, i] * (torch.randn(d).to(device) * 0.5)
                
        # Merged representation:
        # z(alpha) = alpha_k * z_expert + (1 - alpha_k) * z_base + interference
        z_merged_k = alpha[b, k] * z_expert_k + (1.0 - alpha[b, k]) * z_base_k + interference
        
        # Classify using expert classification head k
        W_k = C_prototypes[k] # [C, d]
        b_logits = torch.mv(W_k, z_merged_k)
        if C_tasks[k] < C:
            b_logits[C_tasks[k]:] = -1e9
        logits[b] = b_logits
        
    return logits

# Parametric routers for training baselines
class LinearRouterModel(nn.Module):
    def __init__(self, init_zero=False):
        super().__init__()
        self.fc = nn.Linear(D, K)
        if init_zero:
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
            
    def forward(self, z):
        return self.fc(z)

# Trainable Calibrated Routers Setup
def train_routers(C_prototypes, cal_features, cal_labels, cal_tasks):
    """
    Calibrates trainable routers (Linear, QWS, L3-Softmax) on the 64-sample split.
    """
    # 1. Unregularized Linear Router
    lin_router = LinearRouterModel(init_zero=False).to(device)
    lin_opt = optim.Adam(lin_router.parameters(), lr=1e-2)
    
    # 2. QWS-Merge SOTA (unregularized wave activation)
    qws_router = LinearRouterModel(init_zero=False).to(device)
    qws_opt = optim.Adam(qws_router.parameters(), lr=1e-2)
    
    # 3. L3-Softmax Well-Regularized Router (zero-initialized, L2 regularized)
    l3_router = LinearRouterModel(init_zero=True).to(device)
    l3_opt = optim.Adam(l3_router.parameters(), lr=1e-2, weight_decay=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    
    for step in range(100):
        # --- Train Linear Router ---
        lin_opt.zero_grad()
        lin_logits_route = lin_router(cal_features)
        lin_alpha = torch.softmax(lin_logits_route, dim=-1)
        lin_pred_logits = compute_merged_logits(cal_features, lin_alpha, C_prototypes, cal_tasks)
        lin_loss = criterion(lin_pred_logits, cal_labels)
        lin_loss.backward()
        lin_opt.step()
        
        # --- Train QWS-Merge ---
        qws_opt.zero_grad()
        qws_logits_route = qws_router(cal_features)
        # Cosine wave activation squared
        qws_alpha = torch.cos(qws_logits_route) ** 2
        qws_alpha = qws_alpha / (torch.sum(qws_alpha, dim=-1, keepdim=True) + 1e-8)
        qws_pred_logits = compute_merged_logits(cal_features, qws_alpha, C_prototypes, cal_tasks)
        qws_loss = criterion(qws_pred_logits, cal_labels)
        qws_loss.backward()
        qws_opt.step()
        
        # --- Train L3-Softmax ---
        l3_opt.zero_grad()
        l3_logits_route = l3_router(cal_features)
        l3_alpha = torch.softmax(l3_logits_route, dim=-1)
        l3_pred_logits = compute_merged_logits(cal_features, l3_alpha, C_prototypes, cal_tasks)
        l3_loss = criterion(l3_pred_logits, cal_labels)
        l3_loss.backward()
        l3_opt.step()
        
    return lin_router, qws_router, l3_router

def evaluate_method(method_name, model_objs, C_prototypes, Fisher_M, test_features, test_labels, test_tasks, batch_size, is_homogeneous):
    """
    Evaluates classification accuracy of a specific model/baseline under heterogeneous/homogeneous batch settings.
    """
    lin_router, qws_router, l3_router = model_objs
    
    total_correct = 0
    total_samples = len(test_features)
    
    # 1. Homogeneous setting: evaluate task-by-task and average results
    if is_homogeneous:
        task_accuracies = []
        for k in range(K):
            mask = (test_tasks == k)
            feat_k = test_features[mask]
            lbl_k = test_labels[mask]
            tasks_k = test_tasks[mask]
            
            num_task_samples = len(feat_k)
            correct_k = 0
            
            # Process in batches
            for i in range(0, num_task_samples, batch_size):
                end_idx = min(i + batch_size, num_task_samples)
                batch_feat = feat_k[i:end_idx]
                batch_lbl = lbl_k[i:end_idx]
                batch_tasks = tasks_k[i:end_idx]
                
                # Compute ensembling coefficients
                if method_name in ["PFSR_MBH", "FIOSR"]:
                    alpha = get_mbh_coefficients(method_name, batch_feat, C_prototypes, Fisher_M)
                else:
                    alpha = get_coefficients(method_name, batch_feat, model_objs, C_prototypes, Fisher_M, batch_size)
                
                # Forward pass
                pred_logits = compute_merged_logits(batch_feat, alpha, C_prototypes, batch_tasks)
                preds = torch.argmax(pred_logits, dim=-1)
                correct_k += torch.sum(preds == batch_lbl).item()
                
            task_accuracies.append(correct_k / num_task_samples)
            
        return np.mean(task_accuracies) * 100.0, task_accuracies
        
    # 2. Heterogeneous setting: evaluate mixed streams with batch-averaging or MBH
    else:
        # Shuffle stream to ensure task-interleaved batching
        shuffled_indices = torch.randperm(total_samples)
        feat_stream = test_features[shuffled_indices]
        lbl_stream = test_labels[shuffled_indices]
        tasks_stream = test_tasks[shuffled_indices]
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_feat = feat_stream[i:end_idx]
            batch_lbl = lbl_stream[i:end_idx]
            batch_tasks = tasks_stream[i:end_idx]
            
            # Dynamic parameter-free routing with Micro-Batch Homogenization (MBH)
            if method_name in ["PFSR_MBH", "FIOSR"]:
                alpha = get_mbh_coefficients(method_name, batch_feat, C_prototypes, Fisher_M)
            else:
                # Parametric or static methods: average coefficients across the batch if batch_size > 1 (heterogeneity collapse)
                alpha = get_coefficients(method_name, batch_feat, model_objs, C_prototypes, Fisher_M, batch_size)
                if len(batch_feat) > 1:
                    batch_avg_alpha = torch.mean(alpha, dim=0, keepdim=True).expand(len(batch_feat), -1)
                    alpha = batch_avg_alpha
                    
            pred_logits = compute_merged_logits(batch_feat, alpha, C_prototypes, batch_tasks)
            preds = torch.argmax(pred_logits, dim=-1)
            total_correct += torch.sum(preds == batch_lbl).item()
            
        return (total_correct / total_samples) * 100.0, None

def get_coefficients(method_name, feat, model_objs, C_prototypes, Fisher_M, batch_size):
    """
    Computes sample-wise dynamic merging coefficients for standard baselines.
    """
    B_size = len(feat)
    lin_router, qws_router, l3_router = model_objs
    
    if method_name == "StaticUniform":
        return torch.ones(B_size, K).to(device) * 0.25
        
    elif method_name == "Linear_Unreg":
        with torch.no_grad():
            logits = lin_router(feat)
            return torch.softmax(logits, dim=-1)
            
    elif method_name == "QWS_Merge_SOTA":
        with torch.no_grad():
            logits = qws_router(feat)
            alpha = torch.cos(logits) ** 2
            return alpha / (torch.sum(alpha, dim=-1, keepdim=True) + 1e-8)
            
    elif method_name == "L3_Softmax_Reg":
        with torch.no_grad():
            logits = l3_router(feat)
            return torch.softmax(logits, dim=-1)
            
    else:
        raise ValueError(f"Unknown baseline: {method_name}")

def get_mbh_coefficients(method_name, feat, C_prototypes, Fisher_M):
    """
    Computes merging coefficients under Micro-Batch Homogenization (MBH).
    Supports standard PFSR (Cosine) and FIOSR (Fisher-weighted Cosine).
    """
    B_size = len(feat)
    z_blocks = feat.view(B_size, K, d)
    
    u = torch.zeros(B_size, K).to(device)
    
    # 1. Compute projection coordinates for each task
    for k in range(K):
        W_k = C_prototypes[k, :C_tasks[k]] # [C_tasks[k], d] (sliced to task-specific class size)
        z_k = z_blocks[:, k, :] # [B, d]
        
        if method_name == "FIOSR":
            F_k = Fisher_M[k, :C_tasks[k]] # [C_tasks[k], d]
            sims = fisher_weighted_cosine_similarity(z_k, W_k, F_k) # [B, C_tasks[k]]
        else: # PFSR_MBH (standard Cosine)
            sims = standard_cosine_similarity(z_k, W_k) # [B, C_tasks[k]]
            
        # Raw coordinate is max similarity across class prototypes
        u[:, k] = torch.max(sims, dim=-1)[0]
        
    # 2. Class-Size Scaling Calibration (CSC)
    u_calibrated = u / csc_denom
    
    # 3. Softmax temperature mapping (near-discrete routing)
    tau = 0.001
    alpha_raw = torch.softmax(u_calibrated / tau, dim=-1)
    
    # 4. Micro-Batch Homogenization (MBH)
    dominant_tasks = torch.argmax(u_calibrated, dim=-1)
    alpha_mbh = alpha_raw.clone()
    
    for k in range(K):
        mask = (dominant_tasks == k)
        if torch.sum(mask) > 0:
            # Average coefficients within the homogeneous micro-batch
            mean_alpha = torch.mean(alpha_raw[mask], dim=0)
            alpha_mbh[mask] = mean_alpha
            
    return alpha_mbh


# --- MAIN EXPERIMENT LOOP ---
if __name__ == "__main__":
    seeds = list(range(42, 52)) # seeds 42 to 51
    methods = ["StaticUniform", "Linear_Unreg", "QWS_Merge_SOTA", "L3_Softmax_Reg", "PFSR_MBH", "FIOSR"]

    # Store final results
    # For Homogeneous (B=256)
    homo_results = {m: [] for m in methods}
    # Store individual task ceiling results for FIOSR
    task_ceilings = {i: [] for i in range(K)}

    # For Heterogeneous Streams over Batch Sizes B
    batch_sizes = [1, 8, 32, 128, 512]
    hetero_results = {m: {b: [] for b in batch_sizes} for m in methods}

    print("Starting Sandbox Experiment across 10 independent random seeds...")

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        
        # Generate data
        C_prototypes, cal_f, cal_l, cal_t, test_f, test_l, test_t = generate_sandbox_data(seed, rho=0.33)
        
        # Pre-calibration mean-centering to eliminate translation bias (Section 3.3 / Appendix)
        mean_cal = cal_f.mean(dim=0, keepdim=True)
        cal_f = cal_f - mean_cal
        test_f = test_f - mean_cal
        
        # Precompute Diagonal Fisher Matrix
        Fisher_M = estimate_diagonal_fisher(C_prototypes, cal_f, cal_l, cal_t)
        
        # Train parametric baselines
        model_objs = train_routers(C_prototypes, cal_f, cal_l, cal_t)
        
        # --- 1. Homogeneous Evaluation (B=256) ---
        for m in methods:
            acc, task_accs = evaluate_method(m, model_objs, C_prototypes, Fisher_M, test_f, test_l, test_t, batch_size=256, is_homogeneous=True)
            homo_results[m].append(acc)
            
            # Log task-level ceilings under FIOSR
            if m == "FIOSR":
                for k in range(K):
                    task_ceilings[k].append(task_accs[k] * 100.0)
                    
        # --- 2. Heterogeneous Stream Evaluation ---
        for b in batch_sizes:
            for m in methods:
                acc, _ = evaluate_method(m, model_objs, C_prototypes, Fisher_M, test_f, test_l, test_t, batch_size=b, is_homogeneous=False)
                hetero_results[m][b].append(acc)
                
    # --- AGGREGATE AND PRINT RESULTS ---
    print("\n" + "="*50)
    print("HOMOGENEOUS EXPERIMENT RESULTS (B=256, mean +/- std %)")
    print("="*50)
    homo_summary = {}
    for m in methods:
        mean = np.mean(homo_results[m])
        std = np.std(homo_results[m])
        homo_summary[m] = {"mean": mean, "std": std}
        print(f"{m:<18}: {mean:.2f}% +/- {std:.2f}%")
        
    print("\n" + "="*50)
    print("FIOSR INDIVIDUAL TASK CEILING SUMMARY")
    print("="*50)
    task_names = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
    for k in range(K):
        mean = np.mean(task_ceilings[k])
        std = np.std(task_ceilings[k])
        print(f"{task_names[k]:<15}: {mean:.2f}% +/- {std:.2f}%")

    print("\n" + "="*50)
    print("HETEROGENEOUS STREAM EXPERIMENT RESULTS (mean %)")
    print("="*50)
    print(f"{'Method':<18} | {'B=1':<7} | {'B=8':<7} | {'B=32':<7} | {'B=128':<7} | {'B=512':<7}")
    print("-"*70)

    hetero_summary = {}
    for m in methods:
        hetero_summary[m] = {}
        row_str = f"{m:<18} | "
        for b in batch_sizes:
            mean = np.mean(hetero_results[m][b])
            hetero_summary[m][b] = mean
            row_str += f"{mean:.2f}%  | "
        print(row_str)
        
    # --- GENERATE RESULTS PLOT ---
    plt.figure(figsize=(10, 6))
    colors = ['gray', 'red', 'orange', 'blue', 'purple', 'green']
    markers = ['o', 'x', '^', 's', 'd', '*']

    for idx, m in enumerate(methods):
        y_vals = [np.mean(hetero_results[m][b]) for b in batch_sizes]
        # Standard deviation error bars
        y_errs = [np.std(hetero_results[m][b]) for b in batch_sizes]
        plt.errorbar(batch_sizes, y_vals, yerr=y_errs, label=m, color=colors[idx], marker=markers[idx], capsize=5, linewidth=2)
        
    plt.xscale('log')
    plt.xticks(batch_sizes, [str(b) for b in batch_sizes])
    plt.xlabel('Batch Size (B) in Heterogeneous Stream', fontsize=12)
    plt.ylabel('Joint Classification Accuracy (%)', fontsize=12)
    plt.title('Vulnerability to Heterogeneity Collapse across Batch Sizes', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fiosr_vs_baselines.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved successfully to results/fiosr_vs_baselines.png")

    # --- SAVE METRICS JSON ---
    metrics_payload = {
        "homogeneous_results": {m: {"mean": np.mean(homo_results[m]), "std": np.std(homo_results[m])} for m in methods},
        "heterogeneous_results": {m: {str(b): {"mean": np.mean(hetero_results[m][b]), "std": np.std(hetero_results[m][b])} for b in batch_sizes} for m in methods},
        "task_ceilings": {task_names[k]: {"mean": np.mean(task_ceilings[k]), "std": np.std(task_ceilings[k])} for k in range(K)}
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("Metrics saved successfully to results/metrics.json")
