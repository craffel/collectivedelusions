import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import ttest_rel
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Setup seed and devices
DEVICE = torch.device("cpu")
NUM_LAYERS = 14
NUM_TASKS = 4
D_FEAT = 192
NUM_CLASSES = 10
NUM_CAL = 10  # 10 samples per task (1 per class, total 40)
NUM_TRAIN = 300  # 300 training samples per task
NUM_TEST = 500  # 500 test samples per task
D_HIDDEN = 64

# Uniform baseline target theta_0
THETA_UNIFORM = np.log(1.0 / (NUM_TASKS - 1)) # ln(1/3) = -1.098612

# ----------------------------------------------------
# Global Cache of Real-World Datasets
# ----------------------------------------------------
print("Caching real-world datasets in memory...")
transform = transforms.Compose([transforms.ToTensor()])

mnist_train = dset.MNIST(root='./data', train=True, download=False, transform=transform)
mnist_test = dset.MNIST(root='./data', train=False, download=False, transform=transform)

fmnist_train = dset.FashionMNIST(root='./data', train=True, download=False, transform=transform)
fmnist_test = dset.FashionMNIST(root='./data', train=False, download=False, transform=transform)

cifar_train = dset.CIFAR10(root='./data', train=True, download=False, transform=transform)
cifar_test = dset.CIFAR10(root='./data', train=False, download=False, transform=transform)

svhn_train = dset.SVHN(root='./data', split='train', download=False, transform=transform)
svhn_test = dset.SVHN(root='./data', split='test', download=False, transform=transform)

class DatasetPool:
    def __init__(self, ds_list):
        self.ds_list = ds_list
        self.total_len = sum(len(d) for d in ds_list)
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, idx):
        offset = 0
        for d in self.ds_list:
            if idx - offset < len(d):
                img, lbl = d[idx - offset]
                return img, lbl
            offset += len(d)
        raise IndexError()

GLOBAL_POOLS = {
    0: DatasetPool([mnist_train, mnist_test]),
    1: DatasetPool([fmnist_train, fmnist_test]),
    2: DatasetPool([cifar_train, cifar_test]),
    3: DatasetPool([svhn_train, svhn_test])
}

# ----------------------------------------------------
# Baselines (TIES, DARE)
# ----------------------------------------------------
def ties_merge_layer(V_param, p_trim=0.20):
    # V_param shape: (K, ...)
    K = V_param.shape[0]
    V_pruned = V_param.clone()
    for k in range(K):
        v = V_param[k]
        flat_v = v.flatten()
        k_val = int(p_trim * flat_v.numel())
        if k_val > 0:
            threshold = torch.topk(torch.abs(flat_v), k_val).values[-1]
            mask = torch.abs(v) >= threshold
            V_pruned[k] = torch.where(mask, v, torch.zeros_like(v))
        else:
            V_pruned[k] = torch.zeros_like(v)
            
    signs = torch.sign(V_pruned)
    sign_sum = torch.sum(signs, dim=0)
    consensus_sign = torch.sign(sign_sum)
    
    matching_mask = (torch.sign(V_pruned) == consensus_sign) & (V_pruned != 0)
    
    sum_vals = torch.sum(torch.where(matching_mask, V_pruned, torch.zeros_like(V_pruned)), dim=0)
    count_vals = torch.sum(matching_mask.float(), dim=0)
    
    merged = torch.where(count_vals > 0, sum_vals / count_vals, torch.zeros_like(sum_vals))
    return merged

def dare_merge_layer(V_param, p_drop=0.90):
    # V_param shape: (K, ...)
    K = V_param.shape[0]
    scale = 1.0 / (1.0 - p_drop)
    V_pruned = []
    for k in range(K):
        v = V_param[k]
        mask = (torch.rand_like(v) >= p_drop).float()
        V_pruned.append(v * mask * scale)
    V_pruned = torch.stack(V_pruned)
    return torch.mean(V_pruned, dim=0)

# ----------------------------------------------------
# Real-World Subsampled & Projected Sandbox
# ----------------------------------------------------
class RealWorldSandbox:
    def __init__(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Build Johnson-Lindenstrauss random projection matrices
        self.proj_matrices = {
            0: torch.randn(784, D_FEAT) / np.sqrt(D_FEAT),
            1: torch.randn(784, D_FEAT) / np.sqrt(D_FEAT),
            2: torch.randn(3072, D_FEAT) / np.sqrt(D_FEAT),
            3: torch.randn(3072, D_FEAT) / np.sqrt(D_FEAT)
        }
        
        # Permute consolidated pools randomly for each seed
        self.permuted_indices = {k: torch.randperm(len(GLOBAL_POOLS[k])) for k in range(NUM_TASKS)}
        
        # Sequential pointer tracking to ensure disjoint train/cal/test splits
        self.drawn_indices = {k: 0 for k in range(NUM_TASKS)}
        
    def generate_data(self, num_samples_per_task):
        features_list = []
        labels_list = []
        task_ids_list = []
        
        for k in range(NUM_TASKS):
            imgs = []
            lbls = []
            start_ptr = self.drawn_indices[k]
            
            for idx in range(start_ptr, start_ptr + num_samples_per_task):
                real_idx = self.permuted_indices[k][idx].item()
                img, lbl = GLOBAL_POOLS[k][real_idx]
                imgs.append(img.flatten())
                lbls.append(lbl)
                
            self.drawn_indices[k] += num_samples_per_task
            
            imgs = torch.stack(imgs)
            lbls = torch.tensor(lbls, dtype=torch.long)
            
            # Apply Johnson-Lindenstrauss random projection
            projected = imgs @ self.proj_matrices[k]
            
            # Z-score normalize features to standard scaling
            projected = (projected - projected.mean(dim=-1, keepdim=True)) / (projected.std(dim=-1, keepdim=True) + 1e-6)
            
            features_list.append(projected)
            labels_list.append(lbls)
            task_ids_list.append(torch.full((projected.size(0),), k, dtype=torch.long))
            
        return features_list, labels_list, task_ids_list

# ----------------------------------------------------
# 14-Layer Deep Residual MLP Architecture (with 0.1 scaling)
# ----------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(D_FEAT, D_HIDDEN)
        self.mid_layers = nn.ModuleList([nn.Linear(D_HIDDEN, D_HIDDEN) for _ in range(12)])
        self.fc_out = nn.Linear(D_HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
    def forward(self, x):
        h = self.relu(self.fc_in(x))
        for idx, layer in enumerate(self.mid_layers):
            if idx % 3 == 0:
                h = h + 0.1 * self.relu(layer(h))  # Stable residual branch scaling
            elif idx % 3 == 1:
                h = h + 0.1 * self.gelu(layer(h))
            else:
                h = h + 0.1 * layer(h)
        return self.fc_out(h)

def get_weights(model):
    W, B = [], []
    W.append(model.fc_in.weight.data.clone())
    B.append(model.fc_in.bias.data.clone())
    for layer in model.mid_layers:
        W.append(layer.weight.data.clone())
        B.append(layer.bias.data.clone())
    W.append(model.fc_out.weight.data.clone())
    B.append(model.fc_out.bias.data.clone())
    return W, B

def set_weights(model, W, B):
    model.fc_in.weight.data = W[0].clone()
    model.fc_in.bias.data = B[0].clone()
    for idx, layer in enumerate(model.mid_layers):
        layer.weight.data = W[idx+1].clone()
        layer.bias.data = B[idx+1].clone()
    model.fc_out.weight.data = W[13].clone()
    model.fc_out.bias.data = B[13].clone()

def forward_functional(x, W, B):
    h = F.relu(F.linear(x, W[0], B[0]))
    for l in range(1, 13):
        idx = l - 1
        if idx % 3 == 0:
            h = h + 0.1 * F.relu(F.linear(h, W[l], B[l]))
        elif idx % 3 == 1:
            h = h + 0.1 * F.gelu(F.linear(h, W[l], B[l]))
        else:
            h = h + 0.1 * F.linear(h, W[l], B[l])
    out = F.linear(h, W[13], B[13])
    return out

def train_classifier(model, features, labels, num_epochs=20, lr=0.01):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    x = features.to(DEVICE)
    y = labels.to(DEVICE)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

def evaluate_classifier(model, features, labels):
    model.eval()
    with torch.no_grad():
        out = model(features.to(DEVICE))
        preds = out.argmax(dim=-1)
        acc = (preds == labels.to(DEVICE)).float().mean().item()
    return acc

def evaluate_merged_model_functional(W, B, test_features_list, test_labels_list):
    accs = []
    with torch.no_grad():
        for k in range(NUM_TASKS):
            out = forward_functional(test_features_list[k].to(DEVICE), W, B)
            preds = out.argmax(dim=-1)
            acc = (preds == test_labels_list[k].to(DEVICE)).float().mean().item()
            accs.append(acc)
    return accs

# ----------------------------------------------------
# Trajectory Expansions: Cubic (Degree 3) Polynomials
# ----------------------------------------------------
def compute_coefficients(theta, device):
    t = torch.linspace(0, 1, NUM_LAYERS, device=device)
    alpha = torch.zeros(NUM_LAYERS, NUM_TASKS, device=device)
    for k in range(NUM_TASKS):
        poly = theta[k, 0] + theta[k, 1] * t + theta[k, 2] * (t ** 2) + theta[k, 3] * (t ** 4) # Degree 3 (4 params)
        alpha[:, k] = torch.sigmoid(poly)
    return alpha

# ----------------------------------------------------
# Helper to run a PAC-Bayes variant for Ablations
# ----------------------------------------------------
def run_pac_bayes_variant(W_base, B_base, V_W, V_B, cal_feats_device, cal_labels_device, test_feats, test_labels, lambda_pac, sigma_pac, device):
    theta_pac = torch.zeros(NUM_TASKS, 4, device=device)
    theta_pac[:, 0] = THETA_UNIFORM
    theta_pac.requires_grad = True
    optimizer = optim.AdamW([theta_pac], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    S = 3
    for step in range(50):
        optimizer.zero_grad()
        ce_loss = 0.0
        for _ in range(S):
            noise = torch.randn_like(theta_pac) * sigma_pac
            theta_sampled = theta_pac + noise
            alpha = compute_coefficients(theta_sampled, device)
            W_m, B_m = [], []
            for l in range(NUM_LAYERS):
                a_l = alpha[l, :]
                W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
            sample_ce = 0.0
            for k in range(NUM_TASKS):
                logits = forward_functional(cal_feats_device[k], W_m, B_m)
                task_loss = criterion(logits, cal_labels_device[k])
                sample_ce += torch.clamp(task_loss, max=5.0)
            ce_loss += sample_ce / NUM_TASKS
        ce_loss /= S
        reg_loss = 0.0
        for k in range(NUM_TASKS):
            reg_loss += (theta_pac[k, 0] - THETA_UNIFORM) ** 2 + theta_pac[k, 1] ** 2 + theta_pac[k, 2] ** 2 + theta_pac[k, 3] ** 2
        total_loss = ce_loss + lambda_pac * reg_loss
        total_loss.backward()
        optimizer.step()
        
    S_test = 5
    pac_accs = []
    with torch.no_grad():
        for k in range(NUM_TASKS):
            accumulated_probs = None
            for _ in range(S_test):
                noise = torch.randn_like(theta_pac) * sigma_pac
                theta_sampled = theta_pac + noise
                alpha_sampled = compute_coefficients(theta_sampled, device)
                W_m, B_m = [], []
                for l in range(NUM_LAYERS):
                    a_l = alpha_sampled[l, :]
                    W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                    B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                logits = forward_functional(test_feats[k].to(device), W_m, B_m)
                probs = F.softmax(logits, dim=-1)
                if accumulated_probs is None:
                    accumulated_probs = probs
                else:
                    accumulated_probs += probs
            avg_probs = accumulated_probs / S_test
            preds = avg_probs.argmax(dim=-1)
            acc = (preds == test_labels[k].to(device)).float().mean().item()
            pac_accs.append(acc)
    return pac_accs

# ----------------------------------------------------
# Main Experiment for a Seed
# ----------------------------------------------------
def run_experiment_for_seed(seed):
    print(f"\n--- Running Seed {seed} ---")
    sandbox = RealWorldSandbox(seed)
    
    # Generate splits
    train_feats, train_labels, _ = sandbox.generate_data(NUM_TRAIN)
    cal_feats_max, cal_labels_max, _ = sandbox.generate_data(20)  # Draw 20 calibration samples per task
    cal_feats = [x[:NUM_CAL] for x in cal_feats_max]
    cal_labels = [y[:NUM_CAL] for y in cal_labels_max]
    test_feats, test_labels, _ = sandbox.generate_data(NUM_TEST)
    
    # Initialize base model
    torch.manual_seed(seed)
    base_model = MLPClassifier().to(DEVICE)
    W_base, B_base = get_weights(base_model)
    
    # Fine-tune experts
    W_experts, B_experts = [], []
    expert_ceilings = []
    
    for k in range(NUM_TASKS):
        print(f"Fine-tuning Expert {k} on Task {k}...")
        expert_model = MLPClassifier()
        set_weights(expert_model, W_base, B_base)
        train_classifier(expert_model, train_feats[k], train_labels[k], num_epochs=20, lr=0.01)
        W_exp, B_exp = get_weights(expert_model)
        W_experts.append(W_exp)
        B_experts.append(B_exp)
        ceil_acc = evaluate_classifier(expert_model, test_feats[k], test_labels[k])
        expert_ceilings.append(ceil_acc)
        
    print(f"Expert Ceilings: {[round(x*100, 2) for x in expert_ceilings]} (Mean: {round(np.mean(expert_ceilings)*100, 2)}%)")
    
    # Form task vectors
    V_W = []
    V_B = []
    for l in range(NUM_LAYERS):
        V_W.append(torch.stack([W_experts[k][l] - W_base[l] for k in range(NUM_TASKS)]).to(DEVICE))
        V_B.append(torch.stack([B_experts[k][l] - B_base[l] for k in range(NUM_TASKS)]).to(DEVICE))
        
    cal_feats_device = [x.to(DEVICE) for x in cal_feats]
    cal_labels_device = [y.to(DEVICE) for y in cal_labels]
    criterion = nn.CrossEntropyLoss()
    
    # ------------------
    # Baseline 1: Static Uniform Merging
    # ------------------
    alpha_uniform = torch.full((NUM_LAYERS, NUM_TASKS), 0.25, device=DEVICE)
    W_uniform, B_uniform = [], []
    for l in range(NUM_LAYERS):
        w_l = W_base[l] + torch.sum(alpha_uniform[l][:, None, None] * V_W[l], dim=0)
        b_l = B_base[l] + torch.sum(alpha_uniform[l][:, None] * V_B[l], dim=0)
        W_uniform.append(w_l)
        B_uniform.append(b_l)
    static_uniform_accs = evaluate_merged_model_functional(W_uniform, B_uniform, test_feats, test_labels)
    print(f"Static Uniform: {[round(x*100, 2) for x in static_uniform_accs]} (Mean: {round(np.mean(static_uniform_accs)*100, 2)}%)")
    
    # ------------------
    # Baseline 1b: TIES-Merge
    # ------------------
    W_ties, B_ties = [], []
    for l in range(NUM_LAYERS):
        w_l = W_base[l] + ties_merge_layer(V_W[l], p_trim=0.80)
        b_l = B_base[l] + ties_merge_layer(V_B[l], p_trim=0.80)
        W_ties.append(w_l)
        B_ties.append(b_l)
    ties_accs = evaluate_merged_model_functional(W_ties, B_ties, test_feats, test_labels)
    print(f"TIES-Merge: {[round(x*100, 2) for x in ties_accs]} (Mean: {round(np.mean(ties_accs)*100, 2)}%)")
    
    # ------------------
    # Baseline 1c: DARE-Merge
    # ------------------
    W_dare, B_dare = [], []
    for l in range(NUM_LAYERS):
        w_l = W_base[l] + dare_merge_layer(V_W[l], p_drop=0.10)
        b_l = B_base[l] + dare_merge_layer(V_B[l], p_drop=0.10)
        W_dare.append(w_l)
        B_dare.append(b_l)
    dare_accs = evaluate_merged_model_functional(W_dare, B_dare, test_feats, test_labels)
    print(f"DARE-Merge: {[round(x*100, 2) for x in dare_accs]} (Mean: {round(np.mean(dare_accs)*100, 2)}%)")
    
    # ------------------
    # Baseline 2: Offline Unconstrained Few-Shot Tuning
    # ------------------
    gamma = torch.full((NUM_TASKS, NUM_LAYERS), THETA_UNIFORM, device=DEVICE, requires_grad=True)
    optimizer = optim.AdamW([gamma], lr=0.01)
    for step in range(50):
        optimizer.zero_grad()
        alpha = torch.sigmoid(gamma)
        W_m, B_m = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha[:, l]
            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        total_loss = 0.0
        for k in range(NUM_TASKS):
            logits = forward_functional(cal_feats_device[k], W_m, B_m)
            task_loss = criterion(logits, cal_labels_device[k])
            total_loss += torch.clamp(task_loss, max=5.0)
        total_loss /= NUM_TASKS
        total_loss.backward()
        optimizer.step()
    with torch.no_grad():
        alpha_opt = torch.sigmoid(gamma)
        W_opt, B_opt = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha_opt[:, l]
            W_opt.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_opt.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        offline_unconstrained_accs = evaluate_merged_model_functional(W_opt, B_opt, test_feats, test_labels)
    print(f"Offline Unconstrained: {[round(x*100, 2) for x in offline_unconstrained_accs]} (Mean: {round(np.mean(offline_unconstrained_accs)*100, 2)}%)")
    
    # ------------------
    # Baseline 3: Rademacher-Bounded Polynomial Merging (RBPM)
    # ------------------
    theta_rbpm = torch.zeros(NUM_TASKS, 4, device=DEVICE)
    theta_rbpm[:, 0] = THETA_UNIFORM
    theta_rbpm.requires_grad = True
    optimizer = optim.AdamW([theta_rbpm], lr=0.01)
    for step in range(50):
        optimizer.zero_grad()
        alpha = compute_coefficients(theta_rbpm, DEVICE)
        W_m, B_m = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha[l, :]
            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        ce_loss = 0.0
        for k in range(NUM_TASKS):
            logits = forward_functional(cal_feats_device[k], W_m, B_m)
            task_loss = criterion(logits, cal_labels_device[k])
            ce_loss += torch.clamp(task_loss, max=5.0)
        ce_loss /= NUM_TASKS
        reg_loss = 0.0
        for k in range(NUM_TASKS):
            reg_loss += torch.abs(theta_rbpm[k, 0] - THETA_UNIFORM)
            reg_loss += torch.abs(theta_rbpm[k, 1])
            reg_loss += torch.abs(theta_rbpm[k, 2])
            reg_loss += torch.abs(theta_rbpm[k, 3])
        total_loss = ce_loss + 0.01 * reg_loss
        total_loss.backward()
        optimizer.step()
    with torch.no_grad():
        alpha_opt = compute_coefficients(theta_rbpm, DEVICE)
        W_opt, B_opt = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha_opt[l, :]
            W_opt.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_opt.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        rbpm_accs = evaluate_merged_model_functional(W_opt, B_opt, test_feats, test_labels)
    print(f"RBPM (L1): {[round(x*100, 2) for x in rbpm_accs]} (Mean: {round(np.mean(rbpm_accs)*100, 2)}%)")
    
    # ------------------
    # Proposed: PAC-Bayes Merge (L2) with Randomized Training & Ensemble Evaluation
    # ------------------
    sigma_pac = 0.05
    lambda_pac = 0.010
    
    theta_pac = torch.zeros(NUM_TASKS, 4, device=DEVICE)
    theta_pac[:, 0] = THETA_UNIFORM
    theta_pac.requires_grad = True
    optimizer = optim.AdamW([theta_pac], lr=0.01)
    
    S = 3
    for step in range(50):
        optimizer.zero_grad()
        ce_loss = 0.0
        for _ in range(S):
            noise = torch.randn_like(theta_pac) * sigma_pac
            theta_sampled = theta_pac + noise
            alpha = compute_coefficients(theta_sampled, DEVICE)
            W_m, B_m = [], []
            for l in range(NUM_LAYERS):
                a_l = alpha[l, :]
                W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
            sample_ce = 0.0
            for k in range(NUM_TASKS):
                logits = forward_functional(cal_feats_device[k], W_m, B_m)
                task_loss = criterion(logits, cal_labels_device[k])
                sample_ce += torch.clamp(task_loss, max=5.0)
            ce_loss += sample_ce / NUM_TASKS
        ce_loss /= S
        reg_loss = 0.0
        for k in range(NUM_TASKS):
            reg_loss += (theta_pac[k, 0] - THETA_UNIFORM) ** 2 + theta_pac[k, 1] ** 2 + theta_pac[k, 2] ** 2 + theta_pac[k, 3] ** 2
        total_loss = ce_loss + lambda_pac * reg_loss
        total_loss.backward()
        optimizer.step()
        
    pac_det_accs = []
    with torch.no_grad():
        alpha_opt = compute_coefficients(theta_pac, DEVICE)
        W_m, B_m = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha_opt[l, :]
            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        pac_det_accs = evaluate_merged_model_functional(W_m, B_m, test_feats, test_labels)
    print(f"PAC-Bayes Merge (Deterministic compiled): {[round(x*100, 2) for x in pac_det_accs]} (Mean: {round(np.mean(pac_det_accs)*100, 2)}%)")

    S_test = 5
    pac_accs = []
    with torch.no_grad():
        for k in range(NUM_TASKS):
            accumulated_probs = None
            for _ in range(S_test):
                noise = torch.randn_like(theta_pac) * sigma_pac
                theta_sampled = theta_pac + noise
                alpha_sampled = compute_coefficients(theta_sampled, DEVICE)
                W_m, B_m = [], []
                for l in range(NUM_LAYERS):
                    a_l = alpha_sampled[l, :]
                    W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                    B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                logits = forward_functional(test_feats[k].to(DEVICE), W_m, B_m)
                probs = F.softmax(logits, dim=-1)
                if accumulated_probs is None:
                    accumulated_probs = probs
                else:
                    accumulated_probs += probs
            avg_probs = accumulated_probs / S_test
            preds = avg_probs.argmax(dim=-1)
            acc = (preds == test_labels[k].to(DEVICE)).float().mean().item()
            pac_accs.append(acc)
    print(f"PAC-Bayes Merge (L2, Ours): {[round(x*100, 2) for x in pac_accs]} (Mean: {round(np.mean(pac_accs)*100, 2)}%)")
    
    # Ablations
    print("Running Ablation Sweep 1: lambda_pac=0.001...")
    ablation_001_accs = run_pac_bayes_variant(W_base, B_base, V_W, V_B, cal_feats_device, cal_labels_device, test_feats, test_labels, 0.001, 0.05, DEVICE)
    print("Running Ablation Sweep 2: lambda_pac=0.50...")
    ablation_05_accs = run_pac_bayes_variant(W_base, B_base, V_W, V_B, cal_feats_device, cal_labels_device, test_feats, test_labels, 0.50, 0.05, DEVICE)
    print("Running Ablation Sweep 3: sigma_pac=0.01...")
    ablation_sig01_accs = run_pac_bayes_variant(W_base, B_base, V_W, V_B, cal_feats_device, cal_labels_device, test_feats, test_labels, 0.010, 0.01, DEVICE)
    print("Running Ablation Sweep 4: sigma_pac=0.15...")
    ablation_sig15_accs = run_pac_bayes_variant(W_base, B_base, V_W, V_B, cal_feats_device, cal_labels_device, test_feats, test_labels, 0.010, 0.15, DEVICE)
    
    # ------------------
    # Proposed Advanced: PAC-Bayes-FIM Merge (L2, FIM)
    # ------------------
    theta_init = torch.zeros(NUM_TASKS, 4, device=DEVICE)
    theta_init[:, 0] = THETA_UNIFORM
    theta_init.requires_grad = True
    fisher_diagonal = torch.zeros_like(theta_init)
    
    for k in range(NUM_TASKS):
        for i in range(cal_feats_device[k].size(0)):
            x_i = cal_feats_device[k][i:i+1]
            y_i = cal_labels_device[k][i:i+1]
            alpha = compute_coefficients(theta_init, DEVICE)
            W_m, B_m = [], []
            for l in range(NUM_LAYERS):
                a_l = alpha[l, :]
                W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
            logits = forward_functional(x_i, W_m, B_m)
            loss = criterion(logits, y_i)
            grads = torch.autograd.grad(loss, theta_init, retain_graph=False)[0]
            fisher_diagonal += grads ** 2
            
    fisher_diagonal = fisher_diagonal / (NUM_TASKS * cal_feats_device[0].size(0))
    fisher_diagonal = torch.clamp(fisher_diagonal, min=1e-5)
    fisher_diagonal = fisher_diagonal / fisher_diagonal.mean()
    
    theta_pac_fim = torch.zeros(NUM_TASKS, 4, device=DEVICE)
    theta_pac_fim[:, 0] = THETA_UNIFORM
    theta_pac_fim.requires_grad = True
    optimizer_fim = optim.AdamW([theta_pac_fim], lr=0.01)
    
    for step in range(50):
        optimizer_fim.zero_grad()
        ce_loss = 0.0
        for _ in range(S):
            noise = torch.randn_like(theta_pac_fim) * sigma_pac
            theta_sampled = theta_pac_fim + noise
            alpha = compute_coefficients(theta_sampled, DEVICE)
            W_m, B_m = [], []
            for l in range(NUM_LAYERS):
                a_l = alpha[l, :]
                W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
            sample_ce = 0.0
            for k in range(NUM_TASKS):
                logits = forward_functional(cal_feats_device[k], W_m, B_m)
                task_loss = criterion(logits, cal_labels_device[k])
                sample_ce += torch.clamp(task_loss, max=5.0)
            ce_loss += sample_ce / NUM_TASKS
        ce_loss /= S
        reg_loss = 0.0
        for k in range(NUM_TASKS):
            reg_loss += fisher_diagonal[k, 0] * (theta_pac_fim[k, 0] - THETA_UNIFORM) ** 2
            reg_loss += fisher_diagonal[k, 1] * theta_pac_fim[k, 1] ** 2
            reg_loss += fisher_diagonal[k, 2] * theta_pac_fim[k, 2] ** 2
            reg_loss += fisher_diagonal[k, 3] * theta_pac_fim[k, 3] ** 2
        total_loss = ce_loss + lambda_pac * reg_loss
        total_loss.backward()
        optimizer_fim.step()
        
    pac_fim_det_accs = []
    with torch.no_grad():
        alpha_opt = compute_coefficients(theta_pac_fim, DEVICE)
        W_m, B_m = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha_opt[l, :]
            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        pac_fim_det_accs = evaluate_merged_model_functional(W_m, B_m, test_feats, test_labels)
    print(f"PAC-Bayes-FIM Merge (Deterministic compiled): {[round(x*100, 2) for x in pac_fim_det_accs]} (Mean: {round(np.mean(pac_fim_det_accs)*100, 2)}%)")
        
    pac_fim_accs = []
    with torch.no_grad():
        for k in range(NUM_TASKS):
            accumulated_probs = None
            for _ in range(S_test):
                noise = torch.randn_like(theta_pac_fim) * sigma_pac
                theta_sampled = theta_pac_fim + noise
                alpha_sampled = compute_coefficients(theta_sampled, DEVICE)
                W_m, B_m = [], []
                for l in range(NUM_LAYERS):
                    a_l = alpha_sampled[l, :]
                    W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                    B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                logits = forward_functional(test_feats[k].to(DEVICE), W_m, B_m)
                probs = F.softmax(logits, dim=-1)
                if accumulated_probs is None:
                    accumulated_probs = probs
                else:
                    accumulated_probs += probs
            avg_probs = accumulated_probs / S_test
            preds = avg_probs.argmax(dim=-1)
            acc = (preds == test_labels[k].to(DEVICE)).float().mean().item()
            pac_fim_accs.append(acc)
    print(f"PAC-Bayes-FIM Merge (Randomized Ensemble): {[round(x*100, 2) for x in pac_fim_accs]} (Mean: {round(np.mean(pac_fim_accs)*100, 2)}%)")
    
    return {
        "expert_ceilings": expert_ceilings,
        "static_uniform": static_uniform_accs,
        "ties_merge": ties_accs,
        "dare_merge": dare_accs,
        "offline_unconstrained": offline_unconstrained_accs,
        "rbpm": rbpm_accs,
        "pac_bayes_det": pac_det_accs,
        "pac_bayes": pac_accs,
        "pac_bayes_fim_det": pac_fim_det_accs,
        "pac_bayes_fim": pac_fim_accs,
        "ablation_0.001": ablation_001_accs,
        "ablation_0.5": ablation_05_accs,
        "ablation_sigma_0.01": ablation_sig01_accs,
        "ablation_sigma_0.15": ablation_sig15_accs,
        "theta_pac": theta_pac.detach().cpu().numpy().tolist()
    }

def main():
    seeds = list(range(10, 25))
    results = {}
    for seed in seeds:
        res = run_experiment_for_seed(seed)
        results[seed] = res
        
    print("\n====================================")
    print("Aggregate Multi-Seed Results")
    print("====================================")
    
    methods = [
        "expert_ceilings", 
        "static_uniform", 
        "ties_merge",
        "dare_merge", 
        "offline_unconstrained", 
        "rbpm", 
        "pac_bayes_det", 
        "pac_bayes", 
        "pac_bayes_fim_det",
        "pac_bayes_fim",
        "ablation_0.001", 
        "ablation_0.5",
        "ablation_sigma_0.01",
        "ablation_sigma_0.15"
    ]
    task_names = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
    
    aggregated = {}
    for m in methods:
        aggregated[m] = {}
        for task_idx, name in enumerate(task_names):
            vals = [results[s][m][task_idx] for s in seeds]
            aggregated[m][name] = {
                "mean": np.mean(vals),
                "std": np.std(vals)
            }
        joint_means = [np.mean(results[s][m]) for s in seeds]
        aggregated[m]["Joint Mean"] = {
            "mean": np.mean(joint_means),
            "std": np.std(joint_means)
        }
        
    print("\n| Method | MNIST (%) | FashionMNIST (%) | CIFAR-10 (%) | SVHN (%) | Joint Mean (%) |")
    print("|---|---|---|---|---|---|")
    for m in methods:
        row_str = f"| {m.replace('_', ' ').title()} | "
        for name in task_names + ["Joint Mean"]:
            mean_val = aggregated[m][name]["mean"] * 100
            std_val = aggregated[m][name]["std"] * 100
            row_str += f"{mean_val:.2f} ± {std_val:.2f}% | "
        print(row_str)
        
    # Plot 1: Trajectories
    plt.figure(figsize=(8, 5))
    t = np.linspace(0, 1, NUM_LAYERS)
    theta_opt = np.array(results[10]["theta_pac"])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for k in range(NUM_TASKS):
        poly = theta_opt[k, 0] + theta_opt[k, 1] * t + theta_opt[k, 2] * (t ** 2) + theta_opt[k, 3] * (t ** 3)
        alpha = 1.0 / (1.0 + np.exp(-poly))
        plt.plot(range(1, NUM_LAYERS + 1), alpha, label=f"Task {k}: {task_names[k]}", color=colors[k], linewidth=2.5, marker='o')
        
    plt.axhline(0.25, color='gray', linestyle='--', alpha=0.7, label="Uniform Baseline")
    plt.title("Learned Polynomial Coefficient Trajectories (PAC-Bayes Merge, Seed 10)", fontsize=12, fontweight='bold')
    plt.xlabel("Layer Group (1 to 14)", fontsize=11)
    plt.ylabel("Ensembling Coefficient $\\alpha_k(l)$", fontsize=11)
    plt.xticks(range(1, NUM_LAYERS + 1))
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig("fig1_pacbayes_trajectories.png", dpi=150)
    plt.close()
    
    # Plot 2: Performance Comparison
    plt.figure(figsize=(10, 5))
    comp_methods = ["static_uniform", "ties_merge", "dare_merge", "offline_unconstrained", "rbpm", "pac_bayes", "pac_bayes_fim"]
    means = [aggregated[m]["Joint Mean"]["mean"] * 100 for m in comp_methods]
    stds = [aggregated[m]["Joint Mean"]["std"] * 100 for m in comp_methods]
    labels = ["Static Uniform", "TIES-Merge", "DARE-Merge", "Offline Unconstrained\n(Overfitted)", "RBPM (L1)", "PAC-Bayes Merge\n(Ours, L2)", "PAC-Bayes-FIM\n(Ours, FIM)"]
    colors = ['#aec7e8', '#dbdb8d', '#ff9896', '#ffbb78', '#98df8a', '#2ca02c', '#9467bd']
    
    bars = plt.bar(labels, means, yerr=stds, color=colors, edgecolor='dimgray', capsize=5, width=0.55)
    plt.ylabel("Multi-task Joint Mean Accuracy (%)", fontsize=11)
    plt.title("Comparative Evaluation of Model Merging Paradigms (15 Seeds)", fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1.5, f"{height:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("fig2_performance_comparison.png", dpi=150)
    plt.close()
    
    with open("results.json", "w") as f:
        json.dump(aggregated, f, indent=2)
        
    print("\nExperiments complete and figures saved!")

    # Statistical significance
    print("\n====================================")
    print("Paired t-test statistical significance analysis:")
    print("====================================")
    pac_bayes_seed_means = [np.mean(results[s]["pac_bayes"]) for s in seeds]
    pac_bayes_det_seed_means = [np.mean(results[s]["pac_bayes_det"]) for s in seeds]
    pac_bayes_fim_seed_means = [np.mean(results[s]["pac_bayes_fim"]) for s in seeds]
    pac_bayes_fim_det_seed_means = [np.mean(results[s]["pac_bayes_fim_det"]) for s in seeds]
    offline_unconstrained_seed_means = [np.mean(results[s]["offline_unconstrained"]) for s in seeds]
    rbpm_seed_means = [np.mean(results[s]["rbpm"]) for s in seeds]
    static_uniform_seed_means = [np.mean(results[s]["static_uniform"]) for s in seeds]

    stat_u, p_u = ttest_rel(pac_bayes_seed_means, offline_unconstrained_seed_means)
    print(f"PAC-Bayes (Ensemble) vs Offline Unconstrained: t = {stat_u:.4f}, p = {p_u:.6f}")
    stat_u_det, p_u_det = ttest_rel(pac_bayes_det_seed_means, offline_unconstrained_seed_means)
    print(f"PAC-Bayes (Deterministic Compiled) vs Offline Unconstrained: t = {stat_u_det:.4f}, p = {p_u_det:.6f}")

    stat_f_u, p_f_u = ttest_rel(pac_bayes_fim_seed_means, offline_unconstrained_seed_means)
    print(f"PAC-Bayes-FIM (Ensemble) vs Offline Unconstrained: t = {stat_f_u:.4f}, p = {p_f_u:.6f}")
    stat_f_u_det, p_f_u_det = ttest_rel(pac_bayes_fim_det_seed_means, offline_unconstrained_seed_means)
    print(f"PAC-Bayes-FIM (Deterministic Compiled) vs Offline Unconstrained: t = {stat_f_u_det:.4f}, p = {p_f_u_det:.6f}")

    stat_r, p_r = ttest_rel(pac_bayes_seed_means, rbpm_seed_means)
    print(f"PAC-Bayes (Ensemble) vs RBPM: t = {stat_r:.4f}, p = {p_r:.6f}")

    stat_s, p_s = ttest_rel(pac_bayes_seed_means, static_uniform_seed_means)
    print(f"PAC-Bayes (Ensemble) vs Static Uniform: t = {stat_s:.4f}, p = {p_s:.6f}")

    run_scarcity_sweep()

def run_scarcity_sweep():
    print("\n====================================")
    print("Running Calibration Scarcity Sweep...")
    print("====================================")
    seeds = list(range(10, 25))
    scarcity_values = [2, 5, 10, 20]
    
    results_scarcity = {
        "static_uniform": {M: [] for M in scarcity_values},
        "offline_unconstrained": {M: [] for M in scarcity_values},
        "pac_bayes": {M: [] for M in scarcity_values},
        "pac_bayes_fim": {M: [] for M in scarcity_values}
    }
    
    for seed in seeds:
        print(f"\n--- Sweep: Seed {seed} ---")
        sandbox = RealWorldSandbox(seed)
        
        train_feats, train_labels, _ = sandbox.generate_data(NUM_TRAIN)
        cal_feats_max, cal_labels_max, _ = sandbox.generate_data(20)  # Draw 20 calibration samples per task
        test_feats, test_labels, _ = sandbox.generate_data(NUM_TEST)
        
        torch.manual_seed(seed)
        base_model = MLPClassifier().to(DEVICE)
        W_base, B_base = get_weights(base_model)
        
        W_experts, B_experts = [], []
        for k in range(NUM_TASKS):
            expert_model = MLPClassifier()
            set_weights(expert_model, W_base, B_base)
            train_classifier(expert_model, train_feats[k], train_labels[k], num_epochs=20, lr=0.01)
            W_exp, B_exp = get_weights(expert_model)
            W_experts.append(W_exp)
            B_experts.append(B_exp)
            
        V_W, V_B = [], []
        for l in range(NUM_LAYERS):
            V_W.append(torch.stack([W_experts[k][l] - W_base[l] for k in range(NUM_TASKS)]).to(DEVICE))
            V_B.append(torch.stack([B_experts[k][l] - B_base[l] for k in range(NUM_TASKS)]).to(DEVICE))
            
        alpha_uniform = torch.full((NUM_LAYERS, NUM_TASKS), 0.25, device=DEVICE)
        W_uniform, B_uniform = [], []
        for l in range(NUM_LAYERS):
            w_l = W_base[l] + torch.sum(alpha_uniform[l][:, None, None] * V_W[l], dim=0)
            b_l = B_base[l] + torch.sum(alpha_uniform[l][:, None] * V_B[l], dim=0)
            W_uniform.append(w_l)
            B_uniform.append(b_l)
        uniform_accs = evaluate_merged_model_functional(W_uniform, B_uniform, test_feats, test_labels)
        uniform_mean = np.mean(uniform_accs)
        
        for M in scarcity_values:
            print(f"Evaluating M = {M} samples per task...")
            cal_feats = [x[:M] for x in cal_feats_max]
            cal_labels = [y[:M] for y in cal_labels_max]
            cal_feats_device = [x.to(DEVICE) for x in cal_feats]
            cal_labels_device = [y.to(DEVICE) for y in cal_labels]
            criterion = nn.CrossEntropyLoss()
            
            results_scarcity["static_uniform"][M].append(uniform_mean)
            
            # Offline Unconstrained
            gamma = torch.full((NUM_TASKS, NUM_LAYERS), THETA_UNIFORM, device=DEVICE, requires_grad=True)
            optimizer = optim.AdamW([gamma], lr=0.01)
            for step in range(50):
                optimizer.zero_grad()
                alpha = torch.sigmoid(gamma)
                W_m, B_m = [], []
                for l in range(NUM_LAYERS):
                    a_l = alpha[:, l]
                    W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                    B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                total_loss = 0.0
                for k in range(NUM_TASKS):
                    logits = forward_functional(cal_feats_device[k], W_m, B_m)
                    total_loss += torch.clamp(criterion(logits, cal_labels_device[k]), max=5.0)
                total_loss /= NUM_TASKS
                total_loss.backward()
                optimizer.step()
            with torch.no_grad():
                alpha_opt = torch.sigmoid(gamma)
                W_opt, B_opt = [], []
                for l in range(NUM_LAYERS):
                    a_l = alpha_opt[:, l]
                    W_opt.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                    B_opt.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                results_scarcity["offline_unconstrained"][M].append(np.mean(evaluate_merged_model_functional(W_opt, B_opt, test_feats, test_labels)))
                
            # Isotropic (with dynamic lambda_pac inversely proportional to M)
            sigma_pac = 0.05
            lambda_pac = 0.10 / M
            theta_pac = torch.zeros(NUM_TASKS, 4, device=DEVICE)
            theta_pac[:, 0] = THETA_UNIFORM
            theta_pac.requires_grad = True
            optimizer = optim.AdamW([theta_pac], lr=0.01)
            S = 3
            for step in range(50):
                optimizer.zero_grad()
                ce_loss = 0.0
                for _ in range(S):
                    noise = torch.randn_like(theta_pac) * sigma_pac
                    theta_sampled = theta_pac + noise
                    alpha = compute_coefficients(theta_sampled, DEVICE)
                    W_m, B_m = [], []
                    for l in range(NUM_LAYERS):
                        a_l = alpha[l, :]
                        W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                        B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                    sample_ce = 0.0
                    for k in range(NUM_TASKS):
                        logits = forward_functional(cal_feats_device[k], W_m, B_m)
                        sample_ce += torch.clamp(criterion(logits, cal_labels_device[k]), max=5.0)
                    ce_loss += sample_ce / NUM_TASKS
                ce_loss /= S
                reg_loss = 0.0
                for k in range(NUM_TASKS):
                    reg_loss += (theta_pac[k, 0] - THETA_UNIFORM) ** 2 + theta_pac[k, 1] ** 2 + theta_pac[k, 2] ** 2 + theta_pac[k, 3] ** 2
                total_loss = ce_loss + lambda_pac * reg_loss
                total_loss.backward()
                optimizer.step()
            with torch.no_grad():
                pac_accs = []
                for k in range(NUM_TASKS):
                    accumulated_probs = None
                    for _ in range(5):
                        noise = torch.randn_like(theta_pac) * sigma_pac
                        theta_sampled = theta_pac + noise
                        alpha_sampled = compute_coefficients(theta_sampled, DEVICE)
                        W_m, B_m = [], []
                        for l in range(NUM_LAYERS):
                            a_l = alpha_sampled[l, :]
                            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                        logits = forward_functional(test_feats[k].to(DEVICE), W_m, B_m)
                        probs = F.softmax(logits, dim=-1)
                        if accumulated_probs is None:
                            accumulated_probs = probs
                        else:
                            accumulated_probs += probs
                    avg_probs = accumulated_probs / 5
                    preds = avg_probs.argmax(dim=-1)
                    acc = (preds == test_labels[k].to(DEVICE)).float().mean().item()
                    pac_accs.append(acc)
                results_scarcity["pac_bayes"][M].append(np.mean(pac_accs))

            # FIM-guided scarcity sweep
            theta_init_f = torch.zeros(NUM_TASKS, 4, device=DEVICE)
            theta_init_f[:, 0] = THETA_UNIFORM
            theta_init_f.requires_grad = True
            fisher_diagonal_f = torch.zeros_like(theta_init_f)
            for k in range(NUM_TASKS):
                for i in range(cal_feats_device[k].size(0)):
                    x_i = cal_feats_device[k][i:i+1]
                    y_i = cal_labels_device[k][i:i+1]
                    alpha_f = compute_coefficients(theta_init_f, DEVICE)
                    W_m, B_m = [], []
                    for l in range(NUM_LAYERS):
                        a_l = alpha_f[l, :]
                        W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                        B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                    logits_f = forward_functional(x_i, W_m, B_m)
                    loss_f = criterion(logits_f, y_i)
                    grads_f = torch.autograd.grad(loss_f, theta_init_f, retain_graph=False)[0]
                    fisher_diagonal_f += grads_f ** 2
            fisher_diagonal_f = fisher_diagonal_f / (NUM_TASKS * cal_feats_device[0].size(0))
            fisher_diagonal_f = torch.clamp(fisher_diagonal_f, min=1e-5)
            fisher_diagonal_f = fisher_diagonal_f / fisher_diagonal_f.mean()

            theta_pac_fim_sc = torch.zeros(NUM_TASKS, 4, device=DEVICE)
            theta_pac_fim_sc[:, 0] = THETA_UNIFORM
            theta_pac_fim_sc.requires_grad = True
            optimizer_fim_sc = optim.AdamW([theta_pac_fim_sc], lr=0.01)
            for step in range(50):
                optimizer_fim_sc.zero_grad()
                ce_loss = 0.0
                for _ in range(S):
                    noise = torch.randn_like(theta_pac_fim_sc) * sigma_pac
                    theta_sampled = theta_pac_fim_sc + noise
                    alpha_s = compute_coefficients(theta_sampled, DEVICE)
                    W_m, B_m = [], []
                    for l in range(NUM_LAYERS):
                        a_l = alpha_s[l, :]
                        W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                        B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                    sample_ce = 0.0
                    for k in range(NUM_TASKS):
                        logits = forward_functional(cal_feats_device[k], W_m, B_m)
                        sample_ce += torch.clamp(criterion(logits, cal_labels_device[k]), max=5.0)
                    ce_loss += sample_ce / NUM_TASKS
                ce_loss /= S
                reg_loss = 0.0
                for k in range(NUM_TASKS):
                    reg_loss += fisher_diagonal_f[k, 0] * (theta_pac_fim_sc[k, 0] - THETA_UNIFORM) ** 2
                    reg_loss += fisher_diagonal_f[k, 1] * theta_pac_fim_sc[k, 1] ** 2
                    reg_loss += fisher_diagonal_f[k, 2] * theta_pac_fim_sc[k, 2] ** 2
                    reg_loss += fisher_diagonal_f[k, 3] * theta_pac_fim_sc[k, 3] ** 2
                total_loss = ce_loss + lambda_pac * reg_loss
                total_loss.backward()
                optimizer_fim_sc.step()
            with torch.no_grad():
                pac_fim_accs = []
                for k in range(NUM_TASKS):
                    accumulated_probs = None
                    for _ in range(5):
                        noise = torch.randn_like(theta_pac_fim_sc) * sigma_pac
                        theta_sampled = theta_pac_fim_sc + noise
                        alpha_sampled = compute_coefficients(theta_sampled, DEVICE)
                        W_m, B_m = [], []
                        for l in range(NUM_LAYERS):
                            a_l = alpha_sampled[l, :]
                            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                        logits = forward_functional(test_feats[k].to(DEVICE), W_m, B_m)
                        probs = F.softmax(logits, dim=-1)
                        if accumulated_probs is None:
                            accumulated_probs = probs
                        else:
                            accumulated_probs += probs
                    avg_probs = accumulated_probs / 5
                    preds = avg_probs.argmax(dim=-1)
                    acc = (preds == test_labels[k].to(DEVICE)).float().mean().item()
                    pac_fim_accs.append(acc)
                results_scarcity["pac_bayes_fim"][M].append(np.mean(pac_fim_accs))

    # Save scarcity results
    scarcity_aggregated = {}
    for method in results_scarcity:
        scarcity_aggregated[method] = {}
        for M in scarcity_values:
            scarcity_aggregated[method][M] = {
                "mean": np.mean(results_scarcity[method][M]),
                "std": np.std(results_scarcity[method][M]),
                "raw": results_scarcity[method][M]
            }
            
    with open("scarcity_results.json", "w") as f:
        json.dump(scarcity_aggregated, f, indent=2)
        
    # Plot 3: Scarcity Sweep
    plt.figure(figsize=(7, 5))
    x_axis = np.array(scarcity_values)
    
    unif_means = [scarcity_aggregated["static_uniform"][M]["mean"] * 100 for M in scarcity_values]
    unif_stds = [scarcity_aggregated["static_uniform"][M]["std"] * 100 for M in scarcity_values]
    plt.errorbar(x_axis, unif_means, yerr=unif_stds, fmt='o-', label="Static Uniform", color='#ff7f0e', linewidth=2, capsize=4)
    
    unc_means = [scarcity_aggregated["offline_unconstrained"][M]["mean"] * 100 for M in scarcity_values]
    unc_stds = [scarcity_aggregated["offline_unconstrained"][M]["std"] * 100 for M in scarcity_values]
    plt.errorbar(x_axis, unc_means, yerr=unc_stds, fmt='s-', label="Offline Unconstrained", color='#d62728', linewidth=2, capsize=4)
    
    pac_means = [scarcity_aggregated["pac_bayes"][M]["mean"] * 100 for M in scarcity_values]
    pac_stds = [scarcity_aggregated["pac_bayes"][M]["std"] * 100 for M in scarcity_values]
    plt.errorbar(x_axis, pac_means, yerr=pac_stds, fmt='^-', label="PAC-Bayes Merge (Ours)", color='#2ca02c', linewidth=2.5, capsize=4)
    
    fim_means = [scarcity_aggregated["pac_bayes_fim"][M]["mean"] * 100 for M in scarcity_values]
    fim_stds = [scarcity_aggregated["pac_bayes_fim"][M]["std"] * 100 for M in scarcity_values]
    plt.errorbar(x_axis, fim_means, yerr=fim_stds, fmt='d-', label="PAC-Bayes-FIM Merge (Ours)", color='#9467bd', linewidth=2.5, capsize=4)
    
    plt.xlabel("Few-Shot Calibration Size $M$ (samples per task)", fontsize=11)
    plt.ylabel("Multi-task Joint Mean Accuracy (%)", fontsize=11)
    plt.title("Generalization Robustness under Calibration Scarcity", fontsize=12, fontweight='bold')
    plt.xticks(scarcity_values)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig("fig3_calibration_scarcity.png", dpi=150)
    plt.close()
    
    print("\nScarcity Sweep complete and fig3_calibration_scarcity.png saved!")

if __name__ == "__main__":
    main()
