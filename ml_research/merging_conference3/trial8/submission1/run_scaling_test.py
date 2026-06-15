import torch
import numpy as np
import sys

# Reset seeds
torch.manual_seed(42)
np.random.seed(42)

# We will import run_regime but we need to modify the hypermerge ensembling inside it.
# To do this without modifying run_experiments.py permanently yet, let's copy the code and run a quick test.

L = 14  # Layers
D = 192  # Dimensions
K = 4   # Tasks
C = 10  # Classes per task
N_test_per_task = 500
N_support_per_task = 16
r = 8   # LoRA rank

def safe_norm(x, dim=-1, keepdim=True, eps=1e-8):
    return torch.norm(x, dim=dim, keepdim=keepdim) + eps

def exp_map(h, c):
    norm_h = safe_norm(h, dim=-1, keepdim=True)
    sqrt_c = (c ** 0.5)
    coeff = torch.tanh(sqrt_c * norm_h) / (sqrt_c * norm_h)
    return coeff * h

def log_map(z, c):
    norm_z = safe_norm(z, dim=-1, keepdim=True)
    sqrt_c = (c ** 0.5)
    val = sqrt_c * norm_z
    val = torch.clamp(val, 0.0, 1.0 - 1e-7)
    artanh_val = 0.5 * torch.log((1.0 + val) / (1.0 - val))
    coeff = artanh_val / val
    return coeff * z

def mobius_add(x, y, c):
    inner_xy = torch.sum(x * y, dim=-1, keepdim=True)
    norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y * y, dim=-1, keepdim=True)
    num_part1 = (1.0 + 2.0 * c * inner_xy + c * norm_y_sq) * x
    num_part2 = (1.0 - c * norm_x_sq) * y
    numerator = num_part1 + num_part2
    denominator = 1.0 + 2.0 * c * inner_xy + (c ** 2) * norm_x_sq * norm_y_sq
    return numerator / (denominator + 1e-9)

def hyperbolic_dist(x, y, c):
    minus_x = -x
    add_res = mobius_add(minus_x, y, c)
    norm_res = safe_norm(add_res, dim=-1, keepdim=True)
    val = (c ** 0.5) * norm_res
    val = torch.clamp(val, 0.0, 1.0 - 1e-7)
    artanh_val = 0.5 * torch.log((1.0 + val) / (1.0 - val))
    dist = (2.0 / (c ** 0.5)) * artanh_val
    return dist

def hyperbolic_centroid(z_support, c):
    z_poincare = exp_map(z_support, c)
    norm_sq = torch.sum(z_poincare * z_poincare, dim=-1, keepdim=True)
    z_klein = (2.0 * z_poincare) / (1.0 + c * norm_sq)
    norm_klein_sq = torch.sum(z_klein * z_klein, dim=-1, keepdim=True)
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - c * norm_klein_sq, min=1e-7))
    mu_klein = torch.sum(gamma * z_klein, dim=0, keepdim=True) / torch.sum(gamma, dim=0, keepdim=True)
    norm_mu_klein_sq = torch.sum(mu_klein * mu_klein, dim=-1, keepdim=True)
    denom = 1.0 + torch.sqrt(torch.clamp(1.0 - c * norm_mu_klein_sq, min=1e-7))
    mu_poincare = mu_klein / denom
    return mu_poincare

def compute_acc(preds, labels):
    return (preds == labels).float().mean().item()

def run_test(subspaces, c_hyperbolic=0.1, tau=0.05, scale_factor=1.0):
    torch.manual_seed(42)
    np.random.seed(42)
    
    prototypes = {}
    for k in range(K):
        start, end = subspaces[k]
        dim = end - start
        task_prototypes = []
        for c_idx in range(C):
            proto = torch.randn(dim)
            proto = proto / torch.norm(proto)
            task_prototypes.append(proto)
        prototypes[k] = torch.stack(task_prototypes)

    def generate_samples(task_idx, N, scale=2.5, noise_std=0.1):
        classes = torch.randint(0, C, (N,))
        samples = torch.zeros(N, D)
        start, end = subspaces[task_idx]
        dim = end - start
        for i in range(N):
            c_val = classes[i]
            proto = prototypes[task_idx][c_val]
            samples[i, start:end] = proto * scale + torch.randn(dim) * noise_std
        for k in range(K):
            if k != task_idx:
                s_start, s_end = subspaces[k]
                s_dim = s_end - s_start
                samples[:, s_start:s_end] += torch.randn(N, s_dim) * noise_std
        return samples, classes

    W_base = [torch.eye(D) for _ in range(L)]

    W_experts = {k: [] for k in range(K)}
    for k in range(K):
        start, end = subspaces[k]
        dim = end - start
        for l in range(L):
            W_l = torch.eye(D)
            U = torch.eye(dim) + torch.randn(dim, dim) * 0.1
            W_l[start:end, start:end] = W_l[start:end, start:end] @ U
            W_experts[k].append(W_l)

    LoRA_experts = {k: [] for k in range(K)}
    for k in range(K):
        for l in range(L):
            V = W_experts[k][l] - W_base[l]
            U, S, Vt = torch.linalg.svd(V)
            A = U[:, :r] * torch.sqrt(S[:r])
            B = torch.diag(torch.sqrt(S[:r])) @ Vt[:r, :]
            LoRA_experts[k].append((A, B))

    transformed_prototypes = {}
    for k in range(K):
        task_protos = prototypes[k]
        start, end = subspaces[k]
        dim = end - start
        h = torch.zeros(C, D)
        h[:, start:end] = task_protos * 2.5
        for l in range(L):
            A, B_mat = LoRA_experts[k][l]
            h = h @ W_base[l] + h @ A @ B_mat
        transformed_prototypes[k] = h[:, start:end]

    W_clf = {}
    for k in range(K):
        W_c = torch.zeros(D, C)
        start, end = subspaces[k]
        normed_trans = transformed_prototypes[k] / torch.norm(transformed_prototypes[k], dim=-1, keepdim=True)
        W_c[start:end, :] = normed_trans.T
        W_clf[k] = W_c

    support_samples = {}
    Eucl_centroids = {}
    HCA_centroids = {}
    for k in range(K):
        x_supp, _ = generate_samples(k, N_support_per_task)
        support_samples[k] = x_supp
        Eucl_centroids[k] = torch.mean(x_supp, dim=0)
        HCA_centroids[k] = hyperbolic_centroid(x_supp, c_hyperbolic).squeeze(0)

    test_samples_list = []
    test_labels_list = []
    test_task_ids_list = []
    for k in range(K):
        x_test, y_test = generate_samples(k, N_test_per_task)
        test_samples_list.append(x_test)
        test_labels_list.append(y_test)
        test_task_ids_list.append(torch.full((N_test_per_task,), k, dtype=torch.long))

    homo_samples = torch.cat(test_samples_list, dim=0)
    homo_labels = torch.cat(test_labels_list, dim=0)
    homo_task_ids = torch.cat(test_task_ids_list, dim=0)

    def get_euclidean_routing(z_b, centroids, tau=0.05):
        B = z_b.shape[0]
        u = torch.zeros(B, K)
        for k in range(K):
            c_k = centroids[k]
            norm_z = safe_norm(z_b, dim=-1, keepdim=True)
            norm_c = safe_norm(c_k, dim=-1, keepdim=True).T
            sim = torch.sum(z_b * c_k, dim=-1, keepdim=True) / (norm_z * norm_c + 1e-8)
            u[:, k] = sim.squeeze(-1)
        alpha = torch.softmax(u / tau, dim=-1)
        return alpha

    def get_hyperbolic_routing(z_b, centroids, c, tau=0.05):
        B = z_b.shape[0]
        u = torch.zeros(B, K)
        z_poincare = exp_map(z_b, c)
        for k in range(K):
            mu_k = centroids[k].unsqueeze(0)
            dist = hyperbolic_dist(z_poincare, mu_k, c)
            u[:, k] = -dist.squeeze(-1)
        alpha = torch.softmax(u / tau, dim=-1)
        return alpha

    # SABLE (Early Routing)
    def run_sable_early(X, centroids, tau=0.05, M=2):
        B = X.shape[0]
        alpha = get_euclidean_routing(X, centroids, tau)
        h = X.clone()
        for l in range(L):
            h_next = torch.zeros_like(h)
            for b in range(B):
                h_base_b = h[b:b+1] @ W_base[l]
                E_merged_b = torch.zeros(1, D)
                for k in range(K):
                    A, B_mat = LoRA_experts[k][l]
                    E_k_b = h[b:b+1] @ A @ B_mat
                    E_merged_b += alpha[b, k] * E_k_b
                h_next[b:b+1] = h_base_b + E_merged_b
            h = h_next
        logits = torch.zeros(B, C)
        for b in range(B):
            val, idx = torch.topk(alpha[b], M)
            alpha_b_m = alpha[b].clone()
            mask = torch.zeros(K)
            mask[idx] = 1.0
            alpha_b_m = alpha_b_m * mask
            alpha_b_m = alpha_b_m / torch.sum(alpha_b_m)
            logits_b = torch.zeros(1, C)
            for k in range(K):
                if alpha_b_m[k] > 0:
                    logits_b += alpha_b_m[k] * (h[b:b+1] @ W_clf[k])
            logits[b] = logits_b.squeeze(0)
        return torch.argmax(logits, dim=-1)

    # HyperMerge
    def run_hypermerge(X, hca_centroids, c=0.1, tau=0.05, M=2):
        B = X.shape[0]
        alpha = get_hyperbolic_routing(X, hca_centroids, c, tau)
        h = X.clone()
        for l in range(L):
            h_next = torch.zeros_like(h)
            for b in range(B):
                h_base_b = h[b:b+1] @ W_base[l]
                
                v_list = []
                for k in range(K):
                    A, B_mat = LoRA_experts[k][l]
                    E_k_b = h[b:b+1] @ A @ B_mat
                    v_k_b = exp_map(E_k_b, c)
                    v_list.append(v_k_b)
                
                w_klein_list = []
                gamma_list = []
                for k in range(K):
                    v = v_list[k]
                    norm_sq = torch.sum(v * v, dim=-1, keepdim=True)
                    w = (2.0 * v) / (1.0 + c * norm_sq)
                    w_klein_list.append(w)
                    
                    w_norm_sq = torch.sum(w * w, dim=-1, keepdim=True)
                    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - c * w_norm_sq, min=1e-7))
                    gamma_list.append(gamma)
                    
                w_merged_numerator = torch.zeros(1, D)
                w_merged_denominator = torch.zeros(1, 1)
                for k in range(K):
                    weight = alpha[b, k] * gamma_list[k]
                    w_merged_numerator += weight * w_klein_list[k]
                    w_merged_denominator += weight
                    
                w_merged_klein = w_merged_numerator / torch.clamp(w_merged_denominator, min=1e-7)
                
                norm_klein_sq = torch.sum(w_merged_klein * w_merged_klein, dim=-1, keepdim=True)
                max_norm = 1.0 / (c ** 0.5) - 1e-7
                norm_klein = torch.sqrt(norm_klein_sq)
                if norm_klein > max_norm:
                    w_merged_klein = w_merged_klein * (max_norm / norm_klein)
                    norm_klein_sq = torch.tensor([[max_norm**2]], dtype=torch.float32)
                    
                denom = 1.0 + torch.sqrt(torch.clamp(1.0 - c * norm_klein_sq, min=1e-7))
                v_merged = w_merged_klein / denom
                
                E_merged_b = log_map(v_merged, c)
                
                # Apply scale_factor to compensate for systematic norm shrinkage
                E_merged_b = E_merged_b * scale_factor
                
                h_next[b:b+1] = h_base_b + E_merged_b
            h = h_next
        logits = torch.zeros(B, C)
        for b in range(B):
            val, idx = torch.topk(alpha[b], M)
            alpha_b_m = alpha[b].clone()
            mask = torch.zeros(K)
            mask[idx] = 1.0
            alpha_b_m = alpha_b_m * mask
            alpha_b_m = alpha_b_m / torch.sum(alpha_b_m)
            logits_b = torch.zeros(1, C)
            for k in range(K):
                if alpha_b_m[k] > 0:
                    logits_b += alpha_b_m[k] * (h[b:b+1] @ W_clf[k])
            logits[b] = logits_b.squeeze(0)
        return torch.argmax(logits, dim=-1)

    acc_sable = compute_acc(run_sable_early(homo_samples, Eucl_centroids, tau), homo_labels)
    acc_hm = compute_acc(run_hypermerge(homo_samples, HCA_centroids, c_hyperbolic, tau), homo_labels)
    return acc_sable, acc_hm

subspaces_overlap = {
    0: (0, 96),
    1: (32, 128),
    2: (64, 160),
    3: (96, 192)
}

print("Starting scaling sweep on overlapping sandbox...")
sys.stdout.flush()

for sf in [0.7, 0.8, 0.9, 1.0]:
    for c in [0.05, 0.1, 0.2]:
        acc_sable, acc_hm = run_test(subspaces_overlap, c_hyperbolic=c, scale_factor=sf)
        print(f"sf={sf:.1f}, c={c:.2f} -> SABLE: {acc_sable*100:.2f}%, HyperMerge: {acc_hm*100:.2f}%")
        sys.stdout.flush()
