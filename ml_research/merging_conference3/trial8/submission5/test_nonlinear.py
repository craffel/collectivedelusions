import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from run_experiments import generate_sandbox_data, get_subspace_range, LORA_SCALE, D, K, C, L, R

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate_model(seed, use_nonlinear=False, use_zpc=False):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    # Initialize base model transformations (random but frozen orthogonal-like transformations at each layer)
    W_base = torch.zeros(L, D, D)
    for l in range(L):
        W_base[l] = torch.eye(D) + torch.randn(D, D) * 0.015

    # Initialize LoRA
    A = torch.randn(K, L, D, R) * LORA_SCALE
    B = torch.randn(K, L, R, D) * LORA_SCALE
    
    for k in range(K):
        sub_start, sub_end = get_subspace_range(k)
        for l in range(L):
            A[k, l, :sub_start, :] *= 0.15
            A[k, l, sub_end:, :] *= 0.15
            B[k, l, :, :sub_start] *= 0.15
            B[k, l, :, sub_end:] *= 0.15
            
    expert_heads_W = torch.randn(K, C, D) * 0.05
    expert_heads_B = torch.zeros(K, C)
    
    for k in range(K):
        sub_start, sub_end = get_subspace_range(k)
        expert_heads_W[k, :, sub_start:sub_end] += prototypes[k]
        
    for k in range(K):
        task_mask = (Y_train_task == k)
        X_task = X_train[task_mask]
        Y_class_task = Y_train_class[task_mask]
        
        z = X_task.clone()
        for l in range(L):
            z = z @ W_base[l]
            delta = z @ B[k, l].t() @ A[k, l].t()
            if use_nonlinear:
                delta = torch.nn.functional.gelu(delta)
            z = z + delta
            
        head_W = nn.Parameter(expert_heads_W[k].clone())
        head_B = nn.Parameter(expert_heads_B[k].clone())
        opt_head = optim.AdamW([head_W, head_B], lr=1e-1)
        
        for _ in range(50):
            opt_head.zero_grad()
            logits = z @ head_W.t() + head_B
            loss = nn.CrossEntropyLoss()(logits, Y_class_task)
            loss.backward()
            opt_head.step()
            
        expert_heads_W[k] = head_W.detach()
        expert_heads_B[k] = head_B.detach()

    # Compute task centroids mu_k if using ZPC
    if use_zpc:
        mu = torch.zeros(K, D)
        for k in range(K):
            task_mask = (Y_cal_task == k)
            mu[k] = X_cal[task_mask].mean(dim=0)
            
        # Compute dispersion using ZPC centroids
        pear_dispersion = torch.zeros(K)
        for k in range(K):
            task_mask = (Y_cal_task == k)
            X_task_cal = X_cal[task_mask]
            sims = []
            for s in X_task_cal:
                cos_sim = torch.dot(s, mu[k]) / (s.norm() * mu[k].norm() + 1e-8)
                sims.append(cos_sim.item())
            pear_dispersion[k] = np.mean(sims)
    else:
        # Compute dispersion using EHA classification heads (mu_class)
        mu_class = torch.zeros(K, C, D)
        for k in range(K):
            for c in range(C):
                class_mask = (Y_cal_task == k) & (Y_cal_class == c)
                if class_mask.sum() > 0:
                    mu_class[k, c] = X_cal[class_mask].mean(dim=0)
                else:
                    mu_class[k, c] = X_cal[Y_cal_task == k].mean(dim=0)

        pear_dispersion = torch.zeros(K)
        for k in range(K):
            task_mask = (Y_cal_task == k)
            X_task_cal = X_cal[task_mask]
            Y_class_cal = Y_cal_class[task_mask]
            sims = []
            for s_idx, s in enumerate(X_task_cal):
                c_label = Y_class_cal[s_idx].item()
                w_c = mu_class[k, c_label]
                cos_sim = torch.dot(s, w_c) / (s.norm() * w_c.norm() + 1e-8)
                sims.append(cos_sim.item())
            pear_dispersion[k] = np.mean(sims)

    def propagate_layers(x_batch, l_start, l_end, coeffs_batch):
        z = x_batch.clone()
        for l in range(l_start, l_end):
            z = z @ W_base[l]
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                if use_nonlinear:
                    delta_k = torch.nn.functional.gelu(delta_k)
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = z + delta_sum
        return z

    def compute_routing_coefficients(xb):
        actual_B = len(xb)
        if use_zpc:
            cos_sims = torch.zeros(actual_B, K)
            for j in range(K):
                mu_j = mu[j]
                mu_j_norm = mu_j.norm() + 1e-8
                for b_idx in range(actual_B):
                    x_b = xb[b_idx]
                    x_b_norm = x_b.norm() + 1e-8
                    cos_sims[b_idx, j] = torch.dot(x_b, mu_j) / (x_b_norm * mu_j_norm)
        else:
            cos_sims = torch.zeros(actual_B, K)
            for j in range(K):
                W_j = expert_heads_W[j]
                W_j_norms = W_j.norm(dim=-1, keepdim=True) + 1e-8
                for b_idx in range(actual_B):
                    x_b = xb[b_idx]
                    x_b_norm = x_b.norm() + 1e-8
                    dot_prods = W_j @ x_b
                    cos_class = dot_prods / (W_j_norms.squeeze() * x_b_norm)
                    cos_sims[b_idx, j] = cos_class.max()
        return cos_sims

    # Evaluate on test set
    perm = torch.randperm(len(X_test))
    X_hetero = X_test[perm]
    Y_task_hetero = Y_test_task[perm]
    Y_class_hetero = Y_test_class[perm]
    
    correct_preds = [0]*K
    total_preds = [0]*K
    
    batch_size = 256
    for i in range(0, len(X_hetero), batch_size):
        xb = X_hetero[i : i + batch_size]
        yb_task = Y_task_hetero[i : i + batch_size]
        yb_class = Y_class_hetero[i : i + batch_size]
        actual_B = len(xb)
        
        with torch.no_grad():
            cos_sims = compute_routing_coefficients(xb)
            calibrated_sims = cos_sims / (pear_dispersion.unsqueeze(0) + 1e-8)
            route_coeffs = torch.softmax(calibrated_sims / 0.001, dim=-1)
            
            # OOD Rejection (disabled or active)
            max_sims, _ = cos_sims.max(dim=-1)
            for b_idx in range(actual_B):
                if max_sims[b_idx] < 0.15:
                    route_coeffs[b_idx] = 1.0 / K
                    
        z = propagate_layers(xb, 0, L, route_coeffs)
        logits = torch.zeros(actual_B, C)
        for j in range(K):
            logits += route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
        preds = logits.argmax(dim=-1)
        
        for b_idx in range(actual_B):
            k = yb_task[b_idx].item()
            correct_preds[k] += (preds[b_idx] == yb_class[b_idx]).item()
            total_preds[k] += 1
            
    accs = [correct_preds[k]/total_preds[k]*100 for k in range(K)]
    return np.mean(accs)

if __name__ == "__main__":
    print("Linear Propagation (EHA):", np.mean([evaluate_model(s, use_nonlinear=False, use_zpc=False) for s in [10, 11, 12, 13, 14]]))
    print("Linear Propagation (ZPC):", np.mean([evaluate_model(s, use_nonlinear=False, use_zpc=True) for s in [10, 11, 12, 13, 14]]))
    print("Non-Linear Propagation (EHA):", np.mean([evaluate_model(s, use_nonlinear=True, use_zpc=False) for s in [10, 11, 12, 13, 14]]))
    print("Non-Linear Propagation (ZPC):", np.mean([evaluate_model(s, use_nonlinear=True, use_zpc=True) for s in [10, 11, 12, 13, 14]]))
