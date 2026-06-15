import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from run_experiments import generate_sandbox_data, get_subspace_range, L, D, K, C, R, LORA_SCALE, OOD_THRESHOLD

L_LORA = LORA_SCALE # standard import or define locally to be safe

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate_early_layer_compromise(seed, layer_route):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    # Initialize base model layer transformations (representation shifts)
    W_base = torch.zeros(L, D, D)
    for l in range(L):
        W_base[l] = torch.eye(D) + torch.randn(D, D) * 0.015

    # Initialize LoRA adapters
    A = torch.randn(K, L, D, R) * L_LORA
    B = torch.randn(K, L, R, D) * L_LORA
    
    # Task localization in LoRA
    for k in range(K):
        start, end = get_subspace_range(k)
        for l in range(L):
            A[k, l, :start, :] *= 0.15
            A[k, l, end:, :] *= 0.15
            B[k, l, :, :start] *= 0.15
            B[k, l, :, end:] *= 0.15
            
    # Expert classification heads
    expert_heads_W = torch.randn(K, C, D) * 0.05
    expert_heads_B = torch.zeros(K, C)
    
    for k in range(K):
        start, end = get_subspace_range(k)
        expert_heads_W[k, :, start:end] += prototypes[k]
        
    for k in range(K):
        task_mask = (Y_train_task == k)
        X_task = X_train[task_mask]
        Y_class_task = Y_train_class[task_mask]
        
        z = X_task.clone()
        for l in range(L):
            z = z @ W_base[l]
            delta = z @ B[k, l].t() @ A[k, l].t()
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

    # Early-Layer Routing Compromise Calibration:
    # Propagate the calibration set through the unadapted base model up to layer_route
    X_cal_route = X_cal.clone()
    for l in range(layer_route):
        X_cal_route = X_cal_route @ W_base[l]

    # Compute centroids in the layer_route space
    mu_class = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_class[k, c] = X_cal_route[class_mask].mean(dim=0)
            else:
                mu_class[k, c] = X_cal_route[Y_cal_task == k].mean(dim=0)
        
    # PEAR Dispersion calibration in the layer_route space
    pear_dispersion = torch.zeros(K)
    for k in range(K):
        task_mask = (Y_cal_task == k)
        X_task_cal_route = X_cal_route[task_mask]
        Y_class_cal = Y_cal_class[task_mask]
        sims = []
        for s_idx, s in enumerate(X_task_cal_route):
            c_label = Y_class_cal[s_idx].item()
            w_c = mu_class[k, c_label]
            cos_sim = torch.dot(s, w_c) / (s.norm() * w_c.norm() + 1e-8)
            sims.append(cos_sim.item())
        pear_dispersion[k] = np.mean(sims)

    def propagate_layers_compromise(xb, coeffs_batch):
        # Propagate through unadapted base layers before the routing point
        z = xb.clone()
        for l in range(layer_route):
            z = z @ W_base[l]
        # Propagate with blended expert adapters after the routing point
        for l in range(layer_route, L):
            base_trans = z @ W_base[l]
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

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
            # Propagate queries to the routing layer of base model to extract routing features
            xb_route = xb.clone()
            for l in range(layer_route):
                xb_route = xb_route @ W_base[l]
                
            cos_sims = torch.zeros(actual_B, K)
            for j in range(K):
                mu_j = mu_class[j]
                mu_j_norms = mu_j.norm(dim=-1, keepdim=True) + 1e-8
                for b_idx in range(actual_B):
                    x_b = xb_route[b_idx]
                    dot_prods = mu_j @ x_b
                    cos_class = dot_prods / (mu_j_norms.squeeze() * x_b.norm() + 1e-8)
                    cos_sims[b_idx, j] = cos_class.max()
                    
            calibrated_sims = cos_sims / (pear_dispersion.unsqueeze(0) + 1e-8)
            route_coeffs = torch.softmax(calibrated_sims / 0.05, dim=-1)
            
            # OOD Fallback
            max_sims, _ = cos_sims.max(dim=-1)
            for b_idx in range(actual_B):
                if max_sims[b_idx] < OOD_THRESHOLD:
                    route_coeffs[b_idx] = 1.0 / K
                    
        z = propagate_layers_compromise(xb, route_coeffs)
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
    seeds = [10, 11, 12, 13, 14]
    routing_layers = [0, 1, 2, 4, 6, 8, 10]
    
    print("================== BENCHMARKING EARLY-LAYER ROUTING COMPROMISE ==================")
    for lr in routing_layers:
        accs = [evaluate_early_layer_compromise(s, lr) for s in seeds]
        print(f"Routing at Layer {lr:<2} (PEAR-Compromise): Joint Mean Accuracy = {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
