import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from test_high_accuracy import generate_sandbox_data_high_accuracy, LORA_SCALE, D, K, C, L, R

def evaluate_model_debug(seed):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data_high_accuracy(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    subspace_dim = 48
    
    # Initialize LoRA
    A = torch.randn(K, L, D, R) * LORA_SCALE
    B = torch.randn(K, L, R, D) * LORA_SCALE
    
    for k in range(K):
        sub_start = k * subspace_dim
        sub_end = (k + 1) * subspace_dim
        for l in range(L):
            A[k, l, :sub_start, :] *= 0.15
            A[k, l, sub_end:, :] *= 0.15
            B[k, l, :, :sub_start] *= 0.15
            B[k, l, :, sub_end:] *= 0.15
            
    expert_heads_W = torch.zeros(K, C, D)
    expert_heads_B = torch.zeros(K, C)
    
    for k in range(K):
        expert_heads_W[k, :, k*subspace_dim : (k+1)*subspace_dim] = prototypes[k]
        
    for k in range(K):
        task_mask = (Y_train_task == k)
        X_task = X_train[task_mask]
        Y_class_task = Y_train_class[task_mask]
        
        z = X_task.clone()
        for l in range(L):
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

    pear_dispersion = torch.zeros(K)
    for k in range(K):
        task_mask = (Y_cal_task == k)
        X_task_cal = X_cal[task_mask]
        Y_class_cal = Y_cal_class[task_mask]
        sims = []
        for s_idx, s in enumerate(X_task_cal):
            c_label = Y_class_cal[s_idx].item()
            w_c = expert_heads_W[k, c_label]
            cos_sim = torch.dot(s, w_c) / (s.norm() * w_c.norm() + 1e-8)
            sims.append(cos_sim.item())
        pear_dispersion[k] = np.mean(sims)

    def propagate_layers(x_batch, l_start, l_end, coeffs_batch):
        z = x_batch.clone()
        for l in range(l_start, l_end):
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = z + delta_sum
        return z

    def compute_max_cos_sims(xb):
        actual_B = len(xb)
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

    perm = torch.randperm(len(X_test))
    X_hetero = X_test[perm]
    Y_task_hetero = Y_test_task[perm]
    Y_class_hetero = Y_test_class[perm]
    
    batch_size = 256
    for i in range(0, len(X_hetero), batch_size):
        xb = X_hetero[i : i + batch_size]
        yb_task = Y_task_hetero[i : i + batch_size]
        yb_class = Y_class_hetero[i : i + batch_size]
        actual_B = len(xb)
        
        with torch.no_grad():
            cos_sims = compute_max_cos_sims(xb)
            calibrated_sims = cos_sims / (pear_dispersion.unsqueeze(0) + 1e-8)
            pear_route_coeffs = torch.softmax(calibrated_sims / 0.001, dim=-1)
            
        z = propagate_layers(xb, 0, L, pear_route_coeffs)
        logits = torch.zeros(actual_B, C)
        for j in range(K):
            logits += pear_route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
        preds = logits.argmax(dim=-1)
        
        # Print routing coeffs for some Task 1 samples inside this mixed batch
        task1_mask = (yb_task == 1)
        if task1_mask.sum() > 0:
            print(f"Hetero batch {i//batch_size} F-MNIST sample routing coeffs:")
            indices = torch.where(task1_mask)[0][:3]
            for idx in indices:
                print(f"  Sample {idx}: Coeffs: {['%.4f' % c.item() for c in pear_route_coeffs[idx]]}, Pred: {preds[idx].item()}, True: {yb_class[idx].item()}")

evaluate_model_debug(13)
