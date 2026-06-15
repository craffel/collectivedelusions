import torch
import torch.nn as nn
import numpy as np

# Experimental Parameters
D = 192
K = 4
num_classes = 10
block_size = D // K  # 48
r = 8  # rank of LoRA
N_calib_per_task = 16
N_test_per_task = 250
B = 16  # Batch size
noise_levels = [0.01, 0.05, 0.28, 1.35]

class PACRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(K))
    def forward(self, x):
        tau = torch.exp(self.log_tau)
        return x / tau

def run_single_experiment_sensitivity(seed, sigma_0_sq):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define task dimensions
    task_dims = {}
    for k in range(K):
        task_dims[k] = list(range(k*block_size, (k+1)*block_size))
        
    # Generate class prototypes
    class_prototypes = {}
    for k in range(K):
        subspace_size = len(task_dims[k])
        U, S, V = torch.svd(torch.randn(subspace_size, num_classes))
        prototypes = torch.zeros(num_classes, D)
        for idx, d_idx in enumerate(task_dims[k]):
            prototypes[:, d_idx] = U.t()[:num_classes, idx]
        class_prototypes[k] = prototypes

    # Classification heads (Layer 14)
    W_head = {}
    for k in range(K):
        head = torch.zeros(D, num_classes)
        for d_idx in task_dims[k]:
            head[d_idx, :] = class_prototypes[k][:, d_idx].t()
        W_head[k] = head

    # Shared base layers (Layers 1-13)
    W_base = {}
    for l in range(1, 14):
        W_base[l] = 0.05 * torch.eye(D)

    # Shared base layers and expert adapters (Layers 4-13)
    A_expert = {}
    B_expert = {}
    for k in range(K):
        A_expert[k] = {}
        B_expert[k] = {}
        P_k = torch.zeros(D, D)
        for d_idx in task_dims[k]:
            P_k[d_idx, d_idx] = 1.0
        
        for l in range(4, 14):
            target = 0.15 * P_k + 0.01 * torch.randn(D, D)
            U, S, V = torch.svd(target)
            A_expert[k][l] = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
            B_expert[k][l] = torch.diag(torch.sqrt(S[:r])) @ V[:, :r].t()

    def run_blended_forward(x_batch, coefs):
        B_size = x_batch.shape[0]
        h = x_batch.clone()
        for l in range(1, 4):
            h = h + torch.relu(h @ W_base[l])
        for l in range(4, 14):
            base_out = h @ W_base[l]
            expert_blend = torch.zeros_like(base_out)
            for k in range(K):
                expert_out = h @ (A_expert[k][l] @ B_expert[k][l])
                expert_blend = expert_blend + coefs[:, k:k+1] * expert_out
            h = h + torch.relu(base_out + expert_blend)
                
        pred_task = torch.argmax(coefs, dim=1)
        logits = torch.zeros(B_size, num_classes)
        for b in range(B_size):
            tk = pred_task[b].item()
            logits[b] = h[b] @ W_head[tk]
        return logits, pred_task

    # Generate calibration and test sets (Split into Subspace and Optimization)
    sub_calib_x = []
    sub_calib_y = []
    opt_calib_x = []
    opt_calib_y = []
    
    for k in range(K):
        for i in range(N_calib_per_task):
            c = np.random.randint(0, num_classes)
            x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
            if i < 8:
                sub_calib_x.append(x)
                sub_calib_y.append(k)
            else:
                opt_calib_x.append(x)
                opt_calib_y.append(k)
                
    sub_calib_x = torch.stack(sub_calib_x)
    sub_calib_y = torch.tensor(sub_calib_y)
    opt_calib_x = torch.stack(opt_calib_x)
    opt_calib_y = torch.tensor(opt_calib_y)

    test_x = []
    test_y = []
    test_class_y = []
    for k in range(K):
        for i in range(N_test_per_task):
            c = np.random.randint(0, num_classes)
            x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
            test_x.append(x)
            test_y.append(k)
            test_class_y.append(c)
    test_x = torch.stack(test_x)
    test_y = torch.tensor(test_y)
    test_class_y = torch.tensor(test_class_y)

    # Extract Layer 3 features
    h_opt = opt_calib_x.clone()
    for l in range(1, 4):
        h_opt = h_opt + torch.relu(h_opt @ W_base[l])
    z_opt = h_opt.clone()

    N_opt = 8 * K
    opt_block_norms = torch.zeros(z_opt.shape[0], K)
    for b in range(K):
        opt_block_norms[:, b] = z_opt[:, b*block_size : (b+1)*block_size].norm(dim=1)

    # Train PAC-ZCA (Block)
    pac_router_block = PACRouter()
    pac_opt_block = torch.optim.Adam(pac_router_block.parameters(), lr=0.05)
    criterion_pac = nn.CrossEntropyLoss()
    w_0 = np.log(0.05)
    for epoch in range(100):
        pac_opt_block.zero_grad()
        logits = pac_router_block(opt_block_norms)
        risk = criterion_pac(logits, opt_calib_y)
        kl = ((pac_router_block.log_tau - w_0) ** 2).sum() / (2.0 * sigma_0_sq)
        bound = risk + torch.sqrt((kl + np.log(2.0 * np.sqrt(N_opt) / 0.05)) / (2.0 * N_opt))
        bound.backward()
        pac_opt_block.step()

    # Get final log-temperatures
    final_log_tau = pac_router_block.log_tau.detach().clone()

    # Evaluate on Homogeneous Stream
    N_total = N_test_per_task * K
    stream_x = test_x.clone()
    stream_y = test_y.clone()
    stream_class_y = test_class_y.clone()
    
    correct = 0
    for b_start in range(0, N_total, B):
        x_b = stream_x[b_start : b_start+B]
        y_b = stream_y[b_start : b_start+B]
        class_y_b = stream_class_y[b_start : b_start+B]
        
        h_b = x_b.clone()
        for l in range(1, 4):
            h_b = h_b + torch.relu(h_b @ W_base[l])
            
        z_block_norms = torch.zeros(x_b.shape[0], K)
        for b in range(K):
            z_block_norms[:, b] = h_b[:, b*block_size : (b+1)*block_size].norm(dim=1)
        
        logits_router = pac_router_block(z_block_norms)
        coefs = torch.softmax(logits_router, dim=1)
        
        logits, pred_task = run_blended_forward(x_b, coefs)
        
        for b_idx in range(x_b.shape[0]):
            pred_class = torch.argmax(logits[b_idx]).item()
            if pred_task[b_idx].item() == y_b[b_idx].item() and pred_class == class_y_b[b_idx].item():
                correct += 1
                
    accuracy = correct / N_total * 100.0
    return accuracy, final_log_tau

seeds = [42, 43, 44, 45, 46]
prior_variances = [0.1, 0.5, 1.0, 5.0, 10.0]

print("Starting Standalone Prior Variance Sensitivity Analysis...")
print("=" * 70)
for p in prior_variances:
    accs = []
    log_taus_list = []
    for seed in seeds:
        acc, log_taus = run_single_experiment_sensitivity(seed, p)
        accs.append(acc)
        log_taus_list.append(log_taus.numpy())
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_log_taus = np.mean(log_taus_list, axis=0)
    mean_taus = np.exp(mean_log_taus)
    
    formatted_taus = ", ".join([f"{t:.3f}" for t in mean_taus])
    print(f"sigma_0^2 = {p:<4} | Accuracy = {mean_acc:.2f}% ± {std_acc:.2f}% | Avg Temperatures: [{formatted_taus}]")
print("=" * 70)
