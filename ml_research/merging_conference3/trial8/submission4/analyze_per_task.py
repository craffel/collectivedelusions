import torch
import torch.nn as nn
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

D = 192
K = 4
L = 14
num_classes = 10
block_size = D // K  # 48
r = 8  # rank of LoRA
N_calib_per_task = 16
N_test_per_task = 250
B = 16  # Batch size
sigma_0_sq = 1.0

noise_levels = [0.01, 0.05, 0.28, 1.35]

class PACRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(K))
    def forward(self, x):
        tau = torch.exp(self.log_tau)
        return x / tau

# 1. Define task dimensions
task_dims = {}
for k in range(K):
    task_dims[k] = list(range(k*block_size, (k+1)*block_size))
    
# 2. Generate class prototypes
class_prototypes = {}
for k in range(K):
    subspace_size = len(task_dims[k])
    U, S, V = torch.svd(torch.randn(subspace_size, num_classes))
    prototypes = torch.zeros(num_classes, D)
    for idx, d_idx in enumerate(task_dims[k]):
        prototypes[:, d_idx] = U.t()[:num_classes, idx]
    class_prototypes[k] = prototypes

# 3. Classification heads
W_head = {}
for k in range(K):
    head = torch.zeros(D, num_classes)
    for d_idx in task_dims[k]:
        head[d_idx, :] = class_prototypes[k][:, d_idx].t()
    W_head[k] = head

# 4. Shared base layers
W_base = {l: 0.05 * torch.eye(D) for l in range(1, 14)}

# 5. Task expert adapters
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

# Helper forward pass
def run_expert_forward(x, k):
    h = x.clone()
    for l in range(1, 4):
        h = h + torch.relu(h @ W_base[l])
    for l in range(4, 14):
        delta_W = A_expert[k][l] @ B_expert[k][l]
        h = h + torch.relu(h @ W_base[l] + h @ delta_W)
    logits = h @ W_head[k]
    return logits

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

# 6. Generate datasets
calib_x, calib_y, calib_class_y = [], [], []
for k in range(K):
    for i in range(N_calib_per_task):
        c = np.random.randint(0, num_classes)
        x = class_prototypes[k][c] + noise_levels[k] * torch.randn(D)
        calib_x.append(x)
        calib_y.append(k)
        calib_class_y.append(c)
calib_x = torch.stack(calib_x)
calib_y = torch.tensor(calib_y)
calib_class_y = torch.tensor(calib_class_y)

test_x, test_y, test_class_y = [], [], []
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
h_calib = calib_x.clone()
for l in range(1, 4):
    h_calib = h_calib + torch.relu(h_calib @ W_base[l])
z_calib = h_calib.clone()

# SVD for PCA-SEP
V_pca = {}
for k in range(K):
    mask = (calib_y == k)
    z_k = z_calib[mask]
    U_k, S_k, V_k = torch.svd(z_k)
    V_pca[k] = V_k[:, :10]

# Norms
N_calib = N_calib_per_task * K
calib_block_norms = torch.zeros(z_calib.shape[0], K)
calib_pca_norms = torch.zeros(z_calib.shape[0], K)
for b in range(K):
    calib_block_norms[:, b] = z_calib[:, b*block_size : (b+1)*block_size].norm(dim=1)
    calib_pca_norms[:, b] = (z_calib @ V_pca[b]).norm(dim=1)

# Train PAC-ZCA (Block)
pac_router_block = PACRouter()
pac_opt_block = torch.optim.Adam(pac_router_block.parameters(), lr=0.05)
for epoch in range(100):
    pac_opt_block.zero_grad()
    logits = pac_router_block(calib_block_norms)
    q = torch.softmax(logits, dim=1)
    risk = 1.0 - q[range(N_calib), calib_y].mean()
    kl = (pac_router_block.log_tau ** 2).sum() / (2.0 * sigma_0_sq)
    bound = risk + torch.sqrt((kl + np.log(2.0 * np.sqrt(N_calib) / 0.05)) / (2.0 * N_calib))
    bound.backward()
    pac_opt_block.step()

# Train PAC-ZCA (PCA)
pac_router_pca = PACRouter()
pac_opt_pca = torch.optim.Adam(pac_router_pca.parameters(), lr=0.05)
for epoch in range(100):
    pac_opt_pca.zero_grad()
    logits = pac_router_pca(calib_pca_norms)
    q = torch.softmax(logits, dim=1)
    risk = 1.0 - q[range(N_calib), calib_y].mean()
    kl = (pac_router_pca.log_tau ** 2).sum() / (2.0 * sigma_0_sq)
    bound = risk + torch.sqrt((kl + np.log(2.0 * np.sqrt(N_calib) / 0.05)) / (2.0 * N_calib))
    bound.backward()
    pac_opt_pca.step()

print("Learned log_tau for PAC-ZCA (Block):", pac_router_block.log_tau.detach().numpy())
print("Learned log_tau for PAC-ZCA (PCA):  ", pac_router_pca.log_tau.detach().numpy())

# Evaluation
methods = ["sable_sep", "pac_zca_block", "sable_pca", "pac_zca_pca"]

def get_coefs(z, method):
    batch_size = z.shape[0]
    if method == "sable_sep":
        z_block_norms = torch.zeros(batch_size, K)
        for b in range(K):
            z_block_norms[:, b] = z[:, b*block_size : (b+1)*block_size].norm(dim=1)
        return torch.softmax(z_block_norms / 0.05, dim=1)
    elif method == "pac_zca_block":
        z_block_norms = torch.zeros(batch_size, K)
        for b in range(K):
            z_block_norms[:, b] = z[:, b*block_size : (b+1)*block_size].norm(dim=1)
        return torch.softmax(pac_router_block(z_block_norms), dim=1)
    elif method == "sable_pca":
        z_pca_norms = torch.zeros(batch_size, K)
        for k in range(K):
            z_pca_norms[:, k] = (z @ V_pca[k]).norm(dim=1)
        return torch.softmax(z_pca_norms / 0.05, dim=1)
    elif method == "pac_zca_pca":
        z_pca_norms = torch.zeros(batch_size, K)
        for k in range(K):
            z_pca_norms[:, k] = (z @ V_pca[k]).norm(dim=1)
        return torch.softmax(pac_router_pca(z_pca_norms), dim=1)

N_total = N_test_per_task * K
test_indices = torch.arange(N_total)
stream_x = test_x[test_indices]
stream_y = test_y[test_indices]
stream_class_y = test_class_y[test_indices]

for m in methods:
    correct_per_task = [0] * K
    routed_per_task = [0] * K  # how many predicted task == true task
    total_per_task = [N_test_per_task] * K
    
    for b_start in range(0, N_total, B):
        x_b = stream_x[b_start : b_start+B]
        y_b = stream_y[b_start : b_start+B]
        class_y_b = stream_class_y[b_start : b_start+B]
        
        h = x_b.clone()
        for l in range(1, 4):
            h = h + torch.relu(h @ W_base[l])
            
        coefs = get_coefs(h, m)
        logits, pred_task = run_blended_forward(x_b, coefs)
        
        for b_idx in range(x_b.shape[0]):
            true_task = y_b[b_idx].item()
            pt = pred_task[b_idx].item()
            pred_class = torch.argmax(logits[b_idx]).item()
            
            if pt == true_task:
                routed_per_task[true_task] += 1
                if pred_class == class_y_b[b_idx].item():
                    correct_per_task[true_task] += 1
                    
    print(f"\nMethod: {m}")
    for k in range(K):
        route_acc = routed_per_task[k] / N_test_per_task * 100.0
        class_acc = correct_per_task[k] / N_test_per_task * 100.0
        print(f"  Task {k}: Route Acc = {route_acc:.2f}%, Joint Acc = {class_acc:.2f}%")
