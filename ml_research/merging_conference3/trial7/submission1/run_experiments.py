import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Dataset loading (28x28 grayscale)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading MNIST dataset...")
full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Helper to filter subsets
def get_task_subset(dataset, classes, size, seed_offset=0):
    targets = dataset.targets
    indices = torch.zeros_like(targets, dtype=torch.bool)
    for c in classes:
        indices |= (targets == c)
    subset_indices = torch.where(indices)[0]
    # Keep only the requested size with an offset based on seed to ensure diversity across seeds
    subset_indices = subset_indices[seed_offset : seed_offset + size]
    return torch.utils.data.Subset(dataset, subset_indices)

# Define the 4-layer CNN (representing TinyCNN-4 style routing, L=4)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 3 * 3, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the 12-layer MLP (representing DeepMLP-12 style routing, L=12)
class DeepMLP12(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(784, 64))
        for _ in range(10):
            self.layers.append(nn.Linear(64, 64))
        self.layers.append(nn.Linear(64, 10))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

# Define the dynamic routing module (d=8)
class DynamicRouter(nn.Module):
    def __init__(self, L, d, K, routing_type="layer-wise", activation="bsigmoid"):
        super().__init__()
        self.L = L
        self.d = d
        self.K = K
        self.routing_type = routing_type
        self.activation = activation
        
        if routing_type == "layer-wise":
            self.W = nn.Parameter(torch.randn(L, d, K, device=device) * 0.01)
            self.B = nn.Parameter(torch.zeros(L, K, device=device))
        elif routing_type == "global":
            self.W = nn.Parameter(torch.randn(1, d, K, device=device) * 0.01)
            self.B = nn.Parameter(torch.zeros(1, K, device=device))
            
    def forward(self, psi):
        if self.routing_type == "layer-wise":
            logits = torch.einsum("bd,ldk->lbk", psi, self.W) + self.B.unsqueeze(1)
        elif self.routing_type == "global":
            logits = torch.einsum("bd,gdk->gbk", psi, self.W) + self.B.unsqueeze(1)
            logits = logits.repeat(self.L, 1, 1)
            
        if self.activation == "bsigmoid":
            # Raw element-wise independent logits
            alpha_raw = torch.sigmoid(logits)
            # Normalize weights to sum to 1.0 across experts at each layer to prevent signal decay!
            alpha_renorm = alpha_raw / (alpha_raw.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.activation == "softmax":
            alpha_renorm = torch.softmax(logits, dim=-1)
        return alpha_renorm

# Frozen random projection matrix from flattened image to low-dimensional routing state (d=8)
P_proj = torch.randn(8, 784, device=device)

# Functional forward pass using batch-averaged merged parameters for high efficiency
def forward_merged_cnn(x, alpha, experts_dict, K):
    if alpha.dim() == 3:
        alpha_avg = alpha.mean(dim=1) # shape (L, K)
    else:
        alpha_avg = alpha
        
    # Merge weights once for the entire batch
    w1 = sum(alpha_avg[0, j] * experts_dict[j].conv1.weight for j in range(K))
    b1 = sum(alpha_avg[0, j] * experts_dict[j].conv1.bias for j in range(K))
    w2 = sum(alpha_avg[1, j] * experts_dict[j].conv2.weight for j in range(K))
    b2 = sum(alpha_avg[1, j] * experts_dict[j].conv2.bias for j in range(K))
    w3 = sum(alpha_avg[2, j] * experts_dict[j].conv3.weight for j in range(K))
    b3 = sum(alpha_avg[2, j] * experts_dict[j].conv3.bias for j in range(K))
    w4 = sum(alpha_avg[3, j] * experts_dict[j].fc.weight for j in range(K))
    b4 = sum(alpha_avg[3, j] * experts_dict[j].fc.bias for j in range(K))
    
    # Run standard batched forward pass
    h = F.conv2d(x, w1, b1, padding=1)
    h = F.relu(h)
    h = F.max_pool2d(h, 2)
    
    h = F.conv2d(h, w2, b2, padding=1)
    h = F.relu(h)
    h = F.max_pool2d(h, 2)
    
    h = F.conv2d(h, w3, b3, padding=1)
    h = F.relu(h)
    h = F.max_pool2d(h, 2)
    
    h = h.view(h.size(0), -1)
    out = F.linear(h, w4, b4)
    return out

def forward_merged_mlp(x, alpha, experts_dict, K):
    if alpha.dim() == 3:
        alpha_avg = alpha.mean(dim=1) # shape (L, K)
    else:
        alpha_avg = alpha
        
    h = x.view(x.size(0), -1)
    for l in range(12):
        w = sum(alpha_avg[l, j] * experts_dict[j].layers[l].weight for j in range(K))
        b = sum(alpha_avg[l, j] * experts_dict[j].layers[l].bias for j in range(K))
        h = F.linear(h, w, b)
        if l < 11:
            h = F.relu(h)
    return h

# Disjoint digit tasks
TASKS = {
    0: [0, 1], # Experts for MNIST digits 0,1
    1: [2, 3], # Experts for MNIST digits 2,3
    2: [4, 5], # Experts for MNIST digits 4,5
    3: [6, 7]  # Experts for MNIST digits 6,7
}

SUITES = {
    "Low-Conflict": {
        "task_keys": [0, 1]  # Digits 0,1 vs 2,3 (K=2)
    },
    "High-Conflict": {
        "task_keys": [2, 3]  # Digits 4,5 vs 6,7 (K=2)
    },
    "Cross-Domain": {
        "task_keys": [0, 1, 2, 3]  # Full suite of digits 0-7 (K=4)
    }
}

MODELS = {
    "TinyCNN-4": {"L": 4, "class": TinyCNN, "forward": forward_merged_cnn},
    "DeepMLP-12": {"L": 12, "class": DeepMLP12, "forward": forward_merged_mlp}
}

SEEDS = list(range(42, 47)) # 5 seeds for fast physical CPU training

results_dict = {}
diagnostics_collinearity = {}
diagnostics_cosine_sim = {}

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

all_pretrained_experts = {}
gradient_tracking_data = {"bsigmoid": [0.0]*40, "softmax": [0.0]*40}

print("Starting upgraded physical experiments on Split-MNIST with shared initialization...")

for model_name, model_cfg in MODELS.items():
    L = model_cfg["L"]
    model_class = model_cfg["class"]
    forward_fn = model_cfg["forward"]
    
    results_dict[model_name] = {}
    diagnostics_collinearity[model_name] = {}
    diagnostics_cosine_sim[model_name] = {}
    
    # Pre-train all 4 expert classifiers once per backbone and seed with SHARED INITIALIZATION
    pretrained_experts = {}
    all_pretrained_experts[model_name] = pretrained_experts
    print(f"\n[Backbone: {model_name}] Pre-training expert classifiers on CPU...")
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pretrained_experts[seed] = {}
        
        # Instantiate a single shared base model for this seed
        base_model = model_class().to(device)
        
        for task_key in range(4):
            classes = TASKS[task_key]
            # Draw distinct offsets based on seed to ensure diversity (offset of 100 per seed is safe)
            subset = get_task_subset(full_dataset, classes, 512, seed_offset=seed*100)
            loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
            
            # Load the exact same base model parameters to guarantee shared initialization basin!
            expert_model = model_class().to(device)
            expert_model.load_state_dict(base_model.state_dict())
            
            optimizer = torch.optim.Adam(expert_model.parameters(), lr=0.002)
            criterion = nn.CrossEntropyLoss()
            
            expert_model.train()
            for epoch in range(12): # 12 epochs
                for img, label in loader:
                    img, label = img.to(device), label.to(device)
                    optimizer.zero_grad()
                    out = expert_model(img)
                    loss = criterion(out, label)
                    loss.backward()
                    optimizer.step()
            expert_model.eval()
            pretrained_experts[seed][task_key] = expert_model
            
        print(f"  Seed {seed} shared-init pre-training complete.")
        
    for suite_name, suite_cfg in SUITES.items():
        task_keys = suite_cfg["task_keys"]
        K = len(task_keys)
        
        suite_results = {
            "Uniform": [],
            "OFS-Tune": [],
            "L1-Global-Router": [],
            "Layer-wise-Router": [],
            "Layer-wise-Router-Softmax": [],
            "Layer-wise-Router-NoReg": [],
            "Oracle": []
        }
        
        col_ratios = []
        cosine_sims = []
        
        print(f"\nEvaluating Suite: {suite_name} on Architecture: {model_name}")
        
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # --- Load pre-trained experts ---
            experts = {k_idx: pretrained_experts[seed][task_key] for k_idx, task_key in enumerate(task_keys)}
            
            # 2. Calibration split (128 samples per task, total 128*K)
            cal_x_list = []
            cal_y_list = []
            for k_idx, task_key in enumerate(task_keys):
                classes = TASKS[task_key]
                # Offset by 1000 from training set to prevent leakage
                subset = get_task_subset(full_dataset, classes, 128, seed_offset=seed*100 + 1000)
                for img, label in subset:
                    cal_x_list.append(img.unsqueeze(0))
                    cal_y_list.append(torch.tensor([label]))
                    
            cal_x = torch.cat(cal_x_list, dim=0).to(device)
            cal_y = torch.cat(cal_y_list, dim=0).to(device)
            
            # 3. Test split (200 samples per task, total 200*K)
            test_x_list = []
            test_y_list = []
            test_splits = {}
            current_start = 0
            for k_idx, task_key in enumerate(task_keys):
                classes = TASKS[task_key]
                subset = get_task_subset(full_test_dataset, classes, 200, seed_offset=seed*20)
                for img, label in subset:
                    test_x_list.append(img.unsqueeze(0))
                    test_y_list.append(torch.tensor([label]))
                test_splits[k_idx] = (current_start, current_start + 200)
                current_start += 200
                
            test_x = torch.cat(test_x_list, dim=0).to(device)
            test_y = torch.cat(test_y_list, dim=0).to(device)
            
            # --- EVALUATE BASELINES ---
            
            # 1. Oracle Upper Bound (Evaluating each expert on its own task)
            oracle_accs = []
            for k_idx in range(K):
                start, end = test_splits[k_idx]
                x_sub = test_x[start:end]
                y_sub = test_y[start:end]
                
                with torch.no_grad():
                    logits = experts[k_idx](x_sub)
                    preds = logits.argmax(dim=-1)
                    acc = (preds == y_sub).float().mean().item() * 100.0
                    oracle_accs.append(acc)
            suite_results["Oracle"].append(np.mean(oracle_accs))
            
            # 2. Uniform Merging Baseline (alpha = 1/K for all, run forward merged)
            alpha_uniform = torch.full((L, test_x.size(0), K), 1.0/K, device=device)
            with torch.no_grad():
                logits_uniform = forward_fn(test_x, alpha_uniform, experts, K)
                preds_uniform = logits_uniform.argmax(dim=-1)
                acc_uniform = (preds_uniform == test_y).float().mean().item() * 100.0
            suite_results["Uniform"].append(acc_uniform)
            
            # 3. OFS-Tune Static Tuning (Learn static alpha_k \in [0, 1], sum to 1.0)
            theta_ofs = torch.zeros(K, device=device, requires_grad=True)
            optimizer_ofs = torch.optim.Adam([theta_ofs], lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            for step in range(30):
                optimizer_ofs.zero_grad()
                lambdas_raw = torch.sigmoid(theta_ofs) # shape (K,)
                lambdas = lambdas_raw / (lambdas_raw.sum() + 1e-8)
                alpha_ofs = lambdas.view(1, 1, K).repeat(L, cal_x.size(0), 1)
                logits = forward_fn(cal_x, alpha_ofs, experts, K)
                loss = criterion(logits, cal_y)
                loss.backward()
                optimizer_ofs.step()
                
            with torch.no_grad():
                final_lambdas_raw = torch.sigmoid(theta_ofs)
                final_lambdas = final_lambdas_raw / (final_lambdas_raw.sum() + 1e-8)
                alpha_ofs_test = final_lambdas.view(1, 1, K).repeat(L, test_x.size(0), 1)
                logits_ofs = forward_fn(test_x, alpha_ofs_test, experts, K)
                preds_ofs = logits_ofs.argmax(dim=-1)
                acc_ofs = (preds_ofs == test_y).float().mean().item() * 100.0
            suite_results["OFS-Tune"].append(acc_ofs)
            
            # 4. L1-Global-Router
            global_router = DynamicRouter(L, d=8, K=K, routing_type="global").to(device)
            optimizer_g = torch.optim.Adam(global_router.parameters(), lr=0.01)
            
            # Calibration
            for step in range(40):
                optimizer_g.zero_grad()
                flat_x = cal_x.view(cal_x.size(0), -1)
                psi = torch.mm(flat_x, P_proj.t())
                alpha_g = global_router(psi) # shape (L, B, K)
                logits = forward_fn(cal_x, alpha_g, experts, K)
                loss = criterion(logits, cal_y)
                loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in global_router.parameters() if p.requires_grad)
                (loss + loss_reg).backward()
                optimizer_g.step()
                
            # Evaluation
            global_router.eval()
            with torch.no_grad():
                flat_x_test = test_x.view(test_x.size(0), -1)
                psi_test = torch.mm(flat_x_test, P_proj.t())
                alpha_g_test = global_router(psi_test)
                logits_g = forward_fn(test_x, alpha_g_test, experts, K)
                preds_g = logits_g.argmax(dim=-1)
                acc_g = (preds_g == test_y).float().mean().item() * 100.0
            suite_results["L1-Global-Router"].append(acc_g)
            
            # 5. Layer-wise-Router (Ours, Regularized)
            layer_router = DynamicRouter(L, d=8, K=K, routing_type="layer-wise").to(device)
            optimizer_l = torch.optim.Adam(layer_router.parameters(), lr=0.01)
            
            # Calibration
            for step in range(40):
                optimizer_l.zero_grad()
                flat_x = cal_x.view(cal_x.size(0), -1)
                psi = torch.mm(flat_x, P_proj.t())
                alpha_l = layer_router(psi) # shape (L, B, K)
                logits = forward_fn(cal_x, alpha_l, experts, K)
                loss = criterion(logits, cal_y)
                loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in layer_router.parameters() if p.requires_grad)
                (loss + loss_reg).backward()
                
                # Track gradient norm for BSigmoid
                grad_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in layer_router.parameters() if p.grad is not None)).item()
                if model_name == "TinyCNN-4" and suite_name == "Cross-Domain":
                    gradient_tracking_data["bsigmoid"][step] += grad_norm / len(SEEDS)
                    
                optimizer_l.step()
                
            # Evaluation
            layer_router.eval()
            with torch.no_grad():
                flat_x_test = test_x.view(test_x.size(0), -1)
                psi_test = torch.mm(flat_x_test, P_proj.t())
                alpha_l_test = layer_router(psi_test)
                logits_l = forward_fn(test_x, alpha_l_test, experts, K)
                preds_l = logits_l.argmax(dim=-1)
                acc_l = (preds_l == test_y).float().mean().item() * 100.0
            suite_results["Layer-wise-Router"].append(acc_l)
            
            # Spectral Diagnostics of the Layer-wise Router
            with torch.no_grad():
                A = alpha_l_test.mean(dim=1) # shape (L, K)
                U, S, V = torch.linalg.svd(A, full_matrices=False)
                col_r = (S[0] / S.sum()).item()
                col_ratios.append(col_r)
                
                A_norm = A / (torch.norm(A, dim=1, keepdim=True) + 1e-8)
                cos_s = torch.mm(A_norm, A_norm.t()).cpu().numpy()
                cosine_sims.append(cos_s)
                
            # 6. Layer-wise-Router-Softmax (Softmax activation with L2 Regularization)
            layer_router_s = DynamicRouter(L, d=8, K=K, routing_type="layer-wise", activation="softmax").to(device)
            optimizer_s = torch.optim.Adam(layer_router_s.parameters(), lr=0.01)
            
            for step in range(40):
                optimizer_s.zero_grad()
                flat_x = cal_x.view(cal_x.size(0), -1)
                psi = torch.mm(flat_x, P_proj.t())
                alpha_s = layer_router_s(psi) # shape (L, B, K)
                logits = forward_fn(cal_x, alpha_s, experts, K)
                loss = criterion(logits, cal_y)
                loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in layer_router_s.parameters() if p.requires_grad)
                (loss + loss_reg).backward()
                
                # Track gradient norm for Softmax
                grad_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in layer_router_s.parameters() if p.grad is not None)).item()
                if model_name == "TinyCNN-4" and suite_name == "Cross-Domain":
                    gradient_tracking_data["softmax"][step] += grad_norm / len(SEEDS)
                    
                optimizer_s.step()
                
            layer_router_s.eval()
            with torch.no_grad():
                alpha_s_test = layer_router_s(psi_test)
                logits_s = forward_fn(test_x, alpha_s_test, experts, K)
                preds_s = logits_s.argmax(dim=-1)
                acc_s = (preds_s == test_y).float().mean().item() * 100.0
            suite_results["Layer-wise-Router-Softmax"].append(acc_s)
                
            # 7. Layer-wise-Router-NoReg (Without L2 Regularization)
            layer_router_nr = DynamicRouter(L, d=8, K=K, routing_type="layer-wise", activation="bsigmoid").to(device)
            optimizer_nr = torch.optim.Adam(layer_router_nr.parameters(), lr=0.01)
            
            for step in range(40):
                optimizer_nr.zero_grad()
                flat_x = cal_x.view(cal_x.size(0), -1)
                psi = torch.mm(flat_x, P_proj.t())
                alpha_nr = layer_router_nr(psi) # shape (L, B, K)
                logits = forward_fn(cal_x, alpha_nr, experts, K)
                loss = criterion(logits, cal_y)
                loss.backward()
                optimizer_nr.step()
                
            layer_router_nr.eval()
            with torch.no_grad():
                alpha_nr_test = layer_router_nr(psi_test)
                logits_nr = forward_fn(test_x, alpha_nr_test, experts, K)
                preds_nr = logits_nr.argmax(dim=-1)
                acc_nr = (preds_nr == test_y).float().mean().item() * 100.0
            suite_results["Layer-wise-Router-NoReg"].append(acc_nr)
            
            print(f"  Seed {seed} completed. Layer-wise (BSigmoid) Acc: {acc_l:.2f}% | Softmax Acc: {acc_s:.2f}% | SVD: {col_r:.4f}")
            
        # Compile results
        results_dict[model_name][suite_name] = {}
        for k, v in suite_results.items():
            results_dict[model_name][suite_name][k] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v))
            }
            
        diagnostics_collinearity[model_name][suite_name] = float(np.mean(col_ratios))
        diagnostics_cosine_sim[model_name][suite_name] = np.mean(cosine_sims, axis=0)
        
        print(f"[{model_name} - {suite_name}] Final Mean Layer-wise: {np.mean(suite_results['Layer-wise-Router']):.2f}%, SVD Collinearity: {np.mean(col_ratios):.4f}")

# --- NEW EXPERIMENT: NATURAL-IMAGE SUITE (CIFAR-10 + SVHN) ---
print("\n=== Running NEW Natural-Image Suite (CIFAR-10 + SVHN) on NaturalCNN-4 ===")

class NaturalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def forward_merged_natural(x, alpha, experts_dict, K):
    if alpha.dim() == 3:
        alpha_avg = alpha.mean(dim=1) # shape (L, K)
    else:
        alpha_avg = alpha
        
    w1 = sum(alpha_avg[0, j] * experts_dict[j].conv1.weight for j in range(K))
    b1 = sum(alpha_avg[0, j] * experts_dict[j].conv1.bias for j in range(K))
    w2 = sum(alpha_avg[1, j] * experts_dict[j].conv2.weight for j in range(K))
    b2 = sum(alpha_avg[1, j] * experts_dict[j].conv2.bias for j in range(K))
    w3 = sum(alpha_avg[2, j] * experts_dict[j].conv3.weight for j in range(K))
    b3 = sum(alpha_avg[2, j] * experts_dict[j].conv3.bias for j in range(K))
    w4 = sum(alpha_avg[3, j] * experts_dict[j].fc.weight for j in range(K))
    b4 = sum(alpha_avg[3, j] * experts_dict[j].fc.bias for j in range(K))
    
    h = F.conv2d(x, w1, b1, padding=1)
    h = F.relu(h)
    h = F.max_pool2d(h, 2)
    h = F.conv2d(h, w2, b2, padding=1)
    h = F.relu(h)
    h = F.max_pool2d(h, 2)
    h = F.conv2d(h, w3, b3, padding=1)
    h = F.relu(h)
    h = F.max_pool2d(h, 2)
    h = h.view(h.size(0), -1)
    out = F.linear(h, w4, b4)
    return out

# Load natural datasets
transform_nat = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading natural datasets (CIFAR-10 and SVHN)...")
full_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_nat)
full_cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_nat)
full_svhn = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_nat)
full_svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_nat)

results_dict["NaturalCNN-4"] = {}
diagnostics_collinearity["NaturalCNN-4"] = {}

suite_results_nat = {
    "Uniform": [],
    "OFS-Tune": [],
    "L1-Global-Router": [],
    "Layer-wise-Router": [],
    "Oracle": []
}
col_ratios_nat = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Pre-train natural experts on 256 samples
    print(f"  NaturalCNN-4 Seed {seed}: Pre-training CIFAR-10 expert...")
    cifar_sub = torch.utils.data.Subset(full_cifar, list(range(seed*100, seed*100 + 256)))
    cifar_loader = torch.utils.data.DataLoader(cifar_sub, batch_size=32, shuffle=True)
    cifar_expert = NaturalCNN().to(device)
    optimizer_c = torch.optim.Adam(cifar_expert.parameters(), lr=0.001)
    criterion_c = nn.CrossEntropyLoss()
    cifar_expert.train()
    for epoch in range(12):
        for img, label in cifar_loader:
            optimizer_c.zero_grad()
            out = cifar_expert(img.to(device))
            loss = criterion_c(out, label.to(device))
            loss.backward()
            optimizer_c.step()
    cifar_expert.eval()
    
    print(f"  NaturalCNN-4 Seed {seed}: Pre-training SVHN expert...")
    svhn_sub = torch.utils.data.Subset(full_svhn, list(range(seed*100, seed*100 + 256)))
    svhn_loader = torch.utils.data.DataLoader(svhn_sub, batch_size=32, shuffle=True)
    svhn_expert = NaturalCNN().to(device)
    optimizer_s = torch.optim.Adam(svhn_expert.parameters(), lr=0.001)
    svhn_expert.train()
    for epoch in range(12):
        for img, label in svhn_loader:
            optimizer_s.zero_grad()
            out = svhn_expert(img.to(device))
            loss = criterion_c(out, label.to(device))
            loss.backward()
            optimizer_s.step()
    svhn_expert.eval()
    
    experts_nat = {0: cifar_expert, 1: svhn_expert}
    
    # Calibration Split: 128 samples per task (total 256)
    cifar_cal = torch.utils.data.Subset(full_cifar, list(range(1000 + seed*100, 1000 + seed*100 + 128)))
    svhn_cal = torch.utils.data.Subset(full_svhn, list(range(1000 + seed*100, 1000 + seed*100 + 128)))
    cal_x_nat_list = []
    cal_y_nat_list = []
    for x, y in cifar_cal:
        cal_x_nat_list.append(x.unsqueeze(0))
        cal_y_nat_list.append(torch.tensor([y]))
    for x, y in svhn_cal:
        cal_x_nat_list.append(x.unsqueeze(0))
        cal_y_nat_list.append(torch.tensor([y]))
    cal_x_nat = torch.cat(cal_x_nat_list, dim=0).to(device)
    cal_y_nat = torch.cat(cal_y_nat_list, dim=0).to(device)
    
    # Test Split: 200 samples per task (total 400)
    cifar_test = torch.utils.data.Subset(full_cifar_test, list(range(seed*100, seed*100 + 200)))
    svhn_test = torch.utils.data.Subset(full_svhn_test, list(range(seed*100, seed*100 + 200)))
    test_x_nat_list = []
    test_y_nat_list = []
    for x, y in cifar_test:
        test_x_nat_list.append(x.unsqueeze(0))
        test_y_nat_list.append(torch.tensor([y]))
    for x, y in svhn_test:
        test_x_nat_list.append(x.unsqueeze(0))
        test_y_nat_list.append(torch.tensor([y]))
    test_x_nat = torch.cat(test_x_nat_list, dim=0).to(device)
    test_y_nat = torch.cat(test_y_nat_list, dim=0).to(device)
    
    # 1. Oracle Upper Bound
    oracle_accs_nat = []
    # CIFAR-10 Expert
    with torch.no_grad():
        logits_c = cifar_expert(test_x_nat[:200])
        preds_c = logits_c.argmax(dim=-1)
        acc_c = (preds_c == test_y_nat[:200]).float().mean().item() * 100.0
        oracle_accs_nat.append(acc_c)
    # SVHN Expert
    with torch.no_grad():
        logits_s = svhn_expert(test_x_nat[200:])
        preds_s = logits_s.argmax(dim=-1)
        acc_s = (preds_s == test_y_nat[200:]).float().mean().item() * 100.0
        oracle_accs_nat.append(acc_s)
    suite_results_nat["Oracle"].append(np.mean(oracle_accs_nat))
    
    # 2. Uniform Merging
    alpha_uniform_nat = torch.full((4, test_x_nat.size(0), 2), 0.5, device=device)
    with torch.no_grad():
        logits_unif = forward_merged_natural(test_x_nat, alpha_uniform_nat, experts_nat, 2)
        preds_unif = logits_unif.argmax(dim=-1)
        acc_unif = (preds_unif == test_y_nat).float().mean().item() * 100.0
    suite_results_nat["Uniform"].append(acc_unif)
    
    # 3. OFS-Tune Static Tuning
    theta_ofs_nat = torch.zeros(2, device=device, requires_grad=True)
    optimizer_ofs_nat = torch.optim.Adam([theta_ofs_nat], lr=0.01)
    for step in range(30):
        optimizer_ofs_nat.zero_grad()
        lambdas_raw = torch.sigmoid(theta_ofs_nat)
        lambdas = lambdas_raw / (lambdas_raw.sum() + 1e-8)
        alpha_ofs = lambdas.view(1, 1, 2).repeat(4, cal_x_nat.size(0), 1)
        logits = forward_merged_natural(cal_x_nat, alpha_ofs, experts_nat, 2)
        loss = criterion_c(logits, cal_y_nat)
        loss.backward()
        optimizer_ofs_nat.step()
        
    with torch.no_grad():
        final_lambdas_raw = torch.sigmoid(theta_ofs_nat)
        final_lambdas = final_lambdas_raw / (final_lambdas_raw.sum() + 1e-8)
        alpha_ofs_test = final_lambdas.view(1, 1, 2).repeat(4, test_x_nat.size(0), 1)
        logits_ofs = forward_merged_natural(test_x_nat, alpha_ofs_test, experts_nat, 2)
        preds_ofs = logits_ofs.argmax(dim=-1)
        acc_ofs = (preds_ofs == test_y_nat).float().mean().item() * 100.0
    suite_results_nat["OFS-Tune"].append(acc_ofs)
    
    # 4. L1-Global-Router
    P_proj_nat = torch.randn(8, 3072, device=device)
    global_router_nat = DynamicRouter(4, d=8, K=2, routing_type="global").to(device)
    optimizer_g_nat = torch.optim.Adam(global_router_nat.parameters(), lr=0.01)
    for step in range(40):
        optimizer_g_nat.zero_grad()
        flat_x = cal_x_nat.view(cal_x_nat.size(0), -1)
        psi = torch.mm(flat_x, P_proj_nat.t())
        alpha_g = global_router_nat(psi)
        logits = forward_merged_natural(cal_x_nat, alpha_g, experts_nat, 2)
        loss = criterion_c(logits, cal_y_nat)
        loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in global_router_nat.parameters() if p.requires_grad)
        (loss + loss_reg).backward()
        optimizer_g_nat.step()
        
    global_router_nat.eval()
    with torch.no_grad():
        flat_x_test = test_x_nat.view(test_x_nat.size(0), -1)
        psi_test = torch.mm(flat_x_test, P_proj_nat.t())
        alpha_g_test = global_router_nat(psi_test)
        logits_g = forward_merged_natural(test_x_nat, alpha_g_test, experts_nat, 2)
        preds_g = logits_g.argmax(dim=-1)
        acc_g = (preds_g == test_y_nat).float().mean().item() * 100.0
    suite_results_nat["L1-Global-Router"].append(acc_g)
    
    # 5. Layer-wise-Router (Ours, Regularized)
    layer_router_nat = DynamicRouter(4, d=8, K=2, routing_type="layer-wise").to(device)
    optimizer_l_nat = torch.optim.Adam(layer_router_nat.parameters(), lr=0.01)
    for step in range(40):
        optimizer_l_nat.zero_grad()
        flat_x = cal_x_nat.view(cal_x_nat.size(0), -1)
        psi = torch.mm(flat_x, P_proj_nat.t())
        alpha_l = layer_router_nat(psi)
        logits = forward_merged_natural(cal_x_nat, alpha_l, experts_nat, 2)
        loss = criterion_c(logits, cal_y_nat)
        loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in layer_router_nat.parameters() if p.requires_grad)
        (loss + loss_reg).backward()
        optimizer_l_nat.step()
        
    layer_router_nat.eval()
    with torch.no_grad():
        flat_x_test = test_x_nat.view(test_x_nat.size(0), -1)
        psi_test = torch.mm(flat_x_test, P_proj_nat.t())
        alpha_l_test = layer_router_nat(psi_test)
        logits_l = forward_merged_natural(test_x_nat, alpha_l_test, experts_nat, 2)
        preds_l = logits_l.argmax(dim=-1)
        acc_l = (preds_l == test_y_nat).float().mean().item() * 100.0
    suite_results_nat["Layer-wise-Router"].append(acc_l)
    
    # Compute spectral collinearity on natural images
    with torch.no_grad():
        A_nat = alpha_l_test.mean(dim=1)
        U_nat, S_nat, V_nat = torch.linalg.svd(A_nat, full_matrices=False)
        col_r_nat = (S_nat[0] / S_nat.sum()).item()
        col_ratios_nat.append(col_r_nat)
        
    print(f"    Seed {seed} Natural Completed. Layer-wise Acc: {acc_l:.2f}% | SVD Collinearity: {col_r_nat:.4f}")

results_dict["NaturalCNN-4"]["Natural-Image-Suite"] = {}
for k, v in suite_results_nat.items():
    results_dict["NaturalCNN-4"]["Natural-Image-Suite"][k] = {
        "mean": float(np.mean(v)),
        "std": float(np.std(v))
    }
diagnostics_collinearity["NaturalCNN-4"]["Natural-Image-Suite"] = float(np.mean(col_ratios_nat))
print(f"[NaturalCNN-4] Natural Image Suite: Layer-wise Mean Acc: {np.mean(suite_results_nat['Layer-wise-Router']):.2f}%, SVD Collinearity: {np.mean(col_ratios_nat):.4f}")


# --- NEW EXPERIMENT: CALIBRATION BUDGET SCALING ---
print("\n=== Running NEW Calibration Budget Scaling Experiment on TinyCNN-4 ===")
scaling_budgets = [64, 128, 256, 512, 1024]
scaling_results = {
    "OFS-Tune": {b: [] for b in scaling_budgets},
    "Layer-wise-Router": {b: [] for b in scaling_budgets}
}

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    experts_scale = {k_idx: all_pretrained_experts["TinyCNN-4"][seed][task_key] for k_idx, task_key in enumerate(SUITES["Cross-Domain"]["task_keys"])}
    K_scale = len(experts_scale)
    L_scale = 4
    
    test_x_list = []
    test_y_list = []
    for k_idx, task_key in enumerate(SUITES["Cross-Domain"]["task_keys"]):
        classes = TASKS[task_key]
        subset = get_task_subset(full_test_dataset, classes, 200, seed_offset=seed*20)
        for img, label in subset:
            test_x_list.append(img.unsqueeze(0))
            test_y_list.append(torch.tensor([label]))
    test_x_scale = torch.cat(test_x_list, dim=0).to(device)
    test_y_scale = torch.cat(test_y_list, dim=0).to(device)
    
    for budget in scaling_budgets:
        cal_x_list = []
        cal_y_list = []
        for k_idx, task_key in enumerate(SUITES["Cross-Domain"]["task_keys"]):
            classes = TASKS[task_key]
            subset = get_task_subset(full_dataset, classes, budget, seed_offset=seed*100 + 1000)
            for img, label in subset:
                cal_x_list.append(img.unsqueeze(0))
                cal_y_list.append(torch.tensor([label]))
        cal_x_scale = torch.cat(cal_x_list, dim=0).to(device)
        cal_y_scale = torch.cat(cal_y_list, dim=0).to(device)
        
        # 1. OFS-Tune Static Tuning
        theta_ofs_scale = torch.zeros(K_scale, device=device, requires_grad=True)
        optimizer_ofs_scale = torch.optim.Adam([theta_ofs_scale], lr=0.01)
        for step in range(30):
            optimizer_ofs_scale.zero_grad()
            lambdas_raw = torch.sigmoid(theta_ofs_scale)
            lambdas = lambdas_raw / (lambdas_raw.sum() + 1e-8)
            alpha_ofs = lambdas.view(1, 1, K_scale).repeat(L_scale, cal_x_scale.size(0), 1)
            logits = forward_merged_cnn(cal_x_scale, alpha_ofs, experts_scale, K_scale)
            loss = criterion(logits, cal_y_scale)
            loss.backward()
            optimizer_ofs_scale.step()
            
        with torch.no_grad():
            final_lambdas_raw = torch.sigmoid(theta_ofs_scale)
            final_lambdas = final_lambdas_raw / (final_lambdas_raw.sum() + 1e-8)
            alpha_ofs_test = final_lambdas.view(1, 1, K_scale).repeat(L_scale, test_x_scale.size(0), 1)
            logits_ofs = forward_merged_cnn(test_x_scale, alpha_ofs_test, experts_scale, K_scale)
            preds_ofs = logits_ofs.argmax(dim=-1)
            acc_ofs = (preds_ofs == test_y_scale).float().mean().item() * 100.0
            scaling_results["OFS-Tune"][budget].append(acc_ofs)
            
        # 2. Layer-wise Router (Ours)
        layer_router_scale = DynamicRouter(L_scale, d=8, K=K_scale, routing_type="layer-wise").to(device)
        optimizer_l_scale = torch.optim.Adam(layer_router_scale.parameters(), lr=0.01)
        for step in range(40):
            optimizer_l_scale.zero_grad()
            flat_x = cal_x_scale.view(cal_x_scale.size(0), -1)
            psi = torch.mm(flat_x, P_proj.t())
            alpha_l = layer_router_scale(psi)
            logits = forward_merged_cnn(cal_x_scale, alpha_l, experts_scale, K_scale)
            loss = criterion(logits, cal_y_scale)
            loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in layer_router_scale.parameters() if p.requires_grad)
            (loss + loss_reg).backward()
            optimizer_l_scale.step()
            
        layer_router_scale.eval()
        with torch.no_grad():
            flat_x_test = test_x_scale.view(test_x_scale.size(0), -1)
            psi_test = torch.mm(flat_x_test, P_proj.t())
            alpha_l_test = layer_router_scale(psi_test)
            logits_l = forward_merged_cnn(test_x_scale, alpha_l_test, experts_scale, K_scale)
            preds_l = logits_l.argmax(dim=-1)
            acc_l = (preds_l == test_y_scale).float().mean().item() * 100.0
            scaling_results["Layer-wise-Router"][budget].append(acc_l)
            
    print(f"    Seed Scaling Completed.")

# Compile budget scaling results
scaling_metrics = {
    "OFS-Tune": {},
    "Layer-wise-Router": {}
}
for m in ["OFS-Tune", "Layer-wise-Router"]:
    for b in scaling_budgets:
        scaling_metrics[m][b] = {
            "mean": float(np.mean(scaling_results[m][b])),
            "std": float(np.std(scaling_results[m][b]))
        }

# Write raw metrics to JSON (including new NaturalCNN, Scaling, and Gradient tracking metrics!)
with open("results/metrics.json", "w") as f:
    json.dump({
        "accuracies": results_dict,
        "collinearity": diagnostics_collinearity,
        "scaling_metrics": scaling_metrics,
        "gradient_tracking_data": gradient_tracking_data
    }, f, indent=4)

# Also write to submission/metrics.json to guarantee complete synchronization!
with open("submission/metrics.json", "w") as f:
    json.dump({
        "accuracies": results_dict,
        "collinearity": diagnostics_collinearity,
        "scaling_metrics": scaling_metrics,
        "gradient_tracking_data": gradient_tracking_data
    }, f, indent=4)

# Generate Figure 4: Calibration Budget Scaling plot (crossover visualization)
plt.figure(figsize=(7, 5))
b_vals = np.array(scaling_budgets)

ofs_means = [scaling_metrics["OFS-Tune"][b]["mean"] for b in scaling_budgets]
ofs_stds = [scaling_metrics["OFS-Tune"][b]["std"] for b in scaling_budgets]

layer_means = [scaling_metrics["Layer-wise-Router"][b]["mean"] for b in scaling_budgets]
layer_stds = [scaling_metrics["Layer-wise-Router"][b]["std"] for b in scaling_budgets]

plt.errorbar(b_vals, ofs_means, yerr=ofs_stds, fmt='-o', color='#2ca02c', label='OFS-Tune (Static)', capsize=4, linewidth=2, markersize=8)
plt.errorbar(b_vals, layer_means, yerr=layer_stds, fmt='-s', color='#ff7f0e', label='Layer-wise Router (Ours)', capsize=4, linewidth=2, markersize=8)

plt.xscale('log')
plt.xticks(scaling_budgets, scaling_budgets)
plt.xlabel("Calibration Samples per Task ($B$)", fontsize=12)
plt.ylabel("Physical Test Accuracy (%)", fontsize=12)
plt.title("Generalization Crossover under Scaled Calibration Budgets", fontsize=13, fontweight='bold')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("fig4_calibration_scaling.png", dpi=300)
plt.savefig("results/fig4_calibration_scaling.png", dpi=300)
plt.close()

print("\nGenerating physical evaluation plots...")

# Figure 1: Collinearity Ratio bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(SUITES))
width = 0.35

vit_cols = [diagnostics_collinearity["DeepMLP-12"][s] for s in SUITES.keys()]
resnet_cols = [diagnostics_collinearity["TinyCNN-4"][s] for s in SUITES.keys()]

rects1 = ax.bar(x - width/2, vit_cols, width, label='DeepMLP-12 (L=12)', color='#1f77b4', edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, resnet_cols, width, label='TinyCNN-4 (L=4)', color='#ff7f0e', edgecolor='black', alpha=0.8)

ax.set_ylabel('SVD Collinearity Ratio ($\\rho_{collinear}$)', fontsize=12)
ax.set_title('Emergent SVD Collinearity of Physical Dynamic Routing', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SUITES.keys(), fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("fig1_collinearity_ratio.png", dpi=300)
plt.savefig("results/fig1_collinearity_ratio.png", dpi=300)
plt.close()

# Figure 2: Cosine Similarity Heatmaps for DeepMLP-12 (Low-Conflict vs Cross-Domain)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(diagnostics_cosine_sim["DeepMLP-12"]["Low-Conflict"], cmap='viridis', vmin=0.5, vmax=1.0)
ax1.set_title("DeepMLP-12 on Digits 0-3 (Low-Conflict)\n(Near-Collinear Layer Routing)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Layer Index", fontsize=10)
ax1.set_ylabel("Layer Index", fontsize=10)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(diagnostics_cosine_sim["DeepMLP-12"]["Cross-Domain"], cmap='viridis', vmin=0.5, vmax=1.0)
ax2.set_title("DeepMLP-12 on Digits 0-7 (Cross-Domain)\n(Depth-Specific Specialized Routing)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Layer Index", fontsize=10)
ax2.set_ylabel("Layer Index", fontsize=10)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.suptitle("Pairwise Inter-Layer Cosine Similarity Matrix $S$ of Physical Routing Coefficients", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("fig2_cosine_similarity.png", dpi=300)
plt.savefig("results/fig2_cosine_similarity.png", dpi=300)
plt.close()

# Figure 3: Accuracy comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
methods_to_plot = ["Uniform", "OFS-Tune", "L1-Global-Router", "Layer-wise-Router", "Oracle"]

for i, suite_name in enumerate(SUITES.keys()):
    ax = axes[i]
    
    vit_means = [results_dict["DeepMLP-12"][suite_name][m]["mean"] for m in methods_to_plot]
    vit_stds = [results_dict["DeepMLP-12"][suite_name][m]["std"] for m in methods_to_plot]
    
    resnet_means = [results_dict["TinyCNN-4"][suite_name][m]["mean"] for m in methods_to_plot]
    resnet_stds = [results_dict["TinyCNN-4"][suite_name][m]["std"] for m in methods_to_plot]
    
    x = np.arange(len(methods_to_plot))
    width = 0.35
    
    ax.bar(x - width/2, vit_means, width, yerr=vit_stds, label='DeepMLP-12', color='#1f77b4', edgecolor='black', alpha=0.8, capsize=4)
    ax.bar(x + width/2, resnet_means, width, yerr=resnet_stds, label='TinyCNN-4', color='#ff7f0e', edgecolor='black', alpha=0.8, capsize=4)
    
    ax.set_title(f"{suite_name} Suite", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["Uniform", "OFS-Tune", "Global\nRouter", "Layer-wise\nRouter (Ours)", "Oracle"], fontsize=9, rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if i == 0:
        ax.set_ylabel("Physical Multi-Task Test Accuracy (%)", fontsize=12)
        ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0, 100)

plt.suptitle("Generalization Performance of Physical weight-space model merging", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("fig3_accuracy_comparison.png", dpi=300)
plt.savefig("results/fig3_accuracy_comparison.png", dpi=300)
plt.close()

print("All upgraded physical experiments complete and plots generated successfully!")
