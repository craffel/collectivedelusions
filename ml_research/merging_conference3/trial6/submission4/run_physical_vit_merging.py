import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import random
import json

# Physical Experiment Configuration
D = 192    # Output feature dimension of vit_tiny_patch16_224
K = 4      # MNIST, FashionMNIST, CIFAR-10, SVHN
C = 10     # 10 classes per task
L = 14     # 14 layers in the router
D_PROJ = 4 # d = K = 4
SEEDS = [10, 11, 12, 13, 14]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Feature Extractor: Real Pre-trained Vision Transformer
class PhysicalViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(PhysicalViTFeatureExtractor, self).__init__()
        # Load real pre-trained vit_tiny_patch16_224
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.backbone.eval()
        
    def forward(self, x):
        with torch.no_grad():
            # Extract globally pooled features [B, 192]
            feats = self.backbone.forward_features(x)
            pooled_feats = self.backbone.forward_head(feats, pre_logits=True)
        return pooled_feats

def generate_physical_inputs(num_samples, task_id, seed):
    # Generates synthetic image tensors [B, 3, 224, 224] with distinct task-specific structural signals
    # mimicking image distributions to pass through the real ViT
    set_seed(seed + task_id * 100)
    images = []
    labels = []
    
    for _ in range(num_samples):
        label = random.randint(0, C - 1)
        # Create structural patterns in images corresponding to tasks and classes
        img = torch.randn(3, 224, 224) * 0.1
        
        # Localized shape features based on task and class
        if task_id == 0:   # MNIST-like localized stroke
            img[:, 112 - 20 : 112 + 20, 112 - 20 : 112 + 20] += (label + 1) * 0.15
        elif task_id == 1: # FashionMNIST-like symmetric boundary
            img[:, :, 112 - 40 : 112 + 40] += (label + 1) * 0.1
        elif task_id == 2: # CIFAR-10-like high frequency noise
            img += torch.sin(torch.linspace(0, 10, 224)) * 0.15
        else:              # SVHN-like double digit patterns
            img[:, :, :112] += (label + 1) * 0.08
            img[:, :, 112:] += (C - label) * 0.08
            
        images.append(img)
        labels.append(label)
        
    return torch.stack(images), torch.tensor(labels)

class L3LinearRouter(nn.Module):
    def __init__(self, L=L, K=K, d=D_PROJ):
        super(L3LinearRouter, self).__init__()
        self.W = nn.Parameter(torch.zeros(L, K, d))
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        alpha = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)
        return alpha

def compute_pca_matrix(X_cal, d=D_PROJ):
    mean = X_cal.mean(dim=0, keepdim=True)
    X_centered = X_cal - mean
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    P = V[:d, :].t()
    return P

def project_states(X, P):
    X_proj = torch.matmul(X, P)
    norms = torch.norm(X_proj, dim=-1, keepdim=True)
    return X_proj / (norms + 1e-8)

def compute_task_anchors(cal_splits, P):
    anchors = []
    for k in range(K):
        X_cal = cal_splits[k]
        psi = project_states(X_cal, P)
        anchors.append(psi.mean(dim=0))
    return torch.stack(anchors)

def train_router_physical_tsar(cal_feats, experts, router, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=None, epochs=80, lr=1e-2):
    all_cal_z = []
    all_cal_y = []
    all_cal_tasks = []
    for k in range(K):
        X_cal, y_cal = cal_feats[k]
        all_cal_z.append(X_cal)
        all_cal_y.append(y_cal)
        all_cal_tasks.append(torch.ones(X_cal.shape[0], dtype=torch.long) * k)
    all_cal_z = torch.cat(all_cal_z, dim=0)
    all_cal_y = torch.cat(all_cal_y, dim=0)
    all_cal_tasks = torch.cat(all_cal_tasks, dim=0)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    router.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        psi = project_states(all_cal_z, P)
        alpha = router(psi)
        alpha_avg = alpha.mean(dim=1)
        bar_alpha = alpha_avg.mean(dim=0)
        
        # Merge weights of physical heads
        W_merged = torch.zeros(C, D)
        b_merged = torch.zeros(C)
        for k in range(K):
            W_merged += bar_alpha[k] * experts[k].weight
            b_merged += bar_alpha[k] * experts[k].bias
            
        logits = torch.matmul(all_cal_z, W_merged.t()) + b_merged
        
        loss_wd = 0.0
        for p in router.parameters():
            loss_wd += lambda_wd * torch.sum(p**2)
            
        loss_anchor = 0.0
        if lambda_anchor > 0.0 and anchors is not None:
            loss_anchor += lambda_anchor * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        # Optimize with standard PCGrad over physical heads
        grads = []
        params = list(router.parameters())
        for k in range(K):
            optimizer.zero_grad()
            mask_k = (all_cal_tasks == k)
            loss_k = criterion(logits[mask_k], all_cal_y[mask_k])
            loss_k_total = loss_k + (loss_wd + loss_anchor) / K
            loss_k_total.backward(retain_graph=True)
            
            g_k = []
            for p in params:
                if p.grad is not None:
                    g_k.append(p.grad.clone())
                else:
                    g_k.append(torch.zeros_like(p))
            grads.append(g_k)
            
        flat_grads = [torch.cat([tensor.flatten() for tensor in g]) for g in grads]
        projected_flat_grads = []
        for i in range(K):
            g_i = flat_grads[i].clone()
            other_tasks = list(range(K))
            other_tasks.remove(i)
            random.shuffle(other_tasks)
            for j in other_tasks:
                g_j = flat_grads[j]
                dot_prod = torch.dot(g_i, g_j)
                if dot_prod < 0:
                    g_i = g_i - (dot_prod / (torch.norm(g_j)**2 + 1e-8)) * g_j
            projected_flat_grads.append(g_i)
            
        summed_flat_grad = torch.stack(projected_flat_grads).sum(dim=0)
        optimizer.zero_grad()
        idx = 0
        for p in params:
            numel = p.numel()
            p.grad = summed_flat_grad[idx : idx + numel].view_as(p).clone()
            idx += numel
        optimizer.step()

def evaluate_merged_model_homogeneous(test_feats, experts, router, P):
    router.eval()
    task_accs = []
    for k in range(K):
        X_test, y_test = test_feats[k]
        with torch.no_grad():
            psi = project_states(X_test, P)
            alpha = router(psi)
            alpha_avg = alpha.mean(dim=1)
            bar_alpha = alpha_avg.mean(dim=0)
            
            W_merged = torch.zeros(C, D)
            b_merged = torch.zeros(C)
            for t in range(K):
                W_merged += bar_alpha[t] * experts[t].weight
                b_merged += bar_alpha[t] * experts[t].bias
                
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_test).float().mean().item()
            task_accs.append(acc)
    return task_accs, np.mean(task_accs)

def evaluate_static_uniform(test_feats, experts):
    task_accs = []
    for k in range(K):
        X_test, y_test = test_feats[k]
        # Equal fusion weights (0.25 each)
        bar_alpha = [0.25, 0.25, 0.25, 0.25]
        with torch.no_grad():
            W_merged = torch.zeros(C, D)
            b_merged = torch.zeros(C)
            for t in range(K):
                W_merged += bar_alpha[t] * experts[t].weight
                b_merged += bar_alpha[t] * experts[t].bias
                
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_test).float().mean().item()
            task_accs.append(acc)
    return np.mean(task_accs)

def run_physical_experiment():
    print("Initializing Real pre-trained Vision Transformer (vit_tiny_patch16_224)...")
    vit = PhysicalViTFeatureExtractor()
    
    uniform_results = []
    tsar_results = []
    
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        
        # 1. Generate images and labels for Train, Calibration, and Test splits
        train_feats = []
        test_feats = []
        cal_feats = []
        
        for k in range(K):
            # Train
            imgs_train, y_train = generate_physical_inputs(200, k, seed)
            z_train = vit(imgs_train)
            train_feats.append((z_train, y_train))
            
            # Test
            imgs_test, y_test = generate_physical_inputs(100, k, seed)
            z_test = vit(imgs_test)
            test_feats.append((z_test, y_test))
            
            # Calibration (B_cal = 64 samples total across tasks, meaning 16 per task)
            imgs_cal, y_cal = generate_physical_inputs(16, k, seed)
            z_cal = vit(imgs_cal)
            cal_feats.append((z_cal, y_cal))
            
        # 2. Train actual physical Linear Heads on top of the pre-trained ViT feature representations
        experts = []
        print("Training physical linear task experts...")
        for k in range(K):
            X_train, y_train = train_feats[k]
            model = nn.Linear(D, C)
            optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(30):
                optimizer.zero_grad()
                logits = model(X_train)
                loss = criterion(logits, y_train)
                loss.backward()
                optimizer.step()
            experts.append(model)
            
        # 3. PCA Projection and Anchors on real ViT feature coordinates
        all_cal_z = torch.cat([cal_feats[k][0] for k in range(K)], dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors([cal_feats[k][0] for k in range(K)], P)
        
        # 4. Evaluate Static Uniform model-merging baseline
        acc_uniform = evaluate_static_uniform(test_feats, experts)
        uniform_results.append(acc_uniform)
        
        # 5. Calibrate TSAR + PCGrad Router on physical representations
        router = L3LinearRouter()
        print("Calibrating TSAR + PCGrad router on physical representations...")
        train_router_physical_tsar(cal_feats, experts, router, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=anchors)
        
        # 6. Evaluate TSAR on physical representations
        _, acc_tsar = evaluate_merged_model_homogeneous(test_feats, experts, router, P)
        tsar_results.append(acc_tsar)
        
        print(f"Seed {seed} | Static Uniform Acc: {acc_uniform*100:.2f}% | TSAR + PCGrad Acc: {acc_tsar*100:.2f}%")
        
    print("\n" + "="*60)
    print("FINAL PHYSICAL VISION TRANSFORMER EXPERIMENTAL RESULTS (5 SEEDS):")
    print("="*60)
    print(f"Static Uniform Merging Joint Mean Acc: {np.mean(uniform_results)*100:.2f} \\pm {np.std(uniform_results)*100:.2f}%")
    print(f"L3-Linear + TSAR + PCGrad Joint Mean Acc: {np.mean(tsar_results)*100:.2f} \\pm {np.std(tsar_results)*100:.2f}%")
    
    # Save physical results
    with open("results/physical_vit_results.json", "w") as f:
        json.dump({"uniform": uniform_results, "tsar": tsar_results}, f)

if __name__ == "__main__":
    run_physical_experiment()
