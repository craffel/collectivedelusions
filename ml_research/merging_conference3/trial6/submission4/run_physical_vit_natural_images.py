import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as dset
import numpy as np
import random
import os
import json
import timm

# Constants
D = 192       # ViT-Tiny feature dimension
K = 2         # Number of tasks (MNIST, CIFAR-10)
C = 10        # Classes per task
L = 14        # Layers in the router
D_PROJ = 4    # Projection dimension d
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
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.backbone.eval()
        
    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            pooled_feats = self.backbone.forward_head(feats, pre_logits=True)
        return pooled_feats

class L3LinearRouter(nn.Module):
    def __init__(self, L=L, K=K, d=D_PROJ):
        super(L3LinearRouter, self).__init__()
        self.W = nn.Parameter(torch.zeros(L, K, d))
        self.B = nn.Parameter(torch.zeros(L, K))
        
    def forward(self, psi):
        # psi: [B, d]
        # output: [B, L, K]
        alpha = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)
        return alpha

def compute_pca_matrix(X_cal, d=D_PROJ):
    mean = X_cal.mean(dim=0, keepdim=True)
    X_centered = X_cal - mean
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    P = V[:d, :].t()  # [D, d]
    return P

def project_states(X, P):
    X_proj = torch.matmul(X, P)
    norms = torch.norm(X_proj, dim=-1, keepdim=True)
    return X_proj / (norms + 1e-8)

def compute_task_anchors(cal_feats_list, P):
    anchors = []
    for k in range(K):
        X_cal_k = cal_feats_list[k]
        X_sphere_k = project_states(X_cal_k, P)
        anchors.append(X_sphere_k.mean(dim=0))
    return torch.stack(anchors)

def train_router_physical_tsar(cal_feats, experts, router, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=None, epochs=80, lr=1e-2):
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    # Pre-project all calibration samples
    psi_list = []
    for k in range(K):
        psi_list.append(project_states(cal_feats[k][0], P))
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # We will collect task-specific losses and backpropagate using PCGrad
        task_losses = []
        for k in range(K):
            psi_k = psi_list[k]
            y_k = cal_feats[k][1]
            alpha_k = router(psi_k) # [B_cal, L, K]
            bar_alpha_k = alpha_k.mean(dim=1) # [B_cal, K]
            
            # Dynamic weight assembly for this task's mini-batch
            batch_loss = 0.0
            for b in range(psi_k.shape[0]):
                bar_alpha_b = bar_alpha_k[b]
                W_merged = torch.zeros(C, D)
                b_merged = torch.zeros(C)
                for t in range(K):
                    W_merged += bar_alpha_b[t] * experts[t].weight
                    b_merged += bar_alpha_b[t] * experts[t].bias
                
                # Classify
                x_feat_b = cal_feats[k][0][b] # [D]
                logits = torch.matmul(W_merged, x_feat_b) + b_merged
                loss_b = criterion(logits.unsqueeze(0), y_k[b].unsqueeze(0))
                batch_loss += loss_b
                
            batch_loss = batch_loss / psi_k.shape[0]
            
            # Add L2 penalty and TSAR penalty
            l2_penalty = lambda_wd * (torch.sum(router.W ** 2) + torch.sum(router.B ** 2))
            tsar_penalty = 0.0
            if anchors is not None:
                for l in range(L):
                    for t in range(K):
                        tsar_penalty += lambda_anchor * torch.sum((router.W[l, t] - anchors[t]) ** 2)
            
            task_losses.append(batch_loss + l2_penalty + tsar_penalty)
            
        # PCGrad backward
        task_grads = []
        for k in range(K):
            optimizer.zero_grad()
            task_losses[k].backward(retain_graph=True)
            
            # Collect gradients
            grads = []
            for p in router.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1).clone())
            task_grads.append(torch.cat(grads))
            
        # Project conflicting gradients
        final_grads = []
        num_tasks = K
        pcgrad_grads = [g.clone() for g in task_grads]
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    dot_prod = torch.dot(pcgrad_grads[i], task_grads[j])
                    if dot_prod < 0:
                        # Project i's gradient onto normal plane of j's gradient
                        j_norm_sq = torch.dot(task_grads[j], task_grads[j]) + 1e-8
                        pcgrad_grads[i] -= (dot_prod / j_norm_sq) * task_grads[j]
                        
        # Average projected gradients and set parameter grad
        avg_grad = torch.stack(pcgrad_grads).mean(dim=0)
        
        # Write back to router parameters
        idx = 0
        optimizer.zero_grad()
        for p in router.parameters():
            num_el = p.numel()
            p.grad = avg_grad[idx : idx + num_el].view(p.shape).clone()
            idx += num_el
            
        optimizer.step()

def evaluate_merged_model_homogeneous(test_feats, experts, router, P):
    router.eval()
    correct_by_task = {k: 0 for k in range(K)}
    total_by_task = {k: 0 for k in range(K)}
    
    for k in range(K):
        X_test, y_test = test_feats[k]
        psi = project_states(X_test, P)
        
        with torch.no_grad():
            alpha = router(psi) # [B, L, K]
            bar_alpha = alpha.mean(dim=1).mean(dim=0) # [K]
            
            W_merged = torch.zeros(C, D)
            b_merged = torch.zeros(C)
            for t in range(K):
                W_merged += bar_alpha[t] * experts[t].weight
                b_merged += bar_alpha[t] * experts[t].bias
                
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            
            correct_by_task[k] = (preds == y_test).sum().item()
            total_by_task[k] = X_test.shape[0]
            
    accuracies = {k: correct_by_task[k] / (total_by_task[k] + 1e-8) for k in range(K)}
    return accuracies, np.mean(list(accuracies.values()))

def evaluate_static_uniform(test_feats, experts):
    correct_by_task = {k: 0 for k in range(K)}
    total_by_task = {k: 0 for k in range(K)}
    
    for k in range(K):
        X_test, y_test = test_feats[k]
        
        W_merged = torch.zeros(C, D)
        b_merged = torch.zeros(C)
        for t in range(K):
            W_merged += (1.0 / K) * experts[t].weight
            b_merged += (1.0 / K) * experts[t].bias
            
        with torch.no_grad():
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            correct_by_task[k] = (preds == y_test).sum().item()
            total_by_task[k] = X_test.shape[0]
            
    accuracies = {k: correct_by_task[k] / (total_by_task[k] + 1e-8) for k in range(K)}
    return np.mean(list(accuracies.values()))

def load_data_and_extract_features(vit, seed):
    set_seed(seed)
    
    # Resize and normalize transforms
    mnist_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)), # Greyscale to RGB
        T.Normalize((0.1307,), (0.3081,))
    ])
    
    cifar_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load natural image datasets
    mnist_train_set = dset.MNIST('.cache/mnist', train=True, download=False, transform=mnist_transform)
    mnist_test_set = dset.MNIST('.cache/mnist', train=False, download=False, transform=mnist_transform)
    
    cifar_train_set = dset.CIFAR10('.cache/cifar10', train=True, download=False, transform=cifar_transform)
    cifar_test_set = dset.CIFAR10('.cache/cifar10', train=False, download=False, transform=cifar_transform)
    
    # Select subset indices randomly to build splits
    # MNIST Train (200), Calibration (16), Test (100)
    mnist_train_idx = random.sample(range(len(mnist_train_set)), 216)
    mnist_test_idx = random.sample(range(len(mnist_test_set)), 100)
    
    # CIFAR Train (200), Calibration (16), Test (100)
    cifar_train_idx = random.sample(range(len(cifar_train_set)), 216)
    cifar_test_idx = random.sample(range(len(cifar_test_set)), 100)
    
    splits = {"train": [], "test": [], "cal": []}
    
    # Process MNIST
    # Train (200)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_train_set, mnist_train_idx[:200]), batch_size=50)
    feats_train, labels_train = [], []
    for x, y in train_loader:
        feats_train.append(vit(x))
        labels_train.append(y)
    splits["train"].append((torch.cat(feats_train, dim=0), torch.cat(labels_train, dim=0)))
    
    # Cal (16)
    cal_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_train_set, mnist_train_idx[200:]), batch_size=16)
    feats_cal, labels_cal = [], []
    for x, y in cal_loader:
        feats_cal.append(vit(x))
        labels_cal.append(y)
    splits["cal"].append((torch.cat(feats_cal, dim=0), torch.cat(labels_cal, dim=0)))
    
    # Test (100)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_test_set, mnist_test_idx), batch_size=50)
    feats_test, labels_test = [], []
    for x, y in test_loader:
        feats_test.append(vit(x))
        labels_test.append(y)
    splits["test"].append((torch.cat(feats_test, dim=0), torch.cat(labels_test, dim=0)))
    
    # Process CIFAR-10
    # Train (200)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar_train_set, cifar_train_idx[:200]), batch_size=50)
    feats_train, labels_train = [], []
    for x, y in train_loader:
        feats_train.append(vit(x))
        labels_train.append(y)
    splits["train"].append((torch.cat(feats_train, dim=0), torch.cat(labels_train, dim=0)))
    
    # Cal (16)
    cal_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar_train_set, cifar_train_idx[200:]), batch_size=16)
    feats_cal, labels_cal = [], []
    for x, y in cal_loader:
        feats_cal.append(vit(x))
        labels_cal.append(y)
    splits["cal"].append((torch.cat(feats_cal, dim=0), torch.cat(labels_cal, dim=0)))
    
    # Test (100)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(cifar_test_set, cifar_test_idx), batch_size=50)
    feats_test, labels_test = [], []
    for x, y in test_loader:
        feats_test.append(vit(x))
        labels_test.append(y)
    splits["test"].append((torch.cat(feats_test, dim=0), torch.cat(labels_test, dim=0)))
    
    return splits

def run_physical_natural_images_experiment():
    print("Initializing Real pre-trained Vision Transformer (vit_tiny_patch16_224)...")
    vit = PhysicalViTFeatureExtractor()
    
    uniform_results = []
    tsar_results = []
    
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        splits = load_data_and_extract_features(vit, seed)
        
        # Train actual physical Linear Heads
        experts = []
        print("Training physical linear task experts on natural image ViT features...")
        for k in range(K):
            X_train, y_train = splits["train"][k]
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
            
            model.eval()
            with torch.no_grad():
                test_logits = model(splits["test"][k][0])
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == splits["test"][k][1]).float().mean().item()
            print(f"Expert {k} (0=MNIST, 1=CIFAR-10) Test Accuracy: {test_acc*100:.2f}%")
            experts.append(model)
            
        # PCA Projection and Anchors
        all_cal_z = torch.cat([splits["cal"][k][0] for k in range(K)], dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors([splits["cal"][k][0] for k in range(K)], P)
        
        # 1. Evaluate Static Uniform
        acc_uniform = evaluate_static_uniform(splits["test"], experts)
        uniform_results.append(acc_uniform)
        
        # 2. Calibrate TSAR + PCGrad
        router = L3LinearRouter()
        print("Calibrating TSAR + PCGrad router on physical natural representations...")
        train_router_physical_tsar(splits["cal"], experts, router, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=anchors)
        
        # 3. Evaluate TSAR + PCGrad
        _, acc_tsar = evaluate_merged_model_homogeneous(splits["test"], experts, router, P)
        tsar_results.append(acc_tsar)
        
        print(f"Seed {seed} | Static Uniform Acc: {acc_uniform*100:.2f}% | TSAR + PCGrad Acc: {acc_tsar*100:.2f}%")
        
    print("\n" + "="*60)
    print("FINAL NATURAL IMAGE PHYSICAL VIT EXPERIMENTAL RESULTS (5 SEEDS):")
    print("="*60)
    print(f"Static Uniform Merging Joint Mean Acc: {np.mean(uniform_results)*100:.2f} \\pm {np.std(uniform_results)*100:.2f}%")
    print(f"L3-Linear + TSAR + PCGrad Joint Mean Acc: {np.mean(tsar_results)*100:.2f} \\pm {np.std(tsar_results)*100:.2f}%")
    
    # Save results
    with open("results/physical_vit_natural_images_results.json", "w") as f:
        json.dump({"uniform": uniform_results, "tsar": tsar_results}, f)

if __name__ == "__main__":
    run_physical_natural_images_experiment()
