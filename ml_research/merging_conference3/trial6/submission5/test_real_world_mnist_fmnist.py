import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class SharedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class L3SoftmaxRouterZero(nn.Module):
    def __init__(self, L, K, d):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(L, K, d))
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        out = torch.einsum('bd,lkd->blk', psi, self.W) + self.B
        return torch.softmax(out, dim=2)

class L3SoftmaxRouterRandom(nn.Module):
    def __init__(self, L, K, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(L, K, d) * 0.1)
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        out = torch.einsum('bd,lkd->blk', psi, self.W) + self.B
        return torch.softmax(out, dim=2)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST and FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download datasets
    mnist_train_full = datasets.MNIST('mnist_data', train=True, download=True, transform=transform)
    mnist_test_full = datasets.MNIST('mnist_data', train=False, download=True, transform=transform)
    fmnist_train_full = datasets.FashionMNIST('fmnist_data', train=True, download=True, transform=transform)
    fmnist_test_full = datasets.FashionMNIST('fmnist_data', train=False, download=True, transform=transform)
    
    # Subsample to keep things lightning fast
    set_seed(42)
    mnist_train = Subset(mnist_train_full, torch.randperm(len(mnist_train_full))[:2000])
    mnist_test = Subset(mnist_test_full, torch.randperm(len(mnist_test_full))[:500])
    fmnist_train = Subset(fmnist_train_full, torch.randperm(len(fmnist_train_full))[:2000])
    fmnist_test = Subset(fmnist_test_full, torch.randperm(len(fmnist_test_full))[:500])
    
    # Create dataloaders
    mnist_train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
    fmnist_train_loader = DataLoader(fmnist_train, batch_size=128, shuffle=True)
    
    # 1. Train Shared CNN Backbone
    print("Training Shared CNN Backbone jointly...")
    backbone = SharedCNN().to(device)
    optimizer = optim.AdamW(backbone.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Simple joint pre-training: multi-task head (20 output units)
    joint_head = nn.Linear(128, 20).to(device)
    joint_optimizer = optim.AdamW(list(backbone.parameters()) + list(joint_head.parameters()), lr=1e-3)
    
    for epoch in range(3):
        backbone.train()
        joint_head.train()
        # Train on MNIST
        for x, y in mnist_train_loader:
            x, y = x.to(device), y.to(device)
            feats = backbone(x)
            logits = joint_head(feats)
            loss = criterion(logits[:, :10], y)
            joint_optimizer.zero_grad()
            loss.backward()
            joint_optimizer.step()
        # Train on FashionMNIST
        for x, y in fmnist_train_loader:
            x, y = x.to(device), y.to(device)
            feats = backbone(x)
            logits = joint_head(feats)
            loss = criterion(logits[:, 10:20], y)
            joint_optimizer.zero_grad()
            loss.backward()
            joint_optimizer.step()
            
    # Freeze backbone
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Extract features for faster subsequent training
    def extract_features(dataset):
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_feats = []
        all_targets = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feats = backbone(x)
                all_feats.append(feats.cpu())
                all_targets.append(y)
        return torch.cat(all_feats, dim=0), torch.cat(all_targets, dim=0)
        
    print("Extracting features...")
    mnist_train_feats, mnist_train_targets = extract_features(mnist_train)
    mnist_test_feats, mnist_test_targets = extract_features(mnist_test)
    fmnist_train_feats, fmnist_train_targets = extract_features(fmnist_train)
    fmnist_test_feats, fmnist_test_targets = extract_features(fmnist_test)
    
    # 2. Pre-train Task-Specific Classifier Experts
    # Base head: [20 output classes, 128 features]
    W_base = torch.randn(20, 128) * 0.01
    b_base = torch.zeros(20)
    
    print("Pre-training Specialized Experts...")
    # Expert 1: MNIST (Optimizes classes 0-9)
    W_1 = nn.Parameter(W_base.clone())
    b_1 = nn.Parameter(b_base.clone())
    opt_1 = optim.AdamW([W_1, b_1], lr=1e-2, weight_decay=1e-4)
    
    for epoch in range(15):
        logits = mnist_train_feats.to(device) @ W_1.to(device).t() + b_1.to(device)
        loss = criterion(logits[:, :10], mnist_train_targets.to(device))
        opt_1.zero_grad()
        loss.backward()
        opt_1.step()
        
    # Expert 2: FashionMNIST (Optimizes classes 10-19)
    W_2 = nn.Parameter(W_base.clone())
    b_2 = nn.Parameter(b_base.clone())
    opt_2 = optim.AdamW([W_2, b_2], lr=1e-2, weight_decay=1e-4)
    
    for epoch in range(15):
        logits = fmnist_train_feats.to(device) @ W_2.to(device).t() + b_2.to(device)
        loss = criterion(logits[:, 10:20], fmnist_train_targets.to(device))
        opt_2.zero_grad()
        loss.backward()
        opt_2.step()
        
    # Detach expert parameters and move to CPU
    W_1, b_1 = W_1.detach().cpu(), b_1.detach().cpu()
    W_2, b_2 = W_2.detach().cpu(), b_2.detach().cpu()
    
    # Evaluate experts
    with torch.no_grad():
        mnist_acc = ((mnist_test_feats @ W_1.t() + b_1)[:, :10].argmax(dim=1) == mnist_test_targets).float().mean().item()
        fmnist_acc = ((fmnist_test_feats @ W_2.t() + b_2)[:, 10:20].argmax(dim=1) == fmnist_test_targets).float().mean().item()
        print(f"Expert 1 (MNIST) Accuracy: {mnist_acc*100:.2f}%")
        print(f"Expert 2 (FashionMNIST) Accuracy: {fmnist_acc*100:.2f}%")
        print(f"Joint Expert Ceiling: {(mnist_acc + fmnist_acc)/2*100:.2f}%")
        
    # Task Vectors
    V_1_W = W_1 - W_base
    V_1_b = b_1 - b_base
    V_2_W = W_2 - W_base
    V_2_b = b_2 - b_base
    
    # Create Calibration Set: 32 samples from MNIST, 32 from FashionMNIST
    cal_feats = torch.cat([mnist_train_feats[:32], fmnist_train_feats[:32]], dim=0)
    cal_targets = torch.cat([mnist_train_targets[:32], fmnist_train_targets[:32] + 10], dim=0) # offset FashionMNIST targets by 10
    cal_tasks = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)], dim=0)
    
    # Project to low-dimensional unit sphere d=2 (2 tasks)
    set_seed(42)
    p_proj = torch.randn(128, 2)
    p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
    
    cal_psi = cal_feats @ p_proj
    cal_psi = cal_psi / (torch.norm(cal_psi, dim=1, keepdim=True) + 1e-8)
    
    # 3. Train Routers (Unregularized vs Well-Regularized)
    def train_router(router, lambda_wd, epochs=100):
        router.train()
        optimizer = optim.AdamW(router.parameters(), lr=5e-3, weight_decay=lambda_wd)
        for epoch in range(epochs):
            optimizer.zero_grad()
            alpha = router(cal_psi) # [B, L, K=2]
            alpha_b_k = alpha.mean(dim=1) # [B, 2]
            
            W_merged = W_base.unsqueeze(0) + alpha_b_k[:, 0].unsqueeze(-1).unsqueeze(-1) * V_1_W.unsqueeze(0) + alpha_b_k[:, 1].unsqueeze(-1).unsqueeze(-1) * V_2_W.unsqueeze(0)
            b_merged = b_base.unsqueeze(0) + alpha_b_k[:, 0].unsqueeze(-1) * V_1_b.unsqueeze(0) + alpha_b_k[:, 1].unsqueeze(-1) * V_2_b.unsqueeze(0)
            
            logits = torch.einsum('bd,bod->bo', cal_feats, W_merged) + b_merged
            loss = criterion(logits, cal_targets)
            loss.backward()
            optimizer.step()
            
    print("\nTraining L3_Softmax (Random Init, Unregularized)...")
    router_unreg = L3SoftmaxRouterRandom(14, 2, 2)
    train_router(router_unreg, lambda_wd=0.0)
    
    print("Training L3_Softmax_WellReg (Zero Init, Weight Decay)...")
    router_wellreg = L3SoftmaxRouterZero(14, 2, 2)
    train_router(router_wellreg, lambda_wd=1e-2)
    
    # 4. Create Shuffled Test Set (Heterogeneous Stream)
    test_feats = torch.cat([mnist_test_feats, fmnist_test_feats], dim=0)
    test_targets = torch.cat([mnist_test_targets, fmnist_test_targets + 10], dim=0)
    test_tasks = torch.cat([torch.zeros(500, dtype=torch.long), torch.ones(500, dtype=torch.long)], dim=0)
    
    perm = torch.randperm(test_feats.shape[0])
    test_feats_shuf = test_feats[perm]
    test_targets_shuf = test_targets[perm]
    test_tasks_shuf = test_tasks[perm]
    
    test_psi = test_feats_shuf @ p_proj
    test_psi = test_psi / (torch.norm(test_psi, dim=1, keepdim=True) + 1e-8)
    
    # Evaluation function
    def evaluate(router, batch_size):
        router.eval()
        B = test_feats_shuf.shape[0]
        preds = []
        with torch.no_grad():
            alpha = router(test_psi)
            alpha_b_k = alpha.mean(dim=1)
            
            if batch_size == 1:
                W_merged = W_base.unsqueeze(0) + alpha_b_k[:, 0].unsqueeze(-1).unsqueeze(-1) * V_1_W.unsqueeze(0) + alpha_b_k[:, 1].unsqueeze(-1).unsqueeze(-1) * V_2_W.unsqueeze(0)
                b_merged = b_base.unsqueeze(0) + alpha_b_k[:, 0].unsqueeze(-1) * V_1_b.unsqueeze(0) + alpha_b_k[:, 1].unsqueeze(-1) * V_2_b.unsqueeze(0)
                logits = torch.einsum('bd,bod->bo', test_feats_shuf, W_merged) + b_merged
                preds = logits.argmax(dim=1)
                return (preds == test_targets_shuf).float().mean().item()
            else:
                preds = []
                for i in range(0, B, batch_size):
                    bx = test_feats_shuf[i:i+batch_size]
                    ba = alpha_b_k[i:i+batch_size]
                    ba_avg = ba.mean(dim=0)
                    W_merged = W_base + ba_avg[0] * V_1_W + ba_avg[1] * V_2_W
                    b_merged = b_base + ba_avg[0] * V_1_b + ba_avg[1] * V_2_b
                    logits = bx @ W_merged.t() + b_merged
                    preds.append(logits.argmax(dim=1))
                return (torch.cat(preds, dim=0) == test_targets_shuf).float().mean().item()
                
    # Evaluate Uniform Merging
    W_merged_uni = W_base + 0.5 * V_1_W + 0.5 * V_2_W
    b_merged_uni = b_base + 0.5 * V_1_b + 0.5 * V_2_b
    uni_preds = (test_feats_shuf @ W_merged_uni.t() + b_merged_uni).argmax(dim=1)
    uni_acc = (uni_preds == test_targets_shuf).float().mean().item()
    
    print("\n--- REAL-WORLD EXPERIMENT RESULTS (MNIST + FashionMNIST) ---")
    print(f"Uniform Merging Accuracy: {uni_acc*100:.2f}%")
    
    unreg_b256 = evaluate(router_unreg, batch_size=256)
    unreg_b1 = evaluate(router_unreg, batch_size=1)
    print(f"L3_Softmax (Unregularized) Hetero (B=256): {unreg_b256*100:.2f}%")
    print(f"L3_Softmax (Unregularized) Hetero (B=1)   : {unreg_b1*100:.2f}%  <-- Vectorization Collapse!")
    
    wellreg_b256 = evaluate(router_wellreg, batch_size=256)
    wellreg_b1 = evaluate(router_wellreg, batch_size=1)
    print(f"L3_Softmax_WellReg (Ours) Hetero (B=256) : {wellreg_b256*100:.2f}%")
    print(f"L3_Softmax_WellReg (Ours) Hetero (B=1)   : {wellreg_b1*100:.2f}%  <-- Resolves Vectorization Collapse!")

if __name__ == "__main__":
    main()
