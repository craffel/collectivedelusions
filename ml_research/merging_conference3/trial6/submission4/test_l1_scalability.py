import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import random

# Massive Multi-Task Sandbox Configuration
D = 200    
K = 20     
C = 10     
L_14 = 14     
D_PROJ = 20 
SEEDS = [10, 11]

from run_pcgrad_scalability import (
    generate_massive_sandbox_data,
    train_experts,
    compute_pca_matrix,
    project_states,
    compute_task_anchors
)

class L1LinearRouter(nn.Module):
    def __init__(self, K=K, d=D_PROJ):
        super(L1LinearRouter, self).__init__()
        self.W = nn.Parameter(torch.zeros(1, K, d)) # L=1
        self.B = nn.Parameter(torch.zeros(1, K))
    def forward(self, psi):
        alpha = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)
        return alpha

def evaluate_uniform_baseline(test_splits, experts):
    task_accs = []
    for k in range(K):
        X_test, y_test = test_splits[k]
        num_task_samples = X_test.shape[0]
        correct = 0
        total = 0
        with torch.no_grad():
            bar_alpha = torch.ones(K) / K
            W_merged = torch.zeros(C, D)
            b_merged = torch.zeros(C)
            for t in range(K):
                W_merged += bar_alpha[t] * experts[t].weight
                b_merged += bar_alpha[t] * experts[t].bias
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_test).sum().item()
            total += X_test.shape[0]
        task_accs.append(correct / total)
    return np.mean(task_accs)

def evaluate_merged_model_l1(test_splits, experts, router, P):
    router.eval()
    task_accs = []
    for k in range(K):
        X_test, y_test = test_splits[k]
        num_task_samples = X_test.shape[0]
        correct = 0
        total = 0
        with torch.no_grad():
            psi = project_states(X_test, P)
            alpha = router(psi) # [B, 1, K]
            alpha_avg = alpha.mean(dim=1) # [B, K]
            bar_alpha = alpha_avg.mean(dim=0) # [K]
            
            W_merged = torch.zeros(C, D)
            b_merged = torch.zeros(C)
            for t in range(K):
                W_merged += bar_alpha[t] * experts[t].weight
                b_merged += bar_alpha[t] * experts[t].bias
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_test).sum().item()
            total += X_test.shape[0]
        task_accs.append(correct / total)
    return np.mean(task_accs)

def train_router_l1(cal_splits, experts, router, P, scheme="none", anchors=None, epochs=50, lr=1e-2):
    all_cal_z = []
    all_cal_y = []
    all_cal_tasks = []
    for k in range(K):
        X_cal, y_cal = cal_splits[k]
        all_cal_z.append(X_cal)
        all_cal_y.append(y_cal)
        all_cal_tasks.append(torch.ones(X_cal.shape[0], dtype=torch.long) * k)
    all_cal_z = torch.cat(all_cal_z, dim=0)
    all_cal_y = torch.cat(all_cal_y, dim=0)
    all_cal_tasks = torch.cat(all_cal_tasks, dim=0)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    router.train()
    params = list(router.parameters())
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        psi = project_states(all_cal_z, P)
        alpha = router(psi)
        alpha_avg = alpha.mean(dim=1)
        bar_alpha = alpha_avg.mean(dim=0)
        
        W_merged = torch.zeros(C, D)
        b_merged = torch.zeros(C)
        for k in range(K):
            W_merged += bar_alpha[k] * experts[k].weight
            b_merged += bar_alpha[k] * experts[k].bias
        logits = torch.matmul(all_cal_z, W_merged.t()) + b_merged
        
        loss_wd = 0.0
        for p in router.parameters():
            loss_wd += 1e-3 * torch.sum(p**2)
        loss_anchor = 0.0
        if anchors is not None:
            loss_anchor += 0.1 * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        if scheme == "none":
            loss_ce = criterion(logits, all_cal_y)
            loss_total = loss_ce + loss_wd + loss_anchor
            loss_total.backward()
            optimizer.step()
        elif scheme == "pcgrad":
            grads = []
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

def run_l1_eval():
    print("Evaluating Single-Layer Global Router (L=1) on K=20 setup...")
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        splits, sigmas = generate_massive_sandbox_data(seed)
        experts = train_experts(splits["train"])
        
        # PCA matrix projection
        all_cal_z = []
        for k in range(K):
            all_cal_z.append(splits["cal"][k][0])
        all_cal_z = torch.cat(all_cal_z, dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors(splits["cal"], P)
        
        uniform_acc = evaluate_uniform_baseline(splits["test"], experts)
        print(f"Static Uniform Merging Accuracy: {uniform_acc*100:.2f}%")
        
        # Evaluate L1 without PCGrad
        router_none = L1LinearRouter(K=K, d=D_PROJ)
        train_router_l1(splits["cal"], experts, router_none, P, scheme="none", anchors=anchors, epochs=50)
        acc_none = evaluate_merged_model_l1(splits["test"], experts, router_none, P)
        print(f"L1-Linear + TSAR (No PCGrad) Accuracy: {acc_none*100:.2f}%")
        
        # Evaluate L1 with PCGrad
        router_pcgrad = L1LinearRouter(K=K, d=D_PROJ)
        train_router_l1(splits["cal"], experts, router_pcgrad, P, scheme="pcgrad", anchors=anchors, epochs=50)
        acc_pcgrad = evaluate_merged_model_l1(splits["test"], experts, router_pcgrad, P)
        print(f"L1-Linear + TSAR + PCGrad Accuracy: {acc_pcgrad*100:.2f}%")

if __name__ == "__main__":
    run_l1_eval()
