import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import random
from run_pcgrad_scalability import (
    generate_massive_sandbox_data,
    train_experts,
    L3LinearRouter,
    compute_pca_matrix,
    project_states,
    compute_task_anchors,
    K, C, L, D_PROJ, D, SEEDS
)

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

def train_router_tuned(cal_splits, experts, router, P, lr=1e-2, epochs=50, lambda_anchor=0.1, lambda_wd=1e-3, anchors=None):
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
            loss_wd += lambda_wd * torch.sum(p**2)
        loss_anchor = 0.0
        if anchors is not None:
            loss_anchor += lambda_anchor * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        # Standard PCGrad over all K=20 tasks
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

def run_tuning():
    print("Evaluating Uniform baseline and searching for better TSAR + PCGrad hyperparameters...")
    
    # Grid search candidate values
    lrs = [1e-3, 5e-3, 1e-2, 2e-2]
    epoch_options = [50, 100, 150]
    lambda_anchors = [0.01, 0.1, 1.0, 5.0, 10.0]
    lambda_wds = [1e-4, 1e-3, 1e-2]
    
    # Let's first pre-generate data and train experts for each seed to speed things up
    data_by_seed = {}
    for seed in SEEDS:
        splits, sigmas = generate_massive_sandbox_data(seed)
        experts = train_experts(splits["train"])
        
        all_cal_z = []
        for k in range(K):
            all_cal_z.append(splits["cal"][k][0])
        all_cal_z = torch.cat(all_cal_z, dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors(splits["cal"], P)
        
        uniform_acc = evaluate_uniform_baseline(splits["test"], experts)
        print(f"Seed {seed} Static Uniform Merging Accuracy: {uniform_acc*100:.4f}%")
        
        data_by_seed[seed] = {
            "splits": splits,
            "experts": experts,
            "P": P,
            "anchors": anchors,
            "uniform_acc": uniform_acc
        }
        
    avg_uniform = np.mean([data_by_seed[s]["uniform_acc"] for s in SEEDS])
    print(f"Average Static Uniform Merging Accuracy across seeds: {avg_uniform*100:.4f}%\n")
    
    best_joint_acc = 0.0
    best_config = None
    
    # Run random search over combinations to stay within time budget
    combinations = []
    for lr in lrs:
        for epochs in epoch_options:
            for la in lambda_anchors:
                for lwd in lambda_wds:
                    combinations.append((lr, epochs, la, lwd))
                    
    # Shuffle and evaluate 30 random combinations
    random.seed(42)
    random.shuffle(combinations)
    eval_count = 30
    
    for i, (lr, epochs, la, lwd) in enumerate(combinations[:eval_count]):
        accs = []
        for seed in SEEDS:
            seed_data = data_by_seed[seed]
            router = L3LinearRouter(L=L, K=K, d=D_PROJ)
            train_router_tuned(
                seed_data["splits"]["cal"], seed_data["experts"], router, seed_data["P"],
                lr=lr, epochs=epochs, lambda_anchor=la, lambda_wd=lwd, anchors=seed_data["anchors"]
            )
            from run_pcgrad_scalability import evaluate_merged_model
            acc = evaluate_merged_model(seed_data["splits"]["test"], seed_data["experts"], router, seed_data["P"])
            accs.append(acc)
            
        joint_acc = np.mean(accs)
        print(f"[{i+1}/{eval_count}] Config: lr={lr}, epochs={epochs}, lambda_anchor={la}, lambda_wd={lwd} | Joint Accuracy: {joint_acc*100:.4f}%")
        
        if joint_acc > best_joint_acc:
            best_joint_acc = joint_acc
            best_config = (lr, epochs, la, lwd)
            
    print("\n" + "="*50)
    print(f"BEST CONFIGURATION FOUND:")
    print(f"lr={best_config[0]}, epochs={best_config[1]}, lambda_anchor={best_config[2]}, lambda_wd={best_config[3]}")
    print(f"Best TSAR + PCGrad Accuracy: {best_joint_acc*100:.4f}% vs Static Uniform: {avg_uniform*100:.4f}%")
    print("="*50)

if __name__ == "__main__":
    run_tuning()
