import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from test_l1_scalability import L1LinearRouter, evaluate_uniform_baseline, evaluate_merged_model_l1, train_router_l1
from run_pcgrad_scalability import (
    generate_massive_sandbox_data,
    train_experts,
    compute_pca_matrix,
    project_states,
    compute_task_anchors,
    K, C, L, D_PROJ, D, SEEDS
)

def run_sweep():
    print("Starting streamlined L1 sweep on K=20 setup...", flush=True)
    
    # Pre-generate data to save time
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
        
        data_by_seed[seed] = {
            "splits": splits,
            "experts": experts,
            "P": P,
            "anchors": anchors,
            "uniform_acc": uniform_acc
        }
        
    avg_uniform = np.mean([data_by_seed[s]["uniform_acc"] for s in SEEDS])
    print(f"Average Uniform Baseline: {avg_uniform*100:.4f}%\n", flush=True)
    
    # Evaluate 3 specific high-potential combinations with real-time output
    candidates = [
        {"lr": 1e-2, "epochs": 100, "la": 0.5},
        {"lr": 1e-2, "epochs": 100, "la": 1.0},
        {"lr": 1e-2, "epochs": 100, "la": 5.0}
    ]
    
    for cand in candidates:
        lr = cand["lr"]
        epochs = cand["epochs"]
        la = cand["la"]
        
        print(f"Evaluating: lr={lr}, epochs={epochs}, lambda_anchor={la} ...", end="", flush=True)
        accs = []
        for seed in SEEDS:
            seed_data = data_by_seed[seed]
            router = L1LinearRouter(K=K, d=D_PROJ)
            train_router_l1(
                seed_data["splits"]["cal"], seed_data["experts"], router, seed_data["P"],
                scheme="pcgrad", anchors=seed_data["anchors"], epochs=epochs, lr=lr
            )
            acc = evaluate_merged_model_l1(seed_data["splits"]["test"], seed_data["experts"], router, seed_data["P"])
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f" Result: {mean_acc*100:.4f}% \\pm {std_acc*100:.4f}%", flush=True)

if __name__ == "__main__":
    run_sweep()
