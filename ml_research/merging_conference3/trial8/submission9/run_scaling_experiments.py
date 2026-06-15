import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Global Constants
D = 192          # representation dimension
C = 10           # number of classes per task
RANK = 8         # LoRA rank
NUM_LAYERS = 9   # middle-to-late blocks
DEVICE = torch.device("cpu")
SEEDS = [10, 11, 12, 13, 14]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ExpertModel(nn.Module):
    def __init__(self, d_in=D, rank=RANK, num_classes=C, num_layers=NUM_LAYERS):
        super(ExpertModel, self).__init__()
        self.num_layers = num_layers
        self.rank = rank
        
        # Trainable LoRA adapters
        self.A = nn.ParameterList([nn.Parameter(torch.randn(d_in, rank) * 0.01) for _ in range(num_layers)])
        self.B = nn.ParameterList([nn.Parameter(torch.randn(rank, d_in) * 0.01) for _ in range(num_layers)])
        
        # Classification head
        self.head = nn.Linear(d_in, num_classes)
        
    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            h = h + torch.matmul(torch.matmul(h, self.A[i]), self.B[i])
        logits = self.head(h)
        return logits, h

def generate_datasets(seed, K):
    set_seed(seed)
    subspace_size = D // K
    
    prototypes = {}
    for k in range(K):
        random_matrix = np.random.randn(subspace_size, C)
        q, _ = np.linalg.qr(random_matrix)
        q = q.T  # shape (C, subspace_size)
        
        task_prototypes = np.zeros((C, D))
        task_prototypes[:, k*subspace_size : (k+1)*subspace_size] = q
        prototypes[k] = torch.tensor(task_prototypes, dtype=torch.float32)
        
    # Interpolated noise levels
    noise_levels = {k: float(np.linspace(0.01, 0.56, K)[k]) for k in range(K)}
    
    train_data = {}
    cal_data = {}
    test_data = {}
    
    for k in range(K):
        train_x, train_y = [], []
        for i in range(1000):
            c = i % C
            noise = torch.randn(D) * noise_levels[k]
            sample = prototypes[k][c] + noise
            train_x.append(sample)
            train_y.append(c)
        train_data[k] = (torch.stack(train_x), torch.tensor(train_y))
        
        cal_x, cal_y = [], []
        for i in range(64):
            c = i % C
            noise = torch.randn(D) * noise_levels[k]
            sample = prototypes[k][c] + noise
            cal_x.append(sample)
            cal_y.append(c)
        cal_data[k] = (torch.stack(cal_x), torch.tensor(cal_y))
        
        test_x, test_y = [], []
        for i in range(250):
            c = i % C
            noise = torch.randn(D) * noise_levels[k]
            sample = prototypes[k][c] + noise
            test_x.append(sample)
            test_y.append(c)
        test_data[k] = (torch.stack(test_x), torch.tensor(test_y))
        
    return train_data, cal_data, test_data, prototypes

def run_evaluation_for_K(K):
    print(f"\n==========================================")
    print(f"EVALUATING SCALING REGISTRY FOR K = {K}")
    print(f"==========================================")
    
    all_sps_zca = []
    all_uniform = []
    all_eer = []
    all_epl_oca_hard = []
    all_epl_oca_soft = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        train_data, cal_data, test_data, prototypes = generate_datasets(seed, K)
        
        # 1. Train K Experts
        experts = {}
        for k in range(K):
            model = ExpertModel().to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            x, y = train_data[k]
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            for epoch in range(60):
                model.train()
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            experts[k] = model
            
        # Prepare streams
        homo_x = []
        homo_y = []
        homo_tasks = []
        for k in range(K):
            test_x, test_y = test_data[k]
            homo_x.append(test_x)
            homo_y.append(test_y)
            homo_tasks.extend([k]*len(test_x))
        homo_x = torch.cat(homo_x).to(DEVICE)
        homo_y = torch.cat(homo_y).to(DEVICE)
        
        set_seed(seed + 100)
        shuffled_idx = torch.randperm(len(homo_x))
        hete_x = homo_x[shuffled_idx]
        hete_y = homo_y[shuffled_idx]
        hete_tasks = np.array(homo_tasks)[shuffled_idx.numpy()]
        
        # 2. Static Uniform Merging
        correct_unif = 0
        total_unif = len(hete_x)
        with torch.no_grad():
            h = hete_x
            for i in range(NUM_LAYERS):
                update = sum([torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)]) / K
                h = h + update
            head_w = sum([experts[j].head.weight for j in range(K)]) / K
            head_b = sum([experts[j].head.bias for j in range(K)]) / K
            logits = torch.matmul(h, head_w.t()) + head_b
            preds = logits.argmax(dim=1)
            correct_unif = (preds == hete_y).sum().item()
        unif_acc = correct_unif / total_unif
        all_uniform.append(unif_acc)
        
        # 3. SPS-ZCA Baseline (Offline)
        zca_centroids = {}
        for k in range(K):
            cal_x, _ = cal_data[k]
            cal_x = cal_x.to(DEVICE)
            zca_centroids[k] = cal_x.mean(dim=0)
            zca_centroids[k] = zca_centroids[k] / (zca_centroids[k].norm(p=2) + 1e-8)
            
        correct_zca = 0
        with torch.no_grad():
            for b in range(len(hete_x)):
                x = hete_x[b:b+1]
                y = hete_y[b:b+1]
                
                u = torch.zeros(K)
                for k in range(K):
                    u[k] = torch.dot(x[0], zca_centroids[k]) / (x[0].norm(p=2) * zca_centroids[k].norm(p=2) + 1e-8)
                
                tau = 0.001
                alpha = torch.softmax(u / tau, dim=0)
                
                h = x
                for i in range(NUM_LAYERS):
                    update = sum([alpha[j] * torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)])
                    h = h + update
                head_w = sum([alpha[j] * experts[j].head.weight for j in range(K)])
                head_b = sum([alpha[j] * experts[j].head.bias for j in range(K)])
                logits = torch.matmul(h, head_w.t()) + head_b
                pred = logits.argmax(dim=1)
                correct_zca += (pred == y).item()
        zca_acc = correct_zca / len(hete_x)
        all_sps_zca.append(zca_acc)
        
        # 4. EER Ours
        correct_eer = 0
        with torch.no_grad():
            for b in range(len(hete_x)):
                x = hete_x[b:b+1]
                y = hete_y[b:b+1]
                
                entropy_vals = torch.zeros(K)
                for k in range(K):
                    logits_k, _ = experts[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    # Normalize Shannon Entropy to neutralize vocab bias
                    entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
                    
                k_star = entropy_vals.argmin().item()
                logits_star, _ = experts[k_star](x)
                pred = logits_star.argmax(dim=1)
                correct_eer += (pred == y).item()
        eer_acc = correct_eer / len(hete_x)
        all_eer.append(eer_acc)
        
        # 5. EPL-OCA Ours (Hard and Soft)
        T_warmup = 200
        running_centroids = torch.zeros(K, D)
        centroid_counts = torch.zeros(K)
        beta = 0.1
        correct_oca_hard = 0
        correct_oca_soft = 0
        total_oca = 0
        
        with torch.no_grad():
            for b in range(len(hete_x)):
                x = hete_x[b:b+1]
                y = hete_y[b:b+1]
                
                if b >= T_warmup:
                    u = torch.zeros(K)
                    for k in range(K):
                        c_v = running_centroids[k] if centroid_counts[k] > 0 else x[0]
                        u[k] = torch.dot(x[0], c_v) / (x[0].norm(p=2) * c_v.norm(p=2) + 1e-8)
                    
                    # Hard prediction (tau = 0.001)
                    alpha_hard = torch.softmax(u / 0.001, dim=0)
                    h_hard = x
                    for i in range(NUM_LAYERS):
                        update = sum([alpha_hard[j] * torch.matmul(torch.matmul(h_hard, experts[j].A[i]), experts[j].B[i]) for j in range(K)])
                        h_hard = h_hard + update
                    head_w_hard = sum([alpha_hard[j] * experts[j].head.weight for j in range(K)])
                    head_b_hard = sum([alpha_hard[j] * experts[j].head.bias for j in range(K)])
                    logits_hard = torch.matmul(h_hard, head_w_hard.t()) + head_b_hard
                    pred_hard = logits_hard.argmax(dim=1)
                    correct_oca_hard += (pred_hard == y).item()
                    
                    # Soft prediction (tau = 0.5)
                    alpha_soft = torch.softmax(u / 0.5, dim=0)
                    h_soft = x
                    for i in range(NUM_LAYERS):
                        update = sum([alpha_soft[j] * torch.matmul(torch.matmul(h_soft, experts[j].A[i]), experts[j].B[i]) for j in range(K)])
                        h_soft = h_soft + update
                    head_w_soft = sum([alpha_soft[j] * experts[j].head.weight for j in range(K)])
                    head_b_soft = sum([alpha_soft[j] * experts[j].head.bias for j in range(K)])
                    logits_soft = torch.matmul(h_soft, head_w_soft.t()) + head_b_soft
                    pred_soft = logits_soft.argmax(dim=1)
                    correct_oca_soft += (pred_soft == y).item()
                    
                    total_oca += 1
                    
                # Pseudo-Label update
                entropy_vals = torch.zeros(K)
                for k in range(K):
                    logits_k, _ = experts[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
                k_star = entropy_vals.argmin().item()
                
                if centroid_counts[k_star] == 0:
                    running_centroids[k_star] = x[0]
                    centroid_counts[k_star] = 1
                else:
                    running_centroids[k_star] = (1 - beta) * running_centroids[k_star] + beta * x[0]
                running_centroids[k_star] = running_centroids[k_star] / (running_centroids[k_star].norm(p=2) + 1e-8)
                
        oca_hard_acc = correct_oca_hard / total_oca
        oca_soft_acc = correct_oca_soft / total_oca
        all_epl_oca_hard.append(oca_hard_acc)
        all_epl_oca_soft.append(oca_soft_acc)
        
        print(f"Seed {seed} | Uniform: {unif_acc*100:.2f}% | SPS-ZCA: {zca_acc*100:.2f}% | EER: {eer_acc*100:.2f}% | EPL-OCA Hard: {oca_hard_acc*100:.2f}% | EPL-OCA Soft: {oca_soft_acc*100:.2f}%")
        
    print(f"\nSummary for K = {K}:")
    print(f"Uniform:       {np.mean(all_uniform)*100:.2f} ± {np.std(all_uniform)*100:.2f}%")
    print(f"SPS-ZCA:       {np.mean(all_sps_zca)*100:.2f} ± {np.std(all_sps_zca)*100:.2f}%")
    print(f"EER:           {np.mean(all_eer)*100:.2f} ± {np.std(all_eer)*100:.2f}%")
    print(f"EPL-OCA Hard:  {np.mean(all_epl_oca_hard)*100:.2f} ± {np.std(all_epl_oca_hard)*100:.2f}%")
    print(f"EPL-OCA Soft:  {np.mean(all_epl_oca_soft)*100:.2f} ± {np.std(all_epl_oca_soft)*100:.2f}%")
    
    return {
        "uniform": (np.mean(all_uniform), np.std(all_uniform)),
        "sps_zca": (np.mean(all_sps_zca), np.std(all_sps_zca)),
        "eer": (np.mean(all_eer), np.std(all_eer)),
        "epl_oca_hard": (np.mean(all_epl_oca_hard), np.std(all_epl_oca_hard)),
        "epl_oca_soft": (np.mean(all_epl_oca_soft), np.std(all_epl_oca_soft))
    }

if __name__ == "__main__":
    K_vals = [4, 8, 12]
    results = {}
    for K in K_vals:
        results[K] = run_evaluation_for_K(K)
        
    # Generate Plot
    plt.figure(figsize=(9, 5))
    x_indices = np.arange(len(K_vals))
    width = 0.15
    
    unif_means = [results[K]["uniform"][0] * 100 for K in K_vals]
    unif_stds = [results[K]["uniform"][1] * 100 for K in K_vals]
    
    zca_means = [results[K]["sps_zca"][0] * 100 for K in K_vals]
    zca_stds = [results[K]["sps_zca"][1] * 100 for K in K_vals]
    
    eer_means = [results[K]["eer"][0] * 100 for K in K_vals]
    eer_stds = [results[K]["eer"][1] * 100 for K in K_vals]
    
    hard_means = [results[K]["epl_oca_hard"][0] * 100 for K in K_vals]
    hard_stds = [results[K]["epl_oca_hard"][1] * 100 for K in K_vals]
    
    soft_means = [results[K]["epl_oca_soft"][0] * 100 for K in K_vals]
    soft_stds = [results[K]["epl_oca_soft"][1] * 100 for K in K_vals]
    
    plt.bar(x_indices - 2.0*width, unif_means, width, yerr=unif_stds, label="Uniform Merging", color="gray", capsize=4)
    plt.bar(x_indices - 1.0*width, zca_means, width, yerr=zca_stds, label="SPS-ZCA (Offline Baseline)", color="red", capsize=4)
    plt.bar(x_indices, hard_means, width, yerr=hard_stds, label="EPL-OCA Hard (Ours, tau=0.001)", color="pink", capsize=4)
    plt.bar(x_indices + 1.0*width, soft_means, width, yerr=soft_stds, label="EPL-OCA Soft (Ours, tau=0.5)", color="orange", capsize=4)
    plt.bar(x_indices + 2.0*width, eer_means, width, yerr=eer_stds, label="EER (Ours, Direct)", color="blue", capsize=4)
    
    plt.xlabel("Registry Scale (Number of Tasks $K$)", fontsize=12)
    plt.ylabel("Joint Mean Test Accuracy (%)", fontsize=12)
    plt.title("Registry Scalability: Direct Entropy Routing vs. Centroid-Based Ensembling", fontsize=13)
    plt.xticks(x_indices, [f"K = {K}" for K in K_vals])
    plt.grid(True, axis="y", ls="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fig4_registry_scaling.png", dpi=300)
    print("Scaling plot saved to results/fig4_registry_scaling.png")
    
    # Save a markdown table to results/scaling_results.md
    with open("results/scaling_results.md", "w") as f:
        f.write("# Registry Scaling Results\n\n")
        f.write("| Method | K = 4 | K = 8 | K = 12 |\n")
        f.write("|---|---|---|---|\n")
        
        methods = [
            ("Uniform Merging", "uniform"),
            ("SPS-ZCA (Offline SOTA)", "sps_zca"),
            ("EPL-OCA Hard (Ours, Centroid, tau=0.001)", "epl_oca_hard"),
            ("EPL-OCA Soft (Ours, Centroid, tau=0.5)", "epl_oca_soft"),
            ("EER (Ours, Direct)", "eer")
        ]
        
        for name, key in methods:
            row = f"| {name} "
            for K in K_vals:
                mean, std = results[K][key]
                row += f"| {mean*100:.2f}% ± {std*100:.2f}% "
            row += "|\n"
            f.write(row)
            
    print("Markdown table saved to results/scaling_results.md")
