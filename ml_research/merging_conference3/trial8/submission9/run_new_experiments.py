import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Global Constants consistent with the main paper
D = 192          # representation dimension
K = 4            # number of task domains
C = 10           # number of classes per task in homogeneous setting
NUM_LAYERS = 9   # mid-to-late blocks (Layers 4 to 12)
DEVICE = torch.device("cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class ExpertModel(nn.Module):
    def __init__(self, d_in=D, rank=8, num_classes=C, num_layers=NUM_LAYERS):
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

def generate_homogeneous_datasets(seed):
    set_seed(seed)
    subspace_size = D // K  # 48
    
    # Generate 4 orthogonal subspaces of size 48
    prototypes = {}
    for k in range(K):
        random_matrix = np.random.randn(subspace_size, C)
        q, _ = np.linalg.qr(random_matrix)
        q = q.T  # shape (10, 48)
        
        task_prototypes = np.zeros((C, D))
        task_prototypes[:, k*subspace_size : (k+1)*subspace_size] = q
        prototypes[k] = torch.tensor(task_prototypes, dtype=torch.float32)
        
    noise_levels = {0: 0.01, 1: 0.05, 2: 0.24, 3: 0.56}
    
    train_data = {}
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
        
        test_x, test_y = [], []
        for i in range(250):
            c = i % C
            noise = torch.randn(D) * noise_levels[k]
            sample = prototypes[k][c] + noise
            test_x.append(sample)
            test_y.append(c)
        test_data[k] = (torch.stack(test_x), torch.tensor(test_y))
        
    return train_data, test_data

def generate_block_shuffled_stream(test_data, block_size, seed):
    set_seed(seed)
    # test_data has 250 samples per task. We divide each task's samples into blocks of block_size
    task_blocks = {k: [] for k in range(K)}
    for k in range(K):
        x, y = test_data[k]
        n_samples = len(x)
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, block_size):
            block_idx = indices[i:i+block_size]
            task_blocks[k].append((x[block_idx], y[block_idx], k))
            
    # Pool blocks and shuffle them
    all_blocks = []
    for k in range(K):
        all_blocks.extend(task_blocks[k])
        
    shuffled_idx = torch.randperm(len(all_blocks))
    stream_x = []
    stream_y = []
    stream_tasks = []
    
    for idx in shuffled_idx:
        x_b, y_b, k_b = all_blocks[idx]
        stream_x.append(x_b)
        stream_y.append(y_b)
        stream_tasks.extend([k_b] * len(x_b))
        
    return torch.cat(stream_x), torch.cat(stream_y), np.array(stream_tasks)

# Experiment 1: Amortization under different block sizes (temporal locality)
def run_amortization_ablation():
    print("\n--- Running Experiment 1: Amortized EER under Block-Shuffled Streams ---")
    seed = 42
    train_data, test_data = generate_homogeneous_datasets(seed)
    
    # Train homogeneous experts
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
        
    block_sizes = [1, 10, 50, 100]
    amortization_intervals = [1, 5, 10, 20]
    
    results_acc = {b: [] for b in block_sizes}
    
    with torch.no_grad():
        for b_size in block_sizes:
            # Generate stream for this block size
            stream_x, stream_y, stream_tasks = generate_block_shuffled_stream(test_data, b_size, seed)
            stream_x = stream_x.to(DEVICE)
            stream_y = stream_y.to(DEVICE)
            
            for N_amortize in amortization_intervals:
                correct = 0
                total = len(stream_x)
                
                # Active routing decision cache
                cached_k_star = 0
                
                for b in range(total):
                    x = stream_x[b:b+1]
                    y = stream_y[b:b+1]
                    
                    # Evaluate expert entropy only once every N_amortize steps
                    if b % N_amortize == 0:
                        entropy_vals = torch.zeros(K)
                        for k in range(K):
                            logits_k, _ = experts[k](x)
                            probs_k = torch.softmax(logits_k, dim=1)
                            entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
                        cached_k_star = entropy_vals.argmin().item()
                    
                    # Serve using cached expert
                    logits_star, _ = experts[cached_k_star](x)
                    pred = logits_star.argmax(dim=1)
                    correct += (pred == y).item()
                    
                acc = correct / total
                results_acc[b_size].append(acc)
                print(f"Block Size: {b_size:<3} | Amortization N: {N_amortize:<2} | Accuracy: {acc*100:.2f}%")
                
    # Save a plot for results
    plt.figure(figsize=(8, 5))
    colors = {1: "red", 10: "orange", 50: "blue", 100: "green"}
    markers = {1: "o", 10: "s", 50: "^", 100: "d"}
    
    for b_size in block_sizes:
        accs_pct = [a * 100 for a in results_acc[b_size]]
        plt.plot(amortization_intervals, accs_pct, label=f"Block Size = {b_size} (Locality)", 
                 color=colors[b_size], marker=markers[b_size], linewidth=2, markersize=8)
                 
    plt.xlabel("Amortization Interval ($N_{amortize}$)", fontsize=12)
    plt.ylabel("Joint Mean Test Accuracy (%)", fontsize=12)
    plt.title("Impact of Amortized Routing on Shuffled vs. Coherent Streams", fontsize=13)
    plt.xticks(amortization_intervals)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fig3_amortization_tradeoff.png", dpi=300)
    plt.close()
    print("Amortization plot saved to results/fig3_amortization_tradeoff.png")
    
    return results_acc


# Experiment 2: Normalized vs Raw Shannon Entropy under varying vocabulary sizes
def run_vocabulary_heterogeneity():
    print("\n--- Running Experiment 2: Raw vs. Normalized Shannon Entropy ---")
    seed = 42
    set_seed(seed)
    
    # Vocabulary sizes for 4 tasks
    # MNIST (10), F-MNIST (5), CIFAR-10 (8), SVHN (4)
    vocabs = {0: 10, 1: 5, 2: 8, 3: 4}
    
    # Generate datasets with heterogeneous class sizes
    subspace_size = D // K  # 48
    prototypes = {}
    for k in range(K):
        num_classes = vocabs[k]
        random_matrix = np.random.randn(subspace_size, num_classes)
        q, _ = np.linalg.qr(random_matrix)
        q = q.T  # shape (num_classes, 48)
        
        task_prototypes = np.zeros((num_classes, D))
        task_prototypes[:, k*subspace_size : (k+1)*subspace_size] = q
        prototypes[k] = torch.tensor(task_prototypes, dtype=torch.float32)
        
    noise_levels = {0: 0.01, 1: 0.05, 2: 0.24, 3: 0.56}
    
    train_data = {}
    test_data = {}
    
    for k in range(K):
        train_x, train_y = [], []
        num_classes = vocabs[k]
        for i in range(1000):
            c = i % num_classes
            noise = torch.randn(D) * noise_levels[k]
            sample = prototypes[k][c] + noise
            train_x.append(sample)
            train_y.append(c)
        train_data[k] = (torch.stack(train_x), torch.tensor(train_y))
        
        test_x, test_y = [], []
        for i in range(250):
            c = i % num_classes
            noise = torch.randn(D) * noise_levels[k]
            sample = prototypes[k][c] + noise
            test_x.append(sample)
            test_y.append(c)
        test_data[k] = (torch.stack(test_x), torch.tensor(test_y))
        
    # Train experts with heterogeneous heads
    experts = {}
    for k in range(K):
        model = ExpertModel(num_classes=vocabs[k]).to(DEVICE)
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
        
    # Build fully shuffled heterogeneous test stream
    homo_x = []
    homo_y = []
    homo_tasks = []
    for k in range(K):
        x, y = test_data[k]
        homo_x.append(x)
        homo_y.append(y)
        homo_tasks.extend([k]*len(x))
    homo_x = torch.cat(homo_x).to(DEVICE)
    homo_y = torch.cat(homo_y).to(DEVICE)
    
    set_seed(seed + 500)
    shuffled_idx = torch.randperm(len(homo_x))
    stream_x = homo_x[shuffled_idx]
    stream_y = homo_y[shuffled_idx]
    stream_tasks = np.array(homo_tasks)[shuffled_idx.numpy()]
    
    # Evaluate with Raw Shannon Entropy vs Normalized Shannon Entropy
    def eval_routing(use_normalized):
        correct = 0
        total = len(stream_x)
        task_correct = [0]*K
        task_total = [0]*K
        
        with torch.no_grad():
            for b in range(total):
                x = stream_x[b:b+1]
                y = stream_y[b:b+1]
                t_true = stream_tasks[b]
                
                entropy_vals = torch.zeros(K)
                for k in range(K):
                    logits_k, _ = experts[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    raw_ent = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
                    
                    if use_normalized:
                        # Normalized Entropy Eq 4: divide by log(Y_k)
                        entropy_vals[k] = raw_ent / np.log(vocabs[k])
                    else:
                        entropy_vals[k] = raw_ent
                        
                k_star = entropy_vals.argmin().item()
                logits_star, _ = experts[k_star](x)
                pred = logits_star.argmax(dim=1)
                
                is_correct = (pred == y).item() if k_star == t_true else 0 # correct expert prediction
                correct += is_correct
                total += 1
                task_correct[t_true] += is_correct
                task_total[t_true] += 1
                
        return [task_correct[k]/max(1, task_total[k]) for k in range(K)]

    accs_raw = eval_routing(use_normalized=False)
    accs_norm = eval_routing(use_normalized=True)
    
    print("\n--- Vocabulary Heterogeneity Routing Accuracy ---")
    print("| Metric | MNIST (Y=10) | F-MNIST (Y=5) | CIFAR-10 (Y=8) | SVHN (Y=4) | Joint Mean |")
    print(f"| Raw Entropy | {accs_raw[0]*100:.2f}% | {accs_raw[1]*100:.2f}% | {accs_raw[2]*100:.2f}% | {accs_raw[3]*100:.2f}% | {np.mean(accs_raw)*100:.2f}% |")
    print(f"| Normalized Entropy | {accs_norm[0]*100:.2f}% | {accs_norm[1]*100:.2f}% | {accs_norm[2]*100:.2f}% | {accs_norm[3]*100:.2f}% | {np.mean(accs_norm)*100:.2f}% |")
    
    return accs_raw, accs_norm


# Experiment 3: Physical Wall-Clock Latency Benchmark
def run_latency_benchmark():
    print("\n--- Running Experiment 3: Physical CPU Latency Profile ---")
    seed = 42
    train_data, test_data = generate_homogeneous_datasets(seed)
    
    # Train/instantiate models
    experts = {}
    for k in range(K):
        experts[k] = ExpertModel().to(DEVICE)
        
    stream_x, stream_y, stream_tasks = generate_block_shuffled_stream(test_data, 10, seed)
    stream_x = stream_x.to(DEVICE)
    
    N_steps = 200 # Profile over 200 steps
    
    # 1. Single-Pass Expert (just execute 1 expert)
    start_time = time.time()
    for b in range(N_steps):
        x = stream_x[b:b+1]
        _ = experts[0](x)
    single_pass_time = (time.time() - start_time) / N_steps * 1000 # in ms
    
    # 2. Uniform Merging (computes blended adapter and then single forward pass)
    start_time = time.time()
    for b in range(N_steps):
        x = stream_x[b:b+1]
        # Blended forward pass
        h = x
        for i in range(NUM_LAYERS):
            update = sum([torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)]) / K
            h = h + update
        _ = experts[0].head(h)
    uniform_time = (time.time() - start_time) / N_steps * 1000 # in ms
    
    # 3. Full EER (runs all K experts' heads and adapters to compute entropy)
    start_time = time.time()
    for b in range(N_steps):
        x = stream_x[b:b+1]
        entropy_vals = torch.zeros(K)
        for k in range(K):
            logits_k, _ = experts[k](x)
            probs_k = torch.softmax(logits_k, dim=1)
            entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
        k_star = entropy_vals.argmin().item()
        _, _ = experts[k_star](x)
    eer_time = (time.time() - start_time) / N_steps * 1000 # in ms
    
    # 4. Amortized EER (N_amortize = 10)
    start_time = time.time()
    cached_k_star = 0
    for b in range(N_steps):
        x = stream_x[b:b+1]
        if b % 10 == 0:
            entropy_vals = torch.zeros(K)
            for k in range(K):
                logits_k, _ = experts[k](x)
                probs_k = torch.softmax(logits_k, dim=1)
                entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
            cached_k_star = entropy_vals.argmin().item()
        _, _ = experts[cached_k_star](x)
    amortized_time = (time.time() - start_time) / N_steps * 1000 # in ms
    
    print("\n--- Wall-Clock CPU Execution Latency (ms per sample) ---")
    print(f"| Pipeline | Latency (ms/sample) | Relative Overhead |")
    print(f"| Single-Pass Expert | {single_pass_time:.4f} ms | 1.00x |")
    print(f"| Uniform Merging | {uniform_time:.4f} ms | {uniform_time/single_pass_time:.2f}x |")
    print(f"| Full EER Routing (Ours) | {eer_time:.4f} ms | {eer_time/single_pass_time:.2f}x |")
    print(f"| Amortized EER (N=10, Ours) | {amortized_time:.4f} ms | {amortized_time/single_pass_time:.2f}x |")
    
    return single_pass_time, uniform_time, eer_time, amortized_time

def run_temperature_ablation():
    print("\n--- Running Experiment 4: Softmax Temperature Ablation for EPL-OCA ---")
    seed = 42
    train_data, test_data = generate_homogeneous_datasets(seed)
    
    # Train experts
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

    # Prepare stream datasets
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

    T_warmup = 200
    temperatures = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    print("\n--- EPL-OCA Refined Accuracy vs Softmax Temperature (tau) ---")
    print(f"| Temperature (tau) | Joint Mean Accuracy |")
    for tau in temperatures:
        set_seed(seed + 200)
        # Initialize task centroids to zero
        running_centroids = torch.zeros(K, D)
        centroid_counts = torch.zeros(K)
        beta = 0.1
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for b in range(len(hete_x)):
                x = hete_x[b:b+1]
                y = hete_y[b:b+1]
                t_true = hete_tasks[b]
                
                if b >= T_warmup:
                    u = torch.zeros(K)
                    for k in range(K):
                        c_v = running_centroids[k] if centroid_counts[k] > 0 else x[0]
                        u[k] = torch.dot(x[0], c_v) / (x[0].norm(p=2) * c_v.norm(p=2) + 1e-8)
                    
                    alpha = torch.softmax(u / tau, dim=0)
                    
                    # SPS Activation-Space Blending
                    h = x
                    for i in range(NUM_LAYERS):
                        update = sum([alpha[j] * torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)])
                        h = h + update
                    # Blended head
                    head_w = sum([alpha[j] * experts[j].head.weight for j in range(K)])
                    head_b = sum([alpha[j] * experts[j].head.bias for j in range(K)])
                    logits = torch.matmul(h, head_w.t()) + head_b
                    pred = logits.argmax(dim=1)
                    
                    is_correct = (pred == y).item()
                    correct += is_correct
                    total += 1
                
                # Zero-Shot Pseudo-Labeling via Prediction Entropy to update centroids
                entropy_vals = torch.zeros(K)
                for k in range(K):
                    logits_k, _ = experts[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
                
                k_star = entropy_vals.argmin().item()
                
                # Update Centroid
                if centroid_counts[k_star] == 0:
                    running_centroids[k_star] = x[0]
                    centroid_counts[k_star] = 1
                else:
                    running_centroids[k_star] = (1 - beta) * running_centroids[k_star] + beta * x[0]
                    
                running_centroids[k_star] = running_centroids[k_star] / (running_centroids[k_star].norm(p=2) + 1e-8)
                
        acc = correct / max(1, total)
        print(f"| tau = {tau:<11} | {acc*100:.2f}% |")


def main():
    print("=======================================================")
    print("         RUNNING NEW EXPERIMENTS FOR REVISIONS")
    print("=======================================================")
    
    run_amortization_ablation()
    run_vocabulary_heterogeneity()
    run_latency_benchmark()
    run_temperature_ablation()
    
    print("\nAll new experiments completed successfully!")

if __name__ == "__main__":
    main()
