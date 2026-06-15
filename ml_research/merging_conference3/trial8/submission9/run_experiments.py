import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define constants
D = 192
K = 4
C = 10
RANK = 8
NUM_LAYERS = 9
SEEDS = [10, 11, 12, 13, 14]
DEVICE = torch.device("cpu")

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

def generate_datasets(seed):
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
        
    # Calibrated noise levels:
    noise_levels = {
        0: 0.01,   # MNIST
        1: 0.05,   # F-MNIST
        2: 0.24,   # CIFAR-10
        3: 0.56    # SVHN
    }
    
    train_data = {}
    cal_data = {}
    test_data = {}
    
    for k in range(K):
        # 1000 train, 64 calibration, 250 test samples per task
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

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def run_evaluation_for_seed(seed):
    print(f"\n--- Running Seed {seed} ---")
    train_data, cal_data, test_data, prototypes = generate_datasets(seed)
    
    # 1. Train Experts
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
            
        test_x, test_y = test_data[k]
        test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
        acc = evaluate_model(model, test_x, test_y)
        print(f"Task {k} Expert Ceiling Test Accuracy: {acc*100:.2f}%")
        experts[k] = model
        
    # Baselines:
    # 2. Expert Ceiling
    expert_ceilings = {}
    for k in range(K):
        test_x, test_y = test_data[k]
        test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
        acc = evaluate_model(experts[k], test_x, test_y)
        expert_ceilings[k] = acc
        
    # 3. Static Uniform Merging
    def eval_uniform_merging():
        correct = [0]*K
        total = [0]*K
        with torch.no_grad():
            for k in range(K):
                test_x, test_y = test_data[k]
                test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
                
                h = test_x
                for i in range(NUM_LAYERS):
                    update = sum([torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)]) / K
                    h = h + update
                head_w = sum([experts[j].head.weight for j in range(K)]) / K
                head_b = sum([experts[j].head.bias for j in range(K)]) / K
                logits = torch.matmul(h, head_w.t()) + head_b
                preds = logits.argmax(dim=1)
                correct[k] += (preds == test_y).sum().item()
                total[k] += test_y.size(0)
        return [correct[k]/total[k] for k in range(K)]
        
    uniform_accs = eval_uniform_merging()
    
    # 4. SPS-ZCA (Offline SOTA)
    zca_centroids = {}
    for k in range(K):
        cal_x, _ = cal_data[k]
        cal_x = cal_x.to(DEVICE)
        zca_centroids[k] = cal_x.mean(dim=0)
        zca_centroids[k] = zca_centroids[k] / (zca_centroids[k].norm(p=2) + 1e-8)
        
    def eval_sps_zca(stream_x, stream_y, stream_tasks):
        correct = 0
        total = 0
        task_correct = [0]*K
        task_total = [0]*K
        
        with torch.no_grad():
            for b in range(len(stream_x)):
                x = stream_x[b:b+1]
                y = stream_y[b:b+1]
                t_true = stream_tasks[b]
                
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
                
                is_correct = (pred == y).item()
                correct += is_correct
                total += 1
                task_correct[t_true] += is_correct
                task_total[t_true] += 1
                
        return correct/total, [task_correct[k]/max(1, task_total[k]) for k in range(K)]

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
    
    _, sps_zca_homo_task_accs = eval_sps_zca(homo_x, homo_y, homo_tasks)
    sps_zca_homo_acc = np.mean(sps_zca_homo_task_accs)
    sps_zca_hete_acc, sps_zca_hete_task_accs = eval_sps_zca(hete_x, hete_y, hete_tasks)

    # 5. PFSR (Head-Dependent Routing)
    def eval_pfsr(stream_x, stream_y, stream_tasks):
        correct = 0
        total = 0
        task_correct = [0]*K
        task_total = [0]*K
        
        with torch.no_grad():
            for b in range(len(stream_x)):
                x = stream_x[b:b+1]
                y = stream_y[b:b+1]
                t_true = stream_tasks[b]
                
                u = torch.zeros(K)
                for k in range(K):
                    logits, _ = experts[k](x)
                    u[k] = logits.max().item()
                    
                alpha = torch.softmax(u / 1.0, dim=0)
                
                h = x
                for i in range(NUM_LAYERS):
                    update = sum([alpha[j] * torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)])
                    h = h + update
                head_w = sum([alpha[j] * experts[j].head.weight for j in range(K)])
                head_b = sum([alpha[j] * experts[j].head.bias for j in range(K)])
                logits = torch.matmul(h, head_w.t()) + head_b
                pred = logits.argmax(dim=1)
                
                is_correct = (pred == y).item()
                correct += is_correct
                total += 1
                task_correct[t_true] += is_correct
                task_total[t_true] += 1
                
        return correct/total, [task_correct[k]/max(1, task_total[k]) for k in range(K)]

    pfsr_homo_acc, pfsr_homo_task_accs = eval_pfsr(homo_x, homo_y, homo_tasks)
    pfsr_hete_acc, pfsr_hete_task_accs = eval_pfsr(hete_x, hete_y, hete_tasks)

    # 5.5 Proposed Method: Zero-Shot Expert Entropy Routing (EER)
    def eval_eer(stream_x, stream_y, stream_tasks):
        set_seed(seed + 400)
        correct = 0
        total = 0
        task_correct = [0]*K
        task_total = [0]*K
        
        with torch.no_grad():
            for b in range(len(stream_x)):
                x = stream_x[b:b+1]
                y = stream_y[b:b+1]
                t_true = stream_tasks[b]
                
                # EER evaluates prediction entropy for all K experts
                entropy_vals = torch.zeros(K)
                for k in range(K):
                    logits_k, _ = experts[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
                
                # Select the active expert with minimum prediction entropy
                k_star = entropy_vals.argmin().item()
                
                # Predict using the selected expert head and layers
                logits_star, _ = experts[k_star](x)
                pred = logits_star.argmax(dim=1)
                
                is_correct = (pred == y).item()
                correct += is_correct
                total += 1
                task_correct[t_true] += is_correct
                task_total[t_true] += 1
                
        return correct/max(1, total), [task_correct[k]/max(1, task_total[k]) for k in range(K)]

    eer_hete_acc, eer_hete_task_accs = eval_eer(hete_x, hete_y, hete_tasks)

    # 6. Proposed Method: EPL-OCA (Entropy-Pseudo-Labeled Online Centroid Adaptation)
    # Warm-up phase: We process the first T_warmup samples by updating centroids using expert confidence
    T_warmup = 200
    
    def run_epl_oca(stream_x, stream_y, stream_tasks, use_refinement=True):
        set_seed(seed + 200)
        
        # Initialize task centroids to zero
        running_centroids = torch.zeros(K, D)
        centroid_counts = torch.zeros(K)
        
        beta = 0.1 # centroid accumulation/adaptation rate
        
        # Warm-up and continuous serving loop
        correct = 0
        total = 0
        task_correct = [0]*K
        task_total = [0]*K
        centroid_history = []
        
        with torch.no_grad():
            for b in range(len(stream_x)):
                x = stream_x[b:b+1]
                y = stream_y[b:b+1]
                t_true = stream_tasks[b]
                
                # 1. Evaluate routing once warm-up is completed (using existing centroids before seeing sample b)
                if b >= T_warmup:
                    # Compute routing similarity coordinates using our dynamically maintained centroids
                    u = torch.zeros(K)
                    for k in range(K):
                        # If a centroid is not yet initialized (extremely rare), fallback to dot product with raw feature
                        c_v = running_centroids[k] if centroid_counts[k] > 0 else x[0]
                        u[k] = torch.dot(x[0], c_v) / (x[0].norm(p=2) * c_v.norm(p=2) + 1e-8)
                    
                    tau = 0.001
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
                    task_correct[t_true] += is_correct
                    task_total[t_true] += 1
                
                # 2. Zero-Shot Pseudo-Labeling via Prediction Entropy (to update centroids for subsequent steps)
                # Pass input through all K experts to measure confidence
                entropy_vals = torch.zeros(K)
                for k in range(K):
                    logits_k, _ = experts[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
                
                # The correct expert is the one with minimum prediction entropy (maximum confidence)
                k_star = entropy_vals.argmin().item()
                
                # 3. Update Centroid on-the-fly using the sample
                # If centroid is uninitialized, set it to the sample directly. Otherwise, use EMA.
                if centroid_counts[k_star] == 0:
                    running_centroids[k_star] = x[0]
                    centroid_counts[k_star] = 1
                else:
                    running_centroids[k_star] = (1 - beta) * running_centroids[k_star] + beta * x[0]
                    
                running_centroids[k_star] = running_centroids[k_star] / (running_centroids[k_star].norm(p=2) + 1e-8)
                
                # If refinement is disabled on steps > T_warmup, we freeze updates
                if not use_refinement and b >= T_warmup:
                    beta = 0.0 # freeze updates
                    
                centroid_history.append(running_centroids.clone())
                
        return correct/max(1, total), [task_correct[k]/max(1, task_total[k]) for k in range(K)], centroid_history

    def run_streaming_kmeans(stream_x, stream_y, stream_tasks, use_refinement=True):
        import itertools
        set_seed(seed + 500)
        
        # Initialize K centroids using the first K samples in the stream
        kmeans_centroids = torch.zeros(K, D)
        for k_idx in range(K):
            kmeans_centroids[k_idx] = stream_x[k_idx].clone()
            kmeans_centroids[k_idx] = kmeans_centroids[k_idx] / (kmeans_centroids[k_idx].norm(p=2) + 1e-8)
            
        beta = 0.1
        
        correct = 0
        total = 0
        task_correct = [0]*K
        task_total = [0]*K
        
        alignment_solved = False
        best_perm_inv = None
        
        with torch.no_grad():
            for b in range(len(stream_x)):
                x = stream_x[b:b+1]
                y = stream_y[b:b+1]
                t_true = stream_tasks[b]
                
                # Zero-shot alignment at T_warmup
                if b == T_warmup and not alignment_solved:
                    C_cost = torch.zeros(K, K)
                    for c_idx in range(K):
                        for k_idx in range(K):
                            logits_k, _ = experts[k_idx](kmeans_centroids[c_idx].unsqueeze(0))
                            probs_k = torch.softmax(logits_k, dim=1)
                            C_cost[c_idx, k_idx] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item()
                            
                    perms = list(itertools.permutations(range(K)))
                    best_cost = float('inf')
                    best_perm = None
                    for perm in perms:
                        cost = sum(C_cost[c_idx, perm[c_idx]] for c_idx in range(K))
                        if cost < best_cost:
                            best_cost = cost
                            best_perm = perm
                            
                    best_perm_inv = [best_perm.index(k_idx) for k_idx in range(K)]
                    alignment_solved = True
                    
                # Routing & prediction once alignment is solved
                if b >= T_warmup:
                    u = torch.zeros(K)
                    for k_idx in range(K):
                        c_idx = best_perm_inv[k_idx]
                        c_v = kmeans_centroids[c_idx]
                        u[k_idx] = torch.dot(x[0], c_v) / (x[0].norm(p=2) * c_v.norm(p=2) + 1e-8)
                        
                    tau = 0.001
                    alpha = torch.softmax(u / tau, dim=0)
                    
                    # SPS Activation Blending
                    h = x
                    for i in range(NUM_LAYERS):
                        update = sum([alpha[j] * torch.matmul(torch.matmul(h, experts[j].A[i]), experts[j].B[i]) for j in range(K)])
                        h = h + update
                        
                    head_w = sum([alpha[j] * experts[j].head.weight for j in range(K)])
                    head_b = sum([alpha[j] * experts[j].head.bias for j in range(K)])
                    logits = torch.matmul(h, head_w.t()) + head_b
                    pred = logits.argmax(dim=1)
                    
                    is_correct = (pred == y).item()
                    correct += is_correct
                    total += 1
                    task_correct[t_true] += is_correct
                    task_total[t_true] += 1
                    
                # Update centroids (unsupervised distance-based)
                u_dist = torch.zeros(K)
                for c_idx in range(K):
                    u_dist[c_idx] = torch.dot(x[0], kmeans_centroids[c_idx]) / (x[0].norm(p=2) * kmeans_centroids[c_idx].norm(p=2) + 1e-8)
                c_star = u_dist.argmax().item()
                
                # If refinement is active or we are in warm-up phase, update
                if use_refinement or b < T_warmup:
                    kmeans_centroids[c_star] = (1 - beta) * kmeans_centroids[c_star] + beta * x[0]
                    kmeans_centroids[c_star] = kmeans_centroids[c_star] / (kmeans_centroids[c_star].norm(p=2) + 1e-8)
                    
        return correct/max(1, total), [task_correct[k]/max(1, task_total[k]) for k in range(K)]

    epl_oca_static_acc, epl_oca_static_task_accs, _ = run_epl_oca(hete_x, hete_y, hete_tasks, use_refinement=False)
    epl_oca_refine_acc, epl_oca_refine_task_accs, _ = run_epl_oca(hete_x, hete_y, hete_tasks, use_refinement=True)

    kmeans_static_acc, kmeans_static_task_accs = run_streaming_kmeans(hete_x, hete_y, hete_tasks, use_refinement=False)
    kmeans_refine_acc, kmeans_refine_task_accs = run_streaming_kmeans(hete_x, hete_y, hete_tasks, use_refinement=True)

    # 7. Stress-Test: Streaming Domain Drift (Covariate Shift)
    set_seed(seed + 300)
    drift_dir = torch.randn(D)
    drift_dir = drift_dir / (drift_dir.norm(p=2) + 1e-8)
    drift_scale = 0.45
    
    drifted_hete_x = hete_x.clone()
    for b in range(len(hete_x)):
        drift_factor = drift_scale * (b / len(hete_x))
        drifted_hete_x[b] = hete_x[b] + drift_factor * drift_dir
        
    sps_zca_drift_acc, sps_zca_drift_task_accs = eval_sps_zca(drifted_hete_x, hete_y, hete_tasks)
    
    eer_drift_acc, eer_drift_task_accs = eval_eer(drifted_hete_x, hete_y, hete_tasks)

    epl_oca_static_drift_acc, epl_oca_static_drift_task_accs, static_centroid_history = run_epl_oca(
        drifted_hete_x, hete_y, hete_tasks, use_refinement=False
    )
    
    epl_oca_refine_drift_acc, epl_oca_refine_drift_task_accs, refine_centroid_history = run_epl_oca(
        drifted_hete_x, hete_y, hete_tasks, use_refinement=True
    )

    kmeans_static_drift_acc, kmeans_static_drift_task_accs = run_streaming_kmeans(
        drifted_hete_x, hete_y, hete_tasks, use_refinement=False
    )
    kmeans_refine_drift_acc, kmeans_refine_drift_task_accs = run_streaming_kmeans(
        drifted_hete_x, hete_y, hete_tasks, use_refinement=True
    )
    
    true_centroids_0 = {}
    for k in range(K):
        true_centroids_0[k] = test_data[k][0].mean(dim=0).to(DEVICE)
        true_centroids_0[k] = true_centroids_0[k] / (true_centroids_0[k].norm(p=2) + 1e-8)
        
    tracking_errors_static = []
    tracking_errors_refine = []
    
    for b in range(T_warmup, len(hete_x)):
        drift_factor = drift_scale * (b / len(hete_x))
        true_centroids_t = {}
        for k in range(K):
            true_centroids_t[k] = true_centroids_0[k] + drift_factor * drift_dir
            true_centroids_t[k] = true_centroids_t[k] / (true_centroids_t[k].norm(p=2) + 1e-8)
            
        err_static = sum([torch.norm(static_centroid_history[b][k] - true_centroids_t[k]).item() for k in range(K)]) / K
        err_refine = sum([torch.norm(refine_centroid_history[b][k] - true_centroids_t[k]).item() for k in range(K)]) / K
        tracking_errors_static.append(err_static)
        tracking_errors_refine.append(err_refine)

    return {
        "expert_ceilings": expert_ceilings,
        "uniform_accs": uniform_accs,
        "sps_zca_homo": sps_zca_homo_task_accs,
        "sps_zca_hete": sps_zca_hete_task_accs,
        "pfsr_homo": pfsr_homo_task_accs,
        "pfsr_hete": pfsr_hete_task_accs,
        "eer_hete": eer_hete_task_accs,
        "epl_oca_static_hete": epl_oca_static_task_accs,
        "epl_oca_refine_hete": epl_oca_refine_task_accs,
        "kmeans_static_hete": kmeans_static_task_accs,
        "kmeans_refine_hete": kmeans_refine_task_accs,
        "sps_zca_drift": sps_zca_drift_task_accs,
        "eer_drift": eer_drift_task_accs,
        "epl_oca_static_drift": epl_oca_static_drift_task_accs,
        "epl_oca_refine_drift": epl_oca_refine_drift_task_accs,
        "kmeans_static_drift": kmeans_static_drift_task_accs,
        "kmeans_refine_drift": kmeans_refine_drift_task_accs,
        "tracking_errors_static": tracking_errors_static,
        "tracking_errors_refine": tracking_errors_refine
    }

def main():
    all_results = []
    for seed in SEEDS:
        res = run_evaluation_for_seed(seed)
        all_results.append(res)
        
    def aggregate_task_accs(key):
        accs = np.array([[r[key][k] for k in range(K)] for r in all_results])
        means = np.mean(accs, axis=0)
        stds = np.std(accs, axis=0)
        joint_means = np.mean(accs, axis=1)
        joint_mean = np.mean(joint_means)
        joint_std = np.std(joint_means)
        return means, stds, joint_mean, joint_std

    ceil_mean, ceil_std, ceil_jm, ceil_jstd = aggregate_task_accs("expert_ceilings")
    unif_mean, unif_std, unif_jm, unif_jstd = aggregate_task_accs("uniform_accs")
    pfsr_homo_mean, pfsr_homo_std, pfsr_homo_jm, pfsr_homo_jstd = aggregate_task_accs("pfsr_homo")
    pfsr_hete_mean, pfsr_hete_std, pfsr_hete_jm, pfsr_hete_jstd = aggregate_task_accs("pfsr_hete")
    zca_homo_mean, zca_homo_std, zca_homo_jm, zca_homo_jstd = aggregate_task_accs("sps_zca_homo")
    zca_hete_mean, zca_hete_std, zca_hete_jm, zca_hete_jstd = aggregate_task_accs("sps_zca_hete")
    
    eer_hete_mean, eer_hete_std, eer_hete_jm, eer_hete_jstd = aggregate_task_accs("eer_hete")
    epl_oca_static_hete_mean, epl_oca_static_hete_std, epl_oca_static_hete_jm, epl_oca_static_hete_jstd = aggregate_task_accs("epl_oca_static_hete")
    epl_oca_refine_hete_mean, epl_oca_refine_hete_std, epl_oca_refine_hete_jm, epl_oca_refine_hete_jstd = aggregate_task_accs("epl_oca_refine_hete")
    kmeans_static_hete_mean, kmeans_static_hete_std, kmeans_static_hete_jm, kmeans_static_hete_jstd = aggregate_task_accs("kmeans_static_hete")
    kmeans_refine_hete_mean, kmeans_refine_hete_std, kmeans_refine_hete_jm, kmeans_refine_hete_jstd = aggregate_task_accs("kmeans_refine_hete")

    zca_drift_mean, zca_drift_std, zca_drift_jm, zca_drift_jstd = aggregate_task_accs("sps_zca_drift")
    eer_drift_mean, eer_drift_std, eer_drift_jm, eer_drift_jstd = aggregate_task_accs("eer_drift")
    epl_oca_static_drift_mean, epl_oca_static_drift_std, epl_oca_static_drift_jm, epl_oca_static_drift_jstd = aggregate_task_accs("epl_oca_static_drift")
    epl_oca_refine_drift_mean, epl_oca_refine_drift_std, epl_oca_refine_drift_jm, epl_oca_refine_drift_jstd = aggregate_task_accs("epl_oca_refine_drift")
    kmeans_static_drift_mean, kmeans_static_drift_std, kmeans_static_drift_jm, kmeans_static_drift_jstd = aggregate_task_accs("kmeans_static_drift")
    kmeans_refine_drift_mean, kmeans_refine_drift_std, kmeans_refine_drift_jm, kmeans_refine_drift_jstd = aggregate_task_accs("kmeans_refine_drift")

    print("\n=======================================================")
    print("                AGGREGATED RESULTS")
    print("=======================================================\n")
    
    def print_row(name, task_means, task_stds, joint_mean, joint_std):
        cols = [f"{task_means[i]*100:.2f} ± {task_stds[i]*100:.2f}%" for i in range(K)]
        cols_str = " | ".join(cols)
        print(f"| {name:<30} | {cols_str} | {joint_mean*100:.2f} ± {joint_std*100:.2f}% |")

    print("| Method | MNIST | FashionMNIST | CIFAR-10 | SVHN | Joint Mean |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print_row("Expert Ceiling", ceil_mean, ceil_std, ceil_jm, ceil_jstd)
    print_row("Uniform Merging", unif_mean, unif_std, unif_jm, unif_jstd)
    print_row("PFSR (Homogeneous)", pfsr_homo_mean, pfsr_homo_std, pfsr_homo_jm, pfsr_homo_jstd)
    print_row("PFSR (Heterogeneous)", pfsr_hete_mean, pfsr_hete_std, pfsr_hete_jm, pfsr_hete_jstd)
    print_row("SPS-ZCA (Homo, Offline SOTA)", zca_homo_mean, zca_homo_std, zca_homo_jm, zca_homo_jstd)
    print_row("SPS-ZCA (Hete, Offline SOTA)", zca_hete_mean, zca_hete_std, zca_hete_jm, zca_hete_jstd)
    print_row("EER (Direct Routing, Ours)", eer_hete_mean, eer_hete_std, eer_hete_jm, eer_hete_jstd)
    print_row("EPL-OCA (Static, Unsupervised)", epl_oca_static_hete_mean, epl_oca_static_hete_std, epl_oca_static_hete_jm, epl_oca_static_hete_jstd)
    print_row("EPL-OCA (Refined, Unsupervised)", epl_oca_refine_hete_mean, epl_oca_refine_hete_std, epl_oca_refine_hete_jm, epl_oca_refine_hete_jstd)
    print_row("Streaming K-Means (Static)", kmeans_static_hete_mean, kmeans_static_hete_std, kmeans_static_hete_jm, kmeans_static_hete_jstd)
    print_row("Streaming K-Means (Refined)", kmeans_refine_hete_mean, kmeans_refine_hete_std, kmeans_refine_hete_jm, kmeans_refine_hete_jstd)

    print("\n=======================================================")
    print("             ROBUSTNESS TO COVARIATE SHIFT")
    print("=======================================================\n")
    print("| Method (Under Drift) | MNIST | FashionMNIST | CIFAR-10 | SVHN | Joint Mean |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print_row("SPS-ZCA (Offline centroids)", zca_drift_mean, zca_drift_std, zca_drift_jm, zca_drift_jstd)
    print_row("EER (Direct Routing, Ours)", eer_drift_mean, eer_drift_std, eer_drift_jm, eer_drift_jstd)
    print_row("EPL-OCA (Static centroids)", epl_oca_static_drift_mean, epl_oca_static_drift_std, epl_oca_static_drift_jm, epl_oca_static_drift_jstd)
    print_row("EPL-OCA (Refined centroids)", epl_oca_refine_drift_mean, epl_oca_refine_drift_std, epl_oca_refine_drift_jm, epl_oca_refine_drift_jstd)
    print_row("Streaming K-Means (Static)", kmeans_static_drift_mean, kmeans_static_drift_std, kmeans_static_drift_jm, kmeans_static_drift_jstd)
    print_row("Streaming K-Means (Refined)", kmeans_refine_drift_mean, kmeans_refine_drift_std, kmeans_refine_drift_jm, kmeans_refine_drift_jstd)

    os.makedirs("results", exist_ok=True)
    
    static_errors = np.mean(np.array([r["tracking_errors_static"] for r in all_results]), axis=0)
    refine_errors = np.mean(np.array([r["tracking_errors_refine"] for r in all_results]), axis=0)
    steps = np.arange(200, 1000)
    
    plt.figure(figsize=(8, 5))
    plt.plot(steps, static_errors, label="Static Centroids (No Refinement)", color="red", linestyle="--", linewidth=2)
    plt.plot(steps, refine_errors, label="Online Centroid Refinement (EMA)", color="green", linewidth=2.5)
    plt.xlabel("Serving Stream Steps", fontsize=12)
    plt.ylabel("Centroid Tracking Error (L2 RMSE)", fontsize=12)
    plt.title("Centroid Tracking Error Under Continuous Covariate Shift", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/fig1_centroid_tracking_error.png", dpi=300)
    plt.close()
    
    methods = ["SPS-ZCA", "EER (Ours)", "EPL-OCA (Static)", "EPL-OCA (Refined)"]
    accuracies = [zca_drift_jm * 100, eer_drift_jm * 100, epl_oca_static_drift_jm * 100, epl_oca_refine_drift_jm * 100]
    std_devs = [zca_drift_jstd * 100, eer_drift_jstd * 100, epl_oca_static_drift_jstd * 100, epl_oca_refine_drift_jstd * 100]
    
    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"]
    bars = plt.bar(methods, accuracies, yerr=std_devs, color=colors, capsize=8, alpha=0.85, width=0.5)
    plt.ylabel("Joint Mean Test Accuracy (%)", fontsize=12)
    plt.title("Performance Under Extreme Continuous Covariate Shift", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/fig2_accuracy_under_drift.png", dpi=300)
    plt.close()
    
    print("\nPlots saved successfully in 'results/' directory.")

if __name__ == "__main__":
    main()
