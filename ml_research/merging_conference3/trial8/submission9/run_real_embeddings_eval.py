import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

# Global Constants
D = 512          # ResNet18 feature dimension
C = 10           # number of classes per task
K = 4            # number of tasks
SEEDS = [42, 43, 44, 45, 46]
T_warmup = 200   # warmup steps for online routing

print("Downloading ResNet18 feature extractor...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Downloading datasets...")
datasets_raw = {
    0: dsets.MNIST(root='./data', train=True, download=True, transform=transform_gray),
    1: dsets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray),
    2: dsets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb),
    3: dsets.SVHN(root='./data', split='train', download=True, transform=transform_rgb)
}

datasets_test_raw = {
    0: dsets.MNIST(root='./data', train=False, download=True, transform=transform_gray),
    1: dsets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray),
    2: dsets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb),
    3: dsets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
}

def extract_features(dataset, num_samples):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    features_list = []
    labels_list = []
    total_extracted = 0
    with torch.no_grad():
        for x, y in loader:
            feats = feature_extractor(x).squeeze(-1).squeeze(-1)
            features_list.append(feats)
            labels_list.append(y)
            total_extracted += len(x)
            if total_extracted >= num_samples:
                break
    return torch.cat(features_list)[:num_samples], torch.cat(labels_list)[:num_samples]

print("Extracting features...")
train_feats = {}
train_labels = {}
cal_feats = {}
cal_labels = {}
test_feats = {}
test_labels = {}

for k in range(K):
    train_feats[k], train_labels[k] = extract_features(datasets_raw[k], 1000)
    cal_feats[k], cal_labels[k] = extract_features(datasets_raw[k], 64)
    test_feats[k], test_labels[k] = extract_features(datasets_test_raw[k], 250)

# We will collect results across 5 seeds
results = {
    "oracle": [],
    "uniform": [],
    "task_arithmetic": [],
    "zca": [],
    "eer": [],
    "cg_eer": [],
    "ucg_eer": [],
    "ucg_eer_t10": [],
    "ucg_eer_t50": [],
    "ucg_eer_t100": [],
    "ucg_eer_t200": [],
    "oca_hard": [],
    "oca_soft": []
}

# Calibration joint set
joint_cal_x = []
joint_cal_tasks = []
for k in range(K):
    joint_cal_x.append(cal_feats[k])
    joint_cal_tasks.extend([k] * len(cal_feats[k]))
joint_cal_x = torch.cat(joint_cal_x)
joint_cal_tasks = np.array(joint_cal_tasks)

for seed in SEEDS:
    print(f"\n--- Evaluation for Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train heads with this seed
    heads = {}
    for k in range(K):
        head = nn.Linear(D, C)
        # Use seed to initialize weights
        nn.init.kaiming_uniform_(head.weight, a=np.sqrt(5))
        if head.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(head.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(head.bias, -bound, bound)
            
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        x = train_feats[k]
        y = train_labels[k]
        
        for epoch in range(50):
            head.train()
            optimizer.zero_grad()
            logits = head(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        head.eval()
        heads[k] = head
        
    # Construct heterogeneous shuffled stream
    homo_x = []
    homo_y = []
    homo_tasks = []
    for k in range(K):
        homo_x.append(test_feats[k])
        homo_y.append(test_labels[k])
        homo_tasks.extend([k]*len(test_feats[k]))
    homo_x = torch.cat(homo_x)
    homo_y = torch.cat(homo_y)
    
    shuffled_idx = torch.randperm(len(homo_x))
    hete_x = homo_x[shuffled_idx]
    hete_y = homo_y[shuffled_idx]
    hete_tasks = np.array(homo_tasks)[shuffled_idx.numpy()]
    
    # 1. Oracle
    oracle_correct = 0
    with torch.no_grad():
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            t = hete_tasks[b]
            logits = heads[t](x)
            pred = logits.argmax(dim=1)
            oracle_correct += (pred == y).item()
    oracle_acc = oracle_correct / len(hete_x) * 100
    results["oracle"].append(oracle_acc)
    
    # 2. Uniform
    uniform_correct = 0
    with torch.no_grad():
        merged_weight = sum([heads[k].weight for k in range(K)]) / K
        merged_bias = sum([heads[k].bias for k in range(K)]) / K
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            logits = torch.matmul(x, merged_weight.t()) + merged_bias
            pred = logits.argmax(dim=1)
            uniform_correct += (pred == y).item()
    uniform_acc = uniform_correct / len(hete_x) * 100
    results["uniform"].append(uniform_acc)
    
    # 2b. Task Arithmetic (Optimized Scaling Factor lambda on Joint Calibration Set)
    best_lambda = 0.25
    best_cal_acc = 0.0
    with torch.no_grad():
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            merged_w_lam = sum([heads[k].weight for k in range(K)]) * lam
            merged_b_lam = sum([heads[k].bias for k in range(K)]) * lam
            cal_correct = 0
            for b_c in range(len(joint_cal_x)):
                x_c = joint_cal_x[b_c:b_c+1]
                t_c = joint_cal_tasks[b_c]
                idx_c = b_c % 64
                y_c = cal_labels[t_c][idx_c:idx_c+1]
                logits_c = torch.matmul(x_c, merged_w_lam.t()) + merged_b_lam
                pred_c = logits_c.argmax(dim=1)
                cal_correct += (pred_c == y_c).item()
            cal_acc = cal_correct / len(joint_cal_x)
            if cal_acc > best_cal_acc:
                best_cal_acc = cal_acc
                best_lambda = lam
                
        # Evaluate best lambda on test set
        merged_weight_opt = sum([heads[k].weight for k in range(K)]) * best_lambda
        merged_bias_opt = sum([heads[k].bias for k in range(K)]) * best_lambda
        ta_correct = 0
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            logits = torch.matmul(x, merged_weight_opt.t()) + merged_bias_opt
            pred = logits.argmax(dim=1)
            ta_correct += (pred == y).item()
    ta_acc = ta_correct / len(hete_x) * 100
    results["task_arithmetic"].append(ta_acc)
    
    # 3. ZCA
    zca_centroids = {}
    for k in range(K):
        zca_centroids[k] = cal_feats[k].mean(dim=0)
        zca_centroids[k] = zca_centroids[k] / (zca_centroids[k].norm(p=2) + 1e-8)
    
    zca_correct = 0
    with torch.no_grad():
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            u = torch.zeros(K)
            for k in range(K):
                u[k] = torch.dot(x[0], zca_centroids[k]) / (x[0].norm(p=2) * zca_centroids[k].norm(p=2) + 1e-8)
            alpha = torch.softmax(u / 0.001, dim=0)
            merged_w = sum([alpha[k] * heads[k].weight for k in range(K)])
            merged_b = sum([alpha[k] * heads[k].bias for k in range(K)])
            logits = torch.matmul(x, merged_w.t()) + merged_b
            pred = logits.argmax(dim=1)
            zca_correct += (pred == y).item()
    zca_acc = zca_correct / len(hete_x) * 100
    results["zca"].append(zca_acc)
    
    # 4. EER
    eer_correct = 0
    with torch.no_grad():
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            entropy_vals = torch.zeros(K)
            for k in range(K):
                logits_k = heads[k](x)
                probs_k = torch.softmax(logits_k, dim=1)
                entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
            k_star = entropy_vals.argmin().item()
            logits_star = heads[k_star](x)
            pred = logits_star.argmax(dim=1)
            eer_correct += (pred == y).item()
    eer_acc = eer_correct / len(hete_x) * 100
    results["eer"].append(eer_acc)
    
    # 4b. CG-EER
    cg_eer_correct = 0
    with torch.no_grad():
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            similarities = torch.zeros(K)
            for k in range(K):
                similarities[k] = torch.dot(x[0], zca_centroids[k]) / (x[0].norm(p=2) * zca_centroids[k].norm(p=2) + 1e-8)
            entropy_vals = torch.zeros(K)
            for k in range(K):
                if similarities[k] < 0.7:
                    entropy_vals[k] = 1.0
                else:
                    logits_k = heads[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
            if (entropy_vals == 1.0).all():
                k_star = similarities.argmax().item()
            else:
                k_star = entropy_vals.argmin().item()
            logits_star = heads[k_star](x)
            pred = logits_star.argmax(dim=1)
            cg_eer_correct += (pred == y).item()
    cg_eer_acc = cg_eer_correct / len(hete_x) * 100
    results["cg_eer"].append(cg_eer_acc)
    
    # 4c. UCG-EER (Unsupervised Centroid-Gated Entropy Routing) with Warm-up Sweep
    for t_w in [10, 50, 100, 200]:
        ucg_eer_correct = 0
        total_ucg = 0
        ucg_running_centroids = torch.zeros(K, D)
        ucg_centroid_counts = torch.zeros(K)
        beta = 0.1
        with torch.no_grad():
            for b in range(len(hete_x)):
                x = hete_x[b:b+1]
                y = hete_y[b:b+1]
                if b >= t_w:
                    similarities = torch.zeros(K)
                    for k in range(K):
                        c_v = ucg_running_centroids[k] if ucg_centroid_counts[k] > 0 else x[0]
                        similarities[k] = torch.dot(x[0], c_v) / (x[0].norm(p=2) * c_v.norm(p=2) + 1e-8)
                    
                    entropy_vals = torch.zeros(K)
                    for k in range(K):
                        # If similarity is below threshold and the centroid exists, gate it out (set entropy to 1.0)
                        if ucg_centroid_counts[k] > 0 and similarities[k] < 0.7:
                            entropy_vals[k] = 1.0
                        else:
                            logits_k = heads[k](x)
                            probs_k = torch.softmax(logits_k, dim=1)
                            entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
                    
                    if (entropy_vals == 1.0).all():
                        k_star = similarities.argmax().item()
                    else:
                        k_star = entropy_vals.argmin().item()
                    
                    logits_star = heads[k_star](x)
                    pred = logits_star.argmax(dim=1)
                    ucg_eer_correct += (pred == y).item()
                    total_ucg += 1
                
                # Update online running centroids using EPL-OCA formula
                entropy_vals_all = torch.zeros(K)
                for k in range(K):
                    logits_k = heads[k](x)
                    probs_k = torch.softmax(logits_k, dim=1)
                    entropy_vals_all[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
                k_star_all = entropy_vals_all.argmin().item()
                if ucg_centroid_counts[k_star_all] == 0:
                    ucg_running_centroids[k_star_all] = x[0]
                    ucg_centroid_counts[k_star_all] = 1
                else:
                    ucg_running_centroids[k_star_all] = (1 - beta) * ucg_running_centroids[k_star_all] + beta * x[0]
                ucg_running_centroids[k_star_all] = ucg_running_centroids[k_star_all] / (ucg_running_centroids[k_star_all].norm(p=2) + 1e-8)
                
        ucg_eer_acc = ucg_eer_correct / max(1, total_ucg) * 100
        results[f"ucg_eer_t{t_w}"].append(ucg_eer_acc)
        if t_w == 200:
            results["ucg_eer"].append(ucg_eer_acc)
    
    # 5. EPL-OCA Hard
    T_warmup = 200
    running_centroids = torch.zeros(K, D)
    centroid_counts = torch.zeros(K)
    beta = 0.1
    oca_hard_correct = 0
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
                alpha = torch.softmax(u / 0.001, dim=0)
                merged_w = sum([alpha[k] * heads[k].weight for k in range(K)])
                merged_b = sum([alpha[k] * heads[k].bias for k in range(K)])
                logits = torch.matmul(x, merged_w.t()) + merged_b
                pred = logits.argmax(dim=1)
                oca_hard_correct += (pred == y).item()
                total_oca += 1
            entropy_vals = torch.zeros(K)
            for k in range(K):
                logits_k = heads[k](x)
                probs_k = torch.softmax(logits_k, dim=1)
                entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
            k_star = entropy_vals.argmin().item()
            if centroid_counts[k_star] == 0:
                running_centroids[k_star] = x[0]
                centroid_counts[k_star] = 1
            else:
                running_centroids[k_star] = (1 - beta) * running_centroids[k_star] + beta * x[0]
            running_centroids[k_star] = running_centroids[k_star] / (running_centroids[k_star].norm(p=2) + 1e-8)
    oca_hard_acc = oca_hard_correct / max(1, total_oca) * 100
    results["oca_hard"].append(oca_hard_acc)
    
    # 6. EPL-OCA Soft
    running_centroids = torch.zeros(K, D)
    centroid_counts = torch.zeros(K)
    oca_soft_correct = 0
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
                alpha = torch.softmax(u / 0.5, dim=0)
                merged_w = sum([alpha[k] * heads[k].weight for k in range(K)])
                merged_b = sum([alpha[k] * heads[k].bias for k in range(K)])
                logits = torch.matmul(x, merged_w.t()) + merged_b
                pred = logits.argmax(dim=1)
                oca_soft_correct += (pred == y).item()
                total_oca += 1
            entropy_vals = torch.zeros(K)
            for k in range(K):
                logits_k = heads[k](x)
                probs_k = torch.softmax(logits_k, dim=1)
                entropy_vals[k] = -torch.sum(probs_k * torch.log(probs_k + 1e-8)).item() / np.log(C)
            k_star = entropy_vals.argmin().item()
            if centroid_counts[k_star] == 0:
                running_centroids[k_star] = x[0]
                centroid_counts[k_star] = 1
            else:
                running_centroids[k_star] = (1 - beta) * running_centroids[k_star] + beta * x[0]
            running_centroids[k_star] = running_centroids[k_star] / (running_centroids[k_star].norm(p=2) + 1e-8)
    oca_soft_acc = oca_soft_correct / max(1, total_oca) * 100
    results["oca_soft"].append(oca_soft_acc)

print("\n================ FINAL RESULTS (MEAN \u00b1 STD OVER 5 SEEDS) ================")
for key, values in results.items():
    print(f"{key.upper()}: {np.mean(values):.2f}% \u00b1 {np.std(values):.2f}%")
