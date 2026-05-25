import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import json

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Shared Backbone Architecture
class SharedBackbone(nn.Module):
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Task-specific Expert Heads
class ExpertHead(nn.Module):
    def __init__(self):
        super(ExpertHead, self).__init__()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, feats):
        return self.fc2(self.relu(feats))

# Dataset rotation wrapper
class RotatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, angle):
        self.dataset = dataset
        self.angle = angle
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # Rotate x by self.angle (x is tensor [1, 28, 28])
        x_rot = transforms.functional.rotate(x, self.angle)
        return x_rot, y

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def get_next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        new_iterator = iter(loader)
        return next(new_iterator), new_iterator


# ==========================================
# EXPERIMENT 1: K=5 EXPERT SCALING
# ==========================================
def run_k5_experiment(mnist_train, kmnist_train, fmnist_train, mnist_test, kmnist_test, fmnist_test, seed=42):
    print("\n--- Running Experiment 1: K=5 Experts Scaling ---")
    set_seed(seed)
    
    # Define rotated datasets
    rot_mnist_train = RotatedDataset(mnist_train, 45)
    rot_kmnist_train = RotatedDataset(kmnist_train, 45)
    rot_mnist_test = RotatedDataset(mnist_test, 45)
    rot_kmnist_test = RotatedDataset(kmnist_test, 45)
    rot_fmnist_test = RotatedDataset(fmnist_test, 45) # OOD
    
    # Train 5 experts
    backbone = SharedBackbone()
    heads = nn.ModuleList([ExpertHead() for _ in range(5)]) # 0: MNIST, 1: KMNIST, 2: Rot MNIST, 3: Rot KMNIST, 4: FashionMNIST
    
    indices = list(range(10000))
    loaders = [
        DataLoader(Subset(mnist_train, indices), batch_size=256, shuffle=True),
        DataLoader(Subset(kmnist_train, indices), batch_size=256, shuffle=True),
        DataLoader(Subset(rot_mnist_train, indices), batch_size=256, shuffle=True),
        DataLoader(Subset(rot_kmnist_train, indices), batch_size=256, shuffle=True),
        DataLoader(Subset(fmnist_train, indices), batch_size=256, shuffle=True)
    ]
    
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(heads.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    
    backbone.train()
    heads.train()
    for epoch in range(2):
        for batches in zip(*loaders):
            optimizer.zero_grad()
            total_loss = 0.0
            for i, (x, y) in enumerate(batches):
                feats = backbone(x)
                out = heads[i](feats)
                total_loss += criterion(out, y)
            total_loss.backward()
            optimizer.step()
            
    backbone.eval()
    heads.eval()
    print("K=5 experts trained successfully.")
    
    # Compute offline prototypes & calibration distances
    cal_indices = list(range(500))
    cal_loaders = [
        DataLoader(Subset(mnist_train, cal_indices), batch_size=500, shuffle=False),
        DataLoader(Subset(kmnist_train, cal_indices), batch_size=500, shuffle=False),
        DataLoader(Subset(rot_mnist_train, cal_indices), batch_size=500, shuffle=False),
        DataLoader(Subset(rot_kmnist_train, cal_indices), batch_size=500, shuffle=False),
        DataLoader(Subset(fmnist_train, cal_indices), batch_size=500, shuffle=False)
    ]
    
    prototypes = []
    cal_dists = []
    with torch.no_grad():
        for i, loader in enumerate(cal_loaders):
            for x, _ in loader:
                feats = backbone(x)
                proto = feats.mean(dim=0)
                cal_dist = torch.mean(torch.norm(feats - proto, dim=1)).item()
                prototypes.append(proto)
                cal_dists.append(cal_dist)
                
    # Unified global calibration distance
    global_cal_dist = np.mean(cal_dists)
    
    # Build 11-phase stream (20 batches each)
    test_loaders = [
        DataLoader(mnist_test, batch_size=128, shuffle=True),
        DataLoader(kmnist_test, batch_size=128, shuffle=True),
        DataLoader(rot_mnist_test, batch_size=128, shuffle=True),
        DataLoader(rot_kmnist_test, batch_size=128, shuffle=True),
        DataLoader(fmnist_test, batch_size=128, shuffle=True),
        DataLoader(rot_fmnist_test, batch_size=128, shuffle=True) # OOD
    ]
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    iters = [iter(loader) for loader in test_loaders]
    
    # Domains: MNIST, KMNIST, RotMNIST, RotKMNIST, FashionMNIST, RotFashionMNIST
    for phase_idx, name in enumerate(['MNIST', 'KMNIST', 'RotMNIST', 'RotKMNIST', 'FashionMNIST']):
        # Clean phase
        for _ in range(20):
            (x, y), iters[phase_idx] = get_next_batch(iters[phase_idx], test_loaders[phase_idx])
            stream_batches.append(x)
            stream_labels.append(y)
            stream_domains.append(f'{name}_clean')
        # Noisy phase
        for _ in range(20):
            (x, y), iters[phase_idx] = get_next_batch(iters[phase_idx], test_loaders[phase_idx])
            noisy_x = x + 1.2 * torch.randn_like(x)
            stream_batches.append(noisy_x)
            stream_labels.append(y)
            stream_domains.append(f'{name}_noisy')
            
    # OOD phase (RotFashionMNIST)
    for _ in range(20):
        (x, y), iters[5] = get_next_batch(iters[5], test_loaders[5])
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('RotFashionMNIST_OOD')
        
    print(f"Constructed K=5 stream with {len(stream_batches)} batches.")
    
    # Simulate Gated EMA-Proto vs. Static-Proto on the 5-expert pool
    methods = ['Static-Proto', 'Ungated EMA-Proto', 'Gated EMA-Proto (Ours)']
    results = {m: [] for m in methods}
    
    alpha = 0.15
    entropy_threshold = 1.45
    dist_threshold = 1.5
    temperature = 0.25
    cw = 8.0
    
    for method in methods:
        curr_protos = [p.clone() for p in prototypes]
        curr_cal_dists = list(cal_dists)
        
        correct_counts = []
        total_counts = []
        
        for t, (x, y_true, domain) in enumerate(zip(stream_batches, stream_labels, stream_domains)):
            with torch.no_grad():
                feats = backbone(x)
                
                # Compute spatial distances to 5 prototypes
                dists = [torch.mean(torch.norm(feats - p, dim=1)).item() for p in curr_protos]
                norm_dists = [dists[i] / curr_cal_dists[i] for i in range(5)]
                
                # Evaluate 5 expert heads
                outs = [heads[i](feats) for i in range(5)]
                probs_list = [torch.softmax(out, dim=1) for out in outs]
                confs = [probs.max(dim=1)[0].mean().item() for probs in probs_list]
                
                # Confidence-Guided Routing
                logits = [-norm_dists[i]/temperature + cw * confs[i] for i in range(5)]
                routing_weights = torch.softmax(torch.tensor(logits), dim=0)
                
                # Ensemble prediction
                probs = torch.zeros_like(probs_list[0])
                for i in range(5):
                    probs += routing_weights[i].item() * probs_list[i]
                    
                entropy = compute_entropy(probs).mean().item()
                _, pred = probs.max(dim=1)
                
                acc = pred.eq(y_true).sum().item() / x.size(0)
                correct_counts.append(pred.eq(y_true).sum().item())
                total_counts.append(x.size(0))
                
                active_expert = torch.argmax(routing_weights).item()
                min_norm_dist = min(norm_dists)
                
                # Gating logic
                is_ood = (entropy > entropy_threshold or min_norm_dist > dist_threshold)
                
                gated_by_confidence = False
                max_conf_idx = np.argmax(confs)
                if active_expert != max_conf_idx:
                    # If active expert has lower confidence than some other expert
                    gated_by_confidence = True
                    
                # Prototype updates
                if method == 'Ungated EMA-Proto':
                    curr_protos[active_expert] = (1 - alpha) * curr_protos[active_expert] + alpha * feats.mean(dim=0)
                    batch_dist = torch.mean(torch.norm(feats - curr_protos[active_expert], dim=1)).item()
                    curr_cal_dists[active_expert] = (1 - alpha) * curr_cal_dists[active_expert] + alpha * batch_dist
                elif method == 'Gated EMA-Proto (Ours)':
                    if not is_ood and not gated_by_confidence:
                        curr_protos[active_expert] = (1 - alpha) * curr_protos[active_expert] + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_protos[active_expert], dim=1)).item()
                        curr_cal_dists[active_expert] = (1 - alpha) * curr_cal_dists[active_expert] + alpha * batch_dist
                        
        overall_acc = 100.0 * sum(correct_counts) / sum(total_counts)
        print(f"Method: {method:<25} | Overall Accuracy: {overall_acc:.2f}%")
        results[method] = overall_acc
        
    return results


# ==========================================
# EXPERIMENT 2: LONG-TERM STREAM (1000 batches)
# ==========================================
def run_long_term_experiment(mnist_train, kmnist_train, fmnist_test, mnist_test, kmnist_test, seed=42):
    print("\n--- Running Experiment 2: Long-Term Stream Stability (1000 Batches) ---")
    set_seed(seed)
    
    # Train standard MNIST/KMNIST model
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    kmnist_head = ExpertHead()
    
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mnist_head.parameters()) + list(kmnist_head.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    
    indices = list(range(10000))
    mnist_loader = DataLoader(Subset(mnist_train, indices), batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(Subset(kmnist_train, indices), batch_size=256, shuffle=True)
    
    backbone.train()
    mnist_head.train()
    kmnist_head.train()
    for epoch in range(2):
        for (x_m, y_m), (x_k, y_k) in zip(mnist_loader, kmnist_loader):
            optimizer.zero_grad()
            loss = criterion(mnist_head(backbone(x_m)), y_m) + criterion(kmnist_head(backbone(x_k)), y_k)
            loss.backward()
            optimizer.step()
            
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    # Compute prototypes
    cal_indices = list(range(500))
    mnist_cal = DataLoader(Subset(mnist_train, cal_indices), batch_size=500, shuffle=False)
    kmnist_cal = DataLoader(Subset(kmnist_train, cal_indices), batch_size=500, shuffle=False)
    
    with torch.no_grad():
        for x, _ in mnist_cal:
            proto_mnist = backbone(x).mean(dim=0)
            cal_dist_mnist = torch.mean(torch.norm(backbone(x) - proto_mnist, dim=1)).item()
        for x, _ in kmnist_cal:
            proto_kmnist = backbone(x).mean(dim=0)
            cal_dist_kmnist = torch.mean(torch.norm(backbone(x) - proto_kmnist, dim=1)).item()
            
    # Unified global calibration distance
    global_cal_dist = (cal_dist_mnist + cal_dist_kmnist) / 2.0
    
    # Construct 1000-batch stream (alternating MNIST and KMNIST, with noise and OOD)
    # 10 phases of 100 batches each:
    # 0. MNIST clean
    # 1. MNIST noisy std=1.2
    # 2. KMNIST clean
    # 3. KMNIST noisy std=1.2
    # 4. MNIST clean
    # 5. MNIST noisy std=1.2
    # 6. KMNIST clean
    # 7. KMNIST noisy std=1.2
    # 8. FashionMNIST OOD
    # 9. FashionMNIST OOD
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=True)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    fmnist_iter = iter(fmnist_test_loader)
    
    phases = [
        ('MNIST', False, mnist_test_loader, mnist_iter),
        ('MNIST', True, mnist_test_loader, mnist_iter),
        ('KMNIST', False, kmnist_test_loader, kmnist_iter),
        ('KMNIST', True, kmnist_test_loader, kmnist_iter),
        ('MNIST', False, mnist_test_loader, mnist_iter),
        ('MNIST', True, mnist_test_loader, mnist_iter),
        ('KMNIST', False, kmnist_test_loader, kmnist_iter),
        ('KMNIST', True, kmnist_test_loader, kmnist_iter),
        ('FashionMNIST', False, fmnist_test_loader, fmnist_iter),
        ('FashionMNIST', False, fmnist_test_loader, fmnist_iter)
    ]
    
    for domain, is_noisy, loader, iterator in phases:
        for _ in range(100):
            (x, y), iterator = get_next_batch(iterator, loader)
            if is_noisy:
                x = x + 1.2 * torch.randn_like(x)
            stream_batches.append(x)
            stream_labels.append(y)
            stream_domains.append(f"{domain}_noisy" if is_noisy else domain)
            
    print(f"Constructed long test stream with {len(stream_batches)} batches.")
    
    # Evaluate 4 versions of Gated EMA-Proto:
    # 1. Standard Gated EMA-Proto (static thresholds, no anchoring)
    # 2. Gated EMA-Proto + Anchoring ($\beta=0.0002$)
    # 3. Gated EMA-Proto + Adaptive Thresholding
    # 4. Gated EMA-Proto + Both (Anchoring + Adaptive)
    
    configs = [
        {'name': 'Gated EMA-Proto (Standard)', 'anchor': False, 'adaptive': False},
        {'name': 'Gated EMA-Proto + Anchoring', 'anchor': True, 'adaptive': False},
        {'name': 'Gated EMA-Proto + Adaptive', 'anchor': False, 'adaptive': True},
        {'name': 'Gated EMA-Proto + Both', 'anchor': True, 'adaptive': True}
    ]
    
    alpha = 0.15
    beta = 0.0002
    entropy_threshold_init = 1.45
    dist_threshold_init = 1.5
    temperature = 0.25
    cw = 8.0
    
    long_term_results = {}
    
    for config in configs:
        name = config['name']
        print(f"Simulating Stream: {name}")
        
        curr_proto_mnist = proto_mnist.clone()
        curr_proto_kmnist = proto_kmnist.clone()
        curr_cal_dist_mnist = global_cal_dist
        curr_cal_dist_kmnist = global_cal_dist
        
        entropy_threshold = entropy_threshold_init
        dist_threshold = dist_threshold_init
        
        score_buffer = [] # for rolling percentiles
        
        correct_counts = []
        total_counts = []
        drifts_mnist = []
        drifts_kmnist = []
        
        for t, (x, y_true, domain) in enumerate(zip(stream_batches, stream_labels, stream_domains)):
            with torch.no_grad():
                feats = backbone(x)
                
                dist_mnist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                dist_kmnist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                
                norm_dist_mnist = dist_mnist / curr_cal_dist_mnist
                norm_dist_kmnist = dist_kmnist / curr_cal_dist_kmnist
                
                # Track drift
                drifts_mnist.append(torch.norm(curr_proto_mnist - proto_mnist).item())
                drifts_kmnist.append(torch.norm(curr_proto_kmnist - proto_kmnist).item())
                
                out_mnist = mnist_head(feats)
                out_kmnist = kmnist_head(feats)
                
                probs_mnist = torch.softmax(out_mnist, dim=1)
                probs_kmnist = torch.softmax(out_kmnist, dim=1)
                
                conf_mnist = probs_mnist.max(dim=1)[0].mean().item()
                conf_kmnist = probs_kmnist.max(dim=1)[0].mean().item()
                
                logit_mnist = -norm_dist_mnist / temperature + cw * conf_mnist
                logit_kmnist = -norm_dist_kmnist / temperature + cw * conf_kmnist
                
                routing_weights = torch.softmax(torch.tensor([logit_mnist, logit_kmnist]), dim=0)
                w_mnist = routing_weights[0].item()
                w_kmnist = routing_weights[1].item()
                
                probs = w_mnist * probs_mnist + w_kmnist * probs_kmnist
                entropy = compute_entropy(probs).mean().item()
                
                _, pred = probs.max(dim=1)
                correct_counts.append(pred.eq(y_true).sum().item())
                total_counts.append(x.size(0))
                
                active_expert = torch.argmax(routing_weights).item()
                min_norm_dist = min(norm_dist_mnist, norm_dist_kmnist)
                
                # Adaptive Thresholding Update
                if config['adaptive']:
                    # Use a rolling percentile of the last 100 batches' scores
                    score_buffer.append((entropy, min_norm_dist))
                    if len(score_buffer) > 100:
                        score_buffer.pop(0)
                    
                    # Compute rolling 95th percentile
                    rolling_entropies = [sb[0] for sb in score_buffer]
                    rolling_dists = [sb[1] for sb in score_buffer]
                    
                    target_entropy_thresh = np.percentile(rolling_entropies, 95)
                    target_dist_thresh = np.percentile(rolling_dists, 95)
                    
                    # Exponential smoothing to prevent sudden jumps
                    entropy_threshold = 0.95 * entropy_threshold + 0.05 * target_entropy_thresh
                    dist_threshold = 0.95 * dist_threshold + 0.05 * target_dist_thresh
                    
                is_ood = (entropy > entropy_threshold or min_norm_dist > dist_threshold)
                
                gated_by_confidence = False
                if active_expert == 0 and conf_mnist < conf_kmnist:
                    gated_by_confidence = True
                elif active_expert == 1 and conf_kmnist < conf_mnist:
                    gated_by_confidence = True
                    
                # Prototype update
                if not is_ood and not gated_by_confidence:
                    if active_expert == 0:
                        curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                        curr_cal_dist_mnist = (1 - alpha) * curr_cal_dist_mnist + alpha * batch_dist
                        
                        # Apply Prototype Anchoring
                        if config['anchor']:
                            curr_proto_mnist = curr_proto_mnist - beta * (curr_proto_mnist - proto_mnist)
                    else:
                        curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                        curr_cal_dist_kmnist = (1 - alpha) * curr_cal_dist_kmnist + alpha * batch_dist
                        
                        # Apply Prototype Anchoring
                        if config['anchor']:
                            curr_proto_kmnist = curr_proto_kmnist - beta * (curr_proto_kmnist - proto_kmnist)
                            
        overall_acc = 100.0 * sum(correct_counts) / sum(total_counts)
        mean_drift_mnist = np.mean(drifts_mnist)
        mean_drift_kmnist = np.mean(drifts_kmnist)
        max_drift_mnist = np.max(drifts_mnist)
        max_drift_kmnist = np.max(drifts_kmnist)
        
        print(f"Method: {name:<30} | Acc: {overall_acc:.2f}% | Mean Drift M: {mean_drift_mnist:.4f} | Max Drift M: {max_drift_mnist:.4f}")
        long_term_results[name] = {
            'Accuracy': overall_acc,
            'Mean_Drift_MNIST': mean_drift_mnist,
            'Max_Drift_MNIST': max_drift_mnist,
            'Mean_Drift_KMNIST': mean_drift_kmnist,
            'Max_Drift_KMNIST': max_drift_kmnist
        }
        
    return long_term_results


# ==========================================
# EXPERIMENT 3: TASK-SIMILARITY-AWARE GATING (TSA-CG)
# ==========================================
def run_tsacg_experiment(mnist_train, mnist_test, seed=42):
    print("\n--- Running Experiment 3: Task-Similarity-Aware Confidence Gating ---")
    set_seed(seed)
    
    # Task A: Clean MNIST
    # Task B: Rotated MNIST by 15 degrees (Highly overlapping/similar!)
    rot_mnist_train = RotatedDataset(mnist_train, 15)
    rot_mnist_test = RotatedDataset(mnist_test, 15)
    
    # Train backbone and 2 heads
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    rot_mnist_head = ExpertHead()
    
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mnist_head.parameters()) + list(rot_mnist_head.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    
    indices = list(range(10000))
    mnist_loader = DataLoader(Subset(mnist_train, indices), batch_size=256, shuffle=True)
    rot_loader = DataLoader(Subset(rot_mnist_train, indices), batch_size=256, shuffle=True)
    
    backbone.train()
    mnist_head.train()
    rot_mnist_head.train()
    for epoch in range(2):
        for (x_m, y_m), (x_r, y_r) in zip(mnist_loader, rot_loader):
            optimizer.zero_grad()
            loss = criterion(mnist_head(backbone(x_m)), y_m) + criterion(rot_mnist_head(backbone(x_r)), y_r)
            loss.backward()
            optimizer.step()
            
    backbone.eval()
    mnist_head.eval()
    rot_mnist_head.eval()
    
    # Compute prototypes
    cal_indices = list(range(500))
    mnist_cal = DataLoader(Subset(mnist_train, cal_indices), batch_size=500, shuffle=False)
    rot_cal = DataLoader(Subset(rot_mnist_train, cal_indices), batch_size=500, shuffle=False)
    
    with torch.no_grad():
        for x, _ in mnist_cal:
            proto_mnist = backbone(x).mean(dim=0)
            cal_dist_mnist = torch.mean(torch.norm(backbone(x) - proto_mnist, dim=1)).item()
        for x, _ in rot_cal:
            proto_rot = backbone(x).mean(dim=0)
            cal_dist_rot = torch.mean(torch.norm(backbone(x) - proto_rot, dim=1)).item()
            
    # Compute similarity: cosine similarity of initial prototypes
    cos_sim = torch.dot(proto_mnist, proto_rot) / (torch.norm(proto_mnist) * torch.norm(proto_rot))
    cos_sim_val = cos_sim.item()
    print(f"Cosine Similarity between MNIST and 15-deg Rotated MNIST Prototypes: {cos_sim_val:.4f} (Very High!)")
    
    global_cal_dist = (cal_dist_mnist + cal_dist_rot) / 2.0
    
    # Build 4-phase stream of highly similar tasks (25 batches each, total 100 batches):
    # 0. MNIST clean
    # 1. 15-deg Rotated MNIST clean
    # 2. MNIST noisy std=1.2
    # 3. 15-deg Rotated MNIST noisy std=1.2
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    rot_test_loader = DataLoader(rot_mnist_test, batch_size=128, shuffle=True)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    mnist_iter = iter(mnist_test_loader)
    rot_iter = iter(rot_test_loader)
    
    phases = [
        ('MNIST', False, mnist_test_loader, mnist_iter),
        ('RotMNIST_15', False, rot_test_loader, rot_iter),
        ('MNIST', True, mnist_test_loader, mnist_iter),
        ('RotMNIST_15', True, rot_test_loader, rot_iter)
    ]
    
    for domain, is_noisy, loader, iterator in phases:
        for _ in range(25):
            (x, y), iterator = get_next_batch(iterator, loader)
            if is_noisy:
                x = x + 1.2 * torch.randn_like(x)
            stream_batches.append(x)
            stream_labels.append(y)
            stream_domains.append(f"{domain}_noisy" if is_noisy else domain)
            
    # Compare Gated EMA-Proto (Standard Confidence Gating) vs. Gated EMA-Proto + TSA-CG
    methods = ['Standard Confidence Gating', 'TSA-CG (Ours)']
    
    alpha = 0.15
    gamma = 0.5
    entropy_threshold = 1.45
    dist_threshold = 1.5
    temperature = 0.25
    cw = 8.0
    
    tsacg_results = {}
    
    for method in methods:
        print(f"Simulating Stream: {method}")
        
        curr_proto_mnist = proto_mnist.clone()
        curr_proto_rot = proto_rot.clone()
        curr_cal_dist_mnist = global_cal_dist
        curr_cal_dist_rot = global_cal_dist
        
        correct_counts = []
        total_counts = []
        gated_updates_count = 0
        total_updates_possible = 0
        
        for t, (x, y_true, domain) in enumerate(zip(stream_batches, stream_labels, stream_domains)):
            with torch.no_grad():
                feats = backbone(x)
                
                dist_mnist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                dist_rot = torch.mean(torch.norm(feats - curr_proto_rot, dim=1)).item()
                
                norm_dist_mnist = dist_mnist / curr_cal_dist_mnist
                norm_dist_rot = dist_rot / curr_cal_dist_rot
                
                out_mnist = mnist_head(feats)
                out_rot = rot_mnist_head(feats)
                
                probs_mnist = torch.softmax(out_mnist, dim=1)
                probs_rot = torch.softmax(out_rot, dim=1)
                
                conf_mnist = probs_mnist.max(dim=1)[0].mean().item()
                conf_rot = probs_rot.max(dim=1)[0].mean().item()
                
                logit_mnist = -norm_dist_mnist / temperature + cw * conf_mnist
                logit_rot = -norm_dist_rot / temperature + cw * conf_rot
                
                routing_weights = torch.softmax(torch.tensor([logit_mnist, logit_rot]), dim=0)
                w_mnist = routing_weights[0].item()
                w_rot = routing_weights[1].item()
                
                probs = w_mnist * probs_mnist + w_rot * probs_rot
                entropy = compute_entropy(probs).mean().item()
                
                _, pred = probs.max(dim=1)
                correct_counts.append(pred.eq(y_true).sum().item())
                total_counts.append(x.size(0))
                
                active_expert = torch.argmax(routing_weights).item()
                min_norm_dist = min(norm_dist_mnist, norm_dist_rot)
                
                is_ood = (entropy > entropy_threshold or min_norm_dist > dist_threshold)
                
                # Confidence gating decision
                gated_by_confidence = False
                if method == 'Standard Confidence Gating':
                    if active_expert == 0 and conf_mnist < conf_rot:
                        gated_by_confidence = True
                    elif active_expert == 1 and conf_rot < conf_mnist:
                        gated_by_confidence = True
                else: # TSA-CG
                    # Relax the gate using pairwise similarity
                    # Similarity of A and B is cos_sim_val
                    if active_expert == 0:
                        # Standard condition: conf_mnist < conf_rot
                        # TSA-CG relaxed condition: conf_mnist < conf_rot - gamma * similarity
                        if conf_mnist < conf_rot - gamma * cos_sim_val:
                            gated_by_confidence = True
                    elif active_expert == 1:
                        if conf_rot < conf_mnist - gamma * cos_sim_val:
                            gated_by_confidence = True
                            
                if is_ood or gated_by_confidence:
                    gated_updates_count += 1
                total_updates_possible += 1
                
                # Prototype update
                if not is_ood and not gated_by_confidence:
                    if active_expert == 0:
                        curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                        curr_cal_dist_mnist = (1 - alpha) * curr_cal_dist_mnist + alpha * batch_dist
                    else:
                        curr_proto_rot = (1 - alpha) * curr_proto_rot + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_rot, dim=1)).item()
                        curr_cal_dist_rot = (1 - alpha) * curr_cal_dist_rot + alpha * batch_dist
                        
        overall_acc = 100.0 * sum(correct_counts) / sum(total_counts)
        print(f"Method: {method:<30} | Acc: {overall_acc:.2f}% | Gated/Total steps: {gated_updates_count}/{total_updates_possible}")
        tsacg_results[method] = {
            'Accuracy': overall_acc,
            'Gated_Steps': gated_updates_count,
            'Total_Steps': total_updates_possible
        }
        
    return tsacg_results


if __name__ == "__main__":
    # Load Datasets
    print("Loading datasets for reviewer validation experiments...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Run the 3 experiments
    r1 = run_k5_experiment(mnist_train, kmnist_train, fmnist_train, mnist_test, kmnist_test, fmnist_test, seed=42)
    r2 = run_long_term_experiment(mnist_train, kmnist_train, fmnist_test, mnist_test, kmnist_test, seed=42)
    r3 = run_tsacg_experiment(mnist_train, mnist_test, seed=42)
    
    # Save the consolidated results
    consolidated = {
        'k5_experts': r1,
        'long_term': r2,
        'tsacg': r3
    }
    
    with open("reviewer_validation_results.json", "w") as f:
        json.dump(consolidated, f, indent=4)
    print("\nAll reviewer validation experiments executed successfully. Consolidated results saved to reviewer_validation_results.json")
