import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import json
import random

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_shared_moe(mnist_dataset, kmnist_dataset, seed, epochs=2):
    print(f"--- Jointly Training Shared Backbone and Expert Heads for Seed {seed} ---")
    set_seed(seed)
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    kmnist_head = ExpertHead()
    
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mnist_head.parameters()) + list(kmnist_head.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    
    indices = list(range(12000))
    mnist_loader = DataLoader(Subset(mnist_dataset, indices), batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(Subset(kmnist_dataset, indices), batch_size=256, shuffle=True)
    
    backbone.train()
    mnist_head.train()
    kmnist_head.train()
    
    for epoch in range(epochs):
        mnist_loss, mnist_correct, mnist_total = 0.0, 0, 0
        kmnist_loss, kmnist_correct, kmnist_total = 0.0, 0, 0
        
        for (x_m, y_m), (x_k, y_k) in zip(mnist_loader, kmnist_loader):
            optimizer.zero_grad()
            
            feats_m = backbone(x_m)
            out_m = mnist_head(feats_m)
            loss_m = criterion(out_m, y_m)
            
            feats_k = backbone(x_k)
            out_k = kmnist_head(feats_k)
            loss_k = criterion(out_k, y_k)
            
            loss = loss_m + loss_k
            loss.backward()
            optimizer.step()
            
            mnist_loss += loss_m.item() * x_m.size(0)
            _, pred_m = out_m.max(1)
            mnist_total += y_m.size(0)
            mnist_correct += pred_m.eq(y_m).sum().item()
            
            kmnist_loss += loss_k.item() * x_k.size(0)
            _, pred_k = out_k.max(1)
            kmnist_total += y_k.size(0)
            kmnist_correct += pred_k.eq(y_k).sum().item()
            
    return backbone, mnist_head, kmnist_head

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def get_next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        new_iterator = iter(loader)
        return next(new_iterator), new_iterator

def run_ablation_for_seed(seed, mnist_train, kmnist_train, mnist_test, kmnist_test, fmnist_test):
    print(f"\n==================== ABLATION RUNNING SEED {seed} ====================")
    set_seed(seed)
    
    # 1. Jointly Train Shared Backbone and Heads
    backbone, mnist_head, kmnist_head = train_shared_moe(mnist_train, kmnist_train, seed, epochs=2)
    
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    # 2. Compute initial prototypes and calibration distances in shared space
    print("Computing initial offline prototypes and calibration distances...")
    mnist_cal_loader = DataLoader(Subset(mnist_train, list(range(500))), batch_size=500, shuffle=False)
    kmnist_cal_loader = DataLoader(Subset(kmnist_train, list(range(500))), batch_size=500, shuffle=False)
    
    with torch.no_grad():
        for x, _ in mnist_cal_loader:
            mnist_feats = backbone(x)
            proto_mnist = mnist_feats.mean(dim=0)
            cal_dist_mnist = torch.mean(torch.norm(mnist_feats - proto_mnist, dim=1)).item()
            
        for x, _ in kmnist_cal_loader:
            kmnist_feats = backbone(x)
            proto_kmnist = kmnist_feats.mean(dim=0)
            cal_dist_kmnist = torch.mean(torch.norm(kmnist_feats - proto_kmnist, dim=1)).item()
            
    # Define stream loaders with 50 batches per phase
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=True)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    fmnist_iter = iter(fmnist_test_loader)
    
    # Phase 0: Clean MNIST (50 batches)
    for _ in range(50):
        (x, y), mnist_iter = get_next_batch(mnist_iter, mnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('MNIST')
        
    # Phase 1: Noisy MNIST (Gaussian Noise std=1.2, 50 batches)
    for _ in range(50):
        (x, y), mnist_iter = get_next_batch(mnist_iter, mnist_test_loader)
        noisy_x = x + 1.2 * torch.randn_like(x)
        stream_batches.append(noisy_x)
        stream_labels.append(y)
        stream_domains.append('Noisy_MNIST')
        
    # Phase 2: Clean KMNIST (50 batches)
    for _ in range(50):
        (x, y), kmnist_iter = get_next_batch(kmnist_iter, kmnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('KMNIST')
        
    # Phase 3: Noisy KMNIST (Gaussian Noise std=1.2, 50 batches)
    for _ in range(50):
        (x, y), kmnist_iter = get_next_batch(kmnist_iter, kmnist_test_loader)
        noisy_x = x + 1.2 * torch.randn_like(x)
        stream_batches.append(noisy_x)
        stream_labels.append(y)
        stream_domains.append('Noisy_KMNIST')
        
    # Phase 4: Novel FashionMNIST (OOD, 50 batches)
    for _ in range(50):
        (x, y), fmnist_iter = get_next_batch(fmnist_iter, fmnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('FashionMNIST')
        
    print(f"Constructed test stream with {len(stream_batches)} batches.")
    
    ablation_modes = [
        'Static-Proto',               # No adaptation
        'Ungated EMA-Proto',          # Adaptation, no gates
        'OOD-Gated Only',             # Adaptation, only OOD gate
        'Conf-Gated Only',            # Adaptation, only Confidence gate
        'Gated EMA-Proto (Ours)'      # Adaptation, both gates
    ]
    
    seed_results = {m: {d: [] for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']} for m in ablation_modes}
    
    alpha = 0.15 
    entropy_threshold = 1.45
    dist_threshold = 1.5  
    temperature = 0.25   
    cw = 8.0  
    
    for mode in ablation_modes:
        print(f"--- Simulating Stream using ablation mode: {mode} ---")
        
        curr_proto_mnist = proto_mnist.clone()
        curr_proto_kmnist = proto_kmnist.clone()
        curr_cal_dist_mnist = cal_dist_mnist
        curr_cal_dist_kmnist = cal_dist_kmnist
        
        for t, (x, y_true, domain) in enumerate(zip(stream_batches, stream_labels, stream_domains)):
            with torch.no_grad():
                feats = backbone(x)
                
                dist_mnist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                dist_kmnist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                
                norm_dist_mnist = dist_mnist / curr_cal_dist_mnist
                norm_dist_kmnist = dist_kmnist / curr_cal_dist_kmnist
                
                out_mnist = mnist_head(feats)
                out_kmnist = kmnist_head(feats)
                
                probs_mnist = torch.softmax(out_mnist, dim=1)
                probs_kmnist = torch.softmax(out_kmnist, dim=1)
                
                conf_mnist = probs_mnist.max(dim=1)[0].mean().item()
                conf_kmnist = probs_kmnist.max(dim=1)[0].mean().item()
                
                # Confidence-Guided Routing Adjustment
                logit_mnist = -norm_dist_mnist / temperature + cw * conf_mnist
                logit_kmnist = -norm_dist_kmnist / temperature + cw * conf_kmnist
                
                routing_weights = torch.softmax(torch.tensor([logit_mnist, logit_kmnist]), dim=0)
                w_mnist = routing_weights[0].item()
                w_kmnist = routing_weights[1].item()
                
                probs = w_mnist * probs_mnist + w_kmnist * probs_kmnist
                entropy = compute_entropy(probs).mean().item()
                
                _, pred = probs.max(1)
                correct_count = pred.eq(y_true).sum().item()
                acc = 100.0 * correct_count / x.size(0)
                
                active_expert = torch.argmax(routing_weights).item()
                min_norm_dist = min(norm_dist_mnist, norm_dist_kmnist)
                
                # Flags
                is_ood = (entropy > entropy_threshold or min_norm_dist > dist_threshold)
                
                gated_by_confidence = False
                if active_expert == 0 and conf_mnist < conf_kmnist:
                    gated_by_confidence = True
                elif active_expert == 1 and conf_kmnist < conf_mnist:
                    gated_by_confidence = True
                
                # Perform updates depending on mode
                should_update = False
                if mode == 'Ungated EMA-Proto':
                    should_update = True
                elif mode == 'OOD-Gated Only':
                    should_update = not is_ood
                elif mode == 'Conf-Gated Only':
                    should_update = not gated_by_confidence
                elif mode == 'Gated EMA-Proto (Ours)':
                    should_update = (not is_ood) and (not gated_by_confidence)
                
                if should_update and mode != 'Static-Proto':
                    if active_expert == 0:
                        curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                        curr_cal_dist_mnist = (1 - alpha) * curr_cal_dist_mnist + alpha * batch_dist
                    else:
                        curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * feats.mean(dim=0)
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                        curr_cal_dist_kmnist = (1 - alpha) * curr_cal_dist_kmnist + alpha * batch_dist
                            
                seed_results[mode][domain].append(acc)
                
    return seed_results

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 1. Load Datasets
    print("Loading datasets...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    seeds = [42, 43, 44, 45, 46]
    ablation_modes = [
        'Static-Proto',               # No adaptation
        'Ungated EMA-Proto',          # Adaptation, no gates
        'OOD-Gated Only',             # Adaptation, only OOD gate
        'Conf-Gated Only',            # Adaptation, only Confidence gate
        'Gated EMA-Proto (Ours)'      # Adaptation, both gates
    ]
    domains = ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']
    
    all_runs = []
    for seed in seeds:
        seed_res = run_ablation_for_seed(seed, mnist_train, kmnist_train, mnist_test, kmnist_test, fmnist_test)
        all_runs.append(seed_res)
        
    # Aggregate results across seeds
    aggregated = {m: {d: [] for d in domains} for m in ablation_modes}
    for run in all_runs:
        for m in ablation_modes:
            for d in domains:
                aggregated[m][d].append(np.mean(run[m][d]))
                
    # Compute overall accuracies for each seed and aggregate
    aggregated_overall = {m: [] for m in ablation_modes}
    for run in all_runs:
        for m in ablation_modes:
            all_accs = []
            for d in domains:
                all_accs.extend(run[m][d])
            aggregated_overall[m].append(np.mean(all_accs))
            
    # Calculate means and standard deviations
    summary_data = {}
    print("\n" + "="*110)
    print("ROBUST MULTI-SEED ABLATION SUMMARY (Mean ± Std % across 5 seeds)")
    print("="*110)
    print(f"{'Ablation Mode':<25} | {'MNIST':<12} | {'Noisy M':<12} | {'KMNIST':<12} | {'Noisy K':<12} | {'F-MNIST':<12} | {'Overall':<12}")
    print("-"*110)
    
    for m in ablation_modes:
        row = {}
        row_str = f"{m:<25} | "
        for d in domains:
            mean_val = np.mean(aggregated[m][d])
            std_val = np.std(aggregated[m][d])
            row[d] = {"mean": mean_val, "std": std_val, "raw": aggregated[m][d]}
            row_str += f"{mean_val:5.2f} ± {std_val:4.2f}% | "
        
        overall_mean = np.mean(aggregated_overall[m])
        overall_std = np.std(aggregated_overall[m])
        row['Overall'] = {"mean": overall_mean, "std": overall_std, "raw": aggregated_overall[m]}
        row_str += f"{overall_mean:5.2f} ± {overall_std:4.2f}%"
        print(row_str)
        summary_data[m] = row
    print("="*110)
    
    # Save robust ablation results
    with open("ablation_results.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("Saved robust ablation results to ablation_results.json")
