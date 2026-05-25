import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
import json

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

# Expert Head
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
            
    return backbone, mnist_head, kmnist_head

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def get_next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        new_iterator = iter(loader)
        return next(new_iterator), new_iterator

def run_experiment_for_seed(seed, mnist_train, kmnist_train, mnist_test, kmnist_test, fmnist_test):
    set_seed(seed)
    
    # Train
    backbone, mnist_head, kmnist_head = train_shared_moe(mnist_train, kmnist_train, seed, epochs=2)
    
    # Compute static prototypes and calibration
    mnist_cal_loader = DataLoader(Subset(mnist_train, list(range(500))), batch_size=500, shuffle=False)
    kmnist_cal_loader = DataLoader(Subset(kmnist_train, list(range(500))), batch_size=500, shuffle=False)
    
    with torch.no_grad():
        backbone.eval()
        for x, _ in mnist_cal_loader:
            mnist_feats = backbone(x)
            proto_mnist = mnist_feats.mean(dim=0)
            cal_dist_mnist = torch.mean(torch.norm(mnist_feats - proto_mnist, dim=1)).item()
            
        for x, _ in kmnist_cal_loader:
            kmnist_feats = backbone(x)
            proto_kmnist = kmnist_feats.mean(dim=0)
            cal_dist_kmnist = torch.mean(torch.norm(kmnist_feats - proto_kmnist, dim=1)).item()
            
    # Loaders
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=True)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    fmnist_iter = iter(fmnist_test_loader)
    
    for _ in range(50):
        (x, y), mnist_iter = get_next_batch(mnist_iter, mnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('MNIST')
        
    for _ in range(50):
        (x, y), mnist_iter = get_next_batch(mnist_iter, mnist_test_loader)
        noisy_x = x + 1.2 * torch.randn_like(x)
        stream_batches.append(noisy_x)
        stream_labels.append(y)
        stream_domains.append('Noisy_MNIST')
        
    for _ in range(50):
        (x, y), kmnist_iter = get_next_batch(kmnist_iter, kmnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('KMNIST')
        
    for _ in range(50):
        (x, y), kmnist_iter = get_next_batch(kmnist_iter, kmnist_test_loader)
        noisy_x = x + 1.2 * torch.randn_like(x)
        stream_batches.append(noisy_x)
        stream_labels.append(y)
        stream_domains.append('Noisy_KMNIST')
        
    for _ in range(50):
        (x, y), fmnist_iter = get_next_batch(fmnist_iter, fmnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('FashionMNIST')
        
    results = {}
    
    # 1. EVALUATE GATED EMA-PROTO (Ours)
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    curr_proto_mnist = proto_mnist.clone()
    curr_proto_kmnist = proto_kmnist.clone()
    curr_cal_dist_mnist = cal_dist_mnist
    curr_cal_dist_kmnist = cal_dist_kmnist
    
    alpha = 0.15 
    entropy_threshold = 1.45
    dist_threshold = 1.5  
    temperature = 0.25   
    cw = 8.0  
    
    gated_ema_accs = {d: [] for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']}
    gated_ema_latencies = []
    
    for x, y_true, domain in zip(stream_batches, stream_labels, stream_domains):
        start_time = time.perf_counter()
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
            
            is_ood = entropy > entropy_threshold or min_norm_dist > dist_threshold
            
            gated_by_confidence = False
            if active_expert == 0 and conf_mnist < conf_kmnist:
                gated_by_confidence = True
            elif active_expert == 1 and conf_kmnist < conf_mnist:
                gated_by_confidence = True
            
            if not is_ood and not gated_by_confidence:
                if active_expert == 0:
                    curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                    batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                    curr_cal_dist_mnist = (1 - alpha) * curr_cal_dist_mnist + alpha * batch_dist
                else:
                    curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * feats.mean(dim=0)
                    batch_dist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                    curr_cal_dist_kmnist = (1 - alpha) * curr_cal_dist_kmnist + alpha * batch_dist
                    
        gated_ema_latencies.append((time.perf_counter() - start_time) * 1000.0) # in ms
        gated_ema_accs[domain].append(acc)
        
    results['Gated EMA-Proto'] = {
        'accs': gated_ema_accs,
        'latencies': gated_ema_latencies,
        'params_updated': 0 # Backprop-free!
    }
    
    # 2. EVALUATE BACKPROP-BASED TTA (TTA-Entropy / Tent-like)
    # Recreate pristine trained model for TTA-Entropy to avoid cross-contamination
    backbone_tta = SharedBackbone()
    backbone_tta.load_state_dict(backbone.state_dict())
    mnist_head_tta = ExpertHead()
    mnist_head_tta.load_state_dict(mnist_head.state_dict())
    kmnist_head_tta = ExpertHead()
    kmnist_head_tta.load_state_dict(kmnist_head.state_dict())
    
    # TTA optimizations parameters
    # Optimize the classification heads to minimize Shannon entropy on the incoming batch
    optimizer_tta = optim.SGD(
        list(backbone_tta.parameters()) + list(mnist_head_tta.parameters()) + list(kmnist_head_tta.parameters()),
        lr=0.001
    )
    
    tta_accs = {d: [] for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']}
    tta_latencies = []
    
    for x, y_true, domain in zip(stream_batches, stream_labels, stream_domains):
        start_time = time.perf_counter()
        
        # TTA Step: Enable gradients
        backbone_tta.train()
        mnist_head_tta.train()
        kmnist_head_tta.train()
        
        optimizer_tta.zero_grad()
        
        feats = backbone_tta(x)
        out_mnist = mnist_head_tta(feats)
        out_kmnist = kmnist_head_tta(feats)
        
        # Ensembled soft predictions
        probs_mnist = torch.softmax(out_mnist, dim=1)
        probs_kmnist = torch.softmax(out_kmnist, dim=1)
        
        # Standard ensembling routing weight (equal routing weight 0.5 for standard TTA)
        probs = 0.5 * probs_mnist + 0.5 * probs_kmnist
        loss = compute_entropy(probs).mean()
        
        loss.backward()
        optimizer_tta.step()
        
        # Evaluation step: Eval mode
        backbone_tta.eval()
        mnist_head_tta.eval()
        kmnist_head_tta.eval()
        
        with torch.no_grad():
            feats_eval = backbone_tta(x)
            out_mnist_eval = mnist_head_tta(feats_eval)
            out_kmnist_eval = kmnist_head_tta(feats_eval)
            
            # Simple average ensemble prediction for accuracy
            probs_eval = 0.5 * torch.softmax(out_mnist_eval, dim=1) + 0.5 * torch.softmax(out_kmnist_eval, dim=1)
            _, pred = probs_eval.max(1)
            correct_count = pred.eq(y_true).sum().item()
            acc = 100.0 * correct_count / x.size(0)
            
        tta_latencies.append((time.perf_counter() - start_time) * 1000.0) # in ms
        tta_accs[domain].append(acc)
        
    total_params = sum(p.numel() for p in backbone_tta.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in mnist_head_tta.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in kmnist_head_tta.parameters() if p.requires_grad)
                   
    results['TTA-Entropy'] = {
        'accs': tta_accs,
        'latencies': tta_latencies,
        'params_updated': total_params
    }
    
    return results

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading datasets...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    seeds = [42, 43, 44, 45, 46]
    all_runs = []
    
    for seed in seeds:
        print(f"Evaluating Seed {seed}...")
        all_runs.append(run_experiment_for_seed(seed, mnist_train, kmnist_train, mnist_test, kmnist_test, fmnist_test))
        
    # Aggregate
    agg = {}
    for method in ['Gated EMA-Proto', 'TTA-Entropy']:
        agg[method] = {}
        # Accs
        for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']:
            accs = [np.mean(run[method]['accs'][d]) for run in all_runs]
            agg[method][d] = (np.mean(accs), np.std(accs))
            
        # Overall
        overall_accs = []
        for run in all_runs:
            total_acc = []
            for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']:
                total_acc.extend(run[method]['accs'][d])
            overall_accs.append(np.mean(total_acc))
        agg[method]['Overall'] = (np.mean(overall_accs), np.std(overall_accs))
        
        # Latency
        latencies = []
        for run in all_runs:
            latencies.extend(run[method]['latencies'])
        agg[method]['Latency'] = (np.mean(latencies), np.std(latencies))
        
        # Params Updated
        agg[method]['ParamsUpdated'] = all_runs[0][method]['params_updated']
        
    print("\n" + "="*90)
    print("COMPARATIVE STUDY: GATED EMA-PROTO VS. BACKPROPAGATION-BASED TTA (TTA-ENTROPY)")
    print("="*90)
    print(f"{'Metric':<30} | {'Gated EMA-Proto (Ours)':<25} | {'TTA-Entropy (Backprop)':<25}")
    print("-"*90)
    print(f"{'Overall Accuracy':<30} | {agg['Gated EMA-Proto']['Overall'][0]:6.2f} ± {agg['Gated EMA-Proto']['Overall'][1]:5.2f}%         | {agg['TTA-Entropy']['Overall'][0]:6.2f} ± {agg['TTA-Entropy']['Overall'][1]:5.2f}%")
    print(f"{'MNIST Accuracy':<30} | {agg['Gated EMA-Proto']['MNIST'][0]:6.2f} ± {agg['Gated EMA-Proto']['MNIST'][1]:5.2f}%         | {agg['TTA-Entropy']['MNIST'][0]:6.2f} ± {agg['TTA-Entropy']['MNIST'][1]:5.2f}%")
    print(f"{'Noisy MNIST Accuracy':<30} | {agg['Gated EMA-Proto']['Noisy_MNIST'][0]:6.2f} ± {agg['Gated EMA-Proto']['Noisy_MNIST'][1]:5.2f}%         | {agg['TTA-Entropy']['Noisy_MNIST'][0]:6.2f} ± {agg['TTA-Entropy']['Noisy_MNIST'][1]:5.2f}%")
    print(f"{'KMNIST Accuracy':<30} | {agg['Gated EMA-Proto']['KMNIST'][0]:6.2f} ± {agg['Gated EMA-Proto']['KMNIST'][1]:5.2f}%         | {agg['TTA-Entropy']['KMNIST'][0]:6.2f} ± {agg['TTA-Entropy']['KMNIST'][1]:5.2f}%")
    print(f"{'Noisy KMNIST Accuracy':<30} | {agg['Gated EMA-Proto']['Noisy_KMNIST'][0]:6.2f} ± {agg['Gated EMA-Proto']['Noisy_KMNIST'][1]:5.2f}%         | {agg['TTA-Entropy']['Noisy_KMNIST'][0]:6.2f} ± {agg['TTA-Entropy']['Noisy_KMNIST'][1]:5.2f}%")
    print(f"{'FashionMNIST (OOD) Accuracy':<30} | {agg['Gated EMA-Proto']['FashionMNIST'][0]:6.2f} ± {agg['Gated EMA-Proto']['FashionMNIST'][1]:5.2f}%         | {agg['TTA-Entropy']['FashionMNIST'][0]:6.2f} ± {agg['TTA-Entropy']['FashionMNIST'][1]:5.2f}%")
    print("-"*90)
    print(f"{'Avg Batch Latency (ms)':<30} | {agg['Gated EMA-Proto']['Latency'][0]:6.2f} ± {agg['Gated EMA-Proto']['Latency'][1]:5.2f} ms          | {agg['TTA-Entropy']['Latency'][0]:6.2f} ± {agg['TTA-Entropy']['Latency'][1]:5.2f} ms")
    print(f"{'Parameters Updated':<30} | {agg['Gated EMA-Proto']['ParamsUpdated']:<25} | {agg['TTA-Entropy']['ParamsUpdated']:<25}")
    print("="*90)
    
    # Save to json
    with open("tta_comparison_results.json", "w") as f:
        json.dump(agg, f, indent=4)
