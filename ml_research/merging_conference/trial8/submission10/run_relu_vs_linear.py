import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
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

# Expert Head WITH ReLU
class ExpertHeadWithReLU(nn.Module):
    def __init__(self):
        super(ExpertHeadWithReLU, self).__init__()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, feats):
        return self.fc2(self.relu(feats))

# Expert Head WITHOUT ReLU (Purely Linear)
class ExpertHeadLinear(nn.Module):
    def __init__(self):
        super(ExpertHeadLinear, self).__init__()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, feats):
        return self.fc2(feats)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_shared_moe(mnist_dataset, kmnist_dataset, seed, use_relu=True, epochs=2):
    set_seed(seed)
    backbone = SharedBackbone()
    mnist_head = ExpertHeadWithReLU() if use_relu else ExpertHeadLinear()
    kmnist_head = ExpertHeadWithReLU() if use_relu else ExpertHeadLinear()
    
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

def run_experiment(use_relu=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    seeds = [42, 43, 44, 45, 46]
    methods = ['Static-Proto (Baseline)', 'Ungated EMA-Proto', 'Gated EMA-Proto (Ours)']
    
    alpha = 0.15 
    entropy_threshold = 1.45
    dist_threshold = 1.5  
    temperature = 0.25   
    cw = 8.0  
    
    all_runs = []
    
    for seed in seeds:
        set_seed(seed)
        backbone, mnist_head, kmnist_head = train_shared_moe(mnist_train, kmnist_train, seed, use_relu=use_relu, epochs=2)
        
        backbone.eval()
        mnist_head.eval()
        kmnist_head.eval()
        
        # Compute prototypes
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
                
        # Build stream
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
            
        seed_results = {m: {d: [] for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']} for m in methods}
        
        for method in methods:
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
                    
                    if method == 'Ungated EMA-Proto':
                        if active_expert == 0:
                            curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                            batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                            curr_cal_dist_mnist = (1 - alpha) * curr_cal_dist_mnist + alpha * batch_dist
                        else:
                            curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * feats.mean(dim=0)
                            batch_dist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                            curr_cal_dist_kmnist = (1 - alpha) * curr_cal_dist_kmnist + alpha * batch_dist
                    elif method == 'Gated EMA-Proto (Ours)':
                        if not is_ood and not gated_by_confidence:
                            if active_expert == 0:
                                curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * feats.mean(dim=0)
                                batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                                curr_cal_dist_mnist = (1 - alpha) * curr_cal_dist_mnist + alpha * batch_dist
                            else:
                                curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * feats.mean(dim=0)
                                batch_dist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                                curr_cal_dist_kmnist = (1 - alpha) * curr_cal_dist_kmnist + alpha * batch_dist
                                
                    seed_results[method][domain].append(acc)
        all_runs.append(seed_results)
        
    # Aggregate results
    agg = {}
    for method in methods:
        agg[method] = {}
        for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']:
            runs_acc = [np.mean(run[method][d]) for run in all_runs]
            agg[method][d] = (np.mean(runs_acc), np.std(runs_acc))
            
        # Overall
        overall_runs_acc = []
        for run in all_runs:
            total_acc = []
            for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']:
                total_acc.extend(run[method][d])
            overall_runs_acc.append(np.mean(total_acc))
        agg[method]['Overall'] = (np.mean(overall_runs_acc), np.std(overall_runs_acc))
        
    return agg

if __name__ == "__main__":
    print("Evaluating Standard Gated EMA-Proto (with ReLU)...")
    with_relu_results = run_experiment(use_relu=True)
    
    print("\nEvaluating Ablated Gated EMA-Proto (Purely Linear)...")
    linear_results = run_experiment(use_relu=False)
    
    print("\n" + "="*80)
    print("COMPARATIVE STUDY: EXPERT HEADS WITH RELU VS. PURELY LINEAR")
    print("="*80)
    print(f"{'Method / Configuration':<30} | {'With ReLU (Standard)':<22} | {'Purely Linear (Ablated)':<22}")
    print("-"*80)
    for method in ['Static-Proto (Baseline)', 'Ungated EMA-Proto', 'Gated EMA-Proto (Ours)']:
        mean_r, std_r = with_relu_results[method]['Overall']
        mean_l, std_l = linear_results[method]['Overall']
        print(f"{method:<30} | {mean_r:6.2f} ± {std_r:5.2f}%         | {mean_l:6.2f} ± {std_l:5.2f}%")
    print("="*80)
    
    # Let's save the JSON output
    comparison = {
        "with_relu": with_relu_results,
        "purely_linear": linear_results
    }
    with open("relu_vs_linear_results.json", "w") as f:
        json.dump(comparison, f, indent=4)
