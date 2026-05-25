import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
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

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def get_next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        new_iterator = iter(loader)
        return next(new_iterator), new_iterator

def run_with_params(backbone, mnist_head, kmnist_head, proto_m, proto_k, cal_m, cal_k, stream_batches, stream_labels, stream_domains, alpha, cw, dist_t, temp, entr_t):
    curr_proto_mnist = proto_m.clone()
    curr_proto_kmnist = proto_k.clone()
    curr_cal_dist_mnist = cal_m
    curr_cal_dist_kmnist = cal_k
    
    correct_all = 0
    total_all = 0
    
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
            
            logit_mnist = -norm_dist_mnist / temp + cw * conf_mnist
            logit_kmnist = -norm_dist_kmnist / temp + cw * conf_kmnist
            
            routing_weights = torch.softmax(torch.tensor([logit_mnist, logit_kmnist]), dim=0)
            w_mnist = routing_weights[0].item()
            w_kmnist = routing_weights[1].item()
            
            probs = w_mnist * probs_mnist + w_kmnist * probs_kmnist
            entropy = compute_entropy(probs).mean().item()
            
            _, pred = probs.max(1)
            correct_count = pred.eq(y_true).sum().item()
            correct_all += correct_count
            total_all += x.size(0)
            
            active_expert = torch.argmax(routing_weights).item()
            min_norm_dist = min(norm_dist_mnist, norm_dist_kmnist)
            
            is_ood = (entropy > entr_t or min_norm_dist > dist_t)
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
                    
    return 100.0 * correct_all / total_all

def main():
    set_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading datasets for sensitivity analysis...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Backbone
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    kmnist_head = ExpertHead()
    
    # Quick Train
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(mnist_head.parameters()) + list(kmnist_head.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    indices = list(range(12000))
    mnist_loader = DataLoader(Subset(mnist_train, indices), batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(Subset(kmnist_train, indices), batch_size=256, shuffle=True)
    
    backbone.train()
    mnist_head.train()
    kmnist_head.train()
    for epoch in range(2):
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
            
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    # Initial prototypes
    mnist_cal_loader = DataLoader(Subset(mnist_train, list(range(500))), batch_size=500, shuffle=False)
    kmnist_cal_loader = DataLoader(Subset(kmnist_train, list(range(500))), batch_size=500, shuffle=False)
    with torch.no_grad():
        for x, _ in mnist_cal_loader:
            mnist_feats = backbone(x)
            proto_m = mnist_feats.mean(dim=0)
            cal_m = torch.mean(torch.norm(mnist_feats - proto_m, dim=1)).item()
        for x, _ in kmnist_cal_loader:
            kmnist_feats = backbone(x)
            proto_k = kmnist_feats.mean(dim=0)
            cal_k = torch.mean(torch.norm(kmnist_feats - proto_k, dim=1)).item()
            
    # Stream loaders
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=True)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    mnist_iter, kmnist_iter, fmnist_iter = iter(mnist_test_loader), iter(kmnist_test_loader), iter(fmnist_test_loader)
    
    for _ in range(50):
        (x, y), mnist_iter = get_next_batch(mnist_iter, mnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('MNIST')
    for _ in range(50):
        (x, y), mnist_iter = get_next_batch(mnist_iter, mnist_test_loader)
        stream_batches.append(x + 1.2 * torch.randn_like(x))
        stream_labels.append(y)
        stream_domains.append('Noisy_MNIST')
    for _ in range(50):
        (x, y), kmnist_iter = get_next_batch(kmnist_iter, kmnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('KMNIST')
    for _ in range(50):
        (x, y), kmnist_iter = get_next_batch(kmnist_iter, kmnist_test_loader)
        stream_batches.append(x + 1.2 * torch.randn_like(x))
        stream_labels.append(y)
        stream_domains.append('Noisy_KMNIST')
    for _ in range(50):
        (x, y), fmnist_iter = get_next_batch(fmnist_iter, fmnist_test_loader)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('FashionMNIST')
        
    # Hyperparameter defaults
    d_alpha = 0.15
    d_cw = 8.0
    d_dist_t = 1.5
    d_temp = 0.25
    d_entr_t = 1.45
    
    # 1. Sensitivity to Alpha
    alphas = [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]
    alpha_accs = []
    for a in alphas:
        acc = run_with_params(backbone, mnist_head, kmnist_head, proto_m, proto_k, cal_m, cal_k, stream_batches, stream_labels, stream_domains, a, d_cw, d_dist_t, d_temp, d_entr_t)
        alpha_accs.append(acc)
        
    # 2. Sensitivity to cw
    cws = [0.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    cw_accs = []
    for c in cws:
        acc = run_with_params(backbone, mnist_head, kmnist_head, proto_m, proto_k, cal_m, cal_k, stream_batches, stream_labels, stream_domains, d_alpha, c, d_dist_t, d_temp, d_entr_t)
        cw_accs.append(acc)
        
    # 3. Sensitivity to dist_threshold (tau_d)
    dist_ts = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    dist_accs = []
    for dt in dist_ts:
        acc = run_with_params(backbone, mnist_head, kmnist_head, proto_m, proto_k, cal_m, cal_k, stream_batches, stream_labels, stream_domains, d_alpha, d_cw, dt, d_temp, d_entr_t)
        dist_accs.append(acc)

    print("\n--- Sensitivity Results ---")
    print("Alpha sensitivity:")
    for a, acc in zip(alphas, alpha_accs):
        print(f"  alpha = {a:.2f}: {acc:.2f}%")
        
    print("\nConfidence Weight (cw) sensitivity:")
    for c, acc in zip(cws, cw_accs):
        print(f"  cw = {c:.1f}: {acc:.2f}%")
        
    print("\nDistance Threshold (tau_d) sensitivity:")
    for dt, acc in zip(dist_ts, dist_accs):
        print(f"  tau_d = {dt:.2f}: {acc:.2f}%")

if __name__ == "__main__":
    main()
