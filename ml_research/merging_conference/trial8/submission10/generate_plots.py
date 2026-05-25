import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
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

def main():
    set_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading datasets for plotting...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Jointly Train Shared Backbone and Heads
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    kmnist_head = ExpertHead()
    
    # Train
    print("Training models briefly...")
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
    
    # Compute initial prototypes
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
            
    # Stream loaders
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=True)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=True)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    fmnist_iter = iter(fmnist_test_loader)
    
    # Construct stream: 50 batches per phase
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
        
    methods = ['Static-Proto (Baseline)', 'Ungated EMA-Proto', 'Gated EMA-Proto (Ours)']
    
    # We will record proto-distance drift and routing weights
    # For distance drift, we'll track ||curr_proto_mnist - initial_proto_mnist||_2
    # and ||curr_proto_kmnist - initial_proto_kmnist||_2
    
    proto_drifts_m = {m: [] for m in methods}
    proto_drifts_k = {m: [] for m in methods}
    routing_weights_m = {m: [] for m in methods}
    
    alpha = 0.15 
    entropy_threshold = 1.45
    dist_threshold = 1.5  
    temperature = 0.25   
    cw = 8.0  
    
    for method in methods:
        print(f"Tracking run for {method}...")
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
                
                active_expert = torch.argmax(routing_weights).item()
                min_norm_dist = min(norm_dist_mnist, norm_dist_kmnist)
                
                is_ood = (entropy > entropy_threshold or min_norm_dist > dist_threshold)
                gated_by_confidence = False
                if active_expert == 0 and conf_mnist < conf_kmnist:
                    gated_by_confidence = True
                elif active_expert == 1 and conf_kmnist < conf_mnist:
                    gated_by_confidence = True
                
                # Update prototypes using EMA
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
                
                # Record metrics
                drift_m = torch.norm(curr_proto_mnist - proto_mnist).item()
                drift_k = torch.norm(curr_proto_kmnist - proto_kmnist).item()
                proto_drifts_m[method].append(drift_m)
                proto_drifts_k[method].append(drift_k)
                routing_weights_m[method].append(w_mnist)
                
    # Plot 1: Prototype Drift Over Time
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_steps = np.arange(len(stream_batches))
    
    # MNIST prototype drift
    for m in methods:
        axes[0].plot(x_steps, proto_drifts_m[m], label=m, lw=2)
    axes[0].set_title("MNIST Prototype Drift", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Batch Step", fontsize=10)
    axes[0].set_ylabel(r"$||\mu_{MNIST}^{(t)} - \mu_{MNIST}^{(0)}||_2$", fontsize=10)
    axes[0].grid(True, ls='--', alpha=0.5)
    
    # Vertical phase split lines
    for p in [50, 100, 150, 200]:
        axes[0].axvline(x=p, color='grey', ls=':', alpha=0.7)
    
    # Add phase labels
    axes[0].text(25, max(proto_drifts_m['Ungated EMA-Proto']) * 0.9, "M", ha='center', fontsize=9, color='grey')
    axes[0].text(75, max(proto_drifts_m['Ungated EMA-Proto']) * 0.9, "Noisy M", ha='center', fontsize=9, color='grey')
    axes[0].text(125, max(proto_drifts_m['Ungated EMA-Proto']) * 0.9, "KM", ha='center', fontsize=9, color='grey')
    axes[0].text(175, max(proto_drifts_m['Ungated EMA-Proto']) * 0.9, "Noisy KM", ha='center', fontsize=9, color='grey')
    axes[0].text(225, max(proto_drifts_m['Ungated EMA-Proto']) * 0.9, "Fashion", ha='center', fontsize=9, color='grey')
    
    # KMNIST prototype drift
    for m in methods:
        axes[1].plot(x_steps, proto_drifts_k[m], label=m, lw=2)
    axes[1].set_title("KMNIST Prototype Drift", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Batch Step", fontsize=10)
    axes[1].set_ylabel(r"$||\mu_{KMNIST}^{(t)} - \mu_{KMNIST}^{(0)}||_2$", fontsize=10)
    axes[1].grid(True, ls='--', alpha=0.5)
    
    for p in [50, 100, 150, 200]:
        axes[1].axvline(x=p, color='grey', ls=':', alpha=0.7)
        
    # Add phase labels
    axes[1].text(25, max(proto_drifts_k['Ungated EMA-Proto']) * 0.9, "M", ha='center', fontsize=9, color='grey')
    axes[1].text(75, max(proto_drifts_k['Ungated EMA-Proto']) * 0.9, "Noisy M", ha='center', fontsize=9, color='grey')
    axes[1].text(125, max(proto_drifts_k['Ungated EMA-Proto']) * 0.9, "KM", ha='center', fontsize=9, color='grey')
    axes[1].text(175, max(proto_drifts_k['Ungated EMA-Proto']) * 0.9, "Noisy KM", ha='center', fontsize=9, color='grey')
    axes[1].text(225, max(proto_drifts_k['Ungated EMA-Proto']) * 0.9, "Fashion", ha='center', fontsize=9, color='grey')
    
    axes[1].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("prototype_drift.pdf", bbox_inches='tight')
    plt.close()
    
    # Plot 2: Routing Weights Over Time
    plt.figure(figsize=(10, 3.5))
    for m in methods:
        # Simple moving average to make routing weights smoother/readable
        weights = np.array(routing_weights_m[m])
        plt.plot(x_steps, weights, label=m, lw=2)
        
    plt.title("Evolution of MNIST Routing Weight over the Stream", fontsize=11, fontweight='bold')
    plt.xlabel("Batch Step", fontsize=10)
    plt.ylabel(r"Routing Weight $w_{MNIST}$", fontsize=10)
    plt.grid(True, ls='--', alpha=0.5)
    plt.ylim(-0.05, 1.05)
    
    for p in [50, 100, 150, 200]:
        plt.axvline(x=p, color='grey', ls=':', alpha=0.7)
        
    plt.text(25, 0.5, "MNIST", ha='center', fontsize=10, color='grey', fontweight='bold')
    plt.text(75, 0.5, "Noisy MNIST", ha='center', fontsize=10, color='grey', fontweight='bold')
    plt.text(125, 0.5, "KMNIST", ha='center', fontsize=10, color='grey', fontweight='bold')
    plt.text(175, 0.5, "Noisy KMNIST", ha='center', fontsize=10, color='grey', fontweight='bold')
    plt.text(225, 0.5, "Fashion (OOD)", ha='center', fontsize=10, color='grey', fontweight='bold')
    
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("routing_weights.pdf", bbox_inches='tight')
    plt.close()
    
    print("Successfully generated prototype_drift.pdf and routing_weights.pdf!")

if __name__ == "__main__":
    main()
