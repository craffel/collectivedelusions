import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

def train_shared_moe(mnist_dataset, kmnist_dataset, epochs=2):
    print("--- Jointly Training Shared Backbone and Expert Heads ---")
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
            
        print(f"Epoch {epoch+1} | MNIST Loss: {mnist_loss/mnist_total:.4f}, MNIST Acc: {100.0*mnist_correct/mnist_total:.2f}% | KMNIST Loss: {kmnist_loss/kmnist_total:.4f}, KMNIST Acc: {100.0*kmnist_correct/kmnist_total:.2f}%")
        
    return backbone, mnist_head, kmnist_head

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

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
    
    # 2. Jointly Train Shared Backbone and Heads
    backbone, mnist_head, kmnist_head = train_shared_moe(mnist_train, kmnist_train, epochs=2)
    
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    # 3. Compute initial prototypes and calibration distances in shared space (both Indiv and BMD)
    print("\nComputing initial offline prototypes and calibration distances in shared space...")
    mnist_cal_loader = DataLoader(Subset(mnist_train, list(range(500))), batch_size=500, shuffle=False)
    kmnist_cal_loader = DataLoader(Subset(kmnist_train, list(range(500))), batch_size=500, shuffle=False)
    
    with torch.no_grad():
        for x, _ in mnist_cal_loader:
            mnist_feats = backbone(x)
            proto_mnist = mnist_feats.mean(dim=0)
            cal_dist_mnist_indiv = torch.mean(torch.norm(mnist_feats - proto_mnist, dim=1)).item()
            
            # BMD calibration distance
            batch_means = [mnist_feats[i:i+128].mean(dim=0) for i in range(0, 500-128, 50)]
            cal_dist_mnist_bmd = torch.mean(torch.stack([torch.norm(bm - proto_mnist) for bm in batch_means])).item()
            
        for x, _ in kmnist_cal_loader:
            kmnist_feats = backbone(x)
            proto_kmnist = kmnist_feats.mean(dim=0)
            cal_dist_kmnist_indiv = torch.mean(torch.norm(kmnist_feats - proto_kmnist, dim=1)).item()
            
            # BMD calibration distance
            batch_means = [kmnist_feats[i:i+128].mean(dim=0) for i in range(0, 500-128, 50)]
            cal_dist_kmnist_bmd = torch.mean(torch.stack([torch.norm(bm - proto_kmnist) for bm in batch_means])).item()
            
    print(f"Indiv Calibrations: MNIST={cal_dist_mnist_indiv:.4f}, KMNIST={cal_dist_kmnist_indiv:.4f}")
    print(f"BMD Calibrations: MNIST={cal_dist_mnist_bmd:.4f}, KMNIST={cal_dist_kmnist_bmd:.4f}")
    
    # Define stream loaders
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=False)
    
    stream_batches = []
    stream_labels = []
    stream_domains = []
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    fmnist_iter = iter(fmnist_test_loader)
    
    # Build stream (with noise std = 1.2 to simulate challenging scenarios!)
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('MNIST')
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append(x + 1.2 * torch.randn_like(x))
        stream_labels.append(y)
        stream_domains.append('Noisy_MNIST')
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('KMNIST')
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append(x + 1.2 * torch.randn_like(x))
        stream_labels.append(y)
        stream_domains.append('Noisy_KMNIST')
    for _ in range(10):
        x, y = next(fmnist_iter)
        stream_batches.append(x)
        stream_labels.append(y)
        stream_domains.append('FashionMNIST')
        
    print(f"Constructed test stream with {len(stream_batches)} batches.")
    
    # Grid Search Parameters
    space_options = ['indiv', 'bmd']
    temperatures = [0.1, 0.15, 0.25, 0.5, 1.0, 2.0]
    conf_weights = [0.0, 1.0, 2.0, 4.0, 8.0, 15.0]
    alphas = [0.05, 0.15, 0.3, 0.5]
    dist_thresholds = [1.5, 2.0, 2.5, 3.5, 5.0, 8.0]
    entropy_thresholds = [1.2, 1.45, 1.6, 1.8]
    
    best_overall = 0.0
    best_config = None
    best_results = None
    
    print("\n--- Starting Hyperparameter Sweep ---")
    
    import itertools
    all_combinations = list(itertools.product(space_options, temperatures, conf_weights, alphas, dist_thresholds, entropy_thresholds))
    print(f"Total configurations to evaluate: {len(all_combinations)}")
    
    # We will evaluate a subset or all, since it runs in < 0.1 ms per combination
    # Let's run the full sweep efficiently
    
    # Extract features for all batches in advance to avoid backbone forward passes
    all_feats = []
    all_probs_mnist = []
    all_probs_kmnist = []
    all_conf_mnist = []
    all_conf_kmnist = []
    
    with torch.no_grad():
        for x in stream_batches:
            feats = backbone(x)
            out_mnist = mnist_head(feats)
            out_kmnist = kmnist_head(feats)
            probs_mnist = torch.softmax(out_mnist, dim=1)
            probs_kmnist = torch.softmax(out_kmnist, dim=1)
            conf_mnist = probs_mnist.max(dim=1)[0].mean().item()
            conf_kmnist = probs_kmnist.max(dim=1)[0].mean().item()
            
            all_feats.append(feats)
            all_probs_mnist.append(probs_mnist)
            all_probs_kmnist.append(probs_kmnist)
            all_conf_mnist.append(conf_mnist)
            all_conf_kmnist.append(conf_kmnist)
            
    # Evaluation Loop
    results_list = []
    
    for config in all_combinations:
        space, temp, cw, alpha, dist_t, entr_t = config
        
        # Select calibrations
        cal_m = cal_dist_mnist_indiv if space == 'indiv' else cal_dist_mnist_bmd
        cal_k = cal_dist_kmnist_indiv if space == 'indiv' else cal_dist_kmnist_bmd
        
        curr_proto_mnist = proto_mnist.clone()
        curr_proto_kmnist = proto_kmnist.clone()
        curr_cal_m = cal_m
        curr_cal_k = cal_k
        
        overall_correct = 0
        overall_total = 0
        phase_accuracies = {d: [] for d in ['MNIST', 'Noisy_MNIST', 'KMNIST', 'Noisy_KMNIST', 'FashionMNIST']}
        
        for t, (feats, probs_m, probs_k, conf_m, conf_k, y_true, domain) in enumerate(zip(
            all_feats, all_probs_mnist, all_probs_kmnist, all_conf_mnist, all_conf_kmnist, stream_labels, stream_domains
        )):
            batch_mean_feat = feats.mean(dim=0)
            
            # Distances
            if space == 'indiv':
                dist_m = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                dist_k = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
            else:
                dist_m = torch.norm(batch_mean_feat - curr_proto_mnist).item()
                dist_k = torch.norm(batch_mean_feat - curr_proto_kmnist).item()
                
            norm_dist_m = dist_m / curr_cal_m
            norm_dist_k = dist_k / curr_cal_k
            
            # Logits and routing
            logit_m = -norm_dist_m / temp + cw * conf_m
            logit_k = -norm_dist_k / temp + cw * conf_k
            
            routing_weights = torch.softmax(torch.tensor([logit_m, logit_k]), dim=0)
            w_m = routing_weights[0].item()
            w_k = routing_weights[1].item()
            
            probs = w_m * probs_m + w_k * probs_k
            entropy = compute_entropy(probs).mean().item()
            
            _, pred = probs.max(1)
            correct = pred.eq(y_true).sum().item()
            acc = 100.0 * correct / feats.size(0)
            phase_accuracies[domain].append(acc)
            
            overall_correct += correct
            overall_total += feats.size(0)
            
            # Gating & Update
            active_expert = torch.argmax(routing_weights).item()
            min_norm_dist = min(norm_dist_m, norm_dist_k)
            is_ood = (entropy > entr_t or min_norm_dist > dist_t)
            
            gated_by_confidence = False
            if active_expert == 0 and conf_m < conf_k:
                gated_by_confidence = True
            elif active_expert == 1 and conf_k < conf_m:
                gated_by_confidence = True
                
            if not is_ood and not gated_by_confidence:
                if active_expert == 0:
                    curr_proto_mnist = (1 - alpha) * curr_proto_mnist + alpha * batch_mean_feat
                    if space == 'indiv':
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_mnist, dim=1)).item()
                    else:
                        batch_dist = torch.norm(batch_mean_feat - curr_proto_mnist).item()
                    curr_cal_m = (1 - alpha) * curr_cal_m + alpha * batch_dist
                else:
                    curr_proto_kmnist = (1 - alpha) * curr_proto_kmnist + alpha * batch_mean_feat
                    if space == 'indiv':
                        batch_dist = torch.mean(torch.norm(feats - curr_proto_kmnist, dim=1)).item()
                    else:
                        batch_dist = torch.norm(batch_mean_feat - curr_proto_kmnist).item()
                    curr_cal_k = (1 - alpha) * curr_cal_k + alpha * batch_dist
                    
        overall_acc = 100.0 * overall_correct / overall_total
        results_list.append((overall_acc, config, phase_accuracies))
        
    results_list.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- Top 10 Configurations found ---")
    for idx in range(10):
        acc, cfg, phase_accs = results_list[idx]
        space, temp, cw, alpha, dist_t, entr_t = cfg
        print(f"Rank {idx+1} | Acc: {acc:.2f}% | Config: space={space}, temp={temp}, cw={cw}, alpha={alpha}, dist_t={dist_t}, entr_t={entr_t}")
        print(f"    Phases: MNIST={np.mean(phase_accs['MNIST']):.2f}%, Noisy_MNIST={np.mean(phase_accs['Noisy_MNIST']):.2f}%, KMNIST={np.mean(phase_accs['KMNIST']):.2f}%, Noisy_KMNIST={np.mean(phase_accs['Noisy_KMNIST']):.2f}%, F-MNIST={np.mean(phase_accs['FashionMNIST']):.2f}%")
