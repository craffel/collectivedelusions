import os
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Differentiable manual Batch Normalization for 2D inputs
def manual_batch_norm2d(x, running_mean, running_var, weight, bias, eps=1e-5):
    # Reshape vectors to (1, C, 1, 1) for broadcasting
    rm = running_mean.view(1, -1, 1, 1)
    rv = running_var.view(1, -1, 1, 1)
    w = weight.view(1, -1, 1, 1)
    b = bias.view(1, -1, 1, 1)
    return (x - rm) / torch.sqrt(rv + eps) * w + b

# Differentiable custom forward pass for merged SimpleCNN model
def forward_merged(x, expert0, expert1, lambdas):
    # lambdas is a list or tensor of size 6: conv1, bn1, conv2, bn2, fc1, classifier
    # Interpolate conv1 parameters
    w_conv1 = lambdas[0] * expert0.conv1.weight + (1 - lambdas[0]) * expert1.conv1.weight
    b_conv1 = lambdas[0] * expert0.conv1.bias + (1 - lambdas[0]) * expert1.conv1.bias
    x = F.conv2d(x, w_conv1, b_conv1, padding=1)
    
    # Interpolate bn1 parameters and buffers
    w_bn1 = lambdas[1] * expert0.bn1.weight + (1 - lambdas[1]) * expert1.bn1.weight
    b_bn1 = lambdas[1] * expert0.bn1.bias + (1 - lambdas[1]) * expert1.bn1.bias
    rm_bn1 = lambdas[1] * expert0.bn1.running_mean + (1 - lambdas[1]) * expert1.bn1.running_mean
    rv_bn1 = lambdas[1] * expert0.bn1.running_var + (1 - lambdas[1]) * expert1.bn1.running_var
    x = manual_batch_norm2d(x, rm_bn1, rv_bn1, w_bn1, b_bn1)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)
    
    # Interpolate conv2 parameters
    w_conv2 = lambdas[2] * expert0.conv2.weight + (1 - lambdas[2]) * expert1.conv2.weight
    b_conv2 = lambdas[2] * expert0.conv2.bias + (1 - lambdas[2]) * expert1.conv2.bias
    x = F.conv2d(x, w_conv2, b_conv2, padding=1)
    
    # Interpolate bn2 parameters and buffers
    w_bn2 = lambdas[3] * expert0.bn2.weight + (1 - lambdas[3]) * expert1.bn2.weight
    b_bn2 = lambdas[3] * expert0.bn2.bias + (1 - lambdas[3]) * expert1.bn2.bias
    rm_bn2 = lambdas[3] * expert0.bn2.running_mean + (1 - lambdas[3]) * expert1.bn2.running_mean
    rv_bn2 = lambdas[3] * expert0.bn2.running_var + (1 - lambdas[3]) * expert1.bn2.running_var
    x = manual_batch_norm2d(x, rm_bn2, rv_bn2, w_bn2, b_bn2)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)
    
    x = torch.flatten(x, 1)
    
    # Interpolate fc1 parameters
    w_fc1 = lambdas[4] * expert0.fc1.weight + (1 - lambdas[4]) * expert1.fc1.weight
    b_fc1 = lambdas[4] * expert0.fc1.bias + (1 - lambdas[4]) * expert1.fc1.bias
    x = F.linear(x, w_fc1, b_fc1)
    x = F.relu(x)
    
    # Interpolate classifier parameters
    w_clf = lambdas[5] * expert0.classifier.weight + (1 - lambdas[5]) * expert1.classifier.weight
    b_clf = lambdas[5] * expert0.classifier.bias + (1 - lambdas[5]) * expert1.classifier.bias
    out = F.linear(x, w_clf, b_clf)
    return out

# Extract batch statistics (representation + predictive confidence) using shared base model
def extract_batch_stats(x, base_model, expert_mnist, expert_fashion):
    base_model.eval()
    expert_mnist.eval()
    expert_fashion.eval()
    with torch.no_grad():
        # Get features
        features = base_model.forward_features(x)  # B x 128
        mean_feat = features.mean(dim=0)           # 128
        std_feat = features.std(dim=0)             # 128
        
        # Get predictions and entropies
        logits_m = expert_mnist.classifier(features)
        probs_m = F.softmax(logits_m, dim=1)
        mean_prob_m = probs_m.mean(dim=0)          # 10
        ent_m = - (probs_m * torch.log(probs_m + 1e-8)).sum(dim=1).mean().unsqueeze(0) # 1
        
        logits_f = expert_fashion.classifier(features)
        probs_f = F.softmax(logits_f, dim=1)
        mean_prob_f = probs_f.mean(dim=0)          # 10
        ent_f = - (probs_f * torch.log(probs_f + 1e-8)).sum(dim=1).mean().unsqueeze(0) # 1
        
        # Concatenate into 278-dimensional batch descriptor
        stats = torch.cat([mean_feat, std_feat, mean_prob_m, ent_m, mean_prob_f, ent_f], dim=0)
    return stats

# Hypernetwork MLP Architecture
class HyperNet(nn.Module):
    def __init__(self, input_dim=278, hidden_dim=128, output_dim=6):
        super(HyperNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Sigmoid to map output directly to the [0, 1] range for merging coefficients
        out = torch.sigmoid(self.fc3(x))
        return out

def add_noise(images, sigma=0.6):
    noise = torch.randn_like(images) * sigma
    return images + noise

def main():
    set_seed(42)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    base_model = SimpleCNN().to(device)
    base_model.load_state_dict(torch.load("base_model.pth", weights_only=True))
    
    expert_mnist = SimpleCNN().to(device)
    expert_mnist.load_state_dict(torch.load("expert_mnist.pth", weights_only=True))
    
    expert_fashion = SimpleCNN().to(device)
    expert_fashion.load_state_dict(torch.load("expert_fashion.pth", weights_only=True))
    
    # Freeze models
    for m in [base_model, expert_mnist, expert_fashion]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Datasets
    print("Loading datasets splits...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=True)

    # Use first 10,000 samples of train datasets for metadata generation to completely avoid leakage
    mnist_meta_pool = Subset(mnist_train, list(range(10000)))
    fashion_meta_pool = Subset(fashion_train, list(range(10000)))
    
    mnist_meta_loader = DataLoader(mnist_meta_pool, batch_size=64, shuffle=True, drop_last=True)
    fashion_meta_loader = DataLoader(fashion_meta_pool, batch_size=64, shuffle=True, drop_last=True)

    # 1. Generating synthetic meta-dataset for the Hypernetwork
    print("\n=== Generating Synthetic Meta-Training Dataset ===")
    num_meta_batches = 1000
    meta_x_data = []
    meta_y_data = []

    mnist_iter = iter(mnist_meta_loader)
    fashion_iter = iter(fashion_meta_loader)

    start_time = time.time()
    for b in range(num_meta_batches):
        # Sample mixing ratio p
        p = random.random()
        
        # Get a batch of MNIST
        try:
            xm, ym = next(mnist_iter)
        except StopIteration:
            mnist_iter = iter(mnist_meta_loader)
            xm, ym = next(mnist_iter)
            
        # Get a batch of FashionMNIST
        try:
            xf, yf = next(fashion_iter)
        except StopIteration:
            fashion_iter = iter(fashion_meta_loader)
            xf, yf = next(fashion_iter)
            
        # Mix the batch
        num_m = int(round(64 * p))
        num_f = 64 - num_m
        
        x_mixed = torch.cat([xm[:num_m], xf[:num_f]], dim=0)
        y_mixed = torch.cat([ym[:num_m], yf[:num_f]], dim=0)
        
        # Apply a random noise transformation (0 to 0.8 sigma)
        sigma = random.uniform(0.0, 0.8)
        if sigma > 0.0:
            x_mixed = add_noise(x_mixed, sigma)
            
        x_mixed, y_mixed = x_mixed.to(device), y_mixed.to(device)
        
        # Compute batch stats (s)
        stats = extract_batch_stats(x_mixed, base_model, expert_mnist, expert_fashion)
        
        # Oracle Optimization to find optimal lambda targets on this batch
        # We find the 6 layer coefficients that minimize cross-entropy loss on the batch
        u = torch.zeros(6, requires_grad=True, device=device)
        opt = optim.Adam([u], lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        for step in range(25):
            opt.zero_grad()
            lambdas = torch.sigmoid(u)
            preds = forward_merged(x_mixed, expert_mnist, expert_fashion, lambdas)
            loss = criterion(preds, y_mixed)
            loss.backward()
            opt.step()
            
        optimal_lambdas = torch.sigmoid(u).detach().cpu()
        
        meta_x_data.append(stats.cpu())
        meta_y_data.append(optimal_lambdas)
        
        if (b + 1) % 100 == 0:
            print(f"Generated {b+1}/{num_meta_batches} batches in {time.time() - start_time:.1f}s")

    meta_x_data = torch.stack(meta_x_data)  # N x 278
    meta_y_data = torch.stack(meta_y_data)  # N x 6
    print(f"Meta-dataset ready. Input shape: {meta_x_data.shape}, Target shape: {meta_y_data.shape}")

    # 2. Train the Hypernetwork
    print("\n=== Training the Hypernetwork ===")
    hypernet = HyperNet().to(device)
    optimizer = optim.Adam(hypernet.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train-test split of meta-dataset (80/20)
    n_samples = len(meta_x_data)
    indices = list(range(n_samples))
    random.shuffle(indices)
    split = int(n_samples * 0.8)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_x, train_y = meta_x_data[train_indices].to(device), meta_y_data[train_indices].to(device)
    val_x, val_y = meta_x_data[val_indices].to(device), meta_y_data[val_indices].to(device)

    best_val_loss = float('inf')
    epochs = 150
    for epoch in range(epochs):
        hypernet.train()
        optimizer.zero_grad()
        preds = hypernet(train_x)
        loss = F.mse_loss(preds, train_y)
        loss.backward()
        optimizer.step()
        
        hypernet.eval()
        with torch.no_grad():
            val_preds = hypernet(val_x)
            val_loss = F.mse_loss(val_preds, val_y).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(hypernet.state_dict(), "best_hypernet.pth")
            
        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train MSE: {loss.item():.5f} - Val MSE: {val_loss:.5f}")

    hypernet.load_state_dict(torch.load("best_hypernet.pth", weights_only=True))
    print(f"Hypernetwork trained and saved. Best Val MSE: {best_val_loss:.5f}")

    # 3. Construct evaluation non-stationary stream (50 batches of size 64)
    print("\n=== Constructing Evaluation Stream ===")
    # Load test loader (independent test splits)
    mnist_eval_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fashion_eval_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
    kmnist_eval_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    eval_mnist_iter = iter(mnist_eval_loader)
    eval_fashion_iter = iter(fashion_eval_loader)
    eval_kmnist_iter = iter(kmnist_eval_loader)

    # 5 Phases:
    # Phase 0: Clean MNIST (0-9)
    # Phase 1: Noisy MNIST (10-19)
    # Phase 2: Clean Fashion (20-29)
    # Phase 3: Noisy Fashion (30-39)
    # Phase 4: Novel KMNIST (40-49)
    test_stream = []
    for batch_idx in range(50):
        phase = batch_idx // 10
        if phase == 0:
            x, y = next(eval_mnist_iter)
        elif phase == 1:
            x, y = next(eval_mnist_iter)
            x = add_noise(x, 0.6)
        elif phase == 2:
            x, y = next(eval_fashion_iter)
        elif phase == 3:
            x, y = next(eval_fashion_iter)
            x = add_noise(x, 0.6)
        elif phase == 4:
            x, y = next(eval_kmnist_iter)
        test_stream.append((x.to(device), y.to(device)))
    print(f"Evaluation stream of 50 batches built successfully.")

    # 4. Evaluation Loop for all Methods
    methods = [
        "Expert MNIST Only",
        "Expert Fashion Only",
        "Uniform Merging (0.5/0.5)",
        "Oracle Merging (Ceiling)",
        "Gradient-based TTA (5 steps)",
        "Hyper-TTMM (Ours, Zero-Shot)"
    ]
    
    results = {m: {"accuracies": [], "latencies": [], "phase_accs": [[] for _ in range(5)]} for m in methods}

    print("\n=== Beginning Evaluation ===")
    for m_name in methods:
        print(f"\nEvaluating: {m_name}...")
        
        for idx, (x, y) in enumerate(test_stream):
            phase = idx // 10
            
            # Start timer
            start_time = time.perf_counter()
            
            if m_name == "Expert MNIST Only":
                lambdas = torch.ones(6, device=device)
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Expert Fashion Only":
                lambdas = torch.zeros(6, device=device)
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Uniform Merging (0.5/0.5)":
                lambdas = torch.full((6,), 0.5, device=device)
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Oracle Merging (Ceiling)":
                # Ground-truth labels are used to find optimal merging lambdas
                u = torch.zeros(6, requires_grad=True, device=device)
                opt = optim.Adam([u], lr=0.1)
                criterion = nn.CrossEntropyLoss()
                for step in range(30):
                    opt.zero_grad()
                    lmbds = torch.sigmoid(u)
                    outs = forward_merged(x, expert_mnist, expert_fashion, lmbds)
                    loss = criterion(outs, y)
                    loss.backward()
                    opt.step()
                lambdas = torch.sigmoid(u).detach()
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Gradient-based TTA (5 steps)":
                # Entropy minimization on layer coefficients
                # Initialize at uniform [0.5, 0.5, ...]
                u = torch.zeros(6, requires_grad=True, device=device)
                opt = optim.SGD([u], lr=0.05)
                for step in range(5):
                    opt.zero_grad()
                    lmbds = torch.sigmoid(u)
                    outs = forward_merged(x, expert_mnist, expert_fashion, lmbds)
                    probs = F.softmax(outs, dim=1)
                    ent = - (probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    ent.backward()
                    opt.step()
                lambdas = torch.sigmoid(u).detach()
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
                
            elif m_name == "Hyper-TTMM (Ours, Zero-Shot)":
                # Zero-shot prediction using our trained Hypernetwork
                stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)
                with torch.no_grad():
                    lambdas = hypernet(stats.to(device))
                preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)

            # Measure latency
            latency = time.perf_counter() - start_time
            
            # Compute accuracy
            _, pred_classes = torch.max(preds, 1)
            correct = (pred_classes == y).sum().item()
            acc = correct / len(y)
            
            results[m_name]["accuracies"].append(acc)
            results[m_name]["latencies"].append(latency)
            results[m_name]["phase_accs"][phase].append(acc)

        # Print statistics
        all_accs = results[m_name]["accuracies"]
        all_lats = results[m_name]["latencies"]
        avg_acc = np.mean(all_accs) * 100
        avg_lat = np.mean(all_lats) * 1000 # in ms
        print(f"-> Overall Acc: {avg_acc:.2f}% | Avg Latency: {avg_lat:.2f}ms")
        for p in range(5):
            p_acc = np.mean(results[m_name]["phase_accs"][p]) * 100
            print(f"   Phase {p} (segment {p*10}-{p*10+9}) Acc: {p_acc:.2f}%")

    # Save results as JSON
    final_report = {}
    for m in methods:
        final_report[m] = {
            "overall_accuracy": float(np.mean(results[m]["accuracies"]) * 100),
            "average_latency_ms": float(np.mean(results[m]["latencies"]) * 1000),
            "phase_accuracies": [float(np.mean(results[m]["phase_accs"][p]) * 100) for p in range(5)]
        }
        
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    print("\nResults successfully saved to results.json.")

if __name__ == "__main__":
    main()
