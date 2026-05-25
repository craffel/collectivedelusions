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
    rm = running_mean.view(1, -1, 1, 1)
    rv = running_var.view(1, -1, 1, 1)
    w = weight.view(1, -1, 1, 1)
    b = bias.view(1, -1, 1, 1)
    return (x - rm) / torch.sqrt(rv + eps) * w + b

# Differentiable custom forward pass for merged SimpleCNN model
def forward_merged(x, expert0, expert1, lambdas):
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

# Configurable Hypernetwork Architecture
class ConfigurableHyperNet(nn.Module):
    def __init__(self, input_dim=278, hidden_layers=2, hidden_dim=128, output_dim=6):
        super(ConfigurableHyperNet, self).__init__()
        layers = []
        curr_dim = input_dim
        
        if hidden_layers == 0:
            # Linear model
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Sigmoid()
            )
        else:
            for _ in range(hidden_layers):
                layers.append(nn.Linear(curr_dim, hidden_dim))
                layers.append(nn.ReLU())
                curr_dim = hidden_dim
            layers.append(nn.Linear(curr_dim, output_dim))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def add_noise(images, sigma=0.6):
    noise = torch.randn_like(images) * sigma
    return images + noise

# Training helper for Hypernetwork
def train_hypernetwork(meta_x, meta_y, input_dim, hidden_layers, hidden_dim, epochs=150, lr=1e-3, weight_decay=1e-5, device="cpu"):
    set_seed(42)
    hypernet = ConfigurableHyperNet(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(hypernet.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Split 80/20
    n_samples = len(meta_x)
    indices = list(range(n_samples))
    random.shuffle(indices)
    split = int(n_samples * 0.8)
    
    train_x, train_y = meta_x[indices[:split]].to(device), meta_y[indices[:split]].to(device)
    val_x, val_y = meta_x[indices[split:]].to(device), meta_y[indices[split:]].to(device)
    
    best_val_loss = float('inf')
    best_state = None
    
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
            best_state = {k: v.cpu() for k, v in hypernet.state_dict().items()}
            
    hypernet.load_state_dict(best_state)
    return hypernet, best_val_loss

def main():
    set_seed(42)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load expert models
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

    print("Loading datasets splits...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=True)

    # Use first 10,000 samples of train datasets for metadata generation
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
        p = random.random()
        
        try:
            xm, ym = next(mnist_iter)
        except StopIteration:
            mnist_iter = iter(mnist_meta_loader)
            xm, ym = next(mnist_iter)
            
        try:
            xf, yf = next(fashion_iter)
        except StopIteration:
            fashion_iter = iter(fashion_meta_loader)
            xf, yf = next(fashion_iter)
            
        num_m = int(round(64 * p))
        num_f = 64 - num_m
        
        x_mixed = torch.cat([xm[:num_m], xf[:num_f]], dim=0)
        y_mixed = torch.cat([ym[:num_m], yf[:num_f]], dim=0)
        
        sigma = random.uniform(0.0, 0.8)
        if sigma > 0.0:
            x_mixed = add_noise(x_mixed, sigma)
            
        x_mixed, y_mixed = x_mixed.to(device), y_mixed.to(device)
        
        # Compute batch stats (s)
        stats = extract_batch_stats(x_mixed, base_model, expert_mnist, expert_fashion)
        
        # Oracle Optimization
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

    meta_x_data = torch.stack(meta_x_data)  # 1000 x 278
    meta_y_data = torch.stack(meta_y_data)  # 1000 x 6
    print(f"Meta-dataset ready. Input shape: {meta_x_data.shape}, Target shape: {meta_y_data.shape}")

    # Construct evaluation non-stationary stream (50 batches of size 64)
    print("\n=== Constructing Evaluation Stream ===")
    mnist_eval_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fashion_eval_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
    kmnist_eval_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    eval_mnist_iter = iter(mnist_eval_loader)
    eval_fashion_iter = iter(fashion_eval_loader)
    eval_kmnist_iter = iter(kmnist_eval_loader)

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
    print(f"Evaluation stream built.")

    # Evaluator helper
    def evaluate_model(lambdas_generator_fn, is_zero_shot=True):
        accuracies = []
        latencies = []
        phase_accs = [[] for _ in range(5)]
        
        for idx, (x, y) in enumerate(test_stream):
            phase = idx // 10
            start_time = time.perf_counter()
            
            lambdas = lambdas_generator_fn(x, y)
            preds = forward_merged(x, expert_mnist, expert_fashion, lambdas)
            
            latency = time.perf_counter() - start_time
            _, pred_classes = torch.max(preds, 1)
            correct = (pred_classes == y).sum().item()
            acc = correct / len(y)
            
            accuracies.append(acc)
            latencies.append(latency)
            phase_accs[phase].append(acc)
            
        return {
            "overall_accuracy": float(np.mean(accuracies) * 100),
            "average_latency_ms": float(np.mean(latencies) * 1000),
            "phase_accuracies": [float(np.mean(phase_accs[p]) * 100) for p in range(5)]
        }

    # PART 1: Main Evaluation of 6 Methods
    print("\n=== Running Part 1: Main Evaluation ===")
    
    # Train the default Hypernetwork first
    print("Training Default Hypernetwork (MLP-Medium, 2x128, Full Stats, 1000 Samples)...")
    default_hypernet, val_mse_default = train_hypernetwork(
        meta_x_data, meta_y_data, input_dim=278, hidden_layers=2, hidden_dim=128, device=device
    )
    # Save the default model
    torch.save(default_hypernet.state_dict(), "best_hypernet.pth")
    print(f"Default Hypernetwork trained with Val MSE: {val_mse_default:.5f}")

    main_results = {}

    # 1. Expert MNIST Only
    main_results["Expert MNIST Only"] = evaluate_model(
        lambda x, y: torch.ones(6, device=device)
    )
    # 2. Expert Fashion Only
    main_results["Expert Fashion Only"] = evaluate_model(
        lambda x, y: torch.zeros(6, device=device)
    )
    # 3. Uniform Merging
    main_results["Uniform Merging (0.5/0.5)"] = evaluate_model(
        lambda x, y: torch.full((6,), 0.5, device=device)
    )
    # 4. Oracle Merging
    def oracle_gen(x, y):
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
        return torch.sigmoid(u).detach()
    main_results["Oracle Merging (Ceiling)"] = evaluate_model(oracle_gen)

    # 5. Gradient-based TTA
    def tta_gen(x, y):
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
        return torch.sigmoid(u).detach()
    main_results["Gradient-based TTA (5 steps)"] = evaluate_model(tta_gen)

    # 6. Hyper-TTMM (Ours)
    def hyper_gen(x, y):
        stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)
        with torch.no_grad():
            lambdas = default_hypernet(stats.to(device))
        return lambdas
    main_results["Hyper-TTMM (Ours, Zero-Shot)"] = evaluate_model(hyper_gen)

    # Print main results
    for m, res in main_results.items():
        print(f"Method: {m:35s} | Acc: {res['overall_accuracy']:.2f}% | Latency: {res['average_latency_ms']:.2f}ms")

    # PART 2: Ablation Study on Hypernetwork Architecture
    print("\n=== Running Part 2: Ablation on Architectures ===")
    arch_results = {}
    
    # 1. Linear Architecture
    print("Training Linear Hypernetwork (0 hidden layers)...")
    linear_net, val_mse_linear = train_hypernetwork(
        meta_x_data, meta_y_data, input_dim=278, hidden_layers=0, hidden_dim=0, device=device
    )
    arch_results["Linear"] = {
        "val_mse": float(val_mse_linear),
        "eval": evaluate_model(lambda x, y: linear_net(extract_batch_stats(x, base_model, expert_mnist, expert_fashion).to(device)).detach())
    }

    # 2. MLP-Small (1x64)
    print("Training MLP-Small (1 hidden layer of 64)...")
    small_net, val_mse_small = train_hypernetwork(
        meta_x_data, meta_y_data, input_dim=278, hidden_layers=1, hidden_dim=64, device=device
    )
    arch_results["MLP-Small (1x64)"] = {
        "val_mse": float(val_mse_small),
        "eval": evaluate_model(lambda x, y: small_net(extract_batch_stats(x, base_model, expert_mnist, expert_fashion).to(device)).detach())
    }

    # 3. MLP-Medium (2x128, Default)
    arch_results["MLP-Medium (2x128)"] = {
        "val_mse": float(val_mse_default),
        "eval": main_results["Hyper-TTMM (Ours, Zero-Shot)"]
    }

    # 4. MLP-Large (2x256)
    print("Training MLP-Large (2 hidden layers of 256)...")
    large_net, val_mse_large = train_hypernetwork(
        meta_x_data, meta_y_data, input_dim=278, hidden_layers=2, hidden_dim=256, device=device
    )
    arch_results["MLP-Large (2x256)"] = {
        "val_mse": float(val_mse_large),
        "eval": evaluate_model(lambda x, y: large_net(extract_batch_stats(x, base_model, expert_mnist, expert_fashion).to(device)).detach())
    }

    # Print architecture ablation results
    for arch, res in arch_results.items():
        print(f"Arch: {arch:20s} | Val MSE: {res['val_mse']:.5f} | Eval Acc: {res['eval']['overall_accuracy']:.2f}% | Latency: {res['eval']['average_latency_ms']:.2f}ms")


    # PART 3: Ablation Study on Input Features
    print("\n=== Running Part 3: Ablation on Features ===")
    feat_results = {}
    
    # Feature Stats is the first 256 dimensions, Prob/Ent is the last 22 dimensions
    # 1. Feature-only
    print("Training with Feature-only statistics (256-dim)...")
    feat_x = meta_x_data[:, :256]
    feat_only_net, val_mse_feat = train_hypernetwork(
        feat_x, meta_y_data, input_dim=256, hidden_layers=2, hidden_dim=128, device=device
    )
    def feat_only_gen(x, y):
        stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)[:256]
        with torch.no_grad():
            lambdas = feat_only_net(stats.to(device))
        return lambdas
    feat_results["Feature-Only (256-dim)"] = {
        "val_mse": float(val_mse_feat),
        "eval": evaluate_model(feat_only_gen)
    }

    # 2. Prob/Ent-only
    print("Training with Probability/Entropy-only statistics (22-dim)...")
    prob_x = meta_x_data[:, 256:]
    prob_only_net, val_mse_prob = train_hypernetwork(
        prob_x, meta_y_data, input_dim=22, hidden_layers=2, hidden_dim=128, device=device
    )
    def prob_only_gen(x, y):
        stats = extract_batch_stats(x, base_model, expert_mnist, expert_fashion)[256:]
        with torch.no_grad():
            lambdas = prob_only_net(stats.to(device))
        return lambdas
    feat_results["Prob-Ent-Only (22-dim)"] = {
        "val_mse": float(val_mse_prob),
        "eval": evaluate_model(prob_only_gen)
    }

    # 3. Full (Default)
    feat_results["Full Statistics (278-dim)"] = {
        "val_mse": float(val_mse_default),
        "eval": main_results["Hyper-TTMM (Ours, Zero-Shot)"]
    }

    # Print feature ablation results
    for feat, res in feat_results.items():
        print(f"Features: {feat:25s} | Val MSE: {res['val_mse']:.5f} | Eval Acc: {res['eval']['overall_accuracy']:.2f}%")


    # PART 4: Ablation Study on Meta-dataset Size
    print("\n=== Running Part 4: Ablation on Dataset Size ===")
    size_results = {}
    
    # 1. Size 250
    print("Training with Meta-dataset size = 250...")
    net_250, val_mse_250 = train_hypernetwork(
        meta_x_data[:250], meta_y_data[:250], input_dim=278, hidden_layers=2, hidden_dim=128, device=device
    )
    size_results["Size 250"] = {
        "val_mse": float(val_mse_250),
        "eval": evaluate_model(lambda x, y: net_250(extract_batch_stats(x, base_model, expert_mnist, expert_fashion).to(device)).detach())
    }

    # 2. Size 500
    print("Training with Meta-dataset size = 500...")
    net_500, val_mse_500 = train_hypernetwork(
        meta_x_data[:500], meta_y_data[:500], input_dim=278, hidden_layers=2, hidden_dim=128, device=device
    )
    size_results["Size 500"] = {
        "val_mse": float(val_mse_500),
        "eval": evaluate_model(lambda x, y: net_500(extract_batch_stats(x, base_model, expert_mnist, expert_fashion).to(device)).detach())
    }

    # 3. Size 1000 (Default)
    size_results["Size 1000 (Default)"] = {
        "val_mse": float(val_mse_default),
        "eval": main_results["Hyper-TTMM (Ours, Zero-Shot)"]
    }

    # Print dataset size ablation results
    for size, res in size_results.items():
        print(f"Meta-Dataset Size: {size:20s} | Val MSE: {res['val_mse']:.5f} | Eval Acc: {res['eval']['overall_accuracy']:.2f}%")


    # Save all results to results.json
    final_output = {
        "main_results": main_results,
        "ablation_architecture": arch_results,
        "ablation_features": feat_results,
        "ablation_datasize": size_results
    }
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print("\nAll results and ablations successfully saved to results.json.")

if __name__ == "__main__":
    main()
