import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

# Neural Network Architecture matching CL W-Fisher / KT-Fisher
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        x = self.dropout2(features)
        logits = self.fc2(x)
        if return_features:
            return logits, features
        return logits

def train_expert(name, dataset_cls, save_path, device, epochs=5):
    print(f"--- Training Expert: {name} ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = dataset_cls(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {acc:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert to {save_path}\n")
    return model

def compute_prototypes(model, dataset, device, num_samples=1000):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    features_list = []
    targets_list = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data = data.to(device)
            feats = model.extract_features(data)
            features_list.append(feats.cpu())
            targets_list.append(target)
            if (i + 1) * 128 >= num_samples:
                break
                
    features = torch.cat(features_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    prototypes = {}
    for c in range(10):
        mask = (targets == c)
        if mask.any():
            prototypes[c] = features[mask].mean(dim=0).to(device)
        else:
            prototypes[c] = torch.zeros(128).to(device)
            
    # Return as tensor of shape (10, 128)
    proto_tensor = torch.stack([prototypes[c] for c in range(10)])
    return proto_tensor

# Differentiable Soft BN and Weight model merging helper
def get_merged_state(model1, model2, layer_lambdas, avg_lambda):
    state = {}
    
    # 1. Merge parameters (weights and biases)
    for name, param in model1.named_parameters():
        p1 = param
        p2 = dict(model2.named_parameters())[name]
        
        # Find if matched
        matched_layer = None
        for layer_name in layer_lambdas.keys():
            if name.startswith(layer_name):
                matched_layer = layer_name
                break
                
        if matched_layer is not None:
            lam = layer_lambdas[matched_layer]
            state[name] = lam * p1 + (1 - lam) * p2
        else:
            state[name] = 0.5 * p1 + 0.5 * p2
            
    # 2. Merge buffers (Batch Normalization running mean/var)
    avg_lambda_detached = avg_lambda.detach() if torch.is_tensor(avg_lambda) else avg_lambda
    for name, buf in model1.named_buffers():
        b1 = buf
        b2 = dict(model2.named_buffers())[name]
        
        if "running_mean" in name:
            state[name] = avg_lambda_detached * b1 + (1 - avg_lambda_detached) * b2
        elif "running_var" in name:
            mean_name = name.replace("running_var", "running_mean")
            mean1 = dict(model1.named_buffers())[mean_name]
            mean2 = dict(model2.named_buffers())[mean_name]
            mean_fused = avg_lambda_detached * mean1 + (1 - avg_lambda_detached) * mean2
            
            var_fused = avg_lambda_detached * (b1 + mean1**2) + (1 - avg_lambda_detached) * (b2 + mean2**2) - mean_fused**2
            state[name] = var_fused
        else:
            state[name] = b1
            
    return state

def main():
    parser = argparse.ArgumentParser(description="Test-Time Model Merging Cosine-CPAL-TTMM Experiment")
    parser.add_argument('--sigma', type=float, default=1.0, help="CPAL contrastive temperature")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate for TTMM adaptation")
    parser.add_argument('--steps', type=int, default=10, help="Number of adaptation steps per batch")
    parser.add_argument('--batch_size', type=int, default=128, help="Stream batch size")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--out_dir', type=str, default="results", help="Output directory")
    parser.add_argument('--noise_std', type=float, default=0.6, help="Standard deviation of additive Gaussian noise")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Experts
    expert_mnist_path = "expert_mnist.pt"
    expert_fashion_path = "expert_fashion.pt"
    
    model1 = SimpleCNN().to(device)
    model1.load_state_dict(torch.load(expert_mnist_path, map_location=device))
    model1.eval()
    
    model2 = SimpleCNN().to(device)
    model2.load_state_dict(torch.load(expert_fashion_path, map_location=device))
    model2.eval()
    
    # 2. Compute class prototypes
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    print("Computing prototypes...")
    proto1 = compute_prototypes(model1, mnist_train, device, num_samples=1000) # (10, 128)
    proto2 = compute_prototypes(model2, fashion_train, device, num_samples=1000) # (10, 128)
    print("Prototypes computed successfully.\n")
    
    # 3. Create test stream datasets
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    class AddGaussianNoise(object):
        def __init__(self, mean=0.0, std=1.0):
            self.std = std
            self.mean = mean
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
            
    transform_noisy = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0.0, args.noise_std)
    ])
    mnist_test_noisy = datasets.MNIST(root='./data', train=False, download=True, transform=transform_noisy)
    fashion_test_noisy = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_noisy)
    
    B = args.batch_size
    
    loader_mnist = DataLoader(mnist_test, batch_size=B, shuffle=False)
    loader_mnist_noisy = DataLoader(mnist_test_noisy, batch_size=B, shuffle=False)
    loader_fashion = DataLoader(fashion_test, batch_size=B, shuffle=False)
    loader_fashion_noisy = DataLoader(fashion_test_noisy, batch_size=B, shuffle=False)
    loader_kmnist = DataLoader(kmnist_test, batch_size=B, shuffle=False)
    
    def get_batches(loader, num_batches=10):
        batches = []
        for i, (data, target) in enumerate(loader):
            batches.append((data, target))
            if len(batches) >= num_batches:
                break
        return batches
        
    stream_batches = []
    stream_batches.extend([("Clean MNIST", d, t) for d, t in get_batches(loader_mnist)])
    stream_batches.extend([("Noisy MNIST", d, t) for d, t in get_batches(loader_mnist_noisy)])
    stream_batches.extend([("Clean Fashion", d, t) for d, t in get_batches(loader_fashion)])
    stream_batches.extend([("Noisy Fashion", d, t) for d, t in get_batches(loader_fashion_noisy)])
    stream_batches.extend([("Novel KMNIST", d, t) for d, t in get_batches(loader_kmnist)])
    
    print(f"Constructed stream with {len(stream_batches)} batches of size {B}.")
    
    mergeable_layers = ["conv1", "conv2", "fc1", "fc2"]
    
    def run_ttmm_eval(mode):
        print(f"\n--- Running TTMM under mode: {mode.upper()} ---")
        
        batch_accs = []
        lambda_trajectories = []
        
        alphas = {layer: torch.tensor(0.0, device=device, requires_grad=True) for layer in mergeable_layers}
        
        for batch_idx, (segment_name, data, target) in enumerate(stream_batches):
            data, target = data.to(device), target.to(device)
            
            # --- EUCLIDEAN ROUTING PRIOR COMPUTATION ---
            model1.eval()
            with torch.no_grad():
                feats1 = model1.extract_features(data) # (B, 128)
                feats1_sq = torch.sum(feats1**2, dim=-1, keepdim=True) # (B, 1)
                proto1_sq = torch.sum(proto1**2, dim=-1, keepdim=True).t() # (1, 10)
                dists1_eucl = feats1_sq - 2.0 * torch.matmul(feats1, proto1.t()) + proto1_sq # (B, 10)
                dists1_eucl = torch.clamp(dists1_eucl, min=0.0)
                avg_d1_eucl = dists1_eucl.min(dim=-1)[0].mean().item()
                
            model2.eval()
            with torch.no_grad():
                feats2 = model2.extract_features(data) # (B, 128)
                feats2_sq = torch.sum(feats2**2, dim=-1, keepdim=True) # (B, 1)
                proto2_sq = torch.sum(proto2**2, dim=-1, keepdim=True).t() # (1, 10)
                dists2_eucl = feats2_sq - 2.0 * torch.matmul(feats2, proto2.t()) + proto2_sq # (B, 10)
                dists2_eucl = torch.clamp(dists2_eucl, min=0.0)
                avg_d2_eucl = dists2_eucl.min(dim=-1)[0].mean().item()
                
            # SCTS Euclidean temperature and routing probability
            s_param = 3.0
            delta_eucl = abs(avg_d2_eucl - avg_d1_eucl)
            tau_eucl = delta_eucl / (s_param + 1e-5)
            if tau_eucl < 1e-6:
                tau_eucl = 1e-6
            p_eucl = np.exp(-avg_d1_eucl / tau_eucl) / (np.exp(-avg_d1_eucl / tau_eucl) + np.exp(-avg_d2_eucl / tau_eucl) + 1e-12)
            p_eucl = max(1e-4, min(1.0 - 1e-4, p_eucl))
            alpha_init_eucl = np.log(p_eucl / (1.0 - p_eucl))
            
            # --- COSINE-NORMALIZED ROUTING PRIOR COMPUTATION ---
            feats1_norm = F.normalize(feats1, p=2, dim=-1)
            proto1_norm = F.normalize(proto1, p=2, dim=-1)
            dists1_cos = 1.0 - torch.matmul(feats1_norm, proto1_norm.t()) # (B, 10)
            avg_d1_cos = dists1_cos.min(dim=-1)[0].mean().item()
            
            feats2_norm = F.normalize(feats2, p=2, dim=-1)
            proto2_norm = F.normalize(proto2, p=2, dim=-1)
            dists2_cos = 1.0 - torch.matmul(feats2_norm, proto2_norm.t()) # (B, 10)
            avg_d2_cos = dists2_cos.min(dim=-1)[0].mean().item()
            
            # SCTS Cosine temperature and routing probability
            delta_cos = abs(avg_d2_cos - avg_d1_cos)
            tau_cos = delta_cos / (s_param + 1e-5)
            if tau_cos < 1e-6:
                tau_cos = 1e-6
            p_cos = np.exp(-avg_d1_cos / tau_cos) / (np.exp(-avg_d1_cos / tau_cos) + np.exp(-avg_d2_cos / tau_cos) + 1e-12)
            p_cos = max(1e-4, min(1.0 - 1e-4, p_cos))
            alpha_init_cos = np.log(p_cos / (1.0 - p_cos))
            
            # 2. Parameter Initialization:
            for layer in mergeable_layers:
                with torch.no_grad():
                    if mode == "none":
                        alphas[layer].fill_(0.0)
                    elif mode in ["scts_euclidean", "entropy_euclidean", "cpal_euclidean"]:
                        alphas[layer].fill_(alpha_init_eucl)
                    elif mode in ["scts_cosine", "entropy_cosine", "cpal_cosine"]:
                        alphas[layer].fill_(alpha_init_cos)
            
            # 3. Optimization using TTA/TTMM adaptation
            if mode in ["entropy_euclidean", "entropy_cosine", "cpal_euclidean", "cpal_cosine"]:
                optimizer = optim.Adam(list(alphas.values()), lr=args.lr)
                
                for step in range(args.steps):
                    layer_lambdas = {layer: torch.sigmoid(alphas[layer]) for layer in mergeable_layers}
                    avg_lambda = torch.stack(list(layer_lambdas.values())).mean()
                    
                    merged_state = get_merged_state(model1, model2, layer_lambdas, avg_lambda)
                    logits, features = functional_call(model1, merged_state, args=(data,), kwargs={"return_features": True})
                    
                    if mode in ["entropy_euclidean", "entropy_cosine"]:
                        probs = F.softmax(logits, dim=-1)
                        loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
                    elif mode == "cpal_euclidean":
                        # Standard Euclidean CPAL Loss (computes raw Euclidean distances to prototypes)
                        protos_all = torch.cat([proto1, proto2], dim=0) # (20, 128)
                        feats_sq = torch.sum(features**2, dim=-1, keepdim=True) # (B, 1)
                        protos_sq = torch.sum(protos_all**2, dim=-1, keepdim=True).t() # (1, 20)
                        eucl_dists = feats_sq - 2.0 * torch.matmul(features, protos_all.t()) + protos_sq # (B, 20)
                        eucl_dists = torch.clamp(eucl_dists, min=0.0)
                        
                        min_dists, _ = eucl_dists.min(dim=-1) # (B,)
                        sigma = eucl_dists.mean().detach() + 1e-6
                        loss = torch.mean(min_dists / sigma + torch.logsumexp(-eucl_dists / sigma, dim=-1))
                    elif mode == "cpal_cosine":
                        # Cosine-Normalized CPAL (C-CPAL) Loss (using angular metrics on unit hypersphere)
                        features_norm = F.normalize(features, p=2, dim=-1)
                        protos_all = torch.cat([proto1, proto2], dim=0) # (20, 128)
                        protos_all_norm = F.normalize(protos_all, p=2, dim=-1)
                        
                        cos_sims = torch.matmul(features_norm, protos_all_norm.t()) # (B, 20)
                        cos_dists = 1.0 - cos_sims # (B, 20)
                        
                        min_dists, _ = cos_dists.min(dim=-1) # (B,)
                        sigma = cos_dists.mean().detach() + 1e-6
                        loss = torch.mean(min_dists / sigma + torch.logsumexp(-cos_dists / sigma, dim=-1))
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Evaluate Accuracy with adapted merging coefficients
            with torch.no_grad():
                layer_lambdas = {layer: torch.sigmoid(alphas[layer]) for layer in mergeable_layers}
                avg_lambda = torch.stack(list(layer_lambdas.values())).mean()
                merged_state = get_merged_state(model1, model2, layer_lambdas, avg_lambda)
                
                logits = functional_call(model1, merged_state, data)
                _, predicted = logits.max(1)
                correct = predicted.eq(target).sum().item()
                acc = 100.0 * correct / data.size(0)
                
            batch_accs.append(acc)
            lambda_trajectories.append(avg_lambda.item())
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx:02d} ({segment_name:15s}) | Accuracy: {acc:6.2f}% | Lambda: {avg_lambda.item():.4f}")
            
        return batch_accs, lambda_trajectories

    results = {}
    modes = ["none", "scts_euclidean", "scts_cosine", "entropy_euclidean", "entropy_cosine", "cpal_euclidean", "cpal_cosine"]
    for m in modes:
        accs, lambdas = run_ttmm_eval(m)
        results[m] = {
            "accs": accs,
            "lambdas": lambdas
        }
        
    segments = ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]
    segment_results = {}
    for m in modes:
        segment_results[m] = {}
        for seg_idx, seg_name in enumerate(segments):
            start = seg_idx * 10
            end = start + 10
            mean_acc = np.mean(results[m]["accs"][start:end])
            mean_lam = np.mean(results[m]["lambdas"][start:end])
            segment_results[m][seg_name] = {
                "acc": mean_acc,
                "lambda": mean_lam
            }
            
    print("\n=================================== ROBUST SUMMARY RESULTS ===================================")
    header = f"{'Segment':15s} | {'Static':8s} | {'SCTS-E':8s} | {'SCTS-C':8s} | {'Entr-E':8s} | {'Entr-C':8s} | {'CPAL-E':8s} | {'CPAL-C (Ours)':12s}"
    print(header)
    print("-" * len(header))
    for seg_name in segments:
        acc_none = segment_results["none"][seg_name]["acc"]
        acc_scts_e = segment_results["scts_euclidean"][seg_name]["acc"]
        acc_scts_c = segment_results["scts_cosine"][seg_name]["acc"]
        acc_ent_e = segment_results["entropy_euclidean"][seg_name]["acc"]
        acc_ent_c = segment_results["entropy_cosine"][seg_name]["acc"]
        acc_cpal_e = segment_results["cpal_euclidean"][seg_name]["acc"]
        acc_cpal_c = segment_results["cpal_cosine"][seg_name]["acc"]
        print(f"{seg_name:15s} | {acc_none:5.2f}% | {acc_scts_e:5.2f}% | {acc_scts_c:5.2f}% | {acc_ent_e:5.2f}% | {acc_ent_c:5.2f}% | {acc_cpal_e:5.2f}% | {acc_cpal_c:5.2f}%")
    print("==============================================================================================")
    
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(segment_results, f, indent=4)
        
    # Generate Plots
    plt.figure(figsize=(14, 7))
    for m in modes:
        plt.plot(results[m]["accs"], label=f"{m.upper()}", linewidth=1.5 if "cosine" in m or "Ours" in m else 1.0)
    for i in range(1, 5):
        plt.axvline(x=i*10, color='gray', linestyle='--', alpha=0.5)
    for i, seg in enumerate(segments):
        plt.text(i*10 + 2, 95, seg, rotation=0, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    plt.title("Robust Test-Time Model Merging Accuracy Comparison")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "accuracy_trajectory.png"))
    plt.close()
    
    plt.figure(figsize=(14, 7))
    for m in modes:
        plt.plot(results[m]["lambdas"], label=f"{m.upper()}", linewidth=1.5 if "cosine" in m or "Ours" in m else 1.0)
    for i in range(1, 5):
        plt.axvline(x=i*10, color='gray', linestyle='--', alpha=0.5)
    for i, seg in enumerate(segments):
        plt.text(i*10 + 2, 0.9, seg, rotation=0, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    plt.title("Robust Test-Time Model Merging Lambda Trajectories Comparison")
    plt.xlabel("Batch Index")
    plt.ylabel("Average Merging Coefficient (Lambda)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "lambda_trajectory.png"))
    plt.close()
    
    print("\nSaved robust plots and results successfully.\n")

if __name__ == "__main__":
    main()
