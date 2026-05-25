import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.func import functional_call

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define SimpleCNN Architecture with configurable margin m
class SimpleCNN(nn.Module):
    def __init__(self, is_cosine=False, margin=0.35):
        super().__init__()
        self.is_cosine = is_cosine
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.50)
        
        if is_cosine:
            self.fc2_weight = nn.Parameter(torch.FloatTensor(10, 128))
            nn.init.xavier_uniform_(self.fc2_weight)
            self.s = 30.0  # Scale factor
            self.m = margin  # Cosine margin
        else:
            self.fc2 = nn.Linear(128, 10)
            
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x

    def forward(self, x, label=None):
        features = self.get_features(x)
        x = self.dropout2(features)
        if self.is_cosine:
            cosine = F.linear(F.normalize(x), F.normalize(self.fc2_weight))
            if self.training and label is not None:
                one_hot = torch.zeros(cosine.size(), device=x.device)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
                output = self.s * (cosine - one_hot * self.m)
            else:
                output = self.s * cosine
            return output
        else:
            return self.fc2(x)

# Load Datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading datasets...")
mnist_train_full = torchvision.datasets.MNIST('.', train=True, download=True, transform=transform)
mnist_test_full = torchvision.datasets.MNIST('.', train=False, download=True, transform=transform)
fmnist_train_full = torchvision.datasets.FashionMNIST('.', train=True, download=True, transform=transform)
fmnist_test_full = torchvision.datasets.FashionMNIST('.', train=False, download=True, transform=transform)
kmnist_test_full = torchvision.datasets.KMNIST('.', train=False, download=True, transform=transform)

# Subset 10,000 training samples
mnist_train_subset = torch.utils.data.Subset(mnist_train_full, list(range(10000)))
fmnist_train_subset = torch.utils.data.Subset(fmnist_train_full, list(range(10000)))

# Training function
def train_expert(model, train_dataset, test_dataset, epochs=2, batch_size=64, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, y)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_acc = 100.0 * correct / total
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, predicted = out.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
        test_acc = 100.0 * test_correct / test_total
    print(f"Finished training: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    return model

# Precompute prototypes
def precompute_prototypes(model, dataset, is_cosine=False, num_samples=256):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    class_features = {i: [] for i in range(10)}
    count = 0
    with torch.no_grad():
        for x, y in loader:
            if count >= num_samples:
                break
            x = x.to(device)
            feat = model.get_features(x).squeeze(0)
            label = y.item()
            if is_cosine:
                feat = F.normalize(feat, p=2, dim=0)
            class_features[label].append(feat)
            count += 1
            
    prototypes = torch.zeros(10, 128, device=device)
    for c in range(10):
        if len(class_features[c]) > 0:
            stacked = torch.stack(class_features[c])
            mean_feat = torch.mean(stacked, dim=0)
            if is_cosine:
                mean_feat = F.normalize(mean_feat, p=2, dim=0)
            prototypes[c] = mean_feat
    return prototypes

# Compute Fisher diagonal
def compute_fisher_diagonal(model, dataset, num_samples=256):
    model.eval()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    count = 0
    for x, y in loader:
        if count >= num_samples:
            break
        model.zero_grad()
        out = model(x.to(device))
        loss = F.cross_entropy(out, y.to(device))
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2
        count += 1
    for name in fisher:
        fisher[name] /= count
    return fisher

def get_normalized_joint_fisher(fisher0, fisher1):
    joint_fisher = {}
    layer_sensitivities = {}
    for name in fisher0:
        joint_fisher[name] = fisher0[name] + fisher1[name]
        layer_sensitivities[name] = torch.mean(joint_fisher[name]).item()
    max_sens = max(layer_sensitivities.values())
    normalized_sensitivities = {name: sens / max_sens for name, sens in layer_sensitivities.items()}
    return normalized_sensitivities

# Stream generation
def get_stream_loader(dataset, start_idx, num_batches, batch_size=64, noise_std=0.0):
    batches = []
    for i in range(num_batches):
        indices = list(range(start_idx + i*batch_size, start_idx + (i+1)*batch_size))
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
        x, y = next(iter(loader))
        if noise_std > 0.0:
            x_noisy = x + noise_std * torch.randn_like(x)
        else:
            x_noisy = x.clone()
        batches.append((x_noisy, y))
    return batches

print("\nConstructing non-stationary stream batches...")
stream_batches = []
stream_batches.extend(get_stream_loader(mnist_test_full, 0, 10, batch_size=64, noise_std=0.0))
stream_batches.extend(get_stream_loader(mnist_test_full, 10*64, 10, batch_size=64, noise_std=0.6))
stream_batches.extend(get_stream_loader(fmnist_test_full, 0, 10, batch_size=64, noise_std=0.0))
stream_batches.extend(get_stream_loader(fmnist_test_full, 10*64, 10, batch_size=64, noise_std=0.6))
stream_batches.extend(get_stream_loader(kmnist_test_full, 0, 10, batch_size=64, noise_std=0.0))


# Magnitude-Bounded Angular Routing Prior
def compute_routing_prior_angular_mb(x, mnist_expert, fmnist_expert, mnist_prototypes_cos, fmnist_prototypes_cos, s=3.0, eps_stab=0.04, eps_mag=0.0):
    mnist_expert.eval()
    fmnist_expert.eval()
    with torch.no_grad():
        f0 = mnist_expert.get_features(x)   # B x 128
        f1 = fmnist_expert.get_features(x)  # B x 128
        
    f0_norm = F.normalize(f0, p=2, dim=1)
    f1_norm = F.normalize(f1, p=2, dim=1)
    
    # Cosine Similarity
    cos0 = torch.matmul(f0_norm, mnist_prototypes_cos.t())
    cos1 = torch.matmul(f1_norm, fmnist_prototypes_cos.t())
    
    # If magnitude bounding is active, scale the cosine similarity
    if eps_mag > 0.0:
        # Calculate batch average norm of the feature vectors
        norm0 = torch.norm(f0, p=2, dim=1, keepdim=True)  # B x 1
        norm1 = torch.norm(f1, p=2, dim=1, keepdim=True)  # B x 1
        
        # Soft-normalize by dividing by (norm + eps_mag) instead of norm
        # This is equivalent to multiplying the pure cosine similarity by (norm / (norm + eps_mag))
        cos0 = cos0 * (norm0 / (norm0 + eps_mag))
        cos1 = cos1 * (norm1 / (norm1 + eps_mag))
        
    # Distance = 1.0 - CosSimilarity
    d0 = torch.min(1.0 - cos0, dim=1)[0]
    d1 = torch.min(1.0 - cos1, dim=1)[0]
    
    mean_d0 = d0.mean().item()
    mean_d1 = d1.mean().item()
    
    delta = abs(mean_d1 - mean_d0)
    tau = delta / s + eps_stab
    
    scores = torch.tensor([-mean_d0 / tau, -mean_d1 / tau], device=x.device)
    w = F.softmax(scores, dim=0)
    return w, mean_d0, mean_d1, delta, tau


# TTA PIPELINE RUNNER WITH MAGNITUDE BOUNDING
def run_tta_pipeline(mnist_expert, fmnist_expert, mnist_proto, fmnist_proto, normalized_fisher, is_cosine=False, s_factor=3.0, eps_val=0.04, eps_mag=0.0):
    base_model = SimpleCNN(is_cosine=is_cosine).to(device)
    expert0_state = {k: v.clone().to(device) for k, v in mnist_expert.state_dict().items()}
    expert1_state = {k: v.clone().to(device) for k, v in fmnist_expert.state_dict().items()}
    
    param_names = [k for k, v in mnist_expert.named_parameters()]
    buffer_names = [k for k, v in mnist_expert.named_buffers()]
    
    batch_accuracies = []
    
    N_step = 5
    eta = 0.05
    beta = 1.5
    gamma = 0.02
    
    for t, (x, y) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # Compute prior using MB-Angular Routing
        w, d0, d1, delta, tau = compute_routing_prior_angular_mb(
            x, mnist_expert, fmnist_expert, mnist_proto, fmnist_proto, 
            s=s_factor, eps_stab=eps_val, eps_mag=eps_mag
        )
            
        p = w[0].item()
        p_clamped = max(1e-4, min(1 - 1e-4, p))
        
        w_global = torch.tensor(np.log(p_clamped / (1.0 - p_clamped)), device=device, requires_grad=True)
        delta_dict = {name: torch.tensor(0.0, device=device, requires_grad=True) for name in param_names}
            
        for step in range(N_step):
            lambda_dict = {}
            for name in param_names:
                lambda_dict[name] = torch.sigmoid(w_global + delta_dict[name])
                
            avg_lambda = torch.mean(torch.stack(list(lambda_dict.values())))
            
            merged_state = {}
            for name in param_names:
                l_val = lambda_dict[name]
                merged_state[name] = l_val * expert0_state[name] + (1.0 - l_val) * expert1_state[name]
                
            for name in buffer_names:
                merged_state[name] = (avg_lambda.detach() * expert0_state[name] + (1.0 - avg_lambda.detach()) * expert1_state[name]).detach()
                
            out = functional_call(base_model, merged_state, (x,))
            probs = F.softmax(out, dim=1)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            
            kl_loss = 0.0
            for name in param_names:
                l_val = lambda_dict[name]
                kl_val = p_clamped * torch.log(p_clamped / (l_val + 1e-8)) + (1.0 - p_clamped) * torch.log((1.0 - p_clamped) / (1.0 - l_val + 1e-8))
                kl_loss += kl_val
            kl_loss = kl_loss / len(param_names)
            
            coherence_loss = 0.0
            for name in param_names:
                coherence_loss += delta_dict[name]**2
                
            loss = entropy_loss + beta * kl_loss + gamma * coherence_loss
                
            if w_global.grad is not None: w_global.grad.zero_()
            for name in param_names:
                if delta_dict[name].grad is not None:
                    delta_dict[name].grad.zero_()
                    
            loss.backward()
            
            with torch.no_grad():
                w_global -= eta * w_global.grad
                for name in param_names:
                    grad_val = delta_dict[name].grad
                    f_sens = normalized_fisher[name]
                    delta_dict[name] -= eta * (1.0 / (f_sens + 0.01)) * grad_val
                        
        with torch.no_grad():
            lambda_dict = {}
            for name in param_names:
                lambda_dict[name] = torch.sigmoid(w_global + delta_dict[name])
            avg_lambda = torch.mean(torch.stack(list(lambda_dict.values()))).item()
            
            merged_state = {}
            for name in param_names:
                l_val = lambda_dict[name]
                merged_state[name] = l_val * expert0_state[name] + (1.0 - l_val) * expert1_state[name]
            for name in buffer_names:
                merged_state[name] = avg_lambda * expert0_state[name] + (1.0 - avg_lambda) * expert1_state[name]
                
            out_eval = functional_call(base_model, merged_state, (x,))
            _, pred = out_eval.max(1)
            correct = pred.eq(y).sum().item()
            acc = 100.0 * correct / x.size(0)
            batch_accuracies.append(acc)
            
    seg_accuracies = {
        "Clean MNIST": np.mean(batch_accuracies[0:10]),
        "Noisy MNIST": np.mean(batch_accuracies[10:20]),
        "Clean Fashion": np.mean(batch_accuracies[20:30]),
        "Noisy Fashion": np.mean(batch_accuracies[30:40]),
        "Novel KMNIST": np.mean(batch_accuracies[40:50]),
    }
    return seg_accuracies


# MAIN EXECUTION FOR THE ENTIRE SWEEP PROGRAM
sweep_results = {}

# 1. Sweep over Cosine Margin m (and train models for each margin)
margins = [0.0, 0.15, 0.25, 0.35, 0.45]
sweep_results["margin_sweep"] = {}

print("\n================== SWEEP 1: COSINE MARGIN m ==================")
for m in margins:
    print(f"\n--- Running Margin m = {m:.2f} ---")
    print("Training MNIST Expert with CosFace...")
    model_mnist = SimpleCNN(is_cosine=True, margin=m)
    train_expert(model_mnist, mnist_train_subset, mnist_test_full)
    
    print("Training FashionMNIST Expert with CosFace...")
    model_fmnist = SimpleCNN(is_cosine=True, margin=m)
    train_expert(model_fmnist, fmnist_train_subset, fmnist_test_full)
    
    proto_mnist = precompute_prototypes(model_mnist, mnist_train_subset, is_cosine=True)
    proto_fmnist = precompute_prototypes(model_fmnist, fmnist_train_subset, is_cosine=True)
    
    fisher_mnist = compute_fisher_diagonal(model_mnist, mnist_train_subset)
    fisher_fmnist = compute_fisher_diagonal(model_fmnist, fmnist_train_subset)
    norm_fisher = get_normalized_joint_fisher(fisher_mnist, fisher_fmnist)
    
    res = run_tta_pipeline(
        model_mnist, model_fmnist, proto_mnist, proto_fmnist, norm_fisher,
        is_cosine=True, s_factor=3.0, eps_val=0.04, eps_mag=0.0
    )
    sweep_results["margin_sweep"][m] = res
    print(f"Margin {m} Result: {res}")


# 2. Sweep over Stability Offset eps_val (on the model trained with m = 0.35)
print("\n================== SWEEP 2: STABILITY OFFSET eps_val ==================")
eps_vals = [0.01, 0.02, 0.04, 0.08, 0.16]
sweep_results["eps_sweep"] = {}

# Use the m=0.35 model from Sweep 1 to save training time
model_mnist_35 = SimpleCNN(is_cosine=True, margin=0.35)
# Load parameters from the m=0.35 trained model in Sweep 1
# We can just train a standard 0.35 model if easier, or reuse Sweep 1's models.
# Since we just trained it, we will just retrain one set of m=0.35 models once and reuse.
model_mnist_35 = SimpleCNN(is_cosine=True, margin=0.35)
train_expert(model_mnist_35, mnist_train_subset, mnist_test_full)
model_fmnist_35 = SimpleCNN(is_cosine=True, margin=0.35)
train_expert(model_fmnist_35, fmnist_train_subset, fmnist_test_full)

proto_mnist_35 = precompute_prototypes(model_mnist_35, mnist_train_subset, is_cosine=True)
proto_fmnist_35 = precompute_prototypes(model_fmnist_35, fmnist_train_subset, is_cosine=True)

fisher_mnist_35 = compute_fisher_diagonal(model_mnist_35, mnist_train_subset)
fisher_fmnist_35 = compute_fisher_diagonal(model_fmnist_35, fmnist_train_subset)
norm_fisher_35 = get_normalized_joint_fisher(fisher_mnist_35, fisher_fmnist_35)

for eps in eps_vals:
    print(f"\n--- Running eps_val = {eps:.2f} ---")
    res = run_tta_pipeline(
        model_mnist_35, model_fmnist_35, proto_mnist_35, proto_fmnist_35, norm_fisher_35,
        is_cosine=True, s_factor=3.0, eps_val=eps, eps_mag=0.0
    )
    sweep_results["eps_sweep"][eps] = res
    print(f"eps_val {eps} Result: {res}")


# 3. Sweep over Magnitude Bounding Parameter eps_mag
print("\n================== SWEEP 3: MAGNITUDE BOUNDING eps_mag ==================")
eps_mags = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0]
sweep_results["mag_sweep"] = {}

for em in eps_mags:
    print(f"\n--- Running eps_mag = {em:.1f} ---")
    res = run_tta_pipeline(
        model_mnist_35, model_fmnist_35, proto_mnist_35, proto_fmnist_35, norm_fisher_35,
        is_cosine=True, s_factor=3.0, eps_val=0.04, eps_mag=em
    )
    sweep_results["mag_sweep"][em] = res
    print(f"eps_mag {em} Result: {res}")


# 4. Sweep over Confidence Scaling Factor s
print("\n================== SWEEP 4: CONFIDENCE SCALE FACTOR s ==================")
s_factors = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
sweep_results["s_sweep"] = {}

for sf in s_factors:
    print(f"\n--- Running s_factor = {sf:.1f} ---")
    res = run_tta_pipeline(
        model_mnist_35, model_fmnist_35, proto_mnist_35, proto_fmnist_35, norm_fisher_35,
        is_cosine=True, s_factor=sf, eps_val=0.04, eps_mag=0.0
    )
    sweep_results["s_sweep"][sf] = res
    print(f"s_factor {sf} Result: {res}")


# WRITE ALL SWEEP RESULTS TO FILE
with open("sweep_results.txt", "w") as f:
    f.write("COMPREHENSIVE SWEEP AND ABLATION RESULTS\n")
    f.write("========================================\n\n")
    
    f.write("1. COSINE MARGIN (m) ABLATION STUDY\n")
    f.write("------------------------------------\n")
    f.write(f"{'Margin m':<10} | {'Clean MNIST':<12} | {'Noisy MNIST':<12} | {'Clean Fashion':<14} | {'Noisy Fashion':<14} | {'Novel KMNIST':<12}\n")
    f.write("-" * 85 + "\n")
    for m in margins:
        r = sweep_results["margin_sweep"][m]
        f.write(f"{m:<10.2f} | {r['Clean MNIST']:<12.2f}% | {r['Noisy MNIST']:<12.2f}% | {r['Clean Fashion']:<14.2f}% | {r['Noisy Fashion']:<14.2f}% | {r['Novel KMNIST']:<12.2f}%\n")
    f.write("\n\n")
    
    f.write("2. TEMPERATURE STABILITY OFFSET (eps_val) SWEEP\n")
    f.write("-----------------------------------------------\n")
    f.write(f"{'eps_val':<10} | {'Clean MNIST':<12} | {'Noisy MNIST':<12} | {'Clean Fashion':<14} | {'Noisy Fashion':<14} | {'Novel KMNIST':<12}\n")
    f.write("-" * 85 + "\n")
    for eps in eps_vals:
        r = sweep_results["eps_sweep"][eps]
        f.write(f"{eps:<10.2f} | {r['Clean MNIST']:<12.2f}% | {r['Noisy MNIST']:<12.2f}% | {r['Clean Fashion']:<14.2f}% | {r['Noisy Fashion']:<14.2f}% | {r['Novel KMNIST']:<12.2f}%\n")
    f.write("\n\n")
    
    f.write("3. MAGNITUDE-BOUNDED ANGULAR ROUTING (eps_mag) STUDY\n")
    f.write("----------------------------------------------------\n")
    f.write(f"{'eps_mag':<10} | {'Clean MNIST':<12} | {'Noisy MNIST':<12} | {'Clean Fashion':<14} | {'Noisy Fashion':<14} | {'Novel KMNIST':<12}\n")
    f.write("-" * 85 + "\n")
    for em in eps_mags:
        r = sweep_results["mag_sweep"][em]
        f.write(f"{em:<10.1f} | {r['Clean MNIST']:<12.2f}% | {r['Noisy MNIST']:<12.2f}% | {r['Clean Fashion']:<14.2f}% | {r['Noisy Fashion']:<14.2f}% | {r['Novel KMNIST']:<12.2f}%\n")
    f.write("\n\n")

    f.write("4. TEMPERATURE SCALE FACTOR (s_factor) SWEEP\n")
    f.write("---------------------------------------------\n")
    f.write(f"{'s_factor':<10} | {'Clean MNIST':<12} | {'Noisy MNIST':<12} | {'Clean Fashion':<14} | {'Noisy Fashion':<14} | {'Novel KMNIST':<12}\n")
    f.write("-" * 85 + "\n")
    for sf in s_factors:
        r = sweep_results["s_sweep"][sf]
        f.write(f"{sf:<10.1f} | {r['Clean MNIST']:<12.2f}% | {r['Noisy MNIST']:<12.2f}% | {r['Clean Fashion']:<14.2f}% | {r['Noisy Fashion']:<14.2f}% | {r['Novel KMNIST']:<12.2f}%\n")

print("\nSweep results successfully written to sweep_results.txt!")
