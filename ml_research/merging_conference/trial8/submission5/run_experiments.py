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

# Define SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self, is_cosine=False):
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
            # CosFace weight: shape (10, 128)
            self.fc2_weight = nn.Parameter(torch.FloatTensor(10, 128))
            nn.init.xavier_uniform_(self.fc2_weight)
            self.s = 30.0  # Scale factor
            self.m = 0.35  # Cosine margin
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
            # Normalize both weights and features
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    return model

print("\n--- Training Standard MNIST Expert ---")
mnist_expert_std = SimpleCNN(is_cosine=False)
train_expert(mnist_expert_std, mnist_train_subset, mnist_test_full)

print("\n--- Training Standard FashionMNIST Expert ---")
fmnist_expert_std = SimpleCNN(is_cosine=False)
train_expert(fmnist_expert_std, fmnist_train_subset, fmnist_test_full)

print("\n--- Training Cosine-Margin MNIST Expert ---")
mnist_expert_cos = SimpleCNN(is_cosine=True)
train_expert(mnist_expert_cos, mnist_train_subset, mnist_test_full)

print("\n--- Training Cosine-Margin FashionMNIST Expert ---")
fmnist_expert_cos = SimpleCNN(is_cosine=True)
train_expert(fmnist_expert_cos, fmnist_train_subset, fmnist_test_full)


# PRECOMPUTE CLASS PROTOTYPES (using 256 clean calibration samples from train subset)
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
            feat = model.get_features(x).squeeze(0)  # 128
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
        else:
            print(f"Warning: Class {c} has zero samples in calibration set!")
            
    return prototypes

print("\nComputing prototypes...")
mnist_proto_std = precompute_prototypes(mnist_expert_std, mnist_train_subset, is_cosine=False)
fmnist_proto_std = precompute_prototypes(fmnist_expert_std, fmnist_train_subset, is_cosine=False)

mnist_proto_cos = precompute_prototypes(mnist_expert_cos, mnist_train_subset, is_cosine=True)
fmnist_proto_cos = precompute_prototypes(fmnist_expert_cos, fmnist_train_subset, is_cosine=True)


# PRECOMPUTE JOINT FISHER DIAGONSAL SENSITIVITIES
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

print("\nComputing Fisher sensitivities...")
mnist_fisher_std = compute_fisher_diagonal(mnist_expert_std, mnist_train_subset)
fmnist_fisher_std = compute_fisher_diagonal(fmnist_expert_std, fmnist_train_subset)

mnist_fisher_cos = compute_fisher_diagonal(mnist_expert_cos, mnist_train_subset)
fmnist_fisher_cos = compute_fisher_diagonal(fmnist_expert_cos, fmnist_train_subset)

def get_normalized_joint_fisher(fisher0, fisher1):
    joint_fisher = {}
    layer_sensitivities = {}
    for name in fisher0:
        joint_fisher[name] = fisher0[name] + fisher1[name]
        layer_sensitivities[name] = torch.mean(joint_fisher[name]).item()
        
    max_sens = max(layer_sensitivities.values())
    normalized_sensitivities = {name: sens / max_sens for name, sens in layer_sensitivities.items()}
    return normalized_sensitivities

joint_fisher_std = get_normalized_joint_fisher(mnist_fisher_std, fmnist_fisher_std)
joint_fisher_cos = get_normalized_joint_fisher(mnist_fisher_cos, fmnist_fisher_cos)


# SETUP THE NON-STATIONARY EVALUATION STREAM
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
# 50 batches total, 10 per phase
stream_batches = []
# Phase 1: Clean MNIST (batches 0-9)
stream_batches.extend(get_stream_loader(mnist_test_full, 0, 10, batch_size=64, noise_std=0.0))
# Phase 2: Noisy MNIST (batches 10-19)
stream_batches.extend(get_stream_loader(mnist_test_full, 10*64, 10, batch_size=64, noise_std=0.6))
# Phase 3: Clean FashionMNIST (batches 20-29)
stream_batches.extend(get_stream_loader(fmnist_test_full, 0, 10, batch_size=64, noise_std=0.0))
# Phase 4: Noisy FashionMNIST (batches 30-39)
stream_batches.extend(get_stream_loader(fmnist_test_full, 10*64, 10, batch_size=64, noise_std=0.6))
# Phase 5: Novel KMNIST (batches 40-49)
stream_batches.extend(get_stream_loader(kmnist_test_full, 0, 10, batch_size=64, noise_std=0.0))


# ROUTING PRIOR FUNCTIONS
def compute_routing_prior_l2(x, mnist_expert, fmnist_expert, mnist_prototypes, fmnist_prototypes, s=3.0, eps_stab=150.0):
    mnist_expert.eval()
    fmnist_expert.eval()
    with torch.no_grad():
        f0 = mnist_expert.get_features(x)   # B x 128
        f1 = fmnist_expert.get_features(x)  # B x 128
        
    # L2 Distance
    d0 = torch.min(torch.sum((f0.unsqueeze(1) - mnist_prototypes.unsqueeze(0))**2, dim=2), dim=1)[0]
    d1 = torch.min(torch.sum((f1.unsqueeze(1) - fmnist_prototypes.unsqueeze(0))**2, dim=2), dim=1)[0]
    
    mean_d0 = d0.mean().item()
    mean_d1 = d1.mean().item()
    
    # Gap
    delta = abs(mean_d1 - mean_d0)
    
    # SCTS Temperature
    tau = delta / s + eps_stab
    
    # Softmax Routing
    scores = torch.tensor([-mean_d0 / tau, -mean_d1 / tau], device=x.device)
    w = F.softmax(scores, dim=0)
    return w, mean_d0, mean_d1, delta, tau

def compute_routing_prior_angular(x, mnist_expert, fmnist_expert, mnist_prototypes_cos, fmnist_prototypes_cos, s=3.0, eps_stab=0.04):
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
    
    # Angular Distance = 1.0 - CosSimilarity
    d0 = torch.min(1.0 - cos0, dim=1)[0]
    d1 = torch.min(1.0 - cos1, dim=1)[0]
    
    mean_d0 = d0.mean().item()
    mean_d1 = d1.mean().item()
    
    # Gap
    delta = abs(mean_d1 - mean_d0)
    
    # SCTS Temperature (scaled for angular)
    tau = delta / s + eps_stab
    
    # Softmax Routing
    scores = torch.tensor([-mean_d0 / tau, -mean_d1 / tau], device=x.device)
    w = F.softmax(scores, dim=0)
    return w, mean_d0, mean_d1, delta, tau


# TEST-TIME ADAPTATION PIPELINE RUNNER
def run_tta_pipeline(method_name, mnist_expert, fmnist_expert, mnist_proto, fmnist_proto, normalized_fisher, is_cosine=False, is_angular=False, s_factor=3.0, eps_val=150.0):
    print(f"\nEvaluating Method: {method_name}...")
    
    # Base model structure for functional_call
    base_model = SimpleCNN(is_cosine=is_cosine).to(device)
    
    # Experts parameters and buffers
    expert0_state = {k: v.clone().to(device) for k, v in mnist_expert.state_dict().items()}
    expert1_state = {k: v.clone().to(device) for k, v in fmnist_expert.state_dict().items()}
    
    param_names = [k for k, v in mnist_expert.named_parameters()]
    buffer_names = [k for k, v in mnist_expert.named_buffers()]
    
    # Tracking logs
    batch_accuracies = []
    batch_lambdas = []
    
    # Parameters for adaptation
    N_step = 5
    eta = 0.05
    beta = 1.5
    gamma = 0.02
    
    for t, (x, y) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        
        # 1. Compute routing prior
        if is_angular:
            w, d0, d1, delta, tau = compute_routing_prior_angular(x, mnist_expert, fmnist_expert, mnist_proto, fmnist_proto, s=s_factor, eps_stab=eps_val)
        else:
            w, d0, d1, delta, tau = compute_routing_prior_l2(x, mnist_expert, fmnist_expert, mnist_proto, fmnist_proto, s=s_factor, eps_stab=eps_val)
            
        p = w[0].item()
        p_clamped = max(1e-4, min(1 - 1e-4, p))
        
        # 2. Prior-Guided Initialization (or Standard resetting if baseline)
        if "Fixed" in method_name or "Reset" in method_name:
            w_global = torch.tensor(0.0, device=device, requires_grad=True)
            delta_dict = {name: torch.tensor(0.0, device=device, requires_grad=True) for name in param_names}
        else:
            w_global = torch.tensor(np.log(p_clamped / (1.0 - p_clamped)), device=device, requires_grad=True)
            delta_dict = {name: torch.tensor(0.0, device=device, requires_grad=True) for name in param_names}
            
        # Optimization Loop
        for step in range(N_step):
            # Dynamic layer-wise merging coefficients
            lambda_dict = {}
            for name in param_names:
                lambda_dict[name] = torch.sigmoid(w_global + delta_dict[name])
                
            # Average lambda for BN buffer blending
            avg_lambda = torch.mean(torch.stack(list(lambda_dict.values())))
            
            # Interpolate parameters
            merged_state = {}
            for name in param_names:
                l_val = lambda_dict[name]
                merged_state[name] = l_val * expert0_state[name] + (1.0 - l_val) * expert1_state[name]
                
            # Blending Batch Normalization running statistics (buffers)
            for name in buffer_names:
                merged_state[name] = (avg_lambda.detach() * expert0_state[name] + (1.0 - avg_lambda.detach()) * expert1_state[name]).detach()
                
            # Run forward pass
            out = functional_call(base_model, merged_state, (x,))
            
            # Losses
            probs = F.softmax(out, dim=1)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            
            # KL divergence to routing prior
            kl_loss = 0.0
            for name in param_names:
                l_val = lambda_dict[name]
                kl_val = p_clamped * torch.log(p_clamped / (l_val + 1e-8)) + (1.0 - p_clamped) * torch.log((1.0 - p_clamped) / (1.0 - l_val + 1e-8))
                kl_loss += kl_val
            kl_loss = kl_loss / len(param_names)
            
            # Consensus Coherence Penalty
            coherence_loss = 0.0
            for name in param_names:
                coherence_loss += delta_dict[name]**2
                
            # Total Loss
            if "Fixed" in method_name:
                # Standard TTA (entropy minimization only)
                loss = entropy_loss
            else:
                loss = entropy_loss + beta * kl_loss + gamma * coherence_loss
                
            # Backpropagation
            # Zero grads manually
            if w_global.grad is not None: w_global.grad.zero_()
            for name in param_names:
                if delta_dict[name].grad is not None:
                    delta_dict[name].grad.zero_()
                    
            loss.backward()
            
            # Parameter updates
            with torch.no_grad():
                w_global -= eta * w_global.grad
                for name in param_names:
                    grad_val = delta_dict[name].grad
                    if "Fixed" in method_name:
                        # Unpreconditioned
                        delta_dict[name] -= eta * grad_val
                    else:
                        # Fisher-Preconditioned
                        f_sens = normalized_fisher[name]
                        delta_dict[name] -= eta * (1.0 / (f_sens + 0.01)) * grad_val
                        
        # Inference with adapted coefficients
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
            batch_lambdas.append(avg_lambda)
            
    # Compute segment-wise averages
    # Stream segments:
    # Seg 1: Clean MNIST (batches 0-9)
    # Seg 2: Noisy MNIST (batches 10-19)
    # Seg 3: Clean FashionMNIST (batches 20-29)
    # Seg 4: Noisy FashionMNIST (batches 30-39)
    # Seg 5: Novel KMNIST (batches 40-49)
    seg_accuracies = {
        "Clean MNIST": np.mean(batch_accuracies[0:10]),
        "Noisy MNIST": np.mean(batch_accuracies[10:20]),
        "Clean Fashion": np.mean(batch_accuracies[20:30]),
        "Noisy Fashion": np.mean(batch_accuracies[30:40]),
        "Novel KMNIST": np.mean(batch_accuracies[40:50]),
    }
    
    print(f"Results for {method_name}:")
    for k, v in seg_accuracies.items():
        print(f"  {k}: {v:.2f}%")
        
    return batch_accuracies, batch_lambdas, seg_accuracies


# RUN PIPELINES FOR ALL FOUR METHODS

# Method A: Fixed TTA + Reset (baseline)
acc_a, lam_a, seg_a = run_tta_pipeline(
    "Fixed TTA + Reset",
    mnist_expert_std, fmnist_expert_std,
    mnist_proto_std, fmnist_proto_std,
    joint_fisher_std,
    is_cosine=False, is_angular=False,
    s_factor=3.0, eps_val=150.0
)

# Method B: CL W-Fisher + SCTS (L2)
acc_b, lam_b, seg_b = run_tta_pipeline(
    "CL W-Fisher + SCTS (L2)",
    mnist_expert_std, fmnist_expert_std,
    mnist_proto_std, fmnist_proto_std,
    joint_fisher_std,
    is_cosine=False, is_angular=False,
    s_factor=3.0, eps_val=150.0
)

# Method C: CL W-Fisher + A-SCTS (Angular, standard model)
acc_c, lam_c, seg_c = run_tta_pipeline(
    "CL W-Fisher + A-SCTS (Angular)",
    mnist_expert_std, fmnist_expert_std,
    # Standard prototypes can be normalized for cosine comparison
    F.normalize(mnist_proto_std, p=2, dim=1), F.normalize(fmnist_proto_std, p=2, dim=1),
    joint_fisher_std,
    is_cosine=False, is_angular=True,
    s_factor=3.0, eps_val=0.04
)

# Method D: CL W-Fisher + A-SCTS + CP-AM (Ours, CosFace models)
acc_d, lam_d, seg_d = run_tta_pipeline(
    "CL W-Fisher + A-SCTS + CP-AM (Ours)",
    mnist_expert_cos, fmnist_expert_cos,
    mnist_proto_cos, fmnist_proto_cos,
    joint_fisher_cos,
    is_cosine=True, is_angular=True,
    s_factor=3.0, eps_val=0.04
)


# SAVE PLOTS
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc_a, label="Fixed TTA + Reset (Baseline)", color='gray', linestyle='--')
plt.plot(acc_b, label="CL W-Fisher + SCTS (L2)", color='blue')
plt.plot(acc_c, label="CL W-Fisher + A-SCTS", color='orange')
plt.plot(acc_d, label="CL W-Fisher + A-SCTS + CP-AM (Ours)", color='red', linewidth=2)
plt.axvline(10, color='gray', alpha=0.5, linestyle=':')
plt.axvline(20, color='gray', alpha=0.5, linestyle=':')
plt.axvline(30, color='gray', alpha=0.5, linestyle=':')
plt.axvline(40, color='gray', alpha=0.5, linestyle=':')
plt.title("Test-Time Model Merging Accuracy")
plt.xlabel("Batch Index")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lam_a, label="Fixed TTA + Reset (Baseline)", color='gray', linestyle='--')
plt.plot(lam_b, label="CL W-Fisher + SCTS (L2)", color='blue')
plt.plot(lam_c, label="CL W-Fisher + A-SCTS", color='orange')
plt.plot(lam_d, label="CL W-Fisher + A-SCTS + CP-AM (Ours)", color='red', linewidth=2)
plt.axvline(10, color='gray', alpha=0.5, linestyle=':')
plt.axvline(20, color='gray', alpha=0.5, linestyle=':')
plt.axvline(30, color='gray', alpha=0.5, linestyle=':')
plt.axvline(40, color='gray', alpha=0.5, linestyle=':')
plt.title("Merging Coefficient (λ0 - MNIST)")
plt.xlabel("Batch Index")
plt.ylabel("MNIST Expert Weight")
plt.legend()

plt.tight_layout()
plt.savefig("tta_results_comparison.png", dpi=300)
print("\nPlot saved successfully as tta_results_comparison.png!")


# WRITE RESULTS SUMMARY FILE FOR THE PAPER
with open("experimental_results.txt", "w") as f:
    f.write("Test-Time Model Merging Experimental Results\n")
    f.write("=========================================\n\n")
    
    for name, segs in [("Fixed TTA + Reset", seg_a), ("CL W-Fisher + SCTS (L2)", seg_b), ("CL W-Fisher + A-SCTS (Angular)", seg_c), ("CL W-Fisher + A-SCTS + CP-AM (Ours)", seg_d)]:
        f.write(f"Method: {name}\n")
        f.write("-----------------------------------------\n")
        for k, v in segs.items():
            f.write(f"  {k}: {v:.2f}%\n")
        f.write("\n")
print("Results summary saved to experimental_results.txt!")
