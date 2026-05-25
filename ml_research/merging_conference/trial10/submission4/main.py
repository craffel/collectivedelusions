import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# CosFace Linear Classifier definition
class CosFaceLinear(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFaceLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        weight_norm = F.normalize(self.weight, dim=1)
        input_norm = F.normalize(input, dim=1)
        cosine = F.linear(input_norm, weight_norm)
        
        if label is None:
            return cosine * self.s
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output

# SimpleCNN expert model definition
class SimpleCNN(nn.Module):
    def __init__(self, is_cosface=False, s=30.0, m=0.35):
        super(SimpleCNN, self).__init__()
        self.is_cosface = is_cosface
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Input size: 28x28 -> after two poolings: 7x7
        # 64 * 7 * 7 = 3136
        self.fc = nn.Linear(3136, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.25)
        
        if is_cosface:
            self.classifier = CosFaceLinear(128, 10, s=s, m=m)
        else:
            self.classifier = nn.Linear(128, 10)

    def get_features(self, x, layer=2):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        if layer == 1:
            return x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        if layer == 2:
            return x
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        if layer == 3:
            return x
        return x

    def forward(self, x, label=None):
        x = self.get_features(x, layer=4)
        x = self.dropout2(x)
        if self.is_cosface:
            output = self.classifier(x, label)
        else:
            output = self.classifier(x)
        return output

# Hoyer sparsity metric
def hoyer_sparsity(f):
    if len(f.shape) > 2:
        f = f.flatten(start_dim=1)
    elif len(f.shape) == 1:
        f = f.unsqueeze(0)
    
    d = f.shape[1]
    norm1 = torch.norm(f, p=1, dim=1)
    norm2 = torch.norm(f, p=2, dim=1)
    norm2 = torch.clamp(norm2, min=1e-8)
    
    numerator = math.sqrt(d) - (norm1 / norm2)
    denominator = math.sqrt(d) - 1.0
    sparsity = numerator / denominator
    return sparsity.mean()

# Pixel-level Hoyer Sparsity calculation
def compute_pixel_sparsity(batch_x):
    # Map normalized [-1, 1] back to [0, 1]
    x_pos = (batch_x + 1.0) / 2.0
    # Threshold denoising
    x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
    return hoyer_sparsity(x_denoised)

# Proposed Feature-level Hoyer Sparsity calculation
def compute_feature_sparsity(batch_x, std_exp0, std_exp1):
    with torch.no_grad():
        feat0 = std_exp0.get_features(batch_x, layer=2) # Shape: (B, 64, 7, 7)
        feat1 = std_exp1.get_features(batch_x, layer=2) # Shape: (B, 64, 7, 7)
        
        # Apply ReLU activation
        act0 = F.relu(feat0)
        act1 = F.relu(feat1)
        
        # Adaptive thresholding based on mean of features
        alpha = 1.5
        th0 = alpha * act0.mean()
        th1 = alpha * act1.mean()
        
        act0 = torch.where(act0 > th0, act0, torch.zeros_like(act0))
        act1 = torch.where(act1 > th1, act1, torch.zeros_like(act1))
        
        sparsity0 = hoyer_sparsity(act0)
        sparsity1 = hoyer_sparsity(act1)
        return 0.5 * (sparsity0 + sparsity1).item()

# Dataset preparation helpers
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download datasets
    os.makedirs("./data", exist_ok=True)
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    fashion_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    
    kmnist_test = datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    # Subset to 10,000 training samples
    mnist_train_sub = Subset(mnist_train, list(range(10000)))
    fashion_train_sub = Subset(fashion_train, list(range(10000)))
    
    return mnist_train_sub, mnist_test, fashion_train_sub, fashion_test, kmnist_test

# Expert training function
def train_expert(model, train_loader, epochs=2, device="cpu"):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if model.is_cosface:
                out = model(x, y)
                loss = F.cross_entropy(out, y)
            else:
                out = model(x)
                loss = F.cross_entropy(out, y)
                
            loss.backward()
            optimizer.step()
            
    return model

# Evaluate an expert on clean test loader
def eval_expert(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Parameter and BatchNorm merging logic
def load_state_dicts(model0, model1):
    params0 = {k: v.clone() for k, v in model0.state_dict().items() if v.requires_grad or 'weight' in k or 'bias' in k}
    params1 = {k: v.clone() for k, v in model1.state_dict().items() if v.requires_grad or 'weight' in k or 'bias' in k}
    
    buffers0 = {k: v.clone() for k, v in model0.state_dict().items() if k not in params0}
    buffers1 = {k: v.clone() for k, v in model1.state_dict().items() if k not in params1}
    
    return params0, params1, buffers0, buffers1

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

def merge_parameters(w_global, offsets, params0, params1, keys):
    merged = {}
    lambdas = {}
    for key in keys:
        lambdas[key] = torch.sigmoid(w_global + offsets[key])
        merged[key] = lambdas[key] * params1[key] + (1.0 - lambdas[key]) * params0[key]
    return merged, lambdas

def merge_bn_buffers(lambda_det, buffers0, buffers1):
    merged = {}
    for key in buffers0.keys():
        if 'running_mean' in key:
            merged[key] = lambda_det * buffers1[key] + (1.0 - lambda_det) * buffers0[key]
        elif 'running_var' in key:
            mean_key = key.replace('running_var', 'running_mean')
            mean0 = buffers0[mean_key]
            mean1 = buffers1[mean_key]
            mean_fused = lambda_det * mean1 + (1.0 - lambda_det) * mean0
            
            var0 = buffers0[key]
            var1 = buffers1[key]
            merged[key] = (lambda_det * (var1 + (mean1 - mean_fused)**2) +
                           (1.0 - lambda_det) * (var0 + (mean0 - mean_fused)**2))
        else:
            merged[key] = buffers0[key]
    return merged

# Precomputing Prototypes
def compute_prototypes(model0, model1, loader0, loader1, device="cpu"):
    model0.to(device)
    model1.to(device)
    model0.eval()
    model1.eval()
    
    proto0 = {c: [] for c in range(10)}
    proto1 = {c: [] for c in range(10)}
    
    with torch.no_grad():
        for x, y in loader0:
            x, y = x.to(device), y.to(device)
            feats = model0.get_features(x, layer=3) # Shape: (B, 128)
            for i in range(y.size(0)):
                label = y[i].item()
                proto0[label].append(feats[i])
                
        for x, y in loader1:
            x, y = x.to(device), y.to(device)
            feats = model1.get_features(x, layer=3)
            for i in range(y.size(0)):
                label = y[i].item()
                proto1[label].append(feats[i])
                
    # Average and conditionally normalize prototypes
    P0 = torch.zeros(10, 128, device=device)
    P1 = torch.zeros(10, 128, device=device)
    
    for c in range(10):
        if len(proto0[c]) > 0:
            feats = torch.stack(proto0[c])
            mean_feat = feats.mean(dim=0)
            if model0.is_cosface:
                P0[c] = F.normalize(mean_feat, p=2, dim=0) # CosFace: normalize to unit sphere
            else:
                P0[c] = mean_feat # Standard: keep unnormalized to avoid scale mismatch
        else:
            P0[c] = torch.zeros(128, device=device)
            
        if len(proto1[c]) > 0:
            feats = torch.stack(proto1[c])
            mean_feat = feats.mean(dim=0)
            if model1.is_cosface:
                P1[c] = F.normalize(mean_feat, p=2, dim=0) # CosFace: normalize to unit sphere
            else:
                P1[c] = mean_feat # Standard: keep unnormalized to avoid scale mismatch
        else:
            P1[c] = torch.zeros(128, device=device)
            
    return P0, P1

# Main TTMM adapting function
def evaluate_stream(std_exp0, std_exp1, P0_std, P1_std,
                    cos_exp0, cos_exp1, P0_cos, P1_cos,
                    test_stream, method, device="cpu"):
    std_exp0.eval()
    std_exp1.eval()
    cos_exp0.eval()
    cos_exp1.eval()
    
    # Tracking results
    accuracies = []
    
    for batch_idx, (batch_x, batch_y) in enumerate(test_stream):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        B = batch_x.size(0)
        
        # 1. Sparsity / Density Estimation
        pixel_sparsity = compute_pixel_sparsity(batch_x).item()
        feat_sparsity = compute_feature_sparsity(batch_x, std_exp0, std_exp1)
        
        # Choose gating based on method
        if method == "BK-AHR":
            # Pixel-level Hoyer Sparsity thresholded
            is_sparse = pixel_sparsity >= 0.50
        elif method == "FL-AHR (Ours)":
            # Proposed Feature-Level activation sparsity
            is_sparse = feat_sparsity >= 0.535
        else:
            is_sparse = False
            
        # 2. Dynamic Routing of Expert Family and SCTS Distance Metric
        if method in ["BK-AHR", "FL-AHR (Ours)"]:
            if is_sparse:
                # Sparse domain -> Standard Experts with Euclidean distance SCTS
                expert0, expert1 = std_exp0, std_exp1
                P0, P1 = P0_std, P1_std
                use_angular = False
            else:
                # Dense domain -> CosFace Experts with Angular SCTS
                expert0, expert1 = cos_exp0, cos_exp1
                P0, P1 = P0_cos, P1_cos
                use_angular = True
        elif method == "CP-AM (Baseline)":
            # Baseline always uses CosFace Experts and Angular SCTS
            expert0, expert1 = cos_exp0, cos_exp1
            P0, P1 = P0_cos, P1_cos
            use_angular = True
        else:
            # Fixed TTA, MoG-L2, MoG-Angular always use Standard Experts
            expert0, expert1 = std_exp0, std_exp1
            P0, P1 = P0_std, P1_std
            use_angular = (method == "MoG-Angular")
            
        # Extract parameters and buffers for the active expert family
        params0, params1, buffers0, buffers1 = load_state_dicts(expert0, expert1)
        trainable_keys = [k for k in params0.keys() if 'weight' in k or 'bias' in k]
        
        # Compute routing priors
        with torch.no_grad():
            feats0 = expert0.get_features(batch_x, layer=3) # Shape: (B, 128)
            feats1 = expert1.get_features(batch_x, layer=3)
            
            if use_angular:
                # Angular distance SCTS
                norm0 = F.normalize(feats0, p=2, dim=1)
                norm1 = F.normalize(feats1, p=2, dim=1)
                d0_batch = []
                d1_batch = []
                for i in range(B):
                    d0_batch.append((1.0 - F.linear(norm0[i].unsqueeze(0), P0)).min())
                    d1_batch.append((1.0 - F.linear(norm1[i].unsqueeze(0), P1)).min())
                D0 = torch.stack(d0_batch).mean()
                D1 = torch.stack(d1_batch).mean()
            else:
                # Euclidean distance SCTS
                norm0 = feats0
                norm1 = feats1
                d0_batch = []
                d1_batch = []
                for i in range(B):
                    d0_batch.append(torch.norm(norm0[i].unsqueeze(0) - P0, p=2, dim=1).min())
                    d1_batch.append(torch.norm(norm1[i].unsqueeze(0) - P1, p=2, dim=1).min())
                D0 = torch.stack(d0_batch).mean()
                D1 = torch.stack(d1_batch).mean()
            
            gap = torch.abs(D0 - D1)
            # Decisive Under Noise (DUN) scaling
            Havg_prior = 0.5
            epsilon_base = 0.04 if use_angular else 0.08
            epsilon_stab = epsilon_base / (1.0 + 2.0 * Havg_prior)
            
            tau = (gap / 3.0) + epsilon_stab
            w1 = torch.exp(-D1 / tau) / (torch.exp(-D0 / tau) + torch.exp(-D1 / tau))
            w0 = 1.0 - w1
            
        # 3. Test-Time Optimization initialization
        w_global = torch.tensor(math.log(w1 / w0), device=device, requires_grad=True)
        offsets = {k: torch.zeros_like(params0[k], device=device, requires_grad=True) for k in trainable_keys}
        
        optimizer_params = [w_global] + list(offsets.values())
        optimizer = optim.SGD(optimizer_params, lr=0.05)
        
        N_step = 5
        for step in range(N_step):
            optimizer.zero_grad()
            
            # Merge parameters dynamically
            merged_params, lambdas = merge_parameters(w_global, offsets, params0, params1, trainable_keys)
            
            # Merge Batch Normalization statistics
            lambda_det = torch.sigmoid(w_global).detach()
            merged_buffers = merge_bn_buffers(lambda_det, buffers0, buffers1)
            
            all_state = {**merged_params, **merged_buffers}
            
            # Predict
            outputs = functional_call(expert0, all_state, batch_x)
            
            # Predictive entropy loss
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            # Prior KL penalty
            lambda_mean = torch.stack([lambdas[k].mean() for k in trainable_keys]).mean()
            kl_loss = lambda_mean * torch.log(lambda_mean / w1) + (1.0 - lambda_mean) * torch.log((1.0 - lambda_mean) / w0)
            
            loss = entropy + 1.5 * kl_loss
            loss.backward()
            optimizer.step()
            
        # Final Evaluation after batch adaptation
        with torch.no_grad():
            merged_params, _ = merge_parameters(w_global, offsets, params0, params1, trainable_keys)
            lambda_det = torch.sigmoid(w_global).detach()
            merged_buffers = merge_bn_buffers(lambda_det, buffers0, buffers1)
            all_state = {**merged_params, **merged_buffers}
            
            outputs = functional_call(expert0, all_state, batch_x)
            preds = outputs.argmax(dim=1)
            acc = (preds == batch_y).sum().item() / B
            accuracies.append(acc)
            
    return np.mean(accuracies), accuracies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Datasets
    print("Preparing Datasets...")
    mnist_train, mnist_test, fashion_train, fashion_test, kmnist_test = get_datasets()
    
    loader_mnist_train = DataLoader(mnist_train, batch_size=64, shuffle=True)
    loader_mnist_test = DataLoader(mnist_test, batch_size=64, shuffle=False)
    
    loader_fashion_train = DataLoader(fashion_train, batch_size=64, shuffle=True)
    loader_fashion_test = DataLoader(fashion_test, batch_size=64, shuffle=False)
    
    loader_kmnist_test = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    # 2. Expert Training or Loading Checkpoints
    os.makedirs("./checkpoints", exist_ok=True)
    
    standard_expert0 = SimpleCNN(is_cosface=False)
    standard_expert1 = SimpleCNN(is_cosface=False)
    cosface_expert0 = SimpleCNN(is_cosface=True)
    cosface_expert1 = SimpleCNN(is_cosface=True)
    
    std_mnist_path = "./checkpoints/standard_expert_mnist.pt"
    std_fashion_path = "./checkpoints/standard_expert_fashion.pt"
    cos_mnist_path = "./checkpoints/cosface_expert_mnist.pt"
    cos_fashion_path = "./checkpoints/cosface_expert_fashion.pt"
    
    # Train Standard Experts
    if os.path.exists(std_mnist_path):
        print("Loading pre-trained Standard Expert 0 (MNIST)...")
        standard_expert0.load_state_dict(torch.load(std_mnist_path, map_location=device, weights_only=True))
    else:
        print("Training Standard Expert 0 (MNIST)...")
        standard_expert0 = train_expert(standard_expert0, loader_mnist_train, epochs=2, device=device)
        torch.save(standard_expert0.state_dict(), std_mnist_path)
        
    if os.path.exists(std_fashion_path):
        print("Loading pre-trained Standard Expert 1 (FashionMNIST)...")
        standard_expert1.load_state_dict(torch.load(std_fashion_path, map_location=device, weights_only=True))
    else:
        print("Training Standard Expert 1 (FashionMNIST)...")
        standard_expert1 = train_expert(standard_expert1, loader_fashion_train, epochs=2, device=device)
        torch.save(standard_expert1.state_dict(), std_fashion_path)
        
    # Train CosFace Experts
    if os.path.exists(cos_mnist_path):
        print("Loading pre-trained CosFace Expert 0 (MNIST)...")
        cosface_expert0.load_state_dict(torch.load(cos_mnist_path, map_location=device, weights_only=True))
    else:
        print("Training CosFace Expert 0 (MNIST)...")
        cosface_expert0 = train_expert(cosface_expert0, loader_mnist_train, epochs=2, device=device)
        torch.save(cosface_expert0.state_dict(), cos_mnist_path)
        
    if os.path.exists(cos_fashion_path):
        print("Loading pre-trained CosFace Expert 1 (FashionMNIST)...")
        cosface_expert1.load_state_dict(torch.load(cos_fashion_path, map_location=device, weights_only=True))
    else:
        print("Training CosFace Expert 1 (FashionMNIST)...")
        cosface_expert1 = train_expert(cosface_expert1, loader_fashion_train, epochs=2, device=device)
        torch.save(cosface_expert1.state_dict(), cos_fashion_path)
        
    # Print Expert accuracies
    print("\nEvaluating expert performance on clean test data:")
    acc_std_mnist = eval_expert(standard_expert0, loader_mnist_test, device=device)
    acc_std_fashion = eval_expert(standard_expert1, loader_fashion_test, device=device)
    acc_cos_mnist = eval_expert(cosface_expert0, loader_mnist_test, device=device)
    acc_cos_fashion = eval_expert(cosface_expert1, loader_fashion_test, device=device)
    print(f"Standard Expert 0 (MNIST) Accuracy: {acc_std_mnist*100:.2f}%")
    print(f"Standard Expert 1 (FashionMNIST) Accuracy: {acc_std_fashion*100:.2f}%")
    print(f"CosFace Expert 0 (MNIST) Accuracy: {acc_cos_mnist*100:.2f}%")
    print(f"CosFace Expert 1 (FashionMNIST) Accuracy: {acc_cos_fashion*100:.2f}%")
    
    # 3. Construct the non-stationary, noisy stream of 50 batches
    print("\nConstructing 50-batch non-stationary test stream...")
    set_seed(42) # Ensure stream construction is deterministic
    
    # Extract batches
    mnist_iter = iter(loader_mnist_test)
    fashion_iter = iter(loader_fashion_test)
    kmnist_iter = iter(loader_kmnist_test)
    
    test_stream = [] # List of tuples: (batch_x, batch_y)
    
    # Phase 1: Clean MNIST (batches 0-9)
    for _ in range(10):
        test_stream.append(next(mnist_iter))
        
    # Phase 2: Noisy MNIST (batches 10-19)
    for _ in range(10):
        x, y = next(mnist_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_stream.append((x_noisy, y))
        
    # Phase 3: Clean FashionMNIST (batches 20-29)
    for _ in range(10):
        test_stream.append(next(fashion_iter))
        
    # Phase 4: Noisy FashionMNIST (batches 30-39)
    for _ in range(10):
        x, y = next(fashion_iter)
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_stream.append((x_noisy, y))
        
    # Phase 5: Novel KMNIST (batches 40-49)
    for _ in range(10):
        test_stream.append(next(kmnist_iter))
        
    # 4. Precompute Class Prototypes for standard and CosFace experts
    print("\nPrecomputing Class Prototypes...")
    P0_std, P1_std = compute_prototypes(standard_expert0, standard_expert1, loader_mnist_test, loader_fashion_test, device=device)
    P0_cos, P1_cos = compute_prototypes(cosface_expert0, cosface_expert1, loader_mnist_test, loader_fashion_test, device=device)
    
    # 5. Run Evaluations
    results = {}
    print("\nRunning stream evaluations:")
    
    method_names = [
        "Fixed TTA",
        "MoG-L2",
        "MoG-Angular",
        "CP-AM (Baseline)",
        "BK-AHR",
        "FL-AHR (Ours)"
    ]
    
    for method_name in method_names:
        print(f"Evaluating {method_name}...")
        mean_acc, accuracies = evaluate_stream(
            standard_expert0, standard_expert1, P0_std, P1_std,
            cosface_expert0, cosface_expert1, P0_cos, P1_cos,
            test_stream, method_name, device=device
        )
        results[method_name] = (mean_acc, accuracies)
        print(f"{method_name} Overall Streaming Accuracy: {mean_acc*100:.2f}%")
        
        # Print breakdown by phase
        print("  Breakdown by Phase:")
        for phase in range(5):
            phase_acc = np.mean(accuracies[phase*10: (phase+1)*10])
            print(f"    Phase {phase+1}: {phase_acc*100:.2f}%")
            
    # Write summary results
    print("\nSummary of Results:")
    for m in method_names:
        acc, _ = results[m]
        print(f"  {m}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
