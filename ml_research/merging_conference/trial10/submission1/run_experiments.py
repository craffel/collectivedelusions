import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.func import functional_call

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

set_seed(42)

# Define SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.use_cosface = use_cosface
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        if self.use_cosface:
            self.classifier = nn.Parameter(torch.randn(num_classes, 128))
            nn.init.xavier_uniform_(self.classifier)
        else:
            self.classifier = nn.Linear(128, num_classes)
            
    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        features = F.relu(self.bn3(self.fc1(x)))
        
        if return_features:
            return features
            
        if self.use_cosface:
            features_norm = F.normalize(features, p=2, dim=1)
            classifier_norm = F.normalize(self.classifier, p=2, dim=1)
            logits = torch.matmul(features_norm, classifier_norm.t())
            return logits
        else:
            logits = self.classifier(features)
            return logits

# Define CosFace Loss
class CosFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.s = s
        self.m = m
        
    def forward(self, logits, targets):
        cos_theta = logits
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        output = self.s * (cos_theta - one_hot * self.m)
        return F.cross_entropy(output, targets)

# Model training function
def train_expert(model, dataset_name, train_loader, device, num_epochs=2, use_cosface=False):
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = CosFaceLoss() if use_cosface else nn.CrossEntropyLoss()
    
    print(f"Training expert on {dataset_name} for {num_epochs} epochs (CosFace={use_cosface})...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    return model

# Function to compute Hoyer Sparsity
def compute_hoyer_sparsity(x):
    f = x.flatten()
    d = len(f)
    l1 = torch.sum(torch.abs(f))
    l2 = torch.sqrt(torch.sum(f**2))
    if l2 == 0:
        return 0.0
    hoyer = (np.sqrt(d) - l1 / l2) / (np.sqrt(d) - 1.0)
    return hoyer.item()

# Moment-matching fusion of Batch Normalization running statistics
def fuse_bn_stats(model0, model1, merged_model, lambda_det):
    state_dict0 = model0.state_dict()
    state_dict1 = model1.state_dict()
    merged_state = merged_model.state_dict()
    
    for name in state_dict0:
        if "running_mean" in name:
            mean0 = state_dict0[name]
            mean1 = state_dict1[name]
            merged_state[name] = (1.0 - lambda_det) * mean0 + lambda_det * mean1
        elif "running_var" in name:
            var0 = state_dict0[name]
            var1 = state_dict1[name]
            mean_name = name.replace("running_var", "running_mean")
            mean0 = state_dict0[mean_name]
            mean1 = state_dict1[mean_name]
            mean_fused = merged_state[mean_name]
            
            merged_state[name] = (1.0 - lambda_det) * (var0 + (mean0 - mean_fused)**2) + \
                                 lambda_det * (var1 + (mean1 - mean_fused)**2)
                                 
    merged_model.load_state_dict(merged_state)

# Function to clone a model (deep copy)
def clone_model(model):
    clone = SimpleCNN(use_cosface=model.use_cosface, num_classes=model.num_classes)
    clone.load_state_dict(model.state_dict())
    device = next(model.parameters()).device
    clone.to(device)
    return clone

# Precompute class prototypes
def compute_prototypes(model, data_loader, device, num_samples=256):
    model.to(device)
    model.eval()
    
    class_features = {c: [] for c in range(10)}
    samples_collected = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model(images, return_features=True)
            
            for f, l in zip(features, labels):
                l_item = l.item()
                if len(class_features[l_item]) < num_samples // 10:
                    class_features[l_item].append(f.cpu())
                    samples_collected += 1
                    
            if samples_collected >= num_samples:
                break
                
    prototypes = torch.zeros(10, 128)
    for c in range(10):
        if class_features[c]:
            features_c = torch.stack(class_features[c])
            mean_f = torch.mean(features_c, dim=0)
            if model.use_cosface:
                mean_f = F.normalize(mean_f, p=2, dim=0)
            prototypes[c] = mean_f
            
    return prototypes.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Download and Prepare Datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    g = torch.Generator().manual_seed(42)
    mnist_train_sub, _ = torch.utils.data.random_split(mnist_train, [10000, len(mnist_train)-10000], generator=g)
    fashion_train_sub, _ = torch.utils.data.random_split(fashion_train, [10000, len(fashion_train)-10000], generator=g)
    
    train_loader_mnist = torch.utils.data.DataLoader(mnist_train_sub, batch_size=64, shuffle=True)
    train_loader_fashion = torch.utils.data.DataLoader(fashion_train_sub, batch_size=64, shuffle=True)
    
    # 2. Train/Load Experts
    expert_mnist_std = SimpleCNN(use_cosface=False)
    expert_fashion_std = SimpleCNN(use_cosface=False)
    expert_mnist_cos = SimpleCNN(use_cosface=True)
    expert_fashion_cos = SimpleCNN(use_cosface=True)
    
    if os.path.exists("mnist_std.pth"):
        expert_mnist_std.load_state_dict(torch.load("mnist_std.pth", map_location=device))
        print("Loaded pre-trained MNIST Standard Expert.")
    else:
        expert_mnist_std = train_expert(expert_mnist_std, "MNIST_Std", train_loader_mnist, device, num_epochs=2, use_cosface=False)
        torch.save(expert_mnist_std.state_dict(), "mnist_std.pth")
        
    if os.path.exists("fashion_std.pth"):
        expert_fashion_std.load_state_dict(torch.load("fashion_std.pth", map_location=device))
        print("Loaded pre-trained Fashion Standard Expert.")
    else:
        expert_fashion_std = train_expert(expert_fashion_std, "Fashion_Std", train_loader_fashion, device, num_epochs=2, use_cosface=False)
        torch.save(expert_fashion_std.state_dict(), "fashion_std.pth")
        
    if os.path.exists("mnist_cos.pth"):
        expert_mnist_cos.load_state_dict(torch.load("mnist_cos.pth", map_location=device))
        print("Loaded pre-trained MNIST CosFace Expert.")
    else:
        expert_mnist_cos = train_expert(expert_mnist_cos, "MNIST_CosFace", train_loader_mnist, device, num_epochs=2, use_cosface=True)
        torch.save(expert_mnist_cos.state_dict(), "mnist_cos.pth")
        
    if os.path.exists("fashion_cos.pth"):
        expert_fashion_cos.load_state_dict(torch.load("fashion_cos.pth", map_location=device))
        print("Loaded pre-trained Fashion CosFace Expert.")
    else:
        expert_fashion_cos = train_expert(expert_fashion_cos, "Fashion_CosFace", train_loader_fashion, device, num_epochs=2, use_cosface=True)
        torch.save(expert_fashion_cos.state_dict(), "fashion_cos.pth")
        
    expert_mnist_std.eval().to(device)
    expert_fashion_std.eval().to(device)
    expert_mnist_cos.eval().to(device)
    expert_fashion_cos.eval().to(device)
    
    # 3. Precompute Class Prototypes
    test_loader_mnist_init = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)
    test_loader_fashion_init = torch.utils.data.DataLoader(fashion_test, batch_size=128, shuffle=False)
    
    proto_mnist_std = compute_prototypes(expert_mnist_std, test_loader_mnist_init, device)
    proto_fashion_std = compute_prototypes(expert_fashion_std, test_loader_fashion_init, device)
    proto_mnist_cos = compute_prototypes(expert_mnist_cos, test_loader_mnist_init, device)
    proto_fashion_cos = compute_prototypes(expert_fashion_cos, test_loader_fashion_init, device)
    
    # 4. Construct Non-Stationary Test Stream
    stream_batches = []
    test_loader_mnist = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(101))
    test_loader_fashion = torch.utils.data.DataLoader(fashion_test, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(102))
    test_loader_kmnist = torch.utils.data.DataLoader(kmnist_test, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(103))
    
    mnist_iter = iter(test_loader_mnist)
    fashion_iter = iter(test_loader_fashion)
    kmnist_iter = iter(test_loader_kmnist)
    
    for _ in range(10):
        images, labels = next(mnist_iter)
        stream_batches.append(("Clean MNIST", images, labels))
    for _ in range(10):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append(("Noisy MNIST", images_noisy, labels))
    for _ in range(10):
        images, labels = next(fashion_iter)
        stream_batches.append(("Clean FashionMNIST", images, labels))
    for _ in range(10):
        images, labels = next(fashion_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append(("Noisy FashionMNIST", images_noisy, labels))
    for _ in range(10):
        images, labels = next(kmnist_iter)
        stream_batches.append(("Novel KMNIST", images, labels))
        
    def entropy_loss_fn(outputs):
        probs = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
        return torch.mean(entropy)
        
    methods = ["Method A", "Method B", "Method C", "Method D", "Method E", "Method F (Ours)"]
    method_accuracies = {m: [] for m in methods}
    
    # Evaluate Method A: Fixed TTA + Reset
    print("\n--- Running Method A: Fixed TTA + Reset (L2) ---")
    for b_idx, (phase, images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        feat0 = expert_mnist_std(images, return_features=True)
        feat1 = expert_fashion_std(images, return_features=True)
        d0 = torch.mean(torch.stack([torch.min(torch.norm(f - proto_mnist_std, dim=1)) for f in feat0]))
        d1 = torch.mean(torch.stack([torch.min(torch.norm(f - proto_fashion_std, dim=1)) for f in feat1]))
        
        gap = abs(d0.item() - d1.item())
        tau = gap / 3.0 + 0.08
        w1 = np.exp(-d0.item() / tau)
        w2 = np.exp(-d1.item() / tau)
        w_sum = w1 + w2
        w1, w2 = w1 / w_sum, w2 / w_sum
        
        merged = clone_model(expert_mnist_std)
        state_merged = merged.state_dict()
        state0 = expert_mnist_std.state_dict()
        state1 = expert_fashion_std.state_dict()
        for key in state_merged:
            if "weight" in key or "bias" in key:
                state_merged[key] = w1 * state0[key] + w2 * state1[key]
        merged.load_state_dict(state_merged)
        fuse_bn_stats(expert_mnist_std, expert_fashion_std, merged, w2)
        
        merged.train()
        opt = optim.SGD(merged.parameters(), lr=0.01)
        opt.zero_grad()
        out = merged(images)
        loss = entropy_loss_fn(out)
        loss.backward()
        opt.step()
        
        merged.eval()
        with torch.no_grad():
            out = merged(images)
            _, pred = out.max(1)
            acc = pred.eq(labels).sum().item() / labels.size(0) * 100.0
            method_accuracies["Method A"].append(acc)
            
    # Evaluate Method B: CL W-Fisher + SCTS (L2) with Corrected Gradient Propagation
    print("--- Running Method B: CL W-Fisher + SCTS (L2) ---")
    for b_idx, (phase, images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        feat0 = expert_mnist_std(images, return_features=True)
        feat1 = expert_fashion_std(images, return_features=True)
        d0 = torch.mean(torch.stack([torch.min(torch.norm(f - proto_mnist_std, dim=1)) for f in feat0])).item()
        d1 = torch.mean(torch.stack([torch.min(torch.norm(f - proto_fashion_std, dim=1)) for f in feat1])).item()
        
        gap = abs(d0 - d1)
        tau = gap / 3.0 + 0.08
        w0 = np.exp(-d0 / tau)
        w1 = np.exp(-d1 / tau)
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        
        merged = clone_model(expert_mnist_std)
        state0 = expert_mnist_std.state_dict()
        state1 = expert_fashion_std.state_dict()
        
        global_w = torch.tensor(np.log(w1 / (w0 + 1e-7)), requires_grad=True, device=device)
        
        for _ in range(5):
            if global_w.grad is not None:
                global_w.grad.zero_()
                
            lam = torch.sigmoid(global_w)
            params_dict = {}
            for name, param in merged.named_parameters():
                params_dict[name] = (1.0 - lam) * state0[name] + lam * state1[name]
                
            fuse_bn_stats(expert_mnist_std, expert_fashion_std, merged, lam.item())
            out = functional_call(merged, params_dict, (images,))
            
            loss = entropy_loss_fn(out) + 0.1 * (lam * torch.log(lam / w1) + (1.0 - lam) * torch.log((1.0 - lam) / w0))
            loss.backward()
            
            with torch.no_grad():
                global_w -= 0.05 * global_w.grad
                
        with torch.no_grad():
            lam = torch.sigmoid(global_w)
            state_merged = merged.state_dict()
            for key in state_merged:
                if "weight" in key or "bias" in key:
                    state_merged[key] = (1.0 - lam) * state0[key] + lam * state1[key]
            merged.load_state_dict(state_merged)
            fuse_bn_stats(expert_mnist_std, expert_fashion_std, merged, lam.item())
            
        merged.eval()
        with torch.no_grad():
            out = merged(images)
            _, pred = out.max(1)
            acc = pred.eq(labels).sum().item() / labels.size(0) * 100.0
            method_accuracies["Method B"].append(acc)

    # Evaluate Method C: CL W-Fisher + A-SCTS (Angular) with Corrected Gradient Propagation
    print("--- Running Method C: CL W-Fisher + A-SCTS (Angular) ---")
    for b_idx, (phase, images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        feat0 = F.normalize(expert_mnist_std(images, return_features=True), p=2, dim=1)
        feat1 = F.normalize(expert_fashion_std(images, return_features=True), p=2, dim=1)
        
        proto0_norm = F.normalize(proto_mnist_std, p=2, dim=1)
        proto1_norm = F.normalize(proto_fashion_std, p=2, dim=1)
        
        d0 = torch.mean(torch.stack([torch.min(torch.norm(f - proto0_norm, dim=1)) for f in feat0])).item()
        d1 = torch.mean(torch.stack([torch.min(torch.norm(f - proto1_norm, dim=1)) for f in feat1])).item()
        
        gap = abs(d0 - d1)
        tau = gap / 3.0 + 0.04
        w0 = np.exp(-d0 / tau)
        w1 = np.exp(-d1 / tau)
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        
        merged = clone_model(expert_mnist_std)
        state0 = expert_mnist_std.state_dict()
        state1 = expert_fashion_std.state_dict()
        
        global_w = torch.tensor(np.log(w1 / (w0 + 1e-7)), requires_grad=True, device=device)
        
        for _ in range(5):
            if global_w.grad is not None:
                global_w.grad.zero_()
                
            lam = torch.sigmoid(global_w)
            params_dict = {}
            for name, param in merged.named_parameters():
                params_dict[name] = (1.0 - lam) * state0[name] + lam * state1[name]
                
            fuse_bn_stats(expert_mnist_std, expert_fashion_std, merged, lam.item())
            out = functional_call(merged, params_dict, (images,))
            
            loss = entropy_loss_fn(out) + 0.1 * (lam * torch.log(lam / w1) + (1.0 - lam) * torch.log((1.0 - lam) / w0))
            loss.backward()
            
            with torch.no_grad():
                global_w -= 0.05 * global_w.grad
                
        with torch.no_grad():
            lam = torch.sigmoid(global_w)
            state_merged = merged.state_dict()
            for key in state_merged:
                if "weight" in key or "bias" in key:
                    state_merged[key] = (1.0 - lam) * state0[key] + lam * state1[key]
            merged.load_state_dict(state_merged)
            fuse_bn_stats(expert_mnist_std, expert_fashion_std, merged, lam.item())
            
        merged.eval()
        with torch.no_grad():
            out = merged(images)
            _, pred = out.max(1)
            acc = pred.eq(labels).sum().item() / labels.size(0) * 100.0
            method_accuracies["Method C"].append(acc)

    # Evaluate Method D: CP-AM with CosFace experts & Corrected Gradient Propagation
    print("--- Running Method D: CP-AM (CosFace experts) ---")
    for b_idx, (phase, images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        feat0 = expert_mnist_cos(images, return_features=True)
        feat1 = expert_fashion_cos(images, return_features=True)
        feat0_norm = F.normalize(feat0, p=2, dim=1)
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        
        d0 = torch.mean(torch.stack([torch.min(torch.norm(f - proto_mnist_cos, dim=1)) for f in feat0_norm])).item()
        d1 = torch.mean(torch.stack([torch.min(torch.norm(f - proto_fashion_cos, dim=1)) for f in feat1_norm])).item()
        
        gap = abs(d0 - d1)
        tau = gap / 3.0 + 0.04
        w0 = np.exp(-d0 / tau)
        w1 = np.exp(-d1 / tau)
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        
        merged = clone_model(expert_mnist_cos)
        state0 = expert_mnist_cos.state_dict()
        state1 = expert_fashion_cos.state_dict()
        
        global_w = torch.tensor(np.log(w1 / (w0 + 1e-7)), requires_grad=True, device=device)
        
        for _ in range(5):
            if global_w.grad is not None:
                global_w.grad.zero_()
                
            lam = torch.sigmoid(global_w)
            params_dict = {}
            for name, param in merged.named_parameters():
                params_dict[name] = (1.0 - lam) * state0[name] + lam * state1[name]
                
            fuse_bn_stats(expert_mnist_cos, expert_fashion_cos, merged, lam.item())
            out = functional_call(merged, params_dict, (images,))
            
            loss = entropy_loss_fn(out) + 0.1 * (lam * torch.log(lam / w1) + (1.0 - lam) * torch.log((1.0 - lam) / w0))
            loss.backward()
            
            with torch.no_grad():
                global_w -= 0.05 * global_w.grad
                
        with torch.no_grad():
            lam = torch.sigmoid(global_w)
            state_merged = merged.state_dict()
            for key in state_merged:
                if "weight" in key or "bias" in key:
                    state_merged[key] = (1.0 - lam) * state0[key] + lam * state1[key]
            merged.load_state_dict(state_merged)
            fuse_bn_stats(expert_mnist_cos, expert_fashion_cos, merged, lam.item())
            
        merged.eval()
        with torch.no_grad():
            out = merged(images)
            _, pred = out.max(1)
            acc = pred.eq(labels).sum().item() / labels.size(0) * 100.0
            method_accuracies["Method D"].append(acc)

    # Evaluate Method E: BK-AHR with Corrected Gradient Propagation
    print("--- Running Method E: BK-AHR (SOTA Baseline) ---")
    for b_idx, (phase, images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        images_pos = (images + 1.0) / 2.0
        images_denoised = torch.where(images_pos > 0.35, images_pos, torch.tensor(0.0, device=device))
        h_sparsity = compute_hoyer_sparsity(images_denoised)
        
        is_sparse = h_sparsity >= 0.50
        
        if is_sparse:
            exp0, exp1 = expert_mnist_std, expert_fashion_std
            p0, p1 = proto_mnist_std, proto_fashion_std
            eps_base = 0.08
        else:
            exp0, exp1 = expert_mnist_cos, expert_fashion_cos
            p0, p1 = proto_mnist_cos, proto_fashion_cos
            eps_base = 0.04
            
        feat0 = exp0(images, return_features=True)
        feat1 = exp1(images, return_features=True)
        if not is_sparse:
            feat0 = F.normalize(feat0, p=2, dim=1)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
        d0 = torch.mean(torch.stack([torch.min(torch.norm(f - p0, dim=1)) for f in feat0])).item()
        d1 = torch.mean(torch.stack([torch.min(torch.norm(f - p1, dim=1)) for f in feat1])).item()
        
        gap = abs(d0 - d1)
        
        with torch.no_grad():
            ent0 = entropy_loss_fn(exp0(images)).item()
            ent1 = entropy_loss_fn(exp1(images)).item()
            h_avg = (ent0 + ent1) / 2.0
            
        eps_stab = eps_base / (1.0 + 2.0 * h_avg)
        tau = gap / 3.0 + eps_stab
        
        w0 = np.exp(-d0 / tau)
        w1 = np.exp(-d1 / tau)
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        
        merged = clone_model(exp0)
        state_merged = merged.state_dict()
        state0 = exp0.state_dict()
        state1 = exp1.state_dict()
        for key in state_merged:
            if "weight" in key or "bias" in key:
                state_merged[key] = (1.0 - w1) * state0[key] + w1 * state1[key]
        merged.load_state_dict(state_merged)
        fuse_bn_stats(exp0, exp1, merged, w1)
        
        merged.zero_grad()
        out_init = merged(images)
        loss_init = entropy_loss_fn(out_init)
        loss_init.backward()
        
        F_sens = {}
        total_F = 0.0
        for name, param in merged.named_parameters():
            if param.requires_grad and param.grad is not None:
                F_sens[name] = torch.mean(param.grad ** 2).item()
                total_F += F_sens[name]
            else:
                F_sens[name] = 1e-5
                
        F_tilde = {name: F_sens[name] / (total_F + 1e-7) for name in F_sens}
        
        w_global = torch.tensor(np.log(w1 / (w0 + 1e-7)), requires_grad=True, device=device)
        offsets = {name: torch.zeros(1, requires_grad=True, device=device) for name, _ in merged.named_parameters() if _.requires_grad}
        
        for step in range(5):
            if w_global.grad is not None:
                w_global.grad.zero_()
            for name in offsets:
                if offsets[name].grad is not None:
                    offsets[name].grad.zero_()
                    
            lam_global = torch.sigmoid(w_global)
            params_dict = {}
            for name, param in merged.named_parameters():
                if name in offsets:
                    lam_j = torch.sigmoid(w_global + offsets[name])
                    params_dict[name] = (1.0 - lam_j) * state0[name] + lam_j * state1[name]
                    
            fuse_bn_stats(exp0, exp1, merged, lam_global.item())
            out = functional_call(merged, params_dict, (images,))
            
            l_ent = entropy_loss_fn(out)
            l_prior = 0.1 * (lam_global * torch.log(lam_global / w1) + (1.0 - lam_global) * torch.log((1.0 - lam_global) / w0))
            l_coherence = sum([0.05 * F_tilde[name] * torch.sum(offsets[name] ** 2) for name in offsets])
            total_loss = l_ent + l_prior + l_coherence
            total_loss.backward()
            
            with torch.no_grad():
                w_global -= 0.05 * w_global.grad
                for name in offsets:
                    precond_lr = 0.05 / (F_tilde[name] + 0.02)
                    offsets[name] -= precond_lr * offsets[name].grad
                    
        with torch.no_grad():
            lam_global = torch.sigmoid(w_global)
            state_m = merged.state_dict()
            for name in state_m:
                if name in offsets:
                    lam_j = torch.sigmoid(w_global + offsets[name])
                    state_m[name] = (1.0 - lam_j) * state0[name] + lam_j * state1[name]
            merged.load_state_dict(state_m)
            fuse_bn_stats(exp0, exp1, merged, lam_global.item())
            
        merged.eval()
        with torch.no_grad():
            out = merged(images)
            _, pred = out.max(1)
            acc = pred.eq(labels).sum().item() / labels.size(0) * 100.0
            method_accuracies["Method E"].append(acc)

    # Evaluate Method F: Ours - CG-SAM-TTMM with Corrected Gradient Propagation
    print("--- Running Method F: CG-SAM-TTMM (Ours) ---")
    for b_idx, (phase, images, labels) in enumerate(stream_batches):
        images, labels = images.to(device), labels.to(device)
        
        images_pos = (images + 1.0) / 2.0
        images_denoised = torch.where(images_pos > 0.35, images_pos, torch.tensor(0.0, device=device))
        h_sparsity = compute_hoyer_sparsity(images_denoised)
        
        is_sparse = h_sparsity >= 0.50
        
        if is_sparse:
            exp0, exp1 = expert_mnist_std, expert_fashion_std
            p0, p1 = proto_mnist_std, proto_fashion_std
            eps_base = 0.08
        else:
            exp0, exp1 = expert_mnist_cos, expert_fashion_cos
            p0, p1 = proto_mnist_cos, proto_fashion_cos
            eps_base = 0.04
            
        feat0 = exp0(images, return_features=True)
        feat1 = exp1(images, return_features=True)
        if not is_sparse:
            feat0 = F.normalize(feat0, p=2, dim=1)
            feat1 = F.normalize(feat1, p=2, dim=1)
            
        d0 = torch.mean(torch.stack([torch.min(torch.norm(f - p0, dim=1)) for f in feat0])).item()
        d1 = torch.mean(torch.stack([torch.min(torch.norm(f - p1, dim=1)) for f in feat1])).item()
        
        gap = abs(d0 - d1)
        
        with torch.no_grad():
            ent0 = entropy_loss_fn(exp0(images)).item()
            ent1 = entropy_loss_fn(exp1(images)).item()
            h_avg = (ent0 + ent1) / 2.0
            
        eps_stab = eps_base / (1.0 + 2.0 * h_avg)
        tau = gap / 3.0 + eps_stab
        
        w0 = np.exp(-d0 / tau)
        w1 = np.exp(-d1 / tau)
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        
        alpha_t = max(0.05, 1.0 - 0.5 * h_avg)
        eta_t = 0.05 * alpha_t
        rho_t = 0.02 * alpha_t
        
        merged = clone_model(exp0)
        state_merged = merged.state_dict()
        state0 = exp0.state_dict()
        state1 = exp1.state_dict()
        for key in state_merged:
            if "weight" in key or "bias" in key:
                state_merged[key] = (1.0 - w1) * state0[key] + w1 * state1[key]
        merged.load_state_dict(state_merged)
        fuse_bn_stats(exp0, exp1, merged, w1)
        
        merged.zero_grad()
        out_init = merged(images)
        loss_init = entropy_loss_fn(out_init)
        loss_init.backward()
        
        F_sens = {}
        total_F = 0.0
        for name, param in merged.named_parameters():
            if param.requires_grad and param.grad is not None:
                F_sens[name] = torch.mean(param.grad ** 2).item()
                total_F += F_sens[name]
            else:
                F_sens[name] = 1e-5
                
        F_tilde = {name: F_sens[name] / (total_F + 1e-7) for name in F_sens}
        
        w_global = torch.tensor(np.log(w1 / (w0 + 1e-7)), requires_grad=True, device=device)
        offsets = {name: torch.zeros(1, requires_grad=True, device=device) for name, _ in merged.named_parameters() if _.requires_grad}
        
        for step in range(5):
            if w_global.grad is not None:
                w_global.grad.zero_()
            for name in offsets:
                if offsets[name].grad is not None:
                    offsets[name].grad.zero_()
                    
            lam_global = torch.sigmoid(w_global)
            params_dict = {}
            for name, param in merged.named_parameters():
                if name in offsets:
                    lam_j = torch.sigmoid(w_global + offsets[name])
                    params_dict[name] = (1.0 - lam_j) * state0[name] + lam_j * state1[name]
                    
            fuse_bn_stats(exp0, exp1, merged, lam_global.item())
            out = functional_call(merged, params_dict, (images,))
            
            l_ent = entropy_loss_fn(out)
            l_prior = 0.1 * (lam_global * torch.log(lam_global / w1) + (1.0 - lam_global) * torch.log((1.0 - lam_global) / w0))
            l_coherence = sum([0.05 * F_tilde[name] * torch.sum(offsets[name] ** 2) for name in offsets])
            total_loss = l_ent + l_prior + l_coherence
            total_loss.backward()
            
            g_w = w_global.grad.clone()
            g_offsets = {name: offsets[name].grad.clone() for name in offsets}
            
            if rho_t > 0:
                d_w = g_w
                d_offsets = {name: g_offsets[name] / (F_tilde[name] + 0.02) for name in offsets}
                norm_sq = d_w.item()**2 + sum([torch.sum(d_offsets[name]**2).item() for name in offsets]) + 0.02
                norm_D = np.sqrt(norm_sq)
                
                eps_w = rho_t * d_w / norm_D
                eps_offsets = {name: rho_t * d_offsets[name] / norm_D for name in offsets}
                
                w_global_pert = w_global + eps_w
                offsets_pert = {name: offsets[name] + eps_offsets[name] for name in offsets}
                
                lam_global_pert = torch.sigmoid(w_global_pert)
                params_dict_pert = {}
                for name, param in merged.named_parameters():
                    if name in offsets:
                        lam_j = torch.sigmoid(w_global_pert + offsets_pert[name])
                        params_dict_pert[name] = (1.0 - lam_j) * state0[name] + lam_j * state1[name]
                
                fuse_bn_stats(exp0, exp1, merged, lam_global_pert.item())
                out_pert = functional_call(merged, params_dict_pert, (images,))
                
                l_ent_pert = entropy_loss_fn(out_pert)
                l_prior_pert = 0.1 * (lam_global_pert * torch.log(lam_global_pert / w1) + (1.0 - lam_global_pert) * torch.log((1.0 - lam_global_pert) / w0))
                l_coherence_pert = sum([0.05 * F_tilde[name] * torch.sum(offsets_pert[name] ** 2) for name in offsets])
                pert_loss = l_ent_pert + l_prior_pert + l_coherence_pert
                
                if w_global.grad is not None:
                    w_global.grad.zero_()
                for name in offsets:
                    if offsets[name].grad is not None:
                        offsets[name].grad.zero_()
                        
                pert_loss.backward()
                g_w_final = w_global.grad.clone()
                g_offsets_final = {name: offsets[name].grad.clone() for name in offsets}
            else:
                g_w_final = g_w
                g_offsets_final = g_offsets
                
            with torch.no_grad():
                w_global -= eta_t * g_w_final
                for name in offsets:
                    precond_lr = eta_t / (F_tilde[name] + 0.02)
                    offsets[name] -= precond_lr * g_offsets_final[name]
                    
        with torch.no_grad():
            lam_global = torch.sigmoid(w_global)
            state_m = merged.state_dict()
            for name in state_m:
                if name in offsets:
                    lam_j = torch.sigmoid(w_global + offsets[name])
                    state_m[name] = (1.0 - lam_j) * state0[name] + lam_j * state1[name]
            merged.load_state_dict(state_m)
            fuse_bn_stats(exp0, exp1, merged, lam_global.item())
            
        merged.eval()
        with torch.no_grad():
            out = merged(images)
            _, pred = out.max(1)
            acc = pred.eq(labels).sum().item() / labels.size(0) * 100.0
            method_accuracies["Method F (Ours)"].append(acc)

    # 5. Output Results Table and Generate Comparison Plot
    print("\n================ EVALUATION SUMMARY =================\n")
    print(f"{'Method':<35} | {'C-MNIST':<8} | {'N-MNIST':<8} | {'C-Fashion':<9} | {'N-Fashion':<9} | {'Novel-K':<8} | {'Overall':<8}")
    print("-" * 100)
    
    overall_means = {}
    for m in methods:
        accs = method_accuracies[m]
        c_mnist_acc = np.mean(accs[0:10])
        n_mnist_acc = np.mean(accs[10:20])
        c_fashion_acc = np.mean(accs[20:30])
        n_fashion_acc = np.mean(accs[30:40])
        novel_k_acc = np.mean(accs[40:50])
        overall_acc = np.mean(accs)
        
        overall_means[m] = overall_acc
        print(f"{m:<35} | {c_mnist_acc:6.2f}% | {n_mnist_acc:6.2f}% | {c_fashion_acc:7.2f}% | {n_fashion_acc:7.2f}% | {novel_k_acc:6.2f}% | {overall_acc:6.2f}%")
    print("=====================================================")
    
    # Save a comparison plot
    plt.figure(figsize=(10, 5))
    for m in methods:
        plt.plot(method_accuracies[m], label=m)
    plt.title("Streaming Test-Time Model Merging Accuracy")
    plt.xlabel("Streaming Batch Index")
    plt.ylabel("Batch Accuracy (%)")
    plt.axvline(x=10, color='gray', linestyle='--')
    plt.axvline(x=20, color='gray', linestyle='--')
    plt.axvline(x=30, color='gray', linestyle='--')
    plt.axvline(x=40, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("stream_accuracies.png")
    print("Saved stream_accuracies.png comparison plot.")
    
    print("\nCorrected Experiments complete!")

if __name__ == "__main__":
    main()
