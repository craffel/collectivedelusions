import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

# Helper function to get subset of data
def get_subset(dataset, num_samples):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)

# Standard transformation: resize to 32x32, convert to 3 channels, convert to tensor and normalize
transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data():
    print("Loading data splits...")
    os.makedirs("./data", exist_ok=True)
    
    # Load raw datasets
    mnist_train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_gray)
    mnist_test_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_gray)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_color)
    cifar_test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_color)
    
    # Create required subsets:
    # - 2000 images for expert training
    # - 128 images for calibration/adaptation
    # - 1000 images for fast testing
    datasets = {
        "MNIST": {
            "train_expert": get_subset(mnist_train_full, 2000),
            "calib": get_subset(mnist_train_full, 128),
            "test": get_subset(mnist_test_full, 1000)
        },
        "Fashion-MNIST": {
            "train_expert": get_subset(fmnist_train_full, 2000),
            "calib": get_subset(fmnist_train_full, 128),
            "test": get_subset(fmnist_test_full, 1000)
        },
        "CIFAR-10": {
            "train_expert": get_subset(cifar_train_full, 2000),
            "calib": get_subset(cifar_train_full, 128),
            "test": get_subset(cifar_test_full, 1000)
        }
    }
    return datasets

# Model Definition
class CustomResNet18(nn.Module):
    def __init__(self, num_tasks=3, num_classes=10):
        super().__init__()
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() # Remove final fc
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.heads[task_id](features)

    def forward_features(self, x):
        # Extract activations at different depths
        activations = {}
        
        # conv1 + bn1 + relu + maxpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        activations['conv1'] = x
        
        # residual layers
        x = self.backbone.layer1(x)
        activations['layer1'] = x
        
        x = self.backbone.layer2(x)
        activations['layer2'] = x
        
        x = self.backbone.layer3(x)
        activations['layer3'] = x
        
        x = self.backbone.layer4(x)
        activations['layer4'] = x
        
        # Global average pool for feature representations
        for k in activations:
            # pool over spatial dimensions
            activations[k] = torch.mean(activations[k], dim=(2, 3))
            
        return activations

# CKA computation function (Memory and computation efficient)
def linear_cka(X, Y):
    # Center the matrices
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute covariances
    cov_XY = torch.matmul(X_c.t(), Y_c)
    cov_XX = torch.matmul(X_c.t(), X_c)
    cov_YY = torch.matmul(Y_c.t(), Y_c)
    
    # Compute norms
    norm_XY = torch.sum(cov_XY ** 2)
    norm_XX = torch.sum(cov_XX ** 2)
    norm_YY = torch.sum(cov_YY ** 2)
    
    # CKA
    cka = norm_XY / (torch.sqrt(norm_XX * norm_YY) + 1e-8)
    return cka.item()

# Train expert models
def train_experts(datasets):
    print("--- Training Experts ---")
    experts = []
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    
    # Check if we already have trained experts saved to save time
    if os.path.exists("./experts.pt"):
        print("Loading saved expert weights...")
        checkpoint = torch.load("./experts.pt", map_location=device)
        for i, task_name in enumerate(task_names):
            model = CustomResNet18().to(device)
            # Load states
            model.load_state_dict(checkpoint[f"expert_{i}"])
            experts.append(model)
        return experts
        
    for i, task_name in enumerate(task_names):
        print(f"Training Expert for {task_name}...")
        model = CustomResNet18().to(device)
        
        # Optimizer & Loss
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        
        # DataLoader
        train_loader = DataLoader(datasets[task_name]["train_expert"], batch_size=64, shuffle=True)
        
        model.train()
        # Train for 3 epochs
        for epoch in range(3):
            total_loss = 0
            correct = 0
            total = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs, i) # i is the task ID
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_acc = 100.0 * correct / total
            epoch_loss = total_loss / total
            print(f"Epoch {epoch+1}/3 - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
        # Quick eval
        test_loader = DataLoader(datasets[task_name]["test"], batch_size=128, shuffle=False)
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, i)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * test_correct / test_total
        print(f"Expert {task_name} Test Accuracy: {test_acc:.2f}%")
        experts.append(model)
        
    # Save experts
    checkpoint = {}
    for i, model in enumerate(experts):
        checkpoint[f"expert_{i}"] = model.state_dict()
    torch.save(checkpoint, "./experts.pt")
    print("Experts saved to experts.pt")
    
    return experts

def evaluate_model(model, datasets):
    model.eval()
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    accs = {}
    with torch.no_grad():
        for i, task_name in enumerate(task_names):
            test_loader = DataLoader(datasets[task_name]["test"], batch_size=128, shuffle=False)
            correct = 0
            total = 0
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, i)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            accs[task_name] = 100.0 * correct / total
    return accs

def get_base_backbone_state():
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    return base_model.state_dict()

def merge_models(experts, base_backbone_state, lambda_val=0.3):
    print("--- Merging Models (WA and TA) ---")
    
    # Weight Averaging
    wa_model = CustomResNet18().to(device)
    wa_state = wa_model.state_dict()
    
    # Task Arithmetic
    ta_model = CustomResNet18().to(device)
    ta_state = ta_model.state_dict()
    
    # 1. Merge task heads
    for i in range(3):
        expert_state = experts[i].state_dict()
        for k in wa_state.keys():
            if k.startswith(f"heads.{i}."):
                wa_state[k].copy_(expert_state[k])
                ta_state[k].copy_(expert_state[k])
                
    # 2. Merge backbones
    theta_base = base_backbone_state
    
    for k in wa_state.keys():
        if k.startswith("backbone."):
            # Average of experts for WA
            weights_exp = [experts[i].state_dict()[k] for i in range(3)]
            wa_state[k].copy_(sum(weights_exp) / 3.0)
            
            # Task Arithmetic
            base_w = theta_base[k.replace("backbone.", "")].to(device)
            task_vectors = [experts[i].state_dict()[k] - base_w for i in range(3)]
            ta_weight = base_w + lambda_val * sum(task_vectors)
            ta_state[k].copy_(ta_weight)
            
    wa_model.load_state_dict(wa_state)
    ta_model.load_state_dict(ta_state)
    
    return wa_model, ta_model

def ties_merge(experts, base_backbone_state, lambda_val=0.3, fraction=0.2):
    print(f"--- TIES Merging (lambda={lambda_val}, fraction={fraction}) ---")
    ties_model = CustomResNet18().to(device)
    ties_state = ties_model.state_dict()
    
    # Copy task heads
    for i in range(3):
        expert_state = experts[i].state_dict()
        for k in ties_state.keys():
            if k.startswith(f"heads.{i}."):
                ties_state[k].copy_(expert_state[k])
                
    theta_base = base_backbone_state
    
    for k in ties_state.keys():
        if k.startswith("backbone."):
            base_w = theta_base[k.replace("backbone.", "")].to(device)
            task_vectors = [experts[i].state_dict()[k] - base_w for i in range(3)]
            
            # Stack task vectors: shape [3, *param_shape]
            tvs = torch.stack(task_vectors, dim=0)
            
            # 1. Trim step (per task vector)
            trimmed_tvs_list = []
            for i in range(3):
                tv = tvs[i]
                abs_tv = torch.abs(tv)
                if abs_tv.numel() <= 1:
                    trimmed_tvs_list.append(tv)
                else:
                    k_val = int(abs_tv.numel() * (1.0 - fraction))
                    k_val = max(1, min(k_val, abs_tv.numel() - 1))
                    thresh = torch.kthvalue(abs_tv.view(-1), k_val).values
                    mask = abs_tv >= thresh
                    trimmed_tvs_list.append(tv * mask)
            
            trimmed_tvs = torch.stack(trimmed_tvs_list, dim=0)
            
            # 2. Elect Sign
            signs = torch.sign(trimmed_tvs)
            sum_signs = torch.sum(signs, dim=0)
            elected_sign = torch.sign(sum_signs)
            
            # 3. Disjoint Merge
            match_mask = (signs == elected_sign) & (signs != 0)
            matching_tvs = trimmed_tvs * match_mask
            counts = match_mask.sum(dim=0).float()
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            
            merged_tv = matching_tvs.sum(dim=0) / counts
            
            ties_weight = base_w + lambda_val * merged_tv
            ties_state[k].copy_(ties_weight)
            
    ties_model.load_state_dict(ties_state)
    return ties_model

def perform_head_only_sft(base_model, datasets, num_epochs=15, lr=1e-3):
    print("--- Training Head-only SFT (Corrected) ---")
    # Clone model
    sft_model = CustomResNet18().to(device)
    sft_model.load_state_dict(base_model.state_dict())
    
    # Freeze backbone
    for param in sft_model.backbone.parameters():
        param.requires_grad = False
        
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    criterion = nn.CrossEntropyLoss()
    
    for i, task_name in enumerate(task_names):
        print(f"SFT for head {task_name}...")
        calib_loader = DataLoader(datasets[task_name]["calib"], batch_size=32, shuffle=True)
        optimizer = optim.AdamW(sft_model.heads[i].parameters(), lr=lr, weight_decay=1e-2)
        
        for epoch in range(num_epochs):
            sft_model.backbone.eval()
            sft_model.heads[i].train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            for imgs, labels in calib_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = sft_model(imgs, i)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_acc = 100.0 * correct / total
            avg_loss = total_loss / total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
    return sft_model

# Layer-wise Scaling Calibration (LSC) Implementation
def register_lsc_hooks(model, scale_factors, task_id):
    hooks = []
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    for l in layers:
        scale = scale_factors[task_id][l]
        def make_hook(s):
            return lambda module, input, output: output * s
        
        if l == 'conv1':
            submod = model.backbone.conv1
        else:
            submod = getattr(model.backbone, l)
            
        handle = submod.register_forward_hook(make_hook(scale))
        hooks.append(handle)
    return hooks

def compute_lsc_scale_factors(experts, base_model, datasets):
    print("--- Computing LSC Scale Factors ---")
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    scale_factors = {t_idx: {l: 1.0 for l in layers} for t_idx in range(3)}
    
    for t_idx, t_name in enumerate(task_names):
        calib_loader = DataLoader(datasets[t_name]["calib"], batch_size=32, shuffle=False)
        expert = experts[t_idx]
        expert.eval()
        base_model.eval()
        
        expert_acts = {l: [] for l in layers}
        base_acts = {l: [] for l in layers}
        
        handles = []
        for l in layers:
            if l == 'conv1':
                sub_exp = expert.backbone.conv1
                sub_base = base_model.backbone.conv1
            else:
                sub_exp = getattr(expert.backbone, l)
                sub_base = getattr(base_model.backbone, l)
                
            def exp_hook_fn(layer_name):
                return lambda m, i, o: expert_acts[layer_name].append(torch.abs(o).mean().item())
            def base_hook_fn(layer_name):
                return lambda m, i, o: base_acts[layer_name].append(torch.abs(o).mean().item())
                
            handles.append(sub_exp.register_forward_hook(exp_hook_fn(l)))
            handles.append(sub_base.register_forward_hook(base_hook_fn(l)))
            
        with torch.no_grad():
            for imgs, _ in calib_loader:
                imgs = imgs.to(device)
                _ = expert(imgs, t_idx)
                _ = base_model(imgs, t_idx)
                
        for h in handles:
            h.remove()
            
        for l in layers:
            exp_mean = np.mean(expert_acts[l])
            base_mean = np.mean(base_acts[l])
            scale = exp_mean / (base_mean + 1e-8)
            scale = np.clip(scale, 0.1, 10.0)
            scale_factors[t_idx][l] = float(scale)
            print(f"Task {t_name}, Layer {l}: Expert Mean={exp_mean:.4f}, Merged Mean={base_mean:.4f}, Scale Factor={scale:.4f}")
            
    return scale_factors

def evaluate_lsc_model(model, datasets, scale_factors):
    model.eval()
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    accs = {}
    for i, task_name in enumerate(task_names):
        hooks = register_lsc_hooks(model, scale_factors, i)
        
        test_loader = DataLoader(datasets[task_name]["test"], batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, i)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accs[task_name] = 100.0 * correct / total
        
        for h in hooks:
            h.remove()
    return accs

# Custom Dataset wrapper for Out-of-Distribution (OOD) corruptions
class CorruptedDataset(Dataset):
    def __init__(self, subset, corruption_type, severity):
        self.subset = subset
        self.corruption_type = corruption_type
        self.severity = severity
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        
        if self.corruption_type == "gaussian_noise":
            corrupted_img = img + torch.randn_like(img) * self.severity
        elif self.corruption_type == "blur":
            kernel_size = int(self.severity * 4) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            corrupted_img = transforms.functional.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=self.severity)
        elif self.corruption_type == "brightness":
            corrupted_img = img * self.severity
        else:
            corrupted_img = img
            
        return corrupted_img, label

def evaluate_model_ood(model, datasets, corruption_type, severity, scale_factors=None):
    model.eval()
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    accs = {}
    
    for i, task_name in enumerate(task_names):
        hooks = []
        if scale_factors is not None:
            hooks = register_lsc_hooks(model, scale_factors, i)
            
        test_subset = datasets[task_name]["test"]
        corrupted_test_set = CorruptedDataset(test_subset, corruption_type, severity)
        test_loader = DataLoader(corrupted_test_set, batch_size=128, shuffle=False)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, i)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accs[task_name] = 100.0 * correct / total
        
        for h in hooks:
            h.remove()
            
    return accs

# Core Analysis: CKA vs. Linear Probe Accuracy
def analyze_representations(experts, ta_model, ties_model, sft_model, datasets):
    print("--- Representation Analysis (CKA vs Linear Probing) ---")
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    results = {}
    
    for task_idx, task_name in enumerate(task_names):
        print(f"\nAnalyzing Task: {task_name}")
        results[task_name] = {}
        
        train_subset = datasets[task_name]["train_expert"]
        test_subset = datasets[task_name]["test"]
        
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
        
        expert_model = experts[task_idx]
        expert_model.eval()
        ta_model.eval()
        ties_model.eval()
        sft_model.eval()
        
        def extract_features(model, loader):
            feat_dict = {l: [] for l in layers}
            labels_list = []
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    feats = model.forward_features(imgs)
                    for l in layers:
                        feat_dict[l].append(feats[l].cpu())
                    labels_list.append(labels)
            for l in layers:
                feat_dict[l] = torch.cat(feat_dict[l], dim=0)
            labels_all = torch.cat(labels_list, dim=0)
            return feat_dict, labels_all
            
        print("Extracting features from Expert backbone...")
        expert_train_feats, train_labels = extract_features(expert_model, train_loader)
        expert_test_feats, test_labels = extract_features(expert_model, test_loader)
        
        print("Extracting features from Merged (TA) backbone...")
        ta_train_feats, _ = extract_features(ta_model, train_loader)
        ta_test_feats, _ = extract_features(ta_model, test_loader)
        
        print("Extracting features from Merged (TIES) backbone...")
        ties_train_feats, _ = extract_features(ties_model, train_loader)
        ties_test_feats, _ = extract_features(ties_model, test_loader)
        
        cka_scores = {}
        ties_cka_scores = {}
        for l in layers:
            score = linear_cka(expert_test_feats[l].to(device), ta_test_feats[l].to(device))
            cka_scores[l] = score
            
            ties_score = linear_cka(expert_test_feats[l].to(device), ties_test_feats[l].to(device))
            ties_cka_scores[l] = ties_score
            print(f"Layer {l} CKA Similarity: (TA)={score:.4f}, (TIES)={ties_score:.4f}")
            
        print("Training Linear Probes on Expert Features...")
        expert_probe_accs = {}
        for l in layers:
            acc = train_and_eval_linear_probe(
                expert_train_feats[l], train_labels, 
                expert_test_feats[l], test_labels
            )
            expert_probe_accs[l] = acc
            print(f"  Expert Probe Accuracy at {l}: {acc:.2f}%")
            
        print("Training Linear Probes on Merged (TA) Features...")
        ta_probe_accs = {}
        for l in layers:
            acc = train_and_eval_linear_probe(
                ta_train_feats[l], train_labels, 
                ta_test_feats[l], test_labels
            )
            ta_probe_accs[l] = acc
            print(f"  Merged (TA) Probe Accuracy at {l}: {acc:.2f}%")
            
        print("Training Linear Probes on Merged (TIES) Features...")
        ties_probe_accs = {}
        for l in layers:
            acc = train_and_eval_linear_probe(
                ties_train_feats[l], train_labels, 
                ties_test_feats[l], test_labels
            )
            ties_probe_accs[l] = acc
            print(f"  Merged (TIES) Probe Accuracy at {l}: {acc:.2f}%")
            
        results[task_name] = {
            "cka_scores": cka_scores,
            "ties_cka_scores": ties_cka_scores,
            "expert_probe_accs": expert_probe_accs,
            "ta_probe_accs": ta_probe_accs,
            "ties_probe_accs": ties_probe_accs
        }
        
    return results

def train_and_eval_linear_probe(train_feats, train_labels, test_feats, test_labels, epochs=15, lr=1e-2):
    num_features = train_feats.shape[1]
    probe = nn.Linear(num_features, 10).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    probe.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = probe(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    probe.eval()
    with torch.no_grad():
        test_x = test_feats.to(device)
        test_y = test_labels.to(device)
        out = probe(test_x)
        _, preds = out.max(1)
        correct = preds.eq(test_y).sum().item()
        acc = 100.0 * correct / len(test_y)
    return acc

def generate_visualizations(results):
    print("--- Generating Visualizations ---")
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        cka = [results[task_name]["cka_scores"][l] for l in layers]
        ties_cka = [results[task_name]["ties_cka_scores"][l] for l in layers]
        exp_acc = [results[task_name]["expert_probe_accs"][l] / 100.0 for l in layers]
        ta_acc = [results[task_name]["ta_probe_accs"][l] / 100.0 for l in layers]
        ties_acc = [results[task_name]["ties_probe_accs"][l] / 100.0 for l in layers]
        
        color = 'tab:blue'
        ax.set_xlabel('Layer Depth', fontsize=12)
        ax.set_ylabel('CKA Representational Similarity', color=color, fontsize=12)
        line1 = ax.plot(layers, cka, marker='o', linestyle='-', color=color, label='Linear CKA (Expert vs TA)', linewidth=2.5)
        line1_ties = ax.plot(layers, ties_cka, marker='d', linestyle=':', color='tab:cyan', label='Linear CKA (Expert vs TIES)', linewidth=2.0)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Linear Probe Classification Accuracy', color=color, fontsize=12)
        line2 = ax2.plot(layers, exp_acc, marker='s', linestyle='--', color='tab:green', label='Expert Probe (Upper Bound)', linewidth=2)
        line3 = ax2.plot(layers, ta_acc, marker='^', linestyle='-.', color=color, label='Merged Probe (TA)', linewidth=2)
        line3_ties = ax2.plot(layers, ties_acc, marker='v', linestyle=':', color='tab:red', label='Merged Probe (TIES)', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.05, 1.05)
        
        lines = line1 + line1_ties + line2 + line3 + line3_ties
        labels = [l.get_label() for l in lines]
        if idx == 0:
            ax.legend(lines, labels, loc='lower left', fontsize=8)
            
        ax.set_title(f"{task_name} Representation Analysis", fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("representation_analysis.png", dpi=300)
    print("Saved plot to representation_analysis.png")
    
    plt.figure(figsize=(8, 6))
    avg_cka = []
    avg_ties_cka = []
    avg_exp_probe = []
    avg_ta_probe = []
    avg_ties_probe = []
    for l in layers:
        avg_cka.append(np.mean([results[t]["cka_scores"][l] for t in task_names]))
        avg_ties_cka.append(np.mean([results[t]["ties_cka_scores"][l] for t in task_names]))
        avg_exp_probe.append(np.mean([results[t]["expert_probe_accs"][l] for t in task_names]))
        avg_ta_probe.append(np.mean([results[t]["ta_probe_accs"][l] for t in task_names]))
        avg_ties_probe.append(np.mean([results[t]["ties_probe_accs"][l] for t in task_names]))
        
    plt.plot(layers, avg_cka, marker='o', color='tab:blue', label='Average CKA (TA)', linewidth=3)
    plt.plot(layers, avg_ties_cka, marker='d', color='tab:cyan', label='Average CKA (TIES)', linewidth=2)
    plt.plot(layers, [acc/100.0 for acc in avg_exp_probe], marker='s', linestyle='--', color='tab:green', label='Average Expert Probe Accuracy', linewidth=2)
    plt.plot(layers, [acc/100.0 for acc in avg_ta_probe], marker='^', linestyle='-.', color='tab:orange', label='Average Merged Probe Accuracy (TA)', linewidth=3)
    plt.plot(layers, [acc/100.0 for acc in avg_ties_probe], marker='v', linestyle=':', color='tab:red', label='Average Merged Probe Accuracy (TIES)', linewidth=2)
    
    plt.title("Deconstructing CKA Deception (Average Across All Tasks)", fontsize=14, fontweight='bold')
    plt.xlabel("Layer Depth", fontsize=12)
    plt.ylabel("Value (Similarity/Accuracy Ratio)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig("cka_deception_expose.png", dpi=300)
    print("Saved second plot to cka_deception_expose.png")

def main():
    datasets = load_data()
    
    # 1. Train or load experts
    experts = train_experts(datasets)
    
    # Evaluate experts
    print("\n--- Evaluating Expert Models (Sanity Check) ---")
    expert_accs_dict = {}
    for i, name in enumerate(["MNIST", "Fashion-MNIST", "CIFAR-10"]):
        accs = evaluate_model(experts[i], datasets)
        expert_accs_dict[name] = accs[name]
        print(f"Expert {name} evaluated on:")
        for t_name, acc in accs.items():
            print(f"  {t_name}: {acc:.2f}%")
            
    # 2. Get base backbone
    base_backbone = get_base_backbone_state()
    
    # 3. Merge models
    wa_model, ta_model = merge_models(experts, base_backbone, lambda_val=0.3)
    ties_model = ties_merge(experts, base_backbone, lambda_val=0.3, fraction=0.2)
    
    # Evaluate uncalibrated merged models
    print("\n--- Evaluating Uncalibrated Merged Models ---")
    wa_accs = evaluate_model(wa_model, datasets)
    ta_accs = evaluate_model(ta_model, datasets)
    ties_accs = evaluate_model(ties_model, datasets)
    
    print("Weight Averaging Accuracies:")
    for t_name, acc in wa_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average WA Accuracy: {np.mean(list(wa_accs.values())):.2f}%")
        
    print("Task Arithmetic (lambda=0.3) Accuracies:")
    for t_name, acc in ta_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average TA Accuracy: {np.mean(list(ta_accs.values())):.2f}%")
    
    print("TIES Merging (lambda=0.3) Accuracies:")
    for t_name, acc in ties_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average TIES Accuracy: {np.mean(list(ties_accs.values())):.2f}%")
    
    # 4. Perform Head-only SFT (Corrected)
    print("\n--- Performing Corrected Head-only SFT on TA ---")
    sft_model = perform_head_only_sft(ta_model, datasets, num_epochs=15, lr=1e-3)
    sft_accs = evaluate_model(sft_model, datasets)
    print("Corrected Head-only SFT (TA) Accuracies:")
    for t_name, acc in sft_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average SFT (TA) Accuracy: {np.mean(list(sft_accs.values())):.2f}%")
    
    print("\n--- Performing Corrected Head-only SFT on TIES ---")
    ties_sft_model = perform_head_only_sft(ties_model, datasets, num_epochs=15, lr=1e-3)
    ties_sft_accs = evaluate_model(ties_sft_model, datasets)
    print("Corrected Head-only SFT (TIES) Accuracies:")
    for t_name, acc in ties_sft_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average SFT (TIES) Accuracy: {np.mean(list(ties_sft_accs.values())):.2f}%")
    
    # 5. Compute LSC Scale Factors and Evaluate LSC-Calibrated Models
    print("\n--- LSC Calibration for TA ---")
    scale_factors = compute_lsc_scale_factors(experts, ta_model, datasets)
    lsc_accs = evaluate_lsc_model(ta_model, datasets, scale_factors)
    print("LSC-Calibrated Task Arithmetic Model Accuracies:")
    for t_name, acc in lsc_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average LSC (TA) Accuracy: {np.mean(list(lsc_accs.values())):.2f}%")
    
    print("\n--- LSC Calibration for TIES ---")
    ties_scale_factors = compute_lsc_scale_factors(experts, ties_model, datasets)
    ties_lsc_accs = evaluate_lsc_model(ties_model, datasets, ties_scale_factors)
    print("LSC-Calibrated TIES Model Accuracies:")
    for t_name, acc in ties_lsc_accs.items():
        print(f"  {t_name}: {acc:.2f}%")
    print(f"  Average LSC (TIES) Accuracy: {np.mean(list(ties_lsc_accs.values())):.2f}%")
    
    # 6. Out-of-Distribution (OOD) Evaluation
    print("\n--- Performing Out-of-Distribution (OOD) Evaluation ---")
    ood_results = {}
    
    corruptions = {
        "gaussian_noise": 0.3,
        "blur": 1.5,
        "brightness": 0.5
    }
    
    models_to_eval = {
        "Expert": (None, None),
        "WA": (wa_model, None),
        "TA": (ta_model, None),
        "SFT": (sft_model, None),
        "LSC": (ta_model, scale_factors),
        "TIES": (ties_model, None),
        "TIES_SFT": (ties_sft_model, None),
        "TIES_LSC": (ties_model, ties_scale_factors)
    }
    
    for c_name, severity in corruptions.items():
        print(f"\nEvaluating under corruption: {c_name} (severity={severity})")
        ood_results[c_name] = {}
        
        for m_name, (m_obj, s_factors) in models_to_eval.items():
            if m_name == "Expert":
                task_accs = {}
                for idx, t_name in enumerate(["MNIST", "Fashion-MNIST", "CIFAR-10"]):
                    test_subset = datasets[t_name]["test"]
                    corrupted_test_set = CorruptedDataset(test_subset, c_name, severity)
                    loader = DataLoader(corrupted_test_set, batch_size=128, shuffle=False)
                    
                    correct = 0
                    total = 0
                    experts[idx].eval()
                    with torch.no_grad():
                        for imgs, labels in loader:
                            imgs, labels = imgs.to(device), labels.to(device)
                            outputs = experts[idx](imgs, idx)
                            _, predicted = outputs.max(1)
                            total += labels.size(0)
                            correct += predicted.eq(labels).sum().item()
                    task_accs[t_name] = 100.0 * correct / total
                ood_results[c_name]["Expert"] = task_accs
            else:
                task_accs = evaluate_model_ood(m_obj, datasets, c_name, severity, s_factors)
                ood_results[c_name][m_name] = task_accs
                
            avg_acc = np.mean(list(task_accs.values()))
            print(f"  Model {m_name}: MNIST={task_accs['MNIST']:.2f}%, Fashion-MNIST={task_accs['Fashion-MNIST']:.2f}%, CIFAR-10={task_accs['CIFAR-10']:.2f}%, Average={avg_acc:.2f}%")
            
    # 7. Core Representation Analysis (CKA vs. Linear Probing)
    results = analyze_representations(experts, ta_model, ties_model, sft_model, datasets)
    
    # Save results to JSON
    with open("representation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to representation_results.json")
    
    # 8. Generate visualizations
    generate_visualizations(results)
    
    summary = {
        "expert_upper_bounds": expert_accs_dict,
        "wa_accuracies": wa_accs,
        "ta_accuracies": ta_accs,
        "ties_accuracies": ties_accs,
        "sft_accuracies": sft_accs,
        "ties_sft_accuracies": ties_sft_accs,
        "lsc_accuracies": lsc_accs,
        "ties_lsc_accuracies": ties_lsc_accs,
        "ood_results": ood_results,
        "representation_analysis": results
    }
    with open("experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
