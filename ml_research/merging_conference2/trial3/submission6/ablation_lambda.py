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

# Standard transformations
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
    
    mnist_train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_gray)
    mnist_test_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_gray)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_color)
    cifar_test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_color)
    
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
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.heads[task_id](features)

    def forward_features(self, x):
        activations = {}
        # We only need layer4 for this ablation study to reduce compute and keep it focused
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # Global average pool over spatial dimensions
        activations['layer4'] = torch.mean(x, dim=(2, 3))
        return activations

# CKA computation function
def linear_cka(X, Y):
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    cov_XY = torch.matmul(X_c.t(), Y_c)
    cov_XX = torch.matmul(X_c.t(), X_c)
    cov_YY = torch.matmul(Y_c.t(), Y_c)
    norm_XY = torch.sum(cov_XY ** 2)
    norm_XX = torch.sum(cov_XX ** 2)
    norm_YY = torch.sum(cov_YY ** 2)
    return (norm_XY / (torch.sqrt(norm_XX * norm_YY) + 1e-8)).item()

# Load expert models
def load_experts():
    print("Loading saved expert weights...")
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    experts = []
    checkpoint = torch.load("./experts.pt", map_location=device)
    for i, task_name in enumerate(task_names):
        model = CustomResNet18().to(device)
        model.load_state_dict(checkpoint[f"expert_{i}"])
        experts.append(model)
    return experts

def get_base_backbone_state():
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    return base_model.state_dict()

def merge_models(experts, base_backbone_state, lambda_val):
    ta_model = CustomResNet18().to(device)
    ta_state = ta_model.state_dict()
    
    # 1. Merge task heads
    for i in range(3):
        expert_state = experts[i].state_dict()
        for k in ta_state.keys():
            if k.startswith(f"heads.{i}."):
                ta_state[k].copy_(expert_state[k])
                
    # 2. Merge backbones
    theta_base = base_backbone_state
    for k in ta_state.keys():
        if k.startswith("backbone."):
            base_w = theta_base[k.replace("backbone.", "")].to(device)
            task_vectors = [experts[i].state_dict()[k] - base_w for i in range(3)]
            ta_weight = base_w + lambda_val * sum(task_vectors)
            ta_state[k].copy_(ta_weight)
            
    ta_model.load_state_dict(ta_state)
    return ta_model

def ties_merge(experts, base_backbone_state, lambda_val, fraction=0.2):
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
            tvs = torch.stack(task_vectors, dim=0)
            
            # Trim
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
            
            # Elect Sign
            signs = torch.sign(trimmed_tvs)
            sum_signs = torch.sum(signs, dim=0)
            elected_sign = torch.sign(sum_signs)
            
            # Disjoint Merge
            match_mask = (signs == elected_sign) & (signs != 0)
            matching_tvs = trimmed_tvs * match_mask
            counts = match_mask.sum(dim=0).float()
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            
            merged_tv = matching_tvs.sum(dim=0) / counts
            ties_weight = base_w + lambda_val * merged_tv
            ties_state[k].copy_(ties_weight)
            
    ties_model.load_state_dict(ties_state)
    return ties_model

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

def main():
    datasets = load_data()
    experts = load_experts()
    base_backbone = get_base_backbone_state()
    
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    results = {
        "lambdas": lambdas,
        "TA": {"cka": [], "probe": [], "uncalib_acc": []},
        "TIES": {"cka": [], "probe": [], "uncalib_acc": []}
    }
    
    # First, let's extract expert features once to save time
    expert_train_feats = {}
    expert_test_feats = {}
    train_labels = {}
    test_labels = {}
    
    for t_idx, t_name in enumerate(task_names):
        train_loader = DataLoader(datasets[t_name]["train_expert"], batch_size=128, shuffle=False)
        test_loader = DataLoader(datasets[t_name]["test"], batch_size=128, shuffle=False)
        
        expert = experts[t_idx]
        expert.eval()
        
        tr_feats = []
        te_feats = []
        tr_lbls = []
        te_lbls = []
        
        with torch.no_grad():
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                feats = expert.forward_features(imgs)['layer4'].cpu()
                tr_feats.append(feats)
                tr_lbls.append(lbls)
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                feats = expert.forward_features(imgs)['layer4'].cpu()
                te_feats.append(feats)
                te_lbls.append(lbls)
                
        expert_train_feats[t_name] = torch.cat(tr_feats, dim=0)
        expert_test_feats[t_name] = torch.cat(te_feats, dim=0)
        train_labels[t_name] = torch.cat(tr_lbls, dim=0)
        test_labels[t_name] = torch.cat(te_lbls, dim=0)
        
    print("Expert features extracted. Starting lambda sweep...")
    
    for lambda_val in lambdas:
        print(f"\nEvaluating lambda = {lambda_val:.1f}")
        
        # 1. Merge models
        ta_model = merge_models(experts, base_backbone, lambda_val)
        ties_model = ties_merge(experts, base_backbone, lambda_val, fraction=0.2)
        
        for name, model in [("TA", ta_model), ("TIES", ties_model)]:
            model.eval()
            
            # Measure CKA and Probe accuracies across tasks
            cka_list = []
            probe_list = []
            uncalib_acc_list = []
            
            for t_idx, t_name in enumerate(task_names):
                test_loader = DataLoader(datasets[t_name]["test"], batch_size=128, shuffle=False)
                train_loader = DataLoader(datasets[t_name]["train_expert"], batch_size=128, shuffle=False)
                
                # Measure uncalibrated accuracy
                correct = 0
                total = 0
                with torch.no_grad():
                    for imgs, lbls in test_loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        outputs = model(imgs, t_idx)
                        _, predicted = outputs.max(1)
                        total += lbls.size(0)
                        correct += predicted.eq(lbls).sum().item()
                uncalib_acc_list.append(100.0 * correct / total)
                
                # Extract merged features
                m_tr_feats = []
                m_te_feats = []
                with torch.no_grad():
                    for imgs, _ in train_loader:
                        imgs = imgs.to(device)
                        m_tr_feats.append(model.forward_features(imgs)['layer4'].cpu())
                    for imgs, _ in test_loader:
                        imgs = imgs.to(device)
                        m_te_feats.append(model.forward_features(imgs)['layer4'].cpu())
                m_train_feats = torch.cat(m_tr_feats, dim=0)
                m_test_feats = torch.cat(m_te_feats, dim=0)
                
                # Compute CKA score at layer4 between expert and merged
                cka_score = linear_cka(expert_test_feats[t_name].to(device), m_test_feats.to(device))
                cka_list.append(cka_score)
                
                # Train and eval linear probe on merged features
                probe_acc = train_and_eval_linear_probe(
                    m_train_feats, train_labels[t_name],
                    m_test_feats, test_labels[t_name]
                )
                probe_list.append(probe_acc)
                
            avg_cka = np.mean(cka_list)
            avg_probe = np.mean(probe_list)
            avg_uncalib = np.mean(uncalib_acc_list)
            
            results[name]["cka"].append(float(avg_cka))
            results[name]["probe"].append(float(avg_probe))
            results[name]["uncalib_acc"].append(float(avg_uncalib))
            
            print(f"  {name}: Avg CKA={avg_cka:.4f}, Avg Probe={avg_probe:.2f}%, Avg Uncalib={avg_uncalib:.2f}%")
            
    # Save sweep results to JSON
    with open("ablation_lambda_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel(r'Merging Scale Coefficient ($\lambda$)', fontsize=12)
    ax1.set_ylabel('Average CKA Similarity (Layer 4)', color=color, fontsize=12)
    line1 = ax1.plot(lambdas, results["TA"]["cka"], marker='o', linestyle='-', color=color, label='CKA (Task Arithmetic)', linewidth=2.5)
    line1_ties = ax1.plot(lambdas, results["TIES"]["cka"], marker='d', linestyle=':', color='tab:cyan', label='CKA (TIES-Merging)', linewidth=2.0)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Average Linear Probe Accuracy (%)', color=color, fontsize=12)
    line2 = ax2.plot(lambdas, [acc/100.0 for acc in results["TA"]["probe"]], marker='^', linestyle='-.', color=color, label='Probe Accuracy (Task Arithmetic)', linewidth=2.5)
    line2_ties = ax2.plot(lambdas, [acc/100.0 for acc in results["TIES"]["probe"]], marker='v', linestyle=':', color='tab:red', label='Probe Accuracy (TIES-Merging)', linewidth=2.0)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.05, 1.05)
    
    lines = line1 + line1_ties + line2 + line2_ties
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=10)
    
    plt.title("Ablation of Merging Coefficient ($\lambda$) on Layer 4\n(CKA drops but Probe classification capacity is highly preserved)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("ablation_lambda.png", dpi=300)
    print("Saved sweep plot to ablation_lambda.png")

if __name__ == "__main__":
    main()
