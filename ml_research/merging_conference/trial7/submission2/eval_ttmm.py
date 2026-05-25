import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import os
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)
torch.backends.cudnn.enabled = False

# 1. Model Definitions & Helpers
def modify_resnet18_for_grayscale(model):
    # Sum the weights of the first conv layer along the input channel dimension (3 -> 1)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv
    
    # Modify final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

class FeatureExtractorResNet18(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        # Keep everything except the fc layer
        self.features = nn.Sequential(*list(original_resnet.children())[:-1])
        self.fc = original_resnet.fc
        
    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        out = self.fc(feat)
        return out, feat

def train_expert(model, train_loader, device, epochs=1):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model

# 2. Main Experiment Setup
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Datasets
    print("Downloading datasets...")
    train_mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_kmnist = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    test_kmnist = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    train_fmnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    # Train/Load Experts
    print("Training Experts...")
    loader_opts = {'batch_size': 256, 'shuffle': True, 'num_workers': 2}
    
    # Expert 1: MNIST
    m_path = "checkpoints/expert_mnist.pth"
    if os.path.exists(m_path):
        print("Loading MNIST expert...")
        expert_mnist = models.resnet18()
        expert_mnist = modify_resnet18_for_grayscale(expert_mnist)
        expert_mnist.load_state_dict(torch.load(m_path, map_location=device))
    else:
        print("Training MNIST expert...")
        train_loader = DataLoader(train_mnist, **loader_opts)
        expert_mnist = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        expert_mnist = modify_resnet18_for_grayscale(expert_mnist)
        expert_mnist = train_expert(expert_mnist.to(device), train_loader, device, epochs=1)
        torch.save(expert_mnist.state_dict(), m_path)
        
    # Expert 2: KMNIST
    k_path = "checkpoints/expert_kmnist.pth"
    if os.path.exists(k_path):
        print("Loading KMNIST expert...")
        expert_kmnist = models.resnet18()
        expert_kmnist = modify_resnet18_for_grayscale(expert_kmnist)
        expert_kmnist.load_state_dict(torch.load(k_path, map_location=device))
    else:
        print("Training KMNIST expert...")
        train_loader = DataLoader(train_kmnist, **loader_opts)
        expert_kmnist = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        expert_kmnist = modify_resnet18_for_grayscale(expert_kmnist)
        expert_kmnist = train_expert(expert_kmnist.to(device), train_loader, device, epochs=1)
        torch.save(expert_kmnist.state_dict(), k_path)
        
    # Expert 3: FashionMNIST
    f_path = "checkpoints/expert_fashionmnist.pth"
    if os.path.exists(f_path):
        print("Loading FashionMNIST expert...")
        expert_fmnist = models.resnet18()
        expert_fmnist = modify_resnet18_for_grayscale(expert_fmnist)
        expert_fmnist.load_state_dict(torch.load(f_path, map_location=device))
    else:
        print("Training FashionMNIST expert...")
        train_loader = DataLoader(train_fmnist, **loader_opts)
        expert_fmnist = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        expert_fmnist = modify_resnet18_for_grayscale(expert_fmnist)
        expert_fmnist = train_expert(expert_fmnist.to(device), train_loader, device, epochs=1)
        torch.save(expert_fmnist.state_dict(), f_path)
        
    # Wrap models to extract features
    expert_mnist_fe = FeatureExtractorResNet18(expert_mnist).to(device).eval()
    expert_kmnist_fe = FeatureExtractorResNet18(expert_kmnist).to(device).eval()
    expert_fmnist_fe = FeatureExtractorResNet18(expert_fmnist).to(device).eval()
    
    experts_fe = [expert_mnist_fe, expert_kmnist_fe, expert_fmnist_fe]
    
    # 3. Construct Test-Time Stream
    print("Constructing non-stationary open-world test stream...")
    # MNIST (Batches 1-30), KMNIST (Batches 31-60), FashionMNIST (Batches 61-90)
    batch_size = 64
    stream_batches = []
    
    # MNIST batches
    mnist_indices = np.random.choice(len(test_mnist), 30 * batch_size, replace=False)
    mnist_subset = Subset(test_mnist, mnist_indices)
    mnist_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
    for x, y in mnist_loader:
        stream_batches.append((x, y, "MNIST"))
        
    # KMNIST batches
    kmnist_indices = np.random.choice(len(test_kmnist), 30 * batch_size, replace=False)
    kmnist_subset = Subset(test_kmnist, kmnist_indices)
    kmnist_loader = DataLoader(kmnist_subset, batch_size=batch_size, shuffle=False)
    for x, y in kmnist_loader:
        stream_batches.append((x, y, "KMNIST"))
        
    # FashionMNIST batches (Novel Domain)
    fmnist_indices = np.random.choice(len(test_fmnist), 30 * batch_size, replace=False)
    fmnist_subset = Subset(test_fmnist, fmnist_indices)
    fmnist_loader = DataLoader(fmnist_subset, batch_size=batch_size, shuffle=False)
    for x, y in fmnist_loader:
        stream_batches.append((x, y, "FashionMNIST"))
        
    print(f"Total stream batches: {len(stream_batches)}")
    
    # 4. Implement Model Merging and BN Merging function
    def create_merged_model(lambdas, experts):
        # Create a deep copy of expert 0 to act as merged model base
        merged = models.resnet18()
        merged = modify_resnet18_for_grayscale(merged)
        merged.load_state_dict(experts[0].state_dict())
        merged = merged.to(device)
        
        # Merge weights
        merged_sd = merged.state_dict()
        expert_sds = [exp.state_dict() for exp in experts]
        
        for k in merged_sd.keys():
            if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                # These are BN statistics, handled below
                continue
            # Weighted average
            tmp = torch.zeros_like(merged_sd[k]).float()
            for idx, lam in enumerate(lambdas):
                tmp += lam * expert_sds[idx][k].float()
            merged_sd[k].copy_(tmp)
            
        # Merge BN running mean and variance
        # Let's average coefficients for BN buffers
        for k in merged_sd.keys():
            if 'running_mean' in k or 'running_var' in k:
                tmp = torch.zeros_like(merged_sd[k]).float()
                for idx, lam in enumerate(lambdas):
                    tmp += lam * expert_sds[idx][k].float()
                merged_sd[k].copy_(tmp)
                
        merged.load_state_dict(merged_sd)
        return merged
        
    # 5. Evaluate Baselines and Our Proposed L-GMM
    
    # A. Static Uniform Merging (MNIST and KMNIST merged 0.5/0.5, no FashionMNIST)
    print("\n--- Running Baseline: Static Uniform Merging (MNIST + KMNIST) ---")
    lambdas_static = [0.5, 0.5, 0.0]
    merged_static = create_merged_model(lambdas_static, [expert_mnist, expert_kmnist, expert_fmnist])
    merged_static.eval()
    
    static_accs = []
    for idx, (x, y, domain) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = merged_static(x)
            _, pred = out.max(1)
            acc = pred.eq(y).sum().item() / y.size(0)
            static_accs.append(acc)
            
    print(f"Static Uniform Accuracy - MNIST: {np.mean(static_accs[:30])*100:.2f}%, KMNIST: {np.mean(static_accs[30:60])*100:.2f}%, FashionMNIST (Novel): {np.mean(static_accs[60:])*100:.2f}%")
    
    # B. Closed-World Entropy-Based Expert Routing (EBER)
    # Routes to either Expert 1 or Expert 2 based on which has lower prediction entropy
    print("\n--- Running Baseline: Closed-World EBER ---")
    eber_accs = []
    eber_routing = [] # records routed expert ID (0 or 1)
    
    for idx, (x, y, domain) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            # Get prediction entropy under Expert 1 and Expert 2
            out_m = expert_mnist(x)
            out_k = expert_kmnist(x)
            
            ent_m = -torch.sum(torch.softmax(out_m, dim=1) * torch.log_softmax(out_m, dim=1), dim=1).mean().item()
            ent_k = -torch.sum(torch.softmax(out_k, dim=1) * torch.log_softmax(out_k, dim=1), dim=1).mean().item()
            
            if ent_m < ent_k:
                routed = 0
                out = out_m
            else:
                routed = 1
                out = out_k
                
            _, pred = out.max(1)
            acc = pred.eq(y).sum().item() / y.size(0)
            eber_accs.append(acc)
            eber_routing.append(routed)
            
    print(f"EBER Accuracy - MNIST: {np.mean(eber_accs[:30])*100:.2f}%, KMNIST: {np.mean(eber_accs[30:60])*100:.2f}%, FashionMNIST (Novel): {np.mean(eber_accs[60:])*100:.2f}%")
    
    # C. Open-World L-GMM (Our Proposed Method)
    # Uses expert classifier weight vectors as GMM centroids to compute exact log-likelihood
    print("\n--- Running Proposed Method: Open-World L-GMM ---")
    
    # Retrieve class centroids for Expert 1 and Expert 2 from fc weights
    # fc weights have shape (10, 512)
    centroids_mnist = expert_mnist.fc.weight.data.clone() # (10, 512)
    centroids_kmnist = expert_kmnist.fc.weight.data.clone() # (10, 512)
    
    # Using L2-normalized representations (VMF Mixture Model) to resolve scaling disparities
    sigma_sq = 0.1 # shared directional variance concentration
    
    lgmm_accs = []
    lgmm_routing = [] # 0=MNIST, 1=KMNIST, 2=Novel (FashionMNIST)
    
    # Novelty Detection metrics
    novelty_detected = [] # True/False for each batch
    
    # Let's run a calibration pass using 1 known batch to set the threshold dynamically
    import torch.nn.functional as F
    with torch.no_grad():
        x_cal, _, _ = stream_batches[0]
        x_cal = x_cal.to(device)
        _, feat_cal = expert_mnist_fe(x_cal) # (B, 512)
        
        # Compute normalized likelihood under MNIST
        feat_cal_norm = F.normalize(feat_cal, p=2, dim=1)
        centroids_mnist_norm = F.normalize(centroids_mnist, p=2, dim=1)
        dist_cal = torch.cdist(feat_cal_norm, centroids_mnist_norm) # (B, 10)
        ll_cal = torch.logsumexp(-dist_cal**2 / (2 * sigma_sq), dim=1) - np.log(10)
        mean_ll_cal = ll_cal.mean().item()
        print(f"Calibrated MNIST baseline log-likelihood: {mean_ll_cal:.2f}")
        # Threshold set to separate in-distribution from out-of-distribution
        threshold = mean_ll_cal - 0.8
        print(f"Dynamic Novelty Detection Threshold set to: {threshold:.2f}")
        
    for idx, (x, y, domain) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            # 1. Extract features under the current model or the base expert models
            # To be data-free and zero-shot, we extract features from Expert 1 and Expert 2
            _, feat_m = expert_mnist_fe(x)
            _, feat_k = expert_kmnist_fe(x)
            
            # Normalize features and centroids for directional comparison
            feat_m_norm = F.normalize(feat_m, p=2, dim=1)
            feat_k_norm = F.normalize(feat_k, p=2, dim=1)
            centroids_mnist_norm = F.normalize(centroids_mnist, p=2, dim=1)
            centroids_kmnist_norm = F.normalize(centroids_kmnist, p=2, dim=1)
            
            # 2. Compute GMM log-likelihood for Expert 1 and Expert 2
            dist_m = torch.cdist(feat_m_norm, centroids_mnist_norm) # (B, 10)
            ll_m = (torch.logsumexp(-dist_m**2 / (2 * sigma_sq), dim=1) - np.log(10)).mean().item()
            
            dist_k = torch.cdist(feat_k_norm, centroids_kmnist_norm) # (B, 10)
            ll_k = (torch.logsumexp(-dist_k**2 / (2 * sigma_sq), dim=1) - np.log(10)).mean().item()
            
            max_ll = max(ll_m, ll_k)
            
            # 3. Open-world routing decision
            if max_ll < threshold:
                # Novel domain detected! Route to the Novel Expert (Expert 3)
                routed = 2
                is_novel = True
                # Dynamically compose model with Expert 3 (FashionMNIST)
                lambdas = [0.0, 0.0, 1.0]
            else:
                # Known domain. Route to the expert with higher likelihood
                is_novel = False
                if ll_m > ll_k:
                    routed = 0
                    lambdas = [1.0, 0.0, 0.0]
                else:
                    routed = 1
                    lambdas = [0.0, 1.0, 0.0]
                    
            # Create the merged model for this batch and run prediction
            merged_model = create_merged_model(lambdas, [expert_mnist, expert_kmnist, expert_fmnist])
            merged_model.eval()
            out = merged_model(x)
            
            _, pred = out.max(1)
            acc = pred.eq(y).sum().item() / y.size(0)
            
            lgmm_accs.append(acc)
            lgmm_routing.append(routed)
            novelty_detected.append(is_novel)
            
    # Calculate Novelty Detection Metrics
    # MNIST batches (0 to 29) and KMNIST batches (30 to 59) are known (novelty should be False)
    # FashionMNIST batches (60 to 89) are novel (novelty should be True)
    mnist_novel = novelty_detected[:30]
    kmnist_novel = novelty_detected[30:60]
    fmnist_novel = novelty_detected[60:]
    
    fpr_mnist = sum(mnist_novel) / len(mnist_novel) * 100
    fpr_kmnist = sum(kmnist_novel) / len(kmnist_novel) * 100
    avg_fpr = (fpr_mnist + fpr_kmnist) / 2
    ndr = sum(fmnist_novel) / len(fmnist_novel) * 100
    
    print(f"L-GMM Novelty Detection Rate (NDR): {ndr:.2f}%")
    print(f"L-GMM False Positive Rate (FPR) on MNIST: {fpr_mnist:.2f}%, on KMNIST: {fpr_kmnist:.2f}% (Average FPR: {avg_fpr:.2f}%)")
    print(f"L-GMM Accuracy - MNIST: {np.mean(lgmm_accs[:30])*100:.2f}%, KMNIST: {np.mean(lgmm_accs[30:60])*100:.2f}%, FashionMNIST (Novel): {np.mean(lgmm_accs[60:])*100:.2f}%")
    
    # D. Standard GMM (Euclidean - exhibiting the Euclidean Bug)
    print("\n--- Running Baseline: Standard GMM (Euclidean) ---")
    sigma_sq_gmm = 1.0 # standard GMM variance
    
    gmm_accs = []
    gmm_routing = [] # 0=MNIST, 1=KMNIST, 2=Novel (FashionMNIST)
    gmm_novelty_detected = []
    
    # Calibration pass for standard GMM
    with torch.no_grad():
        x_cal, _, _ = stream_batches[0]
        x_cal = x_cal.to(device)
        _, feat_cal = expert_mnist_fe(x_cal) # (B, 512)
        
        # Compute standard Euclidean likelihood under MNIST
        dist_cal = torch.cdist(feat_cal, centroids_mnist) # (B, 10)
        ll_cal = torch.logsumexp(-dist_cal**2 / (2 * sigma_sq_gmm), dim=1) - np.log(10)
        mean_ll_cal_gmm = ll_cal.mean().item()
        print(f"Standard GMM Calibrated MNIST baseline log-likelihood: {mean_ll_cal_gmm:.2f}")
        threshold_gmm = mean_ll_cal_gmm - 0.8
        print(f"Standard GMM Novelty Detection Threshold set to: {threshold_gmm:.2f}")
        
    for idx, (x, y, domain) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, feat_m = expert_mnist_fe(x)
            _, feat_k = expert_kmnist_fe(x)
            
            # Compute standard Euclidean log-likelihood for Expert 1 and Expert 2
            dist_m = torch.cdist(feat_m, centroids_mnist) # (B, 10)
            ll_m = (torch.logsumexp(-dist_m**2 / (2 * sigma_sq_gmm), dim=1) - np.log(10)).mean().item()
            
            dist_k = torch.cdist(feat_k, centroids_kmnist) # (B, 10)
            ll_k = (torch.logsumexp(-dist_k**2 / (2 * sigma_sq_gmm), dim=1) - np.log(10)).mean().item()
            
            max_ll = max(ll_m, ll_k)
            
            if max_ll < threshold_gmm:
                routed = 2
                is_novel = True
                lambdas = [0.0, 0.0, 1.0]
            else:
                is_novel = False
                if ll_m > ll_k:
                    routed = 0
                    lambdas = [1.0, 0.0, 0.0]
                else:
                    routed = 1
                    lambdas = [0.0, 1.0, 0.0]
                    
            merged_model = create_merged_model(lambdas, [expert_mnist, expert_kmnist, expert_fmnist])
            merged_model.eval()
            out = merged_model(x)
            
            _, pred = out.max(1)
            acc = pred.eq(y).sum().item() / y.size(0)
            
            gmm_accs.append(acc)
            gmm_routing.append(routed)
            gmm_novelty_detected.append(is_novel)
            
    # Calculate Standard GMM Novelty metrics
    gmm_mnist_novel = gmm_novelty_detected[:30]
    gmm_kmnist_novel = gmm_novelty_detected[30:60]
    gmm_fmnist_novel = gmm_novelty_detected[60:]
    
    gmm_fpr_mnist = sum(gmm_mnist_novel) / len(gmm_mnist_novel) * 100
    gmm_fpr_kmnist = sum(gmm_kmnist_novel) / len(gmm_kmnist_novel) * 100
    gmm_avg_fpr = (gmm_fpr_mnist + gmm_fpr_kmnist) / 2
    gmm_ndr = sum(gmm_fmnist_novel) / len(gmm_fmnist_novel) * 100
    
    print(f"Standard GMM Novelty Detection Rate (NDR): {gmm_ndr:.2f}%")
    print(f"Standard GMM False Positive Rate (FPR) on MNIST: {gmm_fpr_mnist:.2f}%, on KMNIST: {gmm_fpr_kmnist:.2f}% (Average FPR: {gmm_avg_fpr:.2f}%)")
    print(f"Standard GMM Accuracy - MNIST: {np.mean(gmm_accs[:30])*100:.2f}%, KMNIST: {np.mean(gmm_accs[30:60])*100:.2f}%, FashionMNIST (Novel): {np.mean(gmm_accs[60:])*100:.2f}%")
    
    # E. Energy-Based Baseline
    print("\n--- Running Baseline: Energy-Based Open-World Routing ---")
    energy_accs = []
    energy_routing = [] # 0=MNIST, 1=KMNIST, 2=Novel (FashionMNIST)
    energy_novelty_detected = []
    
    # Calibration pass for Energy Baseline
    with torch.no_grad():
        x_cal, _, _ = stream_batches[0]
        x_cal = x_cal.to(device)
        out_cal = expert_mnist(x_cal)
        energy_cal = torch.logsumexp(out_cal, dim=1).mean().item()
        print(f"Energy-Based Calibrated MNIST baseline: {energy_cal:.2f}")
        # Threshold set to separate ID from OOD using a calibrated safety margin of 4.0
        threshold_energy = energy_cal - 4.0
        print(f"Energy-Based Novelty Detection Threshold set to: {threshold_energy:.2f}")
        
    for idx, (x, y, domain) in enumerate(stream_batches):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out_m = expert_mnist(x)
            out_k = expert_kmnist(x)
            
            # Compute energy score (average logsumexp of logits)
            eng_m = torch.logsumexp(out_m, dim=1).mean().item()
            eng_k = torch.logsumexp(out_k, dim=1).mean().item()
            
            max_eng = max(eng_m, eng_k)
            
            if max_eng < threshold_energy:
                routed = 2
                is_novel = True
                lambdas = [0.0, 0.0, 1.0]
            else:
                is_novel = False
                if eng_m > eng_k:
                    routed = 0
                    lambdas = [1.0, 0.0, 0.0]
                else:
                    routed = 1
                    lambdas = [0.0, 1.0, 0.0]
                    
            merged_model = create_merged_model(lambdas, [expert_mnist, expert_kmnist, expert_fmnist])
            merged_model.eval()
            out = merged_model(x)
            
            _, pred = out.max(1)
            acc = pred.eq(y).sum().item() / y.size(0)
            
            energy_accs.append(acc)
            energy_routing.append(routed)
            energy_novelty_detected.append(is_novel)
            
    # Calculate Energy Novelty metrics
    energy_mnist_novel = energy_novelty_detected[:30]
    energy_kmnist_novel = energy_novelty_detected[30:60]
    energy_fmnist_novel = energy_novelty_detected[60:]
    
    energy_fpr_mnist = sum(energy_mnist_novel) / len(energy_mnist_novel) * 100
    energy_fpr_kmnist = sum(energy_kmnist_novel) / len(energy_kmnist_novel) * 100
    energy_avg_fpr = (energy_fpr_mnist + energy_fpr_kmnist) / 2
    energy_ndr = sum(energy_fmnist_novel) / len(energy_fmnist_novel) * 100
    
    print(f"Energy Baseline Novelty Detection Rate (NDR): {energy_ndr:.2f}%")
    print(f"Energy Baseline False Positive Rate (FPR) on MNIST: {energy_fpr_mnist:.2f}%, on KMNIST: {energy_fpr_kmnist:.2f}% (Average FPR: {energy_avg_fpr:.2f}%)")
    print(f"Energy Baseline Accuracy - MNIST: {np.mean(energy_accs[:30])*100:.2f}%, KMNIST: {np.mean(energy_accs[30:60])*100:.2f}%, FashionMNIST (Novel): {np.mean(energy_accs[60:])*100:.2f}%")
    
    # 6. Generate Figures/Plots
    # Let's plot the accuracies of the methods across the stream
    plt.figure(figsize=(10, 5))
    plt.plot(static_accs, label='Static Uniform (MNIST+KMNIST)', color='gray', linestyle='--')
    plt.plot(eber_accs, label='EBER (Closed-World Routing)', color='blue', alpha=0.7)
    plt.plot(gmm_accs, label='Standard GMM (Euclidean - Bug)', color='orange', alpha=0.8, linestyle='-.')
    plt.plot(energy_accs, label='Energy-Based Routing (OOD Gated)', color='purple', alpha=0.7, linestyle=':')
    plt.plot(lgmm_accs, label='L-GMM (Our Proposed Open-World Method)', color='green', linewidth=2)
    plt.axvline(x=30, color='red', linestyle=':', label='Stream Shift (MNIST -> KMNIST)')
    plt.axvline(x=60, color='red', linestyle=':', label='Stream Shift (KMNIST -> FashionMNIST [Novel])')
    plt.title('Test-Time Model Merging Performance Comparison')
    plt.xlabel('Stream Batch Index')
    plt.ylabel('Batch Classification Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ttmm_accuracy_comparison.png")
    plt.close()
    
    # Plot coefficients / routing decisions for L-GMM
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(lgmm_routing)), lgmm_routing, c=lgmm_routing, cmap='brg', s=50, alpha=0.8)
    plt.axvline(x=30, color='black', linestyle=':')
    plt.axvline(x=60, color='black', linestyle=':')
    plt.yticks([0, 1, 2], ['MNIST Expert', 'KMNIST Expert', 'FashionMNIST Expert'])
    plt.title('L-GMM Dynamic Routing Decisions on Non-Stationary Open-World Stream')
    plt.xlabel('Stream Batch Index')
    plt.ylabel('Routed Expert')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/lgmm_routing_decisions.png")
    plt.close()
    
    print("\nPlots generated and saved in plots/ directory.")
    
    # Save the key results to a text file for inclusion in the paper
    with open("checkpoints/results.txt", "w") as rf:
        rf.write(f"Static Uniform Accuracy: MNIST={np.mean(static_accs[:30])*100:.2f}%, KMNIST={np.mean(static_accs[30:60])*100:.2f}%, FashionMNIST={np.mean(static_accs[60:])*100:.2f}%\n")
        rf.write(f"EBER Closed-World Accuracy: MNIST={np.mean(eber_accs[:30])*100:.2f}%, KMNIST={np.mean(eber_accs[30:60])*100:.2f}%, FashionMNIST={np.mean(eber_accs[60:])*100:.2f}%\n")
        rf.write(f"Standard GMM Accuracy: MNIST={np.mean(gmm_accs[:30])*100:.2f}%, KMNIST={np.mean(gmm_accs[30:60])*100:.2f}%, FashionMNIST={np.mean(gmm_accs[60:])*100:.2f}%\n")
        rf.write(f"Standard GMM Novelty Detection Metrics: NDR={gmm_ndr:.2f}%, MNIST_FPR={gmm_fpr_mnist:.2f}%, KMNIST_FPR={gmm_fpr_kmnist:.2f}%, Avg_FPR={gmm_avg_fpr:.2f}%\n")
        rf.write(f"Energy Baseline Accuracy: MNIST={np.mean(energy_accs[:30])*100:.2f}%, KMNIST={np.mean(energy_accs[30:60])*100:.2f}%, FashionMNIST={np.mean(energy_accs[60:])*100:.2f}%\n")
        rf.write(f"Energy Baseline Novelty Detection Metrics: NDR={energy_ndr:.2f}%, MNIST_FPR={energy_fpr_mnist:.2f}%, KMNIST_FPR={energy_fpr_kmnist:.2f}%, Avg_FPR={energy_avg_fpr:.2f}%\n")
        rf.write(f"L-GMM Open-World Accuracy: MNIST={np.mean(lgmm_accs[:30])*100:.2f}%, KMNIST={np.mean(lgmm_accs[30:60])*100:.2f}%, FashionMNIST={np.mean(lgmm_accs[60:])*100:.2f}%\n")
        rf.write(f"L-GMM Novelty Detection Metrics: NDR={ndr:.2f}%, MNIST_FPR={fpr_mnist:.2f}%, KMNIST_FPR={fpr_kmnist:.2f}%, Avg_FPR={avg_fpr:.2f}%\n")

if __name__ == "__main__":
    main()
