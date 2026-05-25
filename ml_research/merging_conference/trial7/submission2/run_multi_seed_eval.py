import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import os

# 1. Model Definitions & Helpers
def modify_resnet18_for_grayscale(model):
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
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

class FeatureExtractorResNet18(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        self.features = nn.Sequential(*list(original_resnet.children())[:-1])
        self.fc = original_resnet.fc
        
    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        out = self.fc(feat)
        return out, feat

def create_merged_model(lambdas, experts, device):
    merged = models.resnet18()
    merged = modify_resnet18_for_grayscale(merged)
    merged.load_state_dict(experts[0].state_dict())
    merged = merged.to(device)
    
    merged_sd = merged.state_dict()
    expert_sds = [exp.state_dict() for exp in experts]
    
    for k in merged_sd.keys():
        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            continue
        tmp = torch.zeros_like(merged_sd[k]).float()
        for idx, lam in enumerate(lambdas):
            tmp += lam * expert_sds[idx][k].float()
        merged_sd[k].copy_(tmp)
        
    for k in merged_sd.keys():
        if 'running_mean' in k or 'running_var' in k:
            tmp = torch.zeros_like(merged_sd[k]).float()
            for idx, lam in enumerate(lambdas):
                tmp += lam * expert_sds[idx][k].float()
            merged_sd[k].copy_(tmp)
            
    merged.load_state_dict(merged_sd)
    return merged

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading datasets...")
    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_kmnist = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    # Load Experts
    print("Loading pre-trained experts...")
    expert_mnist = models.resnet18()
    expert_mnist = modify_resnet18_for_grayscale(expert_mnist)
    expert_mnist.load_state_dict(torch.load("checkpoints/expert_mnist.pth", map_location=device))
    expert_mnist = expert_mnist.to(device).eval()
    
    expert_kmnist = models.resnet18()
    expert_kmnist = modify_resnet18_for_grayscale(expert_kmnist)
    expert_kmnist.load_state_dict(torch.load("checkpoints/expert_kmnist.pth", map_location=device))
    expert_kmnist = expert_kmnist.to(device).eval()
    
    expert_fmnist = models.resnet18()
    expert_fmnist = modify_resnet18_for_grayscale(expert_fmnist)
    expert_fmnist.load_state_dict(torch.load("checkpoints/expert_fashionmnist.pth", map_location=device))
    expert_fmnist = expert_fmnist.to(device).eval()
    
    expert_mnist_fe = FeatureExtractorResNet18(expert_mnist).to(device).eval()
    expert_kmnist_fe = FeatureExtractorResNet18(expert_kmnist).to(device).eval()
    expert_fmnist_fe = FeatureExtractorResNet18(expert_fmnist).to(device).eval()
    
    experts_fe = [expert_mnist_fe, expert_kmnist_fe, expert_fmnist_fe]
    experts = [expert_mnist, expert_kmnist, expert_fmnist]
    
    centroids_mnist = expert_mnist.fc.weight.data.clone()
    centroids_kmnist = expert_kmnist.fc.weight.data.clone()
    
    seeds = [2026, 2027, 2028, 2029, 2030]
    methods = ["Uniform", "EBER", "Standard GMM", "Energy", "L-GMM"]
    
    # Initialize dictionary to store results for all seeds
    results = {m: {
        "MNIST": [], "KMNIST": [], "Fashion": [], "NDR": [], "FPR": []
    } for m in methods}
    
    batch_size = 64
    
    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Re-construct stream
        mnist_indices = np.random.choice(len(test_mnist), 30 * batch_size, replace=False)
        mnist_subset = Subset(test_mnist, mnist_indices)
        mnist_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
        
        kmnist_indices = np.random.choice(len(test_kmnist), 30 * batch_size, replace=False)
        kmnist_subset = Subset(test_kmnist, kmnist_indices)
        kmnist_loader = DataLoader(kmnist_subset, batch_size=batch_size, shuffle=False)
        
        fmnist_indices = np.random.choice(len(test_fmnist), 30 * batch_size, replace=False)
        fmnist_subset = Subset(test_fmnist, fmnist_indices)
        fmnist_loader = DataLoader(fmnist_subset, batch_size=batch_size, shuffle=False)
        
        stream_batches = []
        for x, y in mnist_loader:
            stream_batches.append((x, y, "MNIST"))
        for x, y in kmnist_loader:
            stream_batches.append((x, y, "KMNIST"))
        for x, y in fmnist_loader:
            stream_batches.append((x, y, "FashionMNIST"))
            
        # A. Static Uniform
        static_accs = []
        merged_static = create_merged_model([0.5, 0.5, 0.0], experts, device)
        merged_static.eval()
        for x, y, _ in stream_batches:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out = merged_static(x)
                _, pred = out.max(1)
                acc = pred.eq(y).sum().item() / y.size(0)
                static_accs.append(acc)
        results["Uniform"]["MNIST"].append(np.mean(static_accs[:30]) * 100)
        results["Uniform"]["KMNIST"].append(np.mean(static_accs[30:60]) * 100)
        results["Uniform"]["Fashion"].append(np.mean(static_accs[60:]) * 100)
        results["Uniform"]["NDR"].append(0.0) # Not applicable
        results["Uniform"]["FPR"].append(0.0) # Not applicable
        
        # B. EBER
        eber_accs = []
        for x, y, _ in stream_batches:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out_m = expert_mnist(x)
                out_k = expert_kmnist(x)
                ent_m = -torch.sum(torch.softmax(out_m, dim=1) * torch.log_softmax(out_m, dim=1), dim=1).mean().item()
                ent_k = -torch.sum(torch.softmax(out_k, dim=1) * torch.log_softmax(out_k, dim=1), dim=1).mean().item()
                out = out_m if ent_m < ent_k else out_k
                _, pred = out.max(1)
                acc = pred.eq(y).sum().item() / y.size(0)
                eber_accs.append(acc)
        results["EBER"]["MNIST"].append(np.mean(eber_accs[:30]) * 100)
        results["EBER"]["KMNIST"].append(np.mean(eber_accs[30:60]) * 100)
        results["EBER"]["Fashion"].append(np.mean(eber_accs[60:]) * 100)
        results["EBER"]["NDR"].append(0.0) # Not applicable
        results["EBER"]["FPR"].append(0.0) # Not applicable
        
        # C. Standard GMM
        sigma_sq_gmm = 1.0
        gmm_accs = []
        gmm_novelty = []
        with torch.no_grad():
            x_cal, _, _ = stream_batches[0]
            _, feat_cal = expert_mnist_fe(x_cal.to(device))
            dist_cal = torch.cdist(feat_cal, centroids_mnist)
            ll_cal = torch.logsumexp(-dist_cal**2 / (2 * sigma_sq_gmm), dim=1) - np.log(10)
            threshold_gmm = ll_cal.mean().item() - 0.8
            
        for x, y, _ in stream_batches:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, feat_m = expert_mnist_fe(x)
                _, feat_k = expert_kmnist_fe(x)
                dist_m = torch.cdist(feat_m, centroids_mnist)
                ll_m = (torch.logsumexp(-dist_m**2 / (2 * sigma_sq_gmm), dim=1) - np.log(10)).mean().item()
                dist_k = torch.cdist(feat_k, centroids_kmnist)
                ll_k = (torch.logsumexp(-dist_k**2 / (2 * sigma_sq_gmm), dim=1) - np.log(10)).mean().item()
                
                max_ll = max(ll_m, ll_k)
                if max_ll < threshold_gmm:
                    is_novel = True
                    lambdas = [0.0, 0.0, 1.0]
                else:
                    is_novel = False
                    lambdas = [1.0, 0.0, 0.0] if ll_m > ll_k else [0.0, 1.0, 0.0]
                    
                merged_model = create_merged_model(lambdas, experts, device)
                merged_model.eval()
                out = merged_model(x)
                _, pred = out.max(1)
                acc = pred.eq(y).sum().item() / y.size(0)
                gmm_accs.append(acc)
                gmm_novelty.append(is_novel)
                
        results["Standard GMM"]["MNIST"].append(np.mean(gmm_accs[:30]) * 100)
        results["Standard GMM"]["KMNIST"].append(np.mean(gmm_accs[30:60]) * 100)
        results["Standard GMM"]["Fashion"].append(np.mean(gmm_accs[60:]) * 100)
        gmm_fpr = sum(gmm_novelty[:60]) / 60 * 100
        gmm_ndr = sum(gmm_novelty[60:]) / 30 * 100
        results["Standard GMM"]["NDR"].append(gmm_ndr)
        results["Standard GMM"]["FPR"].append(gmm_fpr)
        
        # D. Energy Baseline
        energy_accs = []
        energy_novelty = []
        with torch.no_grad():
            x_cal, _, _ = stream_batches[0]
            out_cal = expert_mnist(x_cal.to(device))
            threshold_energy = torch.logsumexp(out_cal, dim=1).mean().item() - 4.0
            
        for x, y, _ in stream_batches:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out_m = expert_mnist(x)
                out_k = expert_kmnist(x)
                eng_m = torch.logsumexp(out_m, dim=1).mean().item()
                eng_k = torch.logsumexp(out_k, dim=1).mean().item()
                
                max_eng = max(eng_m, eng_k)
                if max_eng < threshold_energy:
                    is_novel = True
                    lambdas = [0.0, 0.0, 1.0]
                else:
                    is_novel = False
                    lambdas = [1.0, 0.0, 0.0] if eng_m > eng_k else [0.0, 1.0, 0.0]
                    
                merged_model = create_merged_model(lambdas, experts, device)
                merged_model.eval()
                out = merged_model(x)
                _, pred = out.max(1)
                acc = pred.eq(y).sum().item() / y.size(0)
                energy_accs.append(acc)
                energy_novelty.append(is_novel)
                
        results["Energy"]["MNIST"].append(np.mean(energy_accs[:30]) * 100)
        results["Energy"]["KMNIST"].append(np.mean(energy_accs[30:60]) * 100)
        results["Energy"]["Fashion"].append(np.mean(energy_accs[60:]) * 100)
        eng_fpr = sum(energy_novelty[:60]) / 60 * 100
        eng_ndr = sum(energy_novelty[60:]) / 30 * 100
        results["Energy"]["NDR"].append(eng_ndr)
        results["Energy"]["FPR"].append(eng_fpr)
        
        # E. L-GMM (Ours)
        sigma_sq = 0.1
        lgmm_accs = []
        lgmm_novelty = []
        with torch.no_grad():
            x_cal, _, _ = stream_batches[0]
            _, feat_cal = expert_mnist_fe(x_cal.to(device))
            feat_cal_norm = F.normalize(feat_cal, p=2, dim=1)
            centroids_mnist_norm = F.normalize(centroids_mnist, p=2, dim=1)
            dist_cal = torch.cdist(feat_cal_norm, centroids_mnist_norm)
            ll_cal = torch.logsumexp(-dist_cal**2 / (2 * sigma_sq), dim=1) - np.log(10)
            threshold = ll_cal.mean().item() - 0.8
            
        for x, y, _ in stream_batches:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, feat_m = expert_mnist_fe(x)
                _, feat_k = expert_kmnist_fe(x)
                
                feat_m_norm = F.normalize(feat_m, p=2, dim=1)
                feat_k_norm = F.normalize(feat_k, p=2, dim=1)
                centroids_mnist_norm = F.normalize(centroids_mnist, p=2, dim=1)
                centroids_kmnist_norm = F.normalize(centroids_kmnist, p=2, dim=1)
                
                dist_m = torch.cdist(feat_m_norm, centroids_mnist_norm)
                ll_m = (torch.logsumexp(-dist_m**2 / (2 * sigma_sq), dim=1) - np.log(10)).mean().item()
                dist_k = torch.cdist(feat_k_norm, centroids_kmnist_norm)
                ll_k = (torch.logsumexp(-dist_k**2 / (2 * sigma_sq), dim=1) - np.log(10)).mean().item()
                
                max_ll = max(ll_m, ll_k)
                if max_ll < threshold:
                    is_novel = True
                    lambdas = [0.0, 0.0, 1.0]
                else:
                    is_novel = False
                    lambdas = [1.0, 0.0, 0.0] if ll_m > ll_k else [0.0, 1.0, 0.0]
                    
                merged_model = create_merged_model(lambdas, experts, device)
                merged_model.eval()
                out = merged_model(x)
                _, pred = out.max(1)
                acc = pred.eq(y).sum().item() / y.size(0)
                lgmm_accs.append(acc)
                lgmm_novelty.append(is_novel)
                
        results["L-GMM"]["MNIST"].append(np.mean(lgmm_accs[:30]) * 100)
        results["L-GMM"]["KMNIST"].append(np.mean(lgmm_accs[30:60]) * 100)
        results["L-GMM"]["Fashion"].append(np.mean(lgmm_accs[60:]) * 100)
        lgmm_fpr = sum(lgmm_novelty[:60]) / 60 * 100
        lgmm_ndr = sum(lgmm_novelty[60:]) / 30 * 100
        results["L-GMM"]["NDR"].append(lgmm_ndr)
        results["L-GMM"]["FPR"].append(lgmm_fpr)
        
    print("\n================== FINAL MULTI-SEED STATISTICS ==================")
    stats = {m: {} for m in methods}
    for m in methods:
        print(f"\nMethod: {m}")
        for key in ["MNIST", "KMNIST", "Fashion", "NDR", "FPR"]:
            arr = np.array(results[m][key])
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            stats[m][key] = (mean_val, std_val)
            print(f"  {key:10s}: {mean_val:6.2f}% \u00B1 {std_val:5.2f}%")
            
    # Save the output report
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/multi_seed_results.txt", "w") as f:
        f.write("Multi-seed Evaluation Results (5 seeds: 2026-2030)\n")
        f.write("================================================\n\n")
        for m in methods:
            f.write(f"Method: {m}\n")
            for key in ["MNIST", "KMNIST", "Fashion", "NDR", "FPR"]:
                mean_val, std_val = stats[m][key]
                f.write(f"  {key:10s}: {mean_val:.2f}% \u00B1 {std_val:.2f}%\n")
            f.write("\n")
            
        f.write("\nLaTeX Table Rows:\n")
        f.write("=================\n")
        for m in methods:
            row_str = f"{m} "
            for key in ["MNIST", "KMNIST", "Fashion", "NDR", "FPR"]:
                mean_val, std_val = stats[m][key]
                if m in ["Uniform", "EBER"] and key in ["NDR", "FPR"]:
                    row_str += "& -- "
                else:
                    row_str += f"& {mean_val:.2f} \\scriptsize{{\\raisebox{{0.1ex}}{{\\tiny$\\pm$}}{std_val:.2f}}} "
            row_str += "\\\\\n"
            f.write(row_str)

if __name__ == "__main__":
    main()
