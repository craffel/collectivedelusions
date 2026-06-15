import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# --- RESNET-18 FOUNDATION FEATURE EXTRACTION ---
print("Loading pre-trained ResNet-18 model as foundation feature extractor...")
# Load pre-trained ResNet-18 and strip the classification head
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval()

# Freeze resnet parameters
for p in resnet.parameters():
    p.requires_grad = False

print("Loading image datasets (MNIST and FashionMNIST)...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
fashion_train = dsets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
fashion_test = dsets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

def get_subset_loader(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        sampler=torch.utils.data.SubsetRandomSampler(subset_indices)
    )
    return loader

print("Extracting high-dimensional foundation features...")
def extract_features(loader):
    features_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in loader:
            # Replicate grayscale channel to 3 channels for ResNet-18
            imgs_3ch = imgs.repeat(1, 3, 1, 1)
            feats = resnet(imgs_3ch)
            features_list.append(feats)
            labels_list.append(labels)
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

# Extract 1000 train samples per task, 500 test samples per task
mnist_train_loader = get_subset_loader(mnist_train, 1000)
fashion_train_loader = get_subset_loader(fashion_train, 1000)
mnist_test_loader = get_subset_loader(mnist_test, 500)
fashion_test_loader = get_subset_loader(fashion_test, 500)

mnist_train_x, mnist_train_y = extract_features(mnist_train_loader)
fashion_train_x, fashion_train_y = extract_features(fashion_train_loader)
mnist_test_x, mnist_test_y = extract_features(mnist_test_loader)
fashion_test_x, fashion_test_y = extract_features(fashion_test_loader)

# Combine test sets
test_x = torch.cat([mnist_test_x, fashion_test_x], dim=0)
test_y = torch.cat([mnist_test_y, fashion_test_y], dim=0)
test_task = torch.cat([torch.zeros(mnist_test_x.shape[0], dtype=torch.long), 
                        torch.ones(fashion_test_x.shape[0], dtype=torch.long)], dim=0)

print(f"Extraction completed:")
print(f"  Feature dimensionality: {mnist_train_x.shape[1]}")
print(f"  MNIST Train: {mnist_train_x.shape[0]} features, Test: {mnist_test_x.shape[0]} features")
print(f"  FashionMNIST Train: {fashion_train_x.shape[0]} features, Test: {fashion_test_x.shape[0]} features")

# --- MULTI-LAYER CLASSIFIER ARCHITECTURE ---
class MultiLayerClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out, h

# Train shared base model on a balanced mixture of both tasks
print("\nTraining shared base classifier model on multitask mixture...")
base_model = MultiLayerClassifier()
optimizer_base = optim.AdamW(base_model.parameters(), lr=1e-3, weight_decay=1e-3)

mix_train_x = torch.cat([mnist_train_x[:500], fashion_train_x[:500]], dim=0)
mix_train_y = torch.cat([mnist_train_y[:500], fashion_train_y[:500]], dim=0)

base_model.train()
for epoch in range(15):
    optimizer_base.zero_grad()
    logits, _ = base_model(mix_train_x)
    loss = F.cross_entropy(logits, mix_train_y)
    loss.backward()
    optimizer_base.step()
base_model.eval()

# Train specialized experts starting from base model
K = 2  # MNIST, FashionMNIST
expert_models = []

print("Training specialized MNIST Expert...")
mnist_expert = MultiLayerClassifier()
mnist_expert.load_state_dict(base_model.state_dict())
optimizer_m = optim.AdamW(mnist_expert.parameters(), lr=2e-3, weight_decay=1e-3)
mnist_expert.train()
for epoch in range(12):
    optimizer_m.zero_grad()
    logits, _ = mnist_expert(mnist_train_x)
    loss = F.cross_entropy(logits, mnist_train_y)
    loss.backward()
    optimizer_m.step()
mnist_expert.eval()
expert_models.append(mnist_expert)

print("Training specialized FashionMNIST Expert...")
fashion_expert = MultiLayerClassifier()
fashion_expert.load_state_dict(base_model.state_dict())
optimizer_f = optim.AdamW(fashion_expert.parameters(), lr=2e-3, weight_decay=1e-3)
fashion_expert.train()
for epoch in range(12):
    optimizer_f.zero_grad()
    logits, _ = fashion_expert(fashion_train_x)
    loss = F.cross_entropy(logits, fashion_train_y)
    loss.backward()
    optimizer_f.step()
fashion_expert.eval()
expert_models.append(fashion_expert)

# Measure standalone expert accuracy
with torch.no_grad():
    m_logits, _ = mnist_expert(mnist_test_x)
    m_acc = (m_logits.argmax(dim=-1) == mnist_test_y).float().mean().item() * 100.0
    
    f_logits, _ = fashion_expert(fashion_test_x)
    f_acc = (f_logits.argmax(dim=-1) == fashion_test_y).float().mean().item() * 100.0

expert_ceilings = [m_acc, f_acc]
print(f"Standalone Expert Accuracies:")
print(f"  MNIST Expert on MNIST: {m_acc:.2f}%")
print(f"  FashionMNIST Expert on FashionMNIST: {f_acc:.2f}%")
print(f"  Expert Ceiling Joint Mean: {np.mean(expert_ceilings):.2f}%")

# Helper accuracy metrics
def get_acc(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item() * 100.0

def evaluate_confounded_accuracy(logits, targets_mnist, targets_fashion):
    # Highly rigorous joint classification metric: Top-2 predictions must retrieve BOTH correct targets
    top2_preds = logits.topk(2, dim=-1).indices # Shape (B, 2)
    success_mnist = (top2_preds == targets_mnist.unsqueeze(-1)).any(dim=-1)
    success_fashion = (top2_preds == targets_fashion.unsqueeze(-1)).any(dim=-1)
    
    same_mask = (targets_mnist == targets_fashion)
    success_same = (logits.argmax(dim=-1) == targets_mnist) & same_mask
    success_diff = success_mnist & success_fashion & (~same_mask)
    
    success = success_same | success_diff
    return success.float().mean().item() * 100.0

# --- ROUTER CENTROIDS ---
# 1. Support-16 Centroids: Average penultimate features of 16 support samples per task
with torch.no_grad():
    _, h_m = base_model(mnist_train_x[:16])
    _, h_f = base_model(fashion_train_x[:16])
centroids_support = torch.stack([h_m.mean(dim=0), h_f.mean(dim=0)], dim=0) # Shape (K, 128)
centroids_support_norm = centroids_support / (centroids_support.norm(dim=-1, keepdim=True) + 1e-12)

# 2. Naive Weight-averaging Zero-Data Centroids: average weights row-wise
w_m_naive = expert_models[0].fc2.weight.data.mean(dim=0)
w_f_naive = expert_models[1].fc2.weight.data.mean(dim=0)
centroids_naive = torch.stack([w_m_naive, w_f_naive], dim=0) # Shape (K, 128)
centroids_naive_norm = centroids_naive / (centroids_naive.norm(dim=-1, keepdim=True) + 1e-12)

# 3. REFINED Weight-averaging Zero-Data Centroids: L2-normalize class vectors BEFORE row-wise averaging
# This resolves vector cancellation and scale mismatches!
w_m_ref = expert_models[0].fc2.weight.data
w_m_ref_norm = w_m_ref / (w_m_ref.norm(dim=-1, keepdim=True) + 1e-12)
w_m_centroid = w_m_ref_norm.mean(dim=0)

w_f_ref = expert_models[1].fc2.weight.data
w_f_ref_norm = w_f_ref / (w_f_ref.norm(dim=-1, keepdim=True) + 1e-12)
w_f_centroid = w_f_ref_norm.mean(dim=0)

centroids_refined = torch.stack([w_m_centroid, w_f_centroid], dim=0) # Shape (K, 128)
centroids_refined_norm = centroids_refined / (centroids_refined.norm(dim=-1, keepdim=True) + 1e-12)

def compute_routing_coefficients(h_features, tau=0.05, centroid_type="support"):
    h_norm = h_features / (h_features.norm(dim=-1, keepdim=True) + 1e-12)
    if centroid_type == "support":
        c_norm = centroids_support_norm
    elif centroid_type == "naive":
        c_norm = centroids_naive_norm
    elif centroid_type == "refined":
        c_norm = centroids_refined_norm
    else:
        raise ValueError("Invalid centroid_type")
        
    sims = torch.matmul(h_norm, c_norm.t()) # Shape (B, K)
    coeffs = torch.softmax(sims / tau, dim=-1)
    return coeffs

# --- LORA SVD DECOMPOSITION ---
W1_base = base_model.fc1.weight.data
b1_base = base_model.fc1.bias.data
W2_base = base_model.fc2.weight.data
b2_base = base_model.fc2.bias.data

# Decompose experts for each rank in {2, 4, 8, 16}
ranks = [2, 4, 8, 16]
decompositions = {}

for r in ranks:
    decompositions[r] = {
        "fc1": {"A": [], "B": [], "db": []},
        "fc2": {"A": [], "B": [], "db": []}
    }
    for k in range(K):
        # Layer 1
        W1_exp = expert_models[k].fc1.weight.data
        b1_exp = expert_models[k].fc1.bias.data
        V1_k = W1_exp - W1_base
        db1_k = b1_exp - b1_base
        
        U1, S1, Vh1 = torch.linalg.svd(V1_k, full_matrices=False)
        U1r = U1[:, :r]
        S1r = torch.diag(torch.sqrt(S1[:r]))
        Vh1r = Vh1[:r, :]
        
        A1 = torch.matmul(U1r, S1r)
        B1 = torch.matmul(S1r, Vh1r)
        
        decompositions[r]["fc1"]["A"].append(A1)
        decompositions[r]["fc1"]["B"].append(B1)
        decompositions[r]["fc1"]["db"].append(db1_k)
        
        # Layer 2
        W2_exp = expert_models[k].fc2.weight.data
        b2_exp = expert_models[k].fc2.bias.data
        V2_k = W2_exp - W2_base
        db2_k = b2_exp - b2_base
        
        U2, S2, Vh2 = torch.linalg.svd(V2_k, full_matrices=False)
        U2r = U2[:, :r]
        S2r = torch.diag(torch.sqrt(S2[:r]))
        Vh2r = Vh2[:r, :]
        
        A2 = torch.matmul(U2r, S2r)
        B2 = torch.matmul(S2r, Vh2r)
        
        decompositions[r]["fc2"]["A"].append(A2)
        decompositions[r]["fc2"]["B"].append(B2)
        decompositions[r]["fc2"]["db"].append(db2_k)

# --- SABLE EVALUATION FOR FOUNDATION FEATURES ---
def evaluate_sable(X, r, hybrid_protocol=False, centroid_type="support", M=2):
    with torch.no_grad():
        # First layer (fc1): Always ensembled dynamically
        # Under SABLE, we run the base pass first to get base activations and penultimate features
        H1_base = F.linear(X, W1_base, b1_base)
        
        # For routing, we pass intermediate activations through the routing block
        # SABLE Late Adaptation routes after the intermediate base activations h1
        # Let's compute coefficients based on H1_base (representing penultimate features before FC2)
        coeffs = compute_routing_coefficients(H1_base, centroid_type=centroid_type)
        
        # Apply Top-M Expert Pruning if M is set
        if M < K:
            top_vals, top_idx = torch.topk(coeffs, M, dim=-1)
            mask = torch.zeros_like(coeffs)
            mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals))
            pruned_coeffs = coeffs * mask
            coeffs = pruned_coeffs / (pruned_coeffs.sum(dim=-1, keepdim=True) + 1e-12)
            
        # Blending Layer 1 activations
        H1_experts = torch.zeros(K, X.shape[0], W1_base.shape[0])
        for k in range(K):
            proj = torch.matmul(X, decompositions[r]["fc1"]["B"][k].t())
            out = torch.matmul(proj, decompositions[r]["fc1"]["A"][k].t()) + decompositions[r]["fc1"]["db"][k]
            H1_experts[k] = out
            
        coeffs_reshaped = coeffs.t().unsqueeze(-1) # (K, B, 1)
        H1_blended = torch.sum(coeffs_reshaped * H1_experts, dim=0)
        h1 = F.relu(H1_base + H1_blended) # Combined hidden activations
        
        # Second layer (fc2 / Output layer)
        H2_base = F.linear(h1, W2_base, b2_base)
        
        if hybrid_protocol:
            # Layer-Dependent Hybrid-Rank Protocol: Use full-rank (full-precision) expert updates for the output layer
            H2_experts = torch.zeros(K, h1.shape[0], W2_base.shape[0])
            for k in range(K):
                W2_exp = expert_models[k].fc2.weight.data
                b2_exp = expert_models[k].fc2.bias.data
                H2_experts[k] = F.linear(h1, W2_exp - W2_base, b2_exp - b2_base)
        else:
            # Strict Low-Rank: Use low-rank r decomposed adapters for fc2
            H2_experts = torch.zeros(K, h1.shape[0], W2_base.shape[0])
            for k in range(K):
                proj = torch.matmul(h1, decompositions[r]["fc2"]["B"][k].t())
                out = torch.matmul(proj, decompositions[r]["fc2"]["A"][k].t()) + decompositions[r]["fc2"]["db"][k]
                H2_experts[k] = out
                
        H2_blended = torch.sum(coeffs_reshaped * H2_experts, dim=0)
        final_logits = H2_base + H2_blended
        return final_logits

# --- BASELINES ---
# 1. Uniform Merging in parameter space
def evaluate_uniform(X):
    with torch.no_grad():
        W1_unif = 0.5 * expert_models[0].fc1.weight.data + 0.5 * expert_models[1].fc1.weight.data
        b1_unif = 0.5 * expert_models[0].fc1.bias.data + 0.5 * expert_models[1].fc1.bias.data
        W2_unif = 0.5 * expert_models[0].fc2.weight.data + 0.5 * expert_models[1].fc2.weight.data
        b2_unif = 0.5 * expert_models[0].fc2.bias.data + 0.5 * expert_models[1].fc2.bias.data
        
        h = F.relu(F.linear(X, W1_unif, b1_unif))
        logits = F.linear(h, W2_unif, b2_unif)
        return logits

# 2. PFSR (Parameter-Free Subspace Routing) under heterogeneous stream
def evaluate_pfsr(X, task_ids, heterogeneous=False):
    with torch.no_grad():
        # Pass through base layer 1 first to extract penultimate routing features
        H1_base = F.linear(X, W1_base, b1_base)
        coeffs = compute_routing_coefficients(H1_base, centroid_type="support")
        
        if heterogeneous:
            # Global batch averaging causes heterogeneity collapse
            mean_coeffs = coeffs.mean(dim=0)
            W1_m = W1_base + sum(mean_coeffs[k] * (expert_models[k].fc1.weight.data - W1_base) for k in range(K))
            b1_m = b1_base + sum(mean_coeffs[k] * (expert_models[k].fc1.bias.data - b1_base) for k in range(K))
            W2_m = W2_base + sum(mean_coeffs[k] * (expert_models[k].fc2.weight.data - W2_base) for k in range(K))
            b2_m = b2_base + sum(mean_coeffs[k] * (expert_models[k].fc2.bias.data - b2_base) for k in range(K))
            
            h = F.relu(F.linear(X, W1_m, b1_m))
            logits = F.linear(h, W2_m, b2_m)
        else:
            # Homogeneous: process in task-homogeneous blocks
            logits = torch.zeros(X.shape[0], 10)
            for k in range(K):
                mask = (task_ids == k)
                if not mask.any():
                    continue
                block_X = X[mask]
                block_H1 = H1_base[mask]
                block_coeffs = compute_routing_coefficients(block_H1, centroid_type="support").mean(dim=0)
                
                W1_m = W1_base + sum(block_coeffs[k_exp] * (expert_models[k_exp].fc1.weight.data - W1_base) for k_exp in range(K))
                b1_m = b1_base + sum(block_coeffs[k_exp] * (expert_models[k_exp].fc1.bias.data - b1_base) for k_exp in range(K))
                W2_m = W2_base + sum(block_coeffs[k_exp] * (expert_models[k_exp].fc2.weight.data - W2_base) for k_exp in range(K))
                b2_m = b2_base + sum(block_coeffs[k_exp] * (expert_models[k_exp].fc2.bias.data - b2_base) for k_exp in range(K))
                
                h = F.relu(F.linear(block_X, W1_m, b1_m))
                logits[mask] = F.linear(h, W2_m, b2_m)
        return logits

# --- EVALUATION RUN ---
uniform_acc = get_acc(evaluate_uniform(test_x), test_y)
pfsr_homog = get_acc(evaluate_pfsr(test_x, test_task, heterogeneous=False), test_y)
pfsr_hetero = get_acc(evaluate_pfsr(test_x, test_task, heterogeneous=True), test_y)

results_summary = {}
for r in ranks:
    results_summary[r] = {}
    for hybrid in [False, True]:
        hybrid_str = "hybrid" if hybrid else "strict"
        results_summary[r][hybrid_str] = {}
        for c_type in ["support", "naive", "refined"]:
            logits = evaluate_sable(test_x, r, hybrid_protocol=hybrid, centroid_type=c_type)
            results_summary[r][hybrid_str][c_type] = get_acc(logits, test_y)

# Domain-confounded inputs evaluation
mnist_blend_x = mnist_test_x[:100]
fashion_blend_x = fashion_test_x[:100]
confounded_x = 0.5 * mnist_blend_x + 0.5 * fashion_blend_x
confounded_targets_m = mnist_test_y[:100]
confounded_targets_f = fashion_test_y[:100]

conf_results = {}
for r in ranks:
    conf_results[r] = {}
    for hybrid in [False, True]:
        hybrid_str = "hybrid" if hybrid else "strict"
        conf_results[r][hybrid_str] = {}
        for c_type in ["support", "naive", "refined"]:
            logits_soft = evaluate_sable(confounded_x, r, hybrid_protocol=hybrid, centroid_type=c_type, M=2)
            logits_hard = evaluate_sable(confounded_x, r, hybrid_protocol=hybrid, centroid_type=c_type, M=1)
            
            conf_results[r][hybrid_str][c_type] = {
                "soft": evaluate_confounded_accuracy(logits_soft, confounded_targets_m, confounded_targets_f),
                "hard": evaluate_confounded_accuracy(logits_hard, confounded_targets_m, confounded_targets_f)
            }

uniform_blend_acc = evaluate_confounded_accuracy(evaluate_uniform(confounded_x), confounded_targets_m, confounded_targets_f)
pfsr_blend_acc = evaluate_confounded_accuracy(evaluate_pfsr(confounded_x, torch.zeros(100), heterogeneous=True), confounded_targets_m, confounded_targets_f)

print("\n" + "="*80)
print("HIGH-DIMENSIONAL RESNET-18 FOUNDATION EXPERIMENT RESULTS")
print("="*80)
print(f"Expert Ceiling Joint Mean: {np.mean(expert_ceilings):.2f}% (MNIST: {m_acc:.2f}%, F-MNIST: {f_acc:.2f}%)")
print(f"Uniform Merging Accuracy:   {uniform_acc:.2f}%")
print(f"PFSR Homogeneous Accuracy:  {pfsr_homog:.2f}%")
print(f"PFSR Heterogeneous Accuracy:{pfsr_hetero:.2f}% (Heterogeneity Collapse!)")
print("-" * 80)
print(f"{'Configuration':<35} | {'Rank r=2':<9} | {'Rank r=4':<9} | {'Rank r=8':<9} | {'Rank r=16':<9}")
print("-" * 80)

# 1. Strict Low-Rank SABLE (Support-16)
print(f"{'SABLE Strict (Support-16)':<35} | " + " | ".join(f"{results_summary[r]['strict']['support']:>7.2f}%" for r in ranks))
# 2. Strict Low-Rank SABLE (Naive Zero-Data)
print(f"{'SABLE Strict (Naive Zero-Data)':<35} | " + " | ".join(f"{results_summary[r]['strict']['naive']:>7.2f}%" for r in ranks))
# 3. Strict Low-Rank SABLE (Refined Zero-Data)
print(f"{'SABLE Strict (Refined Zero)':<35} | " + " | ".join(f"{results_summary[r]['strict']['refined']:>7.2f}%" for r in ranks))

print("-" * 80)
# 4. Hybrid-Rank SABLE (Support-16)
print(f"{'SABLE Hybrid (Support-16)':<35} | " + " | ".join(f"{results_summary[r]['hybrid']['support']:>7.2f}%" for r in ranks))
# 5. Hybrid-Rank SABLE (Naive Zero-Data)
print(f"{'SABLE Hybrid (Naive Zero-Data)':<35} | " + " | ".join(f"{results_summary[r]['hybrid']['naive']:>7.2f}%" for r in ranks))
# 6. Hybrid-Rank SABLE (Refined Zero-Data)
print(f"{'SABLE Hybrid (Refined Zero)':<35} | " + " | ".join(f"{results_summary[r]['hybrid']['refined']:>7.2f}%" for r in ranks))
print("="*80)

print("\nDomain-Confounded Blended Streams (Recall@2 Joint Classification Success):")
print(f"  Uniform Merging:            {uniform_blend_acc:.2f}%")
print(f"  PFSR Weight Merging:        {pfsr_blend_acc:.2f}%")
print("-" * 80)
print(f"{'SABLE Blend (r=2) [Strict, Support]':<40} | Soft (M=2): {conf_results[2]['strict']['support']['soft']:>5.2f}% | Hard (M=1): {conf_results[2]['strict']['support']['hard']:>5.2f}%")
print(f"{'SABLE Blend (r=2) [Hybrid, Support]':<40} | Soft (M=2): {conf_results[2]['hybrid']['support']['soft']:>5.2f}% | Hard (M=1): {conf_results[2]['hybrid']['support']['hard']:>5.2f}%")
print(f"{'SABLE Blend (r=8) [Strict, Support]':<40} | Soft (M=2): {conf_results[8]['strict']['support']['soft']:>5.2f}% | Hard (M=1): {conf_results[8]['strict']['support']['hard']:>5.2f}%")
print(f"{'SABLE Blend (r=8) [Hybrid, Support]':<40} | Soft (M=2): {conf_results[8]['hybrid']['support']['soft']:>5.2f}% | Hard (M=1): {conf_results[8]['hybrid']['support']['hard']:>5.2f}%")
print(f"{'SABLE Blend (r=8) [Hybrid, Refined]':<40} | Soft (M=2): {conf_results[8]['hybrid']['refined']['soft']:>5.2f}% | Hard (M=1): {conf_results[8]['hybrid']['refined']['hard']:>5.2f}%")
print("="*80)

# Let's save the results to a file for easy paper writing!
with open("foundation_experiment_results.md", "w") as f:
    f.write(f"# High-Dimensional ResNet-18 Foundation Experiment Results\n\n")
    f.write(f"This experiment extracts 512-dimensional real-world image representations from a pre-trained ImageNet ResNet-18 model and trains a 2-layer MLP classifier on top, evaluating SABLE's performance on real foundation features.\n\n")
    f.write(f"- **Expert Ceiling Joint Mean:** {np.mean(expert_ceilings):.2f}% (MNIST Expert: {m_acc:.2f}%, F-MNIST Expert: {f_acc:.2f}%)\n")
    f.write(f"- **Uniform Merging:** {uniform_acc:.2f}%\n")
    f.write(f"- **PFSR Homogeneous:** {pfsr_homog:.2f}%\n")
    f.write(f"- **PFSR Heterogeneous:** {pfsr_hetero:.2f}% (Heterogeneity Collapse!)\n\n")
    
    f.write("## Standard Stream Accuracies (MNIST + FashionMNIST)\n\n")
    f.write("| Configuration | r=2 | r=4 | r=8 | r=16 |\n")
    f.write("| --- | --- | --- | --- | --- |\n")
    f.write(f"| SABLE Strict (Support-16) | {results_summary[2]['strict']['support']:.2f}% | {results_summary[4]['strict']['support']:.2f}% | {results_summary[8]['strict']['support']:.2f}% | {results_summary[16]['strict']['support']:.2f}% |\n")
    f.write(f"| SABLE Strict (Naive Zero) | {results_summary[2]['strict']['naive']:.2f}% | {results_summary[4]['strict']['naive']:.2f}% | {results_summary[8]['strict']['naive']:.2f}% | {results_summary[16]['strict']['naive']:.2f}% |\n")
    f.write(f"| SABLE Strict (Refined Zero) | {results_summary[2]['strict']['refined']:.2f}% | {results_summary[4]['strict']['refined']:.2f}% | {results_summary[8]['strict']['refined']:.2f}% | {results_summary[16]['strict']['refined']:.2f}% |\n")
    f.write(f"| SABLE Hybrid (Support-16) | {results_summary[2]['hybrid']['support']:.2f}% | {results_summary[4]['hybrid']['support']:.2f}% | {results_summary[8]['hybrid']['support']:.2f}% | {results_summary[16]['hybrid']['support']:.2f}% |\n")
    f.write(f"| SABLE Hybrid (Naive Zero) | {results_summary[2]['hybrid']['naive']:.2f}% | {results_summary[4]['hybrid']['naive']:.2f}% | {results_summary[8]['hybrid']['naive']:.2f}% | {results_summary[16]['hybrid']['naive']:.2f}% |\n")
    f.write(f"| SABLE Hybrid (Refined Zero) | {results_summary[2]['hybrid']['refined']:.2f}% | {results_summary[4]['hybrid']['refined']:.2f}% | {results_summary[8]['hybrid']['refined']:.2f}% | {results_summary[16]['hybrid']['refined']:.2f}% |\n\n")
    
    f.write("## Domain-Confounded Blended Streams (Recall@2 Joint Success)\n\n")
    f.write(f"- Uniform Merging: {uniform_blend_acc:.2f}%\n")
    f.write(f"- PFSR Weight Merging: {pfsr_blend_acc:.2f}%\n")
    for r in [2, 8]:
        for hybrid in [False, True]:
            h_str = "Hybrid" if hybrid else "Strict"
            for c_type in ["support", "refined"]:
                c_str = "Support" if c_type == "support" else "Refined Zero"
                f.write(f"- SABLE {h_str} (r={r}) [{c_str}] Soft: {conf_results[r]['hybrid' if hybrid else 'strict'][c_type]['soft']:.2f}% | Hard: {conf_results[r]['hybrid' if hybrid else 'strict'][c_type]['hard']:.2f}%\n")

print("Saved results to foundation_experiment_results.md")
