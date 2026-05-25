import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import os
import numpy as np
import random
from torch.func import functional_call

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Disable cuDNN to avoid initialization errors on the cluster
torch.backends.cudnn.enabled = False

# Custom ResNet-18 wrapper to return features and logits
class ResNetWithFeatures(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base_model = resnet18()
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        # Extract features from average pooling layer
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        features = torch.flatten(x, 1) # [B, 512]
        
        logits = self.base_model.fc(features) # [B, num_classes]
        return features, logits

# Helper for simplex projection
def project_simplex(v):
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0)
    indices = torch.arange(1, len(v) + 1, device=v.device)
    cond = sorted_v - (cssv - 1.0) / indices > 0
    idx = torch.sum(cond) - 1
    theta = (cssv[idx] - 1.0) / (idx + 1)
    return torch.clamp(v - theta, min=0.0)

# Empirical Fisher computation
def compute_empirical_fisher(model, dataset, num_samples=500, batch_size=32, device='cuda'):
    model.eval()
    model.to(device)
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    samples_processed = 0
    
    for x, y in dataloader:
        if samples_processed >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        _, logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        bs = len(x)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.data ** 2) * bs
        samples_processed += bs
        
    for name in fisher:
        fisher[name] /= samples_processed
    return fisher

# Prototype computation
def compute_prototypes(model, dataset, num_samples=500, batch_size=32, device='cuda'):
    model.eval()
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    class_features = {c: [] for c in range(10)}
    samples_processed = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            if samples_processed >= num_samples:
                break
            x = x.to(device)
            features, _ = model(x)
            # Normalize features
            features = F.normalize(features, p=2, dim=-1)
            
            for i in range(len(x)):
                c = y[i].item()
                class_features[c].append(features[i])
            samples_processed += len(x)
            
    prototypes = {}
    for c in range(10):
        if len(class_features[c]) > 0:
            stacked = torch.stack(class_features[c])
            mean_feat = torch.mean(stacked, dim=0)
            prototypes[c] = F.normalize(mean_feat, p=2, dim=0)
        else:
            prototypes[c] = torch.zeros(512, device=device)
    return prototypes

# Construct merged model state dict
def construct_merged_params(experts, lambdas, device='cuda'):
    merged_dict = {}
    first_expert = experts[0]
    
    # Process parameters
    for name, param in first_expert.named_parameters():
        if name in lambdas:
            l = F.softmax(lambdas[name], dim=0)
            merged_param = 0.0
            for k, expert in enumerate(experts):
                expert_param = dict(expert.named_parameters())[name]
                merged_param = merged_param + l[k] * expert_param
            merged_dict[name] = merged_param
        else:
            merged_dict[name] = param.to(device)
            
    # Process buffers
    for name, buffer in first_expert.named_buffers():
        if buffer.dtype.is_floating_point:
            merged_buf = 0.0
            for expert in experts:
                merged_buf = merged_buf + (1.0 / len(experts)) * dict(expert.named_buffers())[name]
            merged_dict[name] = merged_buf.to(device)
        else:
            merged_dict[name] = buffer.to(device)
            
    return merged_dict

# Information-Geometric Gradient Surgery (pairwise)
def riemannian_inner_product(g1, g2, G):
    val = 0.0
    for name in g1:
        if name in g2 and name in G:
            val += G[name] * torch.dot(g1[name], g2[name])
    return val

def riemannian_gradient_surgery(gradients, G, eps=1e-8):
    if len(gradients) == 0:
        return {}
    if len(gradients) == 1:
        return gradients[0]
        
    projected_grads = []
    for g in gradients:
        projected_grads.append({name: val.clone() for name, val in g.items()})
        
    num_grads = len(projected_grads)
    indices = list(range(num_grads))
    
    for i in indices:
        g_i = projected_grads[i]
        other_indices = [j for j in indices if j != i]
        # Shuffle for order-invariance approximation
        random.shuffle(other_indices)
        for j in other_indices:
            # Project against the original unmodified gradient of j
            g_j_orig = gradients[j]
            ip = riemannian_inner_product(g_i, g_j_orig, G)
            if ip < 0:
                norm_sq_j = riemannian_inner_product(g_j_orig, g_j_orig, G)
                factor = ip / (norm_sq_j + eps)
                for name in g_i:
                    if name in g_j_orig:
                        g_i[name] -= factor * g_j_orig[name]
                        
    summed_grad = {}
    for name in projected_grads[0]:
        summed_grad[name] = torch.zeros_like(projected_grads[0][name])
        for g in projected_grads:
            if name in g:
                summed_grad[name] += g[name]
                
    return summed_grad

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load experts
    print("Loading experts...")
    experts = []
    for name in ['expert1_cifar', 'expert2_svhn', 'expert3_fmnist']:
        model = ResNetWithFeatures()
        model.base_model.load_state_dict(torch.load(f"experts/{name}.pt", map_location=device))
        model = model.to(device)
        model.eval()
        experts.append(model)
    print("Experts loaded successfully.")
    
    # 2. Setup datasets
    print("Setting up test datasets...")
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    # Pre-compute prototypes on small validation subsets
    print("Pre-computing expert class prototypes and Fisher information...")
    cifar_val = Subset(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb), list(range(500)))
    svhn_val = Subset(datasets.SVHN(root='./data', split='train', download=True, transform=transform_rgb), list(range(500)))
    
    # Prototypes for Expert 1 (CIFAR-10) and Expert 2 (SVHN)
    p1 = compute_prototypes(experts[0], cifar_val, num_samples=500, device=device)
    p2 = compute_prototypes(experts[1], svhn_val, num_samples=500, device=device)
    
    # Compute individual expert feature means
    mu1 = torch.zeros(512, device=device)
    mu2 = torch.zeros(512, device=device)
    for c in range(10):
        mu1 += p1[c]
        mu2 += p2[c]
    mu1 = F.normalize(mu1 / 10.0, p=2, dim=0)
    mu2 = F.normalize(mu2 / 10.0, p=2, dim=0)
    
    # Center and normalize prototypes to match Isotropic Feature Centering
    p1 = {c: F.normalize(p1[c] - mu1, p=2, dim=0) for c in range(10)}
    p2 = {c: F.normalize(p2[c] - mu2, p=2, dim=0) for c in range(10)}
    
    # Precompute Fisher Information
    f1 = compute_empirical_fisher(experts[0], cifar_val, num_samples=500, device=device)
    f2 = compute_empirical_fisher(experts[1], svhn_val, num_samples=500, device=device)
    
    # Average diagonal Fisher sensitivities for each parameter
    G = {}
    alpha = 0.5
    eps_scale = 1e-6
    for name in f1:
        if name in f2:
            # Joint Fisher
            joint_f = 0.5 * (f1[name] + f2[name])
            # Tensor-level average sensitivity
            avg_sens = torch.mean(joint_f).item()
            G[name] = (avg_sens + eps_scale) ** alpha
            
    print("Pre-computation complete.")
    
    # 3. Create non-stationary test stream (90 batches of size 64)
    # Batches 1-30: CIFAR-10
    # Batches 31-60: SVHN
    # Batches 61-90: FashionMNIST (Novel)
    test_batches = []
    
    # Sample subsets from test sets
    cifar_loader = DataLoader(Subset(cifar_test, list(range(30 * 64))), batch_size=64, shuffle=False)
    svhn_loader = DataLoader(Subset(svhn_test, list(range(30 * 64))), batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(Subset(fmnist_test, list(range(30 * 64))), batch_size=64, shuffle=False)
    
    for x, y in cifar_loader:
        test_batches.append((x, y, 'Task A (CIFAR-10)'))
    for x, y in svhn_loader:
        test_batches.append((x, y, 'Task B (SVHN)'))
    for x, y in fmnist_loader:
        test_batches.append((x, y, 'Task C (FashionMNIST)'))
        
    print(f"Constructed test stream with {len(test_batches)} batches.")
    
    # 4. Define evaluation function for a method
    def evaluate_method(method_name):
        print(f"\nEvaluating: {method_name}")
        # Initialize coefficients λ for each parameter layer as raw logits (uniform over K=3 experts)
        learnable_names = [name for name, p in experts[0].named_parameters() if p.requires_grad]
        lambdas = {}
        for name in learnable_names:
            # λ_w of shape [3] (K=3 experts)
            lambdas[name] = torch.zeros(3, device=device, requires_grad=True)
            
        # Target merged model wrapper
        merged_model = ResNetWithFeatures().to(device)
        merged_model.eval()
        
        # Tracking metrics
        task_accs = {
            'Task A (CIFAR-10)': [],
            'Task B (SVHN)': [],
            'Task C (FashionMNIST)': []
        }
        
        ndr_correct = 0
        ndr_total = 0
        cohesion_scores = []
        
        # Online novel prototypes representation
        p_novel = {c: None for c in range(10)}
        gamma = 0.1 # Online prototype refinement rate
        
        # Test-time hyperparameters
        tau_N = 0.10
        tau_p = 0.90
        tau = 0.10
        eta = 2.0
        alpha_ema = 0.99
        
        in_novel_state = False
        
        for b_idx, (x, y, task) in enumerate(test_batches):
            x, y = x.to(device), y.to(device)
            
            # Step 1: Forward pass with current merged model
            merged_params = construct_merged_params(experts, lambdas, device=device)
            features, logits = functional_call(merged_model, merged_params, (x,))
            
            # Compute accuracy on current batch
            _, preds = logits.max(1)
            acc = 100.0 * preds.eq(y).sum().item() / len(y)
            task_accs[task].append(acc)
            
            # Isotropic Feature Centering (IFC)
            # mu_merged = lambda_1 * mu1 + lambda_2 * mu2
            # Use average lambda across all layers (applying softmax first)
            avg_lambda = torch.stack([F.softmax(lambdas[name], dim=0) for name in learnable_names]).mean(dim=0)
            mu_merged = avg_lambda[0] * mu1 + avg_lambda[1] * mu2
            
            z_centered = features - mu_merged
            z_normalized = F.normalize(z_centered, p=2, dim=-1)
            
            # Unbiased Routing (UR) via Prototype Cohesion (Fixed-Expert style to break feedback trap!)
            with torch.no_grad():
                # Expert 1 (CIFAR-10)
                f1, _ = experts[0](x)
                f1_centered = F.normalize(f1 - mu1, p=2, dim=-1)
                c1_scores = []
                for i in range(len(x)):
                    sims1 = [torch.dot(f1_centered[i], p1[c]).item() for c in range(10)]
                    c1_scores.append(max(sims1))
                c1_score = np.mean(c1_scores)
                
                # Expert 2 (SVHN)
                f2, _ = experts[1](x)
                f2_centered = F.normalize(f2 - mu2, p=2, dim=-1)
                c2_scores = []
                for i in range(len(x)):
                    sims2 = [torch.dot(f2_centered[i], p2[c]).item() for c in range(10)]
                    c2_scores.append(max(sims2))
                c2_score = np.mean(c2_scores)
                
            max_cohesion = max(c1_score, c2_score)
            cohesion_scores.append(max_cohesion)
            
            # Stateful hysteresis routing to prevent false-negative resets on novel domains
            if not in_novel_state:
                if max_cohesion < tau_N:
                    in_novel_state = True
            else:
                if max_cohesion > 0.15: # High threshold to exit novel state
                    in_novel_state = False
                    
            is_novel = in_novel_state
            if b_idx % 10 == 0:
                print(f"  [Batch {b_idx:02d} - {task}] cohesion1={c1_score:.4f}, cohesion2={c2_score:.4f}, max={max_cohesion:.4f}, is_novel={is_novel}")
            
            # Ground-truth novelty (Task C is novel, Task A & B are known)
            gt_novel = (task == 'Task C (FashionMNIST)')
            if gt_novel:
                ndr_total += 1
                if is_novel:
                    ndr_correct += 1
                    
            # Adaptation logic based on method
            if method_name == 'Static':
                # No adaptation
                pass
                
            elif method_name == 'CPA-Merge':
                # SOTA closed-world routing via prediction entropy on individual experts
                with torch.no_grad():
                    # Evaluate negative entropy for each expert
                    entropies = []
                    for k in range(3):
                        _, exp_logits = experts[k](x)
                        probs = F.softmax(exp_logits, dim=-1)
                        ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
                        entropies.append(ent)
                    # Lower entropy means higher confidence
                    k_star = np.argmin(entropies)
                    # EMA update on logits (scale target to represent strong confidence without saturation)
                    target_logits = torch.full((3,), -1.0, device=device)
                    target_logits[k_star] = 1.0
                    for name in learnable_names:
                        lambdas[name].copy_((1.0 - alpha_ema) * lambdas[name] + alpha_ema * target_logits)
                        
            elif method_name in ['PROTO-TTMM', 'IGGS-PROTO (Ours)']:
                if not is_novel:
                    # Known domain: route to highest cohesion expert
                    k_star = np.argmax([c1_score, c2_score])
                    with torch.no_grad():
                        target_logits = torch.full((3,), -1.0, device=device)
                        target_logits[k_star] = 1.0
                        for name in learnable_names:
                            lambdas[name].copy_((1.0 - alpha_ema) * lambdas[name] + alpha_ema * target_logits)
                else:
                    # Novel domain detected: generate/refine online prototypes and adapt λ
                    # If this is the very first batch of the novel domain, perform Parameter Reset to uniform
                    if all(p is None for p in p_novel.values()):
                        with torch.no_grad():
                            for name in learnable_names:
                                lambdas[name].fill_(0.0)
                        # Recompute merged features/logits with reset lambdas
                        merged_params = construct_merged_params(experts, lambdas, device=device)
                        features, logits = functional_call(merged_model, merged_params, (x,))
                        
                        # Recompute centered features with the new uniform mean
                        avg_lambda = torch.stack([F.softmax(lambdas[name], dim=0) for name in learnable_names]).mean(dim=0)
                        mu_merged = avg_lambda[0] * mu1 + avg_lambda[1] * mu2
                        z_centered = features - mu_merged
                        z_normalized = F.normalize(z_centered, p=2, dim=-1)
                        
                    # Unsupervised Specialist Bootstrapping (USB)
                    with torch.no_grad():
                        entropies = []
                        for k in range(3):
                            _, exp_logits = experts[k](x)
                            probs = F.softmax(exp_logits, dim=-1)
                            ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
                            entropies.append(ent)
                        best_expert_idx = np.argmin(entropies)
                        
                        # Extract features and logits using the best specialist expert
                        best_features, best_logits = experts[best_expert_idx](x)
                        best_mu = mu1 if best_expert_idx == 0 else mu2 if best_expert_idx == 1 else (0.5 * mu1 + 0.5 * mu2)
                        best_z_centered = best_features - best_mu
                        best_z_normalized = F.normalize(best_z_centered, p=2, dim=-1)
                        
                        # Generate pseudo-labels and confidence using the specialist
                        best_probs = F.softmax(best_logits, dim=-1)
                        confidences, pseudo_labels = best_probs.max(1)
                    
                    # Track novel prototypes
                    batch_class_feats = {c: [] for c in range(10)}
                    for i in range(len(x)):
                        if confidences[i].item() > tau_p:
                            c = pseudo_labels[i].item()
                            batch_class_feats[c].append(best_z_normalized[i])
                            
                    # Update novel prototypes
                    for c in range(10):
                        if len(batch_class_feats[c]) > 0:
                            avg_feat = torch.stack(batch_class_feats[c]).mean(dim=0)
                            avg_feat = F.normalize(avg_feat, p=2, dim=0)
                            if p_novel[c] is None:
                                p_novel[c] = avg_feat.detach()
                            else:
                                p_novel[c] = F.normalize((1.0 - gamma) * p_novel[c] + gamma * avg_feat, p=2, dim=0).detach()
                                
                    # Compute Lalign
                    # We compute class-specific losses for gradient surgery
                    class_losses = {}
                    for c in range(10):
                        # Filter samples with pseudo-label c and high confidence
                        c_mask = (pseudo_labels == c) & (confidences > tau_p)
                        if c_mask.sum().item() > 0:
                            # Features for this class
                            c_feats = z_normalized[c_mask]
                            
                            # Logits against all prototypes (p1, p2, p_novel)
                            # We collect all valid prototype tensors
                            all_protos = []
                            # Expert 1 and Expert 2
                            for pc in range(10):
                                all_protos.append(p1[pc])
                            for pc in range(10):
                                all_protos.append(p2[pc])
                            # Novel prototypes (fallback to zero if None)
                            for pc in range(10):
                                if p_novel[pc] is not None:
                                    all_protos.append(p_novel[pc])
                                else:
                                    all_protos.append(torch.zeros(512, device=device))
                                    
                            stacked_protos = torch.stack(all_protos) # [30, 512]
                            
                            # Cosine similarity logits
                            sim_matrix = torch.matmul(c_feats, stacked_protos.T) / tau # [B_c, 30]
                            
                            # Target is the index of the corresponding novel prototype (index 20 + c)
                            target_idx = 20 + c
                            target_tensor = torch.full((len(c_feats),), target_idx, device=device, dtype=torch.long)
                            
                            class_losses[c] = F.cross_entropy(sim_matrix, target_tensor)
                            
                    if len(class_losses) > 0:
                        if b_idx % 5 == 0 or b_idx == 61:
                            # Average lambdas (softmaxed)
                            avg_lam = torch.stack([F.softmax(lambdas[name], dim=0) for name in learnable_names]).mean(dim=0).tolist()
                            print(f"      [Batch {b_idx}] class_losses keys={list(class_losses.keys())}, loss_sum={sum(class_losses.values()).item():.4f}, lambdas={['%.3f' % l for l in avg_lam]}")
                        
                        if method_name == 'PROTO-TTMM':
                            # Standard PROTO-TTMM: compute total alignment loss and backprop
                            total_loss = sum(class_losses.values()) / len(class_losses)
                            
                            # Zero gradients of lambdas
                            for name in learnable_names:
                                if lambdas[name].grad is not None:
                                    lambdas[name].grad.zero_()
                                    
                            total_loss.backward()
                            
                            # Update lambdas via gradient step on raw logits
                            with torch.no_grad():
                                for name in learnable_names:
                                    if lambdas[name].grad is not None:
                                        lambdas[name].sub_(eta * lambdas[name].grad)
                                        
                        elif method_name == 'IGGS-PROTO (Ours)':
                            # Our proposed: Information-Geometric Gradient Surgery
                            class_grads = []
                            for c, loss_val in class_losses.items():
                                # Compute class-specific gradients
                                for name in learnable_names:
                                    if lambdas[name].grad is not None:
                                        lambdas[name].grad.zero_()
                                        
                                loss_val.backward(retain_graph=True)
                                
                                c_grad = {}
                                for name in learnable_names:
                                    if lambdas[name].grad is not None:
                                        c_grad[name] = lambdas[name].grad.clone()
                                    else:
                                        c_grad[name] = torch.zeros(3, device=device)
                                class_grads.append(c_grad)
                                
                            # Perform Riemannian Gradient Surgery
                            final_grad = riemannian_gradient_surgery(class_grads, G)
                            
                            # Update lambdas via gradient step on raw logits
                            with torch.no_grad():
                                for name in learnable_names:
                                    if name in final_grad:
                                        lambdas[name].sub_(eta * final_grad[name])
                                        
        # Compile metrics
        avg_acc_A = np.mean(task_accs['Task A (CIFAR-10)'])
        avg_acc_B = np.mean(task_accs['Task B (SVHN)'])
        avg_acc_C = np.mean(task_accs['Task C (FashionMNIST)'])
        overall_acc = (avg_acc_A + avg_acc_B + avg_acc_C) / 3.0
        ndr = 100.0 * ndr_correct / ndr_total if ndr_total > 0 else 0.0
        avg_cohesion = np.mean(cohesion_scores)
        
        print(f"Results for {method_name}:")
        print(f"  Task A (CIFAR-10) Acc: {avg_acc_A:.2f}%")
        print(f"  Task B (SVHN) Acc:     {avg_acc_B:.2f}%")
        print(f"  Task C (Novel FMNIST): {avg_acc_C:.2f}%")
        print(f"  Overall Accuracy:      {overall_acc:.2f}%")
        print(f"  Novelty Detection Rate: {ndr:.2f}%")
        print(f"  Average Cohesion:       {avg_cohesion:.4f}")
        
        return {
            'Task A': avg_acc_A,
            'Task B': avg_acc_B,
            'Task C': avg_acc_C,
            'Overall': overall_acc,
            'NDR': ndr,
            'Cohesion': avg_cohesion
        }
        
    # Evaluate all 4 methods
    results = {}
    methods = ['Static', 'CPA-Merge', 'PROTO-TTMM', 'IGGS-PROTO (Ours)']
    for m in methods:
        results[m] = evaluate_method(m)
        
    # Print comparison table
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"{'Method':<20} | {'Task A':<8} | {'Task B':<8} | {'Task C (N)':<10} | {'Overall':<8} | {'NDR (%)':<8}")
    print("-"*60)
    for m in methods:
        r = results[m]
        print(f"{m:<20} | {r['Task A']:.2f}% | {r['Task B']:.2f}% | {r['Task C']:.2f}% | {r['Overall']:.2f}% | {r['NDR']:.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()
