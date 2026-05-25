import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
import random
from torch.func import functional_call
from run_eval import ResNetWithFeatures, compute_prototypes, compute_empirical_fisher, construct_merged_params, riemannian_gradient_surgery

def main():
    # Set seeds for reproducibility
    random.seed(42)
    np.seed_42 = 42 # standard
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

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
    
    # Prototypes
    p1 = compute_prototypes(experts[0], cifar_val, num_samples=500, device=device)
    p2 = compute_prototypes(experts[1], svhn_val, num_samples=500, device=device)
    
    # Individual expert feature means
    mu1 = torch.zeros(512, device=device)
    mu2 = torch.zeros(512, device=device)
    for c in range(10):
        mu1 += p1[c]
        mu2 += p2[c]
    mu1 = F.normalize(mu1 / 10.0, p=2, dim=0)
    mu2 = F.normalize(mu2 / 10.0, p=2, dim=0)
    
    # Center and normalize prototypes
    p1 = {c: F.normalize(p1[c] - mu1, p=2, dim=0) for c in range(10)}
    p2 = {c: F.normalize(p2[c] - mu2, p=2, dim=0) for c in range(10)}
    
    # Precompute Fisher Information
    print("Computing Fisher Information for experts...")
    f1 = compute_empirical_fisher(experts[0], cifar_val, num_samples=500, device=device)
    f2 = compute_empirical_fisher(experts[1], svhn_val, num_samples=500, device=device)
    print("Pre-computation complete.")
    
    # 3. Create non-stationary test stream (90 batches of size 64)
    test_batches = []
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
    
    # 4. Define evaluation function
    def evaluate_with_G(G, label):
        print(f"\nEvaluating: {label}")
        learnable_names = [name for name, p in experts[0].named_parameters() if p.requires_grad]
        lambdas = {}
        for name in learnable_names:
            lambdas[name] = torch.zeros(3, device=device, requires_grad=True)
            
        merged_model = ResNetWithFeatures().to(device)
        merged_model.eval()
        
        task_accs = {
            'Task A (CIFAR-10)': [],
            'Task B (SVHN)': [],
            'Task C (FashionMNIST)': []
        }
        
        ndr_correct = 0
        ndr_total = 0
        p_novel = {c: None for c in range(10)}
        gamma = 0.1
        
        tau_N = 0.10
        tau_p = 0.90
        tau = 0.10
        eta = 2.0
        alpha_ema = 0.99
        
        in_novel_state = False
        
        for b_idx, (x, y, task) in enumerate(test_batches):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            merged_params = construct_merged_params(experts, lambdas, device=device)
            features, logits = functional_call(merged_model, merged_params, (x,))
            
            _, preds = logits.max(1)
            acc = 100.0 * preds.eq(y).sum().item() / len(y)
            task_accs[task].append(acc)
            
            avg_lambda = torch.stack([F.softmax(lambdas[name], dim=0) for name in learnable_names]).mean(dim=0)
            mu_merged = avg_lambda[0] * mu1 + avg_lambda[1] * mu2
            
            z_centered = features - mu_merged
            z_normalized = F.normalize(z_centered, p=2, dim=-1)
            
            # Unbiased Routing
            with torch.no_grad():
                f1, _ = experts[0](x)
                f1_centered = F.normalize(f1 - mu1, p=2, dim=-1)
                c1_scores = [max([torch.dot(f1_centered[i], p1[c]).item() for c in range(10)]) for i in range(len(x))]
                c1_score = np.mean(c1_scores)
                
                f2, _ = experts[1](x)
                f2_centered = F.normalize(f2 - mu2, p=2, dim=-1)
                c2_scores = [max([torch.dot(f2_centered[i], p2[c]).item() for c in range(10)]) for i in range(len(x))]
                c2_score = np.mean(c2_scores)
                
            max_cohesion = max(c1_score, c2_score)
            if not in_novel_state:
                if max_cohesion < tau_N:
                    in_novel_state = True
            else:
                if max_cohesion > 0.15:
                    in_novel_state = False
                    
            is_novel = in_novel_state
            
            gt_novel = (task == 'Task C (FashionMNIST)')
            if gt_novel:
                ndr_total += 1
                if is_novel:
                    ndr_correct += 1
                    
            if not is_novel:
                k_star = np.argmax([c1_score, c2_score])
                with torch.no_grad():
                    target_logits = torch.full((3,), -1.0, device=device)
                    target_logits[k_star] = 1.0
                    for name in learnable_names:
                        lambdas[name].copy_((1.0 - alpha_ema) * lambdas[name] + alpha_ema * target_logits)
            else:
                # Novel domain
                if all(p is None for p in p_novel.values()):
                    with torch.no_grad():
                        for name in learnable_names:
                            lambdas[name].fill_(0.0)
                    merged_params = construct_merged_params(experts, lambdas, device=device)
                    features, logits = functional_call(merged_model, merged_params, (x,))
                    
                    avg_lambda = torch.stack([F.softmax(lambdas[name], dim=0) for name in learnable_names]).mean(dim=0)
                    mu_merged = avg_lambda[0] * mu1 + avg_lambda[1] * mu2
                    z_centered = features - mu_merged
                    z_normalized = F.normalize(z_centered, p=2, dim=-1)
                    
                with torch.no_grad():
                    entropies = []
                    for k in range(3):
                        _, exp_logits = experts[k](x)
                        probs = F.softmax(exp_logits, dim=-1)
                        ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
                        entropies.append(ent)
                    best_expert_idx = np.argmin(entropies)
                    best_features, best_logits = experts[best_expert_idx](x)
                    best_mu = mu1 if best_expert_idx == 0 else mu2 if best_expert_idx == 1 else (0.5 * mu1 + 0.5 * mu2)
                    best_z_centered = best_features - best_mu
                    best_z_normalized = F.normalize(best_z_centered, p=2, dim=-1)
                    
                    best_probs = F.softmax(best_logits, dim=-1)
                    confidences, pseudo_labels = best_probs.max(1)
                    
                batch_class_feats = {c: [] for c in range(10)}
                for i in range(len(x)):
                    if confidences[i].item() > tau_p:
                        c = pseudo_labels[i].item()
                        batch_class_feats[c].append(best_z_normalized[i])
                        
                for c in range(10):
                    if len(batch_class_feats[c]) > 0:
                        avg_feat = torch.stack(batch_class_feats[c]).mean(dim=0)
                        avg_feat = F.normalize(avg_feat, p=2, dim=0)
                        if p_novel[c] is None:
                            p_novel[c] = avg_feat.detach()
                        else:
                            p_novel[c] = F.normalize((1.0 - gamma) * p_novel[c] + gamma * avg_feat, p=2, dim=0).detach()
                            
                class_losses = {}
                for c in range(10):
                    c_mask = (pseudo_labels == c) & (confidences > tau_p)
                    if c_mask.sum().item() > 0:
                        c_feats = z_normalized[c_mask]
                        all_protos = []
                        for pc in range(10):
                            all_protos.append(p1[pc])
                        for pc in range(10):
                            all_protos.append(p2[pc])
                        for pc in range(10):
                            if p_novel[pc] is not None:
                                all_protos.append(p_novel[pc])
                            else:
                                all_protos.append(torch.zeros(512, device=device))
                        stacked_protos = torch.stack(all_protos)
                        sim_matrix = torch.matmul(c_feats, stacked_protos.T) / tau
                        target_idx = 20 + c
                        target_tensor = torch.full((len(c_feats),), target_idx, device=device, dtype=torch.long)
                        class_losses[c] = F.cross_entropy(sim_matrix, target_tensor)
                        
                if len(class_losses) > 0:
                    class_grads = []
                    for c, loss_val in class_losses.items():
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
                        
                    final_grad = riemannian_gradient_surgery(class_grads, G)
                    with torch.no_grad():
                        for name in learnable_names:
                            if name in final_grad:
                                lambdas[name].sub_(eta * final_grad[name])
                                
        avg_acc_A = np.mean(task_accs['Task A (CIFAR-10)'])
        avg_acc_B = np.mean(task_accs['Task B (SVHN)'])
        avg_acc_C = np.mean(task_accs['Task C (FashionMNIST)'])
        overall_acc = (avg_acc_A + avg_acc_B + avg_acc_C) / 3.0
        ndr = 100.0 * ndr_correct / ndr_total if ndr_total > 0 else 0.0
        
        print(f"Results for {label}:")
        print(f"  Task A Acc: {avg_acc_A:.2f}%")
        print(f"  Task B Acc: {avg_acc_B:.2f}%")
        print(f"  Task C Acc: {avg_acc_C:.2f}%")
        print(f"  Overall:    {overall_acc:.2f}%")
        
        return {
            'Task A': avg_acc_A,
            'Task B': avg_acc_B,
            'Task C': avg_acc_C,
            'Overall': overall_acc,
            'NDR': ndr
        }

    # Run over alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    eps_scale = 1e-6
    
    for alpha in alphas:
        # Construct G for this alpha
        G = {}
        for name in f1:
            if name in f2:
                joint_f = 0.5 * (f1[name] + f2[name])
                avg_sens = torch.mean(joint_f).item()
                G[name] = (avg_sens + eps_scale) ** alpha
        label = f"IGGS-PROTO (alpha={alpha:.2f})"
        if alpha == 0.0:
            label = "Euc-GS-PROTO (alpha=0.0)"
        results[alpha] = evaluate_with_G(G, label)
        
    print("\n" + "="*70)
    print("ABLATION STUDY: EFFECT OF FISHER EXPONENT alpha")
    print("="*70)
    print(f"{'alpha':<10} | {'Task A':<10} | {'Task B':<10} | {'Task C (Novel)':<15} | {'Overall':<10} | {'NDR (%)':<8}")
    print("-"*70)
    for alpha in alphas:
        r = results[alpha]
        print(f"{alpha:<10.2f} | {r['Task A']:.2f}% | {r['Task B']:.2f}% | {r['Task C']:.2f}% | {r['Overall']:.2f}% | {r['NDR']:.2f}%")
    print("="*70)

if __name__ == '__main__':
    main()
