import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define Model Wrapper (for training only)
class CLIPClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.model = copy.deepcopy(base_model)
        
        # Unfreeze all visual parameters
        for name, param in self.model.named_parameters():
            if "vision_model" in name or "visual_projection" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, pixel_values):
        image_features = self.model.get_image_features(pixel_values=pixel_values).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.head(image_features)
        return logits

# Normalization for CLIP
normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

# Transforms
transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# Loading Datasets
def load_datasets(dataset_name, is_gray, num_train=2000, num_test=1000, batch_size=128):
    print(f"Loading {dataset_name} dataset...")
    transform = transform_gray if is_gray else transform_rgb
    
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    # Get subsets for fast training/evaluation
    train_indices = list(range(min(num_train, len(train_dataset))))
    test_indices = list(range(min(num_test, len(test_dataset))))
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

# Training Loop (only for non-cached experts)
def train_expert(dataset_name, num_classes, is_gray, base_model, epochs=3, lr=1e-3):
    print(f"\n--- Training Full-Network Expert for {dataset_name} ---")
    train_loader, test_loader = load_datasets(dataset_name, is_gray, num_train=2000, num_test=1000)
    
    model = CLIPClassifier(base_model, num_classes).to(device)
    
    # Backbone fine-tuning uses smaller LR (1e-5), classifier head uses normal LR (1e-3)
    optimizer = optim.AdamW(
        [
            {"params": [p for n, p in model.model.named_parameters() if p.requires_grad], "lr": lr * 0.01},
            {"params": model.head.parameters(), "lr": lr}
        ],
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
        # Eval
        eval_acc = evaluate_model_during_train(model, test_loader)
        print(f"Test Acc: {eval_acc:.2f}%")
        
    # Save checkpoint
    checkpoint_path = f"checkpoints/{dataset_name}_expert.pt"
    visual_state_dict = {n: p.data.cpu() for n, p in model.model.named_parameters() if "vision_model" in n or "visual_projection" in n}
    torch.save({
        'visual_state_dict': visual_state_dict,
        'head_weight': model.head.weight.data.cpu(),
        'head_bias': model.head.bias.data.cpu(),
        'num_classes': num_classes,
        'is_gray': is_gray
    }, checkpoint_path)
    print(f"Saved full-network expert checkpoint to {checkpoint_path}")
    
    return checkpoint_path

# Standard Evaluation Loop (used during training)
def evaluate_model_during_train(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# Merging Core Logic for Full Network
def merge_full_network(base_model, expert_dicts, lambdas, method="ta", **kwargs):
    merged_state_dict = {}
    base_state_dict = base_model.state_dict()
    
    for key in expert_dicts[0].keys():
        if key not in base_state_dict:
            continue
            
        base_weight = base_state_dict[key].to(device)
        expert_weights = [exp[key].to(device) for exp in expert_dicts]
        
        # 1D parameters (biases, layer norms, position/class embeddings)
        if base_weight.dim() < 2:
            task_vectors = [exp - base_weight for exp in expert_weights]
            merged_weight = base_weight + sum(l * T for l, T in zip(lambdas, task_vectors))
        else:
            # 2D or higher parameters (weight matrices, conv filters)
            if method == "ta":
                task_vectors = [exp - base_weight for exp in expert_weights]
                merged_weight = base_weight + sum(l * T for l, T in zip(lambdas, task_vectors))
            elif method == "dare":
                p = kwargs.get("p", 0.2)
                task_vectors = [exp - base_weight for exp in expert_weights]
                dare_task_vectors = []
                for T in task_vectors:
                    mask = (torch.rand_like(T) >= p).float()
                    T_dare = T * mask / (1.0 - p)
                    dare_task_vectors.append(T_dare)
                merged_weight = base_weight + sum(l * T_d for l, T_d in zip(lambdas, dare_task_vectors))
            elif method == "ties":
                K = kwargs.get("K", 0.2)
                original_shape = base_weight.shape
                base_flat = base_weight.reshape(-1)
                expert_flats = [exp.reshape(-1) for exp in expert_weights]
                task_vectors = [exp - base_flat for exp in expert_flats]
                
                trimmed_tvs = []
                for T in task_vectors:
                    k_val = max(1, min(int(len(T) * K), len(T)))
                    threshold = torch.topk(T.abs(), k_val).values[-1]
                    mask = T.abs() >= threshold
                    trimmed_tvs.append(T * mask.float())
                    
                stacked_tvs = torch.stack(trimmed_tvs, dim=0)
                signs = torch.sign(stacked_tvs)
                signs_sum = signs.sum(dim=0)
                consensus_sign = torch.sign(signs_sum)
                consensus_sign[consensus_sign == 0] = torch.sign(stacked_tvs.sum(dim=0))[consensus_sign == 0]
                consensus_sign[consensus_sign == 0] = 1.0
                
                matching_mask = (signs == consensus_sign.unsqueeze(0)) & (stacked_tvs != 0)
                selected_entries = stacked_tvs * matching_mask.float()
                
                counts = (selected_entries != 0).sum(dim=0).float()
                merged_tv = selected_entries.sum(dim=0) / torch.clamp(counts, min=1.0)
                
                l_val = lambdas[0] if len(lambdas) > 0 else 0.5
                merged_weight_flat = base_flat + l_val * merged_tv
                merged_weight = merged_weight_flat.view(original_shape)
            elif method == "svs":
                rank = kwargs.get("rank", 64)
                apply_bwn = kwargs.get("apply_bwn", True)
                apply_entropy = kwargs.get("apply_entropy", False)
                rank_list_out = kwargs.get("rank_list_out", None)
                svd_cache = kwargs.get("svd_cache", None)
                
                original_shape = base_weight.shape
                if base_weight.dim() > 2:
                    base_2d = base_weight.reshape(original_shape[0], -1)
                    expert_2ds = [exp.reshape(original_shape[0], -1) for exp in expert_weights]
                else:
                    base_2d = base_weight
                    expert_2ds = expert_weights
                    
                task_vectors = [exp - base_2d for exp in expert_2ds]
                sliced_task_vectors = []
                
                for exp_idx, T in enumerate(task_vectors):
                    cache_key = (key, exp_idx)
                    if svd_cache is not None and cache_key in svd_cache:
                        U, S, Vh = svd_cache[cache_key]
                    else:
                        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
                        if svd_cache is not None:
                            svd_cache[cache_key] = (U, S, Vh)
                    
                    if apply_entropy and len(S) > 1:
                        # Compute spectral entropy of the task vector
                        S_sum = S.sum()
                        if S_sum > 0:
                            p = S / S_sum
                            entropy = -torch.sum(p * torch.log(p + 1e-10))
                            max_entropy = np.log(len(S))
                            norm_entropy = (entropy / max_entropy).item()
                            # Dynamic rank allocation based on singular value entropy
                            entropy_mult = kwargs.get("entropy_mult", 1.0)
                            r_val = max(1, min(int(round(rank * norm_entropy * entropy_mult)), len(S)))
                        else:
                            r_val = min(rank, len(S))
                    else:
                        r_val = min(rank, len(S))
                        
                    if isinstance(rank_list_out, list):
                        rank_list_out.append(r_val)
                        
                    U_r = U[:, :r_val]
                    S_r = torch.diag(S[:r_val])
                    Vh_r = Vh[:r_val, :]
                    
                    T_sliced = U_r @ S_r @ Vh_r
                    sliced_task_vectors.append(T_sliced)
                    
                merged_weight_2d = base_2d + sum(l * T_s for l, T_s in zip(lambdas, sliced_task_vectors))
                
                if apply_bwn:
                    expert_norms = [torch.linalg.norm(exp, ord='fro') for exp in expert_2ds]
                    total_lambda = sum(lambdas)
                    if total_lambda == 0:
                        weights = [1.0 / len(lambdas)] * len(lambdas)
                    else:
                        weights = [l / total_lambda for l in lambdas]
                        
                    target_norm = sum(w * norm for w, norm in zip(weights, expert_norms))
                    merged_norm = torch.linalg.norm(merged_weight_2d, ord='fro')
                    
                    alpha = target_norm / torch.clamp(merged_norm, min=1e-8)
                    merged_weight_2d = alpha * merged_weight_2d
                    
                if base_weight.dim() > 2:
                    merged_weight = merged_weight_2d.view(original_shape)
                else:
                    merged_weight = merged_weight_2d
                    
        merged_state_dict[key] = merged_weight.cpu()
        
    merged_model = copy.deepcopy(base_model)
    merged_model.load_state_dict(merged_state_dict, strict=False)
    return merged_model


# Pipeline Execution
def main():
    # Load Pre-trained CLIP Base Model
    print("Loading pre-trained CLIP model...")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    
    # Define tasks
    tasks = [
        {'name': 'MNIST', 'num_classes': 10, 'is_gray': True},
        {'name': 'FashionMNIST', 'num_classes': 10, 'is_gray': True},
        {'name': 'CIFAR10', 'num_classes': 10, 'is_gray': False},
        {'name': 'SVHN', 'num_classes': 10, 'is_gray': False}
    ]
    
    checkpoint_paths = []
    
    # 1. Train or load task experts
    for task in tasks:
        ckpt_path = f"checkpoints/{task['name']}_expert.pt"
        if os.path.exists(ckpt_path):
            print(f"Found existing checkpoint for {task['name']}, loading...")
            checkpoint_paths.append(ckpt_path)
        else:
            ckpt_path = train_expert(
                dataset_name=task['name'],
                num_classes=task['num_classes'],
                is_gray=task['is_gray'],
                base_model=base_model,
                epochs=3,
                lr=1e-3
            )
            checkpoint_paths.append(ckpt_path)
            
    # Load loaders for evaluation
    loaders = {}
    for task in tasks:
        _, test_loader = load_datasets(task['name'], task['is_gray'], num_train=2000, num_test=1000)
        loaders[task['name']] = test_loader
        
    # Load heads of the experts
    heads = {}
    expert_dicts = []
    expert_individual_accs = {}
    
    for task, ckpt_path in zip(tasks, checkpoint_paths):
        ckpt = torch.load(ckpt_path)
        head = nn.Linear(512, task['num_classes'])
        head.weight.data.copy_(ckpt['head_weight'])
        head.bias.data.copy_(ckpt['head_bias'])
        heads[task['name']] = head.to(device).eval()
        expert_dicts.append(ckpt['visual_state_dict'])

    # Helper for evaluating a given merged model
    def evaluate_merged_model_multi_task(merged_model):
        merged_model.eval()
        accuracies = {}
        
        with torch.no_grad():
            for task in tasks:
                correct = 0
                total = 0
                loader = loaders[task['name']]
                head = heads[task['name']]
                
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    # Run images through the merged visual model
                    image_features = merged_model.get_image_features(pixel_values=images).pooler_output
                    # Normalize visual features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits = head(image_features)
                    
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                accuracies[task['name']] = 100. * correct / total
                
        accuracies['Average'] = np.mean([accuracies[t['name']] for t in tasks])
        return accuracies

    # Evaluation: Zero-shot baselines
    print("\n--- Evaluating Zero-Shot Baseline ---")
    zero_shot_accs = evaluate_merged_model_multi_task(base_model)
    print("Zero-shot Accuracies:", zero_shot_accs)
    
    # Evaluation: Individual Expert Baselines on their own tasks
    print("\n--- Evaluating Task Experts on their own tasks ---")
    for task, exp_dict in zip(tasks, expert_dicts):
        # Temp model with this expert's weights
        exp_model = copy.deepcopy(base_model)
        exp_model.load_state_dict(exp_dict, strict=False)
        accs = evaluate_merged_model_multi_task(exp_model)
        expert_individual_accs[task['name']] = accs[task['name']]
    print("Individual Expert Accuracies on own tasks:", expert_individual_accs)

    # 2. Sweep over scaling coefficient lambda for Task Arithmetic
    print("\n--- Sweeping lambda for Task Arithmetic ---")
    lambda_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ta_results = {}
    best_ta_lambda = 0.3
    best_ta_avg = 0.0
    
    for l in lambda_sweep:
        lambdas = [l] * len(tasks)
        merged_model = merge_full_network(base_model, expert_dicts, lambdas, method="ta")
        accs = evaluate_merged_model_multi_task(merged_model)
        ta_results[l] = accs
        print(f"TA Lambda={l:.1f} | Avg Acc: {accs['Average']:.2f}%")
        if accs['Average'] > best_ta_avg:
            best_ta_avg = accs['Average']
            best_ta_lambda = l
            
    print(f"Best TA Lambda: {best_ta_lambda} with Average Accuracy: {best_ta_avg:.2f}%")
    
    # 3. Sweep over Rank k and lambda for SVS and SVS without BWN
    print("\n--- Sweeping Rank k and lambda for SVS ---")
    ranks = [16, 32, 64, 128, 256]
    
    svs_bwn_results = {}
    svs_nobwn_results = {}
    
    best_svs_bwn = {'rank': 64, 'lambda': 0.3, 'acc': 0.0, 'results': None}
    best_svs_nobwn = {'rank': 64, 'lambda': 0.3, 'acc': 0.0, 'results': None}
    
    for r in ranks:
        svs_bwn_results[r] = {}
        svs_nobwn_results[r] = {}
        
        for l in lambda_sweep:
            lambdas = [l] * len(tasks)
            
            # SVS with BWN
            merged_model_bwn = merge_full_network(base_model, expert_dicts, lambdas, method="svs", rank=r, apply_bwn=True)
            accs_bwn = evaluate_merged_model_multi_task(merged_model_bwn)
            svs_bwn_results[r][l] = accs_bwn
            
            if accs_bwn['Average'] > best_svs_bwn['acc']:
                best_svs_bwn['acc'] = accs_bwn['Average']
                best_svs_bwn['rank'] = r
                best_svs_bwn['lambda'] = l
                best_svs_bwn['results'] = accs_bwn
                
            # SVS without BWN
            merged_model_nobwn = merge_full_network(base_model, expert_dicts, lambdas, method="svs", rank=r, apply_bwn=False)
            accs_nobwn = evaluate_merged_model_multi_task(merged_model_nobwn)
            svs_nobwn_results[r][l] = accs_nobwn
            
            if accs_nobwn['Average'] > best_svs_nobwn['acc']:
                best_svs_nobwn['acc'] = accs_nobwn['Average']
                best_svs_nobwn['rank'] = r
                best_svs_nobwn['lambda'] = l
                best_svs_nobwn['results'] = accs_nobwn
                
            print(f"Rank={r:3d} | Lambda={l:.1f} | SVS+BWN: {accs_bwn['Average']:.2f}% | SVS no BWN: {accs_nobwn['Average']:.2f}%")
            
    print(f"\nBest SVS+BWN Setup: Rank={best_svs_bwn['rank']}, Lambda={best_svs_bwn['lambda']} | Avg Acc: {best_svs_bwn['acc']:.2f}%")
    print(f"Best SVS No BWN Setup: Rank={best_svs_nobwn['rank']}, Lambda={best_svs_nobwn['lambda']} | Avg Acc: {best_svs_nobwn['acc']:.2f}%")

    # 4. Sweep over DARE threshold p and lambda
    print("\n--- Sweeping DARE dropout p and lambda ---")
    dare_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    dare_results = {}
    best_dare = {'p': 0.1, 'lambda': 0.3, 'acc': 0.0, 'results': None}
    
    for p in dare_probs:
        dare_results[p] = {}
        for l in lambda_sweep:
            lambdas = [l] * len(tasks)
            merged_model_dare = merge_full_network(base_model, expert_dicts, lambdas, method="dare", p=p)
            accs_dare = evaluate_merged_model_multi_task(merged_model_dare)
            dare_results[p][l] = accs_dare
            
            if accs_dare['Average'] > best_dare['acc']:
                best_dare['acc'] = accs_dare['Average']
                best_dare['p'] = p
                best_dare['lambda'] = l
                best_dare['results'] = accs_dare
                
        print(f"DARE p={p:.1f} | Best Avg Acc: {max(dare_results[p][l]['Average'] for l in lambda_sweep):.2f}%")
        
    print(f"Best DARE Setup: p={best_dare['p']}, Lambda={best_dare['lambda']} | Avg Acc: {best_dare['acc']:.2f}%")

    # 5. Sweep over TIES trim fraction K and lambda
    print("\n--- Sweeping TIES trim fraction K and lambda ---")
    ties_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    ties_results = {}
    best_ties = {'K': 0.1, 'lambda': 0.3, 'acc': 0.0, 'results': None}
    
    for K in ties_fractions:
        ties_results[K] = {}
        for l in lambda_sweep:
            lambdas = [l] * len(tasks)
            merged_model_ties = merge_full_network(base_model, expert_dicts, lambdas, method="ties", K=K)
            accs_ties = evaluate_merged_model_multi_task(merged_model_ties)
            ties_results[K][l] = accs_ties
            
            if accs_ties['Average'] > best_ties['acc']:
                best_ties['acc'] = accs_ties['Average']
                best_ties['K'] = K
                best_ties['lambda'] = l
                best_ties['results'] = accs_ties
                
        print(f"TIES K={K:.1f} | Best Avg Acc: {max(ties_results[K][l]['Average'] for l in lambda_sweep):.2f}%")
        
    print(f"Best TIES Setup: K={best_ties['K']}, Lambda={best_ties['lambda']} | Avg Acc: {best_ties['acc']:.2f}%")

    # Generate figures
    # Figure 1: Accuracy vs Lambda for Task Arithmetic vs SVS (at best rank k=64)
    plt.figure(figsize=(10, 6))
    ta_lambdas_plot = list(ta_results.keys())
    ta_accs_plot = [ta_results[l]['Average'] for l in ta_lambdas_plot]
    
    plt.plot(ta_lambdas_plot, ta_accs_plot, label="Task Arithmetic (TA)", color='black', marker='o', linewidth=2)
    
    colors_dict = {16: 'red', 64: 'blue', 256: 'green'}
    for r in [16, 64, 256]:
        svs_accs_plot = [svs_bwn_results[r][l]['Average'] for l in lambda_sweep]
        plt.plot(lambda_sweep, svs_accs_plot, label=f"SVS + BWN (Rank {r})", color=colors_dict[r], marker='s', linestyle='--')
        
    plt.title("Multi-Task Averaged Accuracy vs. Scaling Coefficient $\lambda$", fontsize=14)
    plt.xlabel("Scaling Coefficient $\lambda$", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/fig1_acc_vs_lambda.png", dpi=150)
    plt.close()
    
    # Figure 2: Ablation Study - SVS (with BWN) vs SVS (without BWN) across Ranks k (at best lambda)
    plt.figure(figsize=(10, 6))
    best_l = best_svs_bwn['lambda']
    bwn_accs_vs_rank = [svs_bwn_results[r][best_l]['Average'] for r in ranks]
    nobwn_accs_vs_rank = [svs_nobwn_results[r][best_l]['Average'] for r in ranks]
    
    plt.plot(ranks, bwn_accs_vs_rank, label="SVS + BWN (Scale Preserved)", color='blue', marker='o', linewidth=2)
    plt.plot(ranks, nobwn_accs_vs_rank, label="SVS without BWN (Raw Slicing)", color='red', marker='x', linestyle='--', linewidth=2)
    plt.axhline(y=ta_results[best_l]['Average'], color='black', linestyle=':', label=f"Task Arithmetic Baseline (at $\lambda$={best_l:.1f})")
    
    plt.title(f"Ablation Study: SVS with and without BWN vs. Rank $k$ (at $\lambda$={best_l:.1f})", fontsize=14)
    plt.xlabel("SVS Rank $k$", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(ranks, [str(r) for r in ranks])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/fig2_ablation_bwn.png", dpi=150)
    plt.close()
    
    # Figure 3: Task-specific Breakdown Comparison (Best TA vs Best SVS+BWN vs Zero-shot)
    fig, ax = plt.subplots(figsize=(10, 6))
    task_names = [t['name'] for t in tasks]
    x_coords = np.arange(len(task_names))
    width = 0.25
    
    zs_vals = [zero_shot_accs[t] for t in task_names]
    ta_vals = [ta_results[best_ta_lambda][t] for t in task_names]
    svs_vals = [best_svs_bwn['results'][t] for t in task_names]
    
    ax.bar(x_coords - width, zs_vals, width, label='Zero-Shot Base', color='gray', alpha=0.7)
    ax.bar(x_coords, ta_vals, width, label=f'Task Arithmetic ($\lambda$={best_ta_lambda:.1f})', color='darkorange', alpha=0.8)
    ax.bar(x_coords + width, svs_vals, width, label=f"SVS+BWN (Rank={best_svs_bwn['rank']}, $\lambda$={best_svs_bwn['lambda']:.1f})", color='royalblue')
    
    ax.set_title("Task-Specific Accuracy Comparison across Merging Frameworks", fontsize=14)
    ax.set_xlabel("Dataset / Task", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x_coords)
    ax.set_xticklabels(task_names)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/fig3_task_comparison.png", dpi=150)
    plt.close()

    # Save results to json
    results_data = {
        'zero_shot': zero_shot_accs,
        'individual_experts': expert_individual_accs,
        'best_ta': {'lambda': best_ta_lambda, 'results': ta_results[best_ta_lambda]},
        'best_svs_bwn': {'rank': best_svs_bwn['rank'], 'lambda': best_svs_bwn['lambda'], 'results': best_svs_bwn['results']},
        'best_svs_nobwn': {'rank': best_svs_nobwn['rank'], 'lambda': best_svs_nobwn['lambda'], 'results': best_svs_nobwn['results']},
        'best_dare': {'p': best_dare['p'], 'lambda': best_dare['lambda'], 'results': best_dare['results']},
        'best_ties': {'K': best_ties['K'], 'lambda': best_ties['lambda'], 'results': best_ties['results']},
        'all_ta': ta_results,
        'all_svs_bwn': svs_bwn_results,
        'all_svs_nobwn': svs_nobwn_results,
        'all_dare': dare_results,
        'all_ties': ties_results
    }
    
    with open("results/metrics_summary.json", "w") as f:
        def serialize(obj):
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64, torch.Tensor)):
                return float(obj.item() if hasattr(obj, 'item') else obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        json.dump(serialize(results_data), f, indent=2)
        
    print("\nAll experiments complete and results saved.")

if __name__ == "__main__":
    main()
