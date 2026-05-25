import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

def get_transforms(task):
    if task == 'fashion_mnist':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataset(task, train=True):
    transform = get_transforms(task)
    if task == 'cifar10':
        return datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif task == 'svhn':
        split = 'train' if train else 'test'
        return datasets.SVHN(root='./data', split=split, download=True, transform=transform)
    elif task == 'fashion_mnist':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")

# Helper to extract ResNet18 backbone features
class ResNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def merge_backbones(pretrained_state, expert_states, lam):
    # Perform Task Arithmetic merging of the backbone weights
    # expert_states is a dict of task_name -> state_dict
    merged_state = copy.deepcopy(pretrained_state)
    
    # Identify backbone keys (keys that don't belong to 'fc')
    backbone_keys = [k for k in pretrained_state.keys() if not k.startswith('fc.')]
    
    for k in backbone_keys:
        delta_sum = torch.zeros_like(pretrained_state[k]).float()
        for task, state in expert_states.items():
            delta = state[k].float() - pretrained_state[k].float()
            delta_sum += delta
        merged_state[k] = pretrained_state[k].float() + lam * delta_sum
        # Cast back to original type
        merged_state[k] = merged_state[k].to(pretrained_state[k].dtype)
        
    return merged_state

def merge_backbones_ties(pretrained_state, expert_states, lam, keep_ratio=0.2):
    merged_state = copy.deepcopy(pretrained_state)
    backbone_keys = [k for k in pretrained_state.keys() if not k.startswith('fc.')]
    
    for k in backbone_keys:
        param_shape = pretrained_state[k].shape
        if len(param_shape) == 0:
            continue
            
        deltas = []
        for task, state in expert_states.items():
            delta = state[k].float() - pretrained_state[k].float()
            deltas.append(delta)
            
        deltas_stacked = torch.stack(deltas, dim=0)
        
        # 1. Trimming (Keep only top-k% by magnitude for each task vector)
        flat_deltas = deltas_stacked.view(len(expert_states), -1)
        num_params = flat_deltas.shape[1]
        k_count = int(keep_ratio * num_params)
        if k_count > 0 and k_count < num_params:
            for t_idx in range(len(expert_states)):
                abs_val = torch.abs(flat_deltas[t_idx])
                threshold = torch.topk(abs_val, k_count).values[-1]
                flat_deltas[t_idx][abs_val < threshold] = 0.0
                
        # 2. Sign consensus
        signs = torch.sign(flat_deltas)
        sign_sum = torch.sum(signs, dim=0)
        consensus_sign = torch.sign(sign_sum)
        
        # 3. Discard task vectors that disagree with consensus sign
        agree_mask = (signs == consensus_sign.unsqueeze(0)) & (consensus_sign.unsqueeze(0) != 0)
        flat_deltas[~agree_mask] = 0.0
        
        # 4. Average remaining updates
        non_zero_count = torch.sum((flat_deltas != 0).float(), dim=0)
        sum_deltas = torch.sum(flat_deltas, dim=0)
        avg_deltas = torch.where(non_zero_count > 0, sum_deltas / non_zero_count, torch.zeros_like(sum_deltas))
        
        # 5. Add back to pre-trained
        merged_flat = pretrained_state[k].float().view(-1) + lam * avg_deltas
        merged_state[k] = merged_flat.view(param_shape).to(pretrained_state[k].dtype)
        
    return merged_state

def merge_backbones_dare(pretrained_state, expert_states, lam, drop_rate=0.5):
    merged_state = copy.deepcopy(pretrained_state)
    backbone_keys = [k for k in pretrained_state.keys() if not k.startswith('fc.')]
    
    torch.manual_seed(42)
    
    for k in backbone_keys:
        param_shape = pretrained_state[k].shape
        if len(param_shape) == 0:
            continue
            
        delta_sum = torch.zeros_like(pretrained_state[k]).float()
        for task, state in expert_states.items():
            delta = state[k].float() - pretrained_state[k].float()
            mask = (torch.rand_like(delta) > drop_rate).float()
            scaled_delta = (delta * mask) / (1.0 - drop_rate)
            delta_sum += scaled_delta
            
        merged_state[k] = pretrained_state[k].float() + lam * delta_sum
        merged_state[k] = merged_state[k].to(pretrained_state[k].dtype)
        
    return merged_state

def get_features(model, loader):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def kl_loss_fn(outputs, targets, T=2.0):
    # KL-divergence loss for soft labels
    p = nn.functional.log_softmax(outputs / T, dim=1)
    q = nn.functional.softmax(targets / T, dim=1)
    return nn.functional.kl_div(p, q, reduction='batchmean') * (T ** 2)

def adapt_head_vanilla(head, cal_feats, cal_targets, epochs=50, lr=1e-2, weight_decay=1e-4):
    head = copy.deepcopy(head).to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    return head

def adapt_head_sam(head, cal_feats, cal_targets, epochs=50, lr=1e-2, weight_decay=1e-4, rho=0.05):
    head = copy.deepcopy(head).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            
            # Step 1: First forward-backward pass to find adversarial perturbation
            head.train()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            
            # Retrieve parameters and gradients
            params = [p for p in head.parameters() if p.requires_grad]
            grads = [p.grad.clone() for p in params if p.grad is not None]
            
            grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
            
            if grad_norm > 0:
                scale = rho / grad_norm
                # Apply perturbation
                for p, g in zip(params, grads):
                    p.data.add_(g * scale)
                    
            # Zero gradients for second pass
            optimizer.zero_grad()
            
            # Step 2: Second forward-backward pass at the perturbed point
            outputs_perturbed = head(feats)
            loss_perturbed = kl_loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            # Restore original weights before optimizer step
            if grad_norm > 0:
                for p, g in zip(params, grads):
                    p.data.sub_(g * scale)
                    
            optimizer.step()
            optimizer.zero_grad()
            
    return head

def main():
    parser = argparse.ArgumentParser(description="Run model merging and single-head adaptation experiments")
    parser.add_argument("--cal-size", type=int, default=500, help="Calibration subset size per task")
    parser.add_argument("--epochs", type=int, default=50, help="Adaptation epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adaptation learning rate")
    parser.add_argument("--rho", type=float, default=0.05, help="SAM perturbation size")
    parser.add_argument("--merge-method", type=str, default="task_arithmetic", choices=["task_arithmetic", "ties", "dare"], help="Backbone merging method")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    print(f"Loading pre-trained ResNet18 and experts...")
    tasks = ["cifar10", "svhn", "fashion_mnist"]
    
    # 1. Load Pretrained ResNet18
    pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_state = pretrained_model.state_dict()
    
    # 2. Load Expert Models
    expert_states = {}
    expert_heads = {}
    for task in tasks:
        ckpt_path = f"checkpoints/expert_{task}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint {ckpt_path} not found. Please train experts first.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        expert_states[task] = ckpt['state_dict']
        
        # Extract classification head
        head = nn.Linear(512, 10)
        # Load only fc weights from expert
        fc_weight = ckpt['state_dict']['fc.weight']
        fc_bias = ckpt['state_dict']['fc.bias']
        head.weight.data.copy_(fc_weight)
        head.bias.data.copy_(fc_bias)
        expert_heads[task] = head

    # 3. Load Datasets
    print("Preparing datasets...")
    cal_loaders = {}
    test_loaders = {}
    
    for task in tasks:
        train_dataset = get_dataset(task, train=True)
        test_dataset = get_dataset(task, train=False)
        
        if args.smoke_test:
            cal_subset = Subset(train_dataset, range(64))
            test_subset = Subset(test_dataset, range(64))
        else:
            # Deterministic subset for calibration
            np.random.seed(42)
            cal_indices = np.random.choice(len(train_dataset), args.cal_size, replace=False)
            cal_subset = Subset(train_dataset, cal_indices)
            test_subset = test_dataset
            
        cal_loaders[task] = DataLoader(cal_subset, batch_size=64, shuffle=False, num_workers=2)
        test_loaders[task] = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)

    # 4. Extract features under original experts
    print("Pre-computing calibration soft targets from expert models...")
    expert_cal_targets = {}
    for task in tasks:
        # Load full expert model
        exp_model = resnet18()
        exp_model.fc = nn.Linear(512, 10)
        exp_model.load_state_dict(expert_states[task])
        exp_model = exp_model.to(device)
        exp_model.eval()
        
        targets = []
        with torch.no_grad():
            for images, _ in cal_loaders[task]:
                images = images.to(device)
                outputs = exp_model(images)
                targets.append(outputs.cpu())
        expert_cal_targets[task] = torch.cat(targets, dim=0)

    # 5. Sweep over merging coefficients (lambda)
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9] if not args.smoke_test else [0.5]
    
    results = {
        'naive': {task: [] for task in tasks},
        'symerge': {task: [] for task in tasks},
        'sa_symerge': {task: [] for task in tasks},
    }
    
    for lam in lambdas:
        print("-"*50)
        print(f"Evaluating lambda = {lam}")
        print("-"*50)
        
        # Merge backbones
        if args.merge_method == "task_arithmetic":
            merged_state = merge_backbones(pretrained_state, expert_states, lam)
        elif args.merge_method == "ties":
            merged_state = merge_backbones_ties(pretrained_state, expert_states, lam)
        elif args.merge_method == "dare":
            merged_state = merge_backbones_dare(pretrained_state, expert_states, lam)
        else:
            raise ValueError(f"Unknown merge method: {args.merge_method}")
        
        # Initialize merged backbone model
        merged_model = resnet18()
        merged_model.fc = nn.Identity() # Outputs backbone features directly
        merged_model.load_state_dict({k: v for k, v in merged_state.items() if not k.startswith('fc.')}, strict=False)
        merged_model = merged_model.to(device)
        merged_model.eval()
        
        for task in tasks:
            print(f"--- Processing task: {task} ---")
            # Extract features of calibration and test datasets under the merged backbone
            cal_feats, _ = get_features(merged_model, cal_loaders[task])
            test_feats, test_labels = get_features(merged_model, test_loaders[task])
            
            # Retrieve original expert head
            orig_head = expert_heads[task].to(device)
            orig_head.eval()
            
            # Evaluate Naive (Original head + Merged backbone)
            with torch.no_grad():
                naive_outputs = orig_head(test_feats.to(device))
                _, predicted = naive_outputs.max(1)
                naive_acc = predicted.eq(test_labels.to(device)).sum().item() / len(test_labels)
            results['naive'][task].append(naive_acc)
            print(f"Naive Acc: {naive_acc*100:.2f}%")
            
            # Adapt head via SyMerge (Vanilla optimization)
            print("Adapting head via SyMerge...")
            adapted_head_vanilla = adapt_head_vanilla(
                orig_head, cal_feats, expert_cal_targets[task], 
                epochs=args.epochs, lr=args.lr
            )
            adapted_head_vanilla.eval()
            with torch.no_grad():
                sy_outputs = adapted_head_vanilla(test_feats.to(device))
                _, predicted = sy_outputs.max(1)
                sy_acc = predicted.eq(test_labels.to(device)).sum().item() / len(test_labels)
            results['symerge'][task].append(sy_acc)
            print(f"SyMerge Acc: {sy_acc*100:.2f}%")
            
            # Adapt head via SA-SyMerge (Sharpness-Aware optimization)
            print("Adapting head via SA-SyMerge...")
            adapted_head_sam = adapt_head_sam(
                orig_head, cal_feats, expert_cal_targets[task], 
                epochs=args.epochs, lr=args.lr, rho=args.rho
            )
            adapted_head_sam.eval()
            with torch.no_grad():
                sa_sy_outputs = adapted_head_sam(test_feats.to(device))
                _, predicted = sa_sy_outputs.max(1)
                sa_sy_acc = predicted.eq(test_labels.to(device)).sum().item() / len(test_labels)
            results['sa_symerge'][task].append(sa_sy_acc)
            print(f"SA-SyMerge Acc: {sa_sy_acc*100:.2f}%")

    if not args.smoke_test:
        # Save results
        np.save(f"results_{args.merge_method}.npy", results)
        if args.merge_method == "task_arithmetic":
            np.save("results.npy", results)
        
        # Print Summary
        print("="*60)
        print(f"FINAL RESULTS SUMMARY FOR {args.merge_method.upper()} (Average Accuracies across tasks)")
        print("="*60)
        for i, lam in enumerate(lambdas):
            naive_avg = np.mean([results['naive'][t][i] for t in tasks])
            sy_avg = np.mean([results['symerge'][t][i] for t in tasks])
            sa_sy_avg = np.mean([results['sa_symerge'][t][i] for t in tasks])
            print(f"Lambda = {lam:.1f} | Naive Avg: {naive_avg*100:.2f}% | SyMerge Avg: {sy_avg*100:.2f}% | SA-SyMerge Avg: {sa_sy_avg*100:.2f}%")
            
        # Plot and save results
        plt.figure(figsize=(12, 4))
        for idx, task in enumerate(tasks):
            plt.subplot(1, 3, idx+1)
            plt.plot(lambdas, results['naive'][task], label='Naive', marker='o')
            plt.plot(lambdas, results['symerge'][task], label='SyMerge', marker='s')
            plt.plot(lambdas, results['sa_symerge'][task], label='SA-SyMerge (Ours)', marker='^')
            plt.title(f"{task.upper()} Accuracy ({args.merge_method})")
            plt.xlabel("Merging Coeff (lambda)")
            plt.ylabel("Accuracy")
            plt.grid(True)
            if idx == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig(f"merging_results_{args.merge_method}.png", dpi=300)
        if args.merge_method == "task_arithmetic":
            plt.savefig("merging_results.png", dpi=300)
        print(f"Plots saved to merging_results_{args.merge_method}.png")

if __name__ == "__main__":
    main()
