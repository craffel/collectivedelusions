import os
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
    merged_state = copy.deepcopy(pretrained_state)
    backbone_keys = [k for k in pretrained_state.keys() if not k.startswith('fc.')]
    for k in backbone_keys:
        delta_sum = torch.zeros_like(pretrained_state[k]).float()
        for task, state in expert_states.items():
            delta = state[k].float() - pretrained_state[k].float()
            delta_sum += delta
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
            
            # Step 1: First forward-backward pass
            head.train()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            
            params = [p for p in head.parameters() if p.requires_grad]
            grads = [p.grad.clone() for p in params if p.grad is not None]
            
            grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
            
            if grad_norm > 0:
                scale = rho / grad_norm
                for p, g in zip(params, grads):
                    p.data.add_(g * scale)
                    
            optimizer.zero_grad()
            
            # Step 2: Second forward-backward pass
            outputs_perturbed = head(feats)
            loss_perturbed = kl_loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            if grad_norm > 0:
                for p, g in zip(params, grads):
                    p.data.sub_(g * scale)
                    
            optimizer.step()
            optimizer.zero_grad()
            
    return head

def main():
    print("Loading pre-trained ResNet18 and experts for noise robustness study...")
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
            raise FileNotFoundError(f"Expert checkpoint {ckpt_path} not found.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        expert_states[task] = ckpt['state_dict']
        
        head = nn.Linear(512, 10)
        head.weight.data.copy_(ckpt['state_dict']['fc.weight'])
        head.bias.data.copy_(ckpt['state_dict']['fc.bias'])
        expert_heads[task] = head

    # 3. Load Datasets
    print("Preparing datasets...")
    cal_loaders = {}
    test_loaders = {}
    
    for task in tasks:
        train_dataset = get_dataset(task, train=True)
        test_dataset = get_dataset(task, train=False)
        
        # Deterministic subset for calibration (size = 1000)
        np.random.seed(42)
        cal_indices = np.random.choice(len(train_dataset), 1000, replace=False)
        cal_subset = Subset(train_dataset, cal_indices)
        
        cal_loaders[task] = DataLoader(cal_subset, batch_size=64, shuffle=False, num_workers=2)
        test_loaders[task] = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 4. Extract features under original experts
    print("Pre-computing calibration soft targets from expert models...")
    expert_cal_targets = {}
    for task in tasks:
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

    # 5. Merge backbone at lambda = 0.3 (the optimal lambda)
    print("Merging backbones at lambda = 0.3 using Task Arithmetic...")
    merged_state = merge_backbones(pretrained_state, expert_states, lam=0.3)
    merged_model = resnet18()
    merged_model.fc = nn.Identity()
    merged_model.load_state_dict({k: v for k, v in merged_state.items() if not k.startswith('fc.')}, strict=False)
    merged_model = merged_model.to(device)
    merged_model.eval()

    # Pre-extract features under merged backbone
    print("Pre-extracting calibration and test features under the merged backbone...")
    cal_feats_dict = {}
    test_feats_dict = {}
    test_labels_dict = {}
    for task in tasks:
        cal_feats_dict[task], _ = get_features(merged_model, cal_loaders[task])
        test_feats_dict[task], test_labels_dict[task] = get_features(merged_model, test_loaders[task])

    # 6. Sweep noise level eta
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {
        'naive': {task: [] for task in tasks},
        'symerge': {task: [] for task in tasks},
        'sa_symerge': {task: [] for task in tasks}
    }

    T = 2.0
    C = 10

    for eta in noise_levels:
        print("-"*50)
        print(f"Evaluating soft-target noise level eta = {eta}")
        print("-"*50)
        
        for task in tasks:
            print(f"Task: {task}")
            # Evaluate Naive
            orig_head = expert_heads[task].to(device)
            orig_head.eval()
            with torch.no_grad():
                naive_outputs = orig_head(test_feats_dict[task].to(device))
                _, predicted = naive_outputs.max(1)
                naive_acc = predicted.eq(test_labels_dict[task].to(device)).sum().item() / len(test_labels_dict[task])
            results['naive'][task].append(naive_acc)
            print(f"  Naive Acc: {naive_acc*100:.2f}%")

            # Corrupt soft targets
            clean_targets = expert_cal_targets[task]
            q = torch.softmax(clean_targets / T, dim=-1)
            q_noisy = (1.0 - eta) * q + eta / C
            noisy_targets = T * torch.log(q_noisy + 1e-12)

            # Adapt head via SyMerge (Vanilla optimization)
            adapted_head_vanilla = adapt_head_vanilla(
                orig_head, cal_feats_dict[task], noisy_targets,
                epochs=100, lr=1e-2
            )
            adapted_head_vanilla.eval()
            with torch.no_grad():
                sy_outputs = adapted_head_vanilla(test_feats_dict[task].to(device))
                _, predicted = sy_outputs.max(1)
                sy_acc = predicted.eq(test_labels_dict[task].to(device)).sum().item() / len(test_labels_dict[task])
            results['symerge'][task].append(sy_acc)
            print(f"  SyMerge Acc: {sy_acc*100:.2f}%")

            # Adapt head via SA-SyMerge
            adapted_head_sam = adapt_head_sam(
                orig_head, cal_feats_dict[task], noisy_targets,
                epochs=100, lr=1e-2, rho=0.05
            )
            adapted_head_sam.eval()
            with torch.no_grad():
                sa_sy_outputs = adapted_head_sam(test_feats_dict[task].to(device))
                _, predicted = sa_sy_outputs.max(1)
                sa_sy_acc = predicted.eq(test_labels_dict[task].to(device)).sum().item() / len(test_labels_dict[task])
            results['sa_symerge'][task].append(sa_sy_acc)
            print(f"  SA-SyMerge Acc: {sa_sy_acc*100:.2f}%")

    # Save results
    np.save("noise_robustness_results.npy", results)
    print("Raw results saved to noise_robustness_results.npy")

    # Plot results
    plt.figure(figsize=(12, 4))
    for idx, task in enumerate(tasks):
        plt.subplot(1, 3, idx+1)
        plt.plot(noise_levels, results['naive'][task], label='Naive', linestyle='--', color='gray')
        plt.plot(noise_levels, results['symerge'][task], label='SyMerge', marker='s', color='tab:orange')
        plt.plot(noise_levels, results['sa_symerge'][task], label='SA-SyMerge (Ours)', marker='^', color='tab:green')
        plt.title(f"{task.upper()} (Robustness to Soft Noise)")
        plt.xlabel("Soft Noise Level (eta)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        if idx == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig("noise_robustness.png", dpi=300)
    print("Noise robustness plot saved to noise_robustness.png")

    # Print Summary
    print("="*60)
    print("SUMMARY OF AVERAGE PERFORMANCE UNDER NOISE")
    print("="*60)
    for i, eta in enumerate(noise_levels):
        naive_avg = np.mean([results['naive'][t][i] for t in tasks])
        sy_avg = np.mean([results['symerge'][t][i] for t in tasks])
        sa_sy_avg = np.mean([results['sa_symerge'][t][i] for t in tasks])
        diff = (sa_sy_avg - sy_avg) * 100
        print(f"Noise Eta = {eta:.1f} | Naive Avg: {naive_avg*100:.2f}% | SyMerge Avg: {sy_avg*100:.2f}% | SA-SyMerge Avg: {sa_sy_avg*100:.2f}% | Diff: +{diff:.2f}%")

if __name__ == "__main__":
    main()
