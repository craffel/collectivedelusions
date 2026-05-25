import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

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

def merge_backbones(pretrained_state, expert_states, lam=0.3):
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

def adapt_head_vanilla(head, cal_feats, cal_targets, epochs=100, lr=1e-2):
    head = copy.deepcopy(head).to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
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

def adapt_head_sam(head, cal_feats, cal_targets, epochs=100, lr=1e-2, rho=0.05):
    head = copy.deepcopy(head).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
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
            outputs_perturbed = head(feats)
            loss_perturbed = kl_loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            if grad_norm > 0:
                for p, g in zip(params, grads):
                    p.data.sub_(g * scale)
                    
            optimizer.step()
            optimizer.zero_grad()
    return head

def compute_sharpness(head, cal_feats, cal_targets):
    head.eval()
    cal_feats = cal_feats.to(device)
    cal_targets = cal_targets.to(device)
    
    # 1. Base Loss
    with torch.no_grad():
        base_outputs = head(cal_feats)
        base_loss = kl_loss_fn(base_outputs, cal_targets).item()
        
    # 2. Adversarial Sharpness (at different rho_eval)
    adv_sharpness = {}
    for rho_eval in [0.01, 0.05, 0.1]:
        # Compute gradient at current parameters to determine adversarial direction
        temp_head = copy.deepcopy(head)
        # Enable grads
        for p in temp_head.parameters():
            p.requires_grad = True
        temp_head.zero_grad()
        outputs = temp_head(cal_feats)
        loss = kl_loss_fn(outputs, cal_targets)
        loss.backward()
        
        params = [p for p in temp_head.parameters() if p.requires_grad]
        grads = [p.grad.clone() for p in params if p.grad is not None]
        grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
        
        if grad_norm > 0:
            scale = rho_eval / grad_norm
            # Move in gradient direction (maximizing loss)
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.data.add_(g * scale)
                adv_outputs = temp_head(cal_feats)
                adv_loss = kl_loss_fn(adv_outputs, cal_targets).item()
                adv_sharpness[rho_eval] = adv_loss - base_loss
        else:
            adv_sharpness[rho_eval] = 0.0
            
    # 3. Random Sharpness (Gaussian perturbations with different sigmas)
    rand_sharpness = {}
    torch.manual_seed(42)
    np.random.seed(42)
    
    for sigma in [0.01, 0.05, 0.1]:
        losses = []
        for _ in range(20): # Average over 20 random trials
            perturbed_head = copy.deepcopy(head)
            with torch.no_grad():
                for p in perturbed_head.parameters():
                    noise = torch.randn_like(p) * sigma
                    p.data.add_(noise)
                perturbed_outputs = perturbed_head(cal_feats)
                losses.append(kl_loss_fn(perturbed_outputs, cal_targets).item())
        rand_sharpness[sigma] = np.mean(losses) - base_loss
        
    return base_loss, adv_sharpness, rand_sharpness

def main():
    print("Measuring Loss Landscape Sharpness...")
    tasks = ["cifar10", "svhn", "fashion_mnist"]
    
    pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_state = pretrained_model.state_dict()
    
    expert_states = {}
    expert_heads = {}
    for task in tasks:
        ckpt_path = f"checkpoints/expert_{task}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        expert_states[task] = ckpt['state_dict']
        
        head = nn.Linear(512, 10)
        head.weight.data.copy_(ckpt['state_dict']['fc.weight'])
        head.bias.data.copy_(ckpt['state_dict']['fc.bias'])
        expert_heads[task] = head

    # Merge at lam = 0.3
    lam = 0.3
    merged_state = merge_backbones(pretrained_state, expert_states, lam)
    
    merged_model = resnet18()
    merged_model.fc = nn.Identity()
    merged_model.load_state_dict({k: v for k, v in merged_state.items() if not k.startswith('fc.')}, strict=False)
    merged_model = merged_model.to(device)
    merged_model.eval()

    print("Loading calibration datasets...")
    results = {}
    
    for task in tasks:
        print(f"\n=== Processing task: {task} ===")
        train_dataset = get_dataset(task, train=True)
        
        # Deterministic subset of 1000 calibration samples
        np.random.seed(42)
        cal_indices = np.random.choice(len(train_dataset), 1000, replace=False)
        cal_subset = Subset(train_dataset, cal_indices)
        cal_loader = DataLoader(cal_subset, batch_size=64, shuffle=False, num_workers=2)
        
        # Precompute expert targets
        exp_model = resnet18()
        exp_model.fc = nn.Linear(512, 10)
        exp_model.load_state_dict(expert_states[task])
        exp_model = exp_model.to(device)
        exp_model.eval()
        
        targets = []
        with torch.no_grad():
            for images, _ in cal_loader:
                images = images.to(device)
                outputs = exp_model(images)
                targets.append(outputs.cpu())
        cal_targets = torch.cat(targets, dim=0)
        
        # Extract features
        cal_feats, _ = get_features(merged_model, cal_loader)
        
        # Original head
        orig_head = expert_heads[task]
        
        # 1. Adapt via SyMerge
        print("Adapting via SyMerge...")
        head_sy = adapt_head_vanilla(orig_head, cal_feats, cal_targets, epochs=100, lr=1e-2)
        
        # 2. Adapt via SA-SyMerge
        print("Adapting via SA-SyMerge...")
        head_sa_sy = adapt_head_sam(orig_head, cal_feats, cal_targets, epochs=100, lr=1e-2, rho=0.05)
        
        # 3. Compute Sharpness
        print("Computing sharpness measures...")
        sy_base, sy_adv, sy_rand = compute_sharpness(head_sy, cal_feats, cal_targets)
        sa_base, sa_adv, sa_rand = compute_sharpness(head_sa_sy, cal_feats, cal_targets)
        
        results[task] = {
            'symerge': {
                'base_loss': sy_base,
                'adv_sharpness': sy_adv,
                'rand_sharpness': sy_rand
            },
            'sa_symerge': {
                'base_loss': sa_base,
                'adv_sharpness': sa_adv,
                'rand_sharpness': sa_rand
            }
        }
        
        print(f"SyMerge:   Base Loss = {sy_base:.4f} | Adv Sharp (rho=0.05) = {sy_adv[0.05]:.4f} | Rand Sharp (sigma=0.05) = {sy_rand[0.05]:.4f}")
        print(f"SA-SyMerge: Base Loss = {sa_base:.4f} | Adv Sharp (rho=0.05) = {sa_adv[0.05]:.4f} | Rand Sharp (sigma=0.05) = {sa_rand[0.05]:.4f}")

    # Save results
    torch.save(results, "sharpness_results.pt")
    print("\nSharpness results saved to sharpness_results.pt!")

if __name__ == "__main__":
    main()
