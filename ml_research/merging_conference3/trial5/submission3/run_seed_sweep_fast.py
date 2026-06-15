import sys
sys.path.insert(0, '/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial5/submission3/my_libs')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
import os
import numpy as np
import copy
import math
import random

# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=mean, std=std)
])

transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Task Names
tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']

def get_datasets():
    train_datasets = {
        'MNIST': torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray),
        'FashionMNIST': torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray),
        'CIFAR10': torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_color),
        'SVHN': torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform_color)
    }
    test_datasets = {
        'MNIST': torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray),
        'FashionMNIST': torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray),
        'CIFAR10': torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color),
        'SVHN': torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_color)
    }
    return train_datasets, test_datasets

def get_backbone_and_head(model_path):
    sd = torch.load(model_path, map_location='cpu')
    backbone_sd = {k: v for k, v in sd.items() if not k.startswith('head.')}
    head_sd = {k.replace('head.', ''): v for k, v in sd.items() if k.startswith('head.')}
    return backbone_sd, head_sd

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(4, 192))
        self.b = nn.Parameter(torch.zeros(4))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
    def forward(self, x, temperature=1.0):
        logits = F.linear(x, self.W, self.b)
        coeffs = F.softmax(logits / temperature, dim=-1)
        return coeffs

class ViTBackboneWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        feats = self.model.forward_features(x)
        return self.model.forward_head(feats, pre_logits=True)

def main():
    train_datasets, test_datasets = get_datasets()
    
    # Load base model
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    base_backbone_sd = {k: v for k, v in base_model.state_dict().items() if not k.startswith('head.')}
    
    # Load expert weights and heads
    expert_paths = {task: f"./checkpoints/expert_{task}.pth" for task in tasks}
    expert_backbones = {}
    heads = {}
    for task in tasks:
        backbone_sd, head_sd = get_backbone_and_head(expert_paths[task])
        expert_backbones[task] = backbone_sd
        
        head = nn.Linear(192, 10)
        head.load_state_dict(head_sd)
        head = head.to(device)
        head.eval()
        heads[task] = head
        
    # Calculate task vectors
    task_vectors = {}
    for task in tasks:
        task_vectors[task] = {}
        for k in base_backbone_sd.keys():
            task_vectors[task][k] = expert_backbones[task][k] - base_backbone_sd[k]

    # Pre-send task vectors to device
    task_vectors_dev = {t: {k: v.to(device) for k, v in task_vectors[t].items()} for t in tasks}
    base_backbone_dev = {k: v.to(device) for k, v in base_backbone_sd.items()}
    
    # Setup backbone wrapper
    eval_model_base = timm.create_model('vit_tiny_patch16_224', pretrained=False).to(device)
    wrapper = ViTBackboneWrapper(eval_model_base).to(device)
    base_model = base_model.to(device)

    # Subsample 500 test samples per task for super fast evaluation
    test_subsets = {}
    for task in tasks:
        dataset = test_datasets[task]
        set_seed(42) # fixed test subset for fair comparison
        indices = torch.randperm(len(dataset))[:500]
        imgs, lbls = [], []
        for idx in indices:
            img, lbl = dataset[idx]
            imgs.append(img.unsqueeze(0))
            lbls.append(lbl)
        test_subsets[task] = (torch.cat(imgs, dim=0).to(device), torch.tensor(lbls).to(device))

    # Helper evaluation function using subset
    def evaluate_dynamic_homogeneous(router, temp=1.0):
        results = {}
        with torch.no_grad():
            for task in tasks:
                imgs, labels = test_subsets[task]
                # Process in small batches of 250 to avoid GPU memory overflow
                correct = 0
                total = imgs.size(0)
                for i in range(0, total, 250):
                    batch_imgs = imgs[i:i+250]
                    batch_labels = labels[i:i+250]
                    
                    router_in = base_model.patch_embed(batch_imgs).mean(dim=1)
                    coeffs = router(router_in, temperature=temp)
                    batch_coeffs = coeffs.mean(dim=0)
                    
                    dynamic_params = {}
                    for key in base_backbone_sd.keys():
                        dynamic_params[f"model.{key}"] = base_backbone_dev[key] + sum(
                            batch_coeffs[k] * task_vectors_dev[tasks[k]][key] for k in range(4)
                        )
                    pooled = torch.func.functional_call(wrapper, dynamic_params, (batch_imgs,))
                    outputs = heads[task](pooled)
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == batch_labels).sum().item()
                results[task] = correct / total
        return results

    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    
    lr_results = []
    rlr_results = []
    
    print("Starting fast seed sweep...")
    for seed in seeds:
        # Sample few-shot calibration set for this seed
        calib_images = {}
        calib_labels = {}
        for task in tasks:
            dataset = train_datasets[task]
            set_seed(seed)
            indices = torch.randperm(len(dataset))[:16]
            imgs, lbls = [], []
            for idx in indices:
                img, lbl = dataset[idx]
                imgs.append(img.unsqueeze(0))
                lbls.append(lbl)
            calib_images[task] = torch.cat(imgs, dim=0).to(device)
            calib_labels[task] = torch.tensor(lbls).to(device)
            
        calib_router_inputs = {}
        for task in tasks:
            with torch.no_grad():
                calib_router_inputs[task] = base_model.patch_embed(calib_images[task]).mean(dim=1)
                
        # 1. Train Linear Router (Unregularized)
        set_seed(seed)
        router_lr = Router().to(device)
        optimizer_lr = torch.optim.Adam(router_lr.parameters(), lr=0.01)
        for step in range(100):
            optimizer_lr.zero_grad()
            loss = 0.0
            for t_idx, task in enumerate(tasks):
                inputs = calib_router_inputs[task]
                coeffs = router_lr(inputs, temperature=1.0)
                batch_coeffs = coeffs.mean(dim=0)
                
                dynamic_params = {}
                for key in base_backbone_sd.keys():
                    dynamic_params[f"model.{key}"] = base_backbone_dev[key] + sum(
                        batch_coeffs[k] * task_vectors_dev[tasks[k]][key] for k in range(4)
                    )
                pooled = torch.func.functional_call(wrapper, dynamic_params, (calib_images[task],))
                logits = heads[task](pooled)
                loss += F.cross_entropy(logits, calib_labels[task])
            loss.backward()
            optimizer_lr.step()
            
        lr_accs = evaluate_dynamic_homogeneous(router_lr, temp=1.0)
        lr_mean = np.mean([lr_accs[t] for t in tasks])
        print(f"Seed {seed:03d} | Linear Router - SVHN: {lr_accs['SVHN']:.4f}, Joint Mean: {lr_mean:.4f}")
        lr_results.append((lr_accs['SVHN'], lr_mean))
        
        # 2. Train Robust Linear Routing (RLR)
        set_seed(seed)
        router_rlr = Router().to(device)
        optimizer_rlr = torch.optim.Adam(router_rlr.parameters(), lr=0.01)
        
        # Use unweighted uniform multi-task calibration loss (per Section 3.3 of the paper)
        task_weights = np.ones(len(tasks))
        
        alpha = 0.005
        for step in range(100):
            optimizer_rlr.zero_grad()
            loss = 0.0
            for t_idx, task in enumerate(tasks):
                inputs = calib_router_inputs[task]
                coeffs = router_rlr(inputs, temperature=2.0)
                batch_coeffs = coeffs.mean(dim=0)
                
                dynamic_params = {}
                for key in base_backbone_sd.keys():
                    dynamic_params[f"model.{key}"] = base_backbone_dev[key] + sum(
                        batch_coeffs[k] * task_vectors_dev[tasks[k]][key] for k in range(4)
                    )
                pooled = torch.func.functional_call(wrapper, dynamic_params, (calib_images[task],))
                logits = heads[task](pooled)
                loss += task_weights[t_idx] * F.cross_entropy(logits, calib_labels[task])
            
            l2_reg = alpha * torch.sum(router_rlr.W ** 2)
            loss += l2_reg
            loss.backward()
            optimizer_rlr.step()
            
        rlr_accs = evaluate_dynamic_homogeneous(router_rlr, temp=2.0)
        rlr_mean = np.mean([rlr_accs[t] for t in tasks])
        print(f"Seed {seed:03d} | RLR           - SVHN: {rlr_accs['SVHN']:.4f}, Joint Mean: {rlr_mean:.4f}")
        rlr_results.append((rlr_accs['SVHN'], rlr_mean))

    print("\n--- RESULTS OVER 10 SEEDS ---")
    lr_svhns = [x[0] for x in lr_results]
    lr_means = [x[1] for x in lr_results]
    rlr_svhns = [x[0] for x in rlr_results]
    rlr_means = [x[1] for x in rlr_results]
    
    print(f"Linear Router SVHN: {np.mean(lr_svhns)*100:.2f}% +/- {np.std(lr_svhns)*100:.2f}% (Min: {np.min(lr_svhns)*100:.2f}%, Max: {np.max(lr_svhns)*100:.2f}%)")
    print(f"Linear Router Mean: {np.mean(lr_means)*100:.2f}% +/- {np.std(lr_means)*100:.2f}% (Min: {np.min(lr_means)*100:.2f}%, Max: {np.max(lr_means)*100:.2f}%)")
    print(f"RLR SVHN:           {np.mean(rlr_svhns)*100:.2f}% +/- {np.std(rlr_svhns)*100:.2f}% (Min: {np.min(rlr_svhns)*100:.2f}%, Max: {np.max(rlr_svhns)*100:.2f}%)")
    print(f"RLR Mean:           {np.mean(rlr_means)*100:.2f}% +/- {np.std(rlr_means)*100:.2f}% (Min: {np.min(rlr_means)*100:.2f}%, Max: {np.max(rlr_means)*100:.2f}%)")

if __name__ == '__main__':
    main()
