import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import open_clip
import matplotlib.pyplot as plt

# Helper functions for loading weights
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def softmax_entropy(x):
    return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1).mean()

# Linear CKA
def linear_cka(X, Y):
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    hsic_XY = torch.norm(torch.matmul(X_c.t(), Y_c), p='fro')**2
    hsic_XX = torch.norm(torch.matmul(X_c.t(), X_c), p='fro')**2
    hsic_YY = torch.norm(torch.matmul(Y_c.t(), Y_c), p='fro')**2
    cka = hsic_XY / (torch.sqrt(hsic_XX * hsic_YY) + 1e-8)
    return cka.item()

# Activation capture hook
activations = {}
def get_activation(name):
    def hook(model, input, output):
        # average over seq_len (dim 0)
        activations[name] = output.detach().mean(dim=0).cpu()
    return hook

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Pre-transforms
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(clip_mean, clip_std)
    ])
    
    print("Loading datasets...")
    datasets_train = {
        "MNIST": torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform_gray),
        "FashionMNIST": torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform_gray),
        "CIFAR10": torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_rgb),
        "SVHN": torchvision.datasets.SVHN("./data", split="train", download=True, transform=transform_rgb)
    }
    datasets_test = {
        "MNIST": torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform_gray),
        "FashionMNIST": torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform_gray),
        "CIFAR10": torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_rgb),
        "SVHN": torchvision.datasets.SVHN("./data", split="test", download=True, transform=transform_rgb)
    }
    
    task_names = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    
    # Draw unique subsets (seed 42)
    train_subsets = {}
    cal_subsets = {}
    test_subsets = {}
    
    for task in task_names:
        train_ds = datasets_train[task]
        test_ds = datasets_test[task]
        
        rng_train = np.random.default_rng(seed)
        indices_train = np.arange(len(train_ds))
        rng_train.shuffle(indices_train)
        train_subsets[task] = Subset(train_ds, indices_train[:512])
        cal_subsets[task] = Subset(train_ds, indices_train[512:512+64])
        
        rng_test = np.random.default_rng(seed)
        indices_test = np.arange(len(test_ds))
        rng_test.shuffle(indices_test)
        test_subsets[task] = Subset(test_ds, indices_test[:512])
        
    print("Caching calibration and test subsets...")
    cal_loaders = {}
    test_loaders = {}
    
    for task in task_names:
        loader_temp = DataLoader(cal_subsets[task], batch_size=64, shuffle=False)
        imgs, lbls = [], []
        for img, lbl in loader_temp:
            imgs.append(img)
            lbls.append(lbl)
        cal_loaders[task] = DataLoader(torch.utils.data.TensorDataset(torch.cat(imgs, dim=0), torch.cat(lbls, dim=0)), batch_size=64, shuffle=False)
        
        loader_temp = DataLoader(test_subsets[task], batch_size=64, shuffle=False)
        imgs, lbls = [], []
        for img, lbl in loader_temp:
            imgs.append(img)
            lbls.append(lbl)
        test_loaders[task] = DataLoader(torch.utils.data.TensorDataset(torch.cat(imgs, dim=0), torch.cat(lbls, dim=0)), batch_size=64, shuffle=False)
        
    # Load base model
    base_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    base_model = base_model.to(device)
    
    # Load cached experts
    expert_encoders = {}
    expert_heads = {}
    task_vectors = {}
    
    for task in task_names:
        encoder_path = f"checkpoints/seed_{seed}_{task}_encoder.pt"
        head_path = f"checkpoints/seed_{seed}_{task}_head.pt"
        
        expert_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        expert_model = expert_model.to(device)
        classification_head = nn.Linear(512, 10).to(device)
        
        expert_model.visual.load_state_dict(torch.load(encoder_path, map_location=device))
        classification_head.load_state_dict(torch.load(head_path, map_location=device))
        
        expert_encoders[task] = expert_model.visual
        expert_heads[task] = classification_head
        
        # Build task vectors
        tv = {}
        base_state = base_model.visual.state_dict()
        expert_state = expert_model.visual.state_dict()
        for k in base_state.keys():
            if base_state[k].dtype.is_floating_point:
                tv[k] = expert_state[k] - base_state[k]
            else:
                tv[k] = expert_state[k].clone()
        task_vectors[task] = tv

    # Prepare merge model
    merge_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    merge_model = merge_model.to(device)
    pretrained_visual_dic = {k: v.detach().clone().to(device) for k, v in merge_model.visual.named_parameters()}
    orig_params, names = make_functional(merge_model.visual)
    
    paramslist = []
    paramslist.append(tuple(pretrained_visual_dic[k].to(device) for k in pretrained_visual_dic.keys()))
    for task in task_names:
        paramslist.append(tuple(task_vectors[task][k].detach().clone().to(device) for k in pretrained_visual_dic.keys()))
        
    def eval_loss(lambdas):
        num_params = len(paramslist[0])
        lambdas_expanded = lambdas.repeat(num_params, 1) if lambdas.shape[0] == 1 else lambdas
        merged_params = []
        for j, p in enumerate(zip(*paramslist)):
            param_merged = sum(pi * lambdas_expanded[j, i] for i, pi in enumerate(p))
            merged_params.append(param_merged)
        load_weights(merge_model.visual, names, merged_params)
        
        total_entropy = 0.0
        merge_model.visual.eval()
        for t in task_names:
            head = expert_heads[t]
            head.eval()
            task_entropy = 0.0
            count = 0
            for images, _ in cal_loaders[t]:
                images = images.to(device)
                features = merge_model.visual(images)
                logits = head(features)
                entropy = softmax_entropy(logits)
                task_entropy += entropy
                count += 1
            total_entropy += task_entropy / count
        return total_entropy

    # Optimize lambdas (Adam GD)
    print("Optimizing merging coefficients via test-time prediction entropy minimization...")
    lambdas_raw_gd = torch.full((len(paramslist[0]), 4), 0.3, device=device, requires_grad=True)
    optimizer_gd = torch.optim.Adam([lambdas_raw_gd], lr=1e-2)
    
    for step in range(200):
        optimizer_gd.zero_grad()
        task_lambdas = torch.clamp(lambdas_raw_gd, min=0.0, max=1.0)
        pretrain_lambdas = torch.ones(len(paramslist[0]), 1, device=device)
        lambdas_gd_step = torch.cat((pretrain_lambdas, task_lambdas), dim=1)
        loss = eval_loss(lambdas_gd_step)
        loss.backward()
        optimizer_gd.step()
        
    lambdas_gd = torch.cat((torch.ones(len(paramslist[0]), 1, device=device), torch.clamp(lambdas_raw_gd, min=0.0, max=1.0)), dim=1).detach()
    
    # Compute Spatial Averaging mean
    mean_tasks_gd_raw = lambdas_gd[:, 1:].mean(dim=0, keepdim=True)
    mean_tasks_gd = torch.cat((torch.ones(1, 1, device=device), mean_tasks_gd_raw), dim=1)
    
    # Compile Task Arithmetic lambdas
    lambdas_ta = torch.cat((torch.ones(1, 1, device=device), torch.full((1, 4), 0.3, device=device)), dim=1)

    print("Computing Layer-by-Layer CKA Representation Similarity on CIFAR-10 inputs...")
    
    # 1. Gather expert CIFAR-10 test activations across all 12 layers
    expert_cifar_encoder = expert_encoders["CIFAR10"]
    expert_cifar_encoder.eval()
    
    expert_hooks = []
    for i in range(12):
        expert_hooks.append(
            expert_cifar_encoder.transformer.resblocks[i].register_forward_hook(get_activation(f"expert_layer_{i}"))
        )
        
    expert_acts = {i: [] for i in range(12)}
    with torch.no_grad():
        for images, _ in test_loaders["CIFAR10"]:
            images = images.to(device)
            _ = expert_cifar_encoder(images)
            for i in range(12):
                expert_acts[i].append(activations[f"expert_layer_{i}"])
                
    for i in range(12):
        expert_acts[i] = torch.cat(expert_acts[i], dim=0)
        
    for h in expert_hooks:
        h.remove()
        
    cka_by_method = {
        "Task Arithmetic (\u03bb=0.3)": [],
        "Optimized AdaMerging (Adam GD)": [],
        "Spatially Averaged (Spatial Mean)": []
    }
    
    for method_key, lamb in [("Task Arithmetic (\u03bb=0.3)", lambdas_ta),
                             ("Optimized AdaMerging (Adam GD)", lambdas_gd),
                             ("Spatially Averaged (Spatial Mean)", mean_tasks_gd)]:
        # Reload functional model
        num_params = len(paramslist[0])
        lamb_expanded = lamb.repeat(num_params, 1) if lamb.shape[0] == 1 else lamb
        merged_params = []
        for j, p in enumerate(zip(*paramslist)):
            param_merged = sum(pi * lamb_expanded[j, i] for i, pi in enumerate(p))
            merged_params.append(param_merged)
        load_weights(merge_model.visual, names, merged_params)
        
        # Hook visual encoder
        merge_model.visual.eval()
        merged_hooks = []
        for i in range(12):
            merged_hooks.append(
                merge_model.visual.transformer.resblocks[i].register_forward_hook(get_activation(f"merged_layer_{i}"))
            )
            
        merged_acts = {i: [] for i in range(12)}
        with torch.no_grad():
            for images, _ in test_loaders["CIFAR10"]:
                images = images.to(device)
                _ = merge_model.visual(images)
                for i in range(12):
                    merged_acts[i].append(activations[f"merged_layer_{i}"])
                    
        for i in range(12):
            merged_acts[i] = torch.cat(merged_acts[i], dim=0)
            
        for h in merged_hooks:
            h.remove()
            
        # Compute layer CKA
        for i in range(12):
            cka_val = linear_cka(expert_acts[i], merged_acts[i])
            cka_by_method[method_key].append(cka_val)
            
        print(f"Method: {method_key} calculated.")
        
    # Generate line plot
    print("Generating layer-by-layer CKA plot...")
    plt.figure(figsize=(9, 6))
    colors = ['gray', 'orange', 'darkblue']
    linestyles = ['dotted', 'solid', 'dashed']
    markers = ['x', 'o', 'v']
    
    layers = np.arange(1, 13)
    for (method, cka_vals), color, linestyle, marker in zip(cka_by_method.items(), colors, linestyles, markers):
        plt.plot(layers, cka_vals, label=method, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=6)
        
    plt.xlabel('Transformer Block Layer Index', fontsize=12)
    plt.ylabel('Linear CKA Representation Similarity', fontsize=12)
    plt.title('Hierarchical Representational Alignment: Expert vs. Merged Models', fontsize=14, fontweight='bold')
    plt.ylim(0.92, 1.002)
    plt.xticks(layers)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower left')
    plt.tight_layout()
    
    # Save fig
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fig4_layer_cka.png")
    print("Saved results/fig4_layer_cka.png.")
    
    # Copy to submission results directory
    os.makedirs("submission/results", exist_ok=True)
    plt.savefig("submission/results/fig4_layer_cka.png")
    print("Copied to submission/results/fig4_layer_cka.png.")
    
if __name__ == "__main__":
    main()
