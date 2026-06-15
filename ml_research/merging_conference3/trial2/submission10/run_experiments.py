import os
import sys
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import open_clip
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Helper Functions for Making Model Functional & Loading Weights
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Linear CKA Similarity Calculation
# ---------------------------------------------------------
def linear_cka(X, Y):
    # Center the matrices
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute Hilbert-Schmidt Independence Criterion (HSIC)
    hsic_XY = torch.norm(torch.matmul(X_c.t(), Y_c), p='fro')**2
    hsic_XX = torch.norm(torch.matmul(X_c.t(), X_c), p='fro')**2
    hsic_YY = torch.norm(torch.matmul(Y_c.t(), Y_c), p='fro')**2
    
    cka = hsic_XY / (torch.sqrt(hsic_XX * hsic_YY) + 1e-8)
    return cka.item()

# ---------------------------------------------------------
# 3. Activation Capture Hook
# ---------------------------------------------------------
activations = {}
def get_activation(name):
    def hook(model, input, output):
        # Output is of shape (SeqLen, Batch, EmbedDim)
        # We average over sequence length dimension to get (Batch, EmbedDim)
        activations[name] = output.detach().mean(dim=0).cpu()
    return hook

# ---------------------------------------------------------
# 4. Main Experimental Pipeline
# ---------------------------------------------------------
def run_all_experiments():
    print("Initializing experimental run...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Define tasks
    task_names = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    seeds = [42, 100, 2026]
    
    # Standard CLIP ViT-B/32 normalizations
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
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
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert 1-channel to 3-channel
        transforms.Normalize(clip_mean, clip_std)
    ])
    
    # Load and cache datasets
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
    
    # To store all evaluation metrics across seeds
    all_results = {}
    
    # Noise sensitivities to sweep
    noise_levels = np.arange(0.05, 0.55, 0.05)
    noise_results = {
        "AdaMerging (1+1 ES)": {f"{gamma:.2f}": [] for gamma in noise_levels},
        "AdaMerging (Adam GD)": {f"{gamma:.2f}": [] for gamma in noise_levels},
        "Flat-AdaMerging (1+1 ES)": {f"{gamma:.2f}": [] for gamma in noise_levels},
        "Flat-AdaMerging (Adam GD)": {f"{gamma:.2f}": [] for gamma in noise_levels}
    }
    
    # CKA activations and experts per seed
    cka_results = {task: [] for task in task_names}
    
    for seed in seeds:
        print(f"\n==========================================")
        print(f"RUNNING TRIAL SEED: {seed}")
        print(f"==========================================")
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # 4.1 Draw unique subsets
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
            
        # Create loaders (precomputed and cached as TensorDatasets to avoid PIL transform bottlenecks on CPU)
        print("Pre-transforming and caching subsets into CPU memory...")
        train_loaders = {}
        cal_loaders = {}
        test_loaders = {}
        test_loaders_full = {}
        
        for task in task_names:
            # 1. Train subset
            loader_temp = DataLoader(train_subsets[task], batch_size=64, shuffle=False)
            imgs, lbls = [], []
            for img, lbl in loader_temp:
                imgs.append(img)
                lbls.append(lbl)
            imgs = torch.cat(imgs, dim=0)
            lbls = torch.cat(lbls, dim=0)
            train_loaders[task] = DataLoader(torch.utils.data.TensorDataset(imgs, lbls), batch_size=64, shuffle=True)
            
            # 2. Calibration subset
            loader_temp = DataLoader(cal_subsets[task], batch_size=64, shuffle=False)
            imgs, lbls = [], []
            for img, lbl in loader_temp:
                imgs.append(img)
                lbls.append(lbl)
            imgs = torch.cat(imgs, dim=0)
            lbls = torch.cat(lbls, dim=0)
            cal_loaders[task] = DataLoader(torch.utils.data.TensorDataset(imgs, lbls), batch_size=64, shuffle=False)
            
            # 3. Test subset (cached 512-image subset for fast sweeps)
            loader_temp = DataLoader(test_subsets[task], batch_size=64, shuffle=False)
            imgs, lbls = [], []
            for img, lbl in loader_temp:
                imgs.append(img)
                lbls.append(lbl)
            imgs = torch.cat(imgs, dim=0)
            lbls = torch.cat(lbls, dim=0)
            test_loaders[task] = DataLoader(torch.utils.data.TensorDataset(imgs, lbls), batch_size=64, shuffle=False)
            
            # 4. Full standard test split (for final evaluation)
            test_loaders_full[task] = DataLoader(datasets_test[task], batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        print("Subsets successfully cached in memory.")
        
        # Load fresh base model
        base_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        base_model = base_model.to(device)
        
        # Store expert visual encoders and classification heads
        expert_encoders = {}
        expert_heads = {}
        task_vectors = {}
        
        # Train experts
        for task in task_names:
            print(f"Training Expert for {task} (Seed {seed})...")
            
            encoder_path = f"checkpoints/seed_{seed}_{task}_encoder.pt"
            head_path = f"checkpoints/seed_{seed}_{task}_head.pt"
            
            # Load clean copy of pre-trained visual encoder and instantiate head
            expert_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            expert_model = expert_model.to(device)
            classification_head = nn.Linear(512, 10).to(device)
            
            if os.path.exists(encoder_path) and os.path.exists(head_path):
                print(f"  Found cached expert weights for {task}. Loading...")
                expert_model.visual.load_state_dict(torch.load(encoder_path, map_location=device))
                classification_head.load_state_dict(torch.load(head_path, map_location=device))
            else:
                # Train the expert
                expert_model.train()
                classification_head.train()
                
                optimizer = torch.optim.AdamW([
                    {'params': expert_model.visual.parameters(), 'lr': 1e-5},
                    {'params': classification_head.parameters(), 'lr': 1e-3}
                ], weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                start_t = time.time()
                for epoch in range(5):
                    for images, labels in train_loaders[task]:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        features = expert_model.encode_image(images)
                        outputs = classification_head(features)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                print(f"  Expert trained in {time.time() - start_t:.1f}s.")
                
                # Save expert weights
                torch.save(expert_model.visual.state_dict(), encoder_path)
                torch.save(classification_head.state_dict(), head_path)
                
            expert_encoders[task] = expert_model.visual
            expert_heads[task] = classification_head
            
            # Extract task vectors
            tv = {}
            base_state = base_model.visual.state_dict()
            expert_state = expert_model.visual.state_dict()
            for k in base_state.keys():
                if base_state[k].dtype.is_floating_point:
                    tv[k] = expert_state[k] - base_state[k]
                else:
                    tv[k] = expert_state[k].clone()
            task_vectors[task] = tv
            
        # 4.2 Evaluate Individual Experts
        expert_accs = {}
        for task in task_names:
            correct = 0
            total = 0
            expert_encoders[task].eval()
            expert_heads[task].eval()
            with torch.no_grad():
                for images, labels in test_loaders_full[task]:
                    images, labels = images.to(device), labels.to(device)
                    features = expert_encoders[task](images)
                    logits = expert_heads[task](features)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            expert_accs[task] = (correct / total) * 100.0
            print(f"Expert {task} Test Accuracy: {expert_accs[task]:.2f}%")
            
        # 4.3 Prepare visual encoder for functional weight reloading
        # We clone the base_model visual encoder to run our merging on
        merge_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        merge_model = merge_model.to(device)
        
        # Save baseline visual encoder state dict
        pretrained_visual_dic = {k: v.detach().clone().to(device) for k, v in merge_model.visual.named_parameters()}
        orig_params, names = make_functional(merge_model.visual)
        
        # Build paramslist
        # paramslist is a list of tuples containing:
        #   Index 0: base pretrained visual model parameters
        #   Index 1 to 4: task vectors for MNIST, FashionMNIST, CIFAR10, SVHN
        paramslist = []
        paramslist.append(tuple(pretrained_visual_dic[k].to(device) for k in pretrained_visual_dic.keys()))
        for task in task_names:
            paramslist.append(tuple(task_vectors[task][k].detach().clone().to(device) for k in pretrained_visual_dic.keys()))
            
        # Helper function to evaluate loss on calibration loaders
        def eval_loss(lambdas):
            # lambdas has shape (num_params, 5) or (1, 5)
            num_params = len(paramslist[0])
            if lambdas.shape[0] == 1:
                lambdas_expanded = lambdas.repeat(num_params, 1)
            else:
                lambdas_expanded = lambdas
                
            # Merge weights
            merged_params = []
            for j, p in enumerate(zip(*paramslist)):
                param_merged = sum(pi * lambdas_expanded[j, i] for i, pi in enumerate(p))
                merged_params.append(param_merged)
                
            # Load weights
            load_weights(merge_model.visual, names, merged_params)
            
            # Compute average prediction entropy on calibration loaders
            total_entropy = 0.0
            merge_model.visual.eval()
            for task in task_names:
                head = expert_heads[task]
                head.eval()
                task_entropy = 0.0
                count = 0
                for images, _ in cal_loaders[task]:
                    images = images.to(device)
                    features = merge_model.visual(images)
                    logits = head(features)
                    entropy = softmax_entropy(logits)
                    task_entropy += entropy
                    count += 1
                total_entropy += task_entropy / count
            return total_entropy
            
        # Helper function to evaluate test accuracy across all tasks (subset)
        def eval_test_acc(lambdas):
            num_params = len(paramslist[0])
            if lambdas.shape[0] == 1:
                lambdas_expanded = lambdas.repeat(num_params, 1)
            else:
                lambdas_expanded = lambdas
                
            # Merge weights
            merged_params = []
            for j, p in enumerate(zip(*paramslist)):
                param_merged = sum(pi * lambdas_expanded[j, i] for i, pi in enumerate(p))
                merged_params.append(param_merged)
                
            # Load weights
            load_weights(merge_model.visual, names, merged_params)
            
            merge_model.visual.eval()
            accs = {}
            for task in task_names:
                head = expert_heads[task]
                head.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loaders[task]:
                        images, labels = images.to(device), labels.to(device)
                        features = merge_model.visual(images)
                        logits = head(features)
                        correct += (logits.argmax(dim=1) == labels).sum().item()
                        total += labels.size(0)
                accs[task] = (correct / total) * 100.0
            accs["Average"] = sum(accs.values()) / len(task_names)
            return accs

        # Helper function to evaluate test accuracy across all tasks (Full test sets)
        def eval_test_acc_full(lambdas):
            num_params = len(paramslist[0])
            if lambdas.shape[0] == 1:
                lambdas_expanded = lambdas.repeat(num_params, 1)
            else:
                lambdas_expanded = lambdas
                
            # Merge weights
            merged_params = []
            for j, p in enumerate(zip(*paramslist)):
                param_merged = sum(pi * lambdas_expanded[j, i] for i, pi in enumerate(p))
                merged_params.append(param_merged)
                
            # Load weights
            load_weights(merge_model.visual, names, merged_params)
            
            merge_model.visual.eval()
            accs = {}
            for task in task_names:
                head = expert_heads[task]
                head.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loaders_full[task]:
                        images, labels = images.to(device), labels.to(device)
                        features = merge_model.visual(images)
                        logits = head(features)
                        correct += (logits.argmax(dim=1) == labels).sum().item()
                        total += labels.size(0)
                accs[task] = (correct / total) * 100.0
            accs["Average"] = sum(accs.values()) / len(task_names)
            return accs

        # 4.4 Optimization Methods Setup
        methods = {}
        
        # METHOD A: Task Arithmetic (Uniform baseline, lambda=0.3)
        lambdas_ta = torch.ones(1, 5)
        lambdas_ta[0, 0] = 1.0 # pretrain
        lambdas_ta[0, 1:] = 0.3 # task vectors
        methods[r"Task Arithmetic (Baseline, \lambda=0.3)"] = lambdas_ta
        
        # Additional Task Arithmetic baselines for the scale sweep
        for lam in [0.1, 0.2, 0.4, 0.5]:
            lambdas_ta_lam = torch.ones(1, 5)
            lambdas_ta_lam[0, 0] = 1.0 # pretrain
            lambdas_ta_lam[0, 1:] = lam # task vectors
            methods[f"Task Arithmetic (Baseline, \\lambda={lam:.1f})"] = lambdas_ta_lam
        
        # Compute initial prediction entropies for calibration
        initial_entropies = {}
        with torch.no_grad():
            num_params = len(paramslist[0])
            lambdas_init_expanded = lambdas_ta.repeat(num_params, 1)
            merged_params_init = []
            for j, p in enumerate(zip(*paramslist)):
                param_merged = sum(pi * lambdas_init_expanded[j, i] for i, pi in enumerate(p))
                merged_params_init.append(param_merged)
            load_weights(merge_model.visual, names, merged_params_init)
            
            merge_model.visual.eval()
            for task in task_names:
                head = expert_heads[task]
                head.eval()
                task_entropy = 0.0
                count = 0
                for images, _ in cal_loaders[task]:
                    images = images.to(device)
                    features = merge_model.visual(images)
                    logits = head(features)
                    entropy = softmax_entropy(logits)
                    task_entropy += entropy
                    count += 1
                initial_entropies[task] = (task_entropy / count).item()
        print(f"Initial calibration entropies: {initial_entropies}")

        def eval_loss_calibrated(lambdas):
            num_params = len(paramslist[0])
            if lambdas.shape[0] == 1:
                lambdas_expanded = lambdas.repeat(num_params, 1)
            else:
                lambdas_expanded = lambdas
            merged_params = []
            for j, p in enumerate(zip(*paramslist)):
                param_merged = sum(pi * lambdas_expanded[j, i] for i, pi in enumerate(p))
                merged_params.append(param_merged)
            load_weights(merge_model.visual, names, merged_params)
            
            total_entropy = 0.0
            merge_model.visual.eval()
            for task in task_names:
                head = expert_heads[task]
                head.eval()
                task_entropy = 0.0
                count = 0
                for images, _ in cal_loaders[task]:
                    images = images.to(device)
                    features = merge_model.visual(images)
                    logits = head(features)
                    entropy = softmax_entropy(logits)
                    task_entropy += entropy
                    count += 1
                total_entropy += (task_entropy / count) / (initial_entropies[task] + 1e-6)
            return total_entropy
        
        # METHOD B: AdaMerging (1+1 ES) - Parameter-wise (Layer-wise SOTA)
        print("Optimizing AdaMerging (1+1 ES) - Parameter-wise...")
        lambdas_es = torch.ones(len(paramslist[0]), 5, device=device)
        lambdas_es[:, 0] = 1.0 # pretrain
        lambdas_es[:, 1:] = 0.3 # task vectors
        sigma = 0.1
        best_loss = eval_loss(lambdas_es).item()
        for step in range(500):
            noise = torch.randn_like(lambdas_es) * sigma
            noise[:, 0] = 0.0 # pretrain coefficient fixed
            candidate = torch.clamp(lambdas_es + noise, min=0.0, max=1.0)
            candidate_loss = eval_loss(candidate).item()
            if candidate_loss < best_loss:
                lambdas_es = candidate
                best_loss = candidate_loss
                sigma = min(sigma * 1.1, 0.5)
            else:
                sigma = max(sigma * 0.9, 1e-4)
        methods["Optimized AdaMerging (1+1 ES)"] = lambdas_es
        
        # METHOD C: AdaMerging (Adam GD) - Parameter-wise (Layer-wise SOTA)
        print("Optimizing AdaMerging (Adam GD) - Parameter-wise...")
        lambdas_raw_gd = torch.full((len(paramslist[0]), 4), 0.3, device=device, requires_grad=True)
        optimizer_gd = torch.optim.Adam([lambdas_raw_gd], lr=1e-2)
        for step in range(200):
            optimizer_gd.zero_grad()
            task_lambdas = torch.clamp(lambdas_raw_gd, min=0.0, max=1.0)
            pretrain_lambdas = torch.ones(task_lambdas.shape[0], 1, device=task_lambdas.device)
            lambdas_gd_step = torch.cat((pretrain_lambdas, task_lambdas), dim=1)
            loss = eval_loss(lambdas_gd_step)
            loss.backward()
            optimizer_gd.step()
        with torch.no_grad():
            task_lambdas = torch.clamp(lambdas_raw_gd, min=0.0, max=1.0)
            pretrain_lambdas = torch.ones(task_lambdas.shape[0], 1, device=task_lambdas.device)
            lambdas_gd = torch.cat((pretrain_lambdas, task_lambdas), dim=1).detach()
        methods["Optimized AdaMerging (Adam GD)"] = lambdas_gd
        
        # METHOD D: Flat-AdaMerging (1+1 ES) - Ours (Minimalist)
        print("Optimizing Flat-AdaMerging (1+1 ES)...")
        lambdas_flat_es = torch.ones(1, 5, device=device)
        lambdas_flat_es[0, 0] = 1.0 # pretrain
        lambdas_flat_es[0, 1:] = 0.3 # task vectors
        sigma_flat = 0.1
        best_loss_flat = eval_loss(lambdas_flat_es).item()
        for step in range(500):
            noise = torch.randn_like(lambdas_flat_es) * sigma_flat
            noise[:, 0] = 0.0 # pretrain coefficient fixed
            candidate = torch.clamp(lambdas_flat_es + noise, min=0.0, max=1.0)
            candidate_loss = eval_loss(candidate).item()
            if candidate_loss < best_loss_flat:
                lambdas_flat_es = candidate
                best_loss_flat = candidate_loss
                sigma_flat = min(sigma_flat * 1.1, 0.5)
            else:
                sigma_flat = max(sigma_flat * 0.9, 1e-4)
        methods["Flat-AdaMerging (Ours - 1+1 ES)"] = lambdas_flat_es
        
        # METHOD E: Flat-AdaMerging (Adam GD) - Ours (Minimalist)
        print("Optimizing Flat-AdaMerging (Adam GD)...")
        lambdas_raw_flat_gd = torch.full((1, 4), 0.3, device=device, requires_grad=True)
        optimizer_flat_gd = torch.optim.Adam([lambdas_raw_flat_gd], lr=1e-2)
        for step in range(200):
            optimizer_flat_gd.zero_grad()
            task_lambdas = torch.clamp(lambdas_raw_flat_gd, min=0.0, max=1.0)
            pretrain_lambdas = torch.ones(task_lambdas.shape[0], 1, device=task_lambdas.device)
            lambdas_gd_step = torch.cat((pretrain_lambdas, task_lambdas), dim=1)
            loss = eval_loss(lambdas_gd_step)
            loss.backward()
            optimizer_flat_gd.step()
        with torch.no_grad():
            task_lambdas = torch.clamp(lambdas_raw_flat_gd, min=0.0, max=1.0)
            pretrain_lambdas = torch.ones(task_lambdas.shape[0], 1, device=task_lambdas.device)
            lambdas_flat_gd = torch.cat((pretrain_lambdas, task_lambdas), dim=1).detach()
        methods["Flat-AdaMerging (Ours - Adam GD)"] = lambdas_flat_gd
        
        # METHOD F: Calibrated Task-wise AdaMerging (1+1 ES) - Ours (Remedy)
        print("Optimizing Calibrated Task-wise AdaMerging (1+1 ES)...")
        lambdas_cal_es = torch.ones(1, 5, device=device)
        lambdas_cal_es[0, 0] = 1.0 # pretrain
        lambdas_cal_es[0, 1:] = 0.3 # task vectors
        sigma_cal = 0.1
        best_loss_cal = eval_loss_calibrated(lambdas_cal_es).item()
        for step in range(500):
            noise = torch.randn_like(lambdas_cal_es) * sigma_cal
            noise[:, 0] = 0.0 # pretrain coefficient fixed
            candidate = torch.clamp(lambdas_cal_es + noise, min=0.0, max=1.0)
            candidate_loss = eval_loss_calibrated(candidate).item()
            if candidate_loss < best_loss_cal:
                lambdas_cal_es = candidate
                best_loss_cal = candidate_loss
                sigma_cal = min(sigma_cal * 1.1, 0.5)
            else:
                sigma_cal = max(sigma_cal * 0.9, 1e-4)
        methods["Calibrated Task-wise AdaMerging (1+1 ES)"] = lambdas_cal_es
        
        # METHOD G: Calibrated Task-wise AdaMerging (Adam GD) - Ours (Remedy)
        print("Optimizing Calibrated Task-wise AdaMerging (Adam GD)...")
        lambdas_raw_cal_gd = torch.full((1, 4), 0.3, device=device, requires_grad=True)
        optimizer_cal_gd = torch.optim.Adam([lambdas_raw_cal_gd], lr=1e-2)
        for step in range(200):
            optimizer_cal_gd.zero_grad()
            task_lambdas = torch.clamp(lambdas_raw_cal_gd, min=0.0, max=1.0)
            pretrain_lambdas = torch.ones(task_lambdas.shape[0], 1, device=task_lambdas.device)
            lambdas_gd_step = torch.cat((pretrain_lambdas, task_lambdas), dim=1)
            loss = eval_loss_calibrated(lambdas_gd_step)
            loss.backward()
            optimizer_cal_gd.step()
        with torch.no_grad():
            task_lambdas = torch.clamp(lambdas_raw_cal_gd, min=0.0, max=1.0)
            pretrain_lambdas = torch.ones(task_lambdas.shape[0], 1, device=task_lambdas.device)
            lambdas_cal_gd = torch.cat((pretrain_lambdas, task_lambdas), dim=1).detach()
        methods["Calibrated Task-wise AdaMerging (Adam GD)"] = lambdas_cal_gd
        
        # TREATMENT 1: Intra-Task Layer Shuffling (1+1 ES)
        # Randomly shuffle the row coefficients for each of the 4 task columns
        shuffled_es = lambdas_es.clone()
        for i in range(1, 5):
            perm = torch.randperm(shuffled_es.shape[0])
            shuffled_es[:, i] = shuffled_es[perm, i]
        methods["Intra-Task Layer Shuffling (1+1 ES)"] = shuffled_es
        
        # TREATMENT 2: Intra-Task Layer Shuffling (Adam GD)
        shuffled_gd = lambdas_gd.clone()
        for i in range(1, 5):
            perm = torch.randperm(shuffled_gd.shape[0])
            shuffled_gd[:, i] = shuffled_gd[perm, i]
        methods["Intra-Task Layer Shuffling (Adam GD)"] = shuffled_gd
        
        # TREATMENT 3: Spatially Averaged (Spatial Mean - 1+1 ES)
        # Take mean of task-wise columns across the parameters
        avg_es = lambdas_es.clone()
        mean_tasks_es_raw = avg_es[:, 1:].mean(dim=0, keepdim=True)
        pretrain_es = torch.ones(1, 1, device=device)
        mean_tasks_es = torch.cat((pretrain_es, mean_tasks_es_raw), dim=1)
        methods["Spatially Averaged (Spatial Mean - 1+1 ES)"] = mean_tasks_es
        
        # TREATMENT 4: Spatially Averaged (Spatial Mean - Adam GD)
        avg_gd = lambdas_gd.clone()
        mean_tasks_gd_raw = avg_gd[:, 1:].mean(dim=0, keepdim=True)
        pretrain_gd = torch.ones(1, 1, device=device)
        mean_tasks_gd = torch.cat((pretrain_gd, mean_tasks_gd_raw), dim=1)
        methods["Spatially Averaged (Spatial Mean - Adam GD)"] = mean_tasks_gd
        
        # 4.5 Run Evaluations
        seed_results = {}
        for m_name, lambdas in methods.items():
            accs = eval_test_acc_full(lambdas)
            seed_results[m_name] = accs
            print(f"  Method {m_name}: Avg Acc = {accs['Average']:.2f}%")
            
        # Let's run TIES-Merging baseline
        print("Evaluating TIES-Merging (Static Baseline)...")
        pretrained_weights = paramslist[0]
        task_v_list = paramslist[1:]
        
        fraction = 0.20
        scaling_coef = 0.30
        
        merged_params = []
        for j, p in enumerate(zip(*task_v_list)):
            tvs = torch.stack(p)
            num_tasks = tvs.shape[0]
            
            flat_tvs = tvs.view(num_tasks, -1)
            thresholds = torch.quantile(flat_tvs.abs(), 1.0 - fraction, dim=1, keepdim=True)
            mask_pruned = flat_tvs.abs() >= thresholds
            pruned_flat = flat_tvs * mask_pruned
            
            signs = torch.sign(pruned_flat)
            pos_count = (signs > 0).sum(dim=0)
            neg_count = (signs < 0).sum(dim=0)
            majority_sign = torch.where(pos_count >= neg_count, 1.0, -1.0)
            
            resolved_flat = torch.where(signs == majority_sign.unsqueeze(0), pruned_flat, torch.zeros_like(pruned_flat))
            
            non_zero_count = (resolved_flat != 0).sum(dim=0).float()
            sum_flat = resolved_flat.sum(dim=0)
            mean_flat = torch.where(non_zero_count > 0, sum_flat / non_zero_count, torch.zeros_like(sum_flat))
            
            param_merged = pretrained_weights[j] + scaling_coef * mean_flat.view_as(pretrained_weights[j])
            merged_params.append(param_merged)
            
        load_weights(merge_model.visual, names, merged_params)
        
        merge_model.visual.eval()
        ties_accs = {}
        for task in task_names:
            head = expert_heads[task]
            head.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loaders_full[task]:
                    images, labels = images.to(device), labels.to(device)
                    features = merge_model.visual(images)
                    logits = head(features)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            ties_accs[task] = (correct / total) * 100.0
        ties_accs["Average"] = sum(ties_accs.values()) / len(task_names)
        print(f"  Method TIES-Merging (Static Baseline): Avg Acc = {ties_accs['Average']:.2f}%")
        
        seed_results["TIES-Merging (Static Baseline)"] = ties_accs

        # Let's run DARE-Merging baseline (Static Baseline)
        print("Evaluating DARE-Merging (Static Baseline)...")
        p_drop = 0.20
        scaling_coef = 0.30
        
        merged_params_dare = []
        g = torch.Generator(device=device)
        g.manual_seed(42)
        for j, p in enumerate(zip(*task_v_list)):
            tvs = torch.stack(p) # shape: (num_tasks, *param_shape)
            num_tasks = tvs.shape[0]
            
            rand_tensor = torch.rand(tvs.shape, generator=g, device=device)
            mask = (rand_tensor >= p_drop).float()
            
            dare_tvs = tvs * mask / (1.0 - p_drop)
            mean_flat = dare_tvs.mean(dim=0)
            
            param_merged = pretrained_weights[j] + scaling_coef * mean_flat
            merged_params_dare.append(param_merged)
            
        load_weights(merge_model.visual, names, merged_params_dare)
        
        merge_model.visual.eval()
        dare_accs = {}
        for task in task_names:
            head = expert_heads[task]
            head.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loaders_full[task]:
                    images, labels = images.to(device), labels.to(device)
                    features = merge_model.visual(images)
                    logits = head(features)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            dare_accs[task] = (correct / total) * 100.0
        dare_accs["Average"] = sum(dare_accs.values()) / len(task_names)
        print(f"  Method DARE-Merging (Static Baseline): Avg Acc = {dare_accs['Average']:.2f}%")
        
        seed_results["DARE-Merging (Static Baseline)"] = dare_accs
            
        all_results[seed] = seed_results
        
        # 4.6 Sweep Noise Sensitivity on Optimized Coefficients
        print("Sweeping noise sensitivity...")
        for gamma in noise_levels:
            # We add relative noise and compute test accuracies, then append to global noise logs
            # Gamma is standard deviation of relative noise
            for m_key, base_coef in [("AdaMerging (1+1 ES)", lambdas_es), 
                                     ("AdaMerging (Adam GD)", lambdas_gd),
                                     ("Flat-AdaMerging (1+1 ES)", lambdas_flat_es),
                                     ("Flat-AdaMerging (Adam GD)", lambdas_flat_gd)]:
                # Generate noise
                noise = torch.randn_like(base_coef) * (gamma * base_coef)
                noise[:, 0] = 0.0 # pretrain coefficient unaffected
                perturbed = torch.clamp(base_coef + noise, min=0.0, max=1.0)
                accs = eval_test_acc(perturbed)
                noise_results[m_key][f"{gamma:.2f}"].append(accs["Average"])
                
        # 4.7 CKA Representational Similarity Analysis
        # Hook into Block 5 of the visual encoder to capture hidden states
        print("Collecting hidden states for CKA similarity...")
        resblock5 = merge_model.visual.transformer.resblocks[5]
        hook_handle = resblock5.register_forward_hook(get_activation("resblock5"))
        
        # Gather expert CIFAR-10 test activations (Reference)
        expert_cifar_encoder = expert_encoders["CIFAR10"]
        expert_cifar_head = expert_heads["CIFAR10"]
        expert_cifar_encoder.eval()
        
        # Hook expert
        expert_hook = expert_cifar_encoder.transformer.resblocks[5].register_forward_hook(get_activation("expert"))
        
        expert_acts = []
        with torch.no_grad():
            for images, _ in test_loaders["CIFAR10"]:
                images = images.to(device)
                _ = expert_cifar_encoder(images)
                expert_acts.append(activations["expert"])
        expert_acts = torch.cat(expert_acts, dim=0)
        expert_hook.remove()
        
        # Evaluate CKA similarities on other merged states:
        for method_key, lamb in [("Optimized (1+1 ES)", lambdas_es),
                                 ("Averaged (1+1 ES)", mean_tasks_es),
                                 ("Optimized (Adam GD)", lambdas_gd),
                                 ("Averaged (Adam GD)", mean_tasks_gd)]:
            # Merge and load weights
            num_params = len(paramslist[0])
            lamb_expanded = lamb.repeat(num_params, 1) if lamb.shape[0] == 1 else lamb
            merged_params = []
            for j, p in enumerate(zip(*paramslist)):
                param_merged = sum(pi * lamb_expanded[j, i] for i, pi in enumerate(p))
                merged_params.append(param_merged)
            load_weights(merge_model.visual, names, merged_params)
            
            # Forward pass on CIFAR-10 test loader to collect activations
            merge_model.visual.eval()
            merged_acts = []
            with torch.no_grad():
                for images, _ in test_loaders["CIFAR10"]:
                    images = images.to(device)
                    _ = merge_model.visual(images)
                    merged_acts.append(activations["resblock5"])
            merged_acts = torch.cat(merged_acts, dim=0)
            
            # Compute CKA similarity
            cka_val = linear_cka(expert_acts, merged_acts)
            cka_results[task_names[2]].append({
                "Method": method_key,
                "CKA": cka_val
            })
            print(f"  CKA Similarity [Expert vs. {method_key}]: {cka_val:.4f}")
            
        hook_handle.remove()
        
    # ---------------------------------------------------------
    # 5. Process & Save Metrics
    # ---------------------------------------------------------
    print("\nProcessing final aggregated results across seeds...")
    summary_table = {}
    methods_to_report = list(all_results[seeds[0]].keys())
    
    for method in methods_to_report:
        summary_table[method] = {}
        for col in task_names + ["Average"]:
            vals = [all_results[seed][method][col] for seed in seeds]
            summary_table[method][col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }
            
    # Process CKA
    cka_table = {m: [] for m in ["Optimized (1+1 ES)", "Averaged (1+1 ES)", "Optimized (Adam GD)", "Averaged (Adam GD)"]}
    for entry in cka_results["CIFAR10"]:
        cka_table[entry["Method"]].append(entry["CKA"])
    for m in cka_table:
        cka_table[m] = {
            "mean": float(np.mean(cka_table[m])),
            "std": float(np.std(cka_table[m]))
        }
        
    # Save JSON metrics
    metrics_to_save = {
        "classification_results": summary_table,
        "noise_results": noise_results,
        "cka_results": cka_table
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    print("Saved results/metrics.json.")
    
    # ---------------------------------------------------------
    # 6. Generate Plots
    # ---------------------------------------------------------
    print("Generating plots...")
    
    # FIG 1: Classification accuracy comparisons of methods under treatments
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    methods_to_plot = [
        r"Task Arithmetic (Baseline, \lambda=0.3)",
        "Optimized AdaMerging (1+1 ES)",
        "Optimized AdaMerging (Adam GD)",
        "Flat-AdaMerging (Ours - 1+1 ES)",
        "Flat-AdaMerging (Ours - Adam GD)",
        "Calibrated Task-wise AdaMerging (1+1 ES)",
        "Calibrated Task-wise AdaMerging (Adam GD)",
        "Intra-Task Layer Shuffling (1+1 ES)",
        "Intra-Task Layer Shuffling (Adam GD)",
        "Spatially Averaged (Spatial Mean - 1+1 ES)",
        "Spatially Averaged (Spatial Mean - Adam GD)"
    ]
    
    means = [summary_table[m]["Average"]["mean"] for m in methods_to_plot]
    stds = [summary_table[m]["Average"]["std"] for m in methods_to_plot]
    short_labels = [
        r"TA (\lambda=0.3)",
        "Ada (1+1 ES)",
        "Ada (Adam GD)",
        "Flat-Ada (1+1 ES)",
        "Flat-Ada (Adam GD)",
        "Cal-Ada (1+1 ES)",
        "Cal-Ada (Adam GD)",
        "Shuffle (1+1 ES)",
        "Shuffle (Adam GD)",
        "Spatial (1+1 ES)",
        "Spatial (Adam GD)"
    ]
    
    bars = ax1.bar(short_labels, means, yerr=stds, align='center', alpha=0.8, ecolor='black', capsize=10, color=['gray', 'blue', 'orange', 'cyan', 'red', 'teal', 'magenta', 'lightgrey', 'bisque', 'darkblue', 'darkorange'])
    ax1.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax1.set_title('Classification Accuracies under Diagnostic Treatments (3 Seeds)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    fig1.savefig("results/fig1_treatments.png")
    plt.close(fig1)
    
    # FIG 2: Noise Sensitivity Curves
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for m_key, color, linestyle in [("AdaMerging (1+1 ES)", 'blue', '-'), 
                                     ("AdaMerging (Adam GD)", 'orange', '-'),
                                     ("Flat-AdaMerging (1+1 ES)", 'cyan', '--'),
                                     ("Flat-AdaMerging (Adam GD)", 'red', '--')]:
        noise_means = []
        noise_stds = []
        for gamma in noise_levels:
            vals = noise_results[m_key][f"{gamma:.2f}"]
            noise_means.append(np.mean(vals))
            noise_stds.append(np.std(vals))
        noise_means = np.array(noise_means)
        noise_stds = np.array(noise_stds)
        
        ax2.plot(noise_levels, noise_means, label=m_key, color=color, linestyle=linestyle, marker='o', linewidth=2)
        ax2.fill_between(noise_levels, noise_means - noise_stds, noise_means + noise_stds, color=color, alpha=0.15)
        
    ax2.set_xlabel(r'Relative Noise Level (\gamma)', fontsize=12)
    ax2.set_ylabel('Average Merged Accuracy (%)', fontsize=12)
    ax2.set_title('Landscape Flatness: Robustness to Coefficient Perturbations', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig2.savefig("results/fig2_noise_sensitivity.png")
    plt.close(fig2)
    
    # FIG 3: CKA representational similarity comparison
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    cka_labels = list(cka_table.keys())
    cka_means = [cka_table[l]["mean"] for l in cka_labels]
    cka_stds = [cka_table[l]["std"] for l in cka_labels]
    
    ax3.bar(cka_labels, cka_means, yerr=cka_stds, align='center', alpha=0.8, ecolor='black', capsize=10, color=['blue', 'cyan', 'orange', 'red'])
    ax3.set_ylabel('Linear CKA Similarity', fontsize=12)
    ax3.set_title('Linear CKA Similarity at Layer 6 on CIFAR-10 Inputs (3 Seeds)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0.9, 1.0)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    fig3.savefig("results/fig3_cka.png")
    plt.close(fig3)
    
    print("All plots generated successfully under results/.")
    print("Phase 2 Experimentation complete!")

if __name__ == "__main__":
    run_all_experiments()
