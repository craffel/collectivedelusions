import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
import numpy as np

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid driver/initialization errors on cluster
    np.random.seed(seed)

def get_deterministic_subset(dataset, num_samples, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    return Subset(dataset, indices)

def get_balanced_subset(dataset, N, seed):
    # Extract targets to ensure exact class balance
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    else:
        # Fallback by querying the dataset
        targets = [dataset[i][1] for i in range(len(dataset))]
    
    targets = torch.as_tensor(targets)
    classes = torch.unique(targets).tolist()
    num_classes = len(classes)
    samples_per_class = N // num_classes
    remainder = N % num_classes
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    selected_indices = []
    for i, c in enumerate(classes):
        class_indices = torch.where(targets == c)[0]
        perm = torch.randperm(len(class_indices), generator=g)
        n_sel = samples_per_class + (1 if i < remainder else 0)
        selected_indices.extend(class_indices[perm[:n_sel]].tolist())
        
    return Subset(dataset, selected_indices)

# Traversal to find Conv-BN pairs in ResNet-18
def get_conv_bn_pairs(model):
    modules = list(model.named_modules())
    pairs = []
    i = 0
    while i < len(modules):
        name, m = modules[i]
        if isinstance(m, nn.Conv2d):
            j = i + 1
            while j < len(modules):
                next_name, next_m = modules[j]
                if isinstance(next_m, nn.BatchNorm2d):
                    pairs.append((name, m, next_name, next_m))
                    break
                if isinstance(next_m, nn.Conv2d):
                    break
                j += 1
        i += 1
    return pairs

# Helper to collect activations at a specific Conv layer
def collect_conv_activations(model, loader, conv_module, device, unfold_params=None):
    activations = []
    def hook(module, input, output):
        if unfold_params is not None:
            # Unfold input to the Conv layer to get X (C_in * Kh * Kw, B * L)
            A = input[0].detach() # Input tensor to Conv
            X = torch.nn.functional.unfold(
                A,
                kernel_size=unfold_params['kernel_size'],
                dilation=unfold_params['dilation'],
                padding=unfold_params['padding'],
                stride=unfold_params['stride']
            ) # B x (C_in * Kh * Kw) x L
            X = X.transpose(1, 2).contiguous().view(-1, X.shape[1]).transpose(0, 1) # (C_in * Kh * Kw) x (B * L)
            activations.append(X.cpu())
        else:
            # Collect output activations (before BN)
            activations.append(output.detach().cpu())
            
    handle = conv_module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
    handle.remove()
    return torch.cat(activations, dim=1 if unfold_params is not None else 0)

# Exact BatchNorm Calibration (BNC)
def perform_bnc(merged_model, cal_loaders, device):
    print("Performing exact BatchNorm Calibration...")
    merged_model.eval()
    conv_bn_pairs = get_conv_bn_pairs(merged_model)
    
    # Process sequentially, layer by layer
    for conv_name, conv_mod, bn_name, bn_mod in conv_bn_pairs:
        # Collect merged model output activations at the Conv layer before BN
        activations = []
        def hook(module, input, output):
            activations.append(output.detach().cpu())
        
        handle = conv_mod.register_forward_hook(hook)
        with torch.no_grad():
            for loader in cal_loaders:
                for x, _ in loader:
                    x = x.to(device)
                    merged_model(x)
        handle.remove()
        
        V = torch.cat(activations, dim=0) # Shape: Total_Samples x C x H x W
        
        # Compute exact empirical mean and variance
        mean = V.mean(dim=(0, 2, 3)) # Size: C
        var = V.var(dim=(0, 2, 3), unbiased=False) # Size: C
        
        # Copy to BatchNorm layer
        bn_mod.running_mean.copy_(mean.to(device))
        bn_mod.running_var.copy_(var.to(device))

# Corrected SP-TAAC Calibration
def perform_sp_taac(merged_model, expert_models, cal_loaders, device):
    print("Performing corrected SP-TAAC Calibration...")
    merged_model.eval()
    for m in expert_models:
        m.eval()
        
    conv_bn_pairs = get_conv_bn_pairs(merged_model)
    expert_conv_bn_pairs = [get_conv_bn_pairs(m) for m in expert_models]
    
    # Process sequentially layer-by-layer
    for idx, (conv_name, conv_mod, bn_name, bn_mod) in enumerate(conv_bn_pairs):
        # 1. Collect target activations from each expert on their respective task data
        expert_acts = []
        for exp_idx, exp_model in enumerate(expert_models):
            exp_conv_mod = expert_conv_bn_pairs[exp_idx][idx][1]
            # Run expert on its own task calibration data
            act = collect_conv_activations(exp_model, cal_loaders[exp_idx], exp_conv_mod, device)
            expert_acts.append(act)
            
        pooled_target = torch.cat(expert_acts, dim=0) # Shape: (3*N) x C x H x W
        sigma_target = pooled_target.std(dim=(0, 2, 3)) # Size: C
        
        # 2. Collect merged model activations
        # We must collect the current merged activations (reflecting previous layer corrections)
        # We run the merged model on a combined calibration dataset
        merged_acts = []
        def hook(module, input, output):
            merged_acts.append(output.detach().cpu())
        
        handle = conv_mod.register_forward_hook(hook)
        with torch.no_grad():
            for loader in cal_loaders:
                for x, _ in loader:
                    x = x.to(device)
                    merged_model(x)
        handle.remove()
        
        V_merged = torch.cat(merged_acts, dim=0) # Shape: (3*N) x C x H x W
        
        # Update running stats of BN to match the merged Conv output
        mean = V_merged.mean(dim=(0, 2, 3))
        var = V_merged.var(dim=(0, 2, 3), unbiased=False)
        bn_mod.running_mean.copy_(mean.to(device))
        bn_mod.running_var.copy_(var.to(device))
        
        # Compute standard deviations
        sigma_merged = V_merged.std(dim=(0, 2, 3))
        
        # Compute scale correction factor gamma
        gamma = sigma_target / (sigma_merged + 1e-5)
        gamma = torch.clamp(gamma, 0.1, 10.0).to(device)
        
        # Scale BN running weight and bias
        bn_mod.weight.data.copy_(bn_mod.weight.data * gamma)
        bn_mod.bias.data.copy_(bn_mod.bias.data * gamma)

# Hybrid Calibration (SP-TAAC + SLR-WBC)
def perform_hybrid(merged_model, expert_models, cal_loaders, device, rank=4, reg_strength=0.1):
    print(f"Performing Hybrid Calibration (SP-TAAC + SLR-WBC, rank={rank}, reg={reg_strength})...")
    merged_model.eval()
    for m in expert_models:
        m.eval()
        
    conv_bn_pairs = get_conv_bn_pairs(merged_model)
    expert_conv_bn_pairs = [get_conv_bn_pairs(m) for m in expert_models]
    
    # Process sequentially, layer by layer
    for idx, (conv_name, conv_mod, bn_name, bn_mod) in enumerate(conv_bn_pairs):
        # 1. Collect target activations from each expert
        expert_acts = []
        for exp_idx, exp_model in enumerate(expert_models):
            exp_conv_mod = expert_conv_bn_pairs[exp_idx][idx][1]
            act = collect_conv_activations(exp_model, cal_loaders[exp_idx], exp_conv_mod, device)
            expert_acts.append(act)
        
        pooled_target = torch.cat(expert_acts, dim=0) # (3*N) x C x H x W
        
        # 2. Collect unfolded input activations (X) of the merged Conv layer
        unfold_params = {
            'kernel_size': conv_mod.kernel_size,
            'dilation': conv_mod.dilation,
            'padding': conv_mod.padding,
            'stride': conv_mod.stride
        }
        X_merged = collect_conv_activations(merged_model, cal_loaders[0], conv_mod, device, unfold_params)
        for loader in cal_loaders[1:]:
            X_merged = torch.cat([X_merged, collect_conv_activations(merged_model, loader, conv_mod, device, unfold_params)], dim=1)
        # X_merged shape: d_in x M
        
        # 3. Flatten pooled target to shape C_out x M
        V_target_flat = pooled_target.transpose(0, 1).contiguous().view(pooled_target.shape[1], -1).to(device) # C_out x M
        
        # 4. Reshape current merged weights to C_out x d_in
        W_curr = conv_mod.weight.view(conv_mod.out_channels, -1).detach() # C_out x d_in (on GPU)
        
        # 5. Compute Error E = V_target - W_curr @ X_merged
        X_merged_gpu = X_merged.to(device)
        E = V_target_flat - W_curr @ X_merged_gpu # C_out x M (on GPU)
        
        # 6. Solve Ridge Regression: delta_W = E @ X^T @ inv(X @ X^T + lambda * I)
        d_in = W_curr.shape[1]
        M = X_merged_gpu.shape[1]
        ridge_lambda = reg_strength * M
        
        # Compute covariance matrix on GPU
        XX_T = X_merged_gpu @ X_merged_gpu.T # d_in x d_in (on GPU)
        inv_term = torch.inverse(XX_T + ridge_lambda * torch.eye(d_in, device=device))
        delta_W_star = E @ X_merged_gpu.T @ inv_term # C_out x d_in (on GPU)
        
        # 7. SVD Rank-r Truncated Correction
        try:
            U, S, V_t = torch.linalg.svd(delta_W_star, full_matrices=False)
            r = min(rank, len(S))
            delta_W_r = U[:, :r] @ torch.diag(S[:r]) @ V_t[:r, :]
        except Exception as e:
            print(f"SVD failed at layer {conv_name}, using standard correction:", e)
            delta_W_r = delta_W_star
            
        # 8. Update Conv weights in-place
        new_weight = W_curr + delta_W_r
        conv_mod.weight.data.copy_(new_weight.view_as(conv_mod.weight).to(device))
        
        # 9. Run the calibration data through the updated layer to get new activations V_new
        merged_acts = []
        def hook(module, input, output):
            merged_acts.append(output.detach().cpu())
        
        handle = conv_mod.register_forward_hook(hook)
        with torch.no_grad():
            for loader in cal_loaders:
                for x, _ in loader:
                    x = x.to(device)
                    merged_model(x)
        handle.remove()
        
        V_merged_new = torch.cat(merged_acts, dim=0) # (3*N) x C x H x W
        
        # Update BN statistics
        mean = V_merged_new.mean(dim=(0, 2, 3))
        var = V_merged_new.var(dim=(0, 2, 3), unbiased=False)
        bn_mod.running_mean.copy_(mean.to(device))
        bn_mod.running_var.copy_(var.to(device))
        
        # 10. Compute standard deviations and apply SP-TAAC scaling
        sigma_target = pooled_target.std(dim=(0, 2, 3))
        sigma_merged_new = V_merged_new.std(dim=(0, 2, 3))
        
        gamma = sigma_target / (sigma_merged_new + 1e-5)
        gamma = torch.clamp(gamma, 0.1, 10.0).to(device)
        
        bn_mod.weight.data.copy_(bn_mod.weight.data * gamma)
        bn_mod.bias.data.copy_(bn_mod.bias.data * gamma)

# Evaluation function
def evaluate_model(model, test_loaders, device):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for loader in test_loaders:
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            accuracies.append(100.0 * correct / total)
    return accuracies

# Main Execution Script
def run_all(seed=42):
    print(f"\n================ Running Seed {seed} ================")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Datasets Preparation
    transform_grayscale = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading datasets...")
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_grayscale)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_grayscale)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_grayscale)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_grayscale)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    # Get 5,000-sample deterministic subsets for training
    mnist_train = get_deterministic_subset(mnist_train_full, 5000, seed)
    fmnist_train = get_deterministic_subset(fmnist_train_full, 5000, seed)
    cifar_train = get_deterministic_subset(cifar_train_full, 5000, seed)
    
    # Get 128-sample class-balanced calibration subsets
    mnist_cal = get_balanced_subset(mnist_train_full, 128, seed)
    fmnist_cal = get_balanced_subset(fmnist_train_full, 128, seed)
    cifar_cal = get_balanced_subset(cifar_train_full, 128, seed)
    
    # Loaders
    train_loaders = [
        DataLoader(mnist_train, batch_size=64, shuffle=True),
        DataLoader(fmnist_train, batch_size=64, shuffle=True),
        DataLoader(cifar_train, batch_size=64, shuffle=True)
    ]
    
    cal_loaders = [
        DataLoader(mnist_cal, batch_size=64, shuffle=False),
        DataLoader(fmnist_cal, batch_size=64, shuffle=False),
        DataLoader(cifar_cal, batch_size=64, shuffle=False)
    ]
    
    test_loaders = [
        DataLoader(mnist_test, batch_size=128, shuffle=False),
        DataLoader(fmnist_test, batch_size=128, shuffle=False),
        DataLoader(cifar_test, batch_size=128, shuffle=False)
    ]
    
    tasks = ["MNIST", "FashionMNIST", "CIFAR10"]
    scenarios = {
        "A": {"name": "SGD_LowReg", "opt": "sgd", "lr": 1e-4, "wd": 1e-4},
        "B": {"name": "SGD_HighReg", "opt": "sgd", "lr": 1e-4, "wd": 1e-2},
        "C": {"name": "AdamW_LowReg", "opt": "adamw", "lr": 1e-4, "wd": 1e-4},
        "D": {"name": "AdamW_HighLR", "opt": "adamw", "lr": 1e-3, "wd": 1e-4},
        "E": {"name": "AdamW_HighReg", "opt": "adamw", "lr": 1e-4, "wd": 1e-2}
    }
    
    results = {}
    
    # Let's run all scenarios
    for sc_id, sc in scenarios.items():
        print(f"\n--- Scenario {sc_id}: {sc['name']} ---")
        expert_paths = []
        os.makedirs(f"models/{sc['name']}", exist_ok=True)
        
        # Load shared pre-trained progenitor
        base_weights = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        
        expert_models = []
        # Train or load expert models
        for t_idx, task_name in enumerate(tasks):
            model_path = f"models/{sc['name']}/{task_name}_seed{seed}.pt"
            expert_paths.append(model_path)
            
            # Instantiate model
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 10)
            model = model.to(device)
            
            if os.path.exists(model_path):
                print(f"Loading pre-trained expert {task_name} from {model_path}...")
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"Fine-tuning expert {task_name} for 5 epochs...")
                criterion = nn.CrossEntropyLoss()
                if sc["opt"] == "sgd":
                    optimizer = optim.SGD(model.parameters(), lr=sc["lr"], momentum=0.9, weight_decay=sc["wd"])
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=sc["wd"])
                
                model.train()
                for epoch in range(5):
                    running_loss = 0.0
                    for x, y in train_loaders[t_idx]:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    print(f"  Epoch {epoch+1}/5 Loss: {running_loss/len(train_loaders[t_idx]):.4f}")
                
                # Save expert weights
                torch.save(model.state_dict(), model_path)
                print(f"Expert {task_name} saved to {model_path}.")
            
            expert_models.append(model)
            
        # 2. Evaluate Individual Experts to confirm training was successful
        expert_accs = []
        for t_idx, task_name in enumerate(tasks):
            accs = evaluate_model(expert_models[t_idx], [test_loaders[t_idx]], device)
            expert_accs.append(accs[0])
            print(f"Expert {task_name} Test Accuracy: {accs[0]:.2f}%")
            
        # 3. Compute Weight Drift and Update Cosine Similarity
        # Flatten and extract updates relative to progenitor
        # Progenitor has base_weights, but with fc dimension mismatch. We align fc weights by starting from base_weights and ignoring fc, or we can just measure parameters that share the same dimensions.
        # Since fc.weight and fc.bias have different shapes between base progenitor (1000 classes) and experts (10 classes), we exclude 'fc.weight' and 'fc.bias' from geometric calculations.
        print("\nComputing weight-space geometries...")
        drifts = []
        updates = []
        for m_idx, model in enumerate(expert_models):
            expert_state = model.state_dict()
            drift_elements = []
            update_elements = []
            for key in base_weights.keys():
                if "fc" in key or "running" in key or "tracked" in key:
                    continue # Exclude classification head and non-trainable BatchNorm buffers
                upd = (expert_state[key].cpu() - base_weights[key].cpu()).numpy().flatten()
                update_elements.append(upd)
                
            flat_update = np.concatenate(update_elements)
            updates.append(flat_update)
            drift_l2 = np.linalg.norm(flat_update)
            drifts.append(float(drift_l2))
            print(f"Expert {tasks[m_idx]} L2-Drift from initialization: {drift_l2:.4f}")
            
        # Cosine Similarities between expert updates
        cos_sim_12 = np.dot(updates[0], updates[1]) / (np.linalg.norm(updates[0]) * np.linalg.norm(updates[1]) + 1e-8)
        cos_sim_13 = np.dot(updates[0], updates[2]) / (np.linalg.norm(updates[0]) * np.linalg.norm(updates[2]) + 1e-8)
        cos_sim_23 = np.dot(updates[1], updates[2]) / (np.linalg.norm(updates[1]) * np.linalg.norm(updates[2]) + 1e-8)
        avg_cos_sim = (cos_sim_12 + cos_sim_13 + cos_sim_23) / 3.0
        print(f"Cosine Similarity (MNIST / F-MNIST): {cos_sim_12:.4f}")
        print(f"Cosine Similarity (MNIST / CIFAR10): {cos_sim_13:.4f}")
        print(f"Cosine Similarity (F-MNIST / CIFAR10): {cos_sim_23:.4f}")
        print(f"Average Update Cosine Similarity: {avg_cos_sim:.4f}")
        
        # 4. Perform Parameter-Space Model Merging (Weight Averaging)
        print("\nMerging models via Weight Averaging...")
        merged_model = resnet18()
        merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
        merged_model = merged_model.to(device)
        
        merged_state = merged_model.state_dict()
        expert_states = [m.state_dict() for m in expert_models]
        
        for key in merged_state.keys():
            # Directly average parameter values across the 3 experts
            merged_state[key].copy_(
                sum(states[key] for states in expert_states) / 3.0
            )
        merged_model.load_state_dict(merged_state)
        
        # 5. Evaluate uncalibrated merged model
        accs_uncal = evaluate_model(merged_model, test_loaders, device)
        avg_uncal = sum(accs_uncal) / len(accs_uncal)
        print(f"Uncalibrated Merged Model - MNIST: {accs_uncal[0]:.2f}%, F-MNIST: {accs_uncal[1]:.2f}%, CIFAR10: {accs_uncal[2]:.2f}%, Avg: {avg_uncal:.2f}%")
        
        # 6. Evaluate exact BatchNorm Calibration (BNC)
        # Restore uncalibrated weights to fresh model for clean calibration tests
        merged_model.load_state_dict(merged_state)
        perform_bnc(merged_model, cal_loaders, device)
        accs_bnc = evaluate_model(merged_model, test_loaders, device)
        avg_bnc = sum(accs_bnc) / len(accs_bnc)
        print(f"BatchNorm Calibrated Model - MNIST: {accs_bnc[0]:.2f}%, F-MNIST: {accs_bnc[1]:.2f}%, CIFAR10: {accs_bnc[2]:.2f}%, Avg: {avg_bnc:.2f}%")
        
        # 7. Evaluate SP-TAAC Calibration
        merged_model.load_state_dict(merged_state)
        perform_sp_taac(merged_model, expert_models, cal_loaders, device)
        accs_sptaac = evaluate_model(merged_model, test_loaders, device)
        avg_sptaac = sum(accs_sptaac) / len(accs_sptaac)
        print(f"SP-TAAC Calibrated Model - MNIST: {accs_sptaac[0]:.2f}%, F-MNIST: {accs_sptaac[1]:.2f}%, CIFAR10: {accs_sptaac[2]:.2f}%, Avg: {avg_sptaac:.2f}%")
        
        # 8. Evaluate Hybrid Calibration (SP-TAAC + SLR-WBC) at Rank 4 and Rank 8
        merged_model.load_state_dict(merged_state)
        perform_hybrid(merged_model, expert_models, cal_loaders, device, rank=4, reg_strength=0.1)
        accs_hybrid4 = evaluate_model(merged_model, test_loaders, device)
        avg_hybrid4 = sum(accs_hybrid4) / len(accs_hybrid4)
        print(f"Hybrid Calibrated (Rank 4) - MNIST: {accs_hybrid4[0]:.2f}%, F-MNIST: {accs_hybrid4[1]:.2f}%, CIFAR10: {accs_hybrid4[2]:.2f}%, Avg: {avg_hybrid4:.2f}%")
        
        merged_model.load_state_dict(merged_state)
        perform_hybrid(merged_model, expert_models, cal_loaders, device, rank=8, reg_strength=0.1)
        accs_hybrid8 = evaluate_model(merged_model, test_loaders, device)
        avg_hybrid8 = sum(accs_hybrid8) / len(accs_hybrid8)
        print(f"Hybrid Calibrated (Rank 8) - MNIST: {accs_hybrid8[0]:.2f}%, F-MNIST: {accs_hybrid8[1]:.2f}%, CIFAR10: {accs_hybrid8[2]:.2f}%, Avg: {avg_hybrid8:.2f}%")
        
        # 9. Profile deep layer (layer4.1.bn2) activations to diagnose representation collapse
        # Collect BN activations of uncalibrated vs. Hybrid(Rank 8) models
        # Also collect expert BN activations for reference
        print("\nProfiling layer4.1.bn2 standard deviations...")
        profile_layer = merged_model.layer4[1].bn2
        
        # Expert profile
        exp_stds = []
        for exp_idx, exp_model in enumerate(expert_models):
            exp_layer = exp_model.layer4[1].bn2
            act = collect_conv_activations(exp_model, cal_loaders[exp_idx], exp_layer, device)
            exp_stds.append(act.std(dim=(0, 2, 3)).cpu().numpy().tolist())
        target_profile = [float(x) for x in np.mean(exp_stds, axis=0)]
        
        # Uncalibrated profile
        merged_model.load_state_dict(merged_state)
        uncal_act = collect_conv_activations(merged_model, cal_loaders[0], profile_layer, device)
        for loader in cal_loaders[1:]:
            uncal_act = torch.cat([uncal_act, collect_conv_activations(merged_model, loader, profile_layer, device)], dim=0)
        uncal_profile = uncal_act.std(dim=(0, 2, 3)).cpu().numpy().tolist()
        
        # Hybrid profile
        merged_model.load_state_dict(merged_state)
        perform_hybrid(merged_model, expert_models, cal_loaders, device, rank=8, reg_strength=0.1)
        hybrid_act = collect_conv_activations(merged_model, cal_loaders[0], profile_layer, device)
        for loader in cal_loaders[1:]:
            hybrid_act = torch.cat([hybrid_act, collect_conv_activations(merged_model, loader, profile_layer, device)], dim=0)
        hybrid_profile = hybrid_act.std(dim=(0, 2, 3)).cpu().numpy().tolist()
        
        # Compute SVD spectrum of uncalibrated vs. Hybrid(Rank 8) vs. target
        print("Computing feature spectrum (SVD) of activations...")
        # Flatten spatial dimensions
        # Shape: C x (3 * N * H * W)
        uncal_flat = uncal_act.transpose(0, 1).contiguous().view(uncal_act.shape[1], -1)
        hybrid_flat = hybrid_act.transpose(0, 1).contiguous().view(hybrid_act.shape[1], -1)
        
        # Expert 1 (as proxy for target spectrum)
        target_act = collect_conv_activations(expert_models[0], cal_loaders[0], expert_models[0].layer4[1].bn2, device)
        target_flat = target_act.transpose(0, 1).contiguous().view(target_act.shape[1], -1)
        
        _, S_uncal, _ = torch.svd(uncal_flat)
        _, S_hybrid, _ = torch.svd(hybrid_flat)
        _, S_target, _ = torch.svd(target_flat)
        
        results[sc_id] = {
            "name": sc["name"],
            "expert_accs": expert_accs,
            "drifts": drifts,
            "cos_sims": [float(cos_sim_12), float(cos_sim_13), float(cos_sim_23)],
            "avg_cos_sim": float(avg_cos_sim),
            "uncal_accs": accs_uncal,
            "avg_uncal": float(avg_uncal),
            "bnc_accs": accs_bnc,
            "avg_bnc": float(avg_bnc),
            "sptaac_accs": accs_sptaac,
            "avg_sptaac": float(avg_sptaac),
            "hybrid4_accs": accs_hybrid4,
            "avg_hybrid4": float(avg_hybrid4),
            "hybrid8_accs": accs_hybrid8,
            "avg_hybrid8": float(avg_hybrid8),
            "profile": {
                "target_std": target_profile,
                "uncal_std": uncal_profile,
                "hybrid_std": hybrid_profile,
                "S_target": S_target.cpu().numpy().tolist(),
                "S_uncal": S_uncal.cpu().numpy().tolist(),
                "S_hybrid": S_hybrid.cpu().numpy().tolist()
            }
        }
        
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Save results to a json file
    with open(f"results_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print(f"Saved results for seed {seed} successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_all(seed=args.seed)
