import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from corruptions import corrupt_dataset_batch, CORRUPTIONS

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False

# Dataset configuration
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper to get loaders
def get_dataloader(task_name, is_train=False, batch_size=256, subset_size=None):
    if task_name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=not is_train, download=False, transform=transform_gray)
    elif task_name == "fashion":
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=not is_train, download=False, transform=transform_gray)
    elif task_name == "cifar":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=not is_train, download=False, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task {task_name}")
    
    if subset_size is not None:
        indices = np.random.choice(len(dataset), min(subset_size, len(dataset)), replace=False)
        dataset = Subset(dataset, indices)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

# Multi-Perturbation Calibration Set (MPCS)
class MPCSDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, corruption_types, severities):
        self.base_dataset = base_dataset
        self.corruption_types = corruption_types
        self.severities = severities
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if np.random.rand() < 0.5:
            corr_name = np.random.choice(self.corruption_types)
            sev = np.random.choice(self.severities)
            img_batch = img.unsqueeze(0)
            corrupted_batch = corrupt_dataset_batch(img_batch, corr_name, sev)
            img = corrupted_batch.squeeze(0)
        return img, label

def get_mpcs_dataloader(task_name, batch_size=64, subset_size=128):
    if task_name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray)
    elif task_name == "fashion":
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray)
    elif task_name == "cifar":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_rgb)
    else:
        raise ValueError()
        
    indices = np.random.choice(len(dataset), min(subset_size, len(dataset)), replace=False)
    subset_dataset = Subset(dataset, indices)
    corruption_types = ["gaussian_noise", "gaussian_blur", "contrast", "brightness"]
    severities = [1, 2]
    
    mpcs_dataset = MPCSDataset(subset_dataset, corruption_types, severities)
    loader = DataLoader(mpcs_dataset, batch_size=batch_size, shuffle=False)
    return loader

# Calibration capturing helpers
def capture_activations(model, dataloader, layer_names, device, num_samples=128):
    activations = {name: [] for name in layer_names}
    hooks = []
    
    def get_hook(name):
        def hook(module, input_tensor, output_tensor):
            activations[name].append(input_tensor[0].detach().cpu())
        return hook
        
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    count = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            count += images.size(0)
            if count >= num_samples:
                break
                
    for h in hooks:
        h.remove()
        
    for name in layer_names:
        activations[name] = torch.cat(activations[name], dim=0)[:num_samples]
    return activations

# Evaluation loop
def evaluate_model(model, dataloader, head, device, corruption_name=None, severity=None):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            if corruption_name is not None:
                images = corrupt_dataset_batch(images, corruption_name, severity)
                
            features = model(images)
            outputs = head(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

# Calibration Methods (Exact Cumulative BN Calibration)
def calibrate_bn(model, dataloader, device='cuda'):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None # exact cumulative average over all seen samples
            
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            
    # Restore standard momentum for subsequent operations
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    model.eval()

def ties_merging(progenitor_state, experts_state, density=0.2):
    merged_state = {}
    for k in progenitor_state.keys():
        p_weight = progenitor_state[k]
        if not p_weight.is_floating_point():
            merged_state[k] = experts_state[0][k].clone()
            continue
            
        trimmed_task_vectors = []
        for exp_state in experts_state:
            task_vector = exp_state[k] - p_weight
            flat_tv = task_vector.flatten()
            num_keep = max(1, int(density * flat_tv.numel()))
            threshold = torch.topk(flat_tv.abs(), num_keep).values[-1]
            mask = (task_vector.abs() >= threshold)
            trimmed_tv = task_vector * mask
            trimmed_task_vectors.append(trimmed_tv)
            
        stacked_signs = torch.stack([torch.sign(tv) for tv in trimmed_task_vectors], dim=0)
        sum_signs = stacked_signs.sum(dim=0)
        elected_sign = torch.sign(sum_signs)
        
        agreed_tvs = []
        for tv in trimmed_task_vectors:
            agree_mask = (torch.sign(tv) == elected_sign) & (tv != 0)
            agreed_tvs.append(tv * agree_mask)
            
        stacked_agreed = torch.stack(agreed_tvs, dim=0)
        num_agreed = (stacked_agreed != 0).sum(dim=0).float()
        sum_agreed = stacked_agreed.sum(dim=0)
        merged_tv = torch.where(num_agreed > 0, sum_agreed / num_agreed, torch.zeros_like(sum_agreed))
        merged_state[k] = p_weight + merged_tv
    return merged_state

def dare_merging(progenitor_state, experts_state, drop_rate=0.5):
    merged_state = {}
    keep_prob = 1.0 - drop_rate
    scale = 1.0 / keep_prob
    for k in progenitor_state.keys():
        p_weight = progenitor_state[k]
        if not p_weight.is_floating_point():
            merged_state[k] = experts_state[0][k].clone()
            continue
            
        tvs = []
        for exp_state in experts_state:
            tv = exp_state[k] - p_weight
            mask = (torch.rand_like(tv) >= drop_rate).float()
            tv_dare = tv * mask * scale
            tvs.append(tv_dare)
            
        avg_tv = torch.stack(tvs, dim=0).mean(dim=0)
        merged_state[k] = p_weight + avg_tv
    return merged_state

def apply_slr_wbc_single_task(merged_model, expert_model, calib_loader, layer_names, device, rank=8, is_robust=False, shrink_c=0.1):
    merged_model.eval()
    expert_model.eval()
    
    for layer_name in layer_names:
        merged_conv = None
        for name, module in merged_model.named_modules():
            if name == layer_name:
                merged_conv = module
                break
        if merged_conv is None:
            continue
            
        act_merged = capture_activations(merged_model, calib_loader, [layer_name], device, num_samples=128)[layer_name]
        act_expert = capture_activations(expert_model, calib_loader, [layer_name], device, num_samples=128)[layer_name]
        
        N, C, H, W = act_merged.shape
        X_m = act_merged.permute(0, 2, 3, 1).reshape(-1, C).double()
        X_e = act_expert.permute(0, 2, 3, 1).reshape(-1, C).double()
        
        gamma = 1e-4 if not is_robust else 1e-2
        XTX = torch.matmul(X_m.t(), X_m)
        XTE = torch.matmul(X_m.t(), X_e)
        
        try:
            P = torch.linalg.solve(XTX + gamma * torch.eye(C, device=XTX.device), XTE)
        except Exception:
            P = torch.matmul(torch.linalg.pinv(XTX + gamma * torch.eye(C, device=XTX.device)), XTE)
            
        U, S, Vh = torch.linalg.svd(P)
        
        if is_robust:
            S_shrunk = (S**2 / (S**2 + shrink_c)) * S
            P = torch.matmul(U, torch.matmul(torch.diag(S_shrunk), Vh)).float()
        else:
            P_diff = P - torch.eye(C, device=P.device)
            U_d, S_d, Vh_d = torch.linalg.svd(P_diff)
            S_d_truncated = S_d.clone()
            S_d_truncated[rank:] = 0.0
            P = torch.eye(C, device=P.device) + torch.matmul(U_d, torch.matmul(torch.diag(S_d_truncated), Vh_d))
            P = P.float()
            
        with torch.no_grad():
            W = merged_conv.weight.data
            W_new = torch.einsum('o j k l, i j -> o i k l', W, P.to(W.device))
            merged_conv.weight.copy_(W_new)

# MSPR Routing Helper
def get_mspr_prototypes(models_dict, calib_loaders, layer_name, device):
    # Extracts task prototypes from the clean representation space of the models in models_dict
    prototypes = {}
    for task_name, model in models_dict.items():
        model.eval()
        act = capture_activations(model, calib_loaders[task_name], [layer_name], device, num_samples=128)[layer_name]
        proto = act.mean(dim=[0, 2, 3])
        prototypes[task_name] = proto.to(device)
    return prototypes

def evaluate_mspr_routed(backbones, test_loaders, heads, prototypes, layer_name, device, corruption_name=None, severity=None, routing_backbone=None):
    # MSPR routed evaluation: sample-by-sample cosine-similarity task identification in batch mode
    for b in backbones.values():
        b.eval()
    for h in heads.values():
        h.eval()
    if routing_backbone is not None:
        routing_backbone.eval()
        
    total_correct = 0
    total_samples = 0
    tasks_list = sorted(list(prototypes.keys()))
    
    for true_task, loader in test_loaders.items():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            if corruption_name is not None:
                images = corrupt_dataset_batch(images, corruption_name, severity)
                
            # Use provided routing backbone, or fallback to first backbone
            base_backbone = routing_backbone if routing_backbone is not None else backbones["None" if "None" in backbones else list(backbones.keys())[0]]
            
            batch_acts = []
            def hook_fn(module, input_val, output_val):
                batch_acts.append(input_val[0].detach())
                
            hook = None
            for name, module in base_backbone.named_modules():
                if name == layer_name:
                    hook = module.register_forward_hook(hook_fn)
                    break
                    
            with torch.no_grad():
                _ = base_backbone(images)
                
            hook.remove()
            
            act = batch_acts[0]
            sample_features = act.mean(dim=[2, 3]) # [B, C]
            
            # Compute cosine similarities with each prototype
            similarities = {}
            for task_name, proto in prototypes.items():
                norm_sample = F.normalize(sample_features, p=2, dim=1)
                norm_proto = F.normalize(proto, p=2, dim=0).unsqueeze(0)
                sim = torch.mm(norm_sample, norm_proto.t()).squeeze(1)
                similarities[task_name] = sim
                
            stacked_sim = torch.stack([similarities[t] for t in tasks_list], dim=1)
            routed_indices = stacked_sim.argmax(dim=1)
            
            # Batched Execution Optimization: Group inputs in the batch by their routed task
            grouped_indices = {task: [] for task in prototypes.keys()}
            for b_idx in range(images.size(0)):
                routed_task = tasks_list[routed_indices[b_idx].item()]
                grouped_indices[routed_task].append(b_idx)
                
            for task, indices in grouped_indices.items():
                if len(indices) == 0:
                    continue
                indices_t = torch.tensor(indices, device=device)
                batch_images = images[indices_t]
                batch_labels = labels[indices_t]
                
                with torch.no_grad():
                    feat = backbones[task](batch_images)
                    output = heads[task](feat)
                    _, pred = output.max(1)
                    
                total_correct += pred.eq(batch_labels).sum().item()
                total_samples += len(indices)
                
    return 100.0 * total_correct / total_samples

# Main Pipeline
def main():
    print("\nLoading trained expert weights...")
    experts = {}
    expert_heads = {}
    for task in ["mnist", "fashion", "cifar"]:
        ckpt_path = f"checkpoints/{task}_expert.pt"
        if not os.path.exists(ckpt_path):
            print(f"Error: {ckpt_path} not found. Please run training first.")
            return
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"Loaded {task} expert with training accuracy: {checkpoint['accuracy']:.2f}%")
        
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model_state = model.state_dict()
        model_state.update(checkpoint['backbone_state_dict'])
        for k, v in checkpoint['fc_state_dict'].items():
            model_state[f"fc.{k}"] = v
            
        model.load_state_dict(model_state)
        model = model.to(device)
        experts[task] = model
        
        head = nn.Linear(512, 10)
        head.load_state_dict(checkpoint['fc_state_dict'])
        head = head.to(device)
        expert_heads[task] = head

    progenitor = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    progenitor_state = {k: v.to(device) for k, v in progenitor.state_dict().items() if not k.startswith('fc.')}

    # Calibration datasets (128 samples per task)
    calib_loaders_clean = {task: get_dataloader(task, is_train=True, batch_size=128, subset_size=128) for task in ["mnist", "fashion", "cifar"]}
    calib_loaders_mpcs = {task: get_mpcs_dataloader(task, batch_size=64, subset_size=128) for task in ["mnist", "fashion", "cifar"]}

    # Full test loaders (1000 samples subset to speed up evaluation)
    test_loaders = {task: get_dataloader(task, is_train=False, batch_size=256, subset_size=1000) for task in ["mnist", "fashion", "cifar"]}

    deep_layer_names = ["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2"]
    early_layer_name = "layer2.1.conv2"

    merge_methods = ["WA", "TA", "TIES", "DARE"]
    calib_strategies = ["None", "BN-Calib", "SLR-WBC", "MSPR", "REC-SVD", "REC-Routing"]
    
    results = {m: {c: {"clean": [], "corrupted": []} for c in calib_strategies} for m in merge_methods}
    
    print("\nEvaluating Individual Experts (No Merging)...")
    for task in ["mnist", "fashion", "cifar"]:
        acc = evaluate_model(experts[task], test_loaders[task], nn.Identity(), device)
        print(f"Expert {task.upper()} Clean Accuracy: {acc:.2f}%")

    for merge_method in merge_methods:
        print(f"\n==========================================")
        print(f" MERGING METHOD: {merge_method}")
        print(f"==========================================")
        
        merged_state = {}
        if merge_method == "WA":
            for k in progenitor_state.keys():
                merged_state[k] = (experts["mnist"].state_dict()[k] + 
                                   experts["fashion"].state_dict()[k] + 
                                   experts["cifar"].state_dict()[k]) / 3.0
        elif merge_method == "TA":
            lam = 0.4
            tau_mnist = {k: experts["mnist"].state_dict()[k] - progenitor_state[k] for k in progenitor_state.keys()}
            tau_fashion = {k: experts["fashion"].state_dict()[k] - progenitor_state[k] for k in progenitor_state.keys()}
            tau_cifar = {k: experts["cifar"].state_dict()[k] - progenitor_state[k] for k in progenitor_state.keys()}
            for k in progenitor_state.keys():
                merged_state[k] = progenitor_state[k] + lam * (tau_mnist[k] + tau_fashion[k] + tau_cifar[k])
        elif merge_method == "TIES":
            experts_list = [experts["mnist"].state_dict(), experts["fashion"].state_dict(), experts["cifar"].state_dict()]
            merged_state = ties_merging(progenitor_state, experts_list, density=0.2)
        elif merge_method == "DARE":
            experts_list = [experts["mnist"].state_dict(), experts["fashion"].state_dict(), experts["cifar"].state_dict()]
            merged_state = dare_merging(progenitor_state, experts_list, drop_rate=0.5)
                
        def get_merged_backbone():
            mb = resnet18()
            mb.fc = nn.Identity()
            mb_state = mb.state_dict()
            mb_state.update(merged_state)
            mb.load_state_dict(mb_state)
            return mb.to(device)

        # Iterate over calibration/alignment strategies
        for calib_strategy in calib_strategies:
            print(f"\n--- Strategy: {calib_strategy} ---")
            
            # Setup task-specific backbones for this strategy
            backbones = {}
            routing_b = None
            
            if calib_strategy == "None":
                shared_b = get_merged_backbone()
                backbones = {task: shared_b for task in ["mnist", "fashion", "cifar"]}
                
            elif calib_strategy == "BN-Calib":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_clean[task], device=device)
                    backbones[task] = mb
                    
            elif calib_strategy == "SLR-WBC":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_clean[task], device=device)
                    apply_slr_wbc_single_task(mb, experts[task], calib_loaders_clean[task], deep_layer_names, device, rank=8, is_robust=False)
                    backbones[task] = mb
                    
            elif calib_strategy == "MSPR":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_clean[task], device=device)
                    backbones[task] = mb
                # Extract prototypes using progenitor (completely unbiased, uncollapsed)
                progenitor_dict = {t: progenitor for t in ["mnist", "fashion", "cifar"]}
                prototypes = get_mspr_prototypes(progenitor_dict, calib_loaders_clean, early_layer_name, device)
                routing_b = progenitor
                
            elif calib_strategy == "REC-SVD":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_mpcs[task], device=device)
                    apply_slr_wbc_single_task(mb, experts[task], calib_loaders_mpcs[task], deep_layer_names, device, is_robust=True, shrink_c=0.15)
                    backbones[task] = mb
                    
            elif calib_strategy == "REC-Routing":
                for task in ["mnist", "fashion", "cifar"]:
                    mb = get_merged_backbone()
                    calibrate_bn(mb, calib_loaders_mpcs[task], device=device)
                    backbones[task] = mb
                # Extract prototypes using progenitor evaluated on MPCS
                progenitor_dict = {t: progenitor for t in ["mnist", "fashion", "cifar"]}
                prototypes = get_mspr_prototypes(progenitor_dict, calib_loaders_mpcs, early_layer_name, device)
                routing_b = progenitor
                
            # 1. Evaluate CLEAN Performance
            clean_accs = []
            if calib_strategy in ["MSPR", "REC-Routing"]:
                acc = evaluate_mspr_routed(backbones, test_loaders, expert_heads, prototypes, early_layer_name, device, routing_backbone=routing_b)
                clean_accs = [acc, acc, acc]
                print(f"Clean Routed Accuracy: {acc:.2f}%")
            else:
                for task in ["mnist", "fashion", "cifar"]:
                    acc = evaluate_model(backbones[task], test_loaders[task], expert_heads[task], device)
                    clean_accs.append(acc)
                    print(f"Clean Accuracy for {task.upper()}: {acc:.2f}%")
                    
            avg_clean = np.mean(clean_accs)
            results[merge_method][calib_strategy]["clean"] = clean_accs
            print(f"Average Clean Accuracy: {avg_clean:.2f}%")
            
            # 2. Evaluate OOD CORRUPTED Performance
            print("Running exhaustive OOD evaluation...")
            all_corrupted_accs = []
            selected_corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "pixelation", "jpeg_compression"]
            selected_severities = [1, 3, 5]
            
            for corr in selected_corruptions:
                for sev in selected_severities:
                    if calib_strategy in ["MSPR", "REC-Routing"]:
                        acc = evaluate_mspr_routed(backbones, test_loaders, expert_heads, prototypes, early_layer_name, device, corruption_name=corr, severity=sev, routing_backbone=routing_b)
                        all_corrupted_accs.append(acc)
                    else:
                        task_accs = []
                        for task in ["mnist", "fashion", "cifar"]:
                            acc = evaluate_model(backbones[task], test_loaders[task], expert_heads[task], device, corruption_name=corr, severity=sev)
                            task_accs.append(acc)
                        all_corrupted_accs.append(np.mean(task_accs))
                        
            avg_corrupted = np.mean(all_corrupted_accs)
            results[merge_method][calib_strategy]["corrupted"] = all_corrupted_accs
            print(f"Average Corrupted (OOD) Accuracy: {avg_corrupted:.2f}%")

    # Display final comparative summary table
    print("\n" + "="*80)
    print(" FINAL BENCHMARK SUMMARY: CLEAN VS OUT-OF-DISTRIBUTION (OOD) ROBUSTNESS")
    print("="*80)
    print(f"{'Merge':<6} | {'Strategy':<15} | {'Clean MNIST':<11} | {'Clean F-MNIST':<13} | {'Clean CIFAR-10':<14} | {'Clean Avg':<9} | {'OOD Avg':<8}")
    print("-"*100)
    for mm in merge_methods:
        for cs in calib_strategies:
            clean_list = results[mm][cs]["clean"]
            avg_clean = np.mean(clean_list)
            avg_ood = np.mean(results[mm][cs]["corrupted"])
            print(f"{mm:<6} | {cs:<15} | {clean_list[0]:10.2f}% | {clean_list[1]:12.2f}% | {clean_list[2]:13.2f}% | {avg_clean:8.2f}% | {avg_ood:7.2f}%")
    print("="*100)

if __name__ == "__main__":
    main()
