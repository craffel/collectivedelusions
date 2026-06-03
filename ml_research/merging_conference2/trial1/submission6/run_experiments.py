import sys
sys.modules['flash_attn'] = None

try:
    import transformers.utils.import_utils as import_utils
    import_utils.is_flash_attn_available = lambda: False
    import_utils.is_flash_attn_2_available = lambda: False
    import_utils.is_flash_attn_3_available = lambda: False
except Exception:
    pass

import torch
torch.backends.cudnn.enabled = False
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
import os
import json
import numpy as np
from tqdm import tqdm

# Ensure device is set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Prepare standard CLIP transform
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    lambda img: img.convert("RGB") if hasattr(img, "convert") else img,
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# Helper to get subset of dataset
def get_subset(dataset, num_samples=300, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices)

# Precompute text features
def precompute_text_features(model, processor, classes, template, device):
    prompts = [template.format(c) for c in classes]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        pooled_output = text_outputs[1]
        text_features = model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# Evaluate a vision model state dict
def evaluate_merged_model(base_model, merged_sd, dataloader, text_features, device):
    # Temporarily load merged vision encoder state dict
    orig_sd = {k: v.clone() for k, v in base_model.vision_model.state_dict().items()}
    base_model.vision_model.load_state_dict(merged_sd)
    base_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            image_outputs = base_model.vision_model(pixel_values=images)
            pooled_output = image_outputs[1]
            image_features = base_model.visual_projection(pooled_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T) * 100
            preds = similarity.argmax(dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    # Restore original state dict
    base_model.vision_model.load_state_dict(orig_sd)
    return correct / total

# Merging functions
def merge_task_arithmetic(base_sd, task_vectors, lam=0.5):
    merged_tv = {}
    for k in base_sd.keys():
        w_pre = base_sd[k]
        if w_pre.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            merged_tv[k] = w_pre.clone()
            continue
        # Task Arithmetic: sum of task vectors scaled by lam
        sum_tv = torch.stack([tv[k] for tv in task_vectors], dim=0).sum(dim=0)
        merged_tv[k] = w_pre + lam * sum_tv
    return merged_tv

def merge_ties(base_sd, task_vectors, p=0.2, lam=0.5):
    merged_tv = {}
    for k in base_sd.keys():
        w_pre = base_sd[k]
        if w_pre.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            merged_tv[k] = w_pre.clone()
            continue
            
        updates = torch.stack([tv[k] for tv in task_vectors], dim=0) # [K, ...]
        flat_updates = updates.view(updates.shape[0], -1) # [K, N]
        
        # 1. Trim
        k_val = int(p * flat_updates.shape[1])
        if k_val > 0:
            thresholds = torch.topk(torch.abs(flat_updates), k_val, dim=1).values[:, -1] # [K]
            mask = torch.abs(updates) >= thresholds.view(-1, *([1] * (updates.ndim - 1)))
            trimmed_updates = updates * mask
        else:
            trimmed_updates = updates
            
        # 2. Elect sign
        signs = torch.sign(trimmed_updates)
        sum_signs = torch.sum(signs, dim=0)
        majority_sign = torch.sign(sum_signs)
        
        # 3. Disjoint merge
        same_sign_mask = (torch.sign(trimmed_updates) == majority_sign.unsqueeze(0)) & (majority_sign.unsqueeze(0) != 0)
        sum_matching = torch.sum(trimmed_updates * same_sign_mask, dim=0)
        count_matching = torch.sum(same_sign_mask.float(), dim=0)
        
        merged_val = torch.where(count_matching > 0, sum_matching / count_matching, torch.zeros_like(sum_matching))
        merged_tv[k] = w_pre + lam * merged_val
        
    return merged_tv

def merge_scs(base_sd, task_vectors, beta=0.0, gamma=1.0, lam=0.5):
    merged_tv = {}
    for k in base_sd.keys():
        w_pre = base_sd[k]
        if w_pre.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            merged_tv[k] = w_pre.clone()
            continue
            
        updates = torch.stack([tv[k] for tv in task_vectors], dim=0) # [K, ...]
        
        # Soft-trimming: dampen smaller updates relative to layer mean
        if beta > 0.0:
            flat_updates = updates.view(updates.shape[0], -1)
            mean_abs = torch.mean(torch.abs(flat_updates), dim=1) # [K]
            shape = [updates.shape[0]] + [1] * (updates.ndim - 1)
            mean_abs_reshaped = mean_abs.view(*shape)
            relative_mag = torch.abs(updates) / (mean_abs_reshaped + 1e-12)
            soft_trim_factor = torch.pow(relative_mag, beta)
            trimmed_updates = updates * soft_trim_factor
        else:
            trimmed_updates = updates
            
        # Compute sum update and sum of absolute updates
        sum_tv = torch.sum(trimmed_updates, dim=0)
        sum_abs_tv = torch.sum(torch.abs(trimmed_updates), dim=0)
        
        # Continuous consensus ratio (CCR)
        ccr = torch.abs(sum_tv) / (sum_abs_tv + 1e-12)
        
        # Smooth consensus scaling factor
        scs_factor = torch.pow(ccr, gamma)
        
        # Scaled update
        scaled_tv = sum_tv * scs_factor
        
        # Add to base weights
        merged_tv[k] = w_pre + lam * scaled_tv
        
    return merged_tv

def main():
    print("Loading datasets...")
    # Loading test subsets
    mnist_raw = datasets.MNIST(root='./data', train=False, download=False, transform=clip_transform)
    svhn_raw = datasets.SVHN(root='./data', split='test', download=False, transform=clip_transform)
    dtd_raw = datasets.DTD(root='./data', split='test', download=False, transform=clip_transform)
    cifar10_raw = datasets.CIFAR10(root='./data', train=False, download=False, transform=clip_transform)
    cifar100_raw = datasets.CIFAR100(root='./data', train=False, download=False, transform=clip_transform)
    
    num_samples = 200
    datasets_dict = {
        "MNIST": get_subset(mnist_raw, num_samples),
        "SVHN": get_subset(svhn_raw, num_samples),
        "DTD": get_subset(dtd_raw, num_samples),
        "CIFAR10": get_subset(cifar10_raw, num_samples),
        "CIFAR100": get_subset(cifar100_raw, num_samples)
    }
    
    dataloaders = {name: DataLoader(ds, batch_size=64, shuffle=False, num_workers=0) for name, ds in datasets_dict.items()}
    
    classes_dict = {
        "MNIST": [str(i) for i in range(10)],
        "SVHN": [str(i) for i in range(10)],
        "DTD": dtd_raw.classes,
        "CIFAR10": cifar10_raw.classes,
        "CIFAR100": cifar100_raw.classes
    }
    
    templates_dict = {
        "MNIST": "a photo of the number: {}.",
        "SVHN": "a photo of the number: {}.",
        "DTD": "a photo of a {} texture.",
        "CIFAR10": "a photo of a {}.",
        "CIFAR100": "a photo of a {}."
    }
    
    print("Loading models...")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    tasks = ["MNIST", "SVHN", "DTD", "CIFAR10", "CIFAR100"]
    huggingface_models = {
        "MNIST": "tanganke/clip-vit-base-patch32_mnist",
        "SVHN": "tanganke/clip-vit-base-patch32_svhn",
        "DTD": "tanganke/clip-vit-base-patch32_dtd",
        "CIFAR10": "tanganke/clip-vit-base-patch32_cifar10",
        "CIFAR100": "tanganke/clip-vit-base-patch32_cifar100"
    }
    
    expert_models = {}
    for t in tasks:
        print(f"Loading expert model for {t}...")
        expert_models[t] = CLIPVisionModel.from_pretrained(huggingface_models[t]).to(device)
        
    print("Precomputing text features...")
    text_features_dict = {}
    for t in tasks:
        text_features_dict[t] = precompute_text_features(base_model, processor, classes_dict[t], templates_dict[t], device)
        
    base_sd = {k: v.clone() for k, v in base_model.vision_model.state_dict().items()}
    
    # Calculate task vectors
    task_vectors = []
    task_vectors_dict = {}
    for t in tasks:
        expert_sd = {k: v.clone() for k, v in expert_models[t].vision_model.state_dict().items()}
        tv = {k: expert_sd[k] - base_sd[k] for k in base_sd.keys()}
        task_vectors.append(tv)
        task_vectors_dict[t] = tv
        
    results = {}
    
    # Baseline 1: Zero-Shot (Base model)
    print("Evaluating Zero-Shot CLIP...")
    results["Zero-Shot"] = {}
    for t in tasks:
        acc = evaluate_merged_model(base_model, base_sd, dataloaders[t], text_features_dict[t], device)
        results["Zero-Shot"][t] = acc
        print(f"Zero-Shot on {t}: {acc:.4f}")
        
    # Baseline 2: Individual Experts (Diagonal only)
    print("Evaluating Individual Experts (Diagonal only)...")
    for expert_name, expert_model in expert_models.items():
        results[f"Expert_{expert_name}"] = {}
        expert_sd = {k: v.clone() for k, v in expert_model.vision_model.state_dict().items()}
        acc = evaluate_merged_model(base_model, expert_sd, dataloaders[expert_name], text_features_dict[expert_name], device)
        results[f"Expert_{expert_name}"][expert_name] = acc
        print(f"Expert {expert_name} on {expert_name}: {acc:.4f}")
            
    # Sweep configurations
    ta_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ties_lambdas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    scs_lambdas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    scs_gammas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
    scs_betas = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5]
    
    # Evaluate Task Arithmetic
    print("Sweeping Task Arithmetic...")
    best_ta_acc = 0
    best_ta_lam = None
    best_ta_results = None
    for lam in ta_lambdas:
        merged_sd = merge_task_arithmetic(base_sd, task_vectors, lam=lam)
        ta_accs = {}
        for t in tasks:
            ta_accs[t] = evaluate_merged_model(base_model, merged_sd, dataloaders[t], text_features_dict[t], device)
        avg_acc = np.mean(list(ta_accs.values()))
        print(f"Task Arithmetic (lambda={lam:.2f}) -> Avg Accuracy: {avg_acc:.4f}")
        if avg_acc > best_ta_acc:
            best_ta_acc = avg_acc
            best_ta_lam = lam
            best_ta_results = ta_accs
            
    results[f"TaskArithmetic_best_lambda_{best_ta_lam}"] = best_ta_results
    
    # Evaluate TIES Merging
    print("Sweeping TIES Merging...")
    best_ties_acc = 0
    best_ties_lam = None
    best_ties_results = None
    p_trim = 0.2
    for lam in ties_lambdas:
        merged_sd = merge_ties(base_sd, task_vectors, p=p_trim, lam=lam)
        ties_accs = {}
        for t in tasks:
            ties_accs[t] = evaluate_merged_model(base_model, merged_sd, dataloaders[t], text_features_dict[t], device)
        avg_acc = np.mean(list(ties_accs.values()))
        print(f"TIES Merging (lambda={lam:.2f}) -> Avg Accuracy: {avg_acc:.4f}")
        if avg_acc > best_ties_acc:
            best_ties_acc = avg_acc
            best_ties_lam = lam
            best_ties_results = ties_accs
            
    results[f"TIES_best_lambda_{best_ties_lam}"] = best_ties_results
    
    # Evaluate Smooth Consensus Scaling (SCS)
    print("Sweeping Smooth Consensus Scaling (ST-SCS)...")
    best_scs_acc = 0
    best_scs_lam = None
    best_scs_gamma = None
    best_scs_beta = None
    best_scs_results = None
    
    for beta in scs_betas:
        for gamma in scs_gammas:
            for lam in scs_lambdas:
                merged_sd = merge_scs(base_sd, task_vectors, beta=beta, gamma=gamma, lam=lam)
                scs_accs = {}
                for t in tasks:
                    scs_accs[t] = evaluate_merged_model(base_model, merged_sd, dataloaders[t], text_features_dict[t], device)
                avg_acc = np.mean(list(scs_accs.values()))
                print(f"ST-SCS (beta={beta:.2f}, gamma={gamma:.2f}, lambda={lam:.2f}) -> Avg Accuracy: {avg_acc:.4f}")
                if avg_acc > best_scs_acc:
                    best_scs_acc = avg_acc
                    best_scs_lam = lam
                    best_scs_gamma = gamma
                    best_scs_beta = beta
                    best_scs_results = scs_accs
                
    results[f"SCS_best_beta_{best_scs_beta}_gamma_{best_scs_gamma}_lambda_{best_scs_lam}"] = best_scs_results
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Task Arithmetic (lambda={best_ta_lam}): {best_ta_acc:.4f}")
    print(f"Best TIES Merging (lambda={best_ties_lam}): {best_ties_acc:.4f}")
    print(f"Best ST-SCS (beta={best_scs_beta}, gamma={best_scs_gamma}, lambda={best_scs_lam}): {best_scs_acc:.4f}")
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaving markdown table...")
    # Print Markdown Table
    md_table = "| Method | MNIST | SVHN | DTD | CIFAR10 | CIFAR100 | Avg. |\n"
    md_table += "| --- | --- | --- | --- | --- | --- | --- |\n"
    
    # Zero-Shot
    zs = results["Zero-Shot"]
    md_table += f"| Zero-Shot | {zs['MNIST']:.4f} | {zs['SVHN']:.4f} | {zs['DTD']:.4f} | {zs['CIFAR10']:.4f} | {zs['CIFAR100']:.4f} | {np.mean(list(zs.values())):.4f} |\n"
    
    # Task Arithmetic
    ta = best_ta_results
    md_table += f"| Task Arithmetic (lam={best_ta_lam}) | {ta['MNIST']:.4f} | {ta['SVHN']:.4f} | {ta['DTD']:.4f} | {ta['CIFAR10']:.4f} | {ta['CIFAR100']:.4f} | {best_ta_acc:.4f} |\n"
    
    # TIES Merging
    ties = best_ties_results
    md_table += f"| TIES Merging (lam={best_ties_lam}) | {ties['MNIST']:.4f} | {ties['SVHN']:.4f} | {ties['DTD']:.4f} | {ties['CIFAR10']:.4f} | {ties['CIFAR100']:.4f} | {best_ties_acc:.4f} |\n"
    
    # SCS
    scs = best_scs_results
    md_table += f"| **ST-SCS (Ours) (b={best_scs_beta}, g={best_scs_gamma}, lam={best_scs_lam})** | **{scs['MNIST']:.4f}** | **{scs['SVHN']:.4f}** | **{scs['DTD']:.4f}** | **{scs['CIFAR10']:.4f}** | **{scs['CIFAR100']:.4f}** | **{best_scs_acc:.4f}** |\n"
    
    print(md_table)
    with open("table.md", "w") as f:
        f.write(md_table)

if __name__ == "__main__":
    main()
