import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn_2_cuda'] = None

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model

def load_adapter_weights(path):
    # Safely load lora weights from safetensors or bin
    from safetensors.torch import load_file
    sf_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    if os.path.exists(sf_path):
        weights = load_file(sf_path)
    elif os.path.exists(bin_path):
        weights = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {path}")
        
    # Map keys to include '.default.' expected by active PEFT model
    fixed_weights = {}
    for k, v in weights.items():
        if "lora" in k and ".default." not in k:
            new_key = k.replace(".weight", ".default.weight")
            fixed_weights[new_key] = v
        else:
            fixed_weights[k] = v
    return fixed_weights

def merge_direct_parameters(weights_1, weights_2, lam):
    merged = {}
    for key in weights_1.keys():
        if key in weights_2:
            merged[key] = lam * weights_1[key] + (1 - lam) * weights_2[key]
        else:
            merged[key] = weights_1[key]
    return merged

def merge_svd_low_rank(weights_1, weights_2, lam, r=8):
    merged = {}
    # Iterate through weights_1 to find lora_B and lora_A pairs
    for key in weights_1.keys():
        if "lora_B" in key and key.endswith("weight"):
            key_A = key.replace("lora_B", "lora_A")
            if key_A in weights_1 and key in weights_2 and key_A in weights_2:
                B1 = weights_1[key]   # [d_out, r]
                A1 = weights_1[key_A] # [r, d_in]
                B2 = weights_2[key]
                A2 = weights_2[key_A]
                
                # Compute full updates
                dW1 = B1 @ A1
                dW2 = B2 @ A2
                
                # Merge full updates
                dW_merged = lam * dW1 + (1 - lam) * dW2
                
                # Project back to rank r using SVD
                U, S, Vh = torch.linalg.svd(dW_merged, full_matrices=False)
                
                # Reconstruct low-rank matrices B and A
                U_r = U[:, :r]
                S_r = S[:r]
                Vh_r = Vh[:r, :]
                
                B_merged = U_r * torch.sqrt(S_r)
                A_merged = torch.sqrt(S_r).unsqueeze(1) * Vh_r
                
                merged[key] = B_merged
                merged[key_A] = A_merged
        elif "lora" not in key:
            # For non-lora weights (e.g. classifier weights if any), average them
            if key in weights_2:
                merged[key] = lam * weights_1[key] + (1 - lam) * weights_2[key]
            else:
                merged[key] = weights_1[key]
    return merged

def merge_ties_svd(weights_1, weights_2, lam, fraction=0.2, r=8):
    merged = {}
    for key in weights_1.keys():
        if "lora_B" in key and key.endswith("weight"):
            key_A = key.replace("lora_B", "lora_A")
            if key_A in weights_1 and key in weights_2 and key_A in weights_2:
                B1 = weights_1[key]
                A1 = weights_1[key_A]
                B2 = weights_2[key]
                A2 = weights_2[key_A]
                
                dW1 = B1 @ A1
                dW2 = B2 @ A2
                
                # Ties merging on full updates dW1 and dW2
                updates = torch.stack([dW1, dW2]) # [2, d_out, d_in]
                
                # 1. Trim (Pruning)
                # Compute quantile on flattened task vectors to avoid multi-dim tuple issue in torch.quantile
                updates_flat = updates.flatten(start_dim=1)
                thresholds_flat = torch.quantile(torch.abs(updates_flat), 1.0 - fraction, dim=1, keepdim=True)
                thresholds = thresholds_flat.unsqueeze(-1) # [2, 1, 1]
                mask = torch.abs(updates) >= thresholds
                trimmed_updates = updates * mask
                
                # 2. Elect Sign
                signs = torch.sign(trimmed_updates)
                sum_signs = torch.sum(signs, dim=0)
                dominant_sign = torch.sign(sum_signs)
                
                # 3. Disjoint Merge
                same_sign_mask = (torch.sign(trimmed_updates) == dominant_sign.unsqueeze(0)) | (dominant_sign.unsqueeze(0) == 0)
                selected_updates = trimmed_updates * same_sign_mask
                
                # Weighted average of active updates
                counts = torch.sum(same_sign_mask.float(), dim=0)
                counts = torch.clamp(counts, min=1.0)
                
                # Merge
                dW_merged = (lam * selected_updates[0] + (1.0 - lam) * selected_updates[1]) / counts
                
                # Project back to rank r using SVD
                U, S, Vh = torch.linalg.svd(dW_merged, full_matrices=False)
                U_r = U[:, :r]
                S_r = S[:r]
                Vh_r = Vh[:r, :]
                
                B_merged = U_r * torch.sqrt(S_r)
                A_merged = torch.sqrt(S_r).unsqueeze(1) * Vh_r
                
                merged[key] = B_merged
                merged[key_A] = A_merged
        elif "lora" not in key:
            if key in weights_2:
                merged[key] = lam * weights_1[key] + (1 - lam) * weights_2[key]
            else:
                merged[key] = weights_1[key]
    return merged

def evaluate_model(model, dataloader, device, classifier_state=None):
    model.eval()
    if classifier_state is not None:
        model.base_model.model.classifier.load_state_dict(classifier_state)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="Merge and evaluate task-specific LoRA models")
    parser.add_argument("--adapter_cifar", type=str, required=True, help="Path to CIFAR-10 adapter directory")
    parser.add_argument("--adapter_svhn", type=str, required=True, help="Path to SVHN adapter directory")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction for Ties-Merging pruning")
    parser.add_argument("--save_results", type=str, default="results.json")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    print(f"Using device: {device}")
    
    # Image preprocessing
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    # Load test datasets
    print("Loading test datasets...")
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    svhn_test = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform)
    
    # Take a stable subset of 2000 images to accelerate evaluation sweeps
    cifar_subset = torch.utils.data.Subset(cifar_test, list(range(2000)))
    svhn_subset = torch.utils.data.Subset(svhn_test, list(range(2000)))
    
    cifar_loader = DataLoader(cifar_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    svhn_loader = DataLoader(svhn_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Load base model
    print("Loading base model...")
    base_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    
    # Apply standard PEFT LoRA wrapper
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["query", "value"],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    
    # Load adapter weights
    print("Loading task-specific weights...")
    weights_cifar = load_adapter_weights(args.adapter_cifar)
    weights_svhn = load_adapter_weights(args.adapter_svhn)
    
    # Load task-specific classifier heads
    print("Loading task-specific classifier heads...")
    cifar_head_path = os.path.join(args.adapter_cifar, "classifier_head.bin")
    svhn_head_path = os.path.join(args.adapter_svhn, "classifier_head.bin")
    
    if os.path.exists(cifar_head_path):
        classifier_cifar = torch.load(cifar_head_path, map_location="cpu")
        print("Successfully loaded CIFAR-10 classifier head.")
    else:
        classifier_cifar = None
        print(f"Warning: No classifier head found at {cifar_head_path}")
        
    if os.path.exists(svhn_head_path):
        classifier_svhn = torch.load(svhn_head_path, map_location="cpu")
        print("Successfully loaded SVHN classifier head.")
    else:
        classifier_svhn = None
        print(f"Warning: No classifier head found at {svhn_head_path}")
    
    # Sweep merging coefficient lambda
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {
        "DPM": [],
        "SVDM": [],
        "Ties-SVDM": []
    }
    
    for lam in lambdas:
        print(f"\n--- Evaluating lambda = {lam:.1f} ---")
        
        # 1. Direct Parameter Merging (DPM)
        print("Merging via DPM...")
        merged_dpm = merge_direct_parameters(weights_cifar, weights_svhn, lam)
        # Load merged weights into the model
        model.load_state_dict(merged_dpm, strict=False)
        cifar_acc = evaluate_model(model, cifar_loader, device, classifier_cifar)
        svhn_acc = evaluate_model(model, svhn_loader, device, classifier_svhn)
        avg_acc = (cifar_acc + svhn_acc) / 2.0
        print(f"DPM: CIFAR-10 Acc = {cifar_acc:.2f}%, SVHN Acc = {svhn_acc:.2f}%, Avg = {avg_acc:.2f}%")
        results["DPM"].append({"lambda": lam, "cifar_acc": cifar_acc, "svhn_acc": svhn_acc, "avg_acc": avg_acc})
        
        # 2. SVD Low-Rank Merging (SVDM)
        print("Merging via SVDM...")
        merged_svdm = merge_svd_low_rank(weights_cifar, weights_svhn, lam, r=args.r)
        model.load_state_dict(merged_svdm, strict=False)
        cifar_acc = evaluate_model(model, cifar_loader, device, classifier_cifar)
        svhn_acc = evaluate_model(model, svhn_loader, device, classifier_svhn)
        avg_acc = (cifar_acc + svhn_acc) / 2.0
        print(f"SVDM: CIFAR-10 Acc = {cifar_acc:.2f}%, SVHN Acc = {svhn_acc:.2f}%, Avg = {avg_acc:.2f}%")
        results["SVDM"].append({"lambda": lam, "cifar_acc": cifar_acc, "svhn_acc": svhn_acc, "avg_acc": avg_acc})
        
        # 3. Ties-SVDM Merging
        print("Merging via Ties-SVDM...")
        merged_ties = merge_ties_svd(weights_cifar, weights_svhn, lam, fraction=args.fraction, r=args.r)
        model.load_state_dict(merged_ties, strict=False)
        cifar_acc = evaluate_model(model, cifar_loader, device, classifier_cifar)
        svhn_acc = evaluate_model(model, svhn_loader, device, classifier_svhn)
        avg_acc = (cifar_acc + svhn_acc) / 2.0
        print(f"Ties-SVDM: CIFAR-10 Acc = {cifar_acc:.2f}%, SVHN Acc = {svhn_acc:.2f}%, Avg = {avg_acc:.2f}%")
        results["Ties-SVDM"].append({"lambda": lam, "cifar_acc": cifar_acc, "svhn_acc": svhn_acc, "avg_acc": avg_acc})
        
    # Save results
    with open(args.save_results, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved all merging results to {args.save_results}")

if __name__ == "__main__":
    main()
