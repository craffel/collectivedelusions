import os
import torch
import torch.nn as nn
from evaluate_merging import (
    get_dataloaders,
    load_lora_updates,
    load_task_heads,
    evaluate_multi_task,
    quantize_dequantize_weight,
    apply_weights,
    restore_weights
)
import timm

def run_global_saws(base_model, original_weights, fp16_merged_weights, expert_updates, cal_batches, loaders, task_heads, device, bits, sym, pc):
    # Standard Global SAWS
    saws_weights = {}
    c_factors = {}
    alpha_constant = 0.08
    
    for l in range(12):
        merged_adapter_update = 0.25 * sum(expert_updates[t][l] for t in ["mnist", "fashionmnist", "cifar10", "svhn"])
        base_frob = torch.norm(original_weights[l], p="fro")
        adapter_frob = torch.norm(merged_adapter_update, p="fro")
        gamma = alpha_constant * (base_frob / (adapter_frob + 1e-8))
        
        # Scaled weights
        scaled_merged = original_weights[l] + gamma * merged_adapter_update
        saws_weights[l] = quantize_dequantize_weight(scaled_merged, bits, sym, pc)
        
    with torch.no_grad():
        for l in range(12):
            w_ref = fp16_merged_weights[l].view(-1)
            w_quant = saws_weights[l].view(-1)
            c_l = torch.dot(w_ref, w_quant) / (torch.norm(w_quant)**2 + 1e-8)
            c_factors[l] = c_l.item()
            
    hooks = []
    for l in range(12):
        layer = base_model.blocks[l].attn.qkv
        def get_hook(scale):
            return lambda module, inp, out: out * scale
        h = layer.register_forward_hook(get_hook(c_factors[l]))
        hooks.append(h)
        
    apply_weights(base_model, saws_weights)
    res = evaluate_multi_task(base_model, task_heads, loaders, device)
    
    # Clean up
    for h in hooks:
        h.remove()
    restore_weights(base_model, original_weights)
    return res

def run_channel_wise_saws(base_model, original_weights, fp16_merged_weights, expert_updates, cal_batches, loaders, task_heads, device, bits, sym, pc):
    # Row-Specific (Channel-Wise) SAWS
    saws_weights = {}
    c_factors = {}
    alpha_constant = 0.08
    
    for l in range(12):
        merged_adapter_update = 0.25 * sum(expert_updates[t][l] for t in ["mnist", "fashionmnist", "cifar10", "svhn"])
        
        # Compute row-wise Frobenius (L2) norms (dim=1)
        base_frob = torch.norm(original_weights[l], p=2, dim=1, keepdim=True) # [out_features, 1]
        adapter_frob = torch.norm(merged_adapter_update, p=2, dim=1, keepdim=True) # [out_features, 1]
        gamma_row = alpha_constant * (base_frob / (adapter_frob + 1e-8)) # [out_features, 1]
        
        # Scaled weights
        scaled_merged = original_weights[l] + gamma_row * merged_adapter_update
        saws_weights[l] = quantize_dequantize_weight(scaled_merged, bits, sym, pc)
        
    with torch.no_grad():
        for l in range(12):
            w_ref = fp16_merged_weights[l] # [out_features, in_features]
            w_quant = saws_weights[l] # [out_features, in_features]
            # Compute row-wise inner products and squared norms
            numerator = torch.sum(w_ref * w_quant, dim=1) # [out_features]
            denominator = torch.sum(w_quant * w_quant, dim=1) + 1e-8 # [out_features]
            c_l_row = numerator / denominator # [out_features]
            c_factors[l] = c_l_row.to(device)
            
    hooks = []
    for l in range(12):
        layer = base_model.blocks[l].attn.qkv
        def get_hook(scale_vector):
            return lambda module, inp, out: out * scale_vector
        h = layer.register_forward_hook(get_hook(c_factors[l]))
        hooks.append(h)
        
    apply_weights(base_model, saws_weights)
    res = evaluate_multi_task(base_model, task_heads, loaders, device)
    
    # Clean up
    for h in hooks:
        h.remove()
    restore_weights(base_model, original_weights)
    return res

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SAWS Ablation running on: {device}")
    
    loaders, cal_batches = get_dataloaders()
    expert_updates = load_lora_updates()
    task_heads = load_task_heads()
    
    # Load base model
    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    base_model.eval()
    
    original_weights = {}
    for l in range(12):
        original_weights[l] = base_model.blocks[l].attn.qkv.weight.clone()
        original_weights[l] = original_weights[l].to(device)
        
    fp16_merged_weights = {}
    for l in range(12):
        fp16_merged_weights[l] = original_weights[l] + 0.25 * sum(expert_updates[t][l] for t in ["mnist", "fashionmnist", "cifar10", "svhn"])
        fp16_merged_weights[l] = fp16_merged_weights[l].to(device)
        
    configs = [
        ("INT8 Symmetric Per-Channel", 8, True, True),
        ("INT4 Symmetric Per-Channel", 4, True, True),
        ("INT4 Asymmetric Per-Channel", 4, False, True),
        ("INT4 Symmetric Per-Tensor", 4, True, False)
    ]
    
    print("\nStarting SAWS Ablation (Global vs. Channel-wise)...")
    results = {}
    for name, bits, sym, pc in configs:
        print(f"\nEvaluating: {name}")
        global_res = run_global_saws(base_model, original_weights, fp16_merged_weights, expert_updates, cal_batches, loaders, task_heads, device, bits, sym, pc)
        channel_res = run_channel_wise_saws(base_model, original_weights, fp16_merged_weights, expert_updates, cal_batches, loaders, task_heads, device, bits, sym, pc)
        
        global_mean = sum(global_res.values()) / 4.0
        channel_mean = sum(channel_res.values()) / 4.0
        
        results[name] = {
            "global": global_res,
            "global_mean": global_mean,
            "channel": channel_res,
            "channel_mean": channel_mean
        }
        print(f"  Global SAWS Mean Acc: {global_mean:.2f}%")
        print(f"  Channel-wise SAWS Mean Acc: {channel_mean:.2f}%")
        
    # Format a Markdown table of results
    print("\n### SAWS Ablation Summary Table")
    print("| Quantization Configuration | Global SAWS (Mean Acc) | Channel-wise SAWS (Mean Acc) | Difference |")
    print("|---|---|---|---|")
    for name in results:
        g = results[name]["global_mean"]
        c = results[name]["channel_mean"]
        diff = c - g
        print(f"| {name} | {g:.2f}% | {c:.2f}% | {diff:+.2f}% |")

if __name__ == "__main__":
    main()
