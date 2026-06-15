import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import timm
import random

# Reuse helper classes and functions from run_experiments_optimized
from run_experiments_optimized import (
    set_seed, get_dataset, get_calibration_dataset,
    StaticMergedModel, OFSTuneModel, DynamicMergedModel,
    LinearRouterLinear, QWSLinear, EpiMergeLinear, ExpertHeadsWrapper
)

def profile_memory_and_latency(device):
    print("=== STARTING MEMORY & LATENCY PROFILING ===")
    
    # 1. Setup Models
    print("Loading base and expert models for profiling...")
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    base_model.reset_classifier(0)
    
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    experts = []
    expert_heads = []
    
    for ds in datasets_list:
        model_path = f'checkpoints/{ds.lower()}_expert.pth'
        expert = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        expert.head = nn.Linear(expert.head.in_features, 10)
        expert.load_state_dict(torch.load(model_path, map_location='cpu'))
        expert = expert.to(device)
        expert_heads.append(copy.deepcopy(expert.head))
        expert.reset_classifier(0)
        experts.append(expert)
        
    print("Models loaded successfully. Initializing wrapper models...")
    
    # We will profile three architectures:
    # 1. OFS-Tune (Supervised Static) - Represents shared/standard model overhead
    # 2. Linear Router (Classical Dynamic) - Represents scalar routing overhead
    # 3. EpiMerge (Ours) - Represents coordinate-wise low-rank scaling overhead
    
    ofs_model = OFSTuneModel(base_model, experts, k_tasks=4).to(device)
    wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)
    
    router_model = DynamicMergedModel(base_model, experts, k_tasks=4, latent_dim=4, layer_class=LinearRouterLinear).to(device)
    wrapped_router = ExpertHeadsWrapper(router_model, expert_heads).to(device)
    
    epimerge_model = DynamicMergedModel(base_model, experts, k_tasks=4, latent_dim=4, layer_class=EpiMergeLinear).to(device)
    wrapped_epimerge = ExpertHeadsWrapper(epimerge_model, expert_heads).to(device)
    
    models = {
        'OFS-Tune (Static)': wrapped_ofs,
        'Linear Router': wrapped_router,
        'EpiMerge (Ours)': wrapped_epimerge
    }
    
    batch_sizes = [1, 8, 16, 32, 64]
    
    # Warmup and clear cache
    torch.cuda.empty_cache()
    dummy_input = torch.randn(64, 3, 224, 224).to(device)
    with torch.no_grad():
        for m in models.values():
            if isinstance(m.model, DynamicMergedModel):
                # Set dummy current_h
                m.model.current_h = torch.randn(64, 4).to(device)
            _ = m.model(dummy_input)
            
    print("\nBatch Size | Model | Latency (ms) | Peak GPU Memory (MB) | Mem Overhead vs Static")
    print("-" * 80)
    
    results = {m_name: {'latency': [], 'memory': []} for m_name in models}
    
    for b in batch_sizes:
        x = torch.randn(b, 3, 224, 224).to(device)
        
        # We need task indices to compute output head
        task_indices = [0] * b
        
        for m_name, m in models.items():
            # Clear CUDA cache before memory measurement
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            
            m.eval()
            with torch.no_grad():
                # Measure latency
                # Run multiple warmups and loops
                for _ in range(5):
                    if isinstance(m.model, DynamicMergedModel):
                        # Trigger latent_dim projection
                        mean_tokens = x.mean(dim=1) if len(x.shape) == 3 else x.mean(dim=(2,3)) # dummy mock of Vit token extraction or custom forward
                        # Let's let the forward run naturally which sets current_h internally
                        pass
                    _ = m(x, task_idx=0) # run wrapper forward directly
                
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                iters = 20
                for _ in range(iters):
                    _ = m(x, task_idx=0)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                avg_latency_ms = ((t1 - t0) / iters) * 1000.0
                
                # Measure peak memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                _ = m(x, task_idx=0)
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                
            results[m_name]['latency'].append(avg_latency_ms)
            results[m_name]['memory'].append(peak_mem_mb)
            
    # Print Markdown table format for directly copying into paper
    md_lines = []
    md_lines.append("| Batch Size | Model | Inference Latency (ms) | Peak GPU Memory (MB) | Relative Memory Overhead |")
    md_lines.append("| :--- | :--- | :---: | :---: | :---: |")
    
    for i, b in enumerate(batch_sizes):
        static_mem = results['OFS-Tune (Static)']['memory'][i]
        for m_name in models:
            mem = results[m_name]['memory'][i]
            lat = results[m_name]['latency'][i]
            rel_overhead = f"+{(mem - static_mem):.2f} MB ({((mem - static_mem)/static_mem)*100:.1f}%)" if m_name != 'OFS-Tune (Static)' else "-"
            md_lines.append(f"| B={b} | {m_name} | {lat:.2f} ms | {mem:.2f} MB | {rel_overhead} |")
            print(f"B={b:2d} | {m_name:18s} | {lat:7.2f} ms | {mem:8.2f} MB | {rel_overhead}")
            
    return md_lines

def analyze_routing_dynamics(device):
    print("\n=== STARTING ROUTING DYNAMICS ANALYSIS ===")
    
    # 1. Setup Models
    print("Loading base and expert models...")
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
    base_model.reset_classifier(0)
    
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    experts = []
    expert_heads = []
    
    for ds in datasets_list:
        model_path = f'checkpoints/{ds.lower()}_expert.pth'
        expert = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        expert.head = nn.Linear(expert.head.in_features, 10)
        expert.load_state_dict(torch.load(model_path, map_location='cpu'))
        expert = expert.to(device)
        expert_heads.append(copy.deepcopy(expert.head))
        expert.reset_classifier(0)
        experts.append(expert)
        
    print("Training models briefly to get calibrated parameters...")
    # Setup and calibrate EpiMerge and Linear Router
    cal_dataset = get_calibration_dataset()
    cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=True)
    
    router_model = DynamicMergedModel(base_model, experts, k_tasks=4, latent_dim=4, layer_class=LinearRouterLinear).to(device)
    wrapped_router = ExpertHeadsWrapper(router_model, expert_heads).to(device)
    
    epimerge_model = DynamicMergedModel(base_model, experts, k_tasks=4, latent_dim=4, layer_class=EpiMergeLinear).to(device)
    wrapped_epimerge = ExpertHeadsWrapper(epimerge_model, expert_heads).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Calibrate Linear Router
    print("Calibrating Linear Router (100 steps)...")
    router_model.train()
    optimizer_r = torch.optim.Adam(router_model.parameters(), lr=1e-3)
    for step in range(100):
        for images, labels, task_indices in cal_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_r.zero_grad()
            features = router_model(images)
            outputs = []
            for b in range(images.size(0)):
                task_idx = task_indices[b].item()
                out = wrapped_router.heads[task_idx](features[b:b+1])
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels) * images.size(0)
            loss.backward()
            optimizer_r.step()
            
    # Calibrate EpiMerge
    print("Calibrating EpiMerge (200 steps)...")
    epimerge_model.train()
    optimizer_e = torch.optim.Adam(epimerge_model.parameters(), lr=1e-3)
    for step in range(200):
        for images, labels, task_indices in cal_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_e.zero_grad()
            features = epimerge_model(images)
            outputs = []
            for b in range(images.size(0)):
                task_idx = task_indices[b].item()
                out = wrapped_epimerge.heads[task_idx](features[b:b+1])
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels) * images.size(0)
            loss.backward()
            optimizer_e.step()
            
    print("Calibration finished. Analyzing routing coefficients...")
    
    router_model.eval()
    epimerge_model.eval()
    
    # Collect sample test images from each task
    task_coefficients = {ds: [] for ds in datasets_list}
    task_epimerge_masks_r = {ds: [] for ds in datasets_list}
    task_epimerge_masks_c = {ds: [] for ds in datasets_list}
    
    with torch.no_grad():
        for task_idx, ds in enumerate(datasets_list):
            test_ds = get_dataset(ds, split='val')
            # Select 20 random samples
            indices = list(range(len(test_ds)))
            random.shuffle(indices)
            indices = indices[:20]
            
            samples = []
            for idx in indices:
                img, _ = test_ds[idx]
                samples.append(img)
            x = torch.stack(samples).to(device)
            
            # --- Linear Router Routing Analysis ---
            # Forward to set current_h
            _ = router_model(x)
            h = router_model.current_h # [B, latent_dim]
            # Compute routing coefficients for this task's inputs
            alpha_layers = []
            for name, module in router_model.base_model.named_modules():
                if isinstance(module, LinearRouterLinear):
                    alpha = torch.sigmoid(F.linear(h, module.W_route)) # [B, K]
                    alpha_layers.append(alpha.cpu().numpy())
            # Average across layers and batch
            avg_alpha = np.mean(np.stack(alpha_layers), axis=(0, 1)) # [K]
            task_coefficients[ds] = avg_alpha
            
            # --- EpiMerge Gating Analysis ---
            _ = epimerge_model(x)
            h_e = epimerge_model.current_h
            r_layers = []
            c_layers = []
            for name, module in epimerge_model.base_model.named_modules():
                if isinstance(module, EpiMergeLinear):
                    r = torch.sigmoid(torch.einsum('kod,bd->kbo', module.U, h_e)) # [K, B, D_out]
                    c = torch.sigmoid(torch.einsum('kid,bd->kbi', module.V, h_e)) # [K, B, D_in]
                    # Average over batch and feature dimensions first to resolve shape differences across layers
                    r_layers.append(r.mean(dim=(1, 2)).cpu().numpy()) # [K]
                    c_layers.append(c.mean(dim=(1, 2)).cpu().numpy()) # [K]
            # Average across layers (axis 0) to get task-wise scaling intensity
            avg_r = np.mean(np.stack(r_layers), axis=0) # [K]
            avg_c = np.mean(np.stack(c_layers), axis=0) # [K]
            task_epimerge_masks_r[ds] = avg_r
            task_epimerge_masks_c[ds] = avg_c
            
    print("\n--- TEST-TIME ROUTING COEFFICIENTS (Linear Router) ---")
    for ds in datasets_list:
        coeffs = task_coefficients[ds]
        print(f"Task {ds:12s} input routing coefficients over experts: [MNIST: {coeffs[0]:.3f}, FashionMNIST: {coeffs[1]:.3f}, CIFAR10: {coeffs[2]:.3f}, SVHN: {coeffs[3]:.3f}]")
        
    print("\n--- TEST-TIME EPIGENETIC ROW GATING INTENSITIES (EpiMerge) ---")
    for ds in datasets_list:
        r_masks = task_epimerge_masks_r[ds]
        print(f"Task {ds:12s} input row gating average over experts: [MNIST: {r_masks[0]:.3f}, FashionMNIST: {r_masks[1]:.3f}, CIFAR10: {r_masks[2]:.3f}, SVHN: {r_masks[3]:.3f}]")
        
    print("\n--- TEST-TIME EPIGENETIC COLUMN GATING INTENSITIES (EpiMerge) ---")
    for ds in datasets_list:
        c_masks = task_epimerge_masks_c[ds]
        print(f"Task {ds:12s} input col gating average over experts: [MNIST: {c_masks[0]:.3f}, FashionMNIST: {c_masks[1]:.3f}, CIFAR10: {c_masks[2]:.3f}, SVHN: {c_masks[3]:.3f}]")

    # Generate Markdown summary for routing analysis
    md_lines = []
    md_lines.append("| Input Task | MNIST Expert Weight | FashionMNIST Expert Weight | CIFAR-10 Expert Weight | SVHN Expert Weight |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for ds in datasets_list:
        coeffs = task_coefficients[ds]
        md_lines.append(f"| **{ds}** | {coeffs[0]:.3f} | {coeffs[1]:.3f} | {coeffs[2]:.3f} | {coeffs[3]:.3f} |")
        
    md_lines.append("\n\n| Input Task | MNIST Row Gating | FashionMNIST Row Gating | CIFAR-10 Row Gating | SVHN Row Gating |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for ds in datasets_list:
        r_masks = task_epimerge_masks_r[ds]
        md_lines.append(f"| **{ds}** | {r_masks[0]:.3f} | {r_masks[1]:.3f} | {r_masks[2]:.3f} | {r_masks[3]:.3f} |")
        
    return md_lines

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running profiling script on {device}...")
    
    mem_md = profile_memory_and_latency(device)
    routing_md = analyze_routing_dynamics(device)
    
    # Save markdown outputs to a temporary result file
    with open('profiling_results.md', 'w') as f:
        f.write("# GPU Memory and Latency Profiling Results\n\n")
        f.write("\n".join(mem_md))
        f.write("\n\n# Test-Time Adaptation and Routing Dynamics\n\n")
        f.write("\n".join(routing_md))
    print("\nSaved profiling results to profiling_results.md")
