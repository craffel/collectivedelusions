import torch
import numpy as np
import open_clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)

target_layers = []
for i in range(12):
    block = model.visual.transformer.resblocks[i]
    target_layers.append((f"Block {i} Attn Out Proj", block.attn.out_proj.weight))
    target_layers.append((f"Block {i} MLP c_fc", block.mlp.c_fc.weight))
    target_layers.append((f"Block {i} MLP c_proj", block.mlp.c_proj.weight))

K = 3
scales = [0.1, 0.5, 2.0]
seeds = [101, 102, 103]

layer_stats = {}
for name, weight in target_layers:
    layer_stats[name] = {'alpha': [], 'lambda': []}

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for name, W_pre_orig in target_layers:
        W_pre = W_pre_orig.detach().to(device).float()
        d1, d2 = W_pre.shape
        
        task_vectors = []
        for k in range(K):
            raw_update = torch.randn(d1, d2, device=device) / np.sqrt(d2)
            rms_raw = torch.sqrt((raw_update**2).mean() + 1e-8)
            tau = scales[k] * (raw_update / rms_raw)
            task_vectors.append(tau)
            
        rms_vals = [torch.sqrt((t**2).mean() + 1e-8) for t in task_vectors]
        norm_vectors = [t / r for t, r in zip(task_vectors, rms_vals)]
        avg_norm = sum(norm_vectors) / len(norm_vectors)
        alpha = torch.sqrt((avg_norm**2).mean() + 1e-8).item()
        
        layer_stats[name]['alpha'].append(alpha)
        layer_stats[name]['lambda'].append(1.0 / alpha)

print(f"--- Layer-wise alignment stats across {len(seeds)} seeds ---")
# Group by block index and print average stats
for i in range(12):
    attn_alpha = np.mean(layer_stats[f"Block {i} Attn Out Proj"]['alpha'])
    attn_lam = np.mean(layer_stats[f"Block {i} Attn Out Proj"]['lambda'])
    cfc_alpha = np.mean(layer_stats[f"Block {i} MLP c_fc"]['alpha'])
    cfc_lam = np.mean(layer_stats[f"Block {i} MLP c_fc"]['lambda'])
    cproj_alpha = np.mean(layer_stats[f"Block {i} MLP c_proj"]['alpha'])
    cproj_lam = np.mean(layer_stats[f"Block {i} MLP c_proj"]['lambda'])
    
    print(f"Block {i:2d}:")
    print(f"  Attn Out Proj: alpha = {attn_alpha:.4f}, lambda = {attn_lam:.4f}")
    print(f"  MLP c_fc     : alpha = {cfc_alpha:.4f}, lambda = {cfc_lam:.4f}")
    print(f"  MLP c_proj   : alpha = {cproj_alpha:.4f}, lambda = {cproj_lam:.4f}")
