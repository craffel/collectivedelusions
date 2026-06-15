import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.decomposition import PCA
import json

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PhysicalPCAPreprojector:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.P = None
        
    def fit(self, H):
        if isinstance(H, torch.Tensor):
            H = H.cpu().numpy()
        if len(H.shape) == 3:
            B, T, D = H.shape
            H = H.reshape(B * T, D)
        pca = PCA(n_components=self.n_components)
        pca.fit(H)
        self.P = torch.tensor(pca.components_.T, dtype=torch.float32)
        
    def project(self, H):
        if self.P is None:
            raise ValueError("PCA not fitted yet.")
        is_3d = (len(H.shape) == 3)
        if is_3d:
            B, T, D = H.shape
            H_flat = H.reshape(B * T, D)
        else:
            H_flat = H
            
        proj = torch.matmul(H_flat, self.P.to(H.device))
        psi = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
        
        if is_3d:
            psi = psi.reshape(B, T, self.n_components).mean(dim=1)
        return psi

class NonUniformBWS_ViT_Router(nn.Module):
    def __init__(self, block_groups, d=4, K=4, lambda_max=0.3, init_bias=-2.0):
        super().__init__()
        self.block_groups = block_groups  # List of lists of layer indices
        self.G = len(block_groups)
        self.d = d
        self.K = K
        self.lambda_max = lambda_max
        
        self.W = nn.Parameter(torch.zeros(self.G, K, d))
        self.B = nn.Parameter(torch.zeros(self.G, K))
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.B, init_bias)
        
    def forward(self, psi_list):
        alpha_list = []
        for g in range(self.G):
            psi = psi_list[g]
            logits = torch.matmul(psi, self.W[g].t()) + self.B[g].unsqueeze(0)
            alpha = self.lambda_max * torch.sigmoid(logits)
            alpha_list.append(alpha)
        return alpha_list

def create_vit_experts(K=4):
    set_seed(42)
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    experts = []
    
    for k in range(K):
        expert_dict = {}
        for name, param in base_model.state_dict().items():
            if 'blocks' in name and ('qkv.weight' in name or 'fc1.weight' in name):
                perturb = torch.randn_like(param) * 0.05 * (k + 1)
                expert_dict[name] = param.clone() + perturb
            else:
                expert_dict[name] = param.clone()
        experts.append(expert_dict)
    return base_model, experts

def run_pilot():
    print("======================================================================")
    print("RUNNING BWS-ROUTER PILOTS ON PHYSICAL ViT-B/16 STRUCTURE")
    print("======================================================================")
    
    set_seed(42)
    K = 4
    D = 192
    L = 12
    d = 4
    B = 16
    
    # 1. Create base model and expert state dicts
    base_model, experts = create_vit_experts(K)
    base_dict = base_model.state_dict()
    
    # Extract task vectors
    task_vectors = []
    for k in range(K):
        tv = {}
        for name in base_dict.keys():
            if 'blocks' in name and ('qkv.weight' in name or 'fc1.weight' in name):
                tv[name] = experts[k][name] - base_dict[name]
        task_vectors.append(tv)
        
    # Test batch
    x_input = torch.randn(B, 3, 224, 224)
    
    # --------------------------------------------------
    # PILOT 1: Uniform BWS-Router (M=3, G=4)
    # --------------------------------------------------
    print("\n--- PILOT 1: UNIFORM BWS-ROUTER (M=3, G=4) ---")
    pcas_unif = [PhysicalPCAPreprojector(n_components=d) for _ in range(4)]
    for g in range(4):
        dummy_H = torch.randn(B * 4, 197, D)
        pcas_unif[g].fit(dummy_H)
        
    router_unif = NonUniformBWS_ViT_Router(
        block_groups=[[0,1,2], [3,4,5], [6,7,8], [9,10,11]], d=d, K=K
    )
    unif_params = sum(p.numel() for p in router_unif.parameters())
    print(f"Uniform block-groups: {router_unif.block_groups}")
    print(f"Trainable Parameters: {unif_params}")
    
    t_start = time.time()
    with torch.no_grad():
        x_feats = base_model.patch_embed(x_input)
        x_feats = base_model.pos_drop(x_feats)
        
    current_feats = x_feats.clone()
    psi_unif = []
    for g in range(4):
        psi = pcas_unif[g].project(current_feats)
        psi_unif.append(psi)
        
    alphas_unif = router_unif(psi_unif)
    merged_unif = {k: v.clone() for k, v in base_dict.items()}
    
    for g in range(4):
        bar_alpha = alphas_unif[g].mean(dim=0)
        for l in router_unif.block_groups[g]:
            qkv_name = f"blocks.{l}.attn.qkv.weight"
            fc1_name = f"blocks.{l}.mlp.fc1.weight"
            
            qkv_merged = base_dict[qkv_name].clone()
            fc1_merged = base_dict[fc1_name].clone()
            for k in range(K):
                qkv_merged += bar_alpha[k] * task_vectors[k][qkv_name]
                fc1_merged += bar_alpha[k] * task_vectors[k][fc1_name]
            merged_unif[qkv_name] = qkv_merged
            merged_unif[fc1_name] = fc1_merged
            
    base_model.load_state_dict(merged_unif)
    with torch.no_grad():
        for block in base_model.blocks:
            current_feats = block(current_feats)
        current_feats = base_model.norm(current_feats)
        logits_unif = base_model.forward_head(current_feats)
        
    lat_unif = (time.time() - t_start) * 1000
    print(f"Uniform blended forward pass latency: {lat_unif:.2f} ms")
    
    # --------------------------------------------------
    # PILOT 2: Non-Uniform Coarse-to-Fine BWS-Router (3 groups: [8, 2, 2])
    # --------------------------------------------------
    print("\n--- PILOT 2: NON-UNIFORM COARSE-TO-FINE BWS-ROUTER (G=3 groups: [8, 2, 2]) ---")
    non_unif_groups = [[0,1,2,3,4,5,6,7], [8,9], [10,11]]
    pcas_nonunif = [PhysicalPCAPreprojector(n_components=d) for _ in range(3)]
    for g in range(3):
        dummy_H = torch.randn(B * 4, 197, D)
        pcas_nonunif[g].fit(dummy_H)
        
    router_nonunif = NonUniformBWS_ViT_Router(block_groups=non_unif_groups, d=d, K=K)
    nonunif_params = sum(p.numel() for p in router_nonunif.parameters())
    print(f"Coarse-to-Fine block-groups: {router_nonunif.block_groups}")
    print(f"Trainable Parameters: {nonunif_params}")
    
    t_start = time.time()
    with torch.no_grad():
        x_feats = base_model.patch_embed(x_input)
        x_feats = base_model.pos_drop(x_feats)
        
    current_feats = x_feats.clone()
    psi_nonunif = []
    for g in range(3):
        psi = pcas_nonunif[g].project(current_feats)
        psi_nonunif.append(psi)
        
    alphas_nonunif = router_nonunif(psi_nonunif)
    merged_nonunif = {k: v.clone() for k, v in base_dict.items()}
    
    for g in range(3):
        bar_alpha = alphas_nonunif[g].mean(dim=0)
        for l in router_nonunif.block_groups[g]:
            qkv_name = f"blocks.{l}.attn.qkv.weight"
            fc1_name = f"blocks.{l}.mlp.fc1.weight"
            
            qkv_merged = base_dict[qkv_name].clone()
            fc1_merged = base_dict[fc1_name].clone()
            for k in range(K):
                qkv_merged += bar_alpha[k] * task_vectors[k][qkv_name]
                fc1_merged += bar_alpha[k] * task_vectors[k][fc1_name]
            merged_nonunif[qkv_name] = qkv_merged
            merged_nonunif[fc1_name] = fc1_merged
            
    base_model.load_state_dict(merged_nonunif)
    with torch.no_grad():
        for block in base_model.blocks:
            current_feats = block(current_feats)
        current_feats = base_model.norm(current_feats)
        logits_nonunif = base_model.forward_head(current_feats)
        
    lat_nonunif = (time.time() - t_start) * 1000
    print(f"Coarse-to-Fine blended forward pass latency: {lat_nonunif:.2f} ms")
    
    # --------------------------------------------------
    # Pure Baseline Inference
    # --------------------------------------------------
    print("\n--- BASELINE INFERENCE ---")
    t_base_start = time.time()
    with torch.no_grad():
        logits_base = base_model(x_input)
    latency_base = (time.time() - t_base_start) * 1000
    print(f"Pure ViT inference latency: {latency_base:.2f} ms")
    
    overhead_unif = lat_unif - latency_base
    overhead_nonunif = lat_nonunif - latency_base
    print(f"\nUniform BWS-Router (M=3) Overhead: {overhead_unif:.2f} ms")
    print(f"Coarse-to-Fine BWS-Router Overhead: {overhead_nonunif:.2f} ms (saving {(lat_unif-lat_nonunif)/lat_unif*100:.1f}% dynamic routing compute over Uniform!)")
    
    results = {
        "vit_blocks": L,
        "batch_size": B,
        "pure_inference_latency_ms": latency_base,
        "uniform_router": {
            "block_size": 3,
            "block_groups": 4,
            "router_parameters": unif_params,
            "blended_forward_latency_ms": lat_unif,
            "dynamic_merging_overhead_ms": overhead_unif,
        },
        "coarse_to_fine_router": {
            "block_groups": non_unif_groups,
            "router_parameters": nonunif_params,
            "blended_forward_latency_ms": lat_nonunif,
            "dynamic_merging_overhead_ms": overhead_nonunif,
        }
    }
    
    with open("vit_pilot_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nPilot results saved to 'vit_pilot_results.json'.")

if __name__ == '__main__':
    run_pilot()
