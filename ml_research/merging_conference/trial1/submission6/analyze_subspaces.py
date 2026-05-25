import os
import torch
from safetensors.torch import load_file
import numpy as np

configs = [
    ("Standard LoRA", "cifar10_standard", "svhn_standard"),
    ("SAM Only", "cifar10_sam", "svhn_sam"),
    ("ISR Only (Rigid)", "cifar10_isr", "svhn_isr"),
    ("ISR Only (Soft)", "cifar10_isr_soft", "svhn_isr_soft"),
    ("SATA-LR (Rigid)", "cifar10_sata_lr", "svhn_sata_lr"),
    ("SATA-LR-Soft", "cifar10_sata_lr_soft", "svhn_sata_lr_soft")
]

checkpoint_dir = "checkpoints"

def load_weights(path):
    sf_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    if os.path.exists(sf_path):
        return load_file(sf_path)
    elif os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights found in {path}")

def analyze_layer_subspaces(B1, A1, B2, A2):
    # W = B @ A
    W1 = B1 @ A1
    W2 = B2 @ A2
    
    # 1. Cosine similarity of the full updates
    cos_sim = torch.sum(W1 * W2) / (torch.norm(W1, p='fro') * torch.norm(W2, p='fro') + 1e-12)
    
    # 2. Subspace overlap of column spaces (B)
    # Get orthonormal bases via QR decomposition
    Q1, _ = torch.linalg.qr(B1)
    Q2, _ = torch.linalg.qr(B2)
    # normalized overlap is trace(Q1^T Q2 Q2^T Q1) / r = ||Q1^T Q2||_F^2 / r
    r = B1.shape[1]
    overlap_B = torch.sum((Q1.t() @ Q2) ** 2).item() / r
    
    # 3. Subspace overlap of row spaces (A^T)
    # A is shape [r, d_in], so A^T is [d_in, r]
    P1, _ = torch.linalg.qr(A1.t())
    P2, _ = torch.linalg.qr(A2.t())
    overlap_A = torch.sum((P1.t() @ P2) ** 2).item() / r
    
    # 4. Spectral entropy of W1 and W2
    def spectral_entropy(W):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S = S + 1e-12
        p = S / torch.sum(S)
        entropy = -torch.sum(p * torch.log(p)) / np.log(len(S))
        return entropy.item()
        
    ent1 = spectral_entropy(W1)
    ent2 = spectral_entropy(W2)
    mean_entropy = (ent1 + ent2) / 2.0
    
    return cos_sim.item(), overlap_B, overlap_A, mean_entropy

def main():
    print("========================================================================")
    print("WEIGHT-SPACE GEOMETRY AND SUBSPACE OVERLAP ANALYSIS")
    print("========================================================================")
    
    for name, cifar_name, svhn_name in configs:
        cifar_path = os.path.join(checkpoint_dir, cifar_name)
        svhn_path = os.path.join(checkpoint_dir, svhn_name)
        
        try:
            w_cifar = load_weights(cifar_path)
            w_svhn = load_weights(svhn_path)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
            
        # Find all keys for LoRA B
        b_keys = [k for k in w_cifar.keys() if "lora_B" in k and k.endswith("weight")]
        
        cos_sims = []
        overlaps_B = []
        overlaps_A = []
        entropies = []
        
        for key in b_keys:
            key_A = key.replace("lora_B", "lora_A")
            if key_A in w_cifar and key in w_svhn and key_A in w_svhn:
                B1 = w_cifar[key].float()
                A1 = w_cifar[key_A].float()
                B2 = w_svhn[key].float()
                A2 = w_svhn[key_A].float()
                
                c_sim, o_B, o_A, ent = analyze_layer_subspaces(B1, A1, B2, A2)
                cos_sims.append(c_sim)
                overlaps_B.append(o_B)
                overlaps_A.append(o_A)
                entropies.append(ent)
                
        if len(cos_sims) > 0:
            print(f"Configuration: {name}")
            print(f"  Mean Task-Vector Cosine Similarity: {np.mean(cos_sims):.4f}")
            print(f"  Mean Output Subspace Overlap (B):   {np.mean(overlaps_B):.4f}")
            print(f"  Mean Input Subspace Overlap (A^T):  {np.mean(overlaps_A):.4f}")
            print(f"  Mean Spectral Entropy (Isotropy):   {np.mean(entropies):.4f}")
            print("-" * 72)
            
if __name__ == "__main__":
    main()
