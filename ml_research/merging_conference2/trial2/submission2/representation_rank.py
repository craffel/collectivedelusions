import torch
import torch.nn as nn
import numpy as np
import copy
from run_experiments import load_expert, get_test_loader, build_merged_real_model, device
from cpos_merging import CPOSResNet
from representation_analysis import extract_activations

def compute_effective_rank(activation_tensor):
    """
    Computes the Effective Rank (erank) of an activation tensor of shape (B, C, H, W).
    We flatten the spatial and channel dimensions to shape (B, C * H * W)
    and compute the singular values of the resulting matrix.
    """
    # Flatten features to shape (Batch, Features)
    B = activation_tensor.size(0)
    H_flat = activation_tensor.view(B, -1).detach().cpu().to(torch.float32)
    
    # Compute singular values
    # Since B <= Features typically (e.g. B=256, Features >= 512), SVD is very fast.
    try:
        S = torch.linalg.svdvals(H_flat)
        # Prevent division by zero and handle tiny singular values
        S_sum = S.sum()
        if S_sum < 1e-12:
            return 1.0
        p = S / S_sum
        # Compute Shannon entropy of the singular value distribution
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        # Effective rank is exp(entropy)
        erank = torch.exp(entropy).item()
        return erank
    except Exception as e:
        # In case SVD fails to converge, return NaN or a default value
        print(f"SVD failed: {e}")
        return float('nan')

def main():
    print("==========================================================================")
    print("REPRESENTATION COMPLEXITY ANALYSIS: EFFECTIVE RANK OF ACTIVATIONS")
    print("==========================================================================")
    
    # 1. Load Experts
    task_A = "cifar10"
    task_B = "fmnist"
    model_A = load_expert(task_A).to(device)
    model_B = load_expert(task_B).to(device)
    
    # 2. Setup Merged Models
    # Weight Averaging State Dict
    state_A = model_A.state_dict()
    state_B = model_B.state_dict()
    merged_wa_state = {}
    for k in state_A.keys():
        if k.startswith("fc."):
            continue
        if k in state_B:
            merged_wa_state[k] = 0.5 * state_A[k] + 0.5 * state_B[k]
            
    # Build WA model (for Task A and Task B evaluation of backbone activations)
    fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
    fc_B_state = {k.replace("fc.", ""): v for k, v in model_B.state_dict().items() if k.startswith("fc.")}
    wa_model_A = build_merged_real_model(merged_wa_state, fc_A_state, device)
    
    # CPOS model
    alpha_val = 1.0 / np.sqrt(2)
    beta_val = 1.0 / np.sqrt(2)
    cpos_model = CPOSResNet(model_A, model_B, alpha=alpha_val, beta=beta_val).to(device)
    
    # 3. Load Dataloader (using batch size 256 for stable SVD estimation)
    loader_A = get_test_loader(task_A, max_samples=256)
    inputs_A, _ = next(iter(loader_A))
    inputs_A = inputs_A.to(device)
    
    model_A.eval()
    wa_model_A.eval()
    cpos_model.eval()
    
    with torch.no_grad():
        # Extracted activations
        act_expert = extract_activations(model_A, inputs_A, is_cpos=False)
        act_wa = extract_activations(wa_model_A, inputs_A, is_cpos=False)
        act_cpos = extract_activations(cpos_model, inputs_A, is_cpos=True)
        
    print("\n--- Effective Rank of activations (CIFAR-10 Input, Batch Size 256) ---")
    print("| Block | Expert A erank | WA erank | CPOS erank | CPOS / WA Ratio |")
    print("|---|---|---|---|---|")
    block_names = ["Stem", "Layer 1.1", "Layer 1.2", "Layer 2.1", "Layer 2.2", "Layer 3.1", "Layer 3.2", "Layer 4.1", "Layer 4.2"]
    
    for i, name in enumerate(block_names):
        rank_exp = compute_effective_rank(act_expert[i])
        rank_wa = compute_effective_rank(act_wa[i])
        rank_cpos = compute_effective_rank(act_cpos[i])
        ratio = rank_cpos / rank_wa if rank_wa > 0 else 1.0
        print(f"| {name} | {rank_exp:.2f} | {rank_wa:.2f} | {rank_cpos:.2f} | {ratio:.2f}x |")

if __name__ == "__main__":
    main()
