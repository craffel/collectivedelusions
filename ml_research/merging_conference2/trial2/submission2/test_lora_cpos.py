import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.models import resnet18, ResNet18_Weights
from run_experiments import load_expert, get_test_loader, evaluate_model, device
from cpos_merging import CPOSResNet

def reconstruct_low_rank_weight(weight_expert, weight_base, rank):
    shape = weight_expert.shape
    if len(shape) == 4:  # Conv layer (C_out, C_in, K, K)
        flat_expert = weight_expert.view(shape[0], -1)
        flat_base = weight_base.view(shape[0], -1)
        delta_W = flat_expert - flat_base
        
        U, S, Vh = torch.linalg.svd(delta_W.cpu(), full_matrices=False)
        r = min(rank, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        
        delta_W_rec = U_r @ torch.diag(S_r) @ Vh_r
        delta_W_rec = delta_W_rec.to(weight_expert.device)
        return (flat_base + delta_W_rec).view(shape)
    elif len(shape) == 2:  # Linear layer (C_out, C_in)
        delta_W = weight_expert - weight_base
        U, S, Vh = torch.linalg.svd(delta_W.cpu(), full_matrices=False)
        r = min(rank, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        
        delta_W_rec = U_r @ torch.diag(S_r) @ Vh_r
        delta_W_rec = delta_W_rec.to(weight_expert.device)
        return weight_base + delta_W_rec
    else:
        return weight_expert

def calculate_lora_params(model_base, rank):
    """
    Calculate the number of task-specific parameters under LoRA for all conv and linear layers.
    Also returns the original total parameter count.
    """
    total_base_params = sum(p.numel() for p in model_base.parameters())
    lora_params = 0
    
    # We also keep the FC layer fully task-specific, which is 512 * 10 + 10 = 5130 parameters
    lora_params += sum(p.numel() for name, p in model_base.named_parameters() if "fc" in name)
    
    # For BatchNorm layers, we keep them task-specific too (weight and bias: 2 * num_features)
    lora_params += sum(p.numel() for name, p in model_base.named_parameters() if "bn" in name)
    
    # For Conv and other Linear layers, we apply LoRA
    for name, module in model_base.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and "fc" not in name:
            weight = module.weight
            shape = weight.shape
            if len(shape) == 4:  # Conv
                d_out = shape[0]
                d_in = shape[1] * shape[2] * shape[3]
            elif len(shape) == 2:  # Linear
                d_out = shape[0]
                d_in = shape[1]
            else:
                continue
            
            # LoRA parameters for rank r: r * (d_out + d_in)
            r = min(rank, d_out, d_in)
            lora_params += r * (d_out + d_in)
            
    return lora_params, total_base_params

def create_reconstructed_expert(model_expert, model_base, rank):
    """
    Creates a new model that replaces all conv and linear weights with SVD low-rank approximations.
    """
    reconstructed_model = resnet18(weights=None)
    reconstructed_model.fc = nn.Linear(512, 10)
    reconstructed_model = reconstructed_model.to(device)
    
    # Load state dict of expert as base starting point
    rec_state_dict = model_expert.state_dict().copy()
    base_state_dict = model_base.state_dict()
    
    for key in rec_state_dict.keys():
        # Only reconstruct Conv and Linear weights (excluding FC and BN layers)
        if ("weight" in key) and ("fc" not in key) and ("bn" not in key) and (key in base_state_dict):
            rec_state_dict[key] = reconstruct_low_rank_weight(
                rec_state_dict[key], 
                base_state_dict[key], 
                rank
            )
            
    reconstructed_model.load_state_dict(rec_state_dict)
    return reconstructed_model

def main():
    print("==========================================================================")
    print("LORA-CPOS SIMULATION: SVD LOW-RANK ADAPTER DECOMPOSITION")
    print("==========================================================================")
    print(f"Using device: {device}")
    
    # 1. Load pre-trained Base Model (weights = DEFAULT)
    # The classification head of ResNet-18 has 1000 classes, so we change it to 10
    # to match the architecture of our experts.
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    base_model = base_model.to(device)
    
    # 2. Load fine-tuned Experts
    task_A = "cifar10"
    task_B = "fmnist"
    expert_A = load_expert(task_A).to(device)
    expert_B = load_expert(task_B).to(device)
    
    # 3. Load Test Data Loaders (1000 samples for speed & statistical significance)
    loader_A = get_test_loader(task_A, max_samples=1000)
    loader_B = get_test_loader(task_B, max_samples=1000)
    
    ranks = [2, 4, 8, 16, 32]
    
    print("\nEvaluating Low-Rank CPOS (LoRA-CPOS) and Low-Rank WA (LoRA-WA):")
    print("| Rank | Task-Specific Params | % of Base Model | LoRA-WA Acc (%) | LoRA-CPOS Acc (%) |")
    print("|---|---|---|---|---|")
    
    # Evaluate full-rank (no SVD truncation)
    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)
    
    for rank in ranks:
        # Reconstruct experts
        rec_expert_A = create_reconstructed_expert(expert_A, base_model, rank)
        rec_expert_B = create_reconstructed_expert(expert_B, base_model, rank)
        
        # Calculate parameters
        lora_p, base_p = calculate_lora_params(base_model, rank)
        p_pct = 100.0 * lora_p / base_p
        
        # Evaluate LoRA-WA (Weight Averaging on reconstructed experts)
        wa_state_dict = rec_expert_A.state_dict().copy()
        state_B = rec_expert_B.state_dict()
        for key in wa_state_dict.keys():
            if "fc" not in key:  # Backbone weights
                wa_state_dict[key] = 0.5 * wa_state_dict[key] + 0.5 * state_B[key]
                
        # We evaluate WA for both tasks by loading the merged backbone and the task's FC head
        # Task A
        wa_model_A = resnet18(weights=None)
        wa_model_A.fc = nn.Linear(512, 10)
        wa_model_A.load_state_dict(wa_state_dict)
        wa_model_A.fc.load_state_dict(rec_expert_A.fc.state_dict())
        wa_model_A = wa_model_A.to(device)
        acc_wa_A = evaluate_model(wa_model_A, loader_A)
        
        # Task B
        wa_model_B = resnet18(weights=None)
        wa_model_B.fc = nn.Linear(512, 10)
        wa_model_B.load_state_dict(wa_state_dict)
        wa_model_B.fc.load_state_dict(rec_expert_B.fc.state_dict())
        wa_model_B = wa_model_B.to(device)
        acc_wa_B = evaluate_model(wa_model_B, loader_B)
        
        avg_wa = (acc_wa_A + acc_wa_B) / 2.0
        
        # Evaluate LoRA-CPOS
        cpos_model = CPOSResNet(rec_expert_A, rec_expert_B, alpha=alpha, beta=beta).to(device)
        
        cpos_model.set_task(0)
        acc_cpos_A = evaluate_model(cpos_model, loader_A)
        
        cpos_model.set_task(1)
        acc_cpos_B = evaluate_model(cpos_model, loader_B)
        
        avg_cpos = (acc_cpos_A + acc_cpos_B) / 2.0
        
        print(f"| r={rank:2d} | {lora_p:,} | {p_pct:.2f}% | {avg_wa:.2f}% | {avg_cpos:.2f}% |")

    # Full Rank comparison
    full_cpos_model = CPOSResNet(expert_A, expert_B, alpha=alpha, beta=beta).to(device)
    full_cpos_model.set_task(0)
    acc_full_A = evaluate_model(full_cpos_model, loader_A)
    full_cpos_model.set_task(1)
    acc_full_B = evaluate_model(full_cpos_model, loader_B)
    avg_full_cpos = (acc_full_A + acc_full_B) / 2.0
    
    # Full Rank WA
    wa_full_state = expert_A.state_dict().copy()
    state_B_full = expert_B.state_dict()
    for key in wa_full_state.keys():
        if "fc" not in key:
            wa_full_state[key] = 0.5 * wa_full_state[key] + 0.5 * state_B_full[key]
            
    wa_full_model_A = resnet18(weights=None)
    wa_full_model_A.fc = nn.Linear(512, 10)
    wa_full_model_A.load_state_dict(wa_full_state)
    wa_full_model_A.fc.load_state_dict(expert_A.fc.state_dict())
    wa_full_model_A = wa_full_model_A.to(device)
    acc_full_wa_A = evaluate_model(wa_full_model_A, loader_A)
    
    wa_full_model_B = resnet18(weights=None)
    wa_full_model_B.fc = nn.Linear(512, 10)
    wa_full_model_B.load_state_dict(wa_full_state)
    wa_full_model_B.fc.load_state_dict(expert_B.fc.state_dict())
    wa_full_model_B = wa_full_model_B.to(device)
    acc_full_wa_B = evaluate_model(wa_full_model_B, loader_B)
    avg_full_wa = (acc_full_wa_A + acc_full_wa_B) / 2.0
    
    print(f"| Full | 11,181,642 | 100.00% | {avg_full_wa:.2f}% | {avg_full_cpos:.2f}% |")

if __name__ == "__main__":
    main()
