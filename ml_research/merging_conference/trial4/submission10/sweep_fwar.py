import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
from torch.func import functional_call

from train_experts import CustomCNN, ExpertModel
from eval_tta import set_seed, apply_corruption, augment_batch, build_test_stream, reconstruct_merged_params

def evaluate_fwar(method_name, stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                  lr_lambda=0.1, lr_head=1e-4, gamma_ewc=100.0, gamma_const=1.0, gamma_fwar=10.0):
    set_seed(42)
    
    encoder_keys = [name for name, param in base_encoder.named_parameters() if param.requires_grad]
    
    # Initialize merging coefficients
    raw_lambdas = {k: torch.zeros(3, device=device, requires_grad=True) for k in encoder_keys}
    
    # Copy expert heads
    heads = {t_idx: copy.deepcopy(original_heads[t_idx]).to(device) for t_idx in original_heads}
    for t_idx in heads:
        heads[t_idx].train()
        
    init_head_params = {t_idx: {name: param.clone().detach() for name, param in heads[t_idx].named_parameters()} for t_idx in heads}
    
    # Get sensitivities
    sensitivities = {}
    for k in encoder_keys:
        f0 = torch.mean(fisher_priors[0][f"base_encoder.{k}"])
        f1 = torch.mean(fisher_priors[1][f"base_encoder.{k}"])
        f2 = torch.mean(fisher_priors[2][f"base_encoder.{k}"])
        sensitivities[k] = (f0 + f1 + f2).item() / 3.0
        
    mean_sens = np.mean(list(sensitivities.values()))
    for k in sensitivities:
        sensitivities[k] /= (mean_sens + 1e-12)
        
    correct = 0
    total = 0
    
    for step, (task_id, images, labels) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        
        # --- Adaptation Step ---
        param_groups = []
        alpha = 0.5 # smoothing factor for LR scaling
        for k in encoder_keys:
            scaled_lr = lr_lambda / (sensitivities[k] + alpha)
            param_groups.append({"params": [raw_lambdas[k]], "lr": scaled_lr})
            
        adapting_heads = "tg" in method_name
        if adapting_heads:
            param_groups.append({"params": list(heads[task_id].parameters()), "lr": lr_head})
            
        optimizer = optim.Adam(param_groups)
        
        merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
        
        if "tg" in method_name:
            # Teacher-Guided (Self-Labeling)
            with torch.no_grad():
                teacher = ExpertModel(base_encoder).to(device)
                expert_state = {}
                for name, param in base_encoder.named_parameters():
                    expert_state[f"base_encoder.{name}"] = expert_states[task_id][name]
                for name, param in original_heads[task_id].named_parameters():
                    expert_state[f"head.{name}"] = param
                teacher.load_state_dict(expert_state)
                teacher.eval()
                teacher_outputs = teacher(images)
                p_expert = torch.softmax(teacher_outputs, dim=-1)
                
            z = functional_call(base_encoder, merged_params, images)
            merged_outputs = heads[task_id](z)
            p_merged = torch.softmax(merged_outputs, dim=-1)
            
            loss_kl = torch.mean(torch.sum(p_expert * (torch.log(p_expert + 1e-12) - torch.log(p_merged + 1e-12)), dim=-1))
            loss = loss_kl
            
            # EWC
            loss_ewc = 0.0
            for name, param in heads[task_id].named_parameters():
                fisher_key = f"head.{name}"
                f_prior = fisher_priors[task_id][fisher_key].to(device)
                init_param = init_head_params[task_id][name].to(device)
                loss_ewc += 0.5 * torch.sum(f_prior * (param - init_param) ** 2)
            loss += gamma_ewc * loss_ewc
            
        else:
            # Teacher-Free (Self-Supervised)
            z = functional_call(base_encoder, merged_params, images)
            merged_outputs = heads[task_id](z)
            p_merged = torch.softmax(merged_outputs, dim=-1)
            loss_ent = -torch.mean(torch.sum(p_merged * torch.log(p_merged + 1e-12), dim=-1))
            
            images_aug = augment_batch(images)
            z_aug = functional_call(base_encoder, merged_params, images_aug)
            merged_outputs_aug = heads[task_id](z_aug)
            p_merged_aug = torch.softmax(merged_outputs_aug, dim=-1)
            
            loss_const = torch.mean(torch.sum(p_merged.detach() * (torch.log(p_merged.detach() + 1e-12) - torch.log(p_merged_aug + 1e-12)), dim=-1))
            
            loss = loss_ent + gamma_const * loss_const
            
        # Add proposed Fisher-Weighted Anchor Regularization (FWAR)
        loss_fwar = 0.0
        w_0 = torch.tensor([1/3, 1/3, 1/3], device=device)
        for k in encoder_keys:
            w = torch.softmax(raw_lambdas[k], dim=0)
            # Regularize weights to stay near uniform w_0, scaled by sensitivity
            loss_fwar += sensitivities[k] * torch.sum((w - w_0) ** 2)
            
        loss += gamma_fwar * loss_fwar
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Inference Step ---
        with torch.no_grad():
            merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
            z = functional_call(base_encoder, merged_params, images)
            outputs = heads[task_id](z)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load models
    base_encoder = CustomCNN().to(device)
    expert_states = []
    original_heads = {}
    expert_names = ["mnist", "fashion", "kmnist"]
    for idx, name in enumerate(expert_names):
        path = f"./experts/expert_{name}.pt"
        state = torch.load(path, map_location=device)
        encoder_state = {k.replace("base_encoder.", ""): v for k, v in state.items() if k.startswith("base_encoder.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        expert_states.append(encoder_state)
        head_model = nn.Linear(128, 10).to(device)
        head_model.load_state_dict(head_state)
        original_heads[idx] = head_model
        
    fisher_priors = {idx: torch.load(f"./fisher/fisher_{name}.pt", map_location=device) for idx, name in enumerate(expert_names)}
    
    mnist_test = torch.load("./data/processed/mnist_test.pt")
    fashion_test = torch.load("./data/processed/fashion_test.pt")
    kmnist_test = torch.load("./data/processed/kmnist_test.pt")
    datasets = [mnist_test, fashion_test, kmnist_test]
    
    # We will test on Clean and Noise corruptions to quickly find the best gamma_fwar
    corruptions = ["clean", "noise"]
    stream_types = ["sequential", "alternating"]
    gamma_values = [0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0]
    
    for stream_type in stream_types:
        print(f"\n==========================================")
        print(f"SWEEPING GAMMA_FWAR ON {stream_type.upper()} STREAM")
        print(f"==========================================")
        
        for corr in corruptions:
            print(f"\nCorruption: {corr.upper()}")
            print(f"{'gamma_fwar':<12} | {'FW-CMS (TF)':<15} | {'FW-CMS (TG)':<15}")
            print("-" * 50)
            
            raw_stream = build_test_stream(datasets, stream_type=stream_type, batch_size=32, num_batches_per_task=50)
            corrupted_stream = []
            for task_id, images, labels in raw_stream:
                corrupted_stream.append((task_id, apply_corruption(images, corr), labels))
                
            for gamma in gamma_values:
                # TF hyperparameters
                acc_tf = evaluate_fwar("fw_cms_tf", corrupted_stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                                       lr_lambda=0.1, gamma_const=1.0, gamma_fwar=gamma)
                # TG hyperparameters
                acc_tg = evaluate_fwar("fw_cms_tg", corrupted_stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                                       lr_lambda=0.2, lr_head=1e-4, gamma_ewc=100.0, gamma_fwar=gamma)
                print(f"{gamma:<12} | {acc_tf:.2f}%          | {acc_tg:.2f}%")

if __name__ == "__main__":
    main()
