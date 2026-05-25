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

def evaluate_gamma_ewc(method_name, stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                       lr_lambda=0.2, lr_head=1e-4, gamma_ewc=100.0, gamma_const=1.0, gamma_fwar=25.0):
    set_seed(42)
    
    encoder_keys = [name for name, param in base_encoder.named_parameters() if param.requires_grad]
    raw_lambdas = {k: torch.zeros(3, device=device, requires_grad=True) for k in encoder_keys}
    
    heads = {t_idx: copy.deepcopy(original_heads[t_idx]).to(device) for t_idx in original_heads}
    for t_idx in heads:
        heads[t_idx].train()
        
    init_head_params = {t_idx: {name: param.clone().detach() for name, param in heads[t_idx].named_parameters()} for t_idx in heads}
    
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
        alpha = 0.5
        for k in encoder_keys:
            scaled_lr = lr_lambda / (sensitivities[k] + alpha)
            param_groups.append({"params": [raw_lambdas[k]], "lr": scaled_lr})
            
        # adapting heads
        param_groups.append({"params": list(heads[task_id].parameters()), "lr": lr_head})
            
        optimizer = optim.Adam(param_groups)
        
        merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
        
        # Teacher-Guided Adaptation
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
        
        if "fwar" in method_name:
            loss_fwar = 0.0
            w_0 = torch.tensor([1/3, 1/3, 1/3], device=device)
            for k in encoder_keys:
                w = torch.softmax(raw_lambdas[k], dim=0)
                loss_fwar += sensitivities[k] * torch.sum((w - w_0) ** 2)
            loss += gamma_fwar * loss_fwar
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Inference / Evaluation Step ---
        with torch.no_grad():
            merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
            z = functional_call(base_encoder, merged_params, images)
            outputs = heads[task_id](z)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
    fisher_priors = {}
    for idx, name in enumerate(expert_names):
        path = f"./fisher/fisher_{name}.pt"
        fisher_priors[idx] = torch.load(path, map_location=device)
        
    mnist_test = torch.load("./data/processed/mnist_test.pt")
    fashion_test = torch.load("./data/processed/fashion_test.pt")
    kmnist_test = torch.load("./data/processed/kmnist_test.pt")
    datasets = [mnist_test, fashion_test, kmnist_test]
    
    # Evaluate Clean Sequential
    print("Sequential Stream - Clean Data")
    stream_clean = build_test_stream(datasets, stream_type="sequential", batch_size=32, num_batches_per_task=50, seed=42)
    
    for ge in [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        acc = evaluate_gamma_ewc("fw_cms_tg_fwar", stream_clean, base_encoder, expert_states, original_heads, fisher_priors, device, gamma_ewc=ge)
        print(f"  gamma_ewc = {ge} -> Accuracy: {acc:.2f}%")

    # Evaluate Noise Sequential
    print("\nSequential Stream - Gaussian Noise")
    stream_noise = [(tid, apply_corruption(imgs, "noise"), lbls) for tid, imgs, lbls in stream_clean]
    
    for ge in [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        acc = evaluate_gamma_ewc("fw_cms_tg_fwar", stream_noise, base_encoder, expert_states, original_heads, fisher_priors, device, gamma_ewc=ge)
        print(f"  gamma_ewc = {ge} -> Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
