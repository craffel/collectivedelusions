import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.func import functional_call

from train_experts import CustomCNN, ExpertModel
from eval_tta import set_seed, apply_corruption, build_test_stream, reconstruct_merged_params, evaluate_method, augment_batch

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 1. Load pre-trained expert model states
    base_encoder = CustomCNN().to(device)
    
    expert_states = []
    original_heads = {}
    
    expert_names = ["mnist", "fashion", "kmnist"]
    for idx, name in enumerate(expert_names):
        path = f"./experts/expert_{name}.pt"
        if not os.path.exists(path):
            print(f"Error: expert {name} not found. Run train_experts.py first.")
            return
        state = torch.load(path, map_location=device)
        
        encoder_state = {}
        head_state = {}
        for k, v in state.items():
            if k.startswith("base_encoder."):
                encoder_state[k.replace("base_encoder.", "")] = v
            elif k.startswith("head."):
                head_state[k.replace("head.", "")] = v
                
        expert_states.append(encoder_state)
        
        head_model = nn.Linear(128, 10).to(device)
        head_model.load_state_dict(head_state)
        original_heads[idx] = head_model
        
    # 2. Load pre-computed Fisher priors
    fisher_priors = {}
    for idx, name in enumerate(expert_names):
        path = f"./fisher/fisher_{name}.pt"
        fisher_priors[idx] = torch.load(path, map_location=device)
        
    # 3. Load processed test datasets
    mnist_test = torch.load("./data/processed/mnist_test.pt")
    fashion_test = torch.load("./data/processed/fashion_test.pt")
    kmnist_test = torch.load("./data/processed/kmnist_test.pt")
    datasets = [mnist_test, fashion_test, kmnist_test]
    
    # Build clean stream first, then apply noise corruption
    raw_stream = build_test_stream(datasets, stream_type="sequential", batch_size=32, num_batches_per_task=50)
    
    corrupted_stream = []
    for task_id, images, labels in raw_stream:
        corrupted_images = apply_corruption(images, "noise")
        corrupted_stream.append((task_id, corrupted_images, labels))
        
    alphas = [0.1, 0.2, 0.5, 1.0, 2.0]
    results = {"tg": [], "tf": []}
    
    print("=== Sweeping Alpha on Sequential Noise stream ===")
    
    for alpha in alphas:
        # Evaluate Teacher-Guided FW-CMS with different alpha
        # Note: We can implement evaluate_method with a custom alpha, but let's do it manually here.
        # We will duplicate the evaluation logic for TG and TF, passing the custom alpha
        
        # TG Evaluation
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
            
        # TG run
        correct_tg = 0
        total_tg = 0
        for step, (task_id, images, labels) in enumerate(corrupted_stream):
            images, labels = images.to(device), labels.to(device)
            param_groups = []
            for k in encoder_keys:
                scaled_lr = 0.2 / (sensitivities[k] + alpha)
                param_groups.append({"params": [raw_lambdas[k]], "lr": scaled_lr})
            param_groups.append({"params": list(heads[task_id].parameters()), "lr": 1e-4})
            optimizer = optim.Adam(param_groups)
            
            merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
            
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
            loss_ewc = 0.0
            for name, param in heads[task_id].named_parameters():
                f_prior = fisher_priors[task_id][f"head.{name}"].to(device)
                init_param = init_head_params[task_id][name].to(device)
                loss_ewc += 0.5 * torch.sum(f_prior * (param - init_param) ** 2)
            loss = loss_kl + 100.0 * loss_ewc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                merged_params_eval = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
                z_eval = functional_call(base_encoder, merged_params_eval, images)
                outputs_eval = heads[task_id](z_eval)
                _, predicted = outputs_eval.max(1)
                total_tg += labels.size(0)
                correct_tg += predicted.eq(labels).sum().item()
                
        acc_tg = 100.0 * correct_tg / total_tg
        results["tg"].append(acc_tg)
        
        # TF Run
        set_seed(42)
        raw_lambdas_tf = {k: torch.zeros(3, device=device, requires_grad=True) for k in encoder_keys}
        heads_tf = {t_idx: copy.deepcopy(original_heads[t_idx]).to(device) for t_idx in original_heads}
        for t_idx in heads_tf:
            heads_tf[t_idx].eval() # frozen
            
        correct_tf = 0
        total_tf = 0
        for step, (task_id, images, labels) in enumerate(corrupted_stream):
            images, labels = images.to(device), labels.to(device)
            param_groups = []
            for k in encoder_keys:
                scaled_lr = 0.1 / (sensitivities[k] + alpha)
                param_groups.append({"params": [raw_lambdas_tf[k]], "lr": scaled_lr})
            optimizer = optim.Adam(param_groups)
            
            merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas_tf, use_softmax=True)
            
            z = functional_call(base_encoder, merged_params, images)
            merged_outputs = heads_tf[task_id](z)
            p_merged = torch.softmax(merged_outputs, dim=-1)
            loss_ent = -torch.mean(torch.sum(p_merged * torch.log(p_merged + 1e-12), dim=-1))
            
            images_aug = augment_batch(images)
            z_aug = functional_call(base_encoder, merged_params, images_aug)
            merged_outputs_aug = heads_tf[task_id](z_aug)
            p_merged_aug = torch.softmax(merged_outputs_aug, dim=-1)
            loss_const = torch.mean(torch.sum(p_merged.detach() * (torch.log(p_merged.detach() + 1e-12) - torch.log(p_merged_aug + 1e-12)), dim=-1))
            
            loss = loss_ent + 1.0 * loss_const
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                merged_params_eval = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas_tf, use_softmax=True)
                z_eval = functional_call(base_encoder, merged_params_eval, images)
                outputs_eval = heads_tf[task_id](z_eval)
                _, predicted = outputs_eval.max(1)
                total_tf += labels.size(0)
                correct_tf += predicted.eq(labels).sum().item()
                
        acc_tf = 100.0 * correct_tf / total_tf
        results["tf"].append(acc_tf)
        
        print(f"Alpha: {alpha:<4} | TG Acc: {acc_tg:.2f}% | TF Acc: {acc_tf:.2f}%")
        
    # Save sweep results
    torch.save(results, "./sweep_results.pt")
    print("Alpha sweep completed.")

if __name__ == "__main__":
    main()
