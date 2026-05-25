import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import functional_call

import train_and_merge
from train_and_merge import (
    device, get_model_and_tokenizer, get_datasets, create_zero_shot_heads,
    get_visual_params_dict, SAM_Optimizer
)

def run_calibration_and_measure(use_sam=False, parameter_wise=False, selective=False, rho=0.01, lr=2e-3, num_steps=40):
    # Setup
    datasets = get_datasets()
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    pretrained_visual_state = copy.deepcopy(model.visual.state_dict())
    text_heads = create_zero_shot_heads(model, tokenizer)
    
    tasks = ["mnist", "fashion", "cifar10"]
    expert_states = {}
    for t in tasks:
        weight_path = f"expert_{t}.pt"
        if os.path.exists(weight_path):
            expert_states[t] = torch.load(weight_path, map_location=device)
        else:
            raise FileNotFoundError(f"Expert weights {weight_path} not found. Run train_and_merge.py first.")
            
    task_vectors = {}
    for t in tasks:
        tv = {}
        for name in pretrained_visual_state.keys():
            if name in expert_states[t]:
                tv[name] = expert_states[t][name] - pretrained_visual_state[name]
        task_vectors[t] = tv
        
    cal_loaders = {t: DataLoader(datasets[t]["cal"], batch_size=32, shuffle=True) for t in tasks}
    
    # Initialize calibration parameters
    coeffs = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
    task_proj_states = {t: copy.deepcopy(expert_states[t]['proj']).to(device) for t in tasks}
    task_projs = {t: nn.Parameter(copy.deepcopy(task_proj_states[t])) for t in tasks}
    
    params_to_opt = [coeffs] + list(task_projs.values())
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr)
    
    if use_sam:
        sam_opt = SAM_Optimizer(params_to_opt, optimizer, rho=rho, parameter_wise=parameter_wise, selective=selective)
        
    # Calibration loop
    for step in range(num_steps):
        for task_idx, t in enumerate(tasks):
            images, _ = next(iter(cal_loaders[t]))
            images = images.to(device)
            
            # Expert soft labels
            expert_params = get_visual_params_dict(
                model.visual, pretrained_visual_state, {t: task_vectors[t]}, torch.tensor([1.0], device=device)
            )
            with torch.no_grad():
                expert_feats = functional_call(model.visual, expert_params, (images,))
                expert_feats = expert_feats / expert_feats.norm(dim=-1, keepdim=True)
                expert_logits = 100.0 * expert_feats @ text_heads[t].T
                expert_probs = F.softmax(expert_logits, dim=-1)
                
            def compute_loss(c_values, p_dict):
                merged_params = get_visual_params_dict(
                    model.visual, pretrained_visual_state, task_vectors, c_values, p_dict, active_task=t
                )
                merged_feats = functional_call(model.visual, merged_params, (images,))
                merged_feats = merged_feats / merged_feats.norm(dim=-1, keepdim=True)
                merged_logits = 100.0 * merged_feats @ text_heads[t].T
                merged_log_probs = F.log_softmax(merged_logits, dim=-1)
                return F.kl_div(merged_log_probs, expert_probs, reduction='batchmean')
                
            if use_sam:
                optimizer.zero_grad()
                loss = compute_loss(coeffs, task_projs)
                loss.backward()
                sam_opt.first_step()
                
                optimizer.zero_grad()
                loss_perturbed = compute_loss(coeffs, task_projs)
                loss_perturbed.backward()
                sam_opt.second_step()
            else:
                optimizer.zero_grad()
                loss = compute_loss(coeffs, task_projs)
                loss.backward()
                optimizer.step()
                
    # --- Measure Empirical Sharpness ---
    # We measure sharpness on a full validation/calibration batch
    sharpness_results = {}
    
    for t in tasks:
        # Load a larger batch for stable evaluation
        loader = DataLoader(datasets[t]["cal"], batch_size=200, shuffle=False)
        images, _ = next(iter(loader))
        images = images.to(device)
        
        # Expert soft labels
        expert_params = get_visual_params_dict(
            model.visual, pretrained_visual_state, {t: task_vectors[t]}, torch.tensor([1.0], device=device)
        )
        with torch.no_grad():
            expert_feats = functional_call(model.visual, expert_params, (images,))
            expert_feats = expert_feats / expert_feats.norm(dim=-1, keepdim=True)
            expert_logits = 100.0 * expert_feats @ text_heads[t].T
            expert_probs = F.softmax(expert_logits, dim=-1)
            
        def eval_loss(c_values, p_dict):
            merged_params = get_visual_params_dict(
                model.visual, pretrained_visual_state, task_vectors, c_values, p_dict, active_task=t
            )
            merged_feats = functional_call(model.visual, merged_params, (images,))
            merged_feats = merged_feats / merged_feats.norm(dim=-1, keepdim=True)
            merged_logits = 100.0 * merged_feats @ text_heads[t].T
            merged_log_probs = F.log_softmax(merged_logits, dim=-1)
            return F.kl_div(merged_log_probs, expert_probs, reduction='batchmean').item()
            
        # Point-wise loss
        loss_orig = eval_loss(coeffs, task_projs)
        
        # Calculate adversarial perturbation to find max loss in rho neighborhood
        # We manually compute the gradient-based perturbation to find worst-case loss
        temp_coeffs = nn.Parameter(coeffs.clone())
        temp_projs = {task_name: nn.Parameter(task_projs[task_name].clone()) for task_name in tasks}
        
        temp_params_to_opt = [temp_coeffs] + list(temp_projs.values())
        
        # Forward pass to compute gradients for perturbation
        merged_params = get_visual_params_dict(
            model.visual, pretrained_visual_state, task_vectors, temp_coeffs, temp_projs, active_task=t
        )
        merged_feats = functional_call(model.visual, merged_params, (images,))
        merged_feats = merged_feats / merged_feats.norm(dim=-1, keepdim=True)
        merged_logits = 100.0 * merged_feats @ text_heads[t].T
        merged_log_probs = F.log_softmax(merged_logits, dim=-1)
        loss_for_grad = F.kl_div(merged_log_probs, expert_probs, reduction='batchmean')
        
        loss_for_grad.backward()
        
        # Apply perturbation
        with torch.no_grad():
            if parameter_wise:
                for p in temp_params_to_opt:
                    if p.grad is not None:
                        if selective and p.numel() <= 10:
                            continue
                        grad_norm = p.grad.norm(2)
                        scale = rho / (grad_norm + 1e-12)
                        p.add_(p.grad * scale)
            else:
                grad_norms = [p.grad.norm(2) for p in temp_params_to_opt if p.grad is not None and not (selective and p.numel() <= 10)]
                if len(grad_norms) > 0:
                    grad_norm = torch.norm(torch.stack(grad_norms), 2)
                    scale = rho / (grad_norm + 1e-12)
                    for p in temp_params_to_opt:
                        if p.grad is not None:
                            if selective and p.numel() <= 10:
                                continue
                            p.add_(p.grad * scale)
                            
        loss_pert = eval_loss(temp_coeffs, temp_projs)
        empirical_sharpness = loss_pert - loss_orig
        
        sharpness_results[t] = {
            "loss_orig": loss_orig,
            "loss_pert": loss_pert,
            "sharpness": empirical_sharpness
        }
        
    return sharpness_results

def main():
    print(">>> Measuring empirical sharpness for standard SyMerge (Gradient Descent)...")
    symerge_sharpness = run_calibration_and_measure(use_sam=False, num_steps=40, rho=0.01)
    
    print("\n>>> Measuring empirical sharpness for our proposed PW-SASLA (Selective Parameter-Wise SAM)...")
    pwsasla_sharpness = run_calibration_and_measure(use_sam=True, parameter_wise=True, selective=True, num_steps=40, rho=0.01)
    
    results = {
        "SyMerge": symerge_sharpness,
        "PW-SASLA": pwsasla_sharpness
    }
    
    print("\n================ SHARPNESS RESULTS ================")
    for method, res in results.items():
        print(f"\nMethod: {method}")
        avg_sharp = 0.0
        for t, metrics in res.items():
            print(f"  Task {t:10s} | Base Loss: {metrics['loss_orig']:.6f} | Perturbed Loss: {metrics['loss_pert']:.6f} | Empirical Sharpness: {metrics['sharpness']:.6f}")
            avg_sharp += metrics['sharpness']
        print(f"  Average Empirical Sharpness: {avg_sharp / len(res):.6f}")
        
    with open("sharpness_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to sharpness_evaluation.json successfully!")

if __name__ == "__main__":
    main()
