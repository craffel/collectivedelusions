import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import functional_call
import argparse

import train_and_merge
from train_and_merge import (
    device, get_model_and_tokenizer, get_datasets, create_zero_shot_heads,
    get_visual_params_dict, get_merged_visual_model, evaluate_model, SAM_Optimizer
)

def run_calibration_ablation(model, pretrained_visual_state, task_vectors, text_heads, cal_loaders, expert_states,
                             use_sam=False, parameter_wise=False, selective=False, opt_mode="both", num_steps=40, rho=0.05, lr=2e-3):
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Initialize coefficients: 0.3 for each task
    coeffs = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
    
    # Initialize task-specific projection layers with experts' proj layers
    task_proj_states = {t: copy.deepcopy(expert_states[t]['proj']).to(device) for t in tasks}
    task_projs = {t: nn.Parameter(copy.deepcopy(task_proj_states[t])) for t in tasks}
    
    # Choose parameters to optimize
    if opt_mode == "both":
        params_to_opt = [coeffs] + list(task_projs.values())
    elif opt_mode == "coeffs":
        params_to_opt = [coeffs]
    elif opt_mode == "proj":
        params_to_opt = list(task_projs.values())
    else:
        raise ValueError(f"Unknown opt_mode: {opt_mode}")
        
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr)
    if use_sam:
        sam_opt = SAM_Optimizer(params_to_opt, optimizer, rho=rho, parameter_wise=parameter_wise, selective=selective)
        
    for step in range(num_steps):
        for task_idx, t in enumerate(tasks):
            images, _ = next(iter(cal_loaders[t]))
            images = images.to(device)
            
            # Get expert soft labels using functional_call
            expert_params = get_visual_params_dict(
                model.visual, pretrained_visual_state, {t: task_vectors[t]}, torch.tensor([1.0], device=device)
            )
            with torch.no_grad():
                expert_feats = functional_call(model.visual, expert_params, (images,))
                expert_feats = expert_feats / expert_feats.norm(dim=-1, keepdim=True)
                expert_logits = 100.0 * expert_feats @ text_heads[t].T
                expert_probs = F.softmax(expert_logits, dim=-1)
                
            def compute_loss(c_values, p_dict):
                # If opt_mode is 'coeffs', we do NOT use task-specific projs
                if opt_mode == "coeffs":
                    proj_dict = None
                else:
                    proj_dict = p_dict
                    
                merged_params = get_visual_params_dict(
                    model.visual, pretrained_visual_state, task_vectors, c_values, proj_dict, active_task=t
                )
                merged_feats = functional_call(model.visual, merged_params, (images,))
                merged_feats = merged_feats / merged_feats.norm(dim=-1, keepdim=True)
                merged_logits = 100.0 * merged_feats @ text_heads[t].T
                merged_log_probs = F.log_softmax(merged_logits, dim=-1)
                
                loss = F.kl_div(merged_log_probs, expert_probs, reduction='batchmean')
                return loss
                
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
                
    final_coeffs = coeffs.detach().clone()
    final_projs = {t: task_projs[t].detach().clone() for t in tasks} if opt_mode != "coeffs" else None
    return final_coeffs, final_projs

def main():
    parser = argparse.ArgumentParser(description="U-SASLA Ablation Studies")
    parser.add_argument("--rho", type=float, default=0.01, help="SAM rho parameter")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of calibration steps")
    parser.add_argument("--output_file", type=str, default="ablation_results.json", help="Output ablation JSON path")
    parser.add_argument("--selective_sam", action="store_true", help="Apply SAM only to projection layers, keeping merging coeffs un-perturbed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Set seed to {args.seed}. Selective SAM is: {args.selective_sam}")
    print("Preparing datasets...")
    datasets = get_datasets()
    
    # Load base model
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    train_and_merge._shared_model = model  # set the cached model in the module
    
    pretrained_visual_state = copy.deepcopy(model.visual.state_dict())
    text_heads = create_zero_shot_heads(model, tokenizer)
    
    tasks = ["mnist", "fashion", "cifar10"]
    expert_states = {}
    for t in tasks:
        weight_path = f"expert_{t}.pt"
        if os.path.exists(weight_path):
            expert_states[t] = torch.load(weight_path, map_location=device)
        else:
            raise FileNotFoundError(f"Expert weights {weight_path} not found. Please train/generate experts first.")

    # Create task vectors
    task_vectors = {}
    for t in tasks:
        tv = {}
        for name in pretrained_visual_state.keys():
            if name in expert_states[t]:
                tv[name] = expert_states[t][name] - pretrained_visual_state[name]
        task_vectors[t] = tv

    # Prepare Calibration Data Loaders
    cal_loaders = {t: DataLoader(datasets[t]["cal"], batch_size=32, shuffle=True) for t in tasks}
    
    # Define Ablation configurations
    ablation_configs = {
        "AdaMerging": {"use_sam": False, "parameter_wise": False, "opt_mode": "coeffs"},
        "SAM-AdaMerging": {"use_sam": True, "parameter_wise": False, "opt_mode": "coeffs"},
        "Single-Layer-GD": {"use_sam": False, "parameter_wise": False, "opt_mode": "proj"},
        "SAM-Single-Layer": {"use_sam": True, "parameter_wise": False, "opt_mode": "proj"},
        "PW-SAM-Single-Layer": {"use_sam": True, "parameter_wise": True, "opt_mode": "proj"},
        "SyMerge": {"use_sam": False, "parameter_wise": False, "opt_mode": "both"},
        "U-SASLA (Global SAM)": {"use_sam": True, "parameter_wise": False, "opt_mode": "both"},
        "PW-SASLA (Ours)": {"use_sam": True, "parameter_wise": True, "opt_mode": "both"}
    }
    
    results = {}
    
    for name, config in ablation_configs.items():
        print(f"\n>>> Running ablation: {name} (use_sam={config['use_sam']}, parameter_wise={config.get('parameter_wise', False)}, opt_mode={config['opt_mode']})")
        # For projection-only learning, we can use slightly different hyperparameters or the same ones. Let's use the provided args.
        coeffs, projs = run_calibration_ablation(
            model=model,
            pretrained_visual_state=pretrained_visual_state,
            task_vectors=task_vectors,
            text_heads=text_heads,
            cal_loaders=cal_loaders,
            expert_states=expert_states,
            use_sam=config["use_sam"],
            parameter_wise=config.get("parameter_wise", False),
            selective=args.selective_sam,
            opt_mode=config["opt_mode"],
            num_steps=args.num_steps,
            rho=args.rho,
            lr=args.lr
        )
        
        # Evaluate configuration
        accs_clean = {}
        accs_corr = {}
        for t in tasks:
            eval_model = get_merged_visual_model(
                pretrained_visual_state, task_vectors, coeffs, projs, active_task=t
            )
            loader_clean = DataLoader(datasets[t]["test_clean"], batch_size=128, shuffle=False)
            loader_corr = DataLoader(datasets[t]["test_corr"], batch_size=128, shuffle=False)
            accs_clean[t] = evaluate_model(eval_model, loader_clean, text_heads[t])
            accs_corr[t] = evaluate_model(eval_model, loader_corr, text_heads[t])
            
        avg_clean = sum(accs_clean.values()) / len(accs_clean)
        avg_corr = sum(accs_corr.values()) / len(accs_corr)
        
        print(f"Results for {name}:")
        print(f"  Clean: MNIST={accs_clean['mnist']:.4f}, Fashion={accs_clean['fashion']:.4f}, CIFAR10={accs_clean['cifar10']:.4f} | Avg={avg_clean:.4f}")
        print(f"  Corr:  MNIST={accs_corr['mnist']:.4f}, Fashion={accs_corr['fashion']:.4f}, CIFAR10={accs_corr['cifar10']:.4f} | Avg={avg_corr:.4f}")
        
        results[name] = {
            "clean": accs_clean,
            "corrupted": accs_corr,
            "avg_clean": avg_clean,
            "avg_corrupted": avg_corr,
            "coeffs": coeffs.tolist()
        }
        
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAblation studies complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
