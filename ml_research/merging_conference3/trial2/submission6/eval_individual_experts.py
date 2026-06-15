import os
import torch
import numpy as np
import timm
from run_experiments import (
    device, TASKS, ExpertModel, get_dataset, transform, transform_gray, quantize_tensor
)
from torch.utils.data import DataLoader, Subset

def evaluate_expert_model(model, task, seed, bits=None):
    t_gray = transform_gray if task in ["MNIST", "FashionMNIST"] else transform
    eval_dataset = get_dataset(task, train=False, transform=t_gray)
    rng = np.random.default_rng(seed)
    eval_indices = rng.permutation(len(eval_dataset))[:512]
    eval_loader = DataLoader(Subset(eval_dataset, eval_indices), batch_size=32, shuffle=False)
    
    # Quantize backbone weights if bits is specified
    model_eval = copy_model(model)
    if bits is not None:
        state_dict = model_eval.backbone.state_dict()
        q_state_dict = {}
        for name, param in state_dict.items():
            if "weight" in name and param.dim() > 1:
                q_state_dict[name] = quantize_tensor(param, bits)
            else:
                q_state_dict[name] = param
        model_eval.backbone.load_state_dict(q_state_dict)
        
    model_eval.eval()
    correct = 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            out = model_eval(x)
            correct += (out.argmax(dim=-1) == y).sum().item()
    return correct / 512

def copy_model(model):
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    m = ExpertModel(backbone).to(device)
    m.load_state_dict(model.state_dict())
    return m

def main():
    seeds = [42, 100, 2026]
    configs = [None, 8, 4]
    
    # Store results: config -> seed -> task -> acc
    raw_results = {None: {}, 8: {}, 4: {}}
    for c in configs:
        for s in seeds:
            raw_results[c][s] = {}
            
    for seed in seeds:
        print(f"Evaluating Seed {seed}...")
        for task in TASKS:
            checkpoint_path = f"checkpoints/seed_{seed}_{task}_expert.pt"
            if not os.path.exists(checkpoint_path):
                print(f"Warning: checkpoint {checkpoint_path} not found.")
                continue
            
            # Load expert
            backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
            model = ExpertModel(backbone).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            
            for config in configs:
                acc = evaluate_expert_model(model, task, seed, bits=config)
                raw_results[config][seed][task] = acc
                print(f"  Task {task} ({'FP16' if config is None else f'{config}-bit'}): {acc*100:.2f}%")
                
    # Aggregate and print results
    for config in configs:
        config_name = "FP16" if config is None else f"{config}-Bit"
        print(f"\n==================== {config_name} Results ====================")
        
        task_accs_all_seeds = {task: [] for task in TASKS}
        avg_accs_all_seeds = []
        
        for seed in seeds:
            seed_accs = []
            for task in TASKS:
                if task in raw_results[config][seed]:
                    acc = raw_results[config][seed][task] * 100
                    task_accs_all_seeds[task].append(acc)
                    seed_accs.append(acc)
            if seed_accs:
                avg_accs_all_seeds.append(np.mean(seed_accs))
                
        # Compute mean and std dev
        row_str = f"| Individual Experts ({config_name}) | "
        for task in TASKS:
            mean = np.mean(task_accs_all_seeds[task])
            std = np.std(task_accs_all_seeds[task])
            row_str += f"${mean:.2f} \\pm {std:.2f}$ | "
        
        mean_avg = np.mean(avg_accs_all_seeds)
        std_avg = np.std(avg_accs_all_seeds)
        row_str += f"$\\mathbf{{{mean_avg:.2f} \\pm {std_avg:.2f}}}$ |"
        print(row_str)

if __name__ == "__main__":
    main()
