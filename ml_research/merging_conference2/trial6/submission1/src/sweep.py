import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on the GPU node
torch.backends.cudnn.enabled = False

from data import get_multi_task_datasets, get_calibration_subset
from methods import (
    get_merged_state_dict,
    apply_sp_taac,
    apply_slr_wbc,
    apply_wrsa,
    get_task_prototypes,
    mspr_route_sample
)
from evaluate import load_expert_model, evaluate_model_on_task

def run_early(model, inputs, layer_name):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    if layer_name == 'layer1':
        return x
    x = model.layer2(x)
    if layer_name == 'layer2':
        return x
    x = model.layer3(x)
    if layer_name == 'layer3':
        return x
    raise ValueError(f"Unsupported routing layer: {layer_name}")

def run_deep(ts_model, x_early, layer_name):
    x = x_early
    if layer_name == 'layer1':
        x = ts_model.layer2(x)
        x = ts_model.layer3(x)
        x = ts_model.layer4(x)
    elif layer_name == 'layer2':
        x = ts_model.layer3(x)
        x = ts_model.layer4(x)
    elif layer_name == 'layer3':
        x = ts_model.layer4(x)
    x = ts_model.avgpool(x)
    x = torch.flatten(x, 1)
    return ts_model.fc(x)

def get_task_prototypes_custom(merged_model, calibration_loaders, layer_name='layer2', normalize=True, device='cuda'):
    merged_model.eval()
    prototypes = {}
    layer_module = dict(merged_model.named_modules())[layer_name]
    
    for task_name, loader in calibration_loaders.items():
        layer_acts = []
        def hook_fn(module, input, output):
            pooled = output.mean(dim=(2, 3)) # (B, C)
            layer_acts.append(pooled.detach().cpu())
            
        handle = layer_module.register_forward_hook(hook_fn)
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = merged_model(inputs)
        handle.remove()
        
        pooled_all = torch.cat(layer_acts, dim=0)
        avg_proto = pooled_all.mean(dim=0)
        if normalize:
            normalized_proto = avg_proto / (avg_proto.norm(p=2) + 1e-8)
            prototypes[task_name] = normalized_proto
        else:
            prototypes[task_name] = avg_proto
        
    return prototypes

def mspr_route_sample_custom(merged_model, x, prototypes, layer_name='layer2', metric='cosine', device='cuda'):
    merged_model.eval()
    layer_module = dict(merged_model.named_modules())[layer_name]
    captured_pooled = []
    def hook_fn(module, input, output):
        pooled = output.mean(dim=(2, 3))
        captured_pooled.append(pooled.detach())
        
    handle = layer_module.register_forward_hook(hook_fn)
    _ = merged_model(x)
    handle.remove()
    
    v = captured_pooled[0]
    task_names = list(prototypes.keys())
    proto_matrix = torch.stack([prototypes[name].to(device) for name in task_names])
    
    if metric == 'cosine':
        v_norm = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-8)
        proto_norm = proto_matrix / (proto_matrix.norm(p=2, dim=1, keepdim=True) + 1e-8)
        similarities = v_norm @ proto_norm.T
        best_task_indices = similarities.argmax(dim=1).cpu().numpy()
    elif metric == 'dot_product':
        similarities = v @ proto_matrix.T
        best_task_indices = similarities.argmax(dim=1).cpu().numpy()
    elif metric == 'euclidean':
        diff = v.unsqueeze(1) - proto_matrix.unsqueeze(0)
        dists = torch.linalg.vector_norm(diff, ord=2, dim=2)
        best_task_indices = dists.argmin(dim=1).cpu().numpy()
    elif metric == 'manhattan':
        diff = v.unsqueeze(1) - proto_matrix.unsqueeze(0)
        dists = torch.linalg.vector_norm(diff, ord=1, dim=2)
        best_task_indices = dists.argmin(dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown routing metric: {metric}")
        
    routed_tasks = [task_names[idx] for idx in best_task_indices]
    return routed_tasks

def ssr_merge_eval_custom(merged_model, task_specific_models, prototypes, test_loader, layer_name='layer2', metric='cosine', device='cuda'):
    merged_model.eval()
    for m in task_specific_models.values():
        m.eval()
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            routed_tasks = mspr_route_sample_custom(merged_model, inputs, prototypes, layer_name=layer_name, metric=metric, device=device)
            
            batch_size = inputs.size(0)
            outputs = torch.zeros(batch_size, 10, device=device)
            
            x_early = run_early(merged_model, inputs, layer_name)
            
            for task_name in prototypes.keys():
                indices = [i for i, t in enumerate(routed_tasks) if t == task_name]
                if len(indices) == 0:
                    continue
                    
                indices_tensor = torch.tensor(indices, device=device)
                x_group = x_early[indices_tensor]
                
                ts_model = task_specific_models[task_name]
                outputs_group = run_deep(ts_model, x_group, layer_name)
                outputs[indices_tensor] = outputs_group
                
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100.0 * correct / total

def run_sweeps(device='cuda'):
    print("==================================================")
    print("🌟 STARTING COMPREHENSIVE EMPIRICAL SWEEPS 🌟")
    print("==================================================")
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = {}
    expert_heads = {}
    expert_state_dicts = []
    
    for t in tasks:
        model = load_expert_model(t, device=device)
        expert_models[t] = model
        expert_heads[t] = copy.deepcopy(model.fc)
        expert_state_dicts.append(copy.deepcopy(model.state_dict()))
        
    try:
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        base_model = resnet18(pretrained=True)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        base_model.fc
    )
    base_state_dict = copy.deepcopy(base_model.state_dict())
    
    wa_state_dict = get_merged_state_dict(expert_state_dicts, mode='wa')
    wa_model = copy.deepcopy(base_model).to(device)
    wa_model.load_state_dict(wa_state_dict)
    
    # ----------------------------------------------------
    # Sweep 1: Multi-Seed Statistical Sweep
    # ----------------------------------------------------
    print("\n--- 1. Multi-Seed Statistical Sweep ---")
    seeds = [42, 100, 2026]
    budgets = [16, 64, 128]
    methods = ['WA', 'TA', 'SP-TAAC', 'SLR-WBC', 'WRSA', 'MSPR', 'SSR-Merge (Ours)']
    
    # Structure to hold results: {method: {N: [seed1, seed2, seed3]}}
    seed_results = {m: {N: [] for N in budgets} for m in methods}
    seed_results['Oracle'] = [] # Oracle is independent of budget
    
    for seed in seeds:
        print(f"\nProcessing Seed {seed}...")
        train_datasets, test_datasets = get_multi_task_datasets(seed=seed)
        test_loaders = {t: DataLoader(test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
        
        # Oracle
        oracle_accs = []
        for t in tasks:
            acc = evaluate_model_on_task(expert_models[t], test_loaders[t], expert_heads[t], device=device)
            oracle_accs.append(acc)
        oracle_avg = sum(oracle_accs) / 3.0
        seed_results['Oracle'].append(oracle_avg)
        print(f"  Oracle Average: {oracle_avg:.2f}%")
        
        # Baselines (Independent of calibration subset but depends on test subset/shuffle if any, though test set is full, so it's mostly constant)
        # Weight Averaging
        wa_accs = []
        for t in tasks:
            acc = evaluate_model_on_task(wa_model, test_loaders[t], expert_heads[t], device=device)
            wa_accs.append(acc)
        wa_avg = sum(wa_accs) / 3.0
        
        # Task Arithmetic
        ta_state_dict = get_merged_state_dict(expert_state_dicts, mode='ta', lam=0.3, base_state_dict=base_state_dict)
        ta_model_inst = copy.deepcopy(base_model).to(device)
        ta_model_inst.load_state_dict(ta_state_dict)
        ta_accs = []
        for t in tasks:
            acc = evaluate_model_on_task(ta_model_inst, test_loaders[t], expert_heads[t], device=device)
            ta_accs.append(acc)
        ta_avg = sum(ta_accs) / 3.0
        
        for N in budgets:
            print(f"  Evaluating Budget N = {N}...")
            cal_subsets = {t: get_calibration_subset(train_datasets[t], N, seed=seed) for t in tasks}
            cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
            
            joint_cal_samples = []
            for t in tasks:
                for inputs, _ in cal_loaders[t]:
                    joint_cal_samples.append(inputs)
            joint_cal_batches = [torch.cat(joint_cal_samples, dim=0)]
            
            # WA and TA
            seed_results['WA'][N].append(wa_avg)
            seed_results['TA'][N].append(ta_avg)
            
            # SP-TAAC
            spt_model = copy.deepcopy(wa_model)
            apply_sp_taac(spt_model, [expert_models[t] for t in tasks], joint_cal_batches, device=device)
            spt_accs = []
            for t in tasks:
                acc = evaluate_model_on_task(spt_model, test_loaders[t], expert_heads[t], device=device)
                spt_accs.append(acc)
            seed_results['SP-TAAC'][N].append(sum(spt_accs) / 3.0)
            
            # SLR-WBC
            slr_model = copy.deepcopy(wa_model)
            apply_slr_wbc(slr_model, [expert_models[t] for t in tasks], joint_cal_batches, rank=2, reg=0.5, device=device)
            slr_accs = []
            for t in tasks:
                acc = evaluate_model_on_task(slr_model, test_loaders[t], expert_heads[t], device=device)
                slr_accs.append(acc)
            seed_results['SLR-WBC'][N].append(sum(slr_accs) / 3.0)
            
            # WRSA
            wrsa_model = copy.deepcopy(wa_model)
            hooks = apply_wrsa(wrsa_model, [expert_models[t] for t in tasks], joint_cal_batches, c=0.30, device=device)
            wrsa_accs = []
            for t in tasks:
                acc = evaluate_model_on_task(wrsa_model, test_loaders[t], expert_heads[t], device=device)
                wrsa_accs.append(acc)
            seed_results['WRSA'][N].append(sum(wrsa_accs) / 3.0)
            for h in hooks:
                h.remove()
                
            # MSPR
            prototypes = get_task_prototypes(wa_model, cal_loaders, device=device)
            mspr_correct = 0
            mspr_total = 0
            for t in tasks:
                for inputs, targets in test_loaders[t]:
                    inputs, targets = inputs.to(device), targets.to(device)
                    routed_tasks = mspr_route_sample(wa_model, inputs, prototypes, device=device)
                    expert_outputs = {}
                    for tn in tasks:
                        wa_model.fc = expert_heads[tn]
                        expert_outputs[tn] = wa_model(inputs)
                    outputs_routed = torch.zeros(inputs.size(0), 10, device=device)
                    for i, r_task in enumerate(routed_tasks):
                        outputs_routed[i] = expert_outputs[r_task][i]
                    _, predicted = outputs_routed.max(1)
                    mspr_total += targets.size(0)
                    mspr_correct += predicted.eq(targets).sum().item()
            seed_results['MSPR'][N].append(100.0 * mspr_correct / mspr_total)
            
            # SSR-Merge (Ours)
            task_specific_models = {}
            for t in tasks:
                ts_model = copy.deepcopy(wa_model)
                task_cal_batches = []
                for inputs, _ in cal_loaders[t]:
                    task_cal_batches.append(inputs)
                task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
                
                apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=4, reg=0.1, device=device)
                ts_model.fc = expert_heads[t]
                task_specific_models[t] = ts_model
                
            ssr_accs = []
            for t in tasks:
                acc = ssr_merge_eval_custom(wa_model, task_specific_models, prototypes, test_loaders[t], layer_name='layer2', device=device)
                ssr_accs.append(acc)
            seed_results['SSR-Merge (Ours)'][N].append(sum(ssr_accs) / 3.0)
            
    # Compile multi-seed summary
    print("\n--- Summary of Seed Sweeps (Mean ± Std) ---")
    summary_data = {}
    for m in methods:
        summary_data[m] = {}
        for N in budgets:
            accs = seed_results[m][N]
            mean_val = sum(accs) / len(accs)
            std_val = (sum((x - mean_val)**2 for x in accs) / len(accs))**0.5
            summary_data[m][N] = (mean_val, std_val)
            print(f"{m} (N={N}): {mean_val:.2f}% ± {std_val:.2f}%")
            
    # ----------------------------------------------------
    # Sweep 2: Hyperparameter Sweeps (Rank and Regularization)
    # ----------------------------------------------------
    print("\n--- 2. Hyperparameter Grid Search for SSR-Merge ---")
    # At N=128, seed=42
    N_sweep = 128
    seed_sweep = 42
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed_sweep)
    test_loaders = {t: DataLoader(test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
    cal_subsets = {t: get_calibration_subset(train_datasets[t], N_sweep, seed=seed_sweep) for t in tasks}
    cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
    prototypes = get_task_prototypes_custom(wa_model, cal_loaders, layer_name='layer2', device=device)
    
    ranks = [1, 2, 4, 8]
    regs = [0.01, 0.1, 0.5]
    hyperparam_results = {}
    
    for r in ranks:
        hyperparam_results[r] = {}
        for reg_val in regs:
            print(f"Running SSR-Merge Grid Search: Rank={r}, Reg={reg_val}...")
            task_specific_models = {}
            for t in tasks:
                ts_model = copy.deepcopy(wa_model)
                task_cal_batches = []
                for inputs, _ in cal_loaders[t]:
                    task_cal_batches.append(inputs)
                task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
                
                apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=r, reg=reg_val, device=device)
                ts_model.fc = expert_heads[t]
                task_specific_models[t] = ts_model
                
            ssr_accs = []
            for t in tasks:
                acc = ssr_merge_eval_custom(wa_model, task_specific_models, prototypes, test_loaders[t], layer_name='layer2', device=device)
                ssr_accs.append(acc)
            avg_acc = sum(ssr_accs) / 3.0
            hyperparam_results[r][reg_val] = avg_acc
            print(f"  Average Accuracy: {avg_acc:.2f}%")
            
    # ----------------------------------------------------
    # Sweep 3: Ablation 1 - Routing Layer Choice
    # ----------------------------------------------------
    print("\n--- 3. Ablation 1: Routing Layer Choice ---")
    routing_layers = ['layer1', 'layer2', 'layer3']
    routing_results = {}
    
    # We calibrate deep layers of task-specific models using rank=4, reg=0.1
    # We load task-specific models calibrated with rank=4, reg=0.1
    task_specific_models_std = {}
    for t in tasks:
        ts_model = copy.deepcopy(wa_model)
        task_cal_batches = []
        for inputs, _ in cal_loaders[t]:
            task_cal_batches.append(inputs)
        task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
        
        apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=4, reg=0.1, device=device)
        ts_model.fc = expert_heads[t]
        task_specific_models_std[t] = ts_model
        
    for layer in routing_layers:
        print(f"Evaluating SSR-Merge with routing at {layer}...")
        # Get prototypes at this layer
        proto_custom = get_task_prototypes_custom(wa_model, cal_loaders, layer_name=layer, device=device)
        
        ssr_accs = []
        for t in tasks:
            acc = ssr_merge_eval_custom(wa_model, task_specific_models_std, proto_custom, test_loaders[t], layer_name=layer, device=device)
            ssr_accs.append(acc)
        avg_acc = sum(ssr_accs) / 3.0
        routing_results[layer] = avg_acc
        print(f"  Routing at {layer} average accuracy: {avg_acc:.2f}%")
        
    # ----------------------------------------------------
    # Sweep 4: Ablation 2 - Calibrated Layers
    # ----------------------------------------------------
    print("\n--- 4. Ablation 2: Calibrated Layers ---")
    # We use routing layer = 'layer2', SVD rank=4, reg=0.1
    calibration_scenarios = {
        'layer4 only': ['layer4'],
        'layer3 + layer4 (default)': ['layer3', 'layer4'],
        'layer2 + layer3 + layer4': ['layer2', 'layer3', 'layer4']
    }
    calib_results = {}
    prototypes_std = get_task_prototypes_custom(wa_model, cal_loaders, layer_name='layer2', device=device)
    
    for name, prefixes in calibration_scenarios.items():
        print(f"Evaluating SSR-Merge with calibration on {name}...")
        task_specific_models_cal = {}
        for t in tasks:
            ts_model = copy.deepcopy(wa_model)
            task_cal_batches = []
            for inputs, _ in cal_loaders[t]:
                task_cal_batches.append(inputs)
            task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
            
            apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=4, reg=0.1, device=device, layer_prefixes=prefixes)
            ts_model.fc = expert_heads[t]
            task_specific_models_cal[t] = ts_model
            
        ssr_accs = []
        for t in tasks:
            acc = ssr_merge_eval_custom(wa_model, task_specific_models_cal, prototypes_std, test_loaders[t], layer_name='layer2', device=device)
            ssr_accs.append(acc)
        avg_acc = sum(ssr_accs) / 3.0
        calib_results[name] = avg_acc
        print(f"  Calibration on {name} average accuracy: {avg_acc:.2f}%")
        
    # ----------------------------------------------------
    # Sweep 5: Ablation 3 - Routing Metric Choice
    # ----------------------------------------------------
    print("\n--- 5. Ablation 3: Routing Metric Choice ---")
    routing_metrics = ['cosine', 'dot_product', 'euclidean', 'manhattan']
    metric_results = {}
    
    for metric in routing_metrics:
        print(f"Evaluating SSR-Merge with {metric} routing metric...")
        normalize_proto = (metric == 'cosine')
        proto_custom = get_task_prototypes_custom(wa_model, cal_loaders, layer_name='layer2', normalize=normalize_proto, device=device)
        
        ssr_accs = []
        for t in tasks:
            acc = ssr_merge_eval_custom(wa_model, task_specific_models_std, proto_custom, test_loaders[t], layer_name='layer2', metric=metric, device=device)
            ssr_accs.append(acc)
        avg_acc = sum(ssr_accs) / 3.0
        metric_results[metric] = avg_acc
        print(f"  {metric} routing metric average accuracy: {avg_acc:.2f}%")
        
    # Save all results to sweep_results.json
    results_to_save = {
        'multi_seed': seed_results,
        'summary': summary_data,
        'hyperparams': hyperparam_results,
        'routing_ablation': routing_results,
        'calibration_ablation': calib_results,
        'metric_ablation': metric_results
    }
    
    # Need to convert dictionary keys if they are ints for json serialization
    # hyperparams has integer keys (rank)
    serializable_hyperparams = {}
    for r, reg_dict in hyperparam_results.items():
        serializable_hyperparams[str(r)] = {str(reg): val for reg, val in reg_dict.items()}
    results_to_save['hyperparams'] = serializable_hyperparams
    
    with open('sweep_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=4)
    print("\n✅ All sweep results successfully written to sweep_results.json!")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_sweeps(device=device)
