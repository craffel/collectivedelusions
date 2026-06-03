import os
import copy
import argparse
import torch
import torch.nn as nn

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on the GPU node
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from data import get_multi_task_datasets, get_calibration_subset
from methods import (
    get_merged_state_dict,
    apply_sp_taac,
    apply_slr_wbc,
    apply_wrsa,
    get_task_prototypes,
    mspr_route_sample
)

def load_expert_model(dataset_name, device='cuda'):
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        model.fc
    )
    checkpoint_path = f"checkpoints/{dataset_name}_expert.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Expert model not found at {checkpoint_path}. Please train experts first.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model_on_task(model, test_loader, task_head, device='cuda'):
    """
    Evaluates the model on a specific task's test loader using the task's custom head.
    """
    model.eval()
    # Temporarily swap classification head to the task-specific head
    orig_fc = model.fc
    model.fc = task_head
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    # Restore original head
    model.fc = orig_fc
    acc = 100.0 * correct / total
    return acc

def ssr_merge_eval(merged_model, task_specific_models, prototypes, test_loader, device='cuda'):
    """
    Evaluates SSR-Merge (Our proposed Routed Low-Rank Calibration method) on a test loader.
    """
    merged_model.eval()
    for m in task_specific_models.values():
        m.eval()
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Route each sample in the batch at Layer 2
            routed_tasks = mspr_route_sample(merged_model, inputs, prototypes, device=device)
            
            batch_size = inputs.size(0)
            outputs = torch.zeros(batch_size, 10, device=device)
            
            # Early layers of merged_model (shared across all routed paths)
            x_early = merged_model.conv1(inputs)
            x_early = merged_model.bn1(x_early)
            x_early = merged_model.relu(x_early)
            x_early = merged_model.maxpool(x_early)
            x_early = merged_model.layer1(x_early)
            x_early = merged_model.layer2(x_early)
            
            # Group by routed task and run deep layers
            for task_name in prototypes.keys():
                indices = [i for i, t in enumerate(routed_tasks) if t == task_name]
                if len(indices) == 0:
                    continue
                    
                indices_tensor = torch.tensor(indices, device=device)
                x_group = x_early[indices_tensor]
                
                # Run deep layers of the task-specific calibrated model for this group
                ts_model = task_specific_models[task_name]
                x_deep = ts_model.layer3(x_group)
                x_deep = ts_model.layer4(x_deep)
                x_deep = ts_model.avgpool(x_deep)
                x_deep = torch.flatten(x_deep, 1)
                outputs_group = ts_model.fc(x_deep)
                
                outputs[indices_tensor] = outputs_group
                
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100.0 * correct / total
    return acc

def run_evaluation(device='cuda', seed=42):
    print(f"==================================================")
    print(f"🚀 Running Comprehensive Merging & Calibration Evaluation")
    print(f"==================================================")
    
    # 1. Load experts
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = {}
    expert_heads = {}
    expert_state_dicts = []
    
    for t in tasks:
        model = load_expert_model(t, device=device)
        expert_models[t] = model
        expert_heads[t] = copy.deepcopy(model.fc)
        expert_state_dicts.append(copy.deepcopy(model.state_dict()))
        
    # Get base pre-trained state dict
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
    
    # 2. Get Datasets
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed)
    test_loaders = {t: DataLoader(test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
    
    # Evaluate Experts (Oracle Upper Bound)
    print("\n--- 🌟 Oracle Experts Accuracy ---")
    oracle_accs = {}
    for t in tasks:
        acc = evaluate_model_on_task(expert_models[t], test_loaders[t], expert_heads[t], device=device)
        oracle_accs[t] = acc
        print(f"Oracle Expert ({t}): {acc:.2f}%")
    print(f"Average Oracle Accuracy: {sum(oracle_accs.values())/3:.2f}%")
    
    # 3. Baseline Merging (WA and TA)
    print("\n--- 📉 Uncalibrated Baselines ---")
    # Weight Averaging
    wa_state_dict = get_merged_state_dict(expert_state_dicts, mode='wa')
    wa_model = copy.deepcopy(base_model).to(device)
    wa_model.load_state_dict(wa_state_dict)
    
    wa_accs = {}
    for t in tasks:
        acc = evaluate_model_on_task(wa_model, test_loaders[t], expert_heads[t], device=device)
        wa_accs[t] = acc
        print(f"Weight Averaging ({t}): {acc:.2f}%")
    print(f"Average WA Accuracy: {sum(wa_accs.values())/3:.2f}%")
    
    # Task Arithmetic (lambda = 0.3)
    ta_state_dict = get_merged_state_dict(expert_state_dicts, mode='ta', lam=0.3, base_state_dict=base_state_dict)
    ta_model = copy.deepcopy(base_model).to(device)
    ta_model.load_state_dict(ta_state_dict)
    
    ta_accs = {}
    for t in tasks:
        acc = evaluate_model_on_task(ta_model, test_loaders[t], expert_heads[t], device=device)
        ta_accs[t] = acc
        print(f"Task Arithmetic ({t}): {acc:.2f}%")
    print(f"Average TA Accuracy: {sum(ta_accs.values())/3:.2f}%")
    
    # 4. Calibration Budgets N in [16, 64, 128]
    budgets = [16, 64, 128]
    
    results = {
        'N': [],
        'method': [],
        'mnist': [],
        'fmnist': [],
        'cifar10': [],
        'avg': []
    }
    
    for N in budgets:
        print(f"\n=========================================")
        print(f"📦 Calibration Budget N = {N} samples per task")
        print(f"=========================================")
        
        # Prepare calibration subsets and loaders
        cal_subsets = {t: get_calibration_subset(train_datasets[t], N, seed=seed) for t in tasks}
        cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
        
        # Prepare pooled calibration batches for joint calibration (like SLR-WBC and SP-TAAC)
        # We pool samples from all tasks to form joint calibration batches
        joint_cal_samples = []
        for t in tasks:
            for inputs, _ in cal_loaders[t]:
                joint_cal_samples.append(inputs)
        joint_cal_batches = [torch.cat(joint_cal_samples, dim=0)]
        
        # ---- Method A: SP-TAAC (Activation scaling) ----
        print("\n--- 🛠️ Evaluating SP-TAAC ---")
        spt_model = copy.deepcopy(wa_model)
        # Collect expert models list
        em_list = [expert_models[t] for t in tasks]
        apply_sp_taac(spt_model, em_list, joint_cal_batches, device=device)
        
        spt_accs = {}
        for t in tasks:
            acc = evaluate_model_on_task(spt_model, test_loaders[t], expert_heads[t], device=device)
            spt_accs[t] = acc
            print(f"SP-TAAC ({t}): {acc:.2f}%")
        spt_avg = sum(spt_accs.values())/3
        print(f"Average SP-TAAC: {spt_avg:.2f}%")
        
        results['N'].append(N)
        results['method'].append('SP-TAAC')
        results['mnist'].append(spt_accs['mnist'])
        results['fmnist'].append(spt_accs['fmnist'])
        results['cifar10'].append(spt_accs['cifar10'])
        results['avg'].append(spt_avg)
        
        # ---- Method B: SLR-WBC (Joint Low-Rank Weight Calibration) ----
        print("\n--- 🛠️ Evaluating SLR-WBC ---")
        slr_model = copy.deepcopy(wa_model)
        # Calibrate deep layers with SLR-WBC (jointly)
        apply_slr_wbc(slr_model, em_list, joint_cal_batches, rank=2, reg=0.5, device=device)
        
        slr_accs = {}
        for t in tasks:
            acc = evaluate_model_on_task(slr_model, test_loaders[t], expert_heads[t], device=device)
            slr_accs[t] = acc
            print(f"SLR-WBC ({t}): {acc:.2f}%")
        slr_avg = sum(slr_accs.values())/3
        print(f"Average SLR-WBC: {slr_avg:.2f}%")
        
        results['N'].append(N)
        results['method'].append('SLR-WBC')
        results['mnist'].append(slr_accs['mnist'])
        results['fmnist'].append(slr_accs['fmnist'])
        results['cifar10'].append(slr_accs['cifar10'])
        results['avg'].append(slr_avg)
        
        # ---- Method C: WRSA (Wiener-Regularized Spectral Alignment) ----
        print("\n--- 🛠️ Evaluating WRSA ---")
        wrsa_model = copy.deepcopy(wa_model)
        hooks = apply_wrsa(wrsa_model, em_list, joint_cal_batches, c=0.30, device=device)
        
        wrsa_accs = {}
        for t in tasks:
            acc = evaluate_model_on_task(wrsa_model, test_loaders[t], expert_heads[t], device=device)
            wrsa_accs[t] = acc
            print(f"WRSA ({t}): {acc:.2f}%")
        wrsa_avg = sum(wrsa_accs.values())/3
        print(f"Average WRSA: {wrsa_avg:.2f}%")
        
        # Remove hooks to clean up
        for h in hooks:
            h.remove()
            
        results['N'].append(N)
        results['method'].append('WRSA')
        results['mnist'].append(wrsa_accs['mnist'])
        results['fmnist'].append(wrsa_accs['fmnist'])
        results['cifar10'].append(wrsa_accs['cifar10'])
        results['avg'].append(wrsa_avg)
        
        # ---- Method D: MSPR (Minimalist Static Prototype Routing) ----
        print("\n--- 🛠️ Evaluating MSPR ---")
        # Extract task prototypes at Layer 2 from uncalibrated merged model
        prototypes = get_task_prototypes(wa_model, cal_loaders, device=device)
        
        mspr_accs = {}
        # Route to uncalibrated WA model with task head
        # MSPR evaluates the test set and routes classification head
        for t in tasks:
            correct = 0
            total = 0
            for inputs, targets in test_loaders[t]:
                inputs, targets = inputs.to(device), targets.to(device)
                routed_tasks = mspr_route_sample(wa_model, inputs, prototypes, device=device)
                
                # Run forward pass of wa_model
                outputs = wa_model(inputs)
                
                # Route classification head dynamically
                # Since we are evaluating accuracy, for each sample we take the logit corresponding
                # to the routed classification head!
                # Wait, to do this efficiently:
                outputs_routed = torch.zeros(inputs.size(0), 10, device=device)
                
                # Each expert head is evaluated
                expert_outputs = {}
                for task_name in tasks:
                    # Temporarily put expert head
                    wa_model.fc = expert_heads[task_name]
                    expert_outputs[task_name] = wa_model(inputs)
                    
                for i, r_task in enumerate(routed_tasks):
                    outputs_routed[i] = expert_outputs[r_task][i]
                    
                _, predicted = outputs_routed.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            mspr_accs[t] = 100.0 * correct / total
            print(f"MSPR ({t}): {mspr_accs[t]:.2f}%")
        mspr_avg = sum(mspr_accs.values())/3
        print(f"Average MSPR: {mspr_avg:.2f}%")
        
        results['N'].append(N)
        results['method'].append('MSPR')
        results['mnist'].append(mspr_accs['mnist'])
        results['fmnist'].append(mspr_accs['fmnist'])
        results['cifar10'].append(mspr_accs['cifar10'])
        results['avg'].append(mspr_avg)
        
        # ---- Method E: SSR-Merge (Our Proposed Method) ----
        print("\n--- 💎 Evaluating SSR-Merge (Ours) ---")
        # Step 1: Compute task-specific SLR-WBC weight corrections
        # For each task, we calibrate specifically to its expert model
        task_specific_models = {}
        for t in tasks:
            ts_model = copy.deepcopy(wa_model)
            # Calibration batches specifically for task t
            task_cal_batches = []
            for inputs, _ in cal_loaders[t]:
                task_cal_batches.append(inputs)
            task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
            
            # Apply task-specific SLR-WBC to deep layers
            apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=4, reg=0.1, device=device)
            
            # Ensure classification head is set
            ts_model.fc = expert_heads[t]
            task_specific_models[t] = ts_model
            
        # Step 2: Extract Layer 2 prototypes
        prototypes = get_task_prototypes(wa_model, cal_loaders, device=device)
        
        # Step 3: Evaluate SSR-Merge using batch routing
        ssr_accs = {}
        for t in tasks:
            acc = ssr_merge_eval(wa_model, task_specific_models, prototypes, test_loaders[t], device=device)
            ssr_accs[t] = acc
            print(f"SSR-Merge ({t}): {acc:.2f}%")
        ssr_avg = sum(ssr_accs.values())/3
        print(f"Average SSR-Merge: {ssr_avg:.2f}%")
        
        results['N'].append(N)
        results['method'].append('SSR-Merge (Ours)')
        results['mnist'].append(ssr_accs['mnist'])
        results['fmnist'].append(ssr_accs['fmnist'])
        results['cifar10'].append(ssr_accs['cifar10'])
        results['avg'].append(ssr_avg)
        
    print("\n" + "="*50)
    print("📊 FINAL EXPERIMENTAL RESULTS SUMMARY TABLE")
    print("="*50)
    print(f"{'N':<5} | {'Method':<20} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*65)
    for i in range(len(results['N'])):
        print(f"{results['N'][i]:<5} | {results['method'][i]:<20} | {results['mnist'][i]:<8.2f} | {results['fmnist'][i]:<8.2f} | {results['cifar10'][i]:<8.2f} | {results['avg'][i]:<8.2f}")
    print("="*65)
    
    # Log results to progress.md
    with open('progress.md', 'a') as f:
        f.write(f"\n\n### Experimental Run (Seed {seed})\n")
        f.write(f"| N | Method | MNIST | F-MNIST | CIFAR-10 | Average |\n")
        f.write(f"|---|--------|-------|---------|----------|---------|\n")
        f.write(f"| - | Oracle Experts | {oracle_accs['mnist']:.2f}% | {oracle_accs['fmnist']:.2f}% | {oracle_accs['cifar10']:.2f}% | {sum(oracle_accs.values())/3:.2f}% |\n")
        f.write(f"| - | Weight Averaging (WA) | {wa_accs['mnist']:.2f}% | {wa_accs['fmnist']:.2f}% | {wa_accs['cifar10']:.2f}% | {sum(wa_accs.values())/3:.2f}% |\n")
        f.write(f"| - | Task Arithmetic (TA) | {ta_accs['mnist']:.2f}% | {ta_accs['fmnist']:.2f}% | {ta_accs['cifar10']:.2f}% | {sum(ta_accs.values())/3:.2f}% |\n")
        for i in range(len(results['N'])):
            f.write(f"| {results['N'][i]} | {results['method'][i]} | {results['mnist'][i]:.2f}% | {results['fmnist'][i]:.2f}% | {results['cifar10'][i]:.2f}% | {results['avg'][i]:.2f}% |\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_evaluation(device=args.device, seed=args.seed)
