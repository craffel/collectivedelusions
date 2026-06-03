import torch
import os
import copy
import json
import numpy as np
import random
from models import (
    get_model, get_dataloaders, evaluate_model, set_seed,
    quantize_model, calibrate_bn,
    merge_wa, merge_ta, merge_wcpr, merge_sc_wcpr,
    sparsify_and_merge_task_vectors
)
from torchvision import transforms

def apply_corruption(inputs, corruption_type):
    if corruption_type == 'noise':
        return inputs + torch.randn_like(inputs) * 0.1
    elif corruption_type == 'blur':
        return transforms.functional.gaussian_blur(inputs, kernel_size=[3, 3], sigma=[1.0, 1.0])
    return inputs

def apply_merged_backbone(expert, merged_model, arch):
    merged_state = merged_model.state_dict()
    expert_state = expert.state_dict()
    
    for name in expert_state.keys():
        if arch.lower() == 'resnet18' and name.startswith('backbone.'):
            expert_state[name].copy_(merged_state[name])
        elif arch.lower() == 'mlp' and not name.startswith('out.'):
            expert_state[name].copy_(merged_state[name])
            
    expert.load_state_dict(expert_state)
    return expert

def evaluate_merged_model_all_tasks(merged_model, experts_dict, test_loaders, arch, device, corruption=None, num_bits=None, per_channel=False):
    # Quantize model if specified
    if num_bits is not None:
        merged_model_eval = quantize_model(merged_model, num_bits, per_channel)
    else:
        merged_model_eval = merged_model
        
    accuracies = {}
    for task_name, expert in experts_dict.items():
        # Apply merged backbone in-place (no deepcopy!)
        eval_expert = apply_merged_backbone(expert, merged_model_eval, arch)
        eval_expert.to(device)
        eval_expert.eval()
        
        loader = test_loaders[task_name]
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if corruption is not None:
                    inputs = apply_corruption(inputs, corruption)
                outputs = eval_expert(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracies[task_name] = 100.0 * correct / total
    return accuracies

def calibrate_bn_fast(model, stacked_tensor, device='cuda'):
    model.to(device)
    model.train()
    
    # Reset all running stats in BatchNorm layers
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None # Cumulative average estimation
            
    # Run forward passes in batches of 32
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(stacked_tensor), batch_size):
            batch_tensor = stacked_tensor[i:i+batch_size].to(device)
            _ = model(batch_tensor)
            
    model.eval()

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    architectures = ['resnet18', 'mlp']
    datasets_list = ['mnist', 'fmnist', 'cifar10']
    
    # Check if checkpoint files exist
    for arch in architectures:
        for ds in datasets_list:
            path = f"./checkpoints/{arch}_{ds}.pt"
            if not os.path.exists(path):
                print(f"ERROR: Missing expert checkpoint {path}. Please run train_experts.py first.")
                return

    # Stable subsets of 1000 test samples per task
    print("Preparing stable test subsets (1000 samples per task)...")
    test_loaders = {}
    test_datasets = {}
    for ds in datasets_list:
        _, test_loader = get_dataloaders(ds, batch_size=256)
        dataset = test_loader.dataset
        subset_indices = list(range(min(1000, len(dataset))))
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=256, shuffle=False, num_workers=0)
        test_loaders[ds] = subset_loader
        test_datasets[ds] = subset_dataset

    # Pre-sample and pre-stack calibration tensors for each seed and sample size to optimize DE-BN
    print("Pre-sampling and caching calibration datasets for fast DE-BN execution...")
    pre_sampled_tensors = {}
    for cal_samples in [16, 64]:
        for cal_seed in [42, 43, 44]:
            np.random.seed(cal_seed)
            random.seed(cal_seed)
            
            calib_samples = []
            for dataset in test_datasets.values():
                indices = np.random.choice(len(dataset), min(cal_samples, len(dataset)), replace=False)
                for idx in indices:
                    img, _ = dataset[idx]
                    calib_samples.append(img)
            
            random.shuffle(calib_samples)
            # Stack into a single tensor on the CPU
            stacked_tensor = torch.stack(calib_samples)
            pre_sampled_tensors[(cal_samples, cal_seed)] = stacked_tensor

    # Reset seed to 42 for main script consistency
    set_seed(42)

    results = {}

    for arch in architectures:
        print(f"\n======================================")
        print(f"Evaluating Architecture: {arch}")
        print(f"======================================")
        results[arch] = {}
        
        # Load progenitor and experts
        progenitor = get_model(arch)
        progenitor.load_state_dict(torch.load(f'./checkpoints/{arch}_progenitor.pt', map_location='cpu'))
        
        experts_dict_orig = {}
        for ds in datasets_list:
            exp = get_model(arch)
            exp.load_state_dict(torch.load(f'./checkpoints/{arch}_{ds}.pt', map_location='cpu'))
            experts_dict_orig[ds] = exp
            
        experts_list = [experts_dict_orig[ds] for ds in datasets_list]
        experts_dict = copy.deepcopy(experts_dict_orig)
        
        # 1. Oracle Baselines
        print("\n--- 1. Evaluating Oracles ---")
        oracle_accs = {}
        for ds, exp in experts_dict.items():
            acc = evaluate_model(exp, test_loaders[ds], device=device)
            oracle_accs[ds] = acc
            print(f"Oracle {ds}: {acc:.2f}%")
        results[arch]['oracles'] = oracle_accs
        results[arch]['oracles_avg'] = np.mean(list(oracle_accs.values()))
        
        # 2. Weight Averaging Baseline
        print("\n--- 2. Evaluating Weight Averaging ---")
        merged_wa = merge_wa(progenitor, experts_list)
        wa_accs = evaluate_merged_model_all_tasks(merged_wa, experts_dict, test_loaders, arch, device)
        results[arch]['wa'] = wa_accs
        results[arch]['wa_avg'] = np.mean(list(wa_accs.values()))
        print(f"WA Accuracies: {wa_accs} | Avg: {results[arch]['wa_avg']:.2f}%")
        
        # 3. Tuned Task Arithmetic Baseline
        print("\n--- 3. Evaluating Tuned Task Arithmetic ---")
        best_lambda = 0.0
        best_ta_avg = 0.0
        best_ta_accs = {}
        ta_sweep = {}
        for lam in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]:
            merged_ta = merge_ta(progenitor, experts_list, scaling_factor=lam)
            ta_accs = evaluate_merged_model_all_tasks(merged_ta, experts_dict, test_loaders, arch, device)
            ta_avg = np.mean(list(ta_accs.values()))
            ta_sweep[f"{lam:.2f}"] = ta_avg
            if ta_avg > best_ta_avg:
                best_ta_avg = ta_avg
                best_lambda = lam
                best_ta_accs = ta_accs
        results[arch]['ta_best_lambda'] = best_lambda
        results[arch]['ta_best'] = best_ta_accs
        results[arch]['ta_best_avg'] = best_ta_avg
        results[arch]['ta_sweep'] = ta_sweep
        print(f"Best TA Scaling Lambda: {best_lambda:.2f} | Accuracies: {best_ta_accs} | Avg: {best_ta_avg:.2f}%")
        
        # 4. Standard WCPR Baseline
        print("\n--- 4. Evaluating Standard WCPR ---")
        merged_wcpr = merge_wcpr(progenitor, experts_list)
        wcpr_accs = evaluate_merged_model_all_tasks(merged_wcpr, experts_dict, test_loaders, arch, device)
        results[arch]['wcpr'] = wcpr_accs
        results[arch]['wcpr_avg'] = np.mean(list(wcpr_accs.values()))
        print(f"Standard WCPR Accuracies: {wcpr_accs} | Avg: {results[arch]['wcpr_avg']:.2f}%")
        
        # 5. Sparsification Baselines (TIES and DARE)
        print("\n--- 5. Evaluating TIES and DARE ---")
        results[arch]['ties'] = {}
        results[arch]['dare'] = {}
        sparsity_ratios = [0.2, 0.4, 0.6, 0.8]
        for sr in sparsity_ratios:
            # TIES
            merged_ties = sparsify_and_merge_task_vectors(progenitor, experts_list, method='ties', sparsity_ratio=sr)
            ties_accs = evaluate_merged_model_all_tasks(merged_ties, experts_dict, test_loaders, arch, device)
            avg_ties = np.mean(list(ties_accs.values()))
            results[arch]['ties'][f"{sr:.1f}"] = {'accs': ties_accs, 'avg': avg_ties}
            
            # DARE
            merged_dare = sparsify_and_merge_task_vectors(progenitor, experts_list, method='dare', sparsity_ratio=sr)
            dare_accs = evaluate_merged_model_all_tasks(merged_dare, experts_dict, test_loaders, arch, device)
            avg_dare = np.mean(list(dare_accs.values()))
            results[arch]['dare'][f"{sr:.1f}"] = {'accs': dare_accs, 'avg': avg_dare}
            print(f"Sparsity: {sr:.1f} | TIES Avg: {avg_ties:.2f}% | DARE Avg: {avg_dare:.2f}%")
            
        # 6. Proposed SC-WCPR Method & Ablations
        print("\n--- 6. Evaluating Proposed SC-WCPR ---")
        results[arch]['sc_wcpr'] = {}
        methods = ['ties', 'dare']
        compensations = ['none', 'sqrt', 'linear', 'inv_sqrt', 'inv_linear', 'dare']
        
        for method in methods:
            results[arch]['sc_wcpr'][method] = {}
            for sr in sparsity_ratios:
                results[arch]['sc_wcpr'][method][f"{sr:.1f}"] = {}
                for comp in compensations:
                    merged_sc = merge_sc_wcpr(progenitor, experts_list, method=method, sparsity_ratio=sr, compensation=comp)
                    sc_accs = evaluate_merged_model_all_tasks(merged_sc, experts_dict, test_loaders, arch, device)
                    avg_sc = np.mean(list(sc_accs.values()))
                    results[arch]['sc_wcpr'][method][f"{sr:.1f}"][comp] = {'accs': sc_accs, 'avg': avg_sc}
                    print(f"SC-WCPR ({method.upper()}) | Sparsity: {sr:.1f} | Comp: {comp} | Avg Acc: {avg_sc:.2f}%")

        # 7. Robustness, Quantization, and Calibration sweeps (ResNet-18 only)
        if arch.lower() == 'resnet18':
            print("\n--- 7. Quantization and Calibration Analysis (ResNet-18) ---")
            results[arch]['quantization'] = {}
            
            # Select key models to evaluate under PTQ and Corruptions:
            sr = 0.4
            models_to_quantize = {
                'wa': merge_wa(progenitor, experts_list),
                'ta': merge_ta(progenitor, experts_list, scaling_factor=best_lambda),
                'ties': sparsify_and_merge_task_vectors(progenitor, experts_list, method='ties', sparsity_ratio=sr),
                'dare': sparsify_and_merge_task_vectors(progenitor, experts_list, method='dare', sparsity_ratio=sr),
                'wcpr': merge_wcpr(progenitor, experts_list),
            }
            # Add only the mathematically optimal SC-WCPR variants to keep execution fast:
            # - 'dare': optimal without DE-BN
            # - 'sqrt': optimal with DE-BN
            for comp in ['dare', 'sqrt']:
                models_to_quantize[f'sc_wcpr_ties_{comp}'] = merge_sc_wcpr(progenitor, experts_list, method='ties', sparsity_ratio=sr, compensation=comp)
                models_to_quantize[f'sc_wcpr_dare_{comp}'] = merge_sc_wcpr(progenitor, experts_list, method='dare', sparsity_ratio=sr, compensation=comp)
            
            for m_name, base_model in models_to_quantize.items():
                results[arch]['quantization'][m_name] = {}
                print(f"Quantization robustness sweep for {m_name}...")
                
                for num_bits in [None, 8, 4]:
                    nb_str = "FP32" if num_bits is None else f"INT{num_bits}"
                    results[arch]['quantization'][m_name][nb_str] = {}
                    
                    for per_channel in [False, True]:
                        if num_bits is None and per_channel:
                            continue # Skip FP32 per-channel redundantly
                        
                        pc_str = "channel" if per_channel else "tensor"
                        results[arch]['quantization'][m_name][nb_str][pc_str] = {}
                        
                        # Apply quantization
                        q_model = quantize_model(base_model, num_bits, per_channel)
                        
                        # Evaluate under Clean, Noise, Blur
                        for corr in [None, 'noise', 'blur']:
                            corr_str = corr if corr is not None else "clean"
                            
                            # Clean evaluation (No DE-BN)
                            accs = evaluate_merged_model_all_tasks(q_model, experts_dict, test_loaders, arch, device, corruption=corr)
                            avg_acc = np.mean(list(accs.values()))
                            results[arch]['quantization'][m_name][nb_str][pc_str][corr_str] = {
                                'no_debn': {'accs': accs, 'avg': avg_acc}
                            }
                            
                            # Run DE-BN calibration and re-evaluate over 3 seeds using cached fast tensors!
                            for cal_samples in [16, 64]:
                                seed_accs_list = []
                                seed_task_accs = {ds: [] for ds in datasets_list}
                                for cal_seed in [42, 43, 44]:
                                    calibrated_model = copy.deepcopy(q_model)
                                    # Fetch pre-sampled tensor and run fast calibration
                                    stacked_tensor = pre_sampled_tensors[(cal_samples, cal_seed)]
                                    calibrate_bn_fast(calibrated_model, stacked_tensor, device=device)
                                    cal_accs = evaluate_merged_model_all_tasks(calibrated_model, experts_dict, test_loaders, arch, device, corruption=corr)
                                    seed_accs_list.append(np.mean(list(cal_accs.values())))
                                    for ds in datasets_list:
                                        seed_task_accs[ds].append(cal_accs[ds])
                                        
                                    del calibrated_model
                                    torch.cuda.empty_cache()
                                        
                                avg_cal_acc = np.mean(seed_accs_list)
                                std_cal_acc = np.std(seed_accs_list)
                                cal_accs_mean = {ds: np.mean(seed_task_accs[ds]) for ds in datasets_list}
                                results[arch]['quantization'][m_name][nb_str][pc_str][corr_str][f"debn_{cal_samples}"] = {
                                    'accs': cal_accs_mean, 'avg': avg_cal_acc, 'std': std_cal_acc
                                }

    # Save results to JSON
    results_path = "./results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n======================================")
    print(f"All experiments finished! Saved results to {results_path}")
    print(f"======================================")

if __name__ == '__main__':
    main()
