import os
import json
import numpy as np
import torch
from run_experiments import (
    MultiTaskModel,
    get_dataloaders,
    evaluate_model,
    merge_models,
    collect_expert_statistics,
    calibrate_merged_model,
    calibrate_tc_merged_model,
    apply_zosf_calibration,
    remove_hooks
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataloaders
    _, cal_loaders, test_loaders = get_dataloaders()

    # Wrap test loaders for fast CPU execution if GPU is not available
    if device == 'cpu':
        print("GPU not available. Subsetting test loaders to 500 samples for ultra-fast CPU sweep execution.")
        from torch.utils.data import Subset, DataLoader
        for task in list(test_loaders.keys()):
            original_dataset = test_loaders[task].dataset
            subset_indices = list(range(min(500, len(original_dataset))))
            sub_dataset = Subset(original_dataset, subset_indices)
            test_loaders[task] = DataLoader(sub_dataset, batch_size=128, shuffle=False)

    # Load experts
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = []
    for task in tasks:
        ckpt_path = f"{task}_expert.pth"
        model = MultiTaskModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        expert_models.append(model)

    # Collect expert statistics
    print("Collecting expert profiles...")
    expert_profiles = collect_expert_statistics(expert_models, cal_loaders, device)

    # 1. Sweep lambda_val for Task Arithmetic (TA)
    print("\n=== Sweeping lambda_val for Task Arithmetic ===")
    lambda_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    ta_sweep_results = {}

    for l_val in lambda_vals:
        print(f"Evaluating lambda = {l_val}...")
        merged_model = merge_models(expert_models, method='TA', lambda_val=l_val).to(device)
        
        # Uncalibrated
        uncal_accs = {}
        for task in tasks:
            uncal_accs[task] = evaluate_model(merged_model, test_loaders[task], task, device)
        uncal_avg = np.mean(list(uncal_accs.values()))

        # Global ZOSF (using default gamma_max = 5.0)
        cal_filters = calibrate_merged_model(merged_model, expert_profiles, cal_loaders, device, gamma_max=5.0)
        zosf_handles = apply_zosf_calibration(merged_model, cal_filters)
        zosf_accs = {}
        for task in tasks:
            zosf_accs[task] = evaluate_model(merged_model, test_loaders[task], task, device)
        zosf_avg = np.mean(list(zosf_accs.values()))
        remove_hooks(zosf_handles)

        # TC-ZOSF (using default gamma_max = 5.0)
        cal_filters_tc = calibrate_tc_merged_model(merged_model, expert_profiles, cal_loaders, device, gamma_max=5.0)
        tc_zosf_accs = {}
        for task in tasks:
            tc_zosf_handles = apply_zosf_calibration(merged_model, cal_filters_tc[task])
            tc_zosf_accs[task] = evaluate_model(merged_model, test_loaders[task], task, device)
            remove_hooks(tc_zosf_handles)
        tc_zosf_avg = np.mean(list(tc_zosf_accs.values()))

        ta_sweep_results[str(l_val)] = {
            'Uncalibrated': {**uncal_accs, 'Average': uncal_avg},
            'ZOSF': {**zosf_accs, 'Average': zosf_avg},
            'TC-ZOSF': {**tc_zosf_accs, 'Average': tc_zosf_avg}
        }
        print(f"  Uncalibrated Avg: {uncal_avg:.2f}% | ZOSF Avg: {zosf_avg:.2f}% | TC-ZOSF Avg: {tc_zosf_avg:.2f}%")

    # 2. Sweep gamma_max for both WA and TA (at lambda = 0.3)
    print("\n=== Sweeping gamma_max for WA and TA ===")
    gamma_max_vals = [2.0, 3.0, 5.0, 8.0, 10.0]
    gamma_sweep_results = {'WA': {}, 'TA': {}}

    for method in ['WA', 'TA']:
        print(f"\n--- Method: {method} ---")
        merged_model = merge_models(expert_models, method=method, lambda_val=0.3).to(device)

        for g_max in gamma_max_vals:
            # Global ZOSF
            cal_filters = calibrate_merged_model(merged_model, expert_profiles, cal_loaders, device, gamma_max=g_max)
            zosf_handles = apply_zosf_calibration(merged_model, cal_filters)
            zosf_accs = {}
            for task in tasks:
                zosf_accs[task] = evaluate_model(merged_model, test_loaders[task], task, device)
            zosf_avg = np.mean(list(zosf_accs.values()))
            remove_hooks(zosf_handles)

            # TC-ZOSF
            cal_filters_tc = calibrate_tc_merged_model(merged_model, expert_profiles, cal_loaders, device, gamma_max=g_max)
            tc_zosf_accs = {}
            for task in tasks:
                tc_zosf_handles = apply_zosf_calibration(merged_model, cal_filters_tc[task])
                tc_zosf_accs[task] = evaluate_model(merged_model, test_loaders[task], task, device)
                remove_hooks(tc_zosf_handles)
            tc_zosf_avg = np.mean(list(tc_zosf_accs.values()))

            gamma_sweep_results[method][str(g_max)] = {
                'ZOSF': {**zosf_accs, 'Average': zosf_avg},
                'TC-ZOSF': {**tc_zosf_accs, 'Average': tc_zosf_avg}
            }
            print(f"  gamma_max = {g_max} | ZOSF Avg: {zosf_avg:.2f}% | TC-ZOSF Avg: {tc_zosf_avg:.2f}%")

    # Save sweep results
    sweep_results = {
        'TA_Lambda_Sweep': ta_sweep_results,
        'Gamma_Max_Sweep': gamma_sweep_results
    }
    with open('sweep_results.json', 'w') as f:
        json.dump(sweep_results, f, indent=4)
    print("\nSaved sweep results to sweep_results.json.")

if __name__ == '__main__':
    main()
