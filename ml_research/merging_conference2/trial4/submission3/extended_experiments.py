import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import necessary modules and functions from experiment.py
from experiment import (
    set_seed,
    get_datasets,
    evaluate_model,
    get_bn_modules,
    merge_models,
    get_calibration_sets,
    run_calibration_and_fusion,
    benchmark_inference_speed,
    resnet18
)

def benchmark_compiled_inference_speed(model, device, num_runs=50):
    model.eval()
    dummy_x = torch.randn(128, 3, 32, 32, device=device)
    
    # Measure compilation time (first forward pass triggers compilation in PyTorch Inductor)
    torch.cuda.synchronize()
    start_comp = time.time()
    with torch.no_grad():
        _ = model(dummy_x)
    torch.cuda.synchronize()
    compilation_time = (time.time() - start_comp) * 1000.0 # ms
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_x)
            
    # Measure steady-state latency
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_x)
    torch.cuda.synchronize()
    avg_latency = (time.time() - start_time) / num_runs * 1000.0  # ms per batch
    return avg_latency, compilation_time

def main():
    print("=== STARTING EXTENDED PRAGMATIST EXPERIMENTS ===")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets = get_datasets()
    task_names = ['mnist', 'fmnist', 'cifar10']
    
    # Load cached expert models
    expert_models = []
    expert_accs = {}
    
    from torchvision.models import ResNet18_Weights
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    
    for task in task_names:
        ckpt_path = f"./checkpoints/expert_{task}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint not found at {ckpt_path}. Please run experiment.py first.")
        
        print(f"Loading cached expert for {task.upper()}...")
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        test_ds = datasets[task][1]
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        acc = evaluate_model(model, test_loader, device)
        expert_models.append(model)
        expert_accs[task] = acc
        print(f"Expert {task.upper()} Acc: {acc:.2f}%")
        
    oracle_avg = np.mean(list(expert_accs.values()))
    print(f"Oracle (Single Experts) Average: {oracle_avg:.2f}%")
    
    # Create calibration dictionaries
    train_datasets_dict = {task: datasets[task][0] for task in task_names}
    
    # ==========================================================
    # EXPERIMENT 1: Calibration Sample Size (N) Sweep
    # ==========================================================
    print("\n--- Running Experiment 1: N Sweep ---")
    N_list = [16, 32, 64, 128, 256]
    merging_modes = ["WA", "TA"]
    calibration_modes = ["SP-TAAC", "TAAC"]
    
    n_sweep_results = {}
    
    for m_mode in merging_modes:
        n_sweep_results[m_mode] = {}
        merged_base = merge_models(base_model, expert_models, mode=m_mode, lambda_ta=0.3)
        merged_base = merged_base.to(device)
        
        # Uncalibrated baseline (constant across N)
        uncal_accs = {}
        for idx, task in enumerate(task_names):
            _, test_ds = datasets[task]
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            merged_base.fc = expert_models[idx].fc.to(device)
            acc = evaluate_model(merged_base, test_loader, device)
            uncal_accs[task] = acc
        uncal_avg = np.mean(list(uncal_accs.values()))
        
        n_sweep_results[m_mode]["uncalibrated"] = {
            "tasks": uncal_accs,
            "average": uncal_avg
        }
        
        for cal_mode in calibration_modes:
            n_sweep_results[m_mode][cal_mode] = {
                "N": [],
                "hooked_avg": [],
                "fused_avg": [],
                "exact_match": []
            }
            
            for N in N_list:
                print(f"Evaluating {m_mode} + {cal_mode} with N = {N}")
                calib_sets = get_calibration_sets(train_datasets_dict, N=N)
                
                # Run calibration and fusion
                fused, hooked, hooks = run_calibration_and_fusion(
                    base_model, expert_models, merged_base, calib_sets, N=N, mode=cal_mode, device=device
                )
                
                # Evaluate hooked model
                hooked_accs = {}
                for idx, task in enumerate(task_names):
                    _, test_ds = datasets[task]
                    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                    hooked.fc = expert_models[idx].fc.to(device)
                    acc = evaluate_model(hooked, test_loader, device)
                    hooked_accs[task] = acc
                hooked_avg = np.mean(list(hooked_accs.values()))
                
                # Evaluate fused model
                fused_accs = {}
                for idx, task in enumerate(task_names):
                    _, test_ds = datasets[task]
                    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                    fused.fc = expert_models[idx].fc.to(device)
                    acc = evaluate_model(fused, test_loader, device)
                    fused_accs[task] = acc
                fused_avg = np.mean(list(fused_accs.values()))
                
                # Parity verification
                diffs = [abs(fused_accs[t] - hooked_accs[t]) for t in task_names]
                max_diff = max(diffs)
                exact_match = max_diff < 1e-4
                
                print(f" N={N} | Hooked: {hooked_avg:.2f}% | Fused: {fused_avg:.2f}% | Parity Check: {exact_match}")
                
                n_sweep_results[m_mode][cal_mode]["N"].append(N)
                n_sweep_results[m_mode][cal_mode]["hooked_avg"].append(hooked_avg)
                n_sweep_results[m_mode][cal_mode]["fused_avg"].append(fused_avg)
                n_sweep_results[m_mode][cal_mode]["exact_match"].append(bool(exact_match))
                
                # Clean up hooks
                for h in hooks:
                    h.remove()
                    
    # ==========================================================
    # EXPERIMENT 2: Task Arithmetic Coefficient (lambda) Sweep
    # ==========================================================
    print("\n--- Running Experiment 2: Lambda Sweep ---")
    lambda_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    lambda_sweep_results = {}
    
    calib_sets_128 = get_calibration_sets(train_datasets_dict, N=128)
    
    for l_val in lambda_list:
        lambda_sweep_results[l_val] = {}
        merged_base = merge_models(base_model, expert_models, mode="TA", lambda_ta=l_val)
        merged_base = merged_base.to(device)
        
        # Uncalibrated TA baseline for this lambda
        uncal_accs = {}
        for idx, task in enumerate(task_names):
            _, test_ds = datasets[task]
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            merged_base.fc = expert_models[idx].fc.to(device)
            acc = evaluate_model(merged_base, test_loader, device)
            uncal_accs[task] = acc
        uncal_avg = np.mean(list(uncal_accs.values()))
        
        lambda_sweep_results[l_val]["uncalibrated"] = {
            "tasks": uncal_accs,
            "average": uncal_avg
        }
        
        for cal_mode in calibration_modes:
            print(f"Evaluating TA (lambda={l_val}) + {cal_mode}")
            fused, hooked, hooks = run_calibration_and_fusion(
                base_model, expert_models, merged_base, calib_sets_128, N=128, mode=cal_mode, device=device
            )
            
            # Evaluate fused
            fused_accs = {}
            for idx, task in enumerate(task_names):
                _, test_ds = datasets[task]
                test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                fused.fc = expert_models[idx].fc.to(device)
                acc = evaluate_model(fused, test_loader, device)
                fused_accs[task] = acc
            fused_avg = np.mean(list(fused_accs.values()))
            
            lambda_sweep_results[l_val][cal_mode] = {
                "tasks": fused_accs,
                "average": fused_avg
            }
            
            for h in hooks:
                h.remove()
                
    # ==========================================================
    # EXPERIMENT 3: torch.compile Benchmarking
    # ==========================================================
    print("\n--- Running Experiment 3: torch.compile Benchmarking ---")
    compilation_results = {}
    
    merged_base = merge_models(base_model, expert_models, mode="WA", lambda_ta=0.3)
    merged_base = merged_base.to(device)
    
    # We will run TAAC calibration with N=128 to generate standard models
    fused, hooked, hooks = run_calibration_and_fusion(
        base_model, expert_models, merged_base, calib_sets_128, N=128, mode="TAAC", device=device
    )
    
    # Create evaluation copies
    model_uncal = copy_model(merged_base, expert_models[0].fc, device)
    model_hooked = copy_model(hooked, expert_models[0].fc, device)
    model_fused = copy_model(fused, expert_models[0].fc, device)
    
    # Measure uncompiled speeds
    print("Profiling UNCOMPILED models...")
    uncal_latency = benchmark_inference_speed(model_uncal, datasets, device)
    hooked_latency = benchmark_inference_speed(model_hooked, datasets, device)
    fused_latency = benchmark_inference_speed(model_fused, datasets, device)
    
    print(f"Uncompiled Uncalibrated: {uncal_latency:.3f} ms/batch")
    print(f"Uncompiled Hooked (TAAC): {hooked_latency:.3f} ms/batch")
    print(f"Uncompiled Fused (TAAC): {fused_latency:.3f} ms/batch")
    
    # Compile models
    print("Compiling models with torch.compile...")
    compiled_uncal = torch.compile(model_uncal, backend="inductor")
    compiled_hooked = torch.compile(model_hooked, backend="inductor")
    compiled_fused = torch.compile(model_fused, backend="inductor")
    
    print("Profiling COMPILED models...")
    comp_uncal_latency, comp_uncal_compile_time = benchmark_compiled_inference_speed(compiled_uncal, device)
    comp_hooked_latency, comp_hooked_compile_time = benchmark_compiled_inference_speed(compiled_hooked, device)
    comp_fused_latency, comp_fused_compile_time = benchmark_compiled_inference_speed(compiled_fused, device)
    
    print(f"Compiled Uncalibrated: {comp_uncal_latency:.3f} ms/batch | Compile Time: {comp_uncal_compile_time:.1f} ms")
    print(f"Compiled Hooked (TAAC): {comp_hooked_latency:.3f} ms/batch | Compile Time: {comp_hooked_compile_time:.1f} ms")
    print(f"Compiled Fused (TAAC): {comp_fused_latency:.3f} ms/batch | Compile Time: {comp_fused_compile_time:.1f} ms")
    
    compilation_results = {
        "uncompiled": {
            "uncalibrated": uncal_latency,
            "hooked": hooked_latency,
            "fused": fused_latency
        },
        "compiled": {
            "uncalibrated": comp_uncal_latency,
            "hooked": comp_hooked_latency,
            "fused": comp_fused_latency
        },
        "compile_times": {
            "uncalibrated": comp_uncal_compile_time,
            "hooked": comp_hooked_compile_time,
            "fused": comp_fused_compile_time
        }
    }
    
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    # Save all results to a JSON file
    all_extended_results = {
        "n_sweep": n_sweep_results,
        "lambda_sweep": lambda_sweep_results,
        "compilation_bench": compilation_results
    }
    
    with open("extended_results.json", "w") as f:
        json.dump(all_extended_results, f, indent=4)
    print("\nExtended results saved to extended_results.json.")
    
    # Generate Plots
    generate_extended_plots(all_extended_results)
    
def copy_model(model, fc_head, device):
    import copy
    copied = copy.deepcopy(model)
    copied.fc = copy.deepcopy(fc_head).to(device)
    return copied.to(device)

def generate_extended_plots(results):
    print("Generating extended plots...")
    
    # Plot 1: Accuracy vs N
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    merging_modes = ["WA", "TA"]
    for idx, m_mode in enumerate(merging_modes):
        uncal_avg = results["n_sweep"][m_mode]["uncalibrated"]["average"]
        n_vals = results["n_sweep"][m_mode]["SP-TAAC"]["N"]
        
        sp_vals = results["n_sweep"][m_mode]["SP-TAAC"]["fused_avg"]
        taac_vals = results["n_sweep"][m_mode]["TAAC"]["fused_avg"]
        
        ax[idx].plot(n_vals, [uncal_avg] * len(n_vals), '--', label="Uncalibrated", color="#d95f02", linewidth=2)
        ax[idx].plot(n_vals, sp_vals, '-o', label="SP-TAAC (Ours)", color="#1b9e77", linewidth=2)
        ax[idx].plot(n_vals, taac_vals, '-s', label="TAAC (Ours)", color="#7570b3", linewidth=2)
        
        ax[idx].set_title(f"Accuracy vs. Calib Size (N) under {m_mode}")
        ax[idx].set_xlabel("Calibration Size N (per task)")
        ax[idx].set_ylabel("Average Accuracy (%)")
        ax[idx].set_xscale("log")
        ax[idx].set_xticks(n_vals)
        ax[idx].get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax[idx].set_ylim(25, 75)
        ax[idx].grid(True, linestyle=":", alpha=0.6)
        ax[idx].legend()
        
    plt.tight_layout()
    plt.savefig("accuracy_vs_n.png", dpi=300)
    plt.close()
    
    # Plot 2: Accuracy vs Lambda
    plt.figure(figsize=(7, 5))
    lambdas = sorted([float(x) for x in results["lambda_sweep"].keys()])
    uncal_vals = [results["lambda_sweep"].get(l, results["lambda_sweep"].get(str(l)))["uncalibrated"]["average"] for l in lambdas]
    sp_vals = [results["lambda_sweep"].get(l, results["lambda_sweep"].get(str(l)))["SP-TAAC"]["average"] for l in lambdas]
    taac_vals = [results["lambda_sweep"].get(l, results["lambda_sweep"].get(str(l)))["TAAC"]["average"] for l in lambdas]
    
    plt.plot(lambdas, uncal_vals, '--o', label="Uncalibrated TA", color="#d95f02", linewidth=2)
    plt.plot(lambdas, sp_vals, '-o', label="TA + SP-TAAC (Ours)", color="#1b9e77", linewidth=2)
    plt.plot(lambdas, taac_vals, '-s', label="TA + TAAC (Ours)", color="#7570b3", linewidth=2)
    
    plt.title("Task Arithmetic Sensitivity to Weight Vector Scale ($\lambda$)")
    plt.xlabel("TA Scale Coefficient $\lambda$")
    plt.ylabel("Average Accuracy (%)")
    plt.ylim(25, 75)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_lambda.png", dpi=300)
    plt.close()
    
    # Plot 3: Compiled vs Uncompiled Latency
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["Uncalibrated", "TAAC (Hooked)", "TAAC (Fused)"]
    uncomp_latencies = [
        results["compilation_bench"]["uncompiled"]["uncalibrated"],
        results["compilation_bench"]["uncompiled"]["hooked"],
        results["compilation_bench"]["uncompiled"]["fused"]
    ]
    comp_latencies = [
        results["compilation_bench"]["compiled"]["uncalibrated"],
        results["compilation_bench"]["compiled"]["hooked"],
        results["compilation_bench"]["compiled"]["fused"]
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, uncomp_latencies, width, label='Uncompiled', color='#fc8d62')
    rects2 = ax.bar(x + width/2, comp_latencies, width, label='torch.compile', color='#66c2a5')
    
    ax.set_ylabel('Inference Latency (ms/batch)')
    ax.set_title('Inference Speedup via Native Graph Compilation')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig("compiled_latency_comparison.png", dpi=300)
    plt.close()
    print("Extended plots saved successfully.")

if __name__ == "__main__":
    main()
