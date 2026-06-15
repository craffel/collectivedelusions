import os
import sys
import torch
import numpy as np
import json
import math
from torch.utils.data import DataLoader, Subset

# Import functions from existing scripts
from run_experiments import (
    SimpleMLP, get_datasets, filter_split_dataset, set_seed, eval_model,
    merge_models_task_arithmetic, merge_models_orthomerge, merge_models_saim,
    merge_models_rimo, merge_models_rimo_pruned, train_model
)
from run_experiments_oft import train_model_ortho
from run_additional_baselines import merge_dare, merge_ties

def run_evaluation_for_seed(seed, device, train_dataset, test_dataset):
    print(f"\n==========================================")
    print(f"Running Seed {seed}...")
    print(f"==========================================")
    
    # 1. Prepare Data Loaders
    classes_t1 = [0, 1, 2, 3, 4]
    classes_t2 = [5, 6, 7, 8, 9]
    
    train_t1 = filter_split_dataset(train_dataset, classes_t1)
    train_t2 = filter_split_dataset(train_dataset, classes_t2)
    test_t1  = filter_split_dataset(test_dataset, classes_t1)
    test_t2  = filter_split_dataset(test_dataset, classes_t2)
    
    loader_train_t1 = DataLoader(train_t1, batch_size=64, shuffle=True)
    loader_train_t2 = DataLoader(train_t2, batch_size=64, shuffle=True)
    loader_test_t1  = DataLoader(test_t1, batch_size=64, shuffle=False)
    loader_test_t2  = DataLoader(test_t2, batch_size=64, shuffle=False)
    
    # Base model pretraining subset (10%)
    mix_train_indices = list(range(0, len(train_dataset), 10))
    mix_train = Subset(train_dataset, mix_train_indices)
    loader_mix = DataLoader(mix_train, batch_size=64, shuffle=True)
    
    # -----------------------------------------------------------------
    # REGIME 1: Standard Training (Non-OFT)
    # -----------------------------------------------------------------
    print(f"\n--- [Seed {seed}] Training Standard Models ---")
    set_seed(seed)
    
    base_std = SimpleMLP()
    base_std = train_model(base_std, loader_mix, epochs=2, lr=0.01, device=device, desc="Base Model Standard")
    
    expert1_std = SimpleMLP()
    expert1_std.load_state_dict(base_std.state_dict())
    expert1_std = train_model(expert1_std, loader_train_t1, epochs=3, lr=0.005, device=device, desc="Expert 1 Standard")
    
    expert2_std = SimpleMLP()
    expert2_std.load_state_dict(base_std.state_dict())
    expert2_std = train_model(expert2_std, loader_train_t2, epochs=3, lr=0.005, device=device, desc="Expert 2 Standard")
    
    # Evaluate Standard Individual Models
    acc_base_std_t1 = eval_model(base_std, loader_test_t1, device)
    acc_base_std_t2 = eval_model(base_std, loader_test_t2, device)
    acc_exp1_std_t1 = eval_model(expert1_std, loader_test_t1, device)
    acc_exp1_std_t2 = eval_model(expert1_std, loader_test_t2, device)
    acc_exp2_std_t1 = eval_model(expert2_std, loader_test_t1, device)
    acc_exp2_std_t2 = eval_model(expert2_std, loader_test_t2, device)
    
    standard_evals = {
        "Base Model": {"t1": acc_base_std_t1, "t2": acc_base_std_t2, "avg": (acc_base_std_t1 + acc_base_std_t2)/2.0},
        "Expert 1": {"t1": acc_exp1_std_t1, "t2": acc_exp1_std_t2, "avg": (acc_exp1_std_t1 + acc_exp1_std_t2)/2.0},
        "Expert 2": {"t1": acc_exp2_std_t1, "t2": acc_exp2_std_t2, "avg": (acc_exp2_std_t1 + acc_exp2_std_t2)/2.0},
    }
    
    # Evaluate Standard Merging Algorithms
    # TA lambda=0.3
    m_ta = merge_models_task_arithmetic([expert1_std, expert2_std], base_std, scaling_factor=0.3)
    standard_evals["TA lambda=0.3"] = {"t1": eval_model(m_ta, loader_test_t1, device), "t2": eval_model(m_ta, loader_test_t2, device)}
    standard_evals["TA lambda=0.3"]["avg"] = (standard_evals["TA lambda=0.3"]["t1"] + standard_evals["TA lambda=0.3"]["t2"])/2.0
    
    # DARE p=0.2
    m_dare = merge_dare([expert1_std, expert2_std], base_std, drop_rate=0.2, scaling_factor=0.5)
    standard_evals["DARE p=0.2"] = {"t1": eval_model(m_dare, loader_test_t1, device), "t2": eval_model(m_dare, loader_test_t2, device)}
    standard_evals["DARE p=0.2"]["avg"] = (standard_evals["DARE p=0.2"]["t1"] + standard_evals["DARE p=0.2"]["t2"])/2.0
    
    # TIES k=0.8
    m_ties = merge_ties([expert1_std, expert2_std], base_std, keep_rate=0.8, scaling_factor=0.5)
    standard_evals["TIES k=0.8"] = {"t1": eval_model(m_ties, loader_test_t1, device), "t2": eval_model(m_ties, loader_test_t2, device)}
    standard_evals["TIES k=0.8"]["avg"] = (standard_evals["TIES k=0.8"]["t1"] + standard_evals["TIES k=0.8"]["t2"])/2.0
    
    # OrthoMerge res_scale=1.0
    m_om = merge_models_orthomerge([expert1_std, expert2_std], base_std, residual_scale=1.0)
    standard_evals["OrthoMerge res_scale=1.0"] = {"t1": eval_model(m_om, loader_test_t1, device), "t2": eval_model(m_om, loader_test_t2, device)}
    standard_evals["OrthoMerge res_scale=1.0"]["avg"] = (standard_evals["OrthoMerge res_scale=1.0"]["t1"] + standard_evals["OrthoMerge res_scale=1.0"]["t2"])/2.0
    
    # SAIM t=1.5
    m_saim = merge_models_saim([expert1_std, expert2_std], base_std, t=1.5, scaling_factor=0.5)
    standard_evals["SAIM t=1.5"] = {"t1": eval_model(m_saim, loader_test_t1, device), "t2": eval_model(m_saim, loader_test_t2, device)}
    standard_evals["SAIM t=1.5"]["avg"] = (standard_evals["SAIM t=1.5"]["t1"] + standard_evals["SAIM t=1.5"]["t2"])/2.0
    
    # RIMO t=1.0, res_scale=1.0
    m_rimo_1 = merge_models_rimo([expert1_std, expert2_std], base_std, t=1.0, residual_scale=1.0)
    standard_evals["RIMO t=1.0 res_scale=1.0"] = {"t1": eval_model(m_rimo_1, loader_test_t1, device), "t2": eval_model(m_rimo_1, loader_test_t2, device)}
    standard_evals["RIMO t=1.0 res_scale=1.0"]["avg"] = (standard_evals["RIMO t=1.0 res_scale=1.0"]["t1"] + standard_evals["RIMO t=1.0 res_scale=1.0"]["t2"])/2.0
    
    # RIMO t=1.5, res_scale=1.0
    m_rimo_1_5 = merge_models_rimo([expert1_std, expert2_std], base_std, t=1.5, residual_scale=1.0)
    standard_evals["RIMO t=1.5 res_scale=1.0"] = {"t1": eval_model(m_rimo_1_5, loader_test_t1, device), "t2": eval_model(m_rimo_1_5, loader_test_t2, device)}
    standard_evals["RIMO t=1.5 res_scale=1.0"]["avg"] = (standard_evals["RIMO t=1.5 res_scale=1.0"]["t1"] + standard_evals["RIMO t=1.5 res_scale=1.0"]["t2"])/2.0
    
    # RIMO t=4.0, res_scale=0.2
    m_rimo_4 = merge_models_rimo([expert1_std, expert2_std], base_std, t=4.0, residual_scale=0.2)
    standard_evals["RIMO t=4.0 res_scale=0.2"] = {"t1": eval_model(m_rimo_4, loader_test_t1, device), "t2": eval_model(m_rimo_4, loader_test_t2, device)}
    standard_evals["RIMO t=4.0 res_scale=0.2"]["avg"] = (standard_evals["RIMO t=4.0 res_scale=0.2"]["t1"] + standard_evals["RIMO t=4.0 res_scale=0.2"]["t2"])/2.0
    
    # RIMO-Pruned keep=0.2, res_scale=0.2
    m_rimo_p = merge_models_rimo_pruned([expert1_std, expert2_std], base_std, keep_ratio=0.2, residual_scale=0.2)
    standard_evals["RIMO-Pruned keep=0.2 res_scale=0.2"] = {"t1": eval_model(m_rimo_p, loader_test_t1, device), "t2": eval_model(m_rimo_p, loader_test_t2, device)}
    standard_evals["RIMO-Pruned keep=0.2 res_scale=0.2"]["avg"] = (standard_evals["RIMO-Pruned keep=0.2 res_scale=0.2"]["t1"] + standard_evals["RIMO-Pruned keep=0.2 res_scale=0.2"]["t2"])/2.0
    
    # -----------------------------------------------------------------
    # REGIME 2: Orthogonal Regularization Training (OFT-like, lambda=2.0)
    # -----------------------------------------------------------------
    print(f"\n--- [Seed {seed}] Training Orthogonalized Models ---")
    set_seed(seed)
    ortho_lambda = 2.0
    
    base_ortho = SimpleMLP()
    base_ortho = train_model_ortho(base_ortho, loader_mix, epochs=2, lr=0.01, device=device, desc="Base Model Ortho", ortho_lambda=ortho_lambda)
    
    expert1_ortho = SimpleMLP()
    expert1_ortho.load_state_dict(base_ortho.state_dict())
    expert1_ortho = train_model_ortho(expert1_ortho, loader_train_t1, epochs=3, lr=0.005, device=device, desc="Expert 1 Ortho", ortho_lambda=ortho_lambda)
    
    expert2_ortho = SimpleMLP()
    expert2_ortho.load_state_dict(base_ortho.state_dict())
    expert2_ortho = train_model_ortho(expert2_ortho, loader_train_t2, epochs=3, lr=0.005, device=device, desc="Expert 2 Ortho", ortho_lambda=ortho_lambda)
    
    # Evaluate Ortho Individual Models
    acc_base_ortho_t1 = eval_model(base_ortho, loader_test_t1, device)
    acc_base_ortho_t2 = eval_model(base_ortho, loader_test_t2, device)
    acc_exp1_ortho_t1 = eval_model(expert1_ortho, loader_test_t1, device)
    acc_exp1_ortho_t2 = eval_model(expert1_ortho, loader_test_t2, device)
    acc_exp2_ortho_t1 = eval_model(expert2_ortho, loader_test_t1, device)
    acc_exp2_ortho_t2 = eval_model(expert2_ortho, loader_test_t2, device)
    
    ortho_evals = {
        "Base Model": {"t1": acc_base_ortho_t1, "t2": acc_base_ortho_t2, "avg": (acc_base_ortho_t1 + acc_base_ortho_t2)/2.0},
        "Expert 1": {"t1": acc_exp1_ortho_t1, "t2": acc_exp1_ortho_t2, "avg": (acc_exp1_ortho_t1 + acc_exp1_ortho_t2)/2.0},
        "Expert 2": {"t1": acc_exp2_ortho_t1, "t2": acc_exp2_ortho_t2, "avg": (acc_exp2_ortho_t1 + acc_exp2_ortho_t2)/2.0},
    }
    
    # Evaluate Ortho Merging Algorithms
    # TA lambda=1.0
    m_ta_o = merge_models_task_arithmetic([expert1_ortho, expert2_ortho], base_ortho, scaling_factor=1.0)
    ortho_evals["TA lambda=1.0"] = {"t1": eval_model(m_ta_o, loader_test_t1, device), "t2": eval_model(m_ta_o, loader_test_t2, device)}
    ortho_evals["TA lambda=1.0"]["avg"] = (ortho_evals["TA lambda=1.0"]["t1"] + ortho_evals["TA lambda=1.0"]["t2"])/2.0
    
    # DARE p=0.1
    m_dare_o = merge_dare([expert1_ortho, expert2_ortho], base_ortho, drop_rate=0.1, scaling_factor=1.0)
    ortho_evals["DARE p=0.1"] = {"t1": eval_model(m_dare_o, loader_test_t1, device), "t2": eval_model(m_dare_o, loader_test_t2, device)}
    ortho_evals["DARE p=0.1"]["avg"] = (ortho_evals["DARE p=0.1"]["t1"] + ortho_evals["DARE p=0.1"]["t2"])/2.0
    
    # TIES k=0.8
    m_ties_o = merge_ties([expert1_ortho, expert2_ortho], base_ortho, keep_rate=0.8, scaling_factor=1.0)
    ortho_evals["TIES k=0.8"] = {"t1": eval_model(m_ties_o, loader_test_t1, device), "t2": eval_model(m_ties_o, loader_test_t2, device)}
    ortho_evals["TIES k=0.8"]["avg"] = (ortho_evals["TIES k=0.8"]["t1"] + ortho_evals["TIES k=0.8"]["t2"])/2.0
    
    # OrthoMerge res_scale=1.0
    m_om_o = merge_models_orthomerge([expert1_ortho, expert2_ortho], base_ortho, residual_scale=1.0)
    ortho_evals["OrthoMerge res_scale=1.0"] = {"t1": eval_model(m_om_o, loader_test_t1, device), "t2": eval_model(m_om_o, loader_test_t2, device)}
    ortho_evals["OrthoMerge res_scale=1.0"]["avg"] = (ortho_evals["OrthoMerge res_scale=1.0"]["t1"] + ortho_evals["OrthoMerge res_scale=1.0"]["t2"])/2.0
    
    # SAIM t=1.0
    m_saim_o = merge_models_saim([expert1_ortho, expert2_ortho], base_ortho, t=1.0, scaling_factor=1.0)
    ortho_evals["SAIM t=1.0"] = {"t1": eval_model(m_saim_o, loader_test_t1, device), "t2": eval_model(m_saim_o, loader_test_t2, device)}
    ortho_evals["SAIM t=1.0"]["avg"] = (ortho_evals["SAIM t=1.0"]["t1"] + ortho_evals["SAIM t=1.0"]["t2"])/2.0
    
    # RIMO t=1.0, res_scale=1.0
    m_rimo_1_o = merge_models_rimo([expert1_ortho, expert2_ortho], base_ortho, t=1.0, residual_scale=1.0)
    ortho_evals["RIMO t=1.0 res_scale=1.0"] = {"t1": eval_model(m_rimo_1_o, loader_test_t1, device), "t2": eval_model(m_rimo_1_o, loader_test_t2, device)}
    ortho_evals["RIMO t=1.0 res_scale=1.0"]["avg"] = (ortho_evals["RIMO t=1.0 res_scale=1.0"]["t1"] + ortho_evals["RIMO t=1.0 res_scale=1.0"]["t2"])/2.0
    
    # RIMO t=1.5, res_scale=1.0
    m_rimo_1_5_o = merge_models_rimo([expert1_ortho, expert2_ortho], base_ortho, t=1.5, residual_scale=1.0)
    ortho_evals["RIMO t=1.5 res_scale=1.0"] = {"t1": eval_model(m_rimo_1_5_o, loader_test_t1, device), "t2": eval_model(m_rimo_1_5_o, loader_test_t2, device)}
    ortho_evals["RIMO t=1.5 res_scale=1.0"]["avg"] = (ortho_evals["RIMO t=1.5 res_scale=1.0"]["t1"] + ortho_evals["RIMO t=1.5 res_scale=1.0"]["t2"])/2.0
    
    # RIMO t=4.0, res_scale=0.2
    m_rimo_4_o = merge_models_rimo([expert1_ortho, expert2_ortho], base_ortho, t=4.0, residual_scale=0.2)
    ortho_evals["RIMO t=4.0 res_scale=0.2"] = {"t1": eval_model(m_rimo_4_o, loader_test_t1, device), "t2": eval_model(m_rimo_4_o, loader_test_t2, device)}
    ortho_evals["RIMO t=4.0 res_scale=0.2"]["avg"] = (ortho_evals["RIMO t=4.0 res_scale=0.2"]["t1"] + ortho_evals["RIMO t=4.0 res_scale=0.2"]["t2"])/2.0
    
    # RIMO-Pruned keep=0.1, res_scale=1.0
    m_rimo_p_o = merge_models_rimo_pruned([expert1_ortho, expert2_ortho], base_ortho, keep_ratio=0.1, residual_scale=1.0)
    ortho_evals["RIMO-Pruned keep=0.1 res_scale=1.0"] = {"t1": eval_model(m_rimo_p_o, loader_test_t1, device), "t2": eval_model(m_rimo_p_o, loader_test_t2, device)}
    ortho_evals["RIMO-Pruned keep=0.1 res_scale=1.0"]["avg"] = (ortho_evals["RIMO-Pruned keep=0.1 res_scale=1.0"]["t1"] + ortho_evals["RIMO-Pruned keep=0.1 res_scale=1.0"]["t2"])/2.0
    
    return standard_evals, ortho_evals

def main():
    seeds = [42, 100, 2026]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Multi-seed Robustness Check over seeds {seeds} on {device}...")
    
    # Load dataset
    train_dataset, test_dataset, _ = get_datasets(use_synthetic=False)
    
    all_std = {k: [] for k in [
        "Base Model", "Expert 1", "Expert 2", "TA lambda=0.3", "DARE p=0.2", "TIES k=0.8",
        "OrthoMerge res_scale=1.0", "SAIM t=1.5", "RIMO t=1.0 res_scale=1.0", "RIMO t=1.5 res_scale=1.0",
        "RIMO t=4.0 res_scale=0.2", "RIMO-Pruned keep=0.2 res_scale=0.2"
    ]}
    
    all_ortho = {k: [] for k in [
        "Base Model", "Expert 1", "Expert 2", "TA lambda=1.0", "DARE p=0.1", "TIES k=0.8",
        "OrthoMerge res_scale=1.0", "SAIM t=1.0", "RIMO t=1.0 res_scale=1.0", "RIMO t=1.5 res_scale=1.0",
        "RIMO t=4.0 res_scale=0.2", "RIMO-Pruned keep=0.1 res_scale=1.0"
    ]}
    
    for seed in seeds:
        std_res, ortho_res = run_evaluation_for_seed(seed, device, train_dataset, test_dataset)
        for k in all_std.keys():
            all_std[k].append(std_res[k]["avg"])
        for k in all_ortho.keys():
            all_ortho[k].append(ortho_res[k]["avg"])
            
    # Calculate stats
    print("\n==========================================")
    print("FINAL AGGREGATED STATS (MEAN ± STD)")
    print("==========================================")
    
    stats_std = {}
    print("\n--- Experiment 1 (Standard Training) ---")
    for k, vals in all_std.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        stats_std[k] = {"mean": mean_val, "std": std_val, "values": vals}
        print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
        
    stats_ortho = {}
    print("\n--- Experiment 2 (Orthogonal Regularization) ---")
    for k, vals in all_ortho.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        stats_ortho[k] = {"mean": mean_val, "std": std_val, "values": vals}
        print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
        
    # Save raw stats to file
    os.makedirs("results", exist_ok=True)
    with open("results/multi_seed_stats.json", "w") as f:
        json.dump({"standard": stats_std, "orthogonal": stats_ortho}, f, indent=2)
        
    print("\nSaved multi-seed results to results/multi_seed_stats.json")

if __name__ == "__main__":
    main()
