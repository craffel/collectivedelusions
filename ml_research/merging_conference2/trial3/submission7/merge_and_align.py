import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_datasets, get_calibration_set
from models import (
    MultiTaskResNet18, 
    merge_models_weight_averaging, 
    merge_models_task_arithmetic,
    calibrate_model_ntaac,
    get_layer_stds,
    apply_lsc_calibration,
    CalibratedMultiTaskModel
)
import argparse
import os
import json

torch.backends.cudnn.enabled = False
torch.set_num_threads(1)

def evaluate_full(model, dataloaders, device):
    """
    Evaluates the model across all three tasks.
    """
    model.to(device)
    model.eval()
    
    accuracies = {}
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    
    for task_id, task_name in task_mapping.items():
        _, test_loader = dataloaders[task_name]
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs, task_id=task_id)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracies[task_name] = 100.0 * correct / total
        
    accuracies['avg'] = sum(accuracies.values()) / len(accuracies)
    return accuracies

def train_head_sft(merged_model, task_calibration_sets, device, epochs=15, lr=1e-3):
    """
    Supervised Fine-Tuning (SFT) of classification heads only.
    """
    merged_model.to(device)
    # Freeze backbone
    for name, param in merged_model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    
    for task_id, task_name in task_mapping.items():
        head_params = merged_model.heads[task_name].parameters()
        optimizer = optim.AdamW(head_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(task_calibration_sets[task_name], batch_size=32, shuffle=True)
        
        merged_model.eval() # Keep BN statistics frozen during head tuning
        for epoch in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = merged_model(imgs, task_id=task_id)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
    # Restore parameter requires_grad defaults
    for param in merged_model.parameters():
        param.requires_grad = True

def train_head_tta(merged_model, experts, task_calibration_sets, device, epochs=15, lr=1e-3):
    """
    Unsupervised Test-Time Adaptation (TTA) of classification heads via KL-distillation from experts.
    """
    merged_model.to(device)
    # Freeze backbone
    for name, param in merged_model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    
    for task_id, task_name in task_mapping.items():
        teacher = experts[task_name].to(device)
        teacher.eval()
        
        head_params = merged_model.heads[task_name].parameters()
        optimizer = optim.AdamW(head_params, lr=lr)
        loader = DataLoader(task_calibration_sets[task_name], batch_size=32, shuffle=True)
        
        merged_model.eval() # Keep BN statistics frozen
        for epoch in range(epochs):
            for imgs, _ in loader: # Unlabeled
                imgs = imgs.to(device)
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = teacher(imgs, task_id=task_id)
                    
                student_logits = merged_model(imgs, task_id=task_id)
                loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits, dim=1)
                )
                loss.backward()
                optimizer.step()
                
    for param in merged_model.parameters():
        param.requires_grad = True

def main():
    parser = argparse.ArgumentParser(description="Model Merging & Alignment Experiments")
    parser.add_argument("--merge_method", type=str, default="wa", choices=["wa", "ta"], help="Merging method: wa (Weight Averaging) or ta (Task Arithmetic)")
    parser.add_argument("--ta_lam", type=float, default=0.3, help="Task Arithmetic coefficient")
    parser.add_argument("--rep_cal", type=str, default="none", choices=["none", "ntaac", "lsc", "tsc"], help="Representation calibration method")
    parser.add_argument("--tsc_tau", type=float, default=1.3, help="TSC threshold")
    parser.add_argument("--head_align", type=str, default="none", choices=["none", "sft", "tta"], help="Decision boundary alignment method")
    parser.add_argument("--cal_size", type=int, default=128, help="Calibration sample size N per task")
    parser.add_argument("--sft_tta_lr", type=float, default=1e-3, help="Learning rate for Head SFT/TTA")
    parser.add_argument("--epochs", type=int, default=15, help="Number of head adaptation epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation results JSON")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Run Configuration ---")
    print(f"Merge Method: {args.merge_method.upper()}")
    if args.merge_method == "ta":
        print(f"Task Arithmetic Lambda: {args.ta_lam}")
    print(f"Representation Calibration: {args.rep_cal.upper()}")
    if args.rep_cal == "tsc":
        print(f"TSC Threshold (tau): {args.tsc_tau}")
    print(f"Head Alignment: {args.head_align.upper()}")
    print(f"Calibration Size N: {args.cal_size}")
    print(f"Seed: {args.seed}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    # 1. Load experts and base model
    print("\nLoading models...")
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    experts = {}
    
    base_model = MultiTaskResNet18(pretrained=False)
    base_model.load_state_dict(torch.load("./checkpoints/base_model.pth", map_location=device))
    
    for task_id, task_name in task_mapping.items():
        expert = MultiTaskResNet18(pretrained=False)
        expert.load_state_dict(torch.load(f"./checkpoints/expert_{task_name}.pth", map_location=device))
        experts[task_name] = expert
        
    experts_list = [experts['mnist'], experts['fmnist'], experts['cifar10']]
    
    # 2. Perform weight merging
    print("Performing weight merging...")
    merged_model = MultiTaskResNet18(pretrained=False)
    if args.merge_method == "wa":
        merged_model = merge_models_weight_averaging(experts_list, merged_model)
    elif args.merge_method == "ta":
        merged_model = merge_models_task_arithmetic(experts_list, base_model, merged_model, lam=args.ta_lam)
        
    # 3. Load datasets and calibration sets
    print("Loading datasets and calibration splits...")
    dataloaders = load_datasets(data_dir="./data", num_train_samples=3000, seed=args.seed)
    joint_cal_set, task_cal_sets = get_calibration_set(data_dir="./data", N=args.cal_size, seed=args.seed)
    joint_cal_loader = DataLoader(joint_cal_set, batch_size=len(joint_cal_set), shuffle=False)
    
    # 4. Perform Representation Calibration (Backbone Alignment)
    if args.rep_cal == "ntaac":
        print("Applying Native Task-Agnostic Activation Calibration (N-TAAC)...")
        merged_model = calibrate_model_ntaac(merged_model, joint_cal_loader, device, momentum=1.0)
        eval_model = merged_model
    elif args.rep_cal in ["lsc", "tsc"]:
        print(f"Applying {args.rep_cal.upper()} representation calibration...")
        # LSC/TSC requires standard deviations of original experts and merged model on task-specific calibration sets
        experts_stds = {}
        merged_stds = {}
        
        for task_id, task_name in task_mapping.items():
            task_cal_loader = DataLoader(task_cal_sets[task_name], batch_size=args.cal_size, shuffle=False)
            # Original expert stds
            experts_stds[task_id] = get_layer_stds(experts[task_name], task_cal_loader, device, task_id)
            # Merged model stds (uncalibrated)
            merged_stds[task_id] = get_layer_stds(merged_model, task_cal_loader, device, task_id)
            
        # Calculate scaling factors
        tau = args.tsc_tau if args.rep_cal == "tsc" else 1.0
        scaling_factors = apply_lsc_calibration(merged_model, experts_stds, merged_stds, tau=tau)
        
        # Wrap merged model in CalibratedMultiTaskModel
        eval_model = CalibratedMultiTaskModel(merged_model, scaling_factors)
    else:
        eval_model = merged_model
        
    # 5. Perform Decision Boundary Alignment (Head Adaptation)
    # Note: Head adaptation is applied on top of the calibrated representation space
    if args.head_align == "sft":
        print("Running Supervised Head Fine-Tuning (SFT) on calibration subset...")
        # If model is wrapped (LSC/TSC), adapt the underlying base model's heads
        target_adap_model = eval_model.base_model if isinstance(eval_model, CalibratedMultiTaskModel) else eval_model
        train_head_sft(target_adap_model, task_cal_sets, device, epochs=args.epochs, lr=args.sft_tta_lr)
    elif args.head_align == "tta":
        print("Running Unsupervised Test-Time Head Adaptation (TTA) via KL-distillation...")
        target_adap_model = eval_model.base_model if isinstance(eval_model, CalibratedMultiTaskModel) else eval_model
        train_head_tta(target_adap_model, experts, task_cal_sets, device, epochs=args.epochs, lr=args.sft_tta_lr)
        
    # 6. Evaluate full multi-task performance
    print("\nEvaluating model on full test sets...")
    eval_model.to(device)
    results = evaluate_full(eval_model, dataloaders, device)
    
    print("\n--- Final Results ---")
    for k, v in results.items():
        print(f"Task {k.upper()}: {v:.2f}%")
        
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved results to {args.output_file}")

if __name__ == "__main__":
    main()
