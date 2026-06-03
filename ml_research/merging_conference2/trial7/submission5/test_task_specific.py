import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models, calibrate_running_stats, evaluate_merged_model, generate_harmonic_patterns

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 1. Load datasets
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    task_names = ['mnist', 'fmnist', 'cifar10']
    test_datasets_dict = {task: datasets_dict[task][1] for task in task_names}
    
    # 2. Load experts
    expert_state_dicts = {}
    heads_state_dicts = {}
    for task in task_names:
        checkpoint = torch.load(f"./checkpoints/expert_{task}.pt", map_location=device)
        expert_state_dicts[task] = checkpoint['state_dict']
        heads_state_dicts[task] = checkpoint['head_state_dict']
        
    # Progenitor
    progenitor = MultiTaskResNet18().to(device)
    progenitor_state_dict = {f"backbone.{k}": v.cpu().clone() for k, v in progenitor.backbone.state_dict().items()}
    
    # Merge using Weight Averaging
    merged_backbone_state = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type='WA', lam=0.5)
    
    merged_model_state_dict = {}
    for k, v in merged_backbone_state.items():
        merged_model_state_dict[k] = v
        
    print("\n--- Testing Task-Specific Calibration ---")
    
    task_specific_results = {}
    for task in task_names:
        print(f"\nCalibrating specifically for task: {task.upper()}")
        
        # Load merged weights
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_model_state_dict, strict=False)
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
        # Option A: Calibrate with 256 real samples of THIS task
        train_sub, _ = datasets_dict[task]
        indices = torch.randperm(len(train_sub))[:256]
        real_samples = torch.stack([train_sub[idx][0] for idx in indices], dim=0)
        
        calibrate_running_stats(model, real_samples, epochs=10, device=device)
        
        # Evaluate on this task's test set
        model.eval()
        test_loader = DataLoader(test_datasets_dict[task], batch_size=256, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, task)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        acc = 100.0 * correct / total
        task_specific_results[task] = acc
        print(f"Task {task.upper()} accuracy after task-specific calibration: {acc:.2f}%")
        
    avg_acc = sum(task_specific_results.values()) / len(task_specific_results)
    print(f"\nTask-Specific Real Calibration Average Accuracy: {avg_acc:.2f}%")

if __name__ == '__main__':
    main()
