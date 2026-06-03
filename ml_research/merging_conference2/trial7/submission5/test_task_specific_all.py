import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models

def calibrate_running_stats_no_reset(model, calibration_data, epochs=10, device='cpu'):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    loader = DataLoader(calibration_data, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            _ = model.backbone(x)
    model.eval()

def evaluate_on_task(model, test_dataset, task_name, device):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, task_name)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Load datasets
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    task_names = ['mnist', 'fmnist', 'cifar10']
    test_datasets_dict = {task: datasets_dict[task][1] for task in task_names}
    
    # Load experts
    expert_state_dicts = {}
    heads_state_dicts = {}
    for task in task_names:
        checkpoint = torch.load(f"./checkpoints/expert_{task}.pt", map_location=device)
        expert_state_dicts[task] = checkpoint['state_dict']
        heads_state_dicts[task] = checkpoint['head_state_dict']
        
    progenitor = MultiTaskResNet18().to(device)
    progenitor_state_dict = {f"backbone.{k}": v.cpu().clone() for k, v in progenitor.backbone.state_dict().items()}
    
    # Merge using Weight Averaging
    merged_backbone_state = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type='WA', lam=0.5)
    
    # Load synthetic datasets (DF-Calib-Gen)
    syn_datasets = torch.load("./checkpoints/synthetic_data.pt", map_location=device)
    
    # Establish a baseline: Uncalibrated
    print("\n--- BASELINE: Uncalibrated ---")
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_backbone_state, strict=False)
    uncal_results = {}
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
        uncal_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"Task {task.upper()}: {uncal_results[task]:.2f}%")
    print(f"Average: {sum(uncal_results.values())/3:.2f}%")
    
    # 1. Joint Real-Data (No Reset)
    print("\n--- 1. Joint Real-Data Calibration (No Reset) ---")
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_backbone_state, strict=False)
    # 256 combined samples
    real_samples = []
    for i, task in enumerate(task_names):
        train_sub, _ = datasets_dict[task]
        num_req = 86 if i == 2 else 85
        indices = torch.randperm(len(train_sub))[:num_req]
        for idx in indices:
            real_samples.append(train_sub[idx.item()][0])
    real_data = torch.stack(real_samples, dim=0)
    calibrate_running_stats_no_reset(model, real_data, epochs=10, device=device)
    
    joint_real_results = {}
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
        joint_real_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"Task {task.upper()}: {joint_real_results[task]:.2f}%")
    print(f"Average: {sum(joint_real_results.values())/3:.2f}%")
    
    # 2. Joint DF-Calib-Gen (No Reset)
    print("\n--- 2. Joint DF-Calib-Gen Calibration (No Reset) ---")
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_backbone_state, strict=False)
    syn_combined = torch.cat([
        syn_datasets['mnist'][:85],
        syn_datasets['fmnist'][:85],
        syn_datasets['cifar10'][:86]
    ], dim=0)
    calibrate_running_stats_no_reset(model, syn_combined, epochs=10, device=device)
    
    joint_syn_results = {}
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
        joint_syn_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"Task {task.upper()}: {joint_syn_results[task]:.2f}%")
    print(f"Average: {sum(joint_syn_results.values())/3:.2f}%")
    
    # 3. Task-Specific Real-Data (No Reset)
    print("\n--- 3. Task-Specific Real-Data Calibration (No Reset) ---")
    ts_real_results = {}
    for task in task_names:
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone_state, strict=False)
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
        train_sub, _ = datasets_dict[task]
        indices = torch.randperm(len(train_sub))[:256]
        real_samples = torch.stack([train_sub[idx][0] for idx in indices], dim=0)
        
        calibrate_running_stats_no_reset(model, real_samples, epochs=10, device=device)
        ts_real_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"Task {task.upper()}: {ts_real_results[task]:.2f}%")
    print(f"Average: {sum(ts_real_results.values())/3:.2f}%")
    
    # 4. Task-Specific DF-Calib-Gen (No Reset)
    print("\n--- 4. Task-Specific DF-Calib-Gen Calibration (No Reset) ---")
    ts_syn_results = {}
    for task in task_names:
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone_state, strict=False)
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
        syn_samples = syn_datasets[task][:256]
        calibrate_running_stats_no_reset(model, syn_samples, epochs=10, device=device)
        ts_syn_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"Task {task.upper()}: {ts_syn_results[task]:.2f}%")
    print(f"Average: {sum(ts_syn_results.values())/3:.2f}%")

if __name__ == '__main__':
    main()
