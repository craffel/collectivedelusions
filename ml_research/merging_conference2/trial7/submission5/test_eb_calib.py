import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models, generate_harmonic_patterns

def calibrate_running_stats_eb(model, expert_state_dicts, task_name, calibration_data, epochs=10, device='cpu', alpha=1.0, reset=True):
    # Put model in train mode
    model.train()
    
    # 1. Reset or initialize running stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if reset:
                m.reset_running_stats()
            m.momentum = 0.1
            
    # 2. Run forward passes to update running stats
    loader = DataLoader(calibration_data, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            _ = model.backbone(x)
            
    # 3. Apply the Expert-Bounding post-processing
    # We load the expert's BN running statistics
    expert_dict = expert_state_dicts[task_name]
    
    model.eval()
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # Retrieve expert's statistics for this layer
                expert_mean = expert_dict[f"{name}.running_mean"].to(device)
                expert_var = expert_dict[f"{name}.running_var"].to(device)
                
                # Expert-Bounding:
                # We bound the calibrated running variance to be at most alpha * expert_var
                m.running_var.copy_(torch.min(m.running_var, alpha * expert_var))
                
                # We can also keep the mean bounded or copy it if it's a quiet channel
                # Quiet channel indicator: where expert_var is extremely small
                quiet_mask = (expert_var < 1e-4)
                m.running_mean.copy_(torch.where(quiet_mask, expert_mean, m.running_mean))

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
    
    # Baseline: Uncalibrated
    print("\n--- BASELINE: Uncalibrated ---")
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_backbone_state, strict=False)
    uncal_results = {}
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
        uncal_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"  Task {task.upper()}: {uncal_results[task]:.2f}%")
    print(f"  Average: {sum(uncal_results.values())/3:.2f}%")
    
    # 1. EB-Calib with Harmonic Patterns (Task-Specific, reset=True)
    print("\n--- 1. Task-Specific EB-Calib using HarmonicCalib (Reset=True, alpha=1.0) ---")
    eb_harmonic_results = {}
    for task in task_names:
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone_state, strict=False)
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
        # Generate generic harmonic patterns
        harmonic_data = generate_harmonic_patterns(num_samples=256, img_size=32, device=device)
        
        # Run EB-Calib
        calibrate_running_stats_eb(model, expert_state_dicts, task, harmonic_data, epochs=10, device=device, alpha=1.0, reset=True)
        
        eb_harmonic_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"  Task {task.upper()}: {eb_harmonic_results[task]:.2f}%")
    print(f"  Average: {sum(eb_harmonic_results.values())/3:.2f}%")
    
    # 2. EB-Calib with Real Data (Task-Specific, reset=True)
    print("\n--- 2. Task-Specific EB-Calib using Real Data (Reset=True, alpha=1.0) ---")
    eb_real_results = {}
    for task in task_names:
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone_state, strict=False)
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
        train_sub, _ = datasets_dict[task]
        indices = torch.randperm(len(train_sub))[:256]
        real_samples = torch.stack([train_sub[idx][0] for idx in indices], dim=0)
        
        calibrate_running_stats_eb(model, expert_state_dicts, task, real_samples, epochs=10, device=device, alpha=1.0, reset=True)
        
        eb_real_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"  Task {task.upper()}: {eb_real_results[task]:.2f}%")
    print(f"  Average: {sum(eb_real_results.values())/3:.2f}%")

if __name__ == '__main__':
    main()
