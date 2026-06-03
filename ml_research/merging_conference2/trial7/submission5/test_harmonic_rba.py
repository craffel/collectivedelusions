import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models, generate_harmonic_patterns

def calibrate_running_stats_no_reset(model, calibration_data, epochs=30, momentum=0.2, device='cpu'):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
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
    
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    task_names = ['mnist', 'fmnist', 'cifar10']
    test_datasets_dict = {task: datasets_dict[task][1] for task in task_names}
    
    expert_state_dicts = {}
    heads_state_dicts = {}
    for task in task_names:
        checkpoint = torch.load(f"./checkpoints/expert_{task}.pt", map_location=device)
        expert_state_dicts[task] = checkpoint['state_dict']
        heads_state_dicts[task] = checkpoint['head_state_dict']
        
    progenitor = MultiTaskResNet18().to(device)
    progenitor_state_dict = {f"backbone.{k}": v.cpu().clone() for k, v in progenitor.backbone.state_dict().items()}
    
    merged_backbone_state = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type='WA', lam=0.5)
    
    # 1. Joint Harmonic-RBA
    print("\n--- 1. Joint Regularized Harmonic Resonance Calibration (no reset, mom=0.2, ep=30) ---")
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_backbone_state, strict=False)
    harmonic_data = generate_harmonic_patterns(num_samples=256, img_size=32, device=device)
    calibrate_running_stats_no_reset(model, harmonic_data, epochs=30, momentum=0.2, device=device)
    
    joint_results = {}
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
        joint_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"  Task {task.upper()}: {joint_results[task]:.2f}%")
    print(f"  Average: {sum(joint_results.values())/3:.2f}%")
    
    # 2. Task-Specific Harmonic-RBA?
    # Wait, harmonic sweeps are task-agnostic, but what if we scale/shift them using the expert's mean/std?
    # Let's see: we can generate harmonic patterns, and then for each channel, we scale them to match the expert's mean/std!
    # Yes! We call this "Expert-Scaled Harmonic sweeps"
    print("\n--- 2. Task-Specific Expert-Scaled Harmonic Calibration (no reset, mom=0.2, ep=30) ---")
    ts_results = {}
    for task in task_names:
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_backbone_state, strict=False)
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
        # Generate base harmonic patterns
        harmonic_data = generate_harmonic_patterns(num_samples=256, img_size=32, device=device)
        
        # Scale the harmonic patterns to match the expert's first layer stats
        # MNIST is grayscale, so its expert bn1 has specific stats. Let's see if we can scale the input harmonic_data to match.
        # Expert first layer: bn1. In ResNet-18, the input is normalized.
        # Let's get the expert's input layer running mean and std
        expert_bn1_mean = expert_state_dicts[task]["backbone.bn1.running_mean"].to(device)
        expert_bn1_var = expert_state_dicts[task]["backbone.bn1.running_var"].to(device)
        
        # For simplicity, let's just scale the input harmonic_data to match the task's general mean/std
        # MNIST mean ~ -0.73, std ~ 0.58
        # CIFAR10 mean ~ -0.04, std ~ 0.52
        # Let's measure the mean and std of the actual train dataset
        train_sub, _ = datasets_dict[task]
        indices = torch.randint(0, len(train_sub), (100,))
        sample_imgs = torch.stack([train_sub[idx][0] for idx in indices], dim=0)
        task_mean = sample_imgs.mean(dim=(0, 2, 3), keepdim=True).to(device) # shape (1, 3, 1, 1)
        task_std = sample_imgs.std(dim=(0, 2, 3), keepdim=True).to(device) # shape (1, 3, 1, 1)
        
        # Rescale harmonic data:
        # Currently, harmonic_data is in [-1, 1] with mean near 0 and std near 0.5
        # We normalize harmonic_data to 0 mean and 1 std, then scale to task_mean and task_std!
        h_mean = harmonic_data.mean(dim=(0, 2, 3), keepdim=True)
        h_std = harmonic_data.std(dim=(0, 2, 3), keepdim=True)
        norm_harmonic_data = (harmonic_data - h_mean) / (h_std + 1e-8)
        
        scaled_harmonic_data = norm_harmonic_data * task_std + task_mean
        scaled_harmonic_data = torch.clamp(scaled_harmonic_data, -2.0, 2.0)
        
        calibrate_running_stats_no_reset(model, scaled_harmonic_data, epochs=30, momentum=0.2, device=device)
        
        ts_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
        print(f"  Task {task.upper()}: {ts_results[task]:.2f}%")
    print(f"  Average: {sum(ts_results.values())/3:.2f}%")

if __name__ == '__main__':
    main()
