import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models, generate_harmonic_patterns

def calibrate_running_stats_no_reset(model, calibration_data, epochs=10, device='cpu'):
    # Put model in train mode to trigger BatchNorm running stats updates
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
        
    print("\n--- Testing Harmonic Calibration (Joint) WITHOUT Resetting Stats ---")
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_model_state_dict, strict=False)
    harmonic_data = generate_harmonic_patterns(num_samples=256, img_size=32, device=device)
    calibrate_running_stats_no_reset(model, harmonic_data, epochs=10, device=device)
    
    joint_harmonic_results = {}
    model.eval()
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
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
        joint_harmonic_results[task] = acc
        print(f"Task {task.upper()} accuracy (Joint Harmonic Cal, no reset): {acc:.2f}%")
        
    print(f"Joint Harmonic Cal Average Accuracy: {sum(joint_harmonic_results.values())/3:.2f}%")

if __name__ == '__main__':
    main()
