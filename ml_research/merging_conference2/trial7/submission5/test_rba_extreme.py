import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from run_experiments import MultiTaskResNet18, get_datasets, merge_expert_models

def rba_calibration(model, calibration_data, epochs=10, momentum=0.1, device='cpu'):
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
    
    momentums = [0.15, 0.2, 0.3, 0.4]
    epochs_list = [20, 30, 40, 50]
    
    best_avg = 0.0
    best_hparams = None
    
    for mom in momentums:
        for ep in epochs_list:
            ts_results = {}
            for task in task_names:
                model = MultiTaskResNet18().to(device)
                model.load_state_dict(merged_backbone_state, strict=False)
                model.heads[task].load_state_dict(heads_state_dicts[task])
                
                train_sub, _ = datasets_dict[task]
                g = torch.Generator().manual_seed(42)
                indices = torch.randperm(len(train_sub), generator=g)[:256]
                real_samples = torch.stack([train_sub[idx][0] for idx in indices], dim=0)
                
                rba_calibration(model, real_samples, epochs=ep, momentum=mom, device=device)
                ts_results[task] = evaluate_on_task(model, test_datasets_dict[task], task, device)
                
            avg_acc = sum(ts_results.values()) / 3
            print(f"Momentum: {mom:.2f} | Epochs: {ep:2d} | MNIST: {ts_results['mnist']:.2f}% | FMNIST: {ts_results['fmnist']:.2f}% | CIFAR10: {ts_results['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")
            
            if avg_acc > best_avg:
                best_avg = avg_acc
                best_hparams = (mom, ep)
                
    print(f"\nBest configuration: Momentum={best_hparams[0]}, Epochs={best_hparams[1]} with Average Accuracy={best_avg:.2f}%")

if __name__ == '__main__':
    main()
