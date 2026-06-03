import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False

from data import get_multi_task_datasets, get_calibration_subset
from methods import (
    get_merged_state_dict,
    get_task_prototypes,
    mspr_route_sample
)
from evaluate import load_expert_model

def get_task_prototypes_custom(merged_model, calibration_loaders, layer_name='layer2', normalize=True, device='cpu'):
    merged_model.eval()
    prototypes = {}
    layer_module = dict(merged_model.named_modules())[layer_name]
    
    for task_name, loader in calibration_loaders.items():
        layer_acts = []
        def hook_fn(module, input, output):
            pooled = output.mean(dim=(2, 3)) # (B, C)
            layer_acts.append(pooled.detach().cpu())
            
        handle = layer_module.register_forward_hook(hook_fn)
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = merged_model(inputs)
        handle.remove()
        
        pooled_all = torch.cat(layer_acts, dim=0)
        avg_proto = pooled_all.mean(dim=0)
        if normalize:
            normalized_proto = avg_proto / (avg_proto.norm(p=2) + 1e-8)
            prototypes[task_name] = normalized_proto
        else:
            prototypes[task_name] = avg_proto
        
    return prototypes

def mspr_route_sample_custom(merged_model, x, prototypes, layer_name='layer2', metric='cosine', device='cpu'):
    merged_model.eval()
    layer_module = dict(merged_model.named_modules())[layer_name]
    captured_pooled = []
    def hook_fn(module, input, output):
        pooled = output.mean(dim=(2, 3))
        captured_pooled.append(pooled.detach())
        
    handle = layer_module.register_forward_hook(hook_fn)
    _ = merged_model(x)
    handle.remove()
    
    v = captured_pooled[0]
    task_names = list(prototypes.keys())
    proto_matrix = torch.stack([prototypes[name].to(device) for name in task_names])
    
    if metric == 'cosine':
        v_norm = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-8)
        proto_norm = proto_matrix / (proto_matrix.norm(p=2, dim=1, keepdim=True) + 1e-8)
        similarities = v_norm @ proto_norm.T
        best_task_indices = similarities.argmax(dim=1).cpu().numpy()
    elif metric == 'dot_product':
        similarities = v @ proto_matrix.T
        best_task_indices = similarities.argmax(dim=1).cpu().numpy()
    elif metric == 'euclidean':
        diff = v.unsqueeze(1) - proto_matrix.unsqueeze(0)
        dists = torch.linalg.vector_norm(diff, ord=2, dim=2)
        best_task_indices = dists.argmin(dim=1).cpu().numpy()
    elif metric == 'manhattan':
        diff = v.unsqueeze(1) - proto_matrix.unsqueeze(0)
        dists = torch.linalg.vector_norm(diff, ord=1, dim=2)
        best_task_indices = dists.argmin(dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown routing metric: {metric}")
        
    routed_tasks = [task_names[idx] for idx in best_task_indices]
    return routed_tasks

def run_routing_analysis(device='cpu', seed=42):
    print("Starting Routing Accuracy Analysis on CPU...")
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    # Load expert models to construct the uncalibrated WA model
    expert_state_dicts = []
    for t in tasks:
        model = load_expert_model(t, device=device)
        expert_state_dicts.append(copy.deepcopy(model.state_dict()))
        
    try:
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        base_model = resnet18(pretrained=True)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        base_model.fc
    )
    
    wa_state_dict = get_merged_state_dict(expert_state_dicts, mode='wa')
    wa_model = base_model.to(device)
    wa_model.load_state_dict(wa_state_dict)
    
    # Load datasets
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed)
    
    # Evaluate across N in [16, 64, 128]
    for N in [16, 64, 128]:
        print(f"\n--- Analysis for Calibration Budget N = {N} ---")
        cal_subsets = {t: get_calibration_subset(train_datasets[t], N, seed=seed) for t in tasks}
        cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
        
        # We'll evaluate different metrics
        for metric in ['cosine', 'dot_product', 'euclidean', 'manhattan']:
            normalize = (metric == 'cosine')
            prototypes = get_task_prototypes_custom(wa_model, cal_loaders, layer_name='layer2', normalize=normalize, device=device)
            
            # Confusion matrix: rows are ground truth, columns are predicted
            # keys: (gt_task, pred_task)
            confusion = {gt: {pred: 0 for pred in tasks} for gt in tasks}
            
            for gt_task in tasks:
                test_loader = DataLoader(test_datasets[gt_task], batch_size=128, shuffle=False)
                # To speed up CPU analysis, we can evaluate on first 1000 samples of test set (which is highly representative)
                count = 0
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    routed = mspr_route_sample_custom(wa_model, inputs, prototypes, layer_name='layer2', metric=metric, device=device)
                    for r in routed:
                        confusion[gt_task][r] += 1
                    count += inputs.size(0)
                    if count >= 1000:
                        break
            
            # Calculate accuracies
            print(f"\nMetric: {metric}")
            total_correct = 0
            total_samples = 0
            for gt_task in tasks:
                gt_total = sum(confusion[gt_task].values())
                correct = confusion[gt_task][gt_task]
                acc = 100.0 * correct / gt_total if gt_total > 0 else 0
                total_correct += correct
                total_samples += gt_total
                print(f"  GT {gt_task:<8} -> Routed as MNIST: {confusion[gt_task]['mnist']:4d} | FMNIST: {confusion[gt_task]['fmnist']:4d} | CIFAR10: {confusion[gt_task]['cifar10']:4d} (Acc: {acc:.2f}%)")
            overall_acc = 100.0 * total_correct / total_samples
            print(f"  Overall Routing Accuracy: {overall_acc:.2f}%")

if __name__ == '__main__':
    run_routing_analysis(device='cpu', seed=42)
