import sys
import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on the GPU node
torch.backends.cudnn.enabled = False

from data import get_multi_task_datasets, get_calibration_subset
from methods import (
    get_merged_state_dict,
    apply_slr_wbc,
)
from evaluate import load_expert_model

def run_early(model, inputs, layer_name):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    if layer_name == 'layer1':
        return x
    x = model.layer2(x)
    if layer_name == 'layer2':
        return x
    x = model.layer3(x)
    if layer_name == 'layer3':
        return x
    raise ValueError(f"Unsupported routing layer: {layer_name}")

def run_deep(ts_model, x_early, layer_name):
    x = x_early
    if layer_name == 'layer1':
        x = ts_model.layer2(x)
        x = ts_model.layer3(x)
        x = ts_model.layer4(x)
    elif layer_name == 'layer2':
        x = ts_model.layer3(x)
        x = ts_model.layer4(x)
    elif layer_name == 'layer3':
        x = ts_model.layer4(x)
    x = ts_model.avgpool(x)
    x = torch.flatten(x, 1)
    return ts_model.fc(x)

def get_task_prototypes_custom(merged_model, calibration_loaders, layer_name='layer2', normalize=True, device='cuda'):
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

def mspr_route_sample_custom(merged_model, x, prototypes, layer_name='layer2', metric='cosine', device='cuda'):
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

def ssr_merge_eval_custom(merged_model, task_specific_models, prototypes, test_loader, layer_name='layer2', metric='cosine', device='cuda'):
    merged_model.eval()
    for m in task_specific_models.values():
        m.eval()
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            routed_tasks = mspr_route_sample_custom(merged_model, inputs, prototypes, layer_name=layer_name, metric=metric, device=device)
            
            batch_size = inputs.size(0)
            outputs = torch.zeros(batch_size, 10, device=device)
            
            x_early = run_early(merged_model, inputs, layer_name)
            
            for task_name in prototypes.keys():
                indices = [i for i, t in enumerate(routed_tasks) if t == task_name]
                if len(indices) == 0:
                    continue
                    
                indices_tensor = torch.tensor(indices, device=device)
                x_group = x_early[indices_tensor]
                
                ts_model = task_specific_models[task_name]
                outputs_group = run_deep(ts_model, x_group, layer_name)
                outputs[indices_tensor] = outputs_group
                
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100.0 * correct / total

def run_metric_ablation(device='cuda'):
    print("==================================================")
    print("🚀 Running Fast Routing Metric Ablation 🚀")
    print("==================================================")
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = {}
    expert_heads = {}
    expert_state_dicts = []
    
    for t in tasks:
        model = load_expert_model(t, device=device)
        expert_models[t] = model
        expert_heads[t] = copy.deepcopy(model.fc)
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
    wa_model = copy.deepcopy(base_model).to(device)
    wa_model.load_state_dict(wa_state_dict)
    
    # Target Seed 42, N=128
    N_sweep = 128
    seed_sweep = 42
    
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed_sweep)
    test_loaders = {t: DataLoader(test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
    cal_subsets = {t: get_calibration_subset(train_datasets[t], N_sweep, seed=seed_sweep) for t in tasks}
    cal_loaders = {t: DataLoader(cal_subsets[t], batch_size=16, shuffle=False) for t in tasks}
    
    # Pre-calibrate standard task-specific models (rank=4, reg=0.1)
    task_specific_models_std = {}
    for t in tasks:
        ts_model = copy.deepcopy(wa_model)
        task_cal_batches = []
        for inputs, _ in cal_loaders[t]:
            task_cal_batches.append(inputs)
        task_cal_batches = [torch.cat(task_cal_batches, dim=0)]
        
        apply_slr_wbc(ts_model, [expert_models[t]], task_cal_batches, rank=4, reg=0.1, device=device)
        ts_model.fc = expert_heads[t]
        task_specific_models_std[t] = ts_model
        
    # Evaluate routing metrics
    routing_metrics = ['cosine', 'dot_product', 'euclidean', 'manhattan']
    metric_results = {}
    
    for metric in routing_metrics:
        print(f"Evaluating SSR-Merge with {metric} routing metric...")
        normalize_proto = (metric == 'cosine')
        proto_custom = get_task_prototypes_custom(wa_model, cal_loaders, layer_name='layer2', normalize=normalize_proto, device=device)
        
        ssr_accs = []
        for t in tasks:
            acc = ssr_merge_eval_custom(wa_model, task_specific_models_std, proto_custom, test_loaders[t], layer_name='layer2', metric=metric, device=device)
            ssr_accs.append(acc)
        avg_acc = sum(ssr_accs) / 3.0
        metric_results[metric] = avg_acc
        print(f"  {metric} routing metric average accuracy: {avg_acc:.2f}%")
        
    # Load sweep_results.json, update it, and write it back
    if os.path.exists('sweep_results.json'):
        with open('sweep_results.json', 'r') as f:
            data = json.load(f)
    else:
        data = {}
        
    data['metric_ablation'] = metric_results
    
    with open('sweep_results.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    print("\n✅ Successfully updated sweep_results.json with metric_ablation!")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_metric_ablation(device=device)
