import torch
import torch.nn as nn
import torchvision.models as models
import copy
import os
from train_and_merge import get_model, get_dataset, evaluate_model, merge_experts

def evaluate_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataloaders
    test_loaders = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        _, test_loaders[task] = get_dataset(task)
        
    # Load progenitor
    print("Loading progenitor...")
    progenitor = get_model()
    progenitor_state = copy.deepcopy(progenitor.state_dict())
    
    # Load expert states
    experts_states = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f"checkpoints/resnet18_{task}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Error: checkpoint {ckpt_path} not found.")
            return
        expert_data = torch.load(ckpt_path, map_location=device)
        experts_states[task] = expert_data['state_dict']

    print("\n--- Ablation: CPR with c=1.732 BUT keeping progenitor ImageNet BN stats ---")
    
    # Create CPR state dict with progenitor BN statistics
    # Standard merge:
    merged_state = merge_experts(progenitor_state, experts_states, 'cpr', c_val=1.732)
    
    # Overwrite BN stats back to progenitor's (ImageNet)
    for key in progenitor_state.keys():
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            merged_state[key] = copy.deepcopy(progenitor_state[key])
            
    # Evaluate
    accuracies = {}
    model = get_model().to(device)
    for task in ['mnist', 'fmnist', 'cifar10']:
        task_state = copy.deepcopy(merged_state)
        task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
        task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
        
        model.load_state_dict(task_state)
        acc = evaluate_model(model, test_loaders[task])
        accuracies[task] = acc
        print(f"  {task.upper()} Accuracy (Progenitor BN): {acc:.2f}%")
        
    avg_acc = sum(accuracies.values()) / len(accuracies)
    print(f"  Average Accuracy (Progenitor BN): {avg_acc:.2f}%")

if __name__ == '__main__':
    evaluate_ablation()
