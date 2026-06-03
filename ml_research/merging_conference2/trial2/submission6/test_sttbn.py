import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from run_experiments import get_dataloaders, get_base_backbone, ExpertModel, evaluate_model, merge_task_arithmetic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model_sequential(model, loader, momentum=0.1, init_mode='merged', expert_state=None):
    # Load expert BN running stats if specified
    if init_mode == 'expert' and expert_state is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                prefix = name + '.' if name else ''
                module_state = {}
                for key, val in expert_state.items():
                    if key.startswith(prefix):
                        sub_key = key[len(prefix):]
                        if sub_key in ['running_mean', 'running_var', 'num_batches_tracked']:
                            module_state[sub_key] = val.clone()
                module.load_state_dict(module_state, strict=False)

    # Set BatchNorm to training mode so that running statistics are updated sequentially
    # while keeping everything else in eval mode
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.training = True
            module.momentum = momentum

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

def main():
    tasks = ['mnist', 'fashion', 'cifar']
    
    # Load experts
    expert_models = {}
    expert_backbone_states = []
    
    pretrained_backbone = get_base_backbone()
    pretrained_backbone_state = copy.deepcopy(pretrained_backbone.state_dict())
    
    for task in tasks:
        ckpt_path = f"expert_{task}.pt"
        if os.path.exists(ckpt_path):
            backbone = get_base_backbone().to(device)
            head = nn.Linear(512, 10).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            backbone.load_state_dict(ckpt['backbone'])
            head.load_state_dict(ckpt['head'])
            model = ExpertModel(backbone, head)
            expert_models[task] = model
            expert_backbone_states.append(model.backbone.state_dict())
            
    # Merge with Lambda = 0.4 (best TA + TTBN lambda)
    ta_state = merge_task_arithmetic(pretrained_backbone_state, expert_backbone_states, 0.4)
    eval_backbone = get_base_backbone().to(device)
    
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    momentums = [0.01, 0.05, 0.1, 0.2]
    
    print("\n" + "="*80)
    print("SEQUENTIAL TEST-TIME BATCHNORM (STTBN) EVALUATION")
    print("="*80)
    
    # 1. Baseline standard TTBN
    print("\n--- Baseline Standard TTBN (No Sequential Updates, momentum=0) ---")
    print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-"*60)
    for bs in batch_sizes:
        bs_loaders = get_dataloaders(batch_size=bs)
        results = {}
        for task in tasks:
            eval_model = ExpertModel(eval_backbone, expert_models[task].head)
            eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
            acc = evaluate_model(eval_model, bs_loaders[task]['test'], bn_mode='train')
            results[task] = acc
        avg_acc = np.mean(list(results.values()))
        print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

    # 2. STTBN with Merged Init
    for mom in momentums:
        print(f"\n--- STTBN (Merged Init, momentum={mom}) ---")
        print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-"*60)
        for bs in batch_sizes:
            bs_loaders = get_dataloaders(batch_size=bs)
            results = {}
            for task in tasks:
                eval_model = ExpertModel(eval_backbone, expert_models[task].head)
                eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
                acc = evaluate_model_sequential(eval_model, bs_loaders[task]['test'], momentum=mom, init_mode='merged')
                results[task] = acc
            avg_acc = np.mean(list(results.values()))
            print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

    # 3. STTBN with Expert Init
    for mom in momentums:
        print(f"\n--- STTBN (Expert Init, momentum={mom}) ---")
        print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-"*60)
        for bs in batch_sizes:
            bs_loaders = get_dataloaders(batch_size=bs)
            results = {}
            for task in tasks:
                eval_model = ExpertModel(eval_backbone, expert_models[task].head)
                eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
                acc = evaluate_model_sequential(eval_model, bs_loaders[task]['test'], momentum=mom, init_mode='expert', expert_state=expert_models[task].backbone.state_dict())
                results[task] = acc
            avg_acc = np.mean(list(results.values()))
            print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

if __name__ == '__main__':
    main()
