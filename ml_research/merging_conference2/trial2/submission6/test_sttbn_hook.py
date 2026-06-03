import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from run_experiments import get_dataloaders, get_base_backbone, ExpertModel, evaluate_model, merge_task_arithmetic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STTBNPreHook:
    def __init__(self, momentum=0.1):
        self.momentum = momentum
        
    def __call__(self, module, input):
        x = input[0] # Shape (N, C, H, W)
        with torch.no_grad():
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running stats in-place before the layer's forward pass
            module.running_mean.copy_((1 - self.momentum) * module.running_mean + self.momentum * batch_mean)
            module.running_var.copy_((1 - self.momentum) * module.running_var + self.momentum * batch_var)

class STTBNPostHook:
    def __init__(self, momentum=0.1):
        self.momentum = momentum
        
    def __call__(self, module, input, output):
        x = input[0] # Shape (N, C, H, W)
        with torch.no_grad():
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running stats in-place after the layer's forward pass
            # so the current batch was normalized using the old running statistics
            module.running_mean.copy_((1 - self.momentum) * module.running_mean + self.momentum * batch_mean)
            module.running_var.copy_((1 - self.momentum) * module.running_var + self.momentum * batch_var)

def evaluate_model_sttbn(model, loader, momentum=0.1, mode='pre', init_mode='merged', expert_state=None):
    model.eval()
    
    # 1. Initialize running statistics
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.training = False # Keep in eval mode
            
            # Init running mean/var
            if init_mode == 'expert' and expert_state is not None:
                clean_name = name[len("backbone."):] if name.startswith("backbone.") else name
                prefix = clean_name + '.' if clean_name else ''
                module_state = {}
                for key, val in expert_state.items():
                    if key.startswith(prefix):
                        sub_key = key[len(prefix):]
                        if sub_key in ['running_mean', 'running_var']:
                            module_state[sub_key] = val.clone()
                module.load_state_dict(module_state, strict=False)

    # 2. Register hooks
    hook_handles = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if mode == 'pre':
                hook = STTBNPreHook(momentum=momentum)
                handle = module.register_forward_pre_hook(hook)
            elif mode == 'post':
                hook = STTBNPostHook(momentum=momentum)
                handle = module.register_forward_hook(hook)
            hook_handles.append(handle)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
        
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
    print("PROPER SEQUENTIAL TEST-TIME BATCHNORM (STTBN) EVALUATION")
    print("="*80)
    
    # 1. Baseline standard TTBN
    print("\n--- Baseline Standard TTBN ---")
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

    # 2. STTBN Pre-Hook with Merged Init
    for mom in momentums:
        print(f"\n--- STTBN (Pre-Hook, Merged Init, momentum={mom}) ---")
        print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-"*60)
        for bs in batch_sizes:
            bs_loaders = get_dataloaders(batch_size=bs)
            results = {}
            for task in tasks:
                eval_model = ExpertModel(eval_backbone, expert_models[task].head)
                eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
                acc = evaluate_model_sttbn(eval_model, bs_loaders[task]['test'], momentum=mom, mode='pre', init_mode='merged')
                results[task] = acc
            avg_acc = np.mean(list(results.values()))
            print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

    # 3. STTBN Pre-Hook with Expert Init
    for mom in momentums:
        print(f"\n--- STTBN (Pre-Hook, Expert Init, momentum={mom}) ---")
        print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-"*60)
        for bs in batch_sizes:
            bs_loaders = get_dataloaders(batch_size=bs)
            results = {}
            for task in tasks:
                eval_model = ExpertModel(eval_backbone, expert_models[task].head)
                eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
                acc = evaluate_model_sttbn(eval_model, bs_loaders[task]['test'], momentum=mom, mode='pre', init_mode='expert', expert_state=expert_models[task].backbone.state_dict())
                results[task] = acc
            avg_acc = np.mean(list(results.values()))
            print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

    # 4. STTBN Post-Hook with Merged Init
    for mom in momentums:
        print(f"\n--- STTBN (Post-Hook, Merged Init, momentum={mom}) ---")
        print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-"*60)
        for bs in batch_sizes:
            bs_loaders = get_dataloaders(batch_size=bs)
            results = {}
            for task in tasks:
                eval_model = ExpertModel(eval_backbone, expert_models[task].head)
                eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
                acc = evaluate_model_sttbn(eval_model, bs_loaders[task]['test'], momentum=mom, mode='post', init_mode='merged')
                results[task] = acc
            avg_acc = np.mean(list(results.values()))
            print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

    # 5. STTBN Post-Hook with Expert Init
    for mom in momentums:
        print(f"\n--- STTBN (Post-Hook, Expert Init, momentum={mom}) ---")
        print(f"{'Batch Size':<12} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8} | {'Average':<8}")
        print("-"*60)
        for bs in batch_sizes:
            bs_loaders = get_dataloaders(batch_size=bs)
            results = {}
            for task in tasks:
                eval_model = ExpertModel(eval_backbone, expert_models[task].head)
                eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
                acc = evaluate_model_sttbn(eval_model, bs_loaders[task]['test'], momentum=mom, mode='post', init_mode='expert', expert_state=expert_models[task].backbone.state_dict())
                results[task] = acc
            avg_acc = np.mean(list(results.values()))
            print(f"{bs:<12} | {results['mnist']:<8.2f} | {results['fashion']:<8.2f} | {results['cifar']:<8.2f} | {avg_acc:<8.2f}")

if __name__ == '__main__':
    main()
