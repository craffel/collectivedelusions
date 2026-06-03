import torch
import torch.nn as nn
import torchvision.models as models
from datasets_utils import get_dataloaders
from calibration_methods import merge_models, calibrate_sequential
import os

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def run_evaluation_for_all(model, experts_dict, loaders, device):
    results = {}
    tasks = ['mnist', 'fashion', 'cifar']
    for task in tasks:
        # Load task-specific expert head
        expert_head = experts_dict[task].fc
        model.fc.weight.data.copy_(expert_head.weight.data)
        model.fc.bias.data.copy_(expert_head.bias.data)
        
        test_loader = loaders[task]['test']
        acc = evaluate(model, test_loader, device)
        results[task] = acc * 100.0
    results['avg'] = sum(results[task] for task in tasks) / len(tasks)
    return results

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load loaders
    loaders = get_dataloaders(batch_size=128)
    
    # Load experts
    mnist_expert = models.resnet18(weights=None)
    mnist_expert.fc = nn.Linear(512, 10)
    mnist_expert.load_state_dict(torch.load('mnist_expert.pt', map_location='cpu'))
    mnist_expert = mnist_expert.to(device)
    
    fashion_expert = models.resnet18(weights=None)
    fashion_expert.fc = nn.Linear(512, 10)
    fashion_expert.load_state_dict(torch.load('fashion_expert.pt', map_location='cpu'))
    fashion_expert = fashion_expert.to(device)
    
    cifar_expert = models.resnet18(weights=None)
    cifar_expert.fc = nn.Linear(512, 10)
    cifar_expert.load_state_dict(torch.load('cifar_expert.pt', map_location='cpu'))
    cifar_expert = cifar_expert.to(device)
    
    experts_list = [mnist_expert, fashion_expert, cifar_expert]
    experts_dict = {
        'mnist': mnist_expert,
        'fashion': fashion_expert,
        'cifar': cifar_expert
    }
    
    # Evaluate Oracle (independent expert specialists)
    oracle_mnist = evaluate(mnist_expert, loaders['mnist']['test'], device) * 100.0
    oracle_fashion = evaluate(fashion_expert, loaders['fashion']['test'], device) * 100.0
    oracle_cifar = evaluate(cifar_expert, loaders['cifar']['test'], device) * 100.0
    oracle_avg = (oracle_mnist + oracle_fashion + oracle_cifar) / 3.0
    
    print("\n================ ORACLE (SPECIALISTS) ================")
    print(f"MNIST Specialist: {oracle_mnist:.2f}%")
    print(f"Fashion-MNIST Specialist: {oracle_fashion:.2f}%")
    print(f"CIFAR-10 Specialist: {oracle_cifar:.2f}%")
    print(f"Oracle Average: {oracle_avg:.2f}%")
    print("======================================================")
    
    merge_modes = ['wa', 'ta']
    calibration_methods = ['none', 'sp_taac', 'taac', 'l_fdsa', 'c_fdsa', 'l_dwss', 'c_dwss', 'abs', 'sbr']
    
    all_results = {}
    
    for mode in merge_modes:
        all_results[mode] = {}
        for method in calibration_methods:
            print(f"\nEvaluating: Merge Mode = {mode.upper()}, Calibration = {method.upper()}...")
            # 1. Merge models
            model = merge_models(merge_mode=mode, lambda_val=0.3)
            model = model.to(device)
            
            # 2. Calibrate model
            model, hooks = calibrate_sequential(model, experts_list, loaders, method, device)
            
            # 3. Evaluate model
            res = run_evaluation_for_all(model, experts_dict, loaders, device)
            
            # Remove FDSA hooks to avoid interfering with subsequent runs if any
            for h in hooks:
                h.remove()
                
            all_results[mode][method] = res
            print(f"Results for {mode.upper()} + {method.upper()}:")
            print(f"  MNIST: {res['mnist']:.2f}% | FASHION: {res['fashion']:.2f}% | CIFAR: {res['cifar']:.2f}% | AVERAGE: {res['avg']:.2f}%")
            
    # Print the final comparative table
    print("\n" + "="*80)
    print(f"{'Merge Mode':<12} | {'Calibration Method':<18} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("="*80)
    print(f"{'Oracle':<12} | {'-':<18} | {oracle_mnist:.2f}%  | {oracle_fashion:.2f}%  | {oracle_cifar:.2f}%  | {oracle_avg:.2f}%")
    print("-"*80)
    for mode in merge_modes:
        for method in calibration_methods:
            res = all_results[mode][method]
            method_disp = method.upper()
            if method == 'l_dwss':
                method_disp = "L-DWSS (Ours)"
            elif method == 'c_dwss':
                method_disp = "C-DWSS (Ours)"
            elif method == 'abs':
                method_disp = "ABS (Ours)"
            elif method == 'sbr':
                method_disp = "SBR (Ours)"
            print(f"{mode.upper():<12} | {method_disp:<18} | {res['mnist']:.2f}%  | {res['fashion']:.2f}%  | {res['cifar']:.2f}%  | {res['avg']:.2f}%")
        print("-"*80)
    print("="*80)

if __name__ == '__main__':
    main()
