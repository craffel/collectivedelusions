import torch
import torch.nn as nn
import torchvision.models as models
from datasets_utils import get_dataloaders
from calibration_methods import merge_models, calibrate_sequential
from merge_and_evaluate import evaluate, run_evaluation_for_all
import os

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loaders = get_dataloaders(batch_size=128)
    
    # Load expert heads for evaluation
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
    
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\n" + "="*60)
    print("EXPERIMENT 4: Task Arithmetic Scaling Sweep (Lambda vs SBR)")
    print("="*60)
    
    results = {}
    
    for l in lambdas:
        results[l] = {}
        for method in ['none', 'sbr']:
            print(f"Running TA: Lambda = {l}, Calibration = {method.upper()}...")
            # 1. Merge models
            model = merge_models(merge_mode='ta', lambda_val=l)
            model = model.to(device)
            
            # 2. Calibrate model
            model, hooks = calibrate_sequential(model, experts_list, loaders, method, device, cal_size=128)
            
            # 3. Evaluate model
            res = run_evaluation_for_all(model, experts_dict, loaders, device)
            
            # Remove hooks
            for h in hooks:
                h.remove()
                
            results[l][method] = res
            print(f"  Lambda={l} + {method.upper()} -> Avg Acc: {res['avg']:.2f}% (MNIST: {res['mnist']:.2f}%, F-MNIST: {res['fashion']:.2f}%, CIFAR: {res['cifar']:.2f}%)")

    # Save to file
    with open("lambda_sweep_results.txt", "w") as f:
        f.write("Task Arithmetic Lambda Sensitivity Study\n")
        f.write("========================================\n\n")
        f.write(f"{'Lambda':<8} | {'Method':<8} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}\n")
        f.write("-"*60 + "\n")
        for l in lambdas:
            for method in ['none', 'sbr']:
                res = results[l][method]
                method_name = "Uncal" if method == 'none' else "SBR"
                f.write(f"{l:<8.1f} | {method_name:<8} | {res['mnist']:<8.2f}% | {res['fashion']:<8.2f}% | {res['cifar']:<8.2f}% | {res['avg']:<8.2f}%\n")
            f.write("-"*60 + "\n")
            
    print("\nSweep completed! Results saved to lambda_sweep_results.txt.")

if __name__ == '__main__':
    main()
