import torch
import torchvision.models as models
import torch.nn as nn
from run_experiments import get_resnet18_progenitor, create_expert_model, merge_task_arithmetic, get_dataloaders, set_task_head

device = "cpu"
progenitor = get_resnet18_progenitor()
expert_paths = {
    'mnist': 'expert_mnist.pt',
    'fmnist': 'expert_fmnist.pt',
    'cifar': 'expert_cifar.pt'
}
expert_models = {}
for task in ['mnist', 'fmnist', 'cifar']:
    expert_models[task] = create_expert_model(progenitor, num_classes=10)
    expert_models[task].load_state_dict(torch.load(expert_paths[task], map_location=device))
    expert_models[task] = expert_models[task].to(device)

loaders = get_dataloaders(batch_size=128, sample_limit=5000)

for lam in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    ta_model = merge_task_arithmetic(progenitor, expert_models, lam=lam)
    ta_model = ta_model.to(device)
    
    accs = {}
    has_nan = False
    
    # check for NaNs in parameters
    for p in ta_model.parameters():
        if torch.isnan(p).any():
            has_nan = True
            break
            
    if has_nan:
        print(f"lam={lam:.2f} | NaNs in parameters!")
        continue
        
    for task in ['mnist', 'fmnist', 'cifar']:
        set_task_head(ta_model, expert_models[task])
        ta_model.eval()
        
        # run 1 batch
        for x, y in loaders['test'][task]:
            x = x.to(device)
            with torch.no_grad():
                outputs = ta_model(x)
                if torch.isnan(outputs).any():
                    has_nan = True
                    break
                _, predicted = outputs.max(1)
                correct = predicted.eq(y.to(device)).sum().item()
                accs[task] = 100.0 * correct / x.size(0)
            break
            
    if has_nan:
        print(f"lam={lam:.2f} | NaNs in activations!")
    else:
        avg = sum(accs.values()) / len(accs)
        print(f"lam={lam:.2f} | MNIST={accs['mnist']:.2f}%, F-MNIST={accs['fmnist']:.2f}%, CIFAR={accs['cifar']:.2f}% | Avg={avg:.2f}%")
