import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False

# Define transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ExpertModel(nn.Module):
    def __init__(self, task_name):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 10)
        self.task_name = task_name
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class MultiTaskMergedModel(nn.Module):
    def __init__(self, tasks=['mnist', 'fashion', 'cifar10']):
        super().__init__()
        self.backbone = models.resnet18()
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in tasks
        })

def get_datasets(task):
    if task == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown task {task}")
    return train_set, test_set

def get_calibration_subset(train_set, N=128, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(train_set), generator=g)[:N].tolist()
    return Subset(train_set, indices)

def get_merged_backbone_state_dict(expert_state_dicts, pretrained_state_dict, method="WA", lambda_coeff=0.3):
    merged_state_dict = {}
    keys = list(expert_state_dicts[0].keys())
    for key in keys:
        if key.startswith('backbone.'):
            params = [state_dict[key] for state_dict in expert_state_dicts]
            is_running_stat = any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked'])
            if is_running_stat:
                if 'num_batches_tracked' in key:
                    merged_state_dict[key] = torch.stack([p.float() for p in params]).mean(dim=0).long()
                else:
                    merged_state_dict[key] = torch.stack(params).mean(dim=0)
            else:
                if method == "WA":
                    merged_state_dict[key] = torch.stack(params).mean(dim=0)
                elif method == "TA":
                    pre_key = key.replace('backbone.', '')
                    pre_param = pretrained_state_dict[pre_key]
                    task_vectors = [p - pre_param for p in params]
                    merged_state_dict[key] = pre_param + lambda_coeff * torch.stack(task_vectors).sum(dim=0)
    return merged_state_dict

def evaluate_model(model, test_loaders, device):
    model.eval()
    task_accuracies = {}
    for task_name, test_loader in test_loaders.items():
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                features = model.backbone(inputs)
                outputs = model.heads[task_name](features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        task_accuracies[task_name] = 100.0 * correct / total
    task_accuracies['average'] = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies

def run_native_calibration(model, joint_loader, target_layers=None, momentum=1.0):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if target_layers is None:
                module.train()
                module.momentum = momentum
            else:
                is_target = any(target in name for target in target_layers)
                if is_target:
                    module.train()
                    module.momentum = momentum
                else:
                    module.eval()
                    module.momentum = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, _ in joint_loader:
            inputs = inputs.to(device)
            _ = model.backbone(inputs)
            break
    model.eval()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running calibration ablation on device: {device}")
    
    tasks = ['mnist', 'fashion', 'cifar10']
    expert_state_dicts = []
    test_loaders = {}
    train_datasets = {}
    
    # Load test sets and checkpoints
    for task in tasks:
        train_set, test_set = get_datasets(task)
        train_datasets[task] = train_set
        test_loaders[task] = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        
        checkpoint_path = f"expert_{task}.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expert_state_dicts.append(checkpoint['model_state_dict'])
        
    pretrained_backbone_state_dict = torch.load("resnet18_pretrained.pth", map_location='cpu')
    
    def create_merged_model(method, lambda_coeff):
        merged_sd = get_merged_backbone_state_dict(expert_state_dicts, pretrained_backbone_state_dict, method, lambda_coeff)
        model = MultiTaskMergedModel(tasks)
        model.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in merged_sd.items() if k.startswith('backbone.')})
        for i, task in enumerate(tasks):
            head_state_dict = {
                'weight': expert_state_dicts[i]['head.weight'],
                'bias': expert_state_dicts[i]['head.bias']
            }
            model.heads[task].load_state_dict(head_state_dict)
        return model.to(device)
        
    N_sizes = [16, 32, 64, 128, 256, 512]
    ablation_results = {
        'WA': {n: {} for n in N_sizes},
        'TA': {n: {} for n in N_sizes}
    }
    
    for N in N_sizes:
        print("\n" + "="*50)
        print(f"Running calibration with size N = {N}")
        print("="*50)
        
        # Build joint loader for size N
        cal_datasets = {}
        for task in tasks:
            cal_datasets[task] = get_calibration_subset(train_datasets[task], N=N, seed=42)
        joint_cal_dataset = ConcatDataset([cal_datasets[t] for t in tasks])
        joint_loader = DataLoader(joint_cal_dataset, batch_size=len(joint_cal_dataset), shuffle=False)
        
        # 1. Weight Averaging
        model_wa = create_merged_model("WA", 0.333)
        # N-TAAC
        run_native_calibration(model_wa, joint_loader, target_layers=None)
        acc_ntaac_wa = evaluate_model(model_wa, test_loaders, device)
        
        # T-NAC (L3+L4)
        model_wa = create_merged_model("WA", 0.333)
        run_native_calibration(model_wa, joint_loader, target_layers=['layer3', 'layer4'])
        acc_tnac34_wa = evaluate_model(model_wa, test_loaders, device)
        
        # T-NAC (L4 only)
        model_wa = create_merged_model("WA", 0.333)
        run_native_calibration(model_wa, joint_loader, target_layers=['layer4'])
        acc_tnac4_wa = evaluate_model(model_wa, test_loaders, device)
        
        ablation_results['WA'][N] = {
            'N_TAAC': acc_ntaac_wa['average'],
            'T_NAC_L34': acc_tnac34_wa['average'],
            'T_NAC_L4': acc_tnac4_wa['average']
        }
        
        print(f"WA (lambda=0.333):")
        print(f"  N-TAAC     (all)   : Avg Acc = {acc_ntaac_wa['average']:.2f}%")
        print(f"  T-NAC      (L3+L4) : Avg Acc = {acc_tnac34_wa['average']:.2f}%")
        print(f"  T-NAC      (L4)    : Avg Acc = {acc_tnac4_wa['average']:.2f}%")
        
        # 2. Task Arithmetic
        model_ta = create_merged_model("TA", 0.4)
        # N-TAAC
        run_native_calibration(model_ta, joint_loader, target_layers=None)
        acc_ntaac_ta = evaluate_model(model_ta, test_loaders, device)
        
        # T-NAC (L3+L4)
        model_ta = create_merged_model("TA", 0.4)
        run_native_calibration(model_ta, joint_loader, target_layers=['layer3', 'layer4'])
        acc_tnac34_ta = evaluate_model(model_ta, test_loaders, device)
        
        # T-NAC (L4 only)
        model_ta = create_merged_model("TA", 0.4)
        run_native_calibration(model_ta, joint_loader, target_layers=['layer4'])
        acc_tnac4_ta = evaluate_model(model_ta, test_loaders, device)
        
        ablation_results['TA'][N] = {
            'N_TAAC': acc_ntaac_ta['average'],
            'T_NAC_L34': acc_tnac34_ta['average'],
            'T_NAC_L4': acc_tnac4_ta['average']
        }
        
        print(f"TA (lambda=0.4):")
        print(f"  N-TAAC     (all)   : Avg Acc = {acc_ntaac_ta['average']:.2f}%")
        print(f"  T-NAC      (L3+L4) : Avg Acc = {acc_tnac34_ta['average']:.2f}%")
        print(f"  T-NAC      (L4)    : Avg Acc = {acc_tnac4_ta['average']:.2f}%")
        
    torch.save(ablation_results, "ablation_results.pt")
    print("\nSuccessfully saved ablation results to ablation_results.pt")
    
    # Print nice markdown table for easy copy-pasting
    print("\n" + "="*50)
    print("SUMMARY ABLATION TABLE (Average Accuracy %)")
    print("="*50)
    print("| Merge Setup | Calibration Size N | N-TAAC (Full) | T-NAC (L3+L4) | T-NAC (L4 only) | Gap (T-NAC34 - N-TAAC) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for merge_type in ['WA', 'TA']:
        for N in N_sizes:
            res = ablation_results[merge_type][N]
            gap = res['T_NAC_L34'] - res['N_TAAC']
            print(f"| {merge_type:<11} | {N:<18} | {res['N_TAAC']:.2f}% | {res['T_NAC_L34']:.2f}% | {res['T_NAC_L4']:.2f}% | {gap:+.2f}% |")

if __name__ == "__main__":
    main()
