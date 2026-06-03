import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models.quantization as quantized_models

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

# Set PyTorch quantization engine for CPU serving
torch.backends.quantized.engine = 'fbgemm'

class QuantizedModelWrapper(nn.Module):
    def __init__(self, quantized_model):
        super().__init__()
        self.quantized_model = quantized_model
        # Extract classification head and keep it in FP32
        self.fc = quantized_model.fc
        # Replace the internal fc layer with Identity
        self.quantized_model.fc = nn.Identity()
        
    def forward(self, x):
        features = self.quantized_model(x)
        return self.fc(features)

def get_resnet18_progenitor():
    import torchvision.models as models
    # Use quantization ready model for progenitor but load floating-point weights
    model = quantized_models.resnet18(weights=models.ResNet18_Weights.DEFAULT, quantize=False)
    return model

def create_expert_model(progenitor, num_classes=10):
    model = quantized_models.resnet18(quantize=False)
    model.fc = nn.Linear(512, progenitor.fc.out_features)
    model.load_state_dict(progenitor.state_dict())
    if progenitor.fc.out_features != num_classes:
        model.fc = nn.Linear(512, num_classes)
    return model

def set_task_head(model, expert_model):
    model.fc = expert_model.fc.to(next(model.parameters()).device)

def merge_weights_averaging(expert_models):
    merged = quantized_models.resnet18(quantize=False)
    merged.fc = nn.Linear(512, 10)
    merged_sd = merged.state_dict()
    expert_sds = [exp.state_dict() for exp in expert_models.values()]
    for key in merged_sd.keys():
        if "fc" not in key:
            stacked = torch.stack([sd[key].float().cpu() for sd in expert_sds], dim=0)
            merged_sd[key] = torch.mean(stacked, dim=0).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

def merge_task_arithmetic(progenitor, expert_models, lam=0.25):
    merged = quantized_models.resnet18(quantize=False)
    merged.fc = nn.Linear(512, 10)
    merged_sd = merged.state_dict()
    prog_sd = progenitor.state_dict()
    expert_sds = {task: exp.state_dict() for task, exp in expert_models.items()}
    for key in merged_sd.keys():
        if "fc" not in key:
            task_vectors = []
            for task, sd in expert_sds.items():
                tv = sd[key].float().cpu() - prog_sd[key].float().cpu()
                task_vectors.append(tv)
            sum_tv = torch.sum(torch.stack(task_vectors, dim=0), dim=0)
            merged_sd[key] = (prog_sd[key].float().cpu() + lam * sum_tv).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

def get_transforms(is_grayscale=False):
    if is_grayscale:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir="./data", batch_size=128):
    os.makedirs(data_dir, exist_ok=True)
    gray_tf = get_transforms(is_grayscale=True)
    color_tf = get_transforms(is_grayscale=False)
    
    mnist_train_full = datasets.MNIST(data_dir, train=True, download=True, transform=gray_tf)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=gray_tf)
    fmnist_train_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=gray_tf)
    fmnist_test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=gray_tf)
    cifar_train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=color_tf)
    cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=color_tf)
    
    loaders = {
        'train': {
            'mnist': DataLoader(mnist_train_full, batch_size=batch_size, shuffle=True),
            'fmnist': DataLoader(fmnist_train_full, batch_size=batch_size, shuffle=True),
            'cifar': DataLoader(cifar_train_full, batch_size=batch_size, shuffle=True)
        },
        'test': {
            'mnist': DataLoader(mnist_test, batch_size=batch_size, shuffle=False),
            'fmnist': DataLoader(fmnist_test, batch_size=batch_size, shuffle=False),
            'cifar': DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
        }
    }
    return loaders

def evaluate_model(model, dataloader, task_name, expert_model, device='cpu'):
    model.eval()
    set_task_head(model, expert_model)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

def run_central_quantization(model, expert_models, train_loaders, N=1024):
    """Standard Centralized Post-Training Quantization (PTQ) Calibration."""
    cloned_model = quantized_models.resnet18(quantize=False)
    cloned_model.fc = nn.Linear(512, 10)
    cloned_model.load_state_dict(model.state_dict())
    
    cloned_model.eval()
    cloned_model.qconfig = quantization.get_default_qconfig('fbgemm')
    cloned_model.fc.qconfig = None # Keep head in FP32
    prepared = quantization.prepare(cloned_model)
    
    # Run forward passes on 1024 samples per task
    with torch.no_grad():
        for task, loader in train_loaders.items():
            set_task_head(prepared, expert_models[task])
            count = 0
            for x, y in loader:
                prepared(x)
                count += x.shape[0]
                if count >= N:
                    break
                    
    quantized_model = quantization.convert(prepared)
    return QuantizedModelWrapper(quantized_model)

def run_federated_quantization_calibration(model, expert_models, train_loaders, N=128):
    """
    Proposed Federated Quantization Calibration (Fed-QC).
    
    Clients run prepared models locally on N private samples to calibrate activation observers,
    send only 1D min/max ranges to the server, and the server averages them before converting.
    """
    model.eval()
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    model.fc.qconfig = None
    
    client_min_max = {}
    
    for task, loader in train_loaders.items():
        # Each client gets a fresh prepared model to prevent cross-contamination
        prep_client = quantized_models.resnet18(quantize=False)
        prep_client.fc = nn.Linear(512, 10)
        prep_client.load_state_dict(model.state_dict())
        prep_client.eval()
        prep_client.qconfig = quantization.get_default_qconfig('fbgemm')
        prep_client.fc.qconfig = None
        prep_client = quantization.prepare(prep_client)
        
        # Set client-specific head
        set_task_head(prep_client, expert_models[task])
        
        # Run local client calibration forward pass on N samples
        count = 0
        with torch.no_grad():
            for x, y in loader:
                prep_client(x)
                count += x.shape[0]
                if count >= N:
                    break
                    
        # Extract observer stats
        local_stats = {}
        for name, module in prep_client.named_modules():
            if 'activation_post_process' in name:
                if hasattr(module, 'min_val') and module.min_val is not None:
                    local_stats[name] = (module.min_val.clone(), module.max_val.clone())
        client_min_max[task] = local_stats
        
    # Server-side: prepare a global model
    global_temp = quantized_models.resnet18(quantize=False)
    global_temp.fc = nn.Linear(512, 10)
    global_temp.load_state_dict(model.state_dict())
    global_temp.eval()
    global_temp.qconfig = quantization.get_default_qconfig('fbgemm')
    global_temp.fc.qconfig = None
    prep_global = quantization.prepare(global_temp)
    
    # Aggregate (average) client stats on the server
    for name, module in prep_global.named_modules():
        if 'activation_post_process' in name:
            if hasattr(module, 'min_val'):
                mins = []
                maxs = []
                for task in train_loaders.keys():
                    if name in client_min_max[task]:
                        mins.append(client_min_max[task][name][0])
                        maxs.append(client_min_max[task][name][1])
                if len(mins) > 0:
                    avg_min = torch.mean(torch.stack(mins), dim=0)
                    avg_max = torch.mean(torch.stack(maxs), dim=0)
                    module.min_val = avg_min
                    module.max_val = avg_max
                    
    # Server-side conversion to INT8
    quantized_model = quantization.convert(prep_global)
    return QuantizedModelWrapper(quantized_model)

def run_naive_quantization(model):
    """Naive static quantization without any forward pass calibration."""
    cloned_model = quantized_models.resnet18(quantize=False)
    cloned_model.fc = nn.Linear(512, 10)
    cloned_model.load_state_dict(model.state_dict())
    
    cloned_model.eval()
    cloned_model.qconfig = quantization.get_default_qconfig('fbgemm')
    cloned_model.fc.qconfig = None
    prepared = quantization.prepare(cloned_model)
    quantized_model = quantization.convert(prepared)
    return QuantizedModelWrapper(quantized_model)

def main():
    print("=== STARTING PRIVACY-PRESERVING FEDERATED QUANTIZATION STUDY ===")
    set_seed(42)
    device = 'cpu'
    
    # Load progenitor
    progenitor = get_resnet18_progenitor()
    
    # Load individual expert models
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
        
    loaders = get_dataloaders(batch_size=128)
    
    # 1. Merge weights via Weight Averaging
    print("\nMerging models via Weight Averaging...")
    wa_model = merge_weights_averaging(expert_models).to(device)
    
    # Evaluate FP32 baseline
    print("Evaluating FP32 WA Model (Uncalibrated)...")
    accs_wa_fp32 = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_wa_fp32[task] = evaluate_model(wa_model, loaders['test'][task], task, expert_models[task], device=device)
    avg_wa_fp32 = sum(accs_wa_fp32.values()) / 3.0
    print(f"  FP32 WA: MNIST={accs_wa_fp32['mnist']:.2f}%, F-MNIST={accs_wa_fp32['fmnist']:.2f}%, CIFAR={accs_wa_fp32['cifar']:.2f}% | Avg={avg_wa_fp32:.2f}%")
    
    # 2. Evaluate Naive Quantization
    print("Evaluating Naive INT8 WA Model (No calibration)...")
    naive_quant_wa = run_naive_quantization(wa_model)
    accs_wa_naive = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_wa_naive[task] = evaluate_model(naive_quant_wa, loaders['test'][task], task, expert_models[task], device=device)
    avg_wa_naive = sum(accs_wa_naive.values()) / 3.0
    print(f"  Naive INT8: MNIST={accs_wa_naive['mnist']:.2f}%, F-MNIST={accs_wa_naive['fmnist']:.2f}%, CIFAR={accs_wa_naive['cifar']:.2f}% | Avg={avg_wa_naive:.2f}%")
    
    # 3. Evaluate Centralized Quantization (Oracle)
    print("Evaluating Centralized INT8 WA Model (N=1024 per task)...")
    central_quant_wa = run_central_quantization(wa_model, expert_models, loaders['train'], N=1024)
    accs_wa_central = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_wa_central[task] = evaluate_model(central_quant_wa, loaders['test'][task], task, expert_models[task], device=device)
    avg_wa_central = sum(accs_wa_central.values()) / 3.0
    print(f"  Centralized INT8 (N=1024): MNIST={accs_wa_central['mnist']:.2f}%, F-MNIST={accs_wa_central['fmnist']:.2f}%, CIFAR={accs_wa_central['cifar']:.2f}% | Avg={avg_wa_central:.2f}%")
    
    # 4. Evaluate our proposed Federated Quantization Calibration (Fed-QC)
    print("Evaluating Federated INT8 WA Model (Fed-QC, N=128 per client)...")
    fed_quant_wa = run_federated_quantization_calibration(wa_model, expert_models, loaders['train'], N=128)
    accs_wa_fed_qc = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_wa_fed_qc[task] = evaluate_model(fed_quant_wa, loaders['test'][task], task, expert_models[task], device=device)
    avg_wa_fed_qc = sum(accs_wa_fed_qc.values()) / 3.0
    print(f"  Federated INT8 (Fed-QC, N=128): MNIST={accs_wa_fed_qc['mnist']:.2f}%, F-MNIST={accs_wa_fed_qc['fmnist']:.2f}%, CIFAR={accs_wa_fed_qc['cifar']:.2f}% | Avg={avg_wa_fed_qc:.2f}%")
    
    # Same study for Task Arithmetic
    print("\nMerging models via Task Arithmetic (lambda=0.25)...")
    ta_model = merge_task_arithmetic(progenitor, expert_models, lam=0.25).to(device)
    
    # Evaluate FP32 baseline
    print("Evaluating FP32 TA Model (Uncalibrated)...")
    accs_ta_fp32 = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_ta_fp32[task] = evaluate_model(ta_model, loaders['test'][task], task, expert_models[task], device=device)
    avg_ta_fp32 = sum(accs_ta_fp32.values()) / 3.0
    print(f"  FP32 TA: MNIST={accs_ta_fp32['mnist']:.2f}%, F-MNIST={accs_ta_fp32['fmnist']:.2f}%, CIFAR={accs_ta_fp32['cifar']:.2f}% | Avg={avg_ta_fp32:.2f}%")
    
    # Naive Quantization
    print("Evaluating Naive INT8 TA Model (No calibration)...")
    naive_quant_ta = run_naive_quantization(ta_model)
    accs_ta_naive = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_ta_naive[task] = evaluate_model(naive_quant_ta, loaders['test'][task], task, expert_models[task], device=device)
    avg_ta_naive = sum(accs_ta_naive.values()) / 3.0
    print(f"  Naive INT8: MNIST={accs_ta_naive['mnist']:.2f}%, F-MNIST={accs_ta_naive['fmnist']:.2f}%, CIFAR={accs_ta_naive['cifar']:.2f}% | Avg={avg_ta_naive:.2f}%")
    
    # Centralized Quantization
    print("Evaluating Centralized INT8 TA Model (N=1024 per task)...")
    central_quant_ta = run_central_quantization(ta_model, expert_models, loaders['train'], N=1024)
    accs_ta_central = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_ta_central[task] = evaluate_model(central_quant_ta, loaders['test'][task], task, expert_models[task], device=device)
    avg_ta_central = sum(accs_ta_central.values()) / 3.0
    print(f"  Centralized INT8 (N=1024): MNIST={accs_ta_central['mnist']:.2f}%, F-MNIST={accs_ta_central['fmnist']:.2f}%, CIFAR={accs_ta_central['cifar']:.2f}% | Avg={avg_ta_central:.2f}%")
    
    # Federated Quantization Calibration (Fed-QC)
    print("Evaluating Federated INT8 TA Model (Fed-QC, N=128 per client)...")
    fed_quant_ta = run_federated_quantization_calibration(ta_model, expert_models, loaders['train'], N=128)
    accs_ta_fed_qc = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        accs_ta_fed_qc[task] = evaluate_model(fed_quant_ta, loaders['test'][task], task, expert_models[task], device=device)
    avg_ta_fed_qc = sum(accs_ta_fed_qc.values()) / 3.0
    print(f"  Federated INT8 (Fed-QC, N=128): MNIST={accs_ta_fed_qc['mnist']:.2f}%, F-MNIST={accs_ta_fed_qc['fmnist']:.2f}%, CIFAR={accs_ta_fed_qc['cifar']:.2f}% | Avg={avg_ta_fed_qc:.2f}%")
    
    # Save results
    results = {
        'WA': {
            'FP32': accs_wa_fp32,
            'Naive_INT8': accs_wa_naive,
            'Central_INT8': accs_wa_central,
            'Fed_QC_INT8': accs_wa_fed_qc,
            'Avg': {
                'FP32': avg_wa_fp32,
                'Naive_INT8': avg_wa_naive,
                'Central_INT8': avg_wa_central,
                'Fed_QC_INT8': avg_wa_fed_qc
            }
        },
        'TA': {
            'FP32': accs_ta_fp32,
            'Naive_INT8': accs_ta_naive,
            'Central_INT8': accs_ta_central,
            'Fed_QC_INT8': accs_ta_fed_qc,
            'Avg': {
                'FP32': avg_ta_fp32,
                'Naive_INT8': avg_ta_naive,
                'Central_INT8': avg_ta_central,
                'Fed_QC_INT8': avg_ta_fed_qc
            }
        }
    }
    
    with open('quantization_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nQuantization experiments complete! Saved to quantization_results.json")

if __name__ == '__main__':
    main()
