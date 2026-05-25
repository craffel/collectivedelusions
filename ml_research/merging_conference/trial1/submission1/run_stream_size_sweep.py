import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Handle potential cuDNN initialization errors on the cluster
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Dataset preparation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading datasets...")
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

train_kmnist = torchvision.datasets.KMNIST(root='./data', train=True, download=False, transform=transform)
test_kmnist = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)

# Define corruptions for out-of-distribution evaluation
def apply_corruption(images, corruption_type):
    if corruption_type == 'clean':
        return images
    elif corruption_type == 'noise':
        noise = torch.randn_like(images) * 0.4
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == 'blur':
        return torchvision.transforms.functional.gaussian_blur(images, kernel_size=(5, 5), sigma=(1.5, 1.5))
    elif corruption_type == 'contrast':
        return torchvision.transforms.functional.adjust_contrast(images, contrast_factor=0.25)
    elif corruption_type == 'rotation':
        return torchvision.transforms.functional.rotate(images, angle=30)
    else:
        return images

# Helper to create a model
def create_resnet18_expert():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

expert_paths = {
    'mnist': 'expert_mnist.pt',
    'fmnist': 'expert_fmnist.pt',
    'kmnist': 'expert_kmnist.pt'
}

datasets = {
    'mnist': (train_mnist, test_mnist),
    'fmnist': (train_fmnist, test_fmnist),
    'kmnist': (train_kmnist, test_kmnist)
}

# Load base pre-trained ResNet18
pretrained_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
pretrained_base.fc = nn.Linear(pretrained_base.fc.in_features, 10)
pretrained_state = pretrained_base.state_dict()

# Load experts
experts = {}
for name, path in expert_paths.items():
    model = create_resnet18_expert()
    model.load_state_dict(torch.load(path, map_location=device))
    experts[name] = model.to(device)

# Extract parameters
def get_encoder_params(state_dict):
    return {k: v.clone() for k, v in state_dict.items() if not k.startswith('fc')}

def get_fc_params(state_dict):
    return {k: v.clone() for k, v in state_dict.items() if k.startswith('fc')}

# Calculate task vectors for encoder
pretrained_encoder = get_encoder_params(pretrained_state)
task_vectors = {}
for name, expert in experts.items():
    expert_encoder = get_encoder_params(expert.state_dict())
    task_vectors[name] = {k: expert_encoder[k] - pretrained_encoder[k].to(device) for k in pretrained_encoder}

# Store original task heads
original_heads = {name: get_fc_params(expert.state_dict()) for name, expert in experts.items()}

# Pre-allocated model for merged evaluations
eval_model = create_resnet18_expert().to(device)
eval_model.eval()

# Evaluation helper (optimized)
def evaluate_merged_model(lambdas, task_heads, corruption='clean'):
    merged_encoder = {}
    names = ['mnist', 'fmnist', 'kmnist']
    for k in pretrained_encoder:
        val = pretrained_encoder[k].to(device)
        for i, name in enumerate(names):
            val = val + lambdas[i] * task_vectors[name][k]
        merged_encoder[k] = val
    
    accuracies = {}
    for i, name in enumerate(names):
        model_state = copy.deepcopy(merged_encoder)
        model_state.update(task_heads[name])
        
        eval_model.load_state_dict(model_state)
        
        test_loader = DataLoader(datasets[name][1], batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = apply_corruption(imgs, corruption)
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = eval_model(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracies[name] = 100.0 * correct / total
    
    accuracies['avg'] = sum(accuracies[name] for name in names) / len(names)
    return accuracies

def reconstruct_task_state(lambdas, task_name, head_params):
    merged_state = {}
    names = ['mnist', 'fmnist', 'kmnist']
    for k in pretrained_encoder:
        # Check if this is a BN buffer
        is_bn_buffer = any(sub in k for sub in ['running_mean', 'running_var', 'num_batches_tracked'])
        if is_bn_buffer:
            # Merge without tracking gradients
            with torch.no_grad():
                val = pretrained_encoder[k].to(device).clone()
                # If it's num_batches_tracked, it is a long/int tensor, so don't multiply by lambdas
                if 'num_batches_tracked' not in k:
                    for i, name in enumerate(names):
                        # Use float value of lambdas to prevent grad tracking
                        val = val + float(lambdas[i].item()) * task_vectors[name][k]
                merged_state[k] = val
        else:
            # Regular parameter, track gradients!
            val = pretrained_encoder[k].to(device)
            for i, name in enumerate(names):
                val = val + lambdas[i] * task_vectors[name][k]
            merged_state[k] = val
    merged_state.update(head_params)
    return merged_state

# Adaptive TTA run functions
def run_tta_for_size(n, method='symerge', rho=0.005, eta=0.01):
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataloaders for specific size n
    mnist_sub = Subset(test_mnist, range(n))
    fmnist_sub = Subset(test_fmnist, range(n))
    kmnist_sub = Subset(test_kmnist, range(n))
    
    batch_size = 32
    mnist_loader = DataLoader(mnist_sub, batch_size=batch_size, shuffle=True)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=batch_size, shuffle=True)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=batch_size, shuffle=True)
    
    lambdas = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
    adapted_heads = {}
    for name in ['mnist', 'fmnist', 'kmnist']:
        fc_w = nn.Parameter(original_heads[name]['fc.weight'].clone())
        fc_b = nn.Parameter(original_heads[name]['fc.bias'].clone())
        adapted_heads[name] = {'fc.weight': fc_w, 'fc.bias': fc_b}
        
    optimizer = optim.Adam([
        {'params': [lambdas], 'lr': 0.001},
        {'params': [adapted_heads[n]['fc.weight'] for n in adapted_heads] + 
                   [adapted_heads[n]['fc.bias'] for n in adapted_heads], 'lr': 0.01}
    ])
    
    steps = 10
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    tta_model = create_resnet18_expert().to(device)
    tta_model.eval()
    
    final_local_loss = 0.0
    
    for step in range(steps):
        try:
            imgs_m, _ = next(mnist_iter)
        except StopIteration:
            mnist_iter = iter(mnist_loader)
            imgs_m, _ = next(mnist_iter)
            
        try:
            imgs_f, _ = next(fmnist_iter)
        except StopIteration:
            fmnist_iter = iter(fmnist_loader)
            imgs_f, _ = next(fmnist_iter)
            
        try:
            imgs_k, _ = next(kmnist_iter)
        except StopIteration:
            kmnist_iter = iter(kmnist_loader)
            imgs_k, _ = next(kmnist_iter)
            
        # 1. First backward pass
        optimizer.zero_grad()
        loss = 0.0
        for name, imgs in [('mnist', imgs_m), ('fmnist', imgs_f), ('kmnist', imgs_k)]:
            imgs = imgs.to(device)
            with torch.no_grad():
                expert_outputs = experts[name](imgs)
                expert_probs = torch.softmax(expert_outputs, dim=1)
                
            head_params = adapted_heads[name]
            state = reconstruct_task_state(lambdas, name, head_params)
            
            merged_outputs = functional_call(tta_model, state, imgs)
            merged_log_probs = torch.log_softmax(merged_outputs, dim=1)
            loss += criterion(merged_log_probs, expert_probs)
            
        loss.backward()
        
        # Perturbation and second backward pass
        if method == 'asam':
            original_values = {}
            for n_key in adapted_heads:
                for p in [adapted_heads[n_key]['fc.weight'], adapted_heads[n_key]['fc.bias']]:
                    if p.grad is not None:
                        w_scale = torch.abs(p.data) + eta
                        g_scaled = w_scale * p.grad.data
                        g_norm = g_scaled.norm(2)
                        if g_norm > 0:
                            eps = (rho / g_norm) * (w_scale ** 2) * p.grad.data
                            original_values[p] = p.data.clone()
                            p.data.add_(eps)
                            
            if len(original_values) > 0:
                optimizer.zero_grad()
                perturbed_loss = 0.0
                for name, imgs in [('mnist', imgs_m), ('fmnist', imgs_f), ('kmnist', imgs_k)]:
                    imgs = imgs.to(device)
                    with torch.no_grad():
                        expert_outputs = experts[name](imgs)
                        expert_probs = torch.softmax(expert_outputs, dim=1)
                        
                    head_params = adapted_heads[name]
                    state = reconstruct_task_state(lambdas, name, head_params)
                    
                    merged_outputs = functional_call(tta_model, state, imgs)
                    merged_log_probs = torch.log_softmax(merged_outputs, dim=1)
                    perturbed_loss += criterion(merged_log_probs, expert_probs)
                    
                perturbed_loss.backward()
                
                for p, val in original_values.items():
                    p.data.copy_(val)
                    
        optimizer.step()
        if step == steps - 1:
            final_local_loss = loss.item()
        
    final_lambdas = lambdas.detach().tolist()
    final_heads = {n: {'fc.weight': adapted_heads[n]['fc.weight'].detach(), 
                       'fc.bias': adapted_heads[n]['fc.bias'].detach()} for n in adapted_heads}
    return final_lambdas, final_heads, final_local_loss

stream_sizes = [64, 128, 256, 512, 1024, 2048]
results_dict = {'symerge': {}, 'asam_0.01': {}, 'asam_0.05': {}}

print("\nStarting Stream Size Sweep...")
for size in stream_sizes:
    print(f"\n--- Stream Size n = {size} ---")
    
    # 1. Run SyMerge
    sm_lambdas, sm_heads, sm_loss = run_tta_for_size(size, method='symerge')
    sm_clean = evaluate_merged_model(sm_lambdas, sm_heads, 'clean')['avg']
    sm_ood_avgs = []
    for corruption in ['noise', 'blur', 'contrast', 'rotation']:
        sm_ood_avgs.append(evaluate_merged_model(sm_lambdas, sm_heads, corruption)['avg'])
    sm_ood = sum(sm_ood_avgs) / len(sm_ood_avgs)
    results_dict['symerge'][size] = {'clean': sm_clean, 'ood': sm_ood, 'local_loss': sm_loss}
    print(f"SyMerge [n={size}]: Loss: {sm_loss:.4f} | Clean: {sm_clean:.2f}% | OOD Avg: {sm_ood:.2f}%")
    
    # 2. Run ASAM-SyMerge (rho=0.01)
    as1_lambdas, as1_heads, as1_loss = run_tta_for_size(size, method='asam', rho=0.01, eta=0.01)
    as1_clean = evaluate_merged_model(as1_lambdas, as1_heads, 'clean')['avg']
    as1_ood_avgs = []
    for corruption in ['noise', 'blur', 'contrast', 'rotation']:
        as1_ood_avgs.append(evaluate_merged_model(as1_lambdas, as1_heads, corruption)['avg'])
    as1_ood = sum(as1_ood_avgs) / len(as1_ood_avgs)
    results_dict['asam_0.01'][size] = {'clean': as1_clean, 'ood': as1_ood, 'local_loss': as1_loss}
    print(f"ASAM (rho=0.01) [n={size}]: Loss: {as1_loss:.4f} | Clean: {as1_clean:.2f}% | OOD Avg: {as1_ood:.2f}%")

    # 3. Run ASAM-SyMerge (rho=0.05)
    as5_lambdas, as5_heads, as5_loss = run_tta_for_size(size, method='asam', rho=0.05, eta=0.01)
    as5_clean = evaluate_merged_model(as5_lambdas, as5_heads, 'clean')['avg']
    as5_ood_avgs = []
    for corruption in ['noise', 'blur', 'contrast', 'rotation']:
        as5_ood_avgs.append(evaluate_merged_model(as5_lambdas, as5_heads, corruption)['avg'])
    as5_ood = sum(as5_ood_avgs) / len(as5_ood_avgs)
    results_dict['asam_0.05'][size] = {'clean': as5_clean, 'ood': as5_ood, 'local_loss': as5_loss}
    print(f"ASAM (rho=0.05) [n={size}]: Loss: {as5_loss:.4f} | Clean: {as5_clean:.2f}% | OOD Avg: {as5_ood:.2f}%")

# Save and print results table
with open('stream_size_sweep_results.txt', 'w') as f:
    f.write("Stream Size n,SyMerge Loss,ASAM(0.01) Loss,ASAM(0.05) Loss,SyMerge Clean,ASAM(0.01) Clean,ASAM(0.05) Clean,SyMerge OOD,ASAM(0.01) OOD,ASAM(0.05) OOD\n")
    for size in stream_sizes:
        s_loss = results_dict['symerge'][size]['local_loss']
        a1_loss = results_dict['asam_0.01'][size]['local_loss']
        a5_loss = results_dict['asam_0.05'][size]['local_loss']
        
        s_cl = results_dict['symerge'][size]['clean']
        a1_cl = results_dict['asam_0.01'][size]['clean']
        a5_cl = results_dict['asam_0.05'][size]['clean']
        
        s_od = results_dict['symerge'][size]['ood']
        a1_od = results_dict['asam_0.01'][size]['ood']
        a5_od = results_dict['asam_0.05'][size]['ood']
        
        f.write(f"{size},{s_loss:.6f},{a1_loss:.6f},{a5_loss:.6f},{s_cl:.4f},{a1_cl:.4f},{a5_cl:.4f},{s_od:.4f},{a1_od:.4f},{a5_od:.4f}\n")

print("\n" + "="*95)
print("                                 STREAM SIZE SWEEP RESULTS")
print("="*95)
print(f"{'Stream Size n':<15} | {'SyMerge Loss':<12} | {'ASAM(0.05) Loss':<15} | {'SyMerge Clean':<13} | {'ASAM(0.05) Clean':<16} | {'SyMerge OOD':<11} | {'ASAM(0.05) OOD':<14}")
print("-"*95)
for size in stream_sizes:
    s_loss = results_dict['symerge'][size]['local_loss']
    a5_loss = results_dict['asam_0.05'][size]['local_loss']
    
    s_cl = results_dict['symerge'][size]['clean']
    a5_cl = results_dict['asam_0.05'][size]['clean']
    
    s_od = results_dict['symerge'][size]['ood']
    a5_od = results_dict['asam_0.05'][size]['ood']
    
    print(f"{size:<15} | {s_loss:.4f}       | {a5_loss:.4f}         | {s_cl:.2f}%        | {a5_cl:.2f}%          | {s_od:.2f}%     | {a5_od:.2f}%")
print("="*95)
print("Saved stream_size_sweep_results.txt")
