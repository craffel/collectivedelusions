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

train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

train_kmnist = torchvision.datasets.KMNIST(root='./data', train=True, download=False, transform=transform)
test_kmnist = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)

# Create small subsets for test-time adaptation
tta_size = 512
batch_size = 32

mnist_tta_dataset = Subset(test_mnist, range(tta_size))
fmnist_tta_dataset = Subset(test_fmnist, range(tta_size))
kmnist_tta_dataset = Subset(test_kmnist, range(tta_size))

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

# Setup loaders for TTA
mnist_tta_loader = DataLoader(mnist_tta_dataset, batch_size=batch_size, shuffle=True)
fmnist_tta_loader = DataLoader(fmnist_tta_dataset, batch_size=batch_size, shuffle=True)
kmnist_tta_loader = DataLoader(kmnist_tta_dataset, batch_size=batch_size, shuffle=True)

def reconstruct_task_state(lambdas, task_name, head_params):
    merged_state = {}
    names = ['mnist', 'fmnist', 'kmnist']
    for k in pretrained_encoder:
        is_bn_buffer = any(sub in k for sub in ['running_mean', 'running_var', 'num_batches_tracked'])
        if is_bn_buffer:
            with torch.no_grad():
                val = pretrained_encoder[k].to(device).clone()
                if 'num_batches_tracked' not in k:
                    for i, name in enumerate(names):
                        val = val + float(lambdas[i].item()) * task_vectors[name][k]
                merged_state[k] = val
        else:
            val = pretrained_encoder[k].to(device)
            for i, name in enumerate(names):
                val = val + lambdas[i] * task_vectors[name][k]
            merged_state[k] = val
    merged_state.update(head_params)
    return merged_state

def run_sat_symerge(rho=0.08, sam_type='global', eta=1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
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
    mnist_iter = iter(mnist_tta_loader)
    fmnist_iter = iter(fmnist_tta_loader)
    kmnist_iter = iter(kmnist_tta_loader)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    tta_model = create_resnet18_expert().to(device)
    tta_model.eval()
    
    for step in range(steps):
        try:
            imgs_m, _ = next(mnist_iter)
        except StopIteration:
            mnist_iter = iter(mnist_tta_loader)
            imgs_m, _ = next(mnist_iter)
            
        try:
            imgs_f, _ = next(fmnist_iter)
        except StopIteration:
            fmnist_iter = iter(fmnist_tta_loader)
            imgs_f, _ = next(fmnist_iter)
            
        try:
            imgs_k, _ = next(kmnist_iter)
        except StopIteration:
            kmnist_iter = iter(kmnist_tta_loader)
            imgs_k, _ = next(kmnist_iter)
            
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
        
        if sam_type == 'none':
            optimizer.step()
            continue
            
        original_values = {}
        
        if sam_type == 'asam':
            # Adaptive SAM (ASAM)
            for n in adapted_heads:
                for p in [adapted_heads[n]['fc.weight'], adapted_heads[n]['fc.bias']]:
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
        
    final_lambdas = lambdas.detach().tolist()
    final_heads = {n: {'fc.weight': adapted_heads[n]['fc.weight'].detach(), 
                       'fc.bias': adapted_heads[n]['fc.bias'].detach()} for n in adapted_heads}
    return final_lambdas, final_heads

print("\n--- Running isolated ASAM (eta=1e-2, rho=0.005) ---")
best_lambdas, best_heads = run_sat_symerge(rho=0.005, sam_type='asam', eta=0.01)
print(f"Adapted ASAM lambdas: {best_lambdas}")
for corruption in ['clean', 'noise', 'blur', 'contrast', 'rotation']:
    res = evaluate_merged_model(best_lambdas, best_heads, corruption)
    print(f"ASAM [{corruption}]: MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | KMNIST: {res['kmnist']:.2f}% | Avg: {res['avg']:.2f}%")
