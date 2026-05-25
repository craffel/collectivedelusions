import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
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

# Create small subsets for test-time adaptation to make it fast and realistic
# In TTA, we typically adapt on a small unlabeled pool (e.g., 500 samples)
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
        # Add Gaussian noise
        noise = torch.randn_like(images) * 0.4
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == 'blur':
        # Apply Gaussian blur
        return torchvision.transforms.functional.gaussian_blur(images, kernel_size=(5, 5), sigma=(1.5, 1.5))
    elif corruption_type == 'contrast':
        # Reduce contrast
        return torchvision.transforms.functional.adjust_contrast(images, contrast_factor=0.25)
    elif corruption_type == 'rotation':
        # Rotate images by 30 degrees
        return torchvision.transforms.functional.rotate(images, angle=30)
    else:
        return images

# Helper to create a model
def create_resnet18_expert():
    # Load ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final fully connected layer with a 10-class classifier
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# 2. Train Expert Models if checkpoints do not exist
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

epochs = 3 # 3 epochs is sufficient for high accuracy on these datasets

for name, (train_set, test_set) in datasets.items():
    path = expert_paths[name]
    if os.path.exists(path):
        print(f"Found saved expert checkpoint for {name}. Skipping training.")
    else:
        print(f"Training expert model for {name}...")
        model = create_resnet18_expert().to(device)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {100.0*correct/total:.2f}%")
        
        # Save checkpoint (save state dict)
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint to {path}")

# Load the base pre-trained ResNet18 model to define the pre-trained weights
pretrained_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
pretrained_base.fc = nn.Linear(pretrained_base.fc.in_features, 10)
pretrained_state = pretrained_base.state_dict()

# Load experts
experts = {}
for name, path in expert_paths.items():
    model = create_resnet18_expert()
    model.load_state_dict(torch.load(path, map_location=device))
    experts[name] = model.to(device)

# Verify expert accuracies
print("\nEvaluating individual expert models on clean test sets:")
for name, expert in experts.items():
    expert.eval()
    test_loader = DataLoader(datasets[name][1], batch_size=128, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = expert(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Expert {name} clean test accuracy: {100.0*correct/total:.2f}%")


# Define model merging encoder reconstruction
# The encoder is everything in ResNet18 except 'fc'
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

# Evaluation helper
def evaluate_merged_model(lambdas, task_heads, corruption='clean'):
    """
    Evaluates the merged model with coefficients `lambdas` and heads `task_heads`.
    `lambdas` is a list of coefficients [lambda_mnist, lambda_fmnist, lambda_kmnist].
    `task_heads` is a dictionary of fc layers for each task.
    """
    # 1. Reconstruct merged encoder
    merged_encoder = {}
    names = ['mnist', 'fmnist', 'kmnist']
    for k in pretrained_encoder:
        val = pretrained_encoder[k].to(device)
        for i, name in enumerate(names):
            val = val + lambdas[i] * task_vectors[name][k]
        merged_encoder[k] = val
    
    # 2. Evaluate for each task
    accuracies = {}
    for i, name in enumerate(names):
        # Build model for task i
        model = create_resnet18_expert().to(device)
        # Load merged encoder
        model_state = model.state_dict()
        model_state.update(merged_encoder)
        # Load specific task head
        model_state.update(task_heads[name])
        model.load_state_dict(model_state)
        model.eval()
        
        # Test loader
        test_loader = DataLoader(datasets[name][1], batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = apply_corruption(imgs, corruption)
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracies[name] = 100.0 * correct / total
    
    accuracies['avg'] = sum(accuracies[name] for name in names) / len(names)
    return accuracies


# Store original task heads
original_heads = {name: get_fc_params(expert.state_dict()) for name, expert in experts.items()}

# --- TEST-TIME ADAPTATION EXPERIMENTS ---

# Setup loaders for TTA
mnist_tta_loader = DataLoader(mnist_tta_dataset, batch_size=batch_size, shuffle=True)
fmnist_tta_loader = DataLoader(fmnist_tta_dataset, batch_size=batch_size, shuffle=True)
kmnist_tta_loader = DataLoader(kmnist_tta_dataset, batch_size=batch_size, shuffle=True)

# Helper to reconstruct a state dict for a specific task using lambdas and a head
def reconstruct_task_state(lambdas, task_name, head_params):
    merged_state = {}
    names = ['mnist', 'fmnist', 'kmnist']
    # Encoder
    for k in pretrained_encoder:
        val = pretrained_encoder[k].to(device)
        for i, name in enumerate(names):
            val = val + lambdas[i] * task_vectors[name][k]
        merged_state[k] = val
    # Head
    merged_state.update(head_params)
    return merged_state


# METHOD 1: Baseline (No Adaptation, Task Arithmetic with lambda = 0.3)
print("\n--- Running Method 1: Task Arithmetic (No Adaptation) ---")
base_lambdas = [0.3, 0.3, 0.3]
baseline_results = {}
for corruption in ['clean', 'noise', 'blur', 'contrast', 'rotation']:
    res = evaluate_merged_model(base_lambdas, original_heads, corruption)
    baseline_results[corruption] = res
    print(f"TA [{corruption}]: MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | KMNIST: {res['kmnist']:.2f}% | Avg: {res['avg']:.2f}%")


# METHOD 2: AdaMerging (Test-time adaptation of merging coefficients via prediction entropy)
print("\n--- Running Method 2: AdaMerging (Entropy Minimization of Coefficients) ---")

def run_adamerging():
    # Trainable merging coefficients, initialized to 0.3
    lambdas = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
    optimizer = optim.Adam([lambdas], lr=0.001)
    
    # We will simulate 10 steps of adaptation using TTA batches
    steps = 10
    mnist_iter = iter(mnist_tta_loader)
    fmnist_iter = iter(fmnist_tta_loader)
    kmnist_iter = iter(kmnist_tta_loader)
    
    for step in range(steps):
        # We sample a batch from each task
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
        
        # We compute entropy of predictions across all tasks
        for name, imgs in [('mnist', imgs_m), ('fmnist', imgs_f), ('kmnist', imgs_k)]:
            imgs = imgs.to(device)
            # Reconstruct model state dict
            head_params = original_heads[name]
            state = reconstruct_task_state(lambdas, name, head_params)
            
            # Temporary model to evaluate
            model = create_resnet18_expert().to(device)
            model.load_state_dict(state)
            model.eval()
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            # Add small epsilon to prevent log(0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            loss += entropy
        
        loss.backward()
        optimizer.step()
        
    return lambdas.detach().tolist()

adamerging_lambdas = run_adamerging()
print(f"Adapted AdaMerging lambdas: {adamerging_lambdas}")

adamerging_results = {}
for corruption in ['clean', 'noise', 'blur', 'contrast', 'rotation']:
    res = evaluate_merged_model(adamerging_lambdas, original_heads, corruption)
    adamerging_results[corruption] = res
    print(f"AdaMerging [{corruption}]: MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | KMNIST: {res['kmnist']:.2f}% | Avg: {res['avg']:.2f}%")


# METHOD 3: SyMerge (Unsupervised self-labeling of heads + coefficients)
print("\n--- Running Method 3: SyMerge (Self-Labeling of Heads + Coefficients) ---")

def run_symerge():
    # Parameters to optimize: lambdas and task heads
    lambdas = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
    
    # Task heads are initialized from original heads
    adapted_heads = {}
    optimizer_params = [lambdas]
    
    for name in ['mnist', 'fmnist', 'kmnist']:
        # Define fc parameters as trainable variables
        fc_w = nn.Parameter(original_heads[name]['fc.weight'].clone())
        fc_b = nn.Parameter(original_heads[name]['fc.bias'].clone())
        adapted_heads[name] = {'fc.weight': fc_w, 'fc.bias': fc_b}
        optimizer_params.append(fc_w)
        optimizer_params.append(fc_b)
        
    # We use different learning rates: 0.001 for coefficients, 0.01 for heads
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
            # 1. Expert target prediction (no grad)
            with torch.no_grad():
                expert_outputs = experts[name](imgs)
                expert_probs = torch.softmax(expert_outputs, dim=1)
                
            # 2. Merged model prediction
            head_params = adapted_heads[name]
            state = reconstruct_task_state(lambdas, name, head_params)
            
            model = create_resnet18_expert().to(device)
            model.load_state_dict(state)
            # Do NOT use model.eval() so gradients propagate, but set to eval mode behavior
            model.eval()
            
            merged_outputs = model(imgs)
            merged_log_probs = torch.log_softmax(merged_outputs, dim=1)
            
            # KL divergence loss
            loss += criterion(merged_log_probs, expert_probs)
            
        loss.backward()
        optimizer.step()
        
    # Format output
    final_lambdas = lambdas.detach().tolist()
    final_heads = {n: {'fc.weight': adapted_heads[n]['fc.weight'].detach(), 
                       'fc.bias': adapted_heads[n]['fc.bias'].detach()} for n in adapted_heads}
    return final_lambdas, final_heads

symerge_lambdas, symerge_heads = run_symerge()
print(f"Adapted SyMerge lambdas: {symerge_lambdas}")

symerge_results = {}
for corruption in ['clean', 'noise', 'blur', 'contrast', 'rotation']:
    res = evaluate_merged_model(symerge_lambdas, symerge_heads, corruption)
    symerge_results[corruption] = res
    print(f"SyMerge [{corruption}]: MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | KMNIST: {res['kmnist']:.2f}% | Avg: {res['avg']:.2f}%")


# METHOD 4: SAT-SyMerge (Proposed Sharpness-Aware Test-Time Synergistic Merging)
print("\n--- Running Method 4: SAT-SyMerge (Proposed Sharpness-Aware Synergy) ---")

def run_sat_symerge(rho=0.08):
    # Parameters to optimize: lambdas and task heads
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
    
    # List of parameters we are optimizing
    params_list = [lambdas]
    for n in adapted_heads:
        params_list.append(adapted_heads[n]['fc.weight'])
        params_list.append(adapted_heads[n]['fc.bias'])
        
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
            
        # --- SAM Step ---
        
        # 1. Compute loss and gradients
        optimizer.zero_grad()
        loss = 0.0
        for name, imgs in [('mnist', imgs_m), ('fmnist', imgs_f), ('kmnist', imgs_k)]:
            imgs = imgs.to(device)
            with torch.no_grad():
                expert_outputs = experts[name](imgs)
                expert_probs = torch.softmax(expert_outputs, dim=1)
                
            head_params = adapted_heads[name]
            state = reconstruct_task_state(lambdas, name, head_params)
            
            model = create_resnet18_expert().to(device)
            model.load_state_dict(state)
            model.eval()
            
            merged_outputs = model(imgs)
            merged_log_probs = torch.log_softmax(merged_outputs, dim=1)
            loss += criterion(merged_log_probs, expert_probs)
            
        loss.backward()
        
        # 2. Compute perturbation and apply to parameters
        # Calculate l2 norm of gradients across all optimized parameters
        grad_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in params_list if p.grad is not None))
        
        if grad_norm > 0:
            scale = rho / grad_norm
            # Store original values and apply perturbation
            original_values = []
            for p in params_list:
                if p.grad is not None:
                    original_values.append(p.data.clone())
                    p.data.add_(p.grad.data * scale)
                else:
                    original_values.append(None)
                    
            # 3. Compute loss and gradients at the perturbed point
            optimizer.zero_grad()
            perturbed_loss = 0.0
            for name, imgs in [('mnist', imgs_m), ('fmnist', imgs_f), ('kmnist', imgs_k)]:
                imgs = imgs.to(device)
                with torch.no_grad():
                    expert_outputs = experts[name](imgs)
                    expert_probs = torch.softmax(expert_outputs, dim=1)
                    
                head_params = adapted_heads[name]
                state = reconstruct_task_state(lambdas, name, head_params)
                
                model = create_resnet18_expert().to(device)
                model.load_state_dict(state)
                model.eval()
                
                merged_outputs = model(imgs)
                merged_log_probs = torch.log_softmax(merged_outputs, dim=1)
                perturbed_loss += criterion(merged_log_probs, expert_probs)
                
            perturbed_loss.backward()
            
            # 4. Restore original parameter values
            for idx, p in enumerate(params_list):
                if original_values[idx] is not None:
                    p.data.copy_(original_values[idx])
                    
        # 5. Perform the optimizer step using the perturbed gradients
        optimizer.step()
        
    final_lambdas = lambdas.detach().tolist()
    final_heads = {n: {'fc.weight': adapted_heads[n]['fc.weight'].detach(), 
                       'fc.bias': adapted_heads[n]['fc.bias'].detach()} for n in adapted_heads}
    return final_lambdas, final_heads

sat_lambdas, sat_heads = run_sat_symerge()
print(f"Adapted SAT-SyMerge lambdas: {sat_lambdas}")

sat_results = {}
for corruption in ['clean', 'noise', 'blur', 'contrast', 'rotation']:
    res = evaluate_merged_model(sat_lambdas, sat_heads, corruption)
    sat_results[corruption] = res
    print(f"SAT-SyMerge [{corruption}]: MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | KMNIST: {res['kmnist']:.2f}% | Avg: {res['avg']:.2f}%")


# --- SUMMARY OF RESULTS ---
print("\n" + "="*50)
print("             SUMMARY OF EXPERIMENTAL RESULTS")
print("="*50)
print(f"{'Method / Evaluation Set':<30} | {'Clean':<8} | {'Noise':<8} | {'Blur':<8} | {'Contrast':<8} | {'Rotation':<8} | {'OOD Avg':<8}")
print("-"*100)

for name, results in [
    ('Method 1: Task Arithmetic', baseline_results),
    ('Method 2: AdaMerging', adamerging_results),
    ('Method 3: SyMerge', symerge_results),
    ('Method 4: SAT-SyMerge (Ours)', sat_results)
]:
    clean_acc = results['clean']['avg']
    noise_acc = results['noise']['avg']
    blur_acc = results['blur']['avg']
    contrast_acc = results['contrast']['avg']
    rotation_acc = results['rotation']['avg']
    
    ood_avg = (noise_acc + blur_acc + contrast_acc + rotation_acc) / 4
    
    print(f"{name:<30} | {clean_acc:.2f}% | {noise_acc:.2f}% | {blur_acc:.2f}% | {contrast_acc:.2f}% | {rotation_acc:.2f}% | {ood_avg:.2f}%")
print("="*100)

# Save metrics to a file for review
with open('experimental_results.txt', 'w') as f:
    f.write("Method,Clean,Noise,Blur,Contrast,Rotation,OOD_Avg\n")
    for name, results in [
        ('Task Arithmetic', baseline_results),
        ('AdaMerging', adamerging_results),
        ('SyMerge', symerge_results),
        ('SAT-SyMerge', sat_results)
    ]:
        clean_acc = results['clean']['avg']
        noise_acc = results['noise']['avg']
        blur_acc = results['blur']['avg']
        contrast_acc = results['contrast']['avg']
        rotation_acc = results['rotation']['avg']
        ood_avg = (noise_acc + blur_acc + contrast_acc + rotation_acc) / 4
        f.write(f"{name},{clean_acc:.4f},{noise_acc:.4f},{blur_acc:.4f},{contrast_acc:.4f},{rotation_acc:.4f},{ood_avg:.4f}\n")
print("Saved experimental results to experimental_results.txt")
