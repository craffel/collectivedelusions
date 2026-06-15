import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
import json

# Setup seed for strict reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load CLIP model and preprocess
print("Loading pre-trained CLIP model (ViT-B-32)...")
clip_model, _, val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
image_encoder = clip_model.visual.to(device)
feature_dim = 512

# Keep a copy of the pre-trained state dict
pretrained_state_dict = {k: v.cpu().clone() for k, v in image_encoder.state_dict().items()}

# Setup datasets transform
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

datasets_dict = {
    'MNIST': (torchvision.datasets.MNIST, 10),
    'FashionMNIST': (torchvision.datasets.FashionMNIST, 10),
    'CIFAR10': (torchvision.datasets.CIFAR10, 10),
    'SVHN': (torchvision.datasets.SVHN, 10)
}

# Determine parameter group mappings
param_to_group = {}
for k in pretrained_state_dict.keys():
    group_idx = 12
    for i in range(12):
        if f'transformer.resblocks.{i}.' in k:
            group_idx = i
            break
    param_to_group[k] = group_idx

train_loaders = {}
test_loaders = {}

for name, (dataset_cls, num_classes) in datasets_dict.items():
    if name == 'SVHN':
        train_data = dataset_cls(root='~/data', split='train', download=True, transform=transform)
        test_data = dataset_cls(root='~/data', split='test', download=True, transform=transform)
    else:
        train_data = dataset_cls(root='~/data', train=True, download=True, transform=transform)
        test_data = dataset_cls(root='~/data', train=False, download=True, transform=transform)
    
    # Using seed 42 subset
    train_indices = list(range(seed, seed + 512))
    test_indices = list(range(seed, seed + 512))
    
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    test_subset = torch.utils.data.Subset(test_data, test_indices)
    
    train_loaders[name] = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loaders[name] = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

# Fine-tune experts
experts_state_dicts = {}
heads = {}

print("\n--- Fine-tuning Task Experts (Seed 42) ---")
for name, (dataset_cls, num_classes) in datasets_dict.items():
    print(f"Training expert for {name}...")
    head = torch.nn.Linear(feature_dim, num_classes).to(device)
    image_encoder.load_state_dict({k: v.to(device) for k, v in pretrained_state_dict.items()})
    
    optimizer = torch.optim.Adam([
        {'params': image_encoder.parameters(), 'lr': 1e-5},
        {'params': head.parameters(), 'lr': 1e-3}
    ])
    criterion = torch.nn.CrossEntropyLoss()
    
    image_encoder.train()
    head.train()
    for epoch in range(5):
        for imgs, labels in train_loaders[name]:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = image_encoder(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    experts_state_dicts[name] = {k: v.cpu().clone() for k, v in image_encoder.state_dict().items()}
    heads[name] = head.cpu()
    
    image_encoder.eval()
    head.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loaders[name]:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = image_encoder(imgs)
            logits = head(feats)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    print(f"  Expert {name} test accuracy: {100 * correct / total:.2f}%")

# Construct Task Vectors
task_vectors = {}
for name in datasets_dict.keys():
    task_vectors[name] = {}
    for k in pretrained_state_dict.keys():
        if pretrained_state_dict[k].dtype in [torch.int64, torch.uint8]:
            continue
        task_vectors[name][k] = experts_state_dicts[name][k] - pretrained_state_dict[k]

# Cache task vectors and pre-trained dict on GPU
task_names = list(datasets_dict.keys())
task_vectors_gpu = {t_name: {k: v.to(device) for k, v in task_vectors[t_name].items()} for t_name in task_names}
pretrained_gpu = {k: v.to(device) for k, v in pretrained_state_dict.items()}

# Define apply weights function
def apply_merged_weights(encoder, task_coeffs):
    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k not in task_vectors[task_names[0]]:
            new_state_dict[k] = v.to(device)
            continue
        
        merged_update = torch.zeros_like(v)
        for name in task_names:
            coeffs = task_coeffs[name]
            if isinstance(coeffs, list) or isinstance(coeffs, np.ndarray):
                g_idx = param_to_group[k]
                coeff = coeffs[g_idx]
            else:
                coeff = coeffs
            merged_update += coeff * task_vectors[name][k]
        
        new_state_dict[k] = (v + merged_update).to(device)
    encoder.load_state_dict(new_state_dict, strict=False)

def evaluate_merged_model(encoder, task_coeffs):
    apply_merged_weights(encoder, task_coeffs)
    encoder.eval()
    accuracies = {}
    for name, loader in test_loaders.items():
        head = heads[name].to(device).eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = encoder(imgs)
                logits = head(feats)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)
        accuracies[name] = correct / total
    return accuracies


# =====================================================================
# PART 1: Explicit Coefficient Regularization Sweep
# =====================================================================
print("\n--- Pilot Part 1: Explicit Coefficient Regularization Sweep ---")
beta_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
reg_results = []

# Get a validation set (64 images per task)
calib_imgs = {}
for name, loader in train_loaders.items():
    imgs, _ = next(iter(loader))
    calib_imgs[name] = imgs[:64].to(device)

for beta in beta_values:
    print(f"Optimizing with Regularization penalty beta = {beta}...")
    task_coeffs_tensor = torch.full((len(task_names), 13), 0.3, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([task_coeffs_tensor], lr=1e-2)
    
    for step in range(200):
        optimizer.zero_grad()
        differentiable_state_dict = {}
        for k, v in pretrained_gpu.items():
            if k not in task_vectors_gpu[task_names[0]]:
                differentiable_state_dict[k] = v
                continue
            
            g_idx = param_to_group[k]
            merged_update = torch.zeros_like(v)
            for task_idx, t_name in enumerate(task_names):
                coeff = task_coeffs_tensor[task_idx, g_idx]
                merged_update = merged_update + coeff * task_vectors_gpu[t_name][k]
            differentiable_state_dict[k] = v + merged_update
            
        # Entropy Loss
        entropy_loss = 0.0
        for task_idx, t_name in enumerate(task_names):
            inputs = calib_imgs[t_name]
            feats = torch.func.functional_call(image_encoder, differentiable_state_dict, (inputs,))
            head = heads[t_name].to(device)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropy_loss = entropy_loss + entropy
            
        # Regularization Loss: L2 distance to uniform initialization (0.3)
        reg_loss = torch.sum((task_coeffs_tensor - 0.3) ** 2)
        
        total_loss = entropy_loss + beta * reg_loss
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            task_coeffs_tensor.clamp_(0.0, 1.0)
            
    final_coeffs = {}
    for task_idx, t_name in enumerate(task_names):
        final_coeffs[t_name] = task_coeffs_tensor[task_idx].detach().cpu().numpy().tolist()
        
    accs = evaluate_merged_model(image_encoder, final_coeffs)
    avg_acc = np.mean(list(accs.values()))
    print(f"  Beta {beta}: Avg Acc = {100*avg_acc:.2f}% | CIFAR10 = {100*accs['CIFAR10']:.2f}% | SVHN = {100*accs['SVHN']:.2f}%")
    
    reg_results.append({
        'beta': beta,
        'mnist': accs['MNIST'],
        'fashion': accs['FashionMNIST'],
        'cifar10': accs['CIFAR10'],
        'svhn': accs['SVHN'],
        'average': avg_acc,
        'coeffs': final_coeffs
    })

# Save Part 1 Plot
plt.figure(figsize=(8, 5))
betas = [r['beta'] for r in reg_results]
cifar10_accs = [r['cifar10'] * 100 for r in reg_results]
svhn_accs = [r['svhn'] * 100 for r in reg_results]
avg_accs = [r['average'] * 100 for r in reg_results]

plt.plot(betas, avg_accs, '-o', label='Average Accuracy', color='black', linewidth=2.5)
plt.plot(betas, cifar10_accs, '-s', label='CIFAR-10 Accuracy', color='blue', linewidth=2)
plt.plot(betas, svhn_accs, '-^', label='SVHN Accuracy', color='red', linewidth=2)
plt.xlabel('Coefficient Proximity Penalty Weight (Beta)')
plt.ylabel('Test Accuracy (%)')
plt.title('Explicit Coefficient Regularization Sweep (Seed 42)')
plt.xscale('symlog', linthresh=0.01)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('results/fig4_regularization_sweep.png')
plt.close()


# =====================================================================
# PART 2: Calibration Sample Size Sweep (N_cal)
# =====================================================================
print("\n--- Pilot Part 2: Calibration Sample Size Sweep (N_cal) ---")
calib_sizes = [8, 16, 32, 64, 128]
size_results = []

for n_cal in calib_sizes:
    print(f"Optimizing with Calibration Sample Size N_cal = {n_cal}...")
    # Get calibration subset of size n_cal
    temp_calib_imgs = {}
    for name, loader in train_loaders.items():
        imgs, _ = next(iter(loader))
        temp_calib_imgs[name] = imgs[:n_cal].to(device)
        
    task_coeffs_tensor = torch.full((len(task_names), 13), 0.3, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([task_coeffs_tensor], lr=1e-2)
    
    for step in range(200):
        optimizer.zero_grad()
        differentiable_state_dict = {}
        for k, v in pretrained_gpu.items():
            if k not in task_vectors_gpu[task_names[0]]:
                differentiable_state_dict[k] = v
                continue
            
            g_idx = param_to_group[k]
            merged_update = torch.zeros_like(v)
            for task_idx, t_name in enumerate(task_names):
                coeff = task_coeffs_tensor[task_idx, g_idx]
                merged_update = merged_update + coeff * task_vectors_gpu[t_name][k]
            differentiable_state_dict[k] = v + merged_update
            
        loss = 0.0
        for task_idx, t_name in enumerate(task_names):
            inputs = temp_calib_imgs[t_name]
            feats = torch.func.functional_call(image_encoder, differentiable_state_dict, (inputs,))
            head = heads[t_name].to(device)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss + entropy
            
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            task_coeffs_tensor.clamp_(0.0, 1.0)
            
    final_coeffs = {}
    for task_idx, t_name in enumerate(task_names):
        final_coeffs[t_name] = task_coeffs_tensor[task_idx].detach().cpu().numpy().tolist()
        
    accs = evaluate_merged_model(image_encoder, final_coeffs)
    avg_acc = np.mean(list(accs.values()))
    print(f"  N_cal {n_cal}: Avg Acc = {100*avg_acc:.2f}% | CIFAR10 = {100*accs['CIFAR10']:.2f}%")
    
    size_results.append({
        'n_cal': n_cal,
        'average': avg_acc,
        'cifar10': accs['CIFAR10'],
        'svhn': accs['SVHN']
    })

# Save Part 2 Plot
plt.figure(figsize=(8, 5))
sizes = [s['n_cal'] for s in size_results]
size_avg_accs = [s['average'] * 100 for s in size_results]
size_cifar10_accs = [s['cifar10'] * 100 for s in size_results]

plt.plot(sizes, size_avg_accs, '-o', label='Average Test Accuracy', color='green', linewidth=2)
plt.plot(sizes, size_cifar10_accs, '-s', label='CIFAR-10 Test Accuracy', color='blue', linewidth=2)
plt.xlabel('Calibration Sample Size per Task (N_cal)')
plt.ylabel('Test Accuracy (%)')
plt.title('Impact of Calibration Sample Size on Overfitting (Seed 42)')
plt.grid(True, ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('results/fig5_calibration_sweep.png')
plt.close()


# =====================================================================
# PART 3: Coefficient Variance Profiles (Appendix visualization)
# =====================================================================
print("\n--- Pilot Part 3: Generating Coefficient Variance Profiles ---")
# Let's extract unregularized 1+1 ES coefficients (optimized from main run if we can or just quickly run 1+1 ES here)
# Since we want to make it self-contained, let's run 1+1 ES for 200 steps to get the profiles.
es_coeffs = {name: [0.3] * 13 for name in task_names}
best_loss = 100.0
# Define entropy evaluation on calib_imgs
def compute_entropy_loss_local(task_coeffs):
    apply_merged_weights(image_encoder, task_coeffs)
    image_encoder.eval()
    loss = 0.0
    with torch.no_grad():
        for name in task_names:
            feats = image_encoder(calib_imgs[name])
            head = heads[name].to(device)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            loss += entropy
    return loss

best_loss = compute_entropy_loss_local(es_coeffs)
sigma = 0.05
for step in range(300):
    candidate_coeffs = {}
    for name in task_names:
        noise = np.random.normal(0, sigma, 13)
        candidate_coeffs[name] = np.clip(np.array(es_coeffs[name]) + noise, 0.0, 1.0).tolist()
        
    candidate_loss = compute_entropy_loss_local(candidate_coeffs)
    if candidate_loss < best_loss:
        best_loss = candidate_loss
        es_coeffs = candidate_coeffs
        sigma = min(sigma * 1.1, 0.5)
    else:
        sigma = max(sigma * 0.9, 1e-4)

# Load Adam unregularized coeffs from beta=0.0
adam_unreg_coeffs = reg_results[0]['coeffs']

# Let's plot the profile of the 13 layers for CIFAR10 and SVHN under ES and Adam
plt.figure(figsize=(10, 5))
layers = np.arange(13)

plt.plot(layers, es_coeffs['CIFAR10'], '-o', label='1+1 ES (CIFAR-10)', color='lightgreen', linewidth=2)
plt.plot(layers, adam_unreg_coeffs['CIFAR10'], '-s', label='Adam GD (CIFAR-10)', color='darkblue', linewidth=2)
plt.plot(layers, es_coeffs['SVHN'], '--o', label='1+1 ES (SVHN)', color='tomato', linewidth=1.5)
plt.plot(layers, adam_unreg_coeffs['SVHN'], '--s', label='Adam GD (SVHN)', color='darkred', linewidth=1.5)

plt.xlabel('Transformer Layer Index (Group)')
plt.ylabel('Optimized Merging Coefficient')
plt.title('Layer-wise Coefficient Profiles: 1+1 ES vs. Adam GD')
plt.xticks(layers, [f"L{i}" for i in range(12)] + ["Proj"])
plt.grid(True, ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('results/fig6_coefficient_profiles.png')
plt.close()

# Save everything to a JSON file
pilot_stats = {
    'regularization_sweep': [{k: v for k, v in r.items() if k != 'coeffs'} for r in reg_results],
    'calibration_sweep': size_results,
    'es_coeffs_profile': es_coeffs,
    'adam_coeffs_profile': adam_unreg_coeffs
}
with open('results/pilot_metrics.json', 'w') as f:
    json.dump(pilot_stats, f, indent=2)

print("\nPilot study execution complete! All plots saved in 'results/'.")
