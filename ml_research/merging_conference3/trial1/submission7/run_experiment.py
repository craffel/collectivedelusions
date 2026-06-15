import os
import time
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
import json

# Define the seeds to run over
seeds = [42, 100, 2026]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load CLIP model and preprocess
print("Loading pre-trained CLIP model (ViT-B-32)...")
clip_model, _, val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
image_encoder = clip_model.visual.to(device)
feature_dim = 512  # ViT-B-32 visual output dim

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

# Determine parameter group mappings (13 groups: 12 transformer blocks + 1 rest)
param_to_group = {}
for k in pretrained_state_dict.keys():
    group_idx = 12  # default
    for i in range(12):
        if f'transformer.resblocks.{i}.' in k:
            group_idx = i
            break
    param_to_group[k] = group_idx

# Helper function to load weights based on coefficients
def apply_merged_weights(encoder, task_coeffs, task_vectors, pretrained_state_dict):
    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k not in task_vectors[list(task_vectors.keys())[0]]:
            new_state_dict[k] = v.to(device)
            continue
        
        merged_update = torch.zeros_like(v)
        for name in task_coeffs.keys():
            coeffs = task_coeffs[name]
            if isinstance(coeffs, list) or isinstance(coeffs, np.ndarray):
                g_idx = param_to_group[k]
                coeff = coeffs[g_idx]
            else:
                coeff = coeffs
            merged_update += coeff * task_vectors[name][k]
        
        new_state_dict[k] = (v + merged_update).to(device)
    encoder.load_state_dict(new_state_dict, strict=False)

# Evaluate merged model helper
def evaluate_merged_model(encoder, task_coeffs, task_vectors, heads_dict, loaders_dict):
    apply_merged_weights(encoder, task_coeffs, task_vectors, pretrained_state_dict)
    encoder.eval()
    accuracies = {}
    for name, loader in loaders_dict.items():
        head = heads_dict[name].to(device).eval()
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

def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = torch.norm(torch.matmul(Y.t(), X), p='fro')**2
    den = torch.norm(torch.matmul(X.t(), X), p='fro') * torch.norm(torch.matmul(Y.t(), Y), p='fro')
    return (num / den).item() if den > 0 else 0.0

# Initialize structures to collect statistics across seeds
seed_results = []

# Loop over random seeds for statistical rigor
for seed in seeds:
    print(f"\n==========================================")
    print(f"RUNNING TRIAL WITH SEED {seed}")
    print(f"==========================================")
    
    # Set seeds for strict reproducibility of this trial
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    train_loaders = {}
    test_loaders = {}
    
    for name, (dataset_cls, num_classes) in datasets_dict.items():
        if name == 'SVHN':
            train_data = dataset_cls(root='~/data', split='train', download=True, transform=transform)
            test_data = dataset_cls(root='~/data', split='test', download=True, transform=transform)
        else:
            train_data = dataset_cls(root='~/data', train=True, download=True, transform=transform)
            test_data = dataset_cls(root='~/data', train=False, download=True, transform=transform)
        
        # Disjoint subsets per seed
        train_indices = list(range(seed, seed + 512))
        test_indices = list(range(seed, seed + 512))
        
        train_subset = torch.utils.data.Subset(train_data, train_indices)
        test_subset = torch.utils.data.Subset(test_data, test_indices)
        
        train_loaders[name] = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loaders[name] = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
        
    # 2. Fine-tune task-specific experts
    experts_state_dicts = {}
    heads = {}
    
    print("\n--- Phase 1: Fine-tuning Task Experts ---")
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
        
        # Evaluate expert
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
        
    # 3. Construct Task Vectors
    task_vectors = {}
    for name in datasets_dict.keys():
        task_vectors[name] = {}
        for k in pretrained_state_dict.keys():
            if pretrained_state_dict[k].dtype in [torch.int64, torch.uint8]:
                continue
            task_vectors[name][k] = experts_state_dicts[name][k] - pretrained_state_dict[k]
            
    # 4. Evaluate Baseline: Task Arithmetic
    print("\n--- Phase 2: Task Arithmetic Baseline ---")
    ta_coeffs = {name: 0.3 for name in datasets_dict.keys()}
    ta_accs = evaluate_merged_model(image_encoder, ta_coeffs, task_vectors, heads, test_loaders)
    for name, acc in ta_accs.items():
        print(f"Task Arithmetic {name} accuracy: {100 * acc:.2f}%")
    print(f"Task Arithmetic Average accuracy: {100 * np.mean(list(ta_accs.values())):.2f}%")
    
    # 5. Setup Calibration set
    calib_imgs = {}
    for name, loader in train_loaders.items():
        imgs, _ = next(iter(loader))
        calib_imgs[name] = imgs[:64].to(device)
        
    def compute_entropy_loss(encoder, task_coeffs):
        apply_merged_weights(encoder, task_coeffs, task_vectors, pretrained_state_dict)
        encoder.eval()
        loss = 0.0
        with torch.no_grad():
            for name in datasets_dict.keys():
                feats = encoder(calib_imgs[name])
                head = heads[name].to(device)
                logits = head(feats)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                loss += entropy
        return loss
        
    # 6. Optimize Coefficients via 1+1 Evolution Strategy
    print("\n--- Phase 3: Optimizing Coefficients via Derivative-Free 1+1 ES ---")
    num_groups = 13
    task_names = list(datasets_dict.keys())
    
    es_coeffs = {name: [0.3] * num_groups for name in task_names}
    best_loss = compute_entropy_loss(image_encoder, es_coeffs)
    print(f"Initial 1+1 ES Entropy Loss: {best_loss:.4f}")
    
    sigma = 0.05
    for step in range(500):
        candidate_coeffs = {}
        for name in task_names:
            noise = np.random.normal(0, sigma, num_groups)
            candidate_coeffs[name] = np.clip(np.array(es_coeffs[name]) + noise, 0.0, 1.0).tolist()
            
        candidate_loss = compute_entropy_loss(image_encoder, candidate_coeffs)
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            es_coeffs = candidate_coeffs
            sigma = min(sigma * 1.1, 0.5)
        else:
            sigma = max(sigma * 0.9, 1e-4)
            
    print(f"Final 1+1 ES Loss after 500 steps: {best_loss:.4f}")
    es_accs = evaluate_merged_model(image_encoder, es_coeffs, task_vectors, heads, test_loaders)
    print(f"AdaMerging (1+1 ES) Average accuracy: {100 * np.mean(list(es_accs.values())):.2f}%")
    
    # 7. Optimize Coefficients via First-Order Gradient Descent (Adam)
    print("\n--- Phase 4: Optimizing Coefficients via First-Order Adam ---")
    task_coeffs_tensor = torch.full((len(task_names), num_groups), 0.3, dtype=torch.float32, device=device, requires_grad=True)
    adam_optimizer = torch.optim.Adam([task_coeffs_tensor], lr=1e-2)
    
    task_vectors_gpu = {t_name: {k: v.to(device) for k, v in task_vectors[t_name].items()} for t_name in task_names}
    pretrained_gpu = {k: v.to(device) for k, v in pretrained_state_dict.items()}
    
    for step in range(200):
        adam_optimizer.zero_grad()
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
            inputs = calib_imgs[t_name]
            feats = torch.func.functional_call(image_encoder, differentiable_state_dict, (inputs,))
            head = heads[t_name].to(device)
            logits = head(feats)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss + entropy
            
        loss.backward()
        adam_optimizer.step()
        
        with torch.no_grad():
            task_coeffs_tensor.clamp_(0.0, 1.0)
            
    adam_coeffs = {}
    for task_idx, t_name in enumerate(task_names):
        adam_coeffs[t_name] = task_coeffs_tensor[task_idx].detach().cpu().numpy().tolist()
        
    print(f"Final First-Order Adam Loss after 200 steps: {loss.item():.4f}")
    adam_accs = evaluate_merged_model(image_encoder, adam_coeffs, task_vectors, heads, test_loaders)
    print(f"AdaMerging (Adam GD) Average accuracy: {100 * np.mean(list(adam_accs.values())):.2f}%")
    
    # 8. Run Sanity-Checking Treatments
    # Diagnostic Treatment 1: Intra-Task Layer Shuffling
    shuffled_es_coeffs = {}
    shuffled_adam_coeffs = {}
    for name in task_names:
        shuffled_es_coeffs[name] = list(es_coeffs[name])
        random.shuffle(shuffled_es_coeffs[name])
        
        shuffled_adam_coeffs[name] = list(adam_coeffs[name])
        random.shuffle(shuffled_adam_coeffs[name])
        
    shuf_es_accs = evaluate_merged_model(image_encoder, shuffled_es_coeffs, task_vectors, heads, test_loaders)
    shuf_adam_accs = evaluate_merged_model(image_encoder, shuffled_adam_coeffs, task_vectors, heads, test_loaders)
    
    # Diagnostic Treatment 2: Spatial Mean Treatment
    mean_es_coeffs = {}
    mean_adam_coeffs = {}
    for name in task_names:
        mean_es_coeffs[name] = float(np.mean(es_coeffs[name]))
        mean_adam_coeffs[name] = float(np.mean(adam_coeffs[name]))
        
    mean_es_accs = evaluate_merged_model(image_encoder, mean_es_coeffs, task_vectors, heads, test_loaders)
    mean_adam_accs = evaluate_merged_model(image_encoder, mean_adam_coeffs, task_vectors, heads, test_loaders)
    
    # Diagnostic Treatment 3: Relative Noise Perturbations
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    noise_es_accs_dict = {}
    noise_adam_accs_dict = {}
    
    for lvl in noise_levels:
        noise_es_coeffs = {}
        noise_adam_coeffs = {}
        for name in task_names:
            noise_es = np.random.normal(0, lvl, num_groups)
            noise_es_coeffs[name] = np.clip(np.array(es_coeffs[name]) + noise_es * np.array(es_coeffs[name]), 0.0, 1.0).tolist()
            
            noise_adam = np.random.normal(0, lvl, num_groups)
            noise_adam_coeffs[name] = np.clip(np.array(adam_coeffs[name]) + noise_adam * np.array(adam_coeffs[name]), 0.0, 1.0).tolist()
            
        n_es_accs = evaluate_merged_model(image_encoder, noise_es_coeffs, task_vectors, heads, test_loaders)
        n_adam_accs = evaluate_merged_model(image_encoder, noise_adam_coeffs, task_vectors, heads, test_loaders)
        
        noise_es_accs_dict[lvl] = float(np.mean(list(n_es_accs.values())))
        noise_adam_accs_dict[lvl] = float(np.mean(list(n_adam_accs.values())))
        
    # 9. CKA Representational Similarity Analysis
    # Get CIFAR10 batch for CKA
    cka_batch, _ = next(iter(test_loaders['CIFAR10']))
    cka_batch = cka_batch.to(device)
    
    # Setup Activation Register hook
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
        
    hook_handle = image_encoder.transformer.resblocks[6].register_forward_hook(get_activation('mid_layer'))
    
    # 1. Expert Activations
    expert_activations = {}
    for name in task_names:
        image_encoder.load_state_dict({k: v.to(device) for k, v in experts_state_dicts[name].items()})
        image_encoder(cka_batch)
        expert_activations[name] = activations['mid_layer'].view(cka_batch.size(0), -1)
        
    # 2. ES Merged Model activations
    apply_merged_weights(image_encoder, es_coeffs, task_vectors, pretrained_state_dict)
    image_encoder(cka_batch)
    es_merged_act = activations['mid_layer'].view(cka_batch.size(0), -1)
    
    # 3. ES Spatial Mean activations
    apply_merged_weights(image_encoder, mean_es_coeffs, task_vectors, pretrained_state_dict)
    image_encoder(cka_batch)
    es_mean_act = activations['mid_layer'].view(cka_batch.size(0), -1)
    
    # 4. Adam Merged Model activations
    apply_merged_weights(image_encoder, adam_coeffs, task_vectors, pretrained_state_dict)
    image_encoder(cka_batch)
    adam_merged_act = activations['mid_layer'].view(cka_batch.size(0), -1)
    
    # 5. Adam Spatial Mean activations
    apply_merged_weights(image_encoder, mean_adam_coeffs, task_vectors, pretrained_state_dict)
    image_encoder(cka_batch)
    adam_mean_act = activations['mid_layer'].view(cka_batch.size(0), -1)
    
    # Remove hook
    hook_handle.remove()
    
    # Compute CKA similarity scores
    es_cka = {}
    es_mean_cka = {}
    adam_cka = {}
    adam_mean_cka = {}
    
    for name in task_names:
        es_cka[name] = linear_cka(expert_activations[name], es_merged_act)
        es_mean_cka[name] = linear_cka(expert_activations[name], es_mean_act)
        adam_cka[name] = linear_cka(expert_activations[name], adam_merged_act)
        adam_mean_cka[name] = linear_cka(expert_activations[name], adam_mean_act)
        
    # Save statistics of this trial
    trial_stats = {
        'seed': seed,
        'task_arithmetic': ta_accs,
        'es_opt': es_accs,
        'adam_opt': adam_accs,
        'shuffled_es': shuf_es_accs,
        'shuffled_adam': shuf_adam_accs,
        'spatial_mean_es': mean_es_accs,
        'spatial_mean_adam': mean_adam_accs,
        'noise_sensitivity_es': noise_es_accs_dict,
        'noise_sensitivity_adam': noise_adam_accs_dict,
        'cka_es': es_cka,
        'cka_mean_es': es_mean_cka,
        'cka_adam': adam_cka,
        'cka_mean_adam': adam_mean_cka
    }
    seed_results.append(trial_stats)

print("\n==========================================")
print("AGGREGATING RESULTS ACROSS SEEDS...")
print("==========================================")

def aggregate_dict(list_of_dicts):
    keys = list_of_dicts[0].keys()
    agg = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts]
        agg[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    return agg

def aggregate_scalar(list_of_dicts):
    # list_of_dicts is a list of scalars if dict has no keys, but let's check
    # e.g., noise sensitivity where keys are float level
    keys = list_of_dicts[0].keys()
    agg = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts]
        agg[float(k)] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    return agg

# Aggregate statistics
ta_stats = aggregate_dict([res['task_arithmetic'] for res in seed_results])
es_opt_stats = aggregate_dict([res['es_opt'] for res in seed_results])
adam_opt_stats = aggregate_dict([res['adam_opt'] for res in seed_results])
shuffled_es_stats = aggregate_dict([res['shuffled_es'] for res in seed_results])
shuffled_adam_stats = aggregate_dict([res['shuffled_adam'] for res in seed_results])
spatial_mean_es_stats = aggregate_dict([res['spatial_mean_es'] for res in seed_results])
spatial_mean_adam_stats = aggregate_dict([res['spatial_mean_adam'] for res in seed_results])

noise_es_stats = aggregate_scalar([res['noise_sensitivity_es'] for res in seed_results])
noise_adam_stats = aggregate_scalar([res['noise_sensitivity_adam'] for res in seed_results])

cka_es_stats = aggregate_dict([res['cka_es'] for res in seed_results])
cka_mean_es_stats = aggregate_dict([res['cka_mean_es'] for res in seed_results])
cka_adam_stats = aggregate_dict([res['cka_adam'] for res in seed_results])
cka_mean_adam_stats = aggregate_dict([res['cka_mean_adam'] for res in seed_results])

# Calculate average accuracies across tasks for each run
avg_ta = [np.mean(list(res['task_arithmetic'].values())) for res in seed_results]
avg_es = [np.mean(list(res['es_opt'].values())) for res in seed_results]
avg_adam = [np.mean(list(res['adam_opt'].values())) for res in seed_results]
avg_shuf_es = [np.mean(list(res['shuffled_es'].values())) for res in seed_results]
avg_shuf_adam = [np.mean(list(res['shuffled_adam'].values())) for res in seed_results]
avg_mean_es = [np.mean(list(res['spatial_mean_es'].values())) for res in seed_results]
avg_mean_adam = [np.mean(list(res['spatial_mean_adam'].values())) for res in seed_results]

print(f"\nTask Arithmetic Average Accuracy: {100 * np.mean(avg_ta):.2f}% +/- {100 * np.std(avg_ta):.2f}%")
print(f"AdaMerging (1+1 ES) Average Accuracy: {100 * np.mean(avg_es):.2f}% +/- {100 * np.std(avg_es):.2f}%")
print(f"AdaMerging (Adam GD) Average Accuracy: {100 * np.mean(avg_adam):.2f}% +/- {100 * np.std(avg_adam):.2f}%")
print(f"Spatially Averaged (1+1 ES) Average Accuracy: {100 * np.mean(avg_mean_es):.2f}% +/- {100 * np.std(avg_mean_es):.2f}%")
print(f"Spatially Averaged (Adam GD) Average Accuracy: {100 * np.mean(avg_mean_adam):.2f}% +/- {100 * np.std(avg_mean_adam):.2f}%")

# Generate Plots and Save results
os.makedirs('results', exist_ok=True)

# Plot 1: Accuracy under different treatments with Error Bars (Task Average)
plt.figure(figsize=(11, 6))
methods = [
    'Task Arithmetic', 
    'AdaMerging\n(1+1 ES)', 
    'AdaMerging\n(Adam GD)', 
    'Spatial Mean\n(1+1 ES)', 
    'Spatial Mean\n(Adam GD)', 
    'Shuffle\n(1+1 ES)', 
    'Shuffle\n(Adam GD)'
]
means = [
    np.mean(avg_ta), 
    np.mean(avg_es), 
    np.mean(avg_adam), 
    np.mean(avg_mean_es), 
    np.mean(avg_mean_adam), 
    np.mean(avg_shuf_es), 
    np.mean(avg_shuf_adam)
]
stds = [
    np.std(avg_ta), 
    np.std(avg_es), 
    np.std(avg_adam), 
    np.std(avg_mean_es), 
    np.std(avg_mean_adam), 
    np.std(avg_shuf_es), 
    np.std(avg_shuf_adam)
]

colors = ['blue', 'green', 'darkgreen', 'purple', 'darkviolet', 'orange', 'orangered']
bars = plt.bar(methods, [m * 100 for m in means], yerr=[s * 100 for s in stds], capsize=5, color=colors)
plt.ylabel('Average Accuracy (%)')
plt.title('Sanity Check: Model Merging Accuracies under Treatments (Aggregated over 3 Seeds)')
plt.ylim(0, 100)
for idx, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2.0, means[idx] * 100 + 1.5, f"{means[idx]*100:.2f}%", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('results/fig1_treatments.png')
plt.close()

# Plot 2: Noise Sensitivity Curve with Shading
plt.figure(figsize=(8, 5))
levels_axis = [0.0] + noise_levels

es_noise_means = [np.mean(avg_es)] + [noise_es_stats[l]['mean'] for l in noise_levels]
es_noise_stds = [np.std(avg_es)] + [noise_es_stats[l]['std'] for l in noise_levels]

adam_noise_means = [np.mean(avg_adam)] + [noise_adam_stats[l]['mean'] for l in noise_levels]
adam_noise_stds = [np.std(avg_adam)] + [noise_adam_stats[l]['std'] for l in noise_levels]

plt.errorbar(levels_axis, [m * 100 for m in es_noise_means], yerr=[s * 100 for s in es_noise_stds], fmt='-o', color='green', capsize=4, label='1+1 ES Optimized', linewidth=2)
plt.errorbar(levels_axis, [m * 100 for m in adam_noise_means], yerr=[s * 100 for s in adam_noise_stds], fmt='-s', color='blue', capsize=4, label='Adam GD Optimized', linewidth=2)

plt.xlabel('Relative Noise Level ($\sigma / \mu$)')
plt.ylabel('Average Accuracy (%)')
plt.title('Noise Sensitivity of Optimized Merging Coefficients (3 Seeds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/fig2_noise_sensitivity.png')
plt.close()

# Plot 3: Representational Similarity (CKA) with standard deviations
plt.figure(figsize=(10, 6))
task_list = list(datasets_dict.keys())
x_axis = np.arange(len(task_list))
width = 0.2

es_cka_m = [cka_es_stats[name]['mean'] for name in task_list]
es_cka_s = [cka_es_stats[name]['std'] for name in task_list]

es_mean_cka_m = [cka_mean_es_stats[name]['mean'] for name in task_list]
es_mean_cka_s = [cka_mean_es_stats[name]['std'] for name in task_list]

adam_cka_m = [cka_adam_stats[name]['mean'] for name in task_list]
adam_cka_s = [cka_adam_stats[name]['std'] for name in task_list]

adam_mean_cka_m = [cka_mean_adam_stats[name]['mean'] for name in task_list]
adam_mean_cka_s = [cka_mean_adam_stats[name]['std'] for name in task_list]

plt.bar(x_axis - 1.5*width, es_cka_m, width, yerr=es_cka_s, capsize=3, label='Optimized (1+1 ES)', color='lightgreen')
plt.bar(x_axis - 0.5*width, es_mean_cka_m, width, yerr=es_mean_cka_s, capsize=3, label='Spatial Mean (1+1 ES)', color='forestgreen')
plt.bar(x_axis + 0.5*width, adam_cka_m, width, yerr=adam_cka_s, capsize=3, label='Optimized (Adam GD)', color='lightblue')
plt.bar(x_axis + 1.5*width, adam_mean_cka_m, width, yerr=adam_mean_cka_s, capsize=3, label='Spatial Mean (Adam GD)', color='darkblue')

plt.xticks(x_axis, task_list)
plt.ylabel('Linear CKA Similarity with Expert')
plt.title('Representational Alignment in Intermediate Visual Layer (Block 6) (3 Seeds)')
plt.ylim(0.95, 1.0)
plt.legend(loc='lower left')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('results/fig3_cka.png')
plt.close()

# Write JSON report data with full statistical report
stats_report = {
    'task_arithmetic': ta_stats,
    'es_opt': es_opt_stats,
    'adam_opt': adam_opt_stats,
    'shuffled_es': shuffled_es_stats,
    'shuffled_adam': shuffled_adam_stats,
    'spatial_mean_es': spatial_mean_es_stats,
    'spatial_mean_adam': spatial_mean_adam_stats,
    'noise_sensitivity_es': {str(k): v for k, v in noise_es_stats.items()},
    'noise_sensitivity_adam': {str(k): v for k, v in noise_adam_stats.items()},
    'cka_es': cka_es_stats,
    'cka_mean_es': cka_mean_es_stats,
    'cka_adam': cka_adam_stats,
    'cka_mean_adam': cka_mean_adam_stats,
    'averages': {
        'task_arithmetic': {'mean': float(np.mean(avg_ta)), 'std': float(np.std(avg_ta))},
        'es_opt': {'mean': float(np.mean(avg_es)), 'std': float(np.std(avg_es))},
        'adam_opt': {'mean': float(np.mean(avg_adam)), 'std': float(np.std(avg_adam))},
        'shuffled_es': {'mean': float(np.mean(avg_shuf_es)), 'std': float(np.std(avg_shuf_es))},
        'shuffled_adam': {'mean': float(np.mean(avg_shuf_adam)), 'std': float(np.std(avg_shuf_adam))},
        'spatial_mean_es': {'mean': float(np.mean(avg_mean_es)), 'std': float(np.std(avg_mean_es))},
        'spatial_mean_adam': {'mean': float(np.mean(avg_mean_adam)), 'std': float(np.std(avg_mean_adam))}
    }
}

with open('results/metrics.json', 'w') as f:
    json.dump(stats_report, f, indent=2)

print("\nAll multi-seed experiments and analyses completed successfully!")
