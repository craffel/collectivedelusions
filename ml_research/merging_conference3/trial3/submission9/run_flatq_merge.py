import os
import random
import json
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import timm
import matplotlib.pyplot as plt

# Define task datasets
TASKS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']

def check_cuda_working():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1).cuda()
        return True
    except Exception as e:
        print(f"CUDA is available but failed to initialize: {e}. Falling back to CPU.")
        return False

DEVICE = 'cuda' if check_cuda_working() else 'cpu'
print(f"Using device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def get_layer_group(param_name):
    if 'patch_embed' in param_name or 'cls_token' in param_name or 'pos_embed' in param_name or 'norm_pre' in param_name:
        return 0
    elif 'blocks' in param_name:
        parts = param_name.split('.')
        block_idx = int(parts[1])
        return block_idx + 1 # Block 0 maps to Group 1, Block 11 to Group 12
    elif 'norm.' in param_name:
        return 13
    else:
        return -1 # Not merged (e.g., classification head)

def get_raw_dataset(task_name, split='train'):
    if task_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    if task_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(root='./data', split=('train' if split=='train' else 'test'), download=False, transform=transform)
    else:
        raise ValueError(f"Unknown task {task_name}")
    return dataset

def get_cached_dataset(task_name, split, size, seed):
    print(f"Caching dataset {task_name} (split={split}, size={size}, seed={seed})...")
    dataset = get_raw_dataset(task_name, split)
    if size is not None and size < len(dataset):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
        dataset = Subset(dataset, indices)
    
    # Load everything into memory in a single batch
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)
    x_all, y_all = next(iter(loader))
    return TensorDataset(x_all, y_all)

def sam_step(model, x, y, optimizer, rho, criterion):
    # 1. First forward-backward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    
    # 2. Perturb weights
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    if grad_norm > 1e-8:
        scale = rho / grad_norm
        epsilon_dict = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                e = p.grad.data * scale
                p.data.add_(e)
                epsilon_dict[n] = e
                
        # 3. Second forward-backward pass
        optimizer.zero_grad()
        outputs_perturbed = model(x)
        loss_perturbed = criterion(outputs_perturbed, y)
        loss_perturbed.backward()
        
        # 4. Restore weights
        for n, p in model.named_parameters():
            if n in epsilon_dict:
                p.data.sub_(epsilon_dict[n])
                
    # 5. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def train_expert(task_name, seed, rho, train_dataset, epochs=10, batch_size=64):
    set_seed(seed)
    print(f"Training expert for task {task_name} (seed={seed}, rho={rho}) on {DEVICE}...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(model.num_features, 10)
    model = model.to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if rho > 0:
                sam_step(model, x, y, optimizer, rho, criterion)
            else:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
    save_path = f"checkpoints/expert_{task_name}_seed{seed}_rho{rho}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Finished. Saved to {save_path}")

def softmax_entropy(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)

def quantize_ste(w, b):
    qmin = -(2**(b-1))
    qmax = 2**(b-1) - 1
    
    dim = list(range(1, w.ndim))
    max_val = w.abs()
    for d in dim:
        max_val = max_val.max(dim=d, keepdim=True)[0]
    scale = max_val / qmax
    scale = torch.clamp(scale, min=1e-8)
    
    scaled_w = w / scale
    rounded_w = torch.clamp(torch.round(scaled_w), qmin, qmax)
    quant_w = (rounded_w - scaled_w).detach() + scaled_w
    return quant_w * scale

def update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits):
    lambdas = torch.clamp(lambdas_raw, min=0.0, max=1.0) # shape [14, 4]
    for n, p_base in base_dict.items():
        l = get_layer_group(n)
        if l >= 0:
            w_merged = p_base + sum(lambdas[l, i] * task_vectors[i][n] for i in range(4))
            if quantize and 'bias' not in n:
                w_quant = quantize_ste(w_merged, bits)
            else:
                w_quant = w_merged
            set_attr(active_model, n.split('.'), w_quant)

def test_time_optimize(active_model, base_dict, task_vectors, expert_heads, quantize, bits, calib_loaders, steps=40, lr=1e-3):
    prior = 0.3
    lambdas_raw = torch.nn.Parameter(torch.ones(14, 4, device=DEVICE) * prior)
    optimizer = torch.optim.Adam([lambdas_raw], lr=lr)
    
    calib_batches = {}
    for task_name, loader in calib_loaders.items():
        x, _ = next(iter(loader))
        calib_batches[task_name] = x.to(DEVICE)
        
    for step in range(steps):
        optimizer.zero_grad()
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits)
        
        loss = 0.0
        for k, task_name in enumerate(calib_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            
            x = calib_batches[task_name]
            outputs = active_model(x)
            entropy = softmax_entropy(outputs).mean(0)
            loss += entropy
            
        loss.backward()
        optimizer.step()
        
    return lambdas_raw.detach().clone()

def evaluate_model(active_model, base_dict, task_vectors, lambdas_raw, expert_heads, quantize, bits, test_loaders):
    active_model.eval()
    task_accs = {}
    with torch.no_grad():
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits)
        for k, task_name in enumerate(test_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            
            correct = 0
            total = 0
            for x, y in test_loaders[task_name]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = active_model(x)
                preds = outputs.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            acc = correct / total * 100.0
            task_accs[task_name] = acc
    return task_accs

def profile_curvature(active_model, base_dict, task_vectors, lambdas_opt, expert_heads, quantize, bits, calib_loaders, sigma_grid):
    active_model.eval()
    
    calib_batches = {}
    for task_name, loader in calib_loaders.items():
        x, _ = next(iter(loader))
        calib_batches[task_name] = x.to(DEVICE)
        
    # L_base
    with torch.no_grad():
        update_model_weights(active_model, base_dict, task_vectors, lambdas_opt, quantize, bits)
        L_base = 0.0
        for k, task_name in enumerate(calib_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            x = calib_batches[task_name]
            outputs = active_model(x)
            L_base += softmax_entropy(outputs).mean(0).item()
            
    curvature_results = {}
    for sigma in sigma_grid:
        diffs = []
        for run in range(10):
            delta = torch.randn_like(lambdas_opt) * sigma
            lambdas_perturbed = lambdas_opt + delta
            
            with torch.no_grad():
                update_model_weights(active_model, base_dict, task_vectors, lambdas_perturbed, quantize, bits)
                L_perturbed = 0.0
                for k, task_name in enumerate(calib_loaders.keys()):
                    active_model.head.weight.data.copy_(expert_heads[k]['weight'])
                    active_model.head.bias.data.copy_(expert_heads[k]['bias'])
                    x = calib_batches[task_name]
                    outputs = active_model(x)
                    L_perturbed += softmax_entropy(outputs).mean(0).item()
                    
            diffs.append(L_perturbed - L_base)
        curvature_results[str(sigma)] = float(np.mean(diffs))
        
    return curvature_results

def run_experiments_for_seed(seed, test_loaders, calib_loaders):
    print(f"\n==========================================")
    print(f"RUNNING EXPERIMENTS FOR SEED {seed}")
    print(f"==========================================\n")
    
    radii = [0.0, 0.01, 0.05, 0.1, 0.2]
    precisions = [8, 4]
    
    # Cache training dataset for this seed (size 512)
    train_datasets = {}
    for task in TASKS:
        train_datasets[task] = get_cached_dataset(task, split='train', size=512, seed=seed)
        
    # 1. Train experts (if they don't exist yet)
    for rho in radii:
        for task in TASKS:
            expert_file = f"checkpoints/expert_{task}_seed{seed}_rho{rho}.pt"
            if not os.path.exists(expert_file):
                train_expert(task, seed, rho, train_datasets[task], epochs=10, batch_size=64)
                
    # Load base model structure
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    base_model = base_model.to(DEVICE)
    active_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    active_model = active_model.to(DEVICE)
    active_model.head = torch.nn.Linear(192, 10).to(DEVICE)
    
    # Run make functional on active_model
    for name, _ in list(active_model.named_parameters()):
        if get_layer_group(name) >= 0:
            del_attr(active_model, name.split('.'))
            
    base_dict = {n: p.clone().detach().to(DEVICE) for n, p in base_model.named_parameters()}
    
    seed_results = {}
    
    # Curvature profile grid
    sigma_grid = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    curvature_profiles = {}
    
    for rho in radii:
        print(f"\n--- Loading Experts for SAM radius rho = {rho} ---")
        expert_heads = []
        expert_dicts = []
        for task in TASKS:
            expert_file = f"checkpoints/expert_{task}_seed{seed}_rho{rho}.pt"
            sd = torch.load(expert_file, map_location=DEVICE)
            expert_dicts.append(sd)
            # Extract head
            head_w = sd['head.weight'].clone().detach().cpu()
            head_b = sd['head.bias'].clone().detach().cpu()
            expert_heads.append({'weight': head_w, 'bias': head_b})
            
        task_vectors = []
        for k in range(4):
            tv = {}
            for n in base_dict:
                if get_layer_group(n) >= 0:
                    tv[n] = expert_dicts[k][n] - base_dict[n]
            task_vectors.append(tv)
            
        rho_results = {}
        
        for bits in precisions:
            print(f"Evaluating precision: {bits}-bit weight quantization")
            
            # Technique 1: FlatQ-Merge (Optimizing coefficients on quantized model)
            print("Running FlatQ-Merge optimization...")
            lambdas_opt = test_time_optimize(active_model, base_dict, task_vectors, expert_heads, quantize=True, bits=bits, calib_loaders=calib_loaders)
            accs_flatq = evaluate_model(active_model, base_dict, task_vectors, lambdas_opt, expert_heads, quantize=True, bits=bits, test_loaders=test_loaders)
            avg_acc_flatq = np.mean(list(accs_flatq.values()))
            
            # Curvature profiling of optimized coefficient space under FlatQ-Merge
            if bits == 8: # Let's profile curvature under 8-bit quantization
                curv_profile = profile_curvature(active_model, base_dict, task_vectors, lambdas_opt, expert_heads, quantize=True, bits=bits, calib_loaders=calib_loaders, sigma_grid=sigma_grid)
                curvature_profiles[str(rho)] = curv_profile
            
            # Technique 2: Naive Uniform Merging (M-then-Q)
            lambdas_uniform = torch.ones(14, 4, device=DEVICE) * 0.3
            accs_uniform = evaluate_model(active_model, base_dict, task_vectors, lambdas_uniform, expert_heads, quantize=True, bits=bits, test_loaders=test_loaders)
            avg_acc_uniform = np.mean(list(accs_uniform.values()))
            
            # Technique 3: AdaMerging (Optimizing in FP, then quantizing post-hoc Q-after-M)
            print("Running AdaMerging (post-hoc quantization) optimization...")
            lambdas_fp = test_time_optimize(active_model, base_dict, task_vectors, expert_heads, quantize=False, bits=bits, calib_loaders=calib_loaders)
            accs_adamerge = evaluate_model(active_model, base_dict, task_vectors, lambdas_fp, expert_heads, quantize=True, bits=bits, test_loaders=test_loaders)
            avg_acc_adamerge = np.mean(list(accs_adamerge.values()))
            
            # Technique 4: Individual SAM Experts (Quantized, Unmerged)
            accs_individual = {}
            for k, task_name in enumerate(TASKS):
                lambdas_ind = torch.zeros(14, 4, device=DEVICE)
                lambdas_ind[:, k] = 1.0
                accs_k = evaluate_model(active_model, base_dict, task_vectors, lambdas_ind, expert_heads, quantize=True, bits=bits, test_loaders={task_name: test_loaders[task_name]})
                accs_individual[task_name] = accs_k[task_name]
            avg_acc_individual = np.mean(list(accs_individual.values()))
            
            rho_results[str(bits)] = {
                'FlatQ-Merge': {
                    'task_accuracies': accs_flatq,
                    'average': float(avg_acc_flatq),
                    'lambdas': lambdas_opt.tolist()
                },
                'NaiveUniform': {
                    'task_accuracies': accs_uniform,
                    'average': float(avg_acc_uniform)
                },
                'AdaMerging-PostQ': {
                    'task_accuracies': accs_adamerge,
                    'average': float(avg_acc_adamerge)
                },
                'Individual-Quantized': {
                    'task_accuracies': accs_individual,
                    'average': float(avg_acc_individual)
                }
            }
            
        seed_results[str(rho)] = rho_results
        
    return seed_results, curvature_profiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=int, default=1000, help="Test set subset size for fast evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to run")
    args = parser.parse_args()
    
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42, 100, 2026]
    
    # We load and cache validation and test datasets once
    print("Pre-loading and caching test loaders (1000 images per task)...")
    test_datasets = {task: get_cached_dataset(task, split='test', size=args.test_size, seed=999) for task in TASKS}
    test_loaders = {task: DataLoader(test_datasets[task], batch_size=128, shuffle=False) for task in TASKS}
    
    print("Pre-loading and caching calibration loaders (16 images per task)...")
    calib_datasets = {task: get_cached_dataset(task, split='test', size=16, seed=777) for task in TASKS}
    calib_loaders = {task: DataLoader(calib_datasets[task], batch_size=16, shuffle=False) for task in TASKS}
    
    all_results = {}
    all_curvature = {}
    
    for seed in seeds:
        seed_results, curv = run_experiments_for_seed(seed, test_loaders, calib_loaders)
        all_results[str(seed)] = seed_results
        all_curvature[str(seed)] = curv
        
    # Save results to JSON
    if args.seed is not None:
        out_filename = f"flatq_merge_results_seed{args.seed}.json"
    else:
        out_filename = "flatq_merge_results.json"
        
    with open(out_filename, "w") as f:
        json.dump({
            'results': all_results,
            'curvature': all_curvature
        }, f, indent=2)
    print(f"Results saved to {out_filename}")
    
    # Generate Plots only if all seeds are run or available
    if args.seed is None:
        generate_plots(all_results, all_curvature)

def generate_plots(all_results, all_curvature):
    radii = [0.0, 0.01, 0.05, 0.1, 0.2]
    techniques = ['FlatQ-Merge', 'NaiveUniform', 'AdaMerging-PostQ', 'Individual-Quantized']
    colors = {'FlatQ-Merge': '#1f77b4', 'NaiveUniform': '#ff7f0e', 'AdaMerging-PostQ': '#2ca02c', 'Individual-Quantized': '#d62728'}
    markers = {'FlatQ-Merge': 'o', 'NaiveUniform': 's', 'AdaMerging-PostQ': '^', 'Individual-Quantized': 'x'}
    
    # Plot 1 & 2: Average accuracy vs. SAM radius (8-bit and 4-bit)
    for bits in [8, 4]:
        plt.figure(figsize=(8, 6))
        for tech in techniques:
            y_vals = []
            y_errs = []
            for rho in radii:
                tech_accs = []
                for seed in ['42', '100', '2026']:
                    tech_accs.append(all_results[seed][str(rho)][str(bits)][tech]['average'])
                y_vals.append(np.mean(tech_accs))
                y_errs.append(np.std(tech_accs))
                
            plt.errorbar(radii, y_vals, yerr=y_errs, label=tech, color=colors[tech], marker=markers[tech], linewidth=2, capsize=5)
            
        plt.title(f"Average Multi-Task Accuracy vs. SAM Radius ({bits}-bit Quantization)", fontsize=14)
        plt.xlabel("SAM Radius (expert pre-merging flatness)", fontsize=12)
        plt.ylabel("Average Test Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.savefig(f"flatq_merge_acc_{bits}bit.png", dpi=300)
        plt.close()
        print(f"Generated plot flatq_merge_acc_{bits}bit.png")
        
    # Plot 3: Curvature Profiling (Entropy change vs perturbation scale)
    plt.figure(figsize=(8, 6))
    sigma_grid = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    for rho in radii:
        y_vals = []
        for sigma in sigma_grid:
            sigma_vals = []
            for seed in ['42', '100', '2026']:
                sigma_vals.append(all_curvature[seed][str(rho)][str(sigma)])
            y_vals.append(np.mean(sigma_vals))
            
        plt.plot(sigma_grid, y_vals, label=f"SAM expert rho = {rho}", marker='o', linewidth=2)
        
    plt.title("Coefficient-Space Curvature Profile (Test-Time Loss Landscape)", fontsize=14)
    plt.xlabel("Coefficient Perturbation Noise Scale (sigma)", fontsize=12)
    plt.ylabel("Prediction Entropy Increase (Delta H)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig("flatq_merge_curvature_profile.png", dpi=300)
    plt.close()
    print("Generated plot flatq_merge_curvature_profile.png")

if __name__ == '__main__':
    main()
