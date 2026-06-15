import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time

# Define the SimpleCNN architecture (must match train_experts.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 4x4
        )
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load the 4 expert models
TASKS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']

def load_experts():
    experts = {}
    for task in TASKS:
        model = SimpleCNN()
        path = f'checkpoints/{task.lower()}_expert.pth'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model not found at {path}. Run train_experts.py first.")
        model.load_state_dict(torch.load(path))
        model.eval()
        experts[task] = model
    return experts

# Define data transforms
def get_transforms(is_grayscale=False):
    if is_grayscale:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# Create calibration (64 samples) and test (400 samples) splits with seed-based randomization
def build_data_splits(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cal_data = []
    cal_labels = []
    cal_task_ids = []
    
    test_data = {task: [] for task in TASKS}
    test_labels = {task: [] for task in TASKS}
    
    for task_id, task in enumerate(TASKS):
        is_gray = task in ['MNIST', 'FashionMNIST']
        transform = get_transforms(is_grayscale=is_gray)
        
        if task == 'SVHN':
            dataset = datasets.SVHN('./data', split='test', download=True, transform=transform)
        else:
            dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform) if task == 'CIFAR10' else \
                      datasets.MNIST('./data', train=False, download=True, transform=transform) if task == 'MNIST' else \
                      datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
                      
        total_len = len(dataset)
        indices = np.random.permutation(total_len)
        cal_indices = indices[:16]
        test_indices = indices[16:116]
        
        # Calibration split (16 samples per task)
        for idx in cal_indices:
            img, lbl = dataset[int(idx)]
            cal_data.append(img)
            cal_labels.append(lbl)
            cal_task_ids.append(task_id)
            
        # Test split (100 samples per task)
        for idx in test_indices:
            img, lbl = dataset[int(idx)]
            test_data[task].append(img)
            test_labels[task].append(lbl)
            
    cal_data = torch.stack(cal_data)
    cal_labels = torch.tensor(cal_labels, dtype=torch.long)
    cal_task_ids = torch.tensor(cal_task_ids, dtype=torch.long)
    
    for task in TASKS:
        test_data[task] = torch.stack(test_data[task])
        test_labels[task] = torch.tensor(test_labels[task], dtype=torch.long)
        
    return cal_data, cal_labels, cal_task_ids, test_data, test_labels

# Routing Head that routes on average pooled Conv1 activations (H_0 features)
class PhysicalRoutingHead(nn.Module):
    def __init__(self, in_features=16, num_tasks=4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_tasks)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        return self.fc(x)

# Perform physical functional parameter ensembling
def blend_parameters_functional(experts, alphas, k=1):
    expert_states = {task: dict(experts[task].named_parameters()) for task in TASKS}
    # MNIST can act as baseline model structure
    base_state = dict(experts['MNIST'].named_parameters())
    
    blended_params = {}
    for name in base_state.keys():
        if 'features.0' in name:
            grp = 0
        elif 'features.3' in name:
            grp = 1
        elif 'features.6' in name:
            grp = 2
        elif 'fc' in name:
            grp = 3
        else:
            grp = 0
            
        if grp < 4 - k:
            # Static Offline Uniform Merge: average weights of all experts (detached)
            uniform_val = sum(expert_states[task][name] for task in TASKS) / 4.0
            blended_params[name] = uniform_val.detach()
        else:
            # Dynamic Merge: blend differentiably based on task coefficients
            dynamic_val = torch.zeros_like(base_state[name])
            for i, task in enumerate(TASKS):
                dynamic_val = dynamic_val + alphas[i] * expert_states[task][name]
            blended_params[name] = dynamic_val
            
    return blended_params

# PyTorch K-means clustering for Dynamic Batch Filtering
def kmeans_pytorch(x, num_clusters, num_iters=10, seed=42):
    torch.manual_seed(seed)
    B, D = x.shape
    if B <= num_clusters:
        return torch.arange(B, device=x.device) % num_clusters
    indices = torch.randperm(B, device=x.device)[:num_clusters]
    centroids = x[indices].clone()

    for _ in range(num_iters):
        dists = torch.cdist(x, centroids)
        cluster_ids = dists.argmin(dim=-1)
        for c in range(num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() > 0:
                centroids[c] = x[mask].mean(dim=0)
    return cluster_ids

# Physical routing calibration loop using direct Task-ID Supervision
def train_physical_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=250, lr=1e-2):
    # Extract H_0 features (average pooled output of Conv1 from the base uniform model)
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    with torch.no_grad():
        h0_features = []
        for idx in range(len(cal_data)):
            img = cal_data[idx:idx+1]
            conv1_feat = torch.func.functional_call(base_model.features[0:2], {
                '0.weight': uniform_params['features.0.weight'],
                '0.bias': uniform_params['features.0.bias']
            }, img)
            h0_features.append(conv1_feat.mean(dim=[2, 3]).view(-1))
        h0_features = torch.stack(h0_features) # shape [64, 16]
        
    # Calculate feature normalization stats
    mean = h0_features.mean(dim=0, keepdim=True)
    std = h0_features.std(dim=0, keepdim=True) + 1e-6
    
    # Normalize features
    norm_features = (h0_features - mean) / std
    
    head = PhysicalRoutingHead(in_features=16, num_tasks=4)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    head.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = head(norm_features)
        loss = criterion(out, cal_task_ids)
        loss.backward()
        optimizer.step()
            
    return head, mean, std

# Evaluate physical merged model on test splits
def evaluate_physical_routing(experts, head, mean, std, test_data, test_labels, k=1, T=0.1):
    head.eval()
    results = {}
    
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for task_id, task in enumerate(TASKS):
            imgs = test_data[task]
            lbls = test_labels[task]
            
            # Extract H_0 features
            task_h0_features = []
            for idx in range(len(imgs)):
                img = imgs[idx:idx+1]
                conv1_feat = torch.func.functional_call(base_model.features[0:2], {
                    '0.weight': uniform_params['features.0.weight'],
                    '0.bias': uniform_params['features.0.bias']
                }, img)
                task_h0_features.append(conv1_feat.mean(dim=[2, 3]).view(-1))
            task_h0_features = torch.stack(task_h0_features)
            
            # Normalize using calibration stats
            task_norm_features = (task_h0_features - mean) / std
            
            # Compute coefficients using Softmax with temperature T
            logits = head(task_norm_features)
            alphas = torch.softmax(logits / T, dim=-1)
            
            # Average coefficients over the task test batch to measure average task behavior
            mean_alphas = alphas.mean(dim=0)
            
            # Assemble the task-specific merged model using averaged coefficients
            task_params = blend_parameters_functional(experts, mean_alphas, k=k)
            
            # Evaluate on task test set
            out = torch.func.functional_call(base_model, task_params, imgs)
            _, predicted = out.max(1)
            correct = predicted.eq(lbls).sum().item()
            acc = 100.0 * correct / len(lbls)
            
            results[task] = acc
            total_correct += correct
            total_samples += len(lbls)
            
    results['Joint Mean'] = 100.0 * total_correct / total_samples
    return results

# Evaluate heterogeneous stream on the physical CNN with or without DBF
def evaluate_physical_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=16, k=4, T=0.1, use_dbf=False, seed=42):
    head.eval()
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    # Create the combined stream
    stream_imgs = []
    stream_lbls = []
    
    # Ensure deterministic shuffle per seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for task in TASKS:
        imgs = test_data[task]
        lbls = test_labels[task]
        for idx in range(len(imgs)):
            stream_imgs.append(imgs[idx])
            stream_lbls.append(lbls[idx])
            
    stream_imgs = torch.stack(stream_imgs)
    stream_lbls = torch.tensor(stream_lbls, dtype=torch.long)
    
    # Shuffle stream
    shuffle_idx = torch.randperm(len(stream_imgs))
    stream_imgs = stream_imgs[shuffle_idx]
    stream_lbls = stream_lbls[shuffle_idx]
    
    num_batches = len(stream_imgs) // batch_size
    total_correct = 0
    total_samples = num_batches * batch_size
    
    total_latency = 0.0
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_imgs = stream_imgs[i*batch_size : (i+1)*batch_size]
            batch_lbls = stream_lbls[i*batch_size : (i+1)*batch_size]
            
            start_time = time.perf_counter()
            
            # 1. Extract H_0 features for the batch
            batch_h0_features = []
            for idx in range(len(batch_imgs)):
                img = batch_imgs[idx:idx+1]
                conv1_feat = torch.func.functional_call(base_model.features[0:2], {
                    '0.weight': uniform_params['features.0.weight'],
                    '0.bias': uniform_params['features.0.bias']
                }, img)
                batch_h0_features.append(conv1_feat.mean(dim=[2, 3]).view(-1))
            batch_h0_features = torch.stack(batch_h0_features)
            
            # 2. Normalize
            batch_norm_features = (batch_h0_features - mean) / std
            
            if use_dbf and batch_size > 1:
                # Use DBF: cluster features into M=4 groups
                cluster_ids = kmeans_pytorch(batch_norm_features, num_clusters=4, num_iters=10, seed=seed)
                
                for c in range(4):
                    cluster_mask = (cluster_ids == c)
                    if cluster_mask.sum() == 0:
                        continue
                    sub_imgs = batch_imgs[cluster_mask]
                    sub_lbls = batch_lbls[cluster_mask]
                    sub_norm_feats = batch_norm_features[cluster_mask]
                    
                    # Compute routing coefficients for this sub-batch
                    logits = head(sub_norm_feats)
                    alphas = torch.softmax(logits / T, dim=-1)
                    mean_alphas = alphas.mean(dim=0)
                    
                    # Assemble task weights for this sub-batch
                    task_params = blend_parameters_functional(experts, mean_alphas, k=k)
                    
                    # Run forward pass
                    out = torch.func.functional_call(base_model, task_params, sub_imgs)
                    _, predicted = out.max(1)
                    total_correct += predicted.eq(sub_lbls).sum().item()
            else:
                # Standard batch routing (batch style blur)
                logits = head(batch_norm_features)
                alphas = torch.softmax(logits / T, dim=-1)
                mean_alphas = alphas.mean(dim=0)
                
                # Assemble task weights
                task_params = blend_parameters_functional(experts, mean_alphas, k=k)
                
                # Run forward pass
                out = torch.func.functional_call(base_model, task_params, batch_imgs)
                _, predicted = out.max(1)
                total_correct += predicted.eq(batch_lbls).sum().item()
                
            latency = (time.perf_counter() - start_time) * 1000.0  # ms
            total_latency += latency
            
    avg_latency = total_latency / num_batches
    accuracy = 100.0 * total_correct / total_samples
    return accuracy, avg_latency

if __name__ == '__main__':
    print("Loading experts...")
    experts = load_experts()
    
    SEEDS = [42, 43, 44]
    
    # 1. BASELINE 1: Uniform Offline Merge (k=0) over 3 seeds
    print("\n" + "="*60)
    print("--- EVALUATION 1: Uniform Offline Merge (k=0) ---")
    print("="*60)
    
    uniform_accs = {task: [] for task in TASKS}
    uniform_joint_accs = []
    
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    for seed in SEEDS:
        _, _, _, test_data, test_labels = build_data_splits(seed=seed)
        
        uniform_correct = 0
        uniform_total = 0
        for task_id, task in enumerate(TASKS):
            imgs = test_data[task]
            lbls = test_labels[task]
            with torch.no_grad():
                out = torch.func.functional_call(base_model, uniform_params, imgs)
                _, predicted = out.max(1)
                correct = predicted.eq(lbls).sum().item()
                acc = 100.0 * correct / len(lbls)
                uniform_accs[task].append(acc)
                uniform_correct += correct
                uniform_total += len(lbls)
        uniform_joint_accs.append(100.0 * uniform_correct / uniform_total)
        
    print("Uniform Merge (k=0) Results (Mean ± Std over 3 seeds):")
    for task in TASKS:
        print(f"  {task:12s}: {np.mean(uniform_accs[task]):.2f}% ± {np.std(uniform_accs[task]):.2f}%")
    print(f"  Joint Mean  : {np.mean(uniform_joint_accs):.2f}% ± {np.std(uniform_joint_accs):.2f}%")
    
    # 2. Sweep over partition depths k in [0, 1, 2, 3, 4] over 3 seeds
    print("\n" + "="*60)
    print("--- EVALUATION 2: Sweeping Partition Depth k over 3 Seeds ---")
    print("="*60)
    
    k_sweeps_results = {k: {task: [] for task in TASKS + ['Joint Mean']} for k in [0, 1, 2, 3, 4]}
    
    for seed in SEEDS:
        print(f"\n[Seed {seed}] Training routing head...")
        cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=seed)
        head, mean, std = train_physical_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=250)
        
        for k in [0, 1, 2, 3, 4]:
            res = evaluate_physical_routing(experts, head, mean, std, test_data, test_labels, k=k, T=0.1)
            for key in res.keys():
                k_sweeps_results[k][key].append(res[key])
                
    print("\n" + "-"*80)
    print("--- SUMMARY OF PHYSICAL SWEEP OVER k (Mean ± Std over 3 seeds) ---")
    print("-"*80)
    print(f"{'Depth (k)':9s} | {'MNIST':15s} | {'FMNIST':15s} | {'CIFAR10':15s} | {'SVHN':15s} | {'Joint Mean':15s}")
    print("-"*95)
    for k in [0, 1, 2, 3, 4]:
        res = k_sweeps_results[k]
        print(f"   {k:2d}     | {np.mean(res['MNIST']):.2f}% ± {np.std(res['MNIST']):.2f}% | "
              f"{np.mean(res['FashionMNIST']):.2f}% ± {np.std(res['FashionMNIST']):.2f}% | "
              f"{np.mean(res['CIFAR10']):.2f}% ± {np.std(res['CIFAR10']):.2f}% | "
              f"{np.mean(res['SVHN']):.2f}% ± {np.std(res['SVHN']):.2f}% | "
              f"{np.mean(res['Joint Mean']):.2f}% ± {np.std(res['Joint Mean']):.2f}%")
              
    # 3. Dynamic Batch Filtering (DBF) Physical Validation under Heterogeneous Streaming
    print("\n" + "="*60)
    print("--- EVALUATION 3: Dynamic Batch Filtering (DBF) under Streaming ---")
    print("="*60)
    
    stream_results = {
        'B16_Std': {'acc': [], 'lat': []},
        'B16_DBF': {'acc': [], 'lat': []},
        'B64_Std': {'acc': [], 'lat': []},
        'B64_DBF': {'acc': [], 'lat': []}
    }
    
    for seed in SEEDS:
        cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=seed)
        head, mean, std = train_physical_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=250)
        
        # Batch size 16
        acc, lat = evaluate_physical_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=16, k=4, T=0.1, use_dbf=False, seed=seed)
        stream_results['B16_Std']['acc'].append(acc)
        stream_results['B16_Std']['lat'].append(lat)
        
        acc, lat = evaluate_physical_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=16, k=4, T=0.1, use_dbf=True, seed=seed)
        stream_results['B16_DBF']['acc'].append(acc)
        stream_results['B16_DBF']['lat'].append(lat)
        
        # Batch size 64
        acc, lat = evaluate_physical_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=64, k=4, T=0.1, use_dbf=False, seed=seed)
        stream_results['B64_Std']['acc'].append(acc)
        stream_results['B64_Std']['lat'].append(lat)
        
        acc, lat = evaluate_physical_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=64, k=4, T=0.1, use_dbf=True, seed=seed)
        stream_results['B64_DBF']['acc'].append(acc)
        stream_results['B64_DBF']['lat'].append(lat)
        
    print("\nPhysical Streaming Benchmark Results (Mean ± Std over 3 seeds):")
    print("-" * 80)
    for cfg in ['B16_Std', 'B16_DBF', 'B64_Std', 'B64_DBF']:
        mean_acc = np.mean(stream_results[cfg]['acc'])
        std_acc = np.std(stream_results[cfg]['acc'])
        mean_lat = np.mean(stream_results[cfg]['lat'])
        std_lat = np.std(stream_results[cfg]['lat'])
        print(f"  {cfg:10s} -> Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}% | Latency per batch: {mean_lat:.3f} ms ± {std_lat:.3f} ms")
    print("-" * 80)
