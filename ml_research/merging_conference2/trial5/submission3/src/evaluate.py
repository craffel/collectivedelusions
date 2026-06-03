import os
import copy
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# --- Preloading Loader for 150x Speedup ---
_PRELOADED_LOADER_CACHE = {}

class PreloadedBatchLoader:
    def __init__(self, dataset, batch_size, device):
        # Store a reference to dataset to prevent its memory ID from being recycled by GC
        self.dataset = dataset

    def __new__(cls, dataset, batch_size, device):
        cache_key = (id(dataset), batch_size, str(device))
        if cache_key in _PRELOADED_LOADER_CACHE:
            return _PRELOADED_LOADER_CACHE[cache_key]
        
        instance = super().__new__(cls)
        instance.batches = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for images, labels in loader:
            instance.batches.append((images.to(device), labels.to(device)))
        instance.dataset = dataset
        _PRELOADED_LOADER_CACHE[cache_key] = instance
        return instance
            
    def __iter__(self):
        return iter(self.batches)
        
    def __len__(self):
        return len(self.batches)

# --- Utility Functions ---

def get_batchnorm_modules(model):
    """Retrieves all BatchNorm2d modules in sequential order."""
    bn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_modules.append((name, module))
    return bn_modules

_BASE_MODEL_CACHED = None
_MODEL_CACHE = {}

def load_base_model():
    """Loads the base ResNet18 model with task-specific head structure."""
    global _BASE_MODEL_CACHED
    if _BASE_MODEL_CACHED is None:
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except ImportError:
            model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        _BASE_MODEL_CACHED = model
    return copy.deepcopy(_BASE_MODEL_CACHED)

_EXPERT_CACHE = {}

def load_expert(task, device):
    """Loads a fine-tuned expert model."""
    cache_key = (task, str(device))
    if cache_key not in _EXPERT_CACHE:
        model = load_base_model()
        path = f'experts/{task}.pt'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model not found at {path}. Please run train_experts.py first.")
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        _EXPERT_CACHE[cache_key] = model
    return _EXPERT_CACHE[cache_key]

# --- Data Preparation ---

_DATASET_CACHE = {}

def get_datasets(N=128, p_corruption=0.0, corruption_type='gaussian', seed=42):
    """
    Returns the test datasets and calibration subsets for MNIST, FMNIST, and CIFAR-10.
    Calibration samples are selected from indices 3000 to 3000+N (disjoint from fine-tuning training set).
    If p_corruption > 0, we corrupt a fraction p of the calibration dataset.
    """
    global _DATASET_CACHE
    
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if 'mnist_train_full' not in _DATASET_CACHE:
        _DATASET_CACHE['mnist_train_full'] = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_mnist)
        _DATASET_CACHE['mnist_test'] = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_mnist)

        _DATASET_CACHE['fmnist_train_full'] = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_mnist)
        _DATASET_CACHE['fmnist_test'] = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_mnist)

        _DATASET_CACHE['cifar_train_full'] = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_cifar)
        _DATASET_CACHE['cifar_test'] = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_cifar)

    mnist_train_full = _DATASET_CACHE['mnist_train_full']
    mnist_test = _DATASET_CACHE['mnist_test']
    fmnist_train_full = _DATASET_CACHE['fmnist_train_full']
    fmnist_test = _DATASET_CACHE['fmnist_test']
    cifar_train_full = _DATASET_CACHE['cifar_train_full']
    cifar_test = _DATASET_CACHE['cifar_test']

    # Base calibration indices (disjoint from training)
    cal_indices = list(range(3000, 3000 + N))

    mnist_cal_base = Subset(mnist_train_full, cal_indices)
    fmnist_cal_base = Subset(fmnist_train_full, cal_indices)
    cifar_cal_base = Subset(cifar_train_full, cal_indices)

    # Corrupt if needed
    if p_corruption > 0.0:
        class CorruptedSubset(torch.utils.data.Dataset):
            def __init__(self, original_subset, p_corr, corr_type='gaussian', seed=42):
                self.original_subset = original_subset
                self.num_samples = len(original_subset)
                self.num_corrupted = int(self.num_samples * p_corr)
                self.corr_type = corr_type
                # Deterministic selection of corrupted indices
                rng = np.random.default_rng(seed)
                self.corrupted_idx = set(rng.choice(self.num_samples, self.num_corrupted, replace=False))
                self.seed = seed
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                img, label = self.original_subset[idx]
                if idx in self.corrupted_idx:
                    if self.corr_type == 'gaussian':
                        img = torch.randn_like(img)
                    elif self.corr_type == 'uniform':
                        img = torch.rand_like(img) * 2.0 - 1.0
                    elif self.corr_type == 'salt_and_pepper':
                        img = img.clone()
                        rng_sp = np.random.default_rng(self.seed + idx)
                        mask_salt = rng_sp.random(img.shape) < 0.1
                        mask_pepper = rng_sp.random(img.shape) < 0.1
                        img[mask_salt] = 1.0
                        img[mask_pepper] = -1.0
                return img, label

        mnist_cal = CorruptedSubset(mnist_cal_base, p_corruption, corr_type=corruption_type, seed=seed)
        fmnist_cal = CorruptedSubset(fmnist_cal_base, p_corruption, corr_type=corruption_type, seed=seed+1)
        cifar_cal = CorruptedSubset(cifar_cal_base, p_corruption, corr_type=corruption_type, seed=seed+2)
    else:
        mnist_cal = mnist_cal_base
        fmnist_cal = fmnist_cal_base
        cifar_cal = cifar_cal_base

    # Joint Calibration Dataset
    class JointDataset(torch.utils.data.Dataset):
        def __init__(self, d1, d2, d3):
            self.d1 = d1
            self.d2 = d2
            self.d3 = d3
            self.l1 = len(d1)
            self.l2 = len(d2)
            self.l3 = len(d3)
            
        def __len__(self):
            return self.l1 + self.l2 + self.l3
            
        def __getitem__(self, idx):
            if idx < self.l1:
                return self.d1[idx]
            elif idx < self.l1 + self.l2:
                return self.d2[idx - self.l1]
            else:
                return self.d3[idx - self.l1 - self.l2]

    joint_cal = JointDataset(mnist_cal, fmnist_cal, cifar_cal)

    return {
        'mnist_cal': mnist_cal,
        'fmnist_cal': fmnist_cal,
        'cifar10_cal': cifar_cal,
        'joint_cal': joint_cal,
        'mnist_test': mnist_test,
        'fmnist_test': fmnist_test,
        'cifar10_test': cifar_test
    }

# --- Merging Core ---

def merge_models(base_model, experts, mode='wa', lam=0.3):
    """
    Merges the backbone parameters of a list of expert models.
    Supports Weight Averaging ('wa') and Task Arithmetic ('ta').
    Note that fc layers (heads) are NOT merged; only the backbone is.
    """
    merged_state = {}
    base_state = base_model.state_dict()
    expert_states = [exp.state_dict() for exp in experts]
    
    for key in base_state.keys():
        if 'fc.' in key:
            continue
        
        if mode == 'wa':
            # Weight Averaging
            tensors = [state[key].float() for state in expert_states]
            merged_state[key] = torch.stack(tensors).mean(dim=0)
        elif mode == 'ta':
            # Task Arithmetic
            task_vectors = []
            for state in expert_states:
                task_vectors.append(state[key].float() - base_state[key].float())
            merged_state[key] = base_state[key].float() + lam * torch.stack(task_vectors).sum(dim=0)
            
    return merged_state

# --- Evaluation Core ---

_EVAL_MODEL_CACHE = {}

def get_eval_model(device):
    global _EVAL_MODEL_CACHE
    device_str = str(device)
    if device_str not in _EVAL_MODEL_CACHE:
        _EVAL_MODEL_CACHE[device_str] = load_base_model().to(device)
    return _EVAL_MODEL_CACHE[device_str]

def evaluate_backbone(backbone_state, expert_head_model, test_loader, device):
    """
    Evaluates the performance of a merged backbone combined with a specific task's expert head.
    """
    model = get_eval_model(device)
    
    # We construct the state dict in a fast, surgical way
    state_dict = {}
    for k, v in backbone_state.items():
        state_dict[k] = v
    state_dict['fc.weight'] = expert_head_model.state_dict()['fc.weight']
    state_dict['fc.bias'] = expert_head_model.state_dict()['fc.bias']
    
    model.load_state_dict(state_dict)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
    return 100.0 * correct / total

# --- Calibration Core (RepSeqCalib & QRC) ---

_CAL_MODEL_CACHE = {}

def get_cal_model(device):
    global _CAL_MODEL_CACHE
    device_str = str(device)
    if device_str not in _CAL_MODEL_CACHE:
        _CAL_MODEL_CACHE[device_str] = load_base_model().to(device)
    return _CAL_MODEL_CACHE[device_str]

def run_sequential_calibration(merged_backbone, experts, cal_data, method='taac', alpha=0.25, device='cpu'):
    """
    Executes Reparameterized Sequential Calibration (RepSeqCalib) on the merged backbone.
    Modifies the BatchNorm layers in-place.
    Supported methods:
      - 'none': No calibration
      - 'sp-taac': Sparsity-Preserving Task-Agnostic Activation Calibration (global scaling only)
      - 'taac': Standard Task-Agnostic Activation Calibration (channel-wise mean/std)
      - 'slf-taac': Sample-Level Filtering TAAC (robust L2 filtering + mean/std)
      - 'qrc': Quantile-based Robust Calibration (robust channel-wise Median/IQR)
      - 'qrc-median': Median-only calibration (IQR is not aligned, global scale=1)
      - 'qrc-iqr': IQR-only calibration (Median shift is not aligned, shift=0)
    """
    # Create temporary models loaded with merged_backbone
    m_model = get_cal_model(device)
    # Fill in a dummy fc just to satisfy load_state_dict (we will use a copy of the merged backbone)
    mb_state = {}
    for k, v in merged_backbone.items():
        mb_state[k] = v
    mb_state['fc.weight'] = experts[0].state_dict()['fc.weight']
    mb_state['fc.bias'] = experts[0].state_dict()['fc.bias']
    m_model.load_state_dict(mb_state)
    m_model.eval()

    # Create loaders
    expert_loaders = [
        PreloadedBatchLoader(cal_data['mnist_cal'], batch_size=128, device=device),
        PreloadedBatchLoader(cal_data['fmnist_cal'], batch_size=128, device=device),
        PreloadedBatchLoader(cal_data['cifar10_cal'], batch_size=128, device=device)
    ]
    joint_loader = PreloadedBatchLoader(cal_data['joint_cal'], batch_size=128, device=device)

    # Get BN modules
    bn_modules_m = get_batchnorm_modules(m_model)
    bn_modules_e = [get_batchnorm_modules(exp) for exp in experts]

    # Hook dictionary to record activation outputs
    activations = {}
    def get_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    print(f"Running RepSeqCalib with method: {method.upper()}...")

    # Precompute Expert activations/statistics for all layers
    expert_stats_cache = [] # list of dicts, one per expert
    for m_idx, exp in enumerate(experts):
        exp_cache = {}
        # Register hooks for all BN layers of this expert
        handles = []
        expert_activations = {}
        
        def make_hook(name_key):
            def hook(module, inp, out):
                if name_key not in expert_activations:
                    expert_activations[name_key] = []
                expert_activations[name_key].append(out.detach().cpu())
            return hook
            
        for name_e, module_e in bn_modules_e[m_idx]:
            handles.append(module_e.register_forward_hook(make_hook(name_e)))
            
        # Single forward pass over the expert's calibration set
        with torch.no_grad():
            for images, _ in expert_loaders[m_idx]:
                images = images.to(device)
                exp(images)
                
        # Remove all hooks
        for handle in handles:
            handle.remove()
            
        # Concatenate and compute statistics for each layer
        for name_e, module_e in bn_modules_e[m_idx]:
            Y_exp = torch.cat(expert_activations[name_e], dim=0) # B_cal x C x H x W
            B, C, H, W = Y_exp.shape
            
            stats = {}
            if method in ['taac', 'sp-taac']:
                Y_exp_flat = Y_exp.transpose(0, 1).contiguous().view(C, -1) # C x (B*H*W)
                stats['mean'] = Y_exp_flat.mean(dim=1)
                stats['std'] = Y_exp_flat.std(dim=1, unbiased=True)
            elif method == 'slf-taac':
                scores = (Y_exp ** 2).view(B, -1).mean(dim=1)
                num_corrupted = int(B * alpha)
                _, sorted_idx = torch.sort(scores)
                clean_idx = sorted_idx[:B - num_corrupted]
                Y_clean = Y_exp[clean_idx]
                Y_clean_flat = Y_clean.transpose(0, 1).contiguous().view(C, -1)
                stats['mean'] = Y_clean_flat.mean(dim=1)
                stats['std'] = Y_clean_flat.std(dim=1, unbiased=True)
            elif method in ['qrc', 'qrc-median', 'qrc-iqr', 'qrc-idr', 'qrc-95'] or 'qrc-tau-' in method:
                Y_exp_flat = Y_exp.transpose(0, 1).contiguous().view(C, -1)
                ch_medians = []
                ch_iqrs = []
                active_threshold = 1e-5
                if 'qrc-tau-' in method:
                    active_threshold = float(method.split('qrc-tau-')[1])
                min_active_count = 10
                q_lower = 0.10 if 'idr' in method else (0.05 if '95' in method else 0.25)
                q_upper = 0.90 if 'idr' in method else (0.95 if '95' in method else 0.75)
                for c in range(C):
                    vals = Y_exp_flat[c]
                    active_vals = vals[vals > active_threshold]
                    if len(active_vals) >= min_active_count:
                        ch_medians.append(torch.median(active_vals))
                        q_l = torch.quantile(active_vals, q_lower)
                        q_u = torch.quantile(active_vals, q_upper)
                        ch_iqrs.append(q_u - q_l)
                    else:
                        ch_medians.append(torch.median(vals))
                        q_l = torch.quantile(vals, q_lower)
                        q_u = torch.quantile(vals, q_upper)
                        ch_iqrs.append(q_u - q_l)
                stats['median'] = torch.stack(ch_medians)
                stats['iqr'] = torch.stack(ch_iqrs)
            exp_cache[name_e] = stats
        expert_stats_cache.append(exp_cache)
    
    # Layer-by-layer sequential calibration
    for l_idx in range(len(bn_modules_m)):
        name_m, module_m = bn_modules_m[l_idx]
        
        # 1. Collect Expert Target Statistics
        target_means = []
        target_stds = []
        target_medians = []
        target_iqrs = []
        
        for m_idx, exp in enumerate(experts):
            name_e, module_e = bn_modules_e[m_idx][l_idx]
            assert name_m == name_e, f"Module mismatch: {name_m} vs {name_e}"
            
            stats = expert_stats_cache[m_idx][name_e]
            if method in ['taac', 'sp-taac', 'slf-taac']:
                target_means.append(stats['mean'])
                target_stds.append(stats['std'])
            elif method in ['qrc', 'qrc-median', 'qrc-iqr', 'qrc-idr', 'qrc-95'] or 'qrc-tau-' in method:
                target_medians.append(stats['median'])
                target_iqrs.append(stats['iqr'])

        # Average targets across tasks
        if method in ['taac', 'sp-taac', 'slf-taac']:
            mean_target = torch.stack(target_means).mean(dim=0).to(device)
            std_target = torch.stack(target_stds).mean(dim=0).to(device)
        elif method in ['qrc', 'qrc-median', 'qrc-iqr', 'qrc-idr', 'qrc-95'] or 'qrc-tau-' in method:
            median_target = torch.stack(target_medians).mean(dim=0).to(device)
            iqr_target = torch.stack(target_iqrs).mean(dim=0).to(device)

        # 2. Collect Merged Model Statistics
        # Register forward hook on merged model
        handle = module_m.register_forward_hook(get_hook('merged'))
        collected_acts_m = []
        with torch.no_grad():
            for images, _ in joint_loader:
                images = images.to(device)
                m_model(images)
                collected_acts_m.append(activations['merged'].cpu())
        handle.remove()
        
        Y_merged = torch.cat(collected_acts_m, dim=0)
        B_m, C_m, H_m, W_m = Y_merged.shape
        
        # Compute merged stats
        if method in ['taac', 'sp-taac']:
            Y_merged_flat = Y_merged.transpose(0, 1).contiguous().view(C_m, -1)
            mean_merged = Y_merged_flat.mean(dim=1).to(device)
            std_merged = Y_merged_flat.std(dim=1, unbiased=True).to(device)
        elif method == 'slf-taac':
            scores_m = (Y_merged ** 2).view(B_m, -1).mean(dim=1)
            num_corrupted_m = int(B_m * alpha)
            _, sorted_idx_m = torch.sort(scores_m)
            clean_idx_m = sorted_idx_m[:B_m - num_corrupted_m]
            Y_clean_m = Y_merged[clean_idx_m]
            Y_clean_m_flat = Y_clean_m.transpose(0, 1).contiguous().view(C_m, -1)
            mean_merged = Y_clean_m_flat.mean(dim=1).to(device)
            std_merged = Y_clean_m_flat.std(dim=1, unbiased=True).to(device)
        elif method in ['qrc', 'qrc-median', 'qrc-iqr', 'qrc-idr', 'qrc-95'] or 'qrc-tau-' in method:
            Y_merged_flat = Y_merged.transpose(0, 1).contiguous().view(C_m, -1)
            median_merged_list = []
            iqr_merged_list = []
            active_threshold = 1e-5
            if 'qrc-tau-' in method:
                active_threshold = float(method.split('qrc-tau-')[1])
            min_active_count = 10
            q_lower = 0.10 if 'idr' in method else (0.05 if '95' in method else 0.25)
            q_upper = 0.90 if 'idr' in method else (0.95 if '95' in method else 0.75)
            for c in range(C_m):
                vals = Y_merged_flat[c]
                active_vals = vals[vals > active_threshold]
                if len(active_vals) >= min_active_count:
                    median_merged_list.append(torch.median(active_vals))
                    q_l = torch.quantile(active_vals, q_lower)
                    q_u = torch.quantile(active_vals, q_upper)
                    iqr_merged_list.append(q_u - q_l)
                else:
                    median_merged_list.append(torch.median(vals))
                    q_l = torch.quantile(vals, q_lower)
                    q_u = torch.quantile(vals, q_upper)
                    iqr_merged_list.append(q_u - q_l)
            median_merged = torch.stack(median_merged_list).to(device)
            iqr_merged = torch.stack(iqr_merged_list).to(device)

        # 3. Compute Calibration Factors and Apply ZIO-CF Weight Fusion
        eps = 1e-6
        if method == 'sp-taac':
            # Global scaling, shift is zero
            gamma = std_target.mean() / (std_merged.mean() + eps)
            s = torch.full_like(std_target, gamma)
            bcal = torch.zeros_like(mean_target)
        elif method in ['taac', 'slf-taac']:
            # Channel-wise affine calibration
            s = std_target / (std_merged + eps)
            bcal = mean_target - s * mean_merged
        elif method in ['qrc', 'qrc-idr', 'qrc-95'] or 'qrc-tau-' in method:
            # Robust Quantile-based Calibration (Scale and Shift)
            s = iqr_target / (iqr_merged + eps)
            bcal = median_target - s * median_merged
            invalid = (iqr_target < 1e-4) | (iqr_merged < 1e-4)
            s[invalid] = 1.0
            bcal[invalid] = 0.0
        elif method == 'qrc-median':
            # Median-only robust alignment (no scaling)
            s = torch.ones_like(median_target)
            bcal = median_target - median_merged
            invalid = (iqr_target < 1e-4) | (iqr_merged < 1e-4)
            bcal[invalid] = 0.0
        elif method == 'qrc-iqr':
            # IQR-only robust alignment (no shift)
            s = iqr_target / (iqr_merged + eps)
            bcal = torch.zeros_like(median_target)
            invalid = (iqr_target < 1e-4) | (iqr_merged < 1e-4)
            s[invalid] = 1.0
        else:
            # None
            s = torch.ones(C_m, device=device)
            bcal = torch.zeros(C_m, device=device)

        # FUSE back into the BatchNorm layer's weight and bias parameters (ZIO-CF)
        # weight_new = s * weight_old
        # bias_new = s * bias_old + bcal
        with torch.no_grad():
            module_m.weight.copy_(s * module_m.weight)
            module_m.bias.copy_(s * module_m.bias + bcal)

    # Return the fully-fused reparameterized state dict of the model
    return {k: v.cpu() for k, v in m_model.state_dict().items() if 'fc.' not in k}

# --- Main Sweep Execution ---

def main():
    print("Initializing evaluation and post-merge calibration study...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation running on device: {device}")

    # Load Base and Expert Models
    base_model = load_base_model().to(device)
    base_model.eval()
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    experts = []
    for task in tasks:
        print(f"Loading expert model for {task.upper()}...")
        experts.append(load_expert(task, device))

    # Evaluate Oracle Accuracy
    print("\n=== Evaluating Individual Expert Oracles (Task-Specific Backbones) ===")
    test_loaders = {}
    # Load basic datasets with N=128 to get test loaders
    data_all = get_datasets(N=128, p_corruption=0.0)
    test_loaders['mnist'] = PreloadedBatchLoader(Subset(data_all['mnist_test'], list(range(1000))), batch_size=128, device=device)
    test_loaders['fmnist'] = PreloadedBatchLoader(Subset(data_all['fmnist_test'], list(range(1000))), batch_size=128, device=device)
    test_loaders['cifar10'] = PreloadedBatchLoader(Subset(data_all['cifar10_test'], list(range(1000))), batch_size=128, device=device)

    oracle_accs = {}
    for i, task in enumerate(tasks):
        acc = evaluate_backbone(experts[i].state_dict(), experts[i], test_loaders[task], device)
        oracle_accs[task] = acc
        print(f"{task.upper()} Oracle Accuracy: {acc:.2f}%")
    print(f"Oracle Average: {sum(oracle_accs.values())/3:.2f}%")

    # Evaluate Uncalibrated Merged Models
    print("\n=== Evaluating Uncalibrated Merged Models ===")
    merged_wa_uncal = merge_models(base_model, experts, mode='wa')
    merged_ta_uncal = merge_models(base_model, experts, mode='ta', lam=0.3)

    for mode, state in [('Weight Averaging (WA)', merged_wa_uncal), ('Task Arithmetic (TA)', merged_ta_uncal)]:
        print(f"\nEvaluating uncalibrated {mode}:")
        accs = {}
        for i, task in enumerate(tasks):
            acc = evaluate_backbone(state, experts[i], test_loaders[task], device)
            accs[task] = acc
            print(f"  {task.upper()} Accuracy: {acc:.2f}%")
        print(f"  Uncalibrated Average: {sum(accs.values())/3:.2f}%")

    seeds = [42, 43, 44]
    print(f"\nRunning evaluations over {len(seeds)} random seeds: {seeds}...")

    # --- Sweep 1: Calibration Outlier Corruption (p) Sweep ---
    print("\n=== SWEEP 1: Outlier Corruption Level (p) Sweep (N=128, WA Merge) ===")
    p_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    methods = ['none', 'sp-taac', 'taac', 'slf-taac', 'qrc']
    
    sweep1_seed_results = {m: {p: [] for p in p_levels} for m in methods}
    
    for p in p_levels:
        print(f"\n--- Testing Corruption Level p = {p:.1f} ---")
        for seed in seeds:
            print(f"  Seed {seed}:")
            cal_data_p = get_datasets(N=128, p_corruption=p, corruption_type='gaussian', seed=seed)
            for m in methods:
                if m == 'none':
                    # No calibration
                    accs = []
                    for i, task in enumerate(tasks):
                        acc = evaluate_backbone(merged_wa_uncal, experts[i], test_loaders[task], device)
                        accs.append(acc)
                    avg_acc = sum(accs) / 3
                else:
                    cal_backbone = run_sequential_calibration(
                        copy.deepcopy(merged_wa_uncal), experts, cal_data_p, method=m, device=device
                    )
                    accs = []
                    for i, task in enumerate(tasks):
                        acc = evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device)
                        accs.append(acc)
                    avg_acc = sum(accs) / 3
                sweep1_seed_results[m][p].append(avg_acc)
                print(f"    {m.upper()}: Average Accuracy = {avg_acc:.2f}%")

    # Compute means and stds for Sweep 1
    sweep1_results = {m: [] for m in methods}
    sweep1_stds = {m: [] for m in methods}
    for m in methods:
        for p in p_levels:
            accs = sweep1_seed_results[m][p]
            sweep1_results[m].append(np.mean(accs))
            sweep1_stds[m].append(np.std(accs))

    # Print Sweep 1 Table
    print("\n=== SWEEP 1 SUMMARY TABLE (Average Multi-Task Accuracy % ± Std) ===")
    print(f"{'Corruption p':<12} | {'Uncalibrated':<15} | {'SP-TAAC':<15} | {'Standard TAAC':<18} | {'SLF-TAAC':<15} | {'Proposed QRC':<18}")
    print("-" * 105)
    for idx, p in enumerate(p_levels):
        print(f"p = {p:.1f}         | "
              f"{sweep1_results['none'][idx]:.2f} ± {sweep1_stds['none'][idx]:.2f}% | "
              f"{sweep1_results['sp-taac'][idx]:.2f} ± {sweep1_stds['sp-taac'][idx]:.2f}% | "
              f"{sweep1_results['taac'][idx]:.2f} ± {sweep1_stds['taac'][idx]:.2f}% | "
              f"{sweep1_results['slf-taac'][idx]:.2f} ± {sweep1_stds['slf-taac'][idx]:.2f}% | "
              f"{sweep1_results['qrc'][idx]:.2f} ± {sweep1_stds['qrc'][idx]:.2f}%")

    # --- Sweep 1b & 1c: Alternative Corruption Types (Uniform and S&P) ---
    print("\n=== SWEEP 1b & 1c: Diverse Noise Families (p = [0.0, 0.2, 0.5], N=128) ===")
    alt_p_levels = [0.0, 0.2, 0.5]
    alt_noise_types = ['uniform', 'salt_and_pepper']
    alt_methods = ['taac', 'slf-taac', 'qrc']
    
    alt_results = {nt: {m: {p: [] for p in alt_p_levels} for m in alt_methods} for nt in alt_noise_types}
    
    for nt in alt_noise_types:
        print(f"\nEvaluating Noise Family: {nt.upper()}")
        for p in alt_p_levels:
            if p == 0.0:
                # clean is already computed in Sweep 1, copy results
                for m in alt_methods:
                    alt_results[nt][m][p] = sweep1_seed_results[m][0.0]
                continue
            print(f"  Corruption p = {p:.1f}:")
            for seed in seeds:
                cal_data_p = get_datasets(N=128, p_corruption=p, corruption_type=nt, seed=seed)
                for m in alt_methods:
                    cal_backbone = run_sequential_calibration(
                        copy.deepcopy(merged_wa_uncal), experts, cal_data_p, method=m, device=device
                    )
                    accs = []
                    for i, task in enumerate(tasks):
                        acc = evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device)
                        accs.append(acc)
                    avg_acc = sum(accs) / 3
                    alt_results[nt][m][p].append(avg_acc)
                    print(f"    Seed {seed} - {m.upper()}: {avg_acc:.2f}%")

    # Print Alt Noise summaries
    for nt in alt_noise_types:
        print(f"\nSummary for {nt.upper()} noise:")
        for p in alt_p_levels:
            print(f"  p = {p:.1f}:")
            for m in alt_methods:
                vals = alt_results[nt][m][p]
                print(f"    {m.upper()}: {np.mean(vals):.2f} ± {np.std(vals):.2f}%")

    # --- Sweep 2: Sample Size Budget (N) Sweep ---
    print("\n=== SWEEP 2: Calibration Budget (N) Sweep (Clean, WA Merge) ===")
    n_budgets = [16, 64, 128, 256]
    sweep2_methods = ['taac', 'slf-taac', 'qrc']
    sweep2_seed_results = {m: {N: [] for N in n_budgets} for m in sweep2_methods}
    
    for N in n_budgets:
        print(f"\n--- Testing Calibration Budget N = {N} ---")
        for seed in seeds:
            print(f"  Seed {seed}:")
            cal_data_n = get_datasets(N=N, p_corruption=0.0, seed=seed)
            for m in sweep2_methods:
                cal_backbone = run_sequential_calibration(
                    copy.deepcopy(merged_wa_uncal), experts, cal_data_n, method=m, device=device
                )
                accs = []
                for i, task in enumerate(tasks):
                    acc = evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device)
                    accs.append(acc)
                avg_acc = sum(accs) / 3
                sweep2_seed_results[m][N].append(avg_acc)
                print(f"    {m.upper()} (N={N}): Average Accuracy = {avg_acc:.2f}%")

    # Compute means and stds for Sweep 2
    sweep2_results = {m: [] for m in sweep2_methods}
    sweep2_stds = {m: [] for m in sweep2_methods}
    for m in sweep2_methods:
        for N in n_budgets:
            accs = sweep2_seed_results[m][N]
            sweep2_results[m].append(np.mean(accs))
            sweep2_stds[m].append(np.std(accs))

    # Print Sweep 2 Table
    print("\n=== SWEEP 2 SUMMARY TABLE (Average Multi-Task Accuracy % ± Std) ===")
    print(f"{'Sample Budget N':<15} | {'Standard TAAC':<18} | {'SLF-TAAC':<18} | {'Proposed QRC':<18}")
    print("-" * 75)
    for idx, N in enumerate(n_budgets):
        print(f"N = {N:<12} | "
              f"{sweep2_results['taac'][idx]:.2f} ± {sweep2_stds['taac'][idx]:.2f}% | "
              f"{sweep2_results['slf-taac'][idx]:.2f} ± {sweep2_stds['slf-taac'][idx]:.2f}% | "
              f"{sweep2_results['qrc'][idx]:.2f} ± {sweep2_stds['qrc'][idx]:.2f}%")

    # --- Sweep 3: Ablation of Robust QRC ---
    print("\n=== SWEEP 3: Ablation of QRC Components (p = 0.2 Corruption, N=128, WA Merge) ===")
    ablation_methods = ['none', 'taac', 'qrc-median', 'qrc-iqr', 'qrc']
    ablation_seed_results = {m: [] for m in ablation_methods}
    
    for seed in seeds:
        print(f"  Seed {seed}:")
        cal_data_abl = get_datasets(N=128, p_corruption=0.2, corruption_type='gaussian', seed=seed)
        for m in ablation_methods:
            if m == 'none':
                accs = [evaluate_backbone(merged_wa_uncal, experts[i], test_loaders[task], device) for i, task in enumerate(tasks)]
            else:
                cal_backbone = run_sequential_calibration(
                    copy.deepcopy(merged_wa_uncal), experts, cal_data_abl, method=m, device=device
                )
                accs = [evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device) for i, task in enumerate(tasks)]
            avg_acc = sum(accs) / 3
            ablation_seed_results[m].append(avg_acc)
            print(f"    {m.upper()}: Average Accuracy = {avg_acc:.2f}%")

    ablation_results = {}
    ablation_stds = {}
    for m in ablation_methods:
        ablation_results[m] = np.mean(ablation_seed_results[m])
        ablation_stds[m] = np.std(ablation_seed_results[m])

    print("\n=== ABLATION STUDY RESULTS (Mean ± Std) ===")
    print(f"  Uncalibrated Merge:            {ablation_results['none']:.2f} ± {ablation_stds['none']:.2f}%")
    print(f"  Standard Non-Robust TAAC:      {ablation_results['taac']:.2f} ± {ablation_stds['taac']:.2f}%")
    print(f"  QRC Component: Median-Only:    {ablation_results['qrc-median']:.2f} ± {ablation_stds['qrc-median']:.2f}%")
    print(f"  QRC Component: IQR-Only:       {ablation_results['qrc-iqr']:.2f} ± {ablation_stds['qrc-iqr']:.2f}%")
    print(f"  Complete QRC (Median + IQR):   {ablation_results['qrc']:.2f} ± {ablation_stds['qrc']:.2f}%")

    # --- Sweep 4: Task Arithmetic (TA) Merging Verification ---
    print("\n=== SWEEP 4: Task Arithmetic (TA, λ=0.3) Merging and Calibration (N=128, p=0.2) ===")
    ta_methods = ['none', 'taac', 'qrc']
    ta_seed_results = {m: [] for m in ta_methods}
    for seed in seeds:
        print(f"  Seed {seed}:")
        cal_data_ta = get_datasets(N=128, p_corruption=0.2, corruption_type='gaussian', seed=seed)
        for m in ta_methods:
            if m == 'none':
                accs = [evaluate_backbone(merged_ta_uncal, experts[i], test_loaders[task], device) for i, task in enumerate(tasks)]
            else:
                cal_backbone = run_sequential_calibration(
                    copy.deepcopy(merged_ta_uncal), experts, cal_data_ta, method=m, device=device
                )
                accs = [evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device) for i, task in enumerate(tasks)]
            avg_acc = sum(accs) / 3
            ta_seed_results[m].append(avg_acc)
            print(f"    TA Merge + {m.upper()}: Average Accuracy = {avg_acc:.2f}%")

    ta_results = {}
    ta_stds = {}
    for m in ta_methods:
        ta_results[m] = np.mean(ta_seed_results[m])
        ta_stds[m] = np.std(ta_seed_results[m])

    print("\n=== TASK ARITHMETIC MERGING SUMMARY (Mean ± Std) ===")
    for m in ta_methods:
        print(f"  TA Merge + {m.upper()}: {ta_results[m]:.2f} ± {ta_stds[m]:.2f}%")

    # --- Sweep 5: Quantile Width Sweep (IQR vs IDR vs Q95) ---
    print("\n=== SWEEP 5: Quantile Width Hyperparameter Sweep (Gaussian Noise, N=128, WA Merge) ===")
    q_p_levels = [0.0, 0.2, 0.5]
    q_methods = ['qrc', 'qrc-idr', 'qrc-95']
    q_seed_results = {m: {p: [] for p in q_p_levels} for m in q_methods}

    for p in q_p_levels:
        print(f"\n  Testing Corruption Level p = {p:.1f}:")
        for seed in seeds:
            cal_data_q = get_datasets(N=128, p_corruption=p, corruption_type='gaussian', seed=seed)
            for m in q_methods:
                cal_backbone = run_sequential_calibration(
                    copy.deepcopy(merged_wa_uncal), experts, cal_data_q, method=m, device=device
                )
                accs = []
                for i, task in enumerate(tasks):
                    acc = evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device)
                    accs.append(acc)
                avg_acc = sum(accs) / 3
                q_seed_results[m][p].append(avg_acc)
                print(f"    Seed {seed} - {m.upper()}: {avg_acc:.2f}%")

    # Print Sweep 5 Summary
    print("\n=== SWEEP 5 SUMMARY TABLE (Mean ± Std) ===")
    print(f"{'Quantile Option':<18} | {'p = 0.0 (Clean)':<18} | {'p = 0.2 (20% Noise)':<18} | {'p = 0.5 (50% Noise)':<18}")
    print("-" * 80)
    for m in q_methods:
        name_map = {'qrc': 'IQR (Q75 - Q25)', 'qrc-idr': 'IDR (Q90 - Q10)', 'qrc-95': 'Extreme (Q95 - Q05)'}
        print(f"{name_map[m]:<18} | "
              f"{np.mean(q_seed_results[m][0.0]):.2f} ± {np.std(q_seed_results[m][0.0]):.2f}% | "
              f"{np.mean(q_seed_results[m][0.2]):.2f} ± {np.std(q_seed_results[m][0.2]):.2f}% | "
              f"{np.mean(q_seed_results[m][0.5]):.2f} ± {np.std(q_seed_results[m][0.5]):.2f}%")

    # --- Sweep 6: Activation Threshold Sensitivity Sweep (tau) ---
    print("\n=== SWEEP 6: Activation Threshold (tau) Sensitivity Study (Gaussian Noise, N=128, WA Merge) ===")
    tau_p_levels = [0.0, 0.2, 0.5]
    tau_methods = ['qrc-tau-0.0', 'qrc-tau-1e-7', 'qrc-tau-1e-5', 'qrc-tau-1e-3', 'qrc-tau-1e-1']
    tau_seed_results = {m: {p: [] for p in tau_p_levels} for m in tau_methods}

    for p in tau_p_levels:
        print(f"\n  Testing Corruption Level p = {p:.1f}:")
        for seed in seeds:
            cal_data_tau = get_datasets(N=128, p_corruption=p, corruption_type='gaussian', seed=seed)
            for m in tau_methods:
                cal_backbone = run_sequential_calibration(
                    copy.deepcopy(merged_wa_uncal), experts, cal_data_tau, method=m, device=device
                )
                accs = []
                for i, task in enumerate(tasks):
                    acc = evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device)
                    accs.append(acc)
                avg_acc = sum(accs) / 3
                tau_seed_results[m][p].append(avg_acc)
                print(f"    Seed {seed} - {m.upper()}: {avg_acc:.2f}%")

    # Print Sweep 6 Summary
    print("\n=== SWEEP 6 SUMMARY TABLE (Mean ± Std) ===")
    print(f"{'Threshold tau':<18} | {'p = 0.0 (Clean)':<18} | {'p = 0.2 (20% Noise)':<18} | {'p = 0.5 (50% Noise)':<18}")
    print("-" * 80)
    for m in tau_methods:
        name_map = {
            'qrc-tau-0.0': 'tau = 0 (No Filter)',
            'qrc-tau-1e-7': 'tau = 1e-7',
            'qrc-tau-1e-5': 'tau = 1e-5 (Default)',
            'qrc-tau-1e-3': 'tau = 1e-3',
            'qrc-tau-1e-1': 'tau = 1e-1'
        }
        print(f"{name_map[m]:<18} | "
              f"{np.mean(tau_seed_results[m][0.0]):.2f} ± {np.std(tau_seed_results[m][0.0]):.2f}% | "
              f"{np.mean(tau_seed_results[m][0.2]):.2f} ± {np.std(tau_seed_results[m][0.2]):.2f}% | "
              f"{np.mean(tau_seed_results[m][0.5]):.2f} ± {np.std(tau_seed_results[m][0.5]):.2f}%")

    # Save results to npz for graphing or writing later
    np.savez('evaluation_results.npz',
             p_levels=p_levels, sweep1_results=sweep1_results, sweep1_stds=sweep1_stds, sweep1_seed_results=sweep1_seed_results,
             n_budgets=n_budgets, sweep2_results=sweep2_results, sweep2_stds=sweep2_stds, sweep2_seed_results=sweep2_seed_results,
             ablation_results=ablation_results, ablation_stds=ablation_stds, ablation_seed_results=ablation_seed_results,
             ta_results=ta_results, ta_stds=ta_stds, ta_seed_results=ta_seed_results,
             alt_results=alt_results, q_seed_results=q_seed_results,
             tau_seed_results=tau_seed_results,
             oracle_accs=oracle_accs)
    print("\nAll evaluation experiments complete. Results saved to evaluation_results.npz.")

if __name__ == '__main__':
    main()
