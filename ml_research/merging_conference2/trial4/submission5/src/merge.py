import os
import argparse
import copy
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

def get_datasets(data_dir):
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_mnist = datasets.MNIST(data_dir, train=True, download=True, transform=transform_mnist)
    test_mnist = datasets.MNIST(data_dir, train=False, download=True, transform=transform_mnist)
    
    train_fashion = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_mnist)
    test_fashion = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform_mnist)
    
    train_cifar = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_cifar)
    test_cifar = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_cifar)
    
    return {
        'mnist': (train_mnist, test_mnist),
        'fashion': (train_fashion, test_fashion),
        'cifar': (train_cifar, test_cifar)
    }

def find_bn_layers(model):
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers

def get_expert_models(experts_dir, device):
    experts = {}
    tasks = ['mnist', 'fashion', 'cifar']
    for task in tasks:
        model = models.resnet18()
        model.fc = nn.Linear(512, 10)
        path = os.path.join(experts_dir, f"{task}_expert.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model for {task} not found at {path}. Run train_experts.py first.")
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        # Disable inplace ReLU to prevent backward hook errors
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        experts[task] = model
    return experts

def compute_fisher_information(experts, calibration_sets, device):
    # Compute the activation Fisher Information for each expert model
    print("--- Computing Activation Fisher Information ---")
    fisher_info = {}
    
    for task, model in experts.items():
        model.eval()
        cal_loader = DataLoader(calibration_sets[task], batch_size=32, shuffle=False)
        bn_layers = find_bn_layers(model)
        
        # Dictionary to accumulate squared gradients
        task_fisher = {name: torch.zeros(module.num_features, device=device) for name, module in bn_layers}
        
        # Register forward hooks that register tensor backward hooks
        hooks = []
        def make_forward_hook(layer_name):
            def hook(module, input, output):
                cloned_out = output.clone()
                def tensor_hook(grad):
                    grad_sq = grad.pow(2).sum(dim=(0, 2, 3))
                    task_fisher[layer_name] += grad_sq.detach()
                    return grad
                cloned_out.register_hook(tensor_hook)
                return cloned_out
            return hook
            
        for name, module in bn_layers:
            hooks.append(module.register_forward_hook(make_forward_hook(name)))
            
        # Run forward and backward passes
        criterion = nn.CrossEntropyLoss()
        for inputs, labels in cal_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Normalize Fisher Information per layer so its mean is 1.0
        normalized_task_fisher = {}
        for name, grads_sq in task_fisher.items():
            # Divide by total samples to get average squared gradient
            grads_avg = grads_sq / len(calibration_sets[task])
            mean_val = grads_avg.mean()
            normalized_task_fisher[name] = grads_avg / (mean_val + 1e-8)
            
        fisher_info[task] = normalized_task_fisher
        
    return fisher_info

def collect_expert_statistics(experts, calibration_sets, device):
    # Collect expert statistics (mean and std) on their respective calibration sets
    print("--- Collecting Expert Statistics ---")
    expert_stats = {}
    
    for task, model in experts.items():
        model.eval()
        cal_loader = DataLoader(calibration_sets[task], batch_size=32, shuffle=False)
        bn_layers = find_bn_layers(model)
        
        # Accumulators
        all_activations = {name: [] for name, _ in bn_layers}
        
        # Forward hook to collect output of each BatchNorm layer
        hooks = []
        def make_forward_hook(layer_name):
            def hook(module, input, output):
                all_activations[layer_name].append(output.detach())
            return hook
            
        for name, module in bn_layers:
            hooks.append(module.register_forward_hook(make_forward_hook(name)))
            
        # Run forward pass
        with torch.no_grad():
            for inputs, _ in cal_loader:
                inputs = inputs.to(device)
                _ = model(inputs)
                
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Compute mean and standard deviation per channel
        task_stats = {}
        for name, acts_list in all_activations.items():
            # Concatenate along batch dimension
            acts = torch.cat(acts_list, dim=0) # Shape: (N, channels, H, W)
            # Compute mean and standard deviation per channel (dims: 0, 2, 3)
            mean = acts.mean(dim=(0, 2, 3))
            std = acts.std(dim=(0, 2, 3), unbiased=False)
            task_stats[name] = {'mean': mean, 'std': std}
            
        expert_stats[task] = task_stats
        
    return expert_stats

def build_merged_model(experts, method='wa', lam=0.3, base_model=None):
    # Construct the merged model backbone
    tasks = list(experts.keys())
    merged_model = copy.deepcopy(experts[tasks[0]])
    
    if method == 'wa':
        print("Merging models via Weight Averaging (WA)...")
        # Average state dicts
        merged_state = merged_model.state_dict()
        expert_states = [experts[t].state_dict() for t in tasks]
        
        for key in merged_state.keys():
            # Skip classification head 'fc'
            if 'fc' in key:
                continue
            merged_state[key] = torch.stack([state[key].float() for state in expert_states]).mean(dim=0)
            
        merged_model.load_state_dict(merged_state)
        
    elif method == 'ta':
        print(f"Merging models via Task Arithmetic (TA, lambda={lam})...")
        if base_model is None:
            # Load standard ImageNet pre-trained ResNet-18 as base
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(next(merged_model.parameters()).device)
            
        base_state = base_model.state_dict()
        expert_states = {t: experts[t].state_dict() for t in tasks}
        merged_state = merged_model.state_dict()
        
        for key in merged_state.keys():
            if 'fc' in key:
                continue
            # Task vectors
            task_vectors = []
            for t in tasks:
                task_vector = expert_states[t][key].float() - base_state[key].float()
                task_vectors.append(task_vector)
                
            # Merged weights
            merged_state[key] = base_state[key].float() + lam * torch.stack(task_vectors).sum(dim=0)
            
        merged_model.load_state_dict(merged_state)
        
    return merged_model

def apply_calibration(merged_model, expert_stats, joint_cal_set, method, fisher_info=None, fwas_lambda=0.5, device='cuda'):
    print(f"--- Applying Calibration: {method.upper()} ---")
    
    bn_layers = find_bn_layers(merged_model)
    tasks = list(expert_stats.keys())
    num_bn = len(bn_layers)
    
    # Pre-computed target statistics averaged over experts
    target_stats = {}
    for name, _ in bn_layers:
        means = torch.stack([expert_stats[t][name]['mean'] for t in tasks])
        stds = torch.stack([expert_stats[t][name]['std'] for t in tasks])
        target_stats[name] = {
            'mean': means.mean(dim=0),
            'std': stds.mean(dim=0)
        }
        
    # We will compute the calibration parameters sequentially layer-by-layer (SeqCalib)
    # to resolve the Parallel Collection Flaw
    
    # Store final calibration parameters (scale and bias) for each BN layer
    cal_params = {name: {'scale': torch.ones(module.num_features, device=device), 
                         'bias': torch.zeros(module.num_features, device=device)}
                  for name, module in bn_layers}
                  
    # Active hooks list
    active_calibration_hooks = []
    
    # Sequential statistic collection loop
    for k in range(num_bn):
        layer_name, layer_module = bn_layers[k]
        
        # 1. Register forward hooks on ALL preceding layers j < k using their finalized parameters
        # Remove any previous preceding hooks to avoid duplicates
        for h in active_calibration_hooks:
            h.remove()
        active_calibration_hooks = []
        
        for j in range(k):
            prec_name, prec_module = bn_layers[j]
            scale = cal_params[prec_name]['scale'].view(1, -1, 1, 1)
            bias = cal_params[prec_name]['bias'].view(1, -1, 1, 1)
            
            def make_cal_hook(s, b):
                return lambda m, inp, out: out * s + b
                
            active_calibration_hooks.append(
                prec_module.register_forward_hook(make_cal_hook(scale, bias))
            )
            
        # 2. Register a temporary collection hook on the CURRENT layer k
        current_acts = []
        def collection_hook(module, inp, out):
            current_acts.append(out.detach())
            
        collection_h = layer_module.register_forward_hook(collection_hook)
        
        # 3. Run a forward pass over the joint calibration set
        joint_loader = DataLoader(joint_cal_set, batch_size=32, shuffle=False)
        with torch.no_grad():
            for inputs, _ in joint_loader:
                inputs = inputs.to(device)
                _ = merged_model(inputs)
                
        # Remove the temporary collection hook
        collection_h.remove()
        
        # 4. Compute statistics of the merged model at layer k
        acts = torch.cat(current_acts, dim=0) # Shape: (N_joint, channels, H, W)
        merged_mean = acts.mean(dim=(0, 2, 3))
        merged_std = acts.std(dim=(0, 2, 3), unbiased=False)
        
        # 5. Compute the calibration parameters based on the chosen method
        target_mean = target_stats[layer_name]['mean']
        target_std = target_stats[layer_name]['std']
        
        num_features = layer_module.num_features
        
        if method == 'none':
            scale = torch.ones(num_features, device=device)
            bias = torch.zeros(num_features, device=device)
            
        elif method == 'taac':
            # Unregularized Task-Agnostic Activation Calibration
            scale = target_std / (merged_std + 1e-8)
            bias = target_mean - scale * merged_mean
            
        elif method == 'r-taac':
            # Regularized Task-Agnostic Activation Calibration (static shrinkage alpha=0.8)
            alpha = 0.8
            # Shrink merged statistics towards uncalibrated weight-averaged running statistics (expert priors)
            # For simplicity, we use the average of the experts' running statistics
            prior_mean = torch.stack([layer_module.running_mean for _ in tasks]).mean(dim=0)
            prior_var = torch.stack([layer_module.running_var for _ in tasks]).mean(dim=0)
            prior_std = torch.sqrt(prior_var + 1e-8)
            
            shrunk_mean = alpha * merged_mean + (1.0 - alpha) * prior_mean
            shrunk_std = alpha * merged_std + (1.0 - alpha) * prior_std
            
            scale = target_std / (shrunk_std + 1e-8)
            bias = target_mean - scale * shrunk_mean
            
        elif method == 's-tcac':
            # Shrinkage-TCAC (static shrinkage alpha=0.2 towards layer average)
            alpha = 0.2
            layer_mean_scalar = merged_mean.mean()
            layer_std_scalar = merged_std.mean()
            
            shrunk_mean = (1.0 - alpha) * merged_mean + alpha * layer_mean_scalar
            shrunk_std = (1.0 - alpha) * merged_std + alpha * layer_std_scalar
            
            scale = target_std / (shrunk_std + 1e-8)
            bias = target_mean - scale * shrunk_mean
            
        elif method == 'sp-taac':
            # Sparsity-Preserving Task-Agnostic Calibration (layer-wise scaling only)
            # Compute global standard deviations across all channels
            global_target_std = torch.sqrt( (target_std.pow(2)).mean() + 1e-8 )
            global_merged_std = torch.sqrt( (merged_std.pow(2)).mean() + 1e-8 )
            
            gamma = global_target_std / (global_merged_std + 1e-8)
            scale = torch.ones(num_features, device=device) * gamma
            bias = torch.zeros(num_features, device=device)
            
        elif method == 'fwas':
            # Our proposed Fisher-Weighted Activation Shrinkage!
            # 1. Compute unregularized TAAC parameters
            taac_scale = target_std / (merged_std + 1e-8)
            taac_bias = target_mean - taac_scale * merged_mean
            
            # 2. Compute SP-TAAC parameters (global layer-wise scale, zero bias)
            global_target_std = torch.sqrt( (target_std.pow(2)).mean() + 1e-8 )
            global_merged_std = torch.sqrt( (merged_merged_std := merged_std).pow(2).mean() + 1e-8 ) # dummy to avoid name clash
            gamma = global_target_std / (global_merged_std + 1e-8)
            
            # 3. Compute joint Fisher Information by averaging experts
            layer_fisher = torch.stack([fisher_info[t][layer_name] for t in tasks]).mean(dim=0)
            
            # 4. Get spatial dimensions of the activations
            H = acts.shape[2]
            W = acts.shape[3]
            spatial_factor = H * W
            
            # 5. Compute channel-specific shrinkage factors alphas scaled by spatial resolution
            # alphas = lambda / (lambda + Fisher * H * W)
            # Highly sensitive or early, high-resolution layers -> alphas -> 0 -> use TAAC
            # Insensitive and deep, low-resolution layers -> alphas -> 1 -> use SP-TAAC
            alphas = fwas_lambda / (fwas_lambda + layer_fisher * spatial_factor + 1e-8)
            
            # 6. Interpolate scale and bias directly (SP-TAAC has bias 0.0)
            scale = (1.0 - alphas) * taac_scale + alphas * gamma
            bias = (1.0 - alphas) * taac_bias
            
        cal_params[layer_name]['scale'] = scale
        cal_params[layer_name]['bias'] = bias
        
    # Remove preceding hooks
    for h in active_calibration_hooks:
        h.remove()
        
    # Register the FINAL calibration hooks on the merged model
    final_hooks = []
    for name, module in bn_layers:
        scale = cal_params[name]['scale'].view(1, -1, 1, 1)
        bias = cal_params[name]['bias'].view(1, -1, 1, 1)
        
        def make_final_hook(s, b):
            return lambda m, inp, out: out * s + b
            
        final_hooks.append(
            module.register_forward_hook(make_final_hook(scale, bias))
        )
        
    return final_hooks

def evaluate_multi_task(model, experts, test_datasets, device):
    results = {}
    tasks = list(experts.keys())
    
    # Create copy of classification heads
    heads = {t: copy.deepcopy(experts[t].fc) for t in tasks}
    
    for task in tasks:
        # Load the task's classification head onto the merged backbone
        model.fc = heads[task]
        model.eval()
        
        test_loader = DataLoader(test_datasets[task], batch_size=128, shuffle=False, num_workers=2)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        results[task] = 100.0 * correct / total
        
    results['avg'] = sum([results[t] for t in tasks]) / len(tasks)
    return results

def run_reda(merged_model, experts, calibration_sets, test_datasets, device):
    # REDA: applying N-TAAC (same as TAAC sequential) first, then adapting classification heads on the calibration set for 15 epochs
    print("--- Running REDA Framework ---")
    
    # 1. Apply backbone calibration (N-TAAC)
    expert_stats = collect_expert_statistics(experts, calibration_sets, device)
    
    # Construct joint calibration set
    joint_cal_indices = []
    for t in experts.keys():
        joint_cal_indices.append(calibration_sets[t])
    from torch.utils.data import ConcatDataset
    joint_cal_set = ConcatDataset(joint_cal_indices)
    
    cal_hooks = apply_calibration(merged_model, expert_stats, joint_cal_set, 'taac', device=device)
    
    # 2. Freeze calibrated backbone
    for param in merged_model.parameters():
        param.requires_grad = False
        
    tasks = list(experts.keys())
    adapted_heads = {}
    
    # 3. Adapt heads
    for task in tasks:
        print(f"Adapting classification head for {task.upper()}...")
        # Get head from expert model
        head = copy.deepcopy(experts[task].fc).to(device)
        for param in head.parameters():
            param.requires_grad = True
            
        merged_model.fc = head
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        
        cal_loader = DataLoader(calibration_sets[task], batch_size=32, shuffle=True)
        
        for epoch in range(15):
            merged_model.train()
            # Ensure backbone layers remain in eval mode to use calibration hooks properly
            # and freeze BatchNorm statistics update
            for name, module in merged_model.named_modules():
                if name != 'fc':
                    module.eval()
                    
            for inputs, labels in cal_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = merged_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        adapted_heads[task] = copy.deepcopy(head).eval()
        
    # Evaluate REDA
    results = {}
    for task in tasks:
        merged_model.fc = adapted_heads[task]
        merged_model.eval()
        
        test_loader = DataLoader(test_datasets[task], batch_size=128, shuffle=False, num_workers=2)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = merged_model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        results[task] = 100.0 * correct / total
        
    results['avg'] = sum([results[t] for t in tasks]) / len(tasks)
    
    # Clean up hooks
    for h in cal_hooks:
        h.remove()
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Calibration Methods")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--experts_dir", type=str, default="./experts", help="Directory for expert models")
    parser.add_argument("--cal_size", type=int, default=128, help="Calibration set size per task")
    parser.add_argument("--fwas_lambda", type=float, default=0.5, help="Regularization parameter for FWAS")
    parser.add_argument("--method", type=str, default="wa", choices=["wa", "ta"], help="Model merging method")
    parser.add_argument("--lam", type=float, default=0.3, help="Scaling factor for Task Arithmetic")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    experts = get_expert_models(args.experts_dir, device)
    all_datasets = get_datasets(args.data_dir)
    
    # Prepare task-specific calibration and test sets
    calibration_sets = {}
    test_datasets = {}
    
    for task in experts.keys():
        train_ds, test_ds = all_datasets[task]
        # Calibration subset: deterministic
        g = torch.Generator().manual_seed(100)
        cal_indices = torch.randperm(len(train_ds), generator=g)[:args.cal_size].tolist()
        calibration_sets[task] = Subset(train_ds, cal_indices)
        test_datasets[task] = test_ds
        
    from torch.utils.data import ConcatDataset
    joint_cal_set = ConcatDataset([calibration_sets[t] for t in experts.keys()])
    
    # 1. Compute Fisher Information on-the-fly using the calibration sets
    fisher_info = compute_fisher_information(experts, calibration_sets, device)
    
    # 2. Collect expert statistics
    expert_stats = collect_expert_statistics(experts, calibration_sets, device)
    
    # Methods to evaluate
    calibration_methods = ['none', 'sp-taac', 'taac', 'r-taac', 's-tcac', 'fwas']
    
    print("\n================ EVALUATION RESULTS ================")
    
    # Baseline experts
    print("Evaluating individual experts:")
    for task, model in experts.items():
        test_loader = DataLoader(test_datasets[task], batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"  {task.upper()} Expert: {100.0 * correct / total:.2f}%")
        
    for cal_method in calibration_methods:
        # Build merged model backbone
        merged_model = build_merged_model(experts, method=args.method, lam=args.lam)
        
        # Apply calibration
        cal_hooks = apply_calibration(
            merged_model=merged_model,
            expert_stats=expert_stats,
            joint_cal_set=joint_cal_set,
            method=cal_method,
            fisher_info=fisher_info,
            fwas_lambda=args.fwas_lambda,
            device=device
        )
        
        # Evaluate
        scores = evaluate_multi_task(merged_model, experts, test_datasets, device)
        print(f"Calibration Method [{cal_method.upper()}] Results:")
        for t in experts.keys():
            print(f"  {t.upper()}: {scores[t]:.2f}%")
        print(f"  Average: {scores['avg']:.2f}%")
        
        # Clean up hooks
        for h in cal_hooks:
            h.remove()
            
    # Run REDA
    merged_model = build_merged_model(experts, method=args.method, lam=args.lam)
    reda_scores = run_reda(merged_model, experts, calibration_sets, test_datasets, device)
    print("Calibration Method [REDA] Results:")
    for t in experts.keys():
        print(f"  {t.upper()}: {reda_scores[t]:.2f}%")
    print(f"  Average: {reda_scores['avg']:.2f}%")
