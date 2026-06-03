import os
import random
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set deterministic seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors

# Dataset wrapper to handle 3-channel duplicate and resize
class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Convert grayscale to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_datasets(data_dir="./data", subset_size=5000):
    os.makedirs(data_dir, exist_ok=True)
    
    # Common transform: resize to 32x32, convert to tensor, normalize with ImageNet stats
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download raw datasets
    raw_mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    raw_mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    
    raw_fashion_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True)
    raw_fashion_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True)
    
    raw_cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    raw_cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    # Wrap datasets
    mnist_train = ProcessedDataset(raw_mnist_train, transform=transform)
    mnist_test = ProcessedDataset(raw_mnist_test, transform=transform)
    
    fashion_train = ProcessedDataset(raw_fashion_train, transform=transform)
    fashion_test = ProcessedDataset(raw_fashion_test, transform=transform)
    
    cifar_train = ProcessedDataset(raw_cifar_train, transform=transform)
    cifar_test = ProcessedDataset(raw_cifar_test, transform=transform)
    
    # Create deterministic subsets for expert training
    def get_subset(dataset, size, seed=42):
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
        return Subset(dataset, indices)
        
    mnist_train_sub = get_subset(mnist_train, subset_size)
    fashion_train_sub = get_subset(fashion_train, subset_size)
    cifar_train_sub = get_subset(cifar_train, subset_size)
    
    return {
        'mnist': {'train': mnist_train_sub, 'test': mnist_test, 'full_train': mnist_train},
        'fashion': {'train': fashion_train_sub, 'test': fashion_test, 'full_train': fashion_train},
        'cifar': {'train': cifar_train_sub, 'test': cifar_test, 'full_train': cifar_train}
    }

# Multi-task model architecture wrapper
class MultiTaskResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load ImageNet pretrained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # We replace the final FC layer with task-specific heads
        # During single-task expert training, we only use the active head.
        self.backbone.fc = nn.Identity() # Remove default head
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(512, num_classes),
            'fashion': nn.Linear(512, num_classes),
            'cifar': nn.Linear(512, num_classes)
        })

    def forward(self, x, task_name):
        features = self.backbone(x)
        logits = self.heads[task_name](features)
        return logits

# Train single task expert
def train_expert(model, train_loader, task_name, epochs=5, lr=5e-4, wd=1e-4, device='cuda'):
    model.to(device)
    model.train()
    
    # We only train the backbone and the active task head
    optimizer = optim.AdamW(
        list(model.backbone.parameters()) + list(model.heads[task_name].parameters()),
        lr=lr, weight_decay=wd
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, task_name)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"[{task_name.upper()} Expert] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

# Evaluate model on test set
@torch.no_grad()
def evaluate_model(model, test_loader, task_name, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs, task_name)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = correct / total * 100.0
    return acc

# Forward hooks for sequential calibration
class CalibrationHook:
    def __init__(self, name):
        self.name = name
        self.method = None
        self.is_early_layer = True
        
        # Scaling/shift parameters
        self.scale = None  # Can be scalar (SP-TAAC) or channel-wise tensor
        self.bias = None   # Channel-wise bias tensor

    def __call__(self, module, input, output):
        # Apply calibration on the output of BatchNorm2d
        if self.scale is None:
            return output
            
        if isinstance(self.scale, torch.Tensor):
            # Channel-wise scaling
            # Output shape: [batch, channels, height, width]
            # scale shape: [1, channels, 1, 1], bias shape: [1, channels, 1, 1]
            scaled = output * self.scale
            if self.bias is not None:
                scaled = scaled + self.bias
            return scaled
        else:
            # Global scalar scaling
            return output * self.scale

def get_bn_layers(model):
    # Returns a list of (name, module) of all BatchNorm2d layers in the backbone in order of execution
    bn_layers = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers

# Sequentially calibrate the model
def run_calibration(merged_model, experts, datasets, method, N=128, split_layer_idx=15, alpha=0.5, seed=42, device='cuda'):
    print(f"\n--- Running Sequential Calibration ({method}) with N={N}, seed={seed} ---")
    set_seed(seed)
    
    # 1. Prepare calibration samples
    cal_loaders = {}
    joint_cal_samples = []
    
    for task_name, d_dict in datasets.items():
        # Deterministically select N samples from full training set
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(d_dict['full_train']), generator=g)[:N].tolist()
        subset = Subset(d_dict['full_train'], indices)
        cal_loaders[task_name] = DataLoader(subset, batch_size=32, shuffle=False)
        
        # Collect joint calibration samples
        for imgs, _ in cal_loaders[task_name]:
            joint_cal_samples.append(imgs)
            
    # Joint calibration dataset as a tensor
    joint_cal_tensor = torch.cat(joint_cal_samples, dim=0).to(device) # shape: [3*N, 3, 32, 32]
    
    # 2. Get BN layers in backbone
    merged_bn = get_bn_layers(merged_model)
    experts_bn = {task: get_bn_layers(exp_model) for task, exp_model in experts.items()}
    
    # Register calibration hooks
    merged_hooks = {}
    merged_handles = []
    for idx, (name, module) in enumerate(merged_bn):
        hook = CalibrationHook(name)
        # Determine if this is an early or deep layer
        hook.is_early_layer = (idx < split_layer_idx)
        merged_hooks[name] = hook
        handle = module.register_forward_hook(hook)
        merged_handles.append(handle)
        
    # We will also register simple activation logging hooks on expert models to capture their activations
    expert_activations = {task: {} for task in experts}
    expert_handles = []
    
    def make_log_hook(task, layer_name):
        def log_hook(module, input, output):
            expert_activations[task][layer_name] = output.detach().cpu()
        return log_hook
        
    for task, exp_model in experts.items():
        exp_model.to(device)
        exp_model.eval()
        for name, module in experts_bn[task]:
            handle = module.register_forward_hook(make_log_hook(task, name))
            expert_handles.append(handle)
            
    # Run a single forward pass on expert models with their respective task calibration loaders to populate expert_activations
    with torch.no_grad():
        for task, loader in cal_loaders.items():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                _ = experts[task](imgs, task)
                
    # Remove expert logging hooks to avoid memory leaks
    for handle in expert_handles:
        handle.remove()
        
    # Now, sequentially calibrate the merged model layer-by-layer
    merged_model.to(device)
    merged_model.eval()
    
    epsilon = 1e-5
    
    for idx, (bn_name, bn_module) in enumerate(merged_bn):
        # 1. Gather expert activations for this layer across all tasks
        # Each has shape [N, channels, H, W] in CPU
        exp_acts = [expert_activations[task][bn_name] for task in experts]
        
        # Compute target statistics
        # We need the global layer-wise variance and channel-wise mean/variance of experts
        expert_global_stds = []
        expert_channel_means = []
        expert_channel_vars = []
        
        for act in exp_acts:
            # act shape: [N, channels, H, W]
            # Global standard deviation across all batch, channel, spatial dims
            g_var = torch.var(act, dim=(0, 1, 2, 3), unbiased=False)
            expert_global_stds.append(torch.sqrt(g_var + epsilon))
            
            # Channel-wise mean and variance
            c_mean = torch.mean(act, dim=(0, 2, 3)) # shape: [channels]
            c_var = torch.var(act, dim=(0, 2, 3), unbiased=False) # shape: [channels]
            expert_channel_means.append(c_mean)
            expert_channel_vars.append(c_var)
            
        target_global_std = torch.mean(torch.stack(expert_global_stds))
        target_channel_mean = torch.mean(torch.stack(expert_channel_means), dim=0).to(device) # [channels]
        target_channel_std = torch.mean(torch.stack([torch.sqrt(v + epsilon) for v in expert_channel_vars]), dim=0).to(device) # [channels]
        
        # 2. Extract uncalibrated merged running statistics (from the BN layer itself)
        # Note: running_mean and running_var of the merged BN are the weight averages of the experts' running stats
        merged_bn_running_mean = bn_module.running_mean # [channels]
        merged_bn_running_var = bn_module.running_var   # [channels]
        
        # 3. Gather merged model's current activations on the joint calibration set
        # Since hooks on layers 0 ... idx-1 are already active and calibrated, passing joint_cal_tensor through
        # the model will produce the correctly conditioned input to layer idx!
        current_merged_act = []
        
        # We can use a temporary hook on the target BN to capture its uncalibrated output at this step
        temp_act = None
        def temp_hook(module, input, output):
            nonlocal temp_act
            temp_act = output.detach()
            
        temp_handle = bn_module.register_forward_hook(temp_hook)
        with torch.no_grad():
            _ = merged_model(joint_cal_tensor, 'mnist') # Task head doesn't affect backbone activations
        temp_handle.remove()
        
        # temp_act has shape [3*N, channels, H, W]
        # Compute merged statistics on joint calibration set
        merged_global_var = torch.var(temp_act, dim=(0, 1, 2, 3), unbiased=False)
        merged_global_std = torch.sqrt(merged_global_var + epsilon)
        
        merged_channel_mean = torch.mean(temp_act, dim=(0, 2, 3)) # [channels]
        merged_channel_var = torch.var(temp_act, dim=(0, 2, 3), unbiased=False) # [channels]
        merged_channel_std = torch.sqrt(merged_channel_var + epsilon)
        
        # 4. Determine calibration parameters based on method and layer index
        hook = merged_hooks[bn_name]
        
        current_method = method
        if method == 'HSC':
            # Hybrid Selective Calibration
            # Early layers: SP-TAAC (global scaling)
            # Deep layers: Channel-wise regularized or standard calibration
            if hook.is_early_layer:
                current_method = 'SP-TAAC'
            else:
                current_method = 'R-TAAC' # Use regularized channel-wise calibration in deep layers
                
        if current_method == 'SP-TAAC':
            # Global layer-wise scaling
            gamma = target_global_std / merged_global_std
            hook.scale = gamma.item()
            hook.bias = None
            
        elif current_method == 'N-TAAC':
            # Pure channel-wise calibration
            scale = target_channel_std / merged_channel_std
            bias = target_channel_mean - scale * merged_channel_mean
            
            hook.scale = scale.view(1, -1, 1, 1)
            hook.bias = bias.view(1, -1, 1, 1)
            
        elif current_method == 'R-TAAC':
            # Regularized channel-wise calibration using shrinkage
            # Shrink joint empirical statistics towards the uncalibrated weight-averaged running statistics
            shrunk_mean = alpha * merged_channel_mean + (1 - alpha) * merged_bn_running_mean
            shrunk_var = alpha * merged_channel_var + (1 - alpha) * merged_bn_running_var
            shrunk_std = torch.sqrt(shrunk_var + epsilon)
            
            scale = target_channel_std / shrunk_std
            bias = target_channel_mean - scale * shrunk_mean
            
            hook.scale = scale.view(1, -1, 1, 1)
            hook.bias = bias.view(1, -1, 1, 1)
            
        else:
            # Uncalibrated (no-op)
            hook.scale = None
            hook.bias = None

    # Return handles to allow cleaning them up if needed
    return merged_handles

# Head supervised fine-tuning (SFT)
def run_head_sft(model, datasets, N=128, epochs=15, lr=1e-3, seed=42, device='cuda'):
    print(f"\n--- Running Head Supervised Fine-Tuning (SFT) with N={N}, epochs={epochs}, lr={lr} ---")
    set_seed(seed)
    model.to(device)
    
    # Freeze the entire backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze only the task classification heads
    for head in model.heads.values():
        for param in head.parameters():
            param.requires_grad = True
            
    criterion = nn.CrossEntropyLoss()
    
    for task_name, d_dict in datasets.items():
        # Deterministically select the SAME N samples used for calibration
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(d_dict['full_train']), generator=g)[:N].tolist()
        subset = Subset(d_dict['full_train'], indices)
        train_loader = DataLoader(subset, batch_size=16, shuffle=True)
        
        optimizer = optim.AdamW(model.heads[task_name].parameters(), lr=lr, weight_decay=1e-4)
        
        model.eval() # We keep the backbone in eval mode (specifically BatchNorm)
        for epoch in range(epochs):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs, task_name)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
        print(f"[{task_name.upper()} Head] Adapted successfully over {epochs} epochs.")
        
    # Set back all requires_grad to True just in case
    for param in model.parameters():
        param.requires_grad = True

# Main execution routine
def main():
    parser = argparse.ArgumentParser(description="Hybrid Selective Calibration for Model Merging")
    parser.add_argument('--mode', type=str, default='sanity_check', choices=['sanity_check', 'train_experts', 'experiments'],
                        help="Execution mode")
    parser.add_argument('--epochs', type=int, default=5, help="Epochs for training experts")
    parser.add_argument('--cal_size', type=int, default=128, help="Calibration size N")
    parser.add_argument('--split_idx', type=int, default=15, help="Layer index to split early vs deep layers in HSC")
    parser.add_argument('--alpha', type=type(0.5), default=0.5, help="Shrinkage parameter alpha for R-TAAC / HSC")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--seeds', type=str, default='42', help="Comma-separated list of random seeds to run sweeps and average over")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.mode == 'sanity_check':
        print("\n=== RUNNING CPU SANITY CHECK ===")
        set_seed(args.seed)
        
        # Load small dataset
        datasets = get_datasets(subset_size=50) # Tiny subset of 50 images
        
        # Create models
        expert_mnist = MultiTaskResNet18()
        expert_fashion = MultiTaskResNet18()
        expert_cifar = MultiTaskResNet18()
        
        # CPU Training: 1 epoch, 10 images each to make sure backpropagation and shapes work
        mnist_loader = DataLoader(datasets['mnist']['train'], batch_size=10, shuffle=True)
        fashion_loader = DataLoader(datasets['fashion']['train'], batch_size=10, shuffle=True)
        cifar_loader = DataLoader(datasets['cifar']['train'], batch_size=10, shuffle=True)
        
        print("\n--- Training MNIST Expert (Sanity) ---")
        train_expert(expert_mnist, mnist_loader, 'mnist', epochs=1, device=device)
        print("\n--- Training Fashion Expert (Sanity) ---")
        train_expert(expert_fashion, fashion_loader, 'fashion', epochs=1, device=device)
        print("\n--- Training CIFAR Expert (Sanity) ---")
        train_expert(expert_cifar, cifar_loader, 'cifar', epochs=1, device=device)
        
        # Check test evaluation
        test_loader = DataLoader(datasets['mnist']['test'], batch_size=10)
        acc = evaluate_model(expert_mnist, test_loader, 'mnist', device=device)
        print(f"MNIST expert accuracy on subset: {acc:.2f}%")
        
        # Merge models (Weight Averaging)
        print("\n--- Merging Experts ---")
        merged_model = MultiTaskResNet18()
        
        # Merge backbone parameters
        for name, param in merged_model.backbone.named_parameters():
            param.data = (expert_mnist.backbone.state_dict()[name].data +
                          expert_fashion.backbone.state_dict()[name].data +
                          expert_cifar.backbone.state_dict()[name].data) / 3.0
                          
        # Merge running stats of BatchNorm layers
        for name, module in merged_model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                mnist_bn = expert_mnist.backbone.state_dict()[name + '.running_mean']
                fashion_bn = expert_fashion.backbone.state_dict()[name + '.running_mean']
                cifar_bn = expert_cifar.backbone.state_dict()[name + '.running_mean']
                module.running_mean.copy_((mnist_bn + fashion_bn + cifar_bn) / 3.0)
                
                mnist_var = expert_mnist.backbone.state_dict()[name + '.running_var']
                fashion_var = expert_fashion.backbone.state_dict()[name + '.running_var']
                cifar_var = expert_cifar.backbone.state_dict()[name + '.running_var']
                module.running_var.copy_((mnist_var + fashion_var + cifar_var) / 3.0)
                
        # Copy task-specific heads directly
        merged_model.heads['mnist'].load_state_dict(expert_mnist.heads['mnist'].state_dict())
        merged_model.heads['fashion'].load_state_dict(expert_fashion.heads['fashion'].state_dict())
        merged_model.heads['cifar'].load_state_dict(expert_cifar.heads['cifar'].state_dict())
        
        experts = {
            'mnist': expert_mnist,
            'fashion': expert_fashion,
            'cifar': expert_cifar
        }
        
        # Test calibration methods
        for method in ['uncalibrated', 'SP-TAAC', 'N-TAAC', 'R-TAAC', 'HSC']:
            print(f"\nEvaluating {method}...")
            # Create a clean copy of the merged model
            m_copy = copy.deepcopy(merged_model)
            
            # Calibrate (using tiny N=4)
            handles = run_calibration(m_copy, experts, datasets, method, N=4, split_layer_idx=15, alpha=args.alpha, seed=args.seed, device=device)
            
            # Evaluate on a small test set
            m_accs = []
            for t in ['mnist', 'fashion', 'cifar']:
                t_loader = DataLoader(datasets[t]['test'], batch_size=10)
                # Just evaluate on first batch for speed
                sub_indices = list(range(10))
                sub_test = Subset(datasets[t]['test'], sub_indices)
                sub_loader = DataLoader(sub_test, batch_size=10)
                acc = evaluate_model(m_copy, sub_loader, t, device=device)
                m_accs.append(acc)
                print(f"-> {t.upper()} accuracy under {method}: {acc:.2f}%")
            print(f"Average accuracy under {method}: {sum(m_accs)/3.0:.2f}%")
            
            # Clean up hooks
            for h in handles:
                h.remove()
                
        # Test Head SFT
        print("\n--- Testing Head SFT (Sanity) ---")
        m_copy = copy.deepcopy(merged_model)
        handles = run_calibration(m_copy, experts, datasets, 'HSC', N=4, split_layer_idx=15, alpha=args.alpha, seed=args.seed, device=device)
        run_head_sft(m_copy, datasets, N=4, epochs=1, lr=1e-3, seed=args.seed, device=device)
        acc = evaluate_model(m_copy, DataLoader(Subset(datasets['mnist']['test'], list(range(10))), batch_size=10), 'mnist', device=device)
        print(f"MNIST Adapted head accuracy on subset: {acc:.2f}%")
        for h in handles:
            h.remove()
            
        print("\n=== CPU SANITY CHECK SUCCESSFUL ===")
        
    elif args.mode == 'train_experts':
        print("\n=== TRAINING FULL RESNET18 EXPERTS ===")
        os.makedirs("experts", exist_ok=True)
        set_seed(args.seed)
        
        # Load dataset of size 5,000
        datasets = get_datasets(subset_size=5000)
        
        # Train and save each expert
        for task in ['mnist', 'fashion', 'cifar']:
            print(f"\n--- Training {task.upper()} Expert ---")
            model = MultiTaskResNet18()
            loader = DataLoader(datasets[task]['train'], batch_size=128, shuffle=True, num_workers=4)
            train_expert(model, loader, task, epochs=args.epochs, device=device)
            
            # Evaluate on test set
            test_loader = DataLoader(datasets[task]['test'], batch_size=256, num_workers=4)
            acc = evaluate_model(model, test_loader, task, device=device)
            print(f"{task.upper()} Expert Test Accuracy: {acc:.2f}%")
            
            # Save checkpoint
            torch.save(model.state_dict(), f"experts/{task}_expert.pth")
            print(f"Saved {task}_expert.pth")
            
    elif args.mode == 'experiments':
        print("\n=== RUNNING CALIBRATION EXPERIMENTS ===")
        set_seed(args.seed)
        
        # Check that expert models are trained
        for task in ['mnist', 'fashion', 'cifar']:
            if not os.path.exists(f"experts/{task}_expert.pth"):
                raise ValueError(f"Expert model experts/{task}_expert.pth not found! Run with --mode train_experts first.")
                
        # Load dataset
        datasets = get_datasets(subset_size=5000)
        
        # Create and load expert models
        experts = {}
        for task in ['mnist', 'fashion', 'cifar']:
            exp_model = MultiTaskResNet18()
            exp_model.load_state_dict(torch.load(f"experts/{task}_expert.pth", map_location=device))
            experts[task] = exp_model
            
        # Create uncalibrated merged model
        merged_model = MultiTaskResNet18()
        for name, param in merged_model.backbone.named_parameters():
            param.data = (experts['mnist'].backbone.state_dict()[name].data +
                          experts['fashion'].backbone.state_dict()[name].data +
                          experts['cifar'].backbone.state_dict()[name].data) / 3.0
                          
        for name, module in merged_model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                m_mean = experts['mnist'].backbone.state_dict()[name + '.running_mean']
                f_mean = experts['fashion'].backbone.state_dict()[name + '.running_mean']
                c_mean = experts['cifar'].backbone.state_dict()[name + '.running_mean']
                module.running_mean.copy_((m_mean + f_mean + c_mean) / 3.0)
                
                m_var = experts['mnist'].backbone.state_dict()[name + '.running_var']
                f_var = experts['fashion'].backbone.state_dict()[name + '.running_var']
                c_var = experts['cifar'].backbone.state_dict()[name + '.running_var']
                module.running_var.copy_((m_var + f_var + c_var) / 3.0)
                
        merged_model.heads['mnist'].load_state_dict(experts['mnist'].heads['mnist'].state_dict())
        merged_model.heads['fashion'].load_state_dict(experts['fashion'].heads['fashion'].state_dict())
        merged_model.heads['cifar'].load_state_dict(experts['cifar'].heads['cifar'].state_dict())
        
        # We will sweep across calibration size N, split_idx, and alpha!
        # Evaluation loaders
        test_loaders = {
            t: DataLoader(datasets[t]['test'], batch_size=256, num_workers=4) for t in ['mnist', 'fashion', 'cifar']
        }
        
        seeds_list = [int(s.strip()) for s in args.seeds.split(',')]
        
        # 1. Sweep over calibration size N and methods
        n_sweep = [4, 16, 64, 128, 256]
        methods_sweep = ['uncalibrated', 'SP-TAAC', 'N-TAAC', 'R-TAAC', 'HSC']
        
        main_results = { (N, method): [] for N in n_sweep for method in methods_sweep }
        
        # 2. Sweep over split index (at N=128)
        split_sweep = [0, 5, 10, 15, 19]
        split_results = { split_val: [] for split_val in split_sweep }
        
        # 3. Sweep over alpha (at N=128, Split=15)
        alpha_sweep = [0.0, 0.25, 0.5, 0.75, 1.0]
        alpha_results = { a_val: [] for a_val in alpha_sweep }
        
        for s in seeds_list:
            print(f"\n" + "="*80)
            print(f"RUNNING ALL EXPERIMENTS WITH SEED {s}")
            print("="*80)
            
            # 1. Sweep over N and methods
            for N in n_sweep:
                for method in methods_sweep:
                    m_copy = copy.deepcopy(merged_model)
                    
                    # Perform calibration
                    handles = run_calibration(
                        m_copy, experts, datasets, method, 
                        N=N, split_layer_idx=args.split_idx, alpha=args.alpha, seed=s, device=device
                    )
                    
                    # Evaluate
                    accs = {}
                    for t in ['mnist', 'fashion', 'cifar']:
                        accs[t] = evaluate_model(m_copy, test_loaders[t], t, device=device)
                    mean_acc = sum(accs.values()) / 3.0
                    
                    print(f"[EVAL] Seed={s:4d} | N={N:3d} | Method: {method:12s} | MNIST: {accs['mnist']:.2f}% | F-MNIST: {accs['fashion']:.2f}% | CIFAR: {accs['cifar']:.2f}% | Mean: {mean_acc:.2f}%")
                    
                    # SFT evaluation (if SFT is applied on top of this representation calibration)
                    sft_accs = {}
                    # Create a copy for head adaptation to keep representation-only metrics clean
                    sft_m_copy = copy.deepcopy(m_copy)
                    run_head_sft(sft_m_copy, datasets, N=N, epochs=15, lr=1e-3, seed=s, device=device)
                    
                    for t in ['mnist', 'fashion', 'cifar']:
                        sft_accs[t] = evaluate_model(sft_m_copy, test_loaders[t], t, device=device)
                    mean_sft_acc = sum(sft_accs.values()) / 3.0
                    print(f"[EVAL + SFT] Seed={s:4d} | N={N:3d} | Method: {method:12s} + SFT | MNIST: {sft_accs['mnist']:.2f}% | F-MNIST: {sft_accs['fashion']:.2f}% | CIFAR: {sft_accs['cifar']:.2f}% | Mean: {mean_sft_acc:.2f}%")
                    
                    main_results[(N, method)].append({
                        'mnist': accs['mnist'],
                        'fashion': accs['fashion'],
                        'cifar': accs['cifar'],
                        'mean': mean_acc,
                        'sft_mnist': sft_accs['mnist'],
                        'sft_fashion': sft_accs['fashion'],
                        'sft_cifar': sft_accs['cifar'],
                        'sft_mean': mean_sft_acc
                    })
                    
                    for h in handles:
                        h.remove()
                        
            # 2. Sweep over split layer index for HSC (at N=128)
            print(f"\n--- Sweeping split index in HSC (N=128) with Seed {s} ---")
            for split_val in split_sweep:
                m_copy = copy.deepcopy(merged_model)
                handles = run_calibration(
                    m_copy, experts, datasets, 'HSC', 
                    N=128, split_layer_idx=split_val, alpha=args.alpha, seed=s, device=device
                )
                accs = {}
                for t in ['mnist', 'fashion', 'cifar']:
                    accs[t] = evaluate_model(m_copy, test_loaders[t], t, device=device)
                mean_acc = sum(accs.values()) / 3.0
                print(f"[HSC Split Sweep] Seed={s:4d} | Split index: {split_val:2d} | Mean: {mean_acc:.2f}%")
                
                # Plus SFT
                sft_m_copy = copy.deepcopy(m_copy)
                run_head_sft(sft_m_copy, datasets, N=128, epochs=15, lr=1e-3, seed=s, device=device)
                sft_accs = {}
                for t in ['mnist', 'fashion', 'cifar']:
                    sft_accs[t] = evaluate_model(sft_m_copy, test_loaders[t], t, device=device)
                mean_sft_acc = sum(sft_accs.values()) / 3.0
                print(f"[HSC Split Sweep + SFT] Seed={s:4d} | Split index: {split_val:2d} + SFT | Mean: {mean_sft_acc:.2f}%")
                
                split_results[split_val].append({
                    'mean': mean_acc,
                    'sft_mean': mean_sft_acc
                })
                
                for h in handles:
                    h.remove()

            # 3. Sweep over alpha shrinkage parameter (at N=128, Split=15)
            print(f"\n--- Sweeping alpha shrinkage parameter in HSC (N=128, Split=15) with Seed {s} ---")
            for a_val in alpha_sweep:
                m_copy = copy.deepcopy(merged_model)
                handles = run_calibration(
                    m_copy, experts, datasets, 'HSC', 
                    N=128, split_layer_idx=15, alpha=a_val, seed=s, device=device
                )
                accs = {}
                for t in ['mnist', 'fashion', 'cifar']:
                    accs[t] = evaluate_model(m_copy, test_loaders[t], t, device=device)
                mean_acc = sum(accs.values()) / 3.0
                print(f"[HSC Alpha Sweep] Seed={s:4d} | Alpha: {a_val:.2f} | Mean: {mean_acc:.2f}%")
                
                # Plus SFT
                sft_m_copy = copy.deepcopy(m_copy)
                run_head_sft(sft_m_copy, datasets, N=128, epochs=15, lr=1e-3, seed=s, device=device)
                sft_accs = {}
                for t in ['mnist', 'fashion', 'cifar']:
                    sft_accs[t] = evaluate_model(sft_m_copy, test_loaders[t], t, device=device)
                mean_sft_acc = sum(sft_accs.values()) / 3.0
                print(f"[HSC Alpha Sweep + SFT] Seed={s:4d} | Alpha: {a_val:.2f} + SFT | Mean: {mean_sft_acc:.2f}%")
                
                alpha_results[a_val].append({
                    'mean': mean_acc,
                    'sft_mean': mean_sft_acc
                })
                
                for h in handles:
                    h.remove()

        # Print final formatted summary table
        import math
        def format_mean_std(values_list):
            if not values_list:
                return "N/A"
            if len(values_list) == 1:
                return f"{values_list[0]:.2f}%"
            mean = sum(values_list) / len(values_list)
            var = sum((x - mean) ** 2 for x in values_list) / (len(values_list) - 1)
            std = math.sqrt(var)
            return f"{mean:.2f}% ± {std:.2f}%"

        print("\n" + "="*115)
        print("FINAL MULTI-SEED AGGREGATED RESULTS TABLE")
        print("="*115)
        print(f"{'N':4s} | {'Method':12s} | {'MNIST':18s} | {'F-MNIST':18s} | {'CIFAR':18s} | {'Mean':18s} | {'Mean+SFT':18s}")
        print("-"*115)
        for N in n_sweep:
            for method in methods_sweep:
                run_list = main_results[(N, method)]
                mnist_str = format_mean_std([r['mnist'] for r in run_list])
                fashion_str = format_mean_std([r['fashion'] for r in run_list])
                cifar_str = format_mean_std([r['cifar'] for r in run_list])
                mean_str = format_mean_std([r['mean'] for r in run_list])
                sft_str = format_mean_std([r['sft_mean'] for r in run_list])
                print(f"{N:4d} | {method:12s} | {mnist_str:18s} | {fashion_str:18s} | {cifar_str:18s} | {mean_str:18s} | {sft_str:18s}")
        print("="*115)

        print("\n" + "="*60)
        print("HSC SPLIT SWEEP (N=128)")
        print("="*60)
        print(f"{'Split Index':12s} | {'Mean Acc':20s} | {'Mean+SFT Acc':20s}")
        print("-"*60)
        for split_val in split_sweep:
            run_list = split_results[split_val]
            mean_str = format_mean_std([r['mean'] for r in run_list])
            sft_str = format_mean_std([r['sft_mean'] for r in run_list])
            print(f"{split_val:12d} | {mean_str:20s} | {sft_str:20s}")
        print("="*60)

        print("\n" + "="*60)
        print("HSC ALPHA SWEEP (N=128, Split=15)")
        print("="*60)
        print(f"{'Alpha':12s} | {'Mean Acc':20s} | {'Mean+SFT Acc':20s}")
        print("-"*60)
        for a_val in alpha_sweep:
            run_list = alpha_results[a_val]
            mean_str = format_mean_std([r['mean'] for r in run_list])
            sft_str = format_mean_std([r['sft_mean'] for r in run_list])
            print(f"{a_val:12.2f} | {mean_str:20s} | {sft_str:20s}")
        print("="*60)

if __name__ == "__main__":
    main()
