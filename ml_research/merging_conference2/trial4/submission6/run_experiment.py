import os
import sys
import copy
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Disable cuDNN to bypass cluster-specific cuDNN initialization errors
        torch.backends.cudnn.enabled = False

class CalibrationHook:
    def __init__(self, name, mode='none'):
        self.name = name
        self.mode = mode
        self.stats_list = []
        self.params = {}
        self.active = False
        self.recording = False
        
    def reset_stats(self):
        self.stats_list = []
        
    def __call__(self, module, input, output):
        if self.recording:
            with torch.no_grad():
                self.stats_list.append(output.detach().clone())
        elif self.active:
            with torch.no_grad():
                if self.mode == 'taac':
                    mean = self.params['mean'].to(output.device)
                    std = self.params['std'].to(output.device)
                    target_mean = self.params['target_mean'].to(output.device)
                    target_std = self.params['target_std'].to(output.device)
                    
                    eps = 1e-5
                    denom = torch.clamp(std, min=eps)
                    scaled = (output - mean.view(1, -1, 1, 1)) / denom.view(1, -1, 1, 1)
                    calibrated = scaled * target_std.view(1, -1, 1, 1) + target_mean.view(1, -1, 1, 1)
                    output.copy_(calibrated)
                    
                elif self.mode == 'sp_taac':
                    scale = self.params['scale'].to(output.device)
                    output.mul_(scale)
                    
                elif self.mode == 'qspa':
                    B, C, H, W = output.shape
                    real = output[:, 0::2, :, :]
                    imag = output[:, 1::2, :, :]
                    z = torch.complex(real, imag)
                    
                    rotation = self.params['rotation'].to(output.device)
                    z_rotated = z * rotation.view(1, -1, 1, 1)
                    
                    if 'amp_scale' in self.params:
                        amp_scale = self.params['amp_scale'].to(output.device)
                        z_rotated = z_rotated * amp_scale.view(1, -1, 1, 1)
                    
                    output[:, 0::2, :, :] = z_rotated.real
                    output[:, 1::2, :, :] = z_rotated.imag
                    
                elif self.mode == 'pra':
                    B, C, H, W = output.shape
                    flat = output.view(B, C // 4, 4, H, W).permute(1, 2, 0, 3, 4).reshape(C // 4, 4, B * H * W)
                    
                    rotation = self.params['rotation'].to(output.device) # shape (C//4, 4, 4)
                    amp_scale = self.params['amp_scale'].to(output.device) # shape (C//4, 1)
                    target_mean = self.params['target_mean'].to(output.device) # shape (C//4, 4, 1)
                    mean_m = self.params['mean_m'].to(output.device) # shape (C//4, 4, 1)
                    
                    flat_c = flat - mean_m
                    
                    flat_scaled = flat_c * amp_scale.view(-1, 1, 1)
                    flat_rotated = torch.bmm(rotation, flat_scaled)
                    flat_aligned = flat_rotated + target_mean
                    
                    aligned = flat_aligned.view(C // 4, 4, B, H, W).permute(2, 0, 1, 3, 4).reshape(B, C, H, W)
                    output.copy_(aligned)
                    
        return output

def register_calibration_hooks(model, mode, layer_filter='all'):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if layer_filter == 'all' or layer_filter in name:
                hook = CalibrationHook(name, mode)
                handle = module.register_forward_hook(hook)
                hooks.append((hook, handle))
    return hooks

def load_datasets(debug=False):
    # Transforms
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Datasets
    os.makedirs("./data", exist_ok=True)
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_fmnist)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_fmnist)
    
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_cifar)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar)

    if debug:
        # Mini subsets for debugging
        mnist_train = Subset(mnist_train, list(range(100)))
        mnist_test = Subset(mnist_test, list(range(50)))
        fmnist_train = Subset(fmnist_train, list(range(100)))
        fmnist_test = Subset(fmnist_test, list(range(50)))
        cifar_train = Subset(cifar_train, list(range(100)))
        cifar_test = Subset(cifar_test, list(range(50)))

    return {
        'mnist': (mnist_train, mnist_test),
        'fmnist': (fmnist_train, fmnist_test),
        'cifar10': (cifar_train, cifar_test)
    }

def train_expert(task, train_dataset, device, num_epochs=5, batch_size=128):
    print(f"--- Training Expert for {task} ---")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    return model

def evaluate_model(model, task_heads, test_loaders, device):
    model.eval()
    results = {}
    for task, loader in test_loaders.items():
        # Plug in task head
        model.fc = task_heads[task].to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[task] = (correct / total) * 100.0
    return results

def merge_models(experts, coefficients):
    merged_model = copy.deepcopy(experts[list(experts.keys())[0]])
    merged_state_dict = merged_model.state_dict()
    for key in merged_state_dict.keys():
        if not key.startswith('fc.'):
            weighted_sum = 0.0
            for task, expert in experts.items():
                weighted_sum += coefficients[task] * expert.state_dict()[key].float()
            merged_state_dict[key] = weighted_sum
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def get_joint_calibration_loader(train_datasets, N, seed=42):
    subsets = []
    g = torch.Generator().manual_seed(seed)
    for task, dataset in train_datasets.items():
        indices = torch.randperm(len(dataset), generator=g)[:N].tolist()
        subsets.append(Subset(dataset, indices))
    joint_dataset = ConcatDataset(subsets)
    # Use larger batch size to feed all calibration samples in 1 or 2 steps
    return DataLoader(joint_dataset, batch_size=512, shuffle=False)

def record_activations(model, hooks, loader, device):
    model.eval()
    # Reset and activate recording
    for hook, _ in hooks:
        hook.reset_stats()
        hook.recording = True
        hook.active = False
        
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
            
    # Deactivate recording and pool activations
    pooled_activations = {}
    for hook, _ in hooks:
        hook.recording = False
        if len(hook.stats_list) > 0:
            pooled_activations[hook.name] = torch.cat(hook.stats_list, dim=0)
        hook.reset_stats()
    return pooled_activations

def calibrate_model(merged_model, merged_hooks, experts_activations, merged_activations, mode='none'):
    if mode == 'none':
        for hook, _ in merged_hooks:
            hook.active = False
        return

    for hook, _ in merged_hooks:
        name = hook.name
        hook.mode = mode
        
        if mode == 'taac':
            # Extract activations
            act_merged = merged_activations[name]
            mean_merged = act_merged.mean(dim=(0, 2, 3))
            std_merged = act_merged.std(dim=(0, 2, 3), unbiased=False)
            
            means_experts = []
            stds_experts = []
            for task in experts_activations.keys():
                act_exp = experts_activations[task][name]
                means_experts.append(act_exp.mean(dim=(0, 2, 3)))
                stds_experts.append(act_exp.std(dim=(0, 2, 3), unbiased=False))
                
            target_mean = torch.stack(means_experts).mean(dim=0)
            target_std = torch.stack(stds_experts).mean(dim=0)
            
            hook.params = {
                'mean': mean_merged.cpu(),
                'std': std_merged.cpu(),
                'target_mean': target_mean.cpu(),
                'target_std': target_std.cpu()
            }
            hook.active = True
            
        elif mode == 'sp_taac':
            act_merged = merged_activations[name]
            std_global_merged = act_merged.std(unbiased=False)
            
            stds_experts = []
            for task in experts_activations.keys():
                act_exp = experts_activations[task][name]
                stds_experts.append(act_exp.std(unbiased=False))
                
            target_std_global = torch.stack(stds_experts).mean()
            
            eps = 1e-12
            scale = target_std_global / (std_global_merged + eps)
            
            hook.params = {
                'scale': scale.cpu()
            }
            hook.active = True
            
        elif mode == 'qspa':
            # Quantum Superposition Phase Alignment (QSPA v2: Bloch Covariance Alignment)
            act_merged = merged_activations[name]
            B, C, H, W = act_merged.shape
            assert C % 2 == 0, f"BatchNorm channel count {C} must be even for QSPA complex pairing"
            
            eps = 1e-12
            reg_eps = 1e-4  # Bayes-style Tikhonov regularization constant to avoid sparsity trap
            
            # Merged model complex activations covariance
            real_m = act_merged[:, 0::2, :, :]
            imag_m = act_merged[:, 1::2, :, :]
            
            # Compute means
            mean_real_m = real_m.mean(dim=(0, 2, 3), keepdim=True)
            mean_imag_m = imag_m.mean(dim=(0, 2, 3), keepdim=True)
            
            real_m_c = real_m - mean_real_m
            imag_m_c = imag_m - mean_imag_m
            
            # Covariance matrix elements
            sig_rr_m = (real_m_c ** 2).mean(dim=(0, 2, 3))
            sig_ii_m = (imag_m_c ** 2).mean(dim=(0, 2, 3))
            sig_ri_m = (real_m_c * imag_m_c).mean(dim=(0, 2, 3))
            
            # Total variance (invariant under rotation)
            V_m = sig_rr_m + sig_ii_m
            
            # Bloch coordinates for merged
            x_m = sig_rr_m - sig_ii_m
            y_m = 2.0 * sig_ri_m
            w_m = torch.complex(x_m, y_m)
            
            # Experts
            w_experts = []
            V_experts = []
            for task in experts_activations.keys():
                act_exp = experts_activations[task][name]
                real_e = act_exp[:, 0::2, :, :]
                imag_e = act_exp[:, 1::2, :, :]
                
                mean_real_e = real_e.mean(dim=(0, 2, 3), keepdim=True)
                mean_imag_e = imag_e.mean(dim=(0, 2, 3), keepdim=True)
                
                real_e_c = real_e - mean_real_e
                imag_e_c = imag_e - mean_imag_e
                
                sig_rr_e = (real_e_c ** 2).mean(dim=(0, 2, 3))
                sig_ii_e = (imag_e_c ** 2).mean(dim=(0, 2, 3))
                sig_ri_e = (real_e_c * imag_e_c).mean(dim=(0, 2, 3))
                
                V_e = sig_rr_e + sig_ii_e
                V_experts.append(V_e)
                
                x_e = sig_rr_e - sig_ii_e
                y_e = 2.0 * sig_ri_e
                w_experts.append(torch.complex(x_e, y_e))
                
            # Consensus Bloch vector and target variance
            w_consensus = torch.stack(w_experts).mean(dim=0)
            V_target = torch.stack(V_experts).mean(dim=0)
            
            # Compute phase rotation angle
            # We align the phase of w_m to w_consensus. Since phase(w) = 2 * theta, theta = 0.5 * delta_phase.
            theta = 0.5 * torch.angle(w_consensus * torch.conj(w_m))
            
            # Regularize phase rotation using anisotropy/confidence
            # If the distribution is near-circular (isotropic), rotation is undefined and we shouldn't rotate.
            anisotropy_m = torch.clamp(torch.abs(w_m) / (V_m + eps), 0.0, 1.0)
            anisotropy_consensus = torch.clamp(torch.abs(w_consensus) / (V_target + eps), 0.0, 1.0)
            confidence = anisotropy_m * anisotropy_consensus
            theta_reg = theta * confidence
            
            # Compute amplitude scale with Tikhonov regularization
            amp_scale = torch.sqrt((V_target + reg_eps) / (V_m + reg_eps))
            
            # Prepare complex rotation phasor polar(1, theta)
            rotation = torch.polar(torch.ones_like(theta_reg), theta_reg)
            
            hook.params = {
                'rotation': rotation.cpu(),
                'amp_scale': amp_scale.cpu()
            }
            hook.active = True
            
        elif mode == 'pra':
            # Procrustes Representation Alignment (PRA: 4-Dimensional SVD Procrustes)
            act_merged = merged_activations[name]
            B, C, H, W = act_merged.shape
            assert C % 4 == 0, f"BatchNorm channel count {C} must be a multiple of 4 for PRA"
            
            # Reshape merged activations to (C//4, 4, B*H*W)
            flat_m = act_merged.view(B, C // 4, 4, H, W).permute(1, 2, 0, 3, 4).reshape(C // 4, 4, B * H * W)
            K_samples = B * H * W
            
            # Compute merged mean and center
            mean_m = flat_m.mean(dim=2, keepdim=True) # (C//4, 4, 1)
            flat_m_c = flat_m - mean_m
            
            # Compute consensus activations and center them
            expert_flats = []
            expert_means = []
            for task in experts_activations.keys():
                act_exp = experts_activations[task][name]
                flat_e = act_exp.view(B, C // 4, 4, H, W).permute(1, 2, 0, 3, 4).reshape(C // 4, 4, B * H * W)
                expert_flats.append(flat_e)
                expert_means.append(flat_e.mean(dim=2, keepdim=True))
                
            # Consensus flat is the mean of expert flats
            flat_t = torch.stack(expert_flats).mean(dim=0) # (C//4, 4, B*H*W)
            target_mean = torch.stack(expert_means).mean(dim=0) # (C//4, 4, 1)
            flat_t_c = flat_t - target_mean
            
            # Compute cross-covariance matrix H = flat_t_c @ flat_m_c.T (C//4, 4, 4)
            H_cov = torch.bmm(flat_t_c, flat_m_c.transpose(1, 2)) / K_samples
            
            # Perform SVD: H = U S V^T
            U, S, Vh = torch.linalg.svd(H_cov) # U: (C//4, 4, 4), S: (C//4, 4), Vh: (C//4, 4, 4)
            
            # R = U @ V^T = U @ Vh
            R = torch.bmm(U, Vh)
            det_R = torch.linalg.det(R) # (C//4,)
            
            # Force proper rotation if det(R) < 0 by flipping the last column of U
            flip_mask = det_R < 0
            if flip_mask.any():
                U_flipped = U.clone()
                U_flipped[flip_mask, :, -1] = -U_flipped[flip_mask, :, -1]
                R = torch.bmm(U_flipped, Vh)
                
            # Compute amplitude scale using variance ratio: sqrt(Tr(Sigma_T) / Tr(Sigma_M))
            sig_m = torch.bmm(flat_m_c, flat_m_c.transpose(1, 2)) / K_samples
            sig_t = torch.bmm(flat_t_c, flat_t_c.transpose(1, 2)) / K_samples
            
            tr_m = torch.diagonal(sig_m, dim1=1, dim2=2).sum(dim=1, keepdim=True) # (C//4, 1)
            tr_t = torch.diagonal(sig_t, dim1=1, dim2=2).sum(dim=1, keepdim=True) # (C//4, 1)
            
            reg_eps = 1e-4
            amp_scale = torch.sqrt((tr_t + reg_eps) / (tr_m + reg_eps)) # (C//4, 1)
            
            hook.params = {
                'rotation': R.cpu(),
                'amp_scale': amp_scale.cpu(),
                'target_mean': target_mean.cpu(),
                'mean_m': mean_m.cpu()
            }
            hook.active = True

def run_head_adaptation(merged_model, experts_activations, merged_activations, train_datasets, test_loaders, task_heads, N, device, calibration_mode='none', lr=1e-3, epochs=10, layer_filter='all'):
    print(f"\n--- Adapting Heads with Calibration: {calibration_mode} (N={N}) ---")
    
    # Deepcopy models/heads to prevent side effects
    model_adapt = copy.deepcopy(merged_model)
    heads_adapt = {task: copy.deepcopy(head) for task, head in task_heads.items()}
    
    # Freeze backbone
    for p in model_adapt.parameters():
        p.requires_grad = False
        
    # Re-register hooks on the adapted model
    hooks_adapt = register_calibration_hooks(model_adapt, calibration_mode, layer_filter)
    
    # If using calibration, apply the params computed previously
    if calibration_mode != 'none':
        calibrate_model(model_adapt, hooks_adapt, experts_activations, merged_activations, mode=calibration_mode)
            
    # Train each head independently on its respective task training subset of size N
    criterion = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(42)
    
    for task, dataset in train_datasets.items():
        print(f"Adapting head for {task}...")
        indices = torch.randperm(len(dataset), generator=g)[:N].tolist()
        task_sub = Subset(dataset, indices)
        loader = DataLoader(task_sub, batch_size=min(32, N), shuffle=True)
        
        head = heads_adapt[task].to(device)
        head.train()
        optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                # Forward pass through backbone
                model_adapt.fc = nn.Identity()
                with torch.no_grad():
                    features = model_adapt(x)
                
                outputs = head(features)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
    # Evaluate
    results = evaluate_model(model_adapt, heads_adapt, test_loaders, device)
    
    # Clean up hooks
    for _, handle in hooks_adapt:
        handle.remove()
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Model Merging & Calibration Experiment")
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'], help="Operation mode")
    parser.add_argument('--debug', action='store_true', help="Run in quick debug mode on CPU")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--n_val', type=int, default=128, help="Calibration size N")
    parser.add_argument('--layer_filter', type=str, default='all', choices=['all', 'layer4'], help="BatchNorm layers to calibrate")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.debug else 'cpu')
    print(f"Using device: {device}")
    
    # Load Datasets
    data_dict = load_datasets(debug=args.debug)
    train_datasets = {task: data_dict[task][0] for task in data_dict}
    test_datasets = {task: data_dict[task][1] for task in data_dict}
    
    # Setup test loaders
    test_loaders = {
        task: DataLoader(test_datasets[task], batch_size=128 if not args.debug else 16, shuffle=False)
        for task in test_datasets
    }
    
    if args.mode == 'train':
        print("Training expert models...")
        experts = {}
        for task in train_datasets:
            # Subsample 3000 images for training if not debugging
            if not args.debug:
                g = torch.Generator().manual_seed(args.seed)
                indices = torch.randperm(len(train_datasets[task]), generator=g)[:3000].tolist()
                sub_dataset = Subset(train_datasets[task], indices)
            else:
                sub_dataset = train_datasets[task]
                
            model = train_expert(task, sub_dataset, device, num_epochs=1 if args.debug else 5)
            experts[task] = model
            torch.save(model.state_dict(), f"expert_{task}.pth")
            print(f"Saved expert_{task}.pth")
            
    elif args.mode == 'eval':
        # Check if expert weights exist
        expert_files = {task: f"expert_{task}.pth" for task in train_datasets}
        for task, file in expert_files.items():
            if not os.path.exists(file):
                print(f"Error: {file} not found. Please run with '--mode train' first.")
                sys.exit(1)
                
        print("Loading expert models...")
        expert_models = {}
        task_heads = {}
        for task, file in expert_files.items():
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(torch.load(file, map_location=device))
            model = model.to(device)
            expert_models[task] = model
            task_heads[task] = copy.deepcopy(model.fc)
            
        # 1. Evaluate Individual Experts as baseline
        print("\n--- Evaluating Individual Experts ---")
        for task, model in expert_models.items():
            loader = test_loaders[task]
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = correct / total * 100.0
            print(f"Expert {task} accuracy on {task} test set: {acc:.2f}%")
            
        # 2. Merge Models with uniform coefficients (1/3 each)
        coeffs = {task: 1/3 for task in expert_models}
        print(f"\nMerging models with coefficients: {coeffs}")
        merged_model = merge_models(expert_models, coeffs)
        
        # 3. Register calibration hooks
        # Create separate hook containers
        expert_hooks = {task: register_calibration_hooks(expert_models[task], 'none', args.layer_filter) for task in expert_models}
        merged_hooks = register_calibration_hooks(merged_model, 'none', args.layer_filter)
        
        # 4. Sweep over calibration sizes N
        N_sweep = [4, 16, 64, 128, 256] if not args.debug else [4, 16]
        
        print("\n==================================================")
        print("RUNNING CALIBRATION SWEEP")
        print("==================================================")
        
        results_all = {}
        
        for N in N_sweep:
            print(f"\n--- Calibrating with N = {N} samples per task ---")
            # Create joint task-agnostic calibration loader
            joint_cal_loader = get_joint_calibration_loader(train_datasets, N, seed=args.seed)
            
            # Record expert activations
            experts_activations = {}
            for task in expert_models:
                experts_activations[task] = record_activations(expert_models[task], expert_hooks[task], joint_cal_loader, device)
                
            # Record merged activations
            merged_activations = record_activations(merged_model, merged_hooks, joint_cal_loader, device)
            
            # Evaluate Calibration Methods
            calibration_methods = ['none', 'taac', 'sp_taac', 'qspa', 'pra']
            results_all[N] = {}
            
            for method in calibration_methods:
                # Calibrate merged model
                calibrate_model(merged_model, merged_hooks, experts_activations, merged_activations, mode=method)
                
                # Evaluate
                accs = evaluate_model(merged_model, task_heads, test_loaders, device)
                avg_acc = sum(accs.values()) / len(accs)
                results_all[N][method] = {
                    'accs': accs,
                    'average': avg_acc
                }
                
                print(f"Method: {method:10s} | MNIST: {accs['mnist']:.2f}% | F-MNIST: {accs['fmnist']:.2f}% | CIFAR-10: {accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")
                
            # 5. Evaluate REDA-style Head-only Adaptation with and without calibration
            # We do this for N samples
            print(f"\nRunning REDA Head-only Adaptation (N={N} labeled samples, frozen backbone)...")
            for method in ['none', 'qspa', 'sp_taac', 'taac', 'pra']:
                accs_adapt = run_head_adaptation(
                    merged_model, experts_activations, merged_activations, train_datasets, test_loaders, task_heads,
                    N=N, device=device, calibration_mode=method, lr=5e-4 if not args.debug else 1e-3, epochs=10 if not args.debug else 1,
                    layer_filter=args.layer_filter
                )
                avg_acc_adapt = sum(accs_adapt.values()) / len(accs_adapt)
                print(f"REDA ({method:7s} + Head Adapt) | MNIST: {accs_adapt['mnist']:.2f}% | F-MNIST: {accs_adapt['fmnist']:.2f}% | CIFAR-10: {accs_adapt['cifar10']:.2f}% | Average: {avg_acc_adapt:.2f}%")
                
        # 6. Sweep over task coefficients (Imbalance Regimes)
        print("\n==================================================")
        print("RUNNING COEFFICIENT SWEEP (N=128)")
        print("==================================================")
        
        imbalance_regimes = [
            {'mnist': 0.8, 'fmnist': 0.1, 'cifar10': 0.1},
            {'mnist': 0.1, 'fmnist': 0.8, 'cifar10': 0.1},
            {'mnist': 0.1, 'fmnist': 0.1, 'cifar10': 0.8}
        ] if not args.debug else [{'mnist': 0.8, 'fmnist': 0.1, 'cifar10': 0.1}]
        
        N_imbalance = 128 if not args.debug else 16
        joint_cal_loader = get_joint_calibration_loader(train_datasets, N_imbalance, seed=args.seed)
        
        for regime in imbalance_regimes:
            print(f"\nRegime: MNIST={regime['mnist']:.1f}, F-MNIST={regime['fmnist']:.1f}, CIFAR={regime['cifar10']:.1f}")
            imb_merged_model = merge_models(expert_models, regime)
            imb_merged_hooks = register_calibration_hooks(imb_merged_model, 'none', args.layer_filter)
            
            # Record merged activations
            imb_merged_activations = record_activations(imb_merged_model, imb_merged_hooks, joint_cal_loader, device)
            
            for method in ['none', 'taac', 'sp_taac', 'qspa', 'pra']:
                calibrate_model(imb_merged_model, imb_merged_hooks, experts_activations, imb_merged_activations, mode=method)
                accs = evaluate_model(imb_merged_model, task_heads, test_loaders, device)
                avg_acc = sum(accs.values()) / len(accs)
                print(f"Method: {method:10s} | MNIST: {accs['mnist']:.2f}% | F-MNIST: {accs['fmnist']:.2f}% | CIFAR-10: {accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")
                
            # Clean up hooks
            for _, handle in imb_merged_hooks:
                handle.remove()
                
        # Clean up all registered handles
        for task in expert_hooks:
            for _, handle in expert_hooks[task]:
                handle.remove()
        for _, handle in merged_hooks:
            handle.remove()

if __name__ == '__main__':
    main()
