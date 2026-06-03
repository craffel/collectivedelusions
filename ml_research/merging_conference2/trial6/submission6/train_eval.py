import os
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set complete determinism
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

set_seed(42)

# Helper Dataset to replicate grayscale to 3 channels
class ReplicateChannelsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x, y

def get_dataloaders(batch_size=64, num_train_samples=5000, num_cal_samples=128):
    # Transforms
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download full datasets
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)

    # Deterministic subset selection
    def get_subsets(full_dataset):
        indices = list(range(len(full_dataset)))
        random.seed(42)
        random.shuffle(indices)
        train_indices = indices[:num_train_samples]
        cal_indices = indices[num_train_samples:num_train_samples + num_cal_samples]
        return Subset(full_dataset, train_indices), Subset(full_dataset, cal_indices)

    mnist_train, mnist_cal = get_subsets(mnist_train_full)
    fmnist_train, fmnist_cal = get_subsets(fmnist_train_full)
    cifar_train, cifar_cal = get_subsets(cifar_train_full)

    # Data loaders
    loaders = {
        'mnist': {
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
            'cal': DataLoader(mnist_cal, batch_size=num_cal_samples, shuffle=False),
            'test': DataLoader(mnist_test, batch_size=256, shuffle=False)
        },
        'fmnist': {
            'train': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True),
            'cal': DataLoader(fmnist_cal, batch_size=num_cal_samples, shuffle=False),
            'test': DataLoader(fmnist_test, batch_size=256, shuffle=False)
        },
        'cifar': {
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True),
            'cal': DataLoader(cifar_cal, batch_size=num_cal_samples, shuffle=False),
            'test': DataLoader(cifar_test, batch_size=256, shuffle=False)
        }
    }
    return loaders

def create_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace classification head with Dropout + Linear
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
    return model

def train_expert(model, init_model, train_loader, device, epochs=5, lr=1e-4, weight_decay=1e-4, l2_sp_lambda=0.0):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Store initial weights for L2-SP
    if l2_sp_lambda > 0.0:
        init_params = {name: param.clone().to(device) for name, param in init_model.named_parameters() if param.requires_grad}

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Add L2-SP penalty if applicable
            if l2_sp_lambda > 0.0:
                l2_sp_penalty = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and 'weight' in name and name in init_params:
                        l2_sp_penalty += torch.sum((param - init_params[name]) ** 2)
                loss += l2_sp_lambda * l2_sp_penalty
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return model

def evaluate(model, loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

def get_conv_bn_pairs(model):
    """Finds all Conv2d and BatchNorm2d pairs in execution order."""
    pairs = []
    # Helper to traverse recursively and capture in execution order
    def traverse(module, name_prefix=""):
        # We can look at modules directly inside ResNet basic block or sequentially
        # In ResNet-18, the submodules are conv1, bn1, layer1, layer2, layer3, layer4.
        # Inside BasicBlock, we have conv1, bn1, conv2, bn2, and optional downsample.
        # Let's target specific submodules in order.
        pass

    # A simpler and guaranteed execution order for ResNet-18 specifically:
    submodules_ordered = []
    
    # Root conv1 and bn1
    submodules_ordered.append(('conv1', model.conv1, 'bn1', model.bn1))
    
    # Helper for basic blocks
    def process_block(block_name, block):
        submodules_ordered.append((f'{block_name}.conv1', block.conv1, f'{block_name}.bn1', block.bn1))
        if block.downsample is not None:
            # Downsample is a Sequential of Conv2d and BatchNorm2d
            submodules_ordered.append((f'{block_name}.downsample.0', block.downsample[0], f'{block_name}.downsample.1', block.downsample[1]))
        submodules_ordered.append((f'{block_name}.conv2', block.conv2, f'{block_name}.bn2', block.bn2))

    for idx, block in enumerate(model.layer1):
        process_block(f'layer1.{idx}', block)
    for idx, block in enumerate(model.layer2):
        process_block(f'layer2.{idx}', block)
    for idx, block in enumerate(model.layer3):
        process_block(f'layer3.{idx}', block)
    for idx, block in enumerate(model.layer4):
        process_block(f'layer4.{idx}', block)
        
    return submodules_ordered

def analyze_drift(experts, init_model):
    """Analyze L2 drift and cosine similarity of expert weight updates."""
    init_state = init_model.state_dict()
    drift_metrics = {}
    
    for task_name, expert in experts.items():
        expert_state = expert.state_dict()
        l2_dist_total = 0.0
        param_count = 0
        weight_norms = 0.0
        
        updates = []
        for name, param in expert.named_parameters():
            if param.requires_grad and 'weight' in name:
                init_p = init_state[name]
                update = (param - init_p).flatten()
                updates.append(update)
                l2_dist_total += torch.sum(update ** 2).item()
                param_count += update.numel()
                weight_norms += torch.sum(param ** 2).item()
                
        l2_dist = np.sqrt(l2_dist_total)
        rms_drift = np.sqrt(l2_dist_total / param_count) if param_count > 0 else 0
        drift_metrics[task_name] = {
            'l2_distance_from_init': float(l2_dist),
            'rms_drift_from_init': float(rms_drift),
            'weight_l2_norm': float(np.sqrt(weight_norms))
        }
        
    # Compute cross-task update cosine similarities
    tasks = list(experts.keys())
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            # Get updates
            u1_list = []
            u2_list = []
            for name in experts[t1].state_dict().keys():
                if 'weight' in name and any(k in name for k in ['conv', 'layer', 'downsample']):
                    init_p = init_state[name]
                    u1 = (experts[t1].state_dict()[name] - init_p).flatten()
                    u2 = (experts[t2].state_dict()[name] - init_p).flatten()
                    u1_list.append(u1)
                    u2_list.append(u2)
            u1_all = torch.cat(u1_list)
            u2_all = torch.cat(u2_list)
            
            cos_sim = torch.dot(u1_all, u2_all) / (torch.norm(u1_all) * torch.norm(u2_all) + 1e-8)
            drift_metrics[f'cos_sim_{t1}_{t2}'] = float(cos_sim.cpu().item())
            
    return drift_metrics

def merge_models_wa(experts, init_model):
    """Averages expert weights to create a merged model."""
    merged = create_model()
    merged_state = merged.state_dict()
    expert_states = {t: expert.state_dict() for t, expert in experts.items()}
    
    # We only average backbone weights. The fc layer remains expert-specific during evaluation.
    for name in merged_state.keys():
        if 'fc' in name:
            # Do not average fc, we can just initialize it
            continue
        # Average weights across tasks if they are floating point or complex
        if torch.is_floating_point(merged_state[name]) or torch.is_complex(merged_state[name]):
            merged_state[name] = torch.stack([expert_states[t][name] for t in experts.keys()]).mean(dim=0)
        else:
            # For non-floating point tensors/buffers (like num_batches_tracked), copy from the first expert
            first_task = list(experts.keys())[0]
            merged_state[name] = expert_states[first_task][name].clone()
        
    merged.load_state_dict(merged_state)
    return merged

# Forward hooks to capture activations
class ActivationHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = input[0].detach()

    def remove(self):
        self.hook.remove()

class OutputActivationHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output.detach()

    def remove(self):
        self.hook.remove()

def calibrate_model(merged, experts, cal_loaders, device, method='none', r=4, reg=0.5):
    """
    Applies sequential calibration to the merged model.
    method can be: 'none', 'sp-taac', 'hybrid' (SP-TAAC + SLR-WBC)
    """
    if method == 'none':
        return merged

    set_seed(42)
    # Create copies for activation computation
    merged_cal = copy.deepcopy(merged).to(device).eval()
    experts_cal = {t: copy.deepcopy(expert).to(device).eval() for t, expert in experts.items()}
    
    # Get Conv-BN pairs in execution order
    merged_pairs = get_conv_bn_pairs(merged_cal)
    expert_pairs = {t: get_conv_bn_pairs(experts_cal[t]) for t in experts.keys()}
    
    num_layers = len(merged_pairs)
    print(f"Calibrating {num_layers} Conv-BN layers using method: {method}")

    # Build D_cal batches
    # Load calibration sets
    cal_batches = {}
    for t in experts.keys():
        for x, _ in cal_loaders[t]['cal']:
            cal_batches[t] = x.to(device)
            break # only 1 batch of size N=128
            
    # Sequential calibration
    for l_idx in range(num_layers):
        conv_name, conv_m, bn_name, bn_m = merged_pairs[l_idx]
        print(f"Layer {l_idx+1}/{num_layers} | {conv_name} & {bn_name}")
        
        # Decide layer type
        is_deep = any(k in conv_name for k in ['layer3', 'layer4'])
        is_deep_calibration = is_deep and method == 'hybrid'
        
        if is_deep_calibration:
            # SLR-WBC activation collection
            hook_X = ActivationHook(conv_m)
            hooks_H = {t: OutputActivationHook(expert_pairs[t][l_idx][3]) for t in experts.keys()}
            
            X_tasks = {}
            H_target_tasks = {}
            
            with torch.no_grad():
                for t in experts.keys():
                    merged_cal.fc = experts_cal[t].fc
                    _ = merged_cal(cal_batches[t])
                    X_tasks[t] = hook_X.activation.clone()
                    
                    _ = experts_cal[t](cal_batches[t])
                    H_target_tasks[t] = hooks_H[t].activation.clone()
                    
            hook_X.remove()
            for t in experts.keys():
                hooks_H[t].remove()
        else:
            # SP-TAAC activation collection: compare outputs of the Conv layers (before-BN)
            hook_V_merged = OutputActivationHook(conv_m)
            hooks_V_expert = {t: OutputActivationHook(expert_pairs[t][l_idx][1]) for t in experts.keys()}
            
            V_merged_tasks = []
            V_expert_tasks = []
            
            with torch.no_grad():
                for t in experts.keys():
                    merged_cal.fc = experts_cal[t].fc
                    _ = merged_cal(cal_batches[t])
                    V_merged_tasks.append(hook_V_merged.activation.clone())
                    
                    _ = experts_cal[t](cal_batches[t])
                    V_expert_tasks.append(hooks_V_expert[t].activation.clone())
                    
            hook_V_merged.remove()
            for t in experts.keys():
                hooks_V_expert[t].remove()
            
        # Perform calibration
        if not is_deep_calibration:
            # SP-TAAC: scale BN layers based on pre-BN Conv outputs
            V_all = torch.cat(V_merged_tasks, dim=0) # (K*N, C, H, W)
            V_expert_all = torch.cat(V_expert_tasks, dim=0) # (K*N, C, H, W)
            
            # Compute channel-wise std deviation
            C = V_all.shape[1]
            V_flat = V_all.permute(0, 2, 3, 1).reshape(-1, C)
            V_expert_flat = V_expert_all.permute(0, 2, 3, 1).reshape(-1, C)
            
            sigma_merged = V_flat.std(dim=0)
            sigma_target = V_expert_flat.std(dim=0)
            
            gamma = sigma_target / (sigma_merged + 1e-5)
            # Clamp for numerical stability
            gamma = torch.clamp(gamma, min=0.1, max=10.0)
            
            # Update BN scale and shift
            bn_m.weight.data.copy_(bn_m.weight.data * gamma)
            bn_m.bias.data.copy_(bn_m.bias.data * gamma)
            
            print(f"  [SP-TAAC] Applied scaling. Mean gamma: {gamma.mean().item():.4f}")
            
        else:
            # SLR-WBC: Apply weight correction and BN alignment
            C_out, C_in, Kh, Kw = conv_m.weight.shape
            
            # Invert expert BN operations to get target Conv outputs V_target
            V_target_tasks = []
            for t in experts.keys():
                H_t = H_target_tasks[t] # (N, C_out, H, W)
                exp_bn = expert_pairs[t][l_idx][3]
                
                # Invert: V = (H - b)/w * sqrt(var + eps) + mu
                w = exp_bn.weight.view(1, C_out, 1, 1)
                b = exp_bn.bias.view(1, C_out, 1, 1)
                mu = exp_bn.running_mean.view(1, C_out, 1, 1)
                var = exp_bn.running_var.view(1, C_out, 1, 1)
                eps = exp_bn.eps
                
                V_t = (H_t - b) / (w + 1e-5) * torch.sqrt(var + eps) + mu
                V_target_tasks.append(V_t)
                
            # Flatten activations for least-squares
            X_unfolded_tasks = []
            for t in experts.keys():
                X_t = X_tasks[t] # (N, C_in, H_in, W_in)
                # Unfold
                X_unf = F.unfold(X_t, kernel_size=(Kh, Kw), dilation=conv_m.dilation, padding=conv_m.padding, stride=conv_m.stride) # (N, d_in, M_spatial)
                X_unfolded_tasks.append(X_unf)
                
            # Concatenate unfolded activations and target outputs across all tasks
            X_unfolded = torch.cat(X_unfolded_tasks, dim=0) # (K*N, d_in, M_spatial)
            d_in = X_unfolded.shape[1]
            # Reshape to (d_in, K*N*M_spatial)
            X_matrix = X_unfolded.transpose(0, 1).reshape(d_in, -1)
            
            # Flatten V_target
            V_target_all = torch.cat(V_target_tasks, dim=0) # (K*N, C_out, H_out, W_out)
            # Reshape to (C_out, K*N*H_out*W_out)
            V_target = V_target_all.transpose(0, 1).reshape(C_out, -1)
            
            M = X_matrix.shape[1]
            lmbda = reg * M
            
            # Flatten current weight
            W_curr = conv_m.weight.view(C_out, -1)
            # Error matrix
            E = V_target - torch.matmul(W_curr, X_matrix)
            
            # Solve ridge regression
            cov = torch.matmul(X_matrix, X_matrix.T)
            cov.add_(torch.eye(d_in, device=device) * lmbda)
            inv_cov = torch.linalg.inv(cov)
            
            # Full rank update
            dW_star = torch.matmul(torch.matmul(E, X_matrix.T), inv_cov)
            
            # Truncated SVD
            U, S, Vh = torch.linalg.svd(dW_star, full_matrices=False)
            dW_r = torch.matmul(U[:, :r], torch.matmul(torch.diag(S[:r]), Vh[:r, :]))
            
            # Update Conv weight
            conv_m.weight.data.copy_(conv_m.weight.data + dW_r.view(conv_m.weight.shape))
            
            # Compute new activations V_new
            W_new = conv_m.weight.view(C_out, -1)
            V_new = torch.matmul(W_new, X_matrix) # (C_out, K*N*H_out*W_out)
            
            # Update BN running statistics
            mu_run = V_new.mean(dim=1)
            var_run = V_new.var(dim=1, unbiased=False)
            bn_m.running_mean.copy_(mu_run)
            bn_m.running_var.copy_(var_run)
            
            # Normalize activations
            V_norm = (V_new - mu_run.unsqueeze(1)) / torch.sqrt(var_run.unsqueeze(1) + bn_m.eps)
            
            # Concatenate expert BN outputs H_target
            H_target_all = torch.cat([H_target_tasks[t] for t in experts.keys()], dim=0) # (K*N, C_out, H_out, W_out)
            H_target = H_target_all.transpose(0, 1).reshape(C_out, -1)
            
            # Solve least-squares BN affine parameters
            w_c = (V_norm * H_target).mean(dim=1)
            b_c = H_target.mean(dim=1)
            
            # Clamp scale
            w_c = torch.clamp(w_c, min=0.1, max=10.0)
            
            bn_m.weight.data.copy_(w_c)
            bn_m.bias.data.copy_(b_c)
            
            print(f"  [SLR-WBC] SLR weight update rank-{r} applied. BN stats updated. Mean scaling w_c: {w_c.mean().item():.4f}")
            
    # Copy calibrated state back to the original merged model
    merged.load_state_dict(merged_cal.state_dict())
    return merged

def run_experiment_suite(device):
    loaders = get_dataloaders()
    
    # We will run 4 training scenarios
    scenarios = {
        'A_low_reg': {'weight_decay': 0.0, 'l2_sp_lambda': 0.0},
        'B_std_reg': {'weight_decay': 1e-4, 'l2_sp_lambda': 0.0},
        'C_high_reg': {'weight_decay': 1e-2, 'l2_sp_lambda': 0.0},
        'D_l2sp_reg': {'weight_decay': 0.0, 'l2_sp_lambda': 1e-3}
    }
    
    results = {}
    
    # For each scenario, train/evaluate/analyze
    for sc_name, config in scenarios.items():
        print(f"\n==================== Running Scenario: {sc_name} ====================")
        set_seed(42)
        
        # Progenitor / initial weights
        init_model = create_model().to(device)
        
        # Train experts for each task
        experts = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            print(f"\n--- Training Expert for {task} ---")
            set_seed(42)
            model = create_model()
            model = train_expert(
                model=model,
                init_model=init_model,
                train_loader=loaders[task]['train'],
                device=device,
                epochs=5,
                lr=1e-4,
                weight_decay=config['weight_decay'],
                l2_sp_lambda=config['l2_sp_lambda']
            )
            
            # Evaluate expert accuracy on its own test set
            acc = evaluate(model, loaders[task]['test'], device)
            print(f"Expert {task} Test Acc: {acc:.2f}%")
            experts[task] = model
            
        # 1. Analyze expert parameter drift
        drift_metrics = analyze_drift(experts, init_model)
        print("Drift Metrics:")
        print(json.dumps(drift_metrics, indent=2))
        
        # 2. Merge models using Weight Averaging (WA)
        merged_base = merge_models_wa(experts, init_model)
        
        # 3. Evaluate different post-merge calibration methods
        calibration_methods = ['none', 'sp-taac', 'hybrid']
        scenario_results = {'drift': drift_metrics}
        
        for method in calibration_methods:
            print(f"\n--- Applying Calibration: {method} ---")
            # Create a copy of merged_base to calibrate in-place
            merged_cal = copy.deepcopy(merged_base)
            merged_cal = calibrate_model(
                merged=merged_cal,
                experts=experts,
                cal_loaders=loaders,
                device=device,
                method=method,
                r=4,
                reg=0.5
            )
            
            # Evaluate merged model on all three tasks
            # Remember to attach the correct task-specific head (fc) during evaluation
            accs = {}
            for task in ['mnist', 'fmnist', 'cifar']:
                merged_cal.fc = experts[task].fc
                acc = evaluate(merged_cal, loaders[task]['test'], device)
                accs[task] = acc
                print(f"Merged model ({method}) on {task} Test Acc: {acc:.2f}%")
            
            accs['average'] = sum(accs.values()) / len(accs)
            scenario_results[method] = accs
            print(f"Average Merged Acc ({method}): {accs['average']:.2f}%")
            
        results[sc_name] = scenario_results
        
    # Save the results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nAll experiments completed and results saved to experiment_results.json!")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device:", device)
    run_experiment_suite(device)
