import os
import copy
import json
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset, DataLoader

class BlendedBatchNorm2d(nn.Module):
    def __init__(self, original_bn, alpha=1.0):
        super().__init__()
        self.original_bn = original_bn
        self.alpha = alpha
        
    def forward(self, x):
        if self.alpha > 0:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            merged_mean = self.original_bn.running_mean
            merged_var = self.original_bn.running_var
            
            active_mean = (1 - self.alpha) * merged_mean + self.alpha * batch_mean
            active_var = (1 - self.alpha) * merged_var + self.alpha * batch_var
        else:
            active_mean = self.original_bn.running_mean
            active_var = self.original_bn.running_var
            
        w = self.original_bn.weight.view(1, -1, 1, 1)
        b = self.original_bn.bias.view(1, -1, 1, 1)
        eps = self.original_bn.eps
        
        x_norm = (x - active_mean.view(1, -1, 1, 1)) / torch.sqrt(active_var.view(1, -1, 1, 1) + eps)
        return w * x_norm + b

def get_subset(dataset, num_samples=5000, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    return Subset(dataset, indices)

def train_expert(model, dataloader, epochs, lr, weight_decay, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    return model

def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return correct / total

def merge_models(progenitor, experts, method='WA', lam=0.5, p_ties=0.2, p_dare=0.9):
    device = next(experts[0].parameters()).device
    progenitor = progenitor.to(device)
    merged = copy.deepcopy(progenitor)
    merged_state_dict = merged.state_dict()
    expert_state_dicts = [e.state_dict() for e in experts]
    
    def trim_tensor(t, p):
        flat = t.view(-1)
        k = int(flat.numel() * p)
        if k == 0:
            return t
        threshold = torch.topk(flat.abs(), k).values[-1]
        mask = flat.abs() >= threshold
        return (flat * mask).view_as(t)

    def elect_sign(updates):
        stacked = torch.stack(updates)
        signs = torch.sign(stacked)
        sum_signs = torch.sum(signs, dim=0)
        consensus_sign = torch.sign(sum_signs)
        return consensus_sign

    def disjoint_merge(updates, consensus_sign):
        stacked = torch.stack(updates)
        signs = torch.sign(stacked)
        mask = (signs == consensus_sign.unsqueeze(0)) & (consensus_sign.unsqueeze(0) != 0)
        sum_agree = torch.sum(stacked * mask, dim=0)
        count_agree = torch.sum(mask, dim=0)
        merged_update = torch.where(count_agree > 0, sum_agree / count_agree, torch.zeros_like(sum_agree))
        return merged_update

    def dare_tensor(t, p):
        if p <= 0.0:
            return t
        if p >= 1.0:
            return torch.zeros_like(t)
        mask = (torch.rand_like(t) > p).float()
        return (t * mask) / (1.0 - p)
    
    for key in merged_state_dict.keys():
        if 'fc' in key:
            continue
        
        if merged_state_dict[key].dtype in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            if 'running_mean' in key or 'running_var' in key:
                stacked = torch.stack([sd[key] for sd in expert_state_dicts])
                merged_state_dict[key] = torch.mean(stacked, dim=0)
            else:
                if method == 'WA':
                    stacked = torch.stack([sd[key] for sd in expert_state_dicts])
                    merged_state_dict[key] = torch.mean(stacked, dim=0)
                elif method == 'TA':
                    prog_val = progenitor.state_dict()[key]
                    updates = [sd[key] - prog_val for sd in expert_state_dicts]
                    merged_state_dict[key] = prog_val + lam * sum(updates)
                elif method == 'TIES':
                    prog_val = progenitor.state_dict()[key]
                    raw_updates = [sd[key] - prog_val for sd in expert_state_dicts]
                    trimmed_updates = [trim_tensor(up, p_ties) for up in raw_updates]
                    consensus_sign = elect_sign(trimmed_updates)
                    merged_update = disjoint_merge(trimmed_updates, consensus_sign)
                    merged_state_dict[key] = prog_val + lam * merged_update
                elif method == 'DARE':
                    prog_val = progenitor.state_dict()[key]
                    raw_updates = [sd[key] - prog_val for sd in expert_state_dicts]
                    dare_updates = [dare_tensor(up, p_dare) for up in raw_updates]
                    merged_update = torch.mean(torch.stack(dare_updates), dim=0)
                    merged_state_dict[key] = prog_val + lam * merged_update
                    
    merged.load_state_dict(merged_state_dict)
    return merged

def get_bn_activations_std(model, dataloader, device):
    model.to(device)
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_modules.append((name, module))
            
    activations = {name: [] for name, _ in conv_modules}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name].append(output.detach().cpu())
        return hook
        
    hooks = []
    for name, module in conv_modules:
        hooks.append(module.register_forward_hook(hook_fn(name)))
        
    model.eval()
    count = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _ = model(x)
            count += x.size(0)
            if count >= 128:
                break
                
    for h in hooks:
        h.remove()
        
    stds = {}
    for name in activations:
        act = torch.cat(activations[name], dim=0)[:128]
        stds[name] = torch.sqrt(act.var(dim=(0, 2, 3), unbiased=False) + 1e-5)
    return stds

def apply_repair(merged_model, experts, calibration_loaders, device):
    conv_to_bn = {}
    prev_conv_name = None
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prev_conv_name = name
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if prev_conv_name is not None:
                conv_to_bn[prev_conv_name] = module
                
    expert_stds = []
    for i, exp in enumerate(experts):
        stds = get_bn_activations_std(exp, calibration_loaders[i], device)
        expert_stds.append(stds)
        
    joint_images = []
    for loader in calibration_loaders:
        count = 0
        for x, _ in loader:
            joint_images.append(x)
            count += x.size(0)
            if count >= 43:
                break
    joint_tensor = torch.cat(joint_images, dim=0)
    joint_dataset = torch.utils.data.TensorDataset(joint_tensor, torch.zeros(joint_tensor.size(0), dtype=torch.long))
    joint_loader = DataLoader(joint_dataset, batch_size=64, shuffle=False)
    
    merged_stds = get_bn_activations_std(merged_model, joint_loader, device)
    
    for conv_name, bn_module in conv_to_bn.items():
        target_std = torch.mean(torch.stack([estds[conv_name] for estds in expert_stds]), dim=0)
        merged_std = merged_stds[conv_name]
        
        gamma = target_std / (merged_std + 1e-5)
        gamma = torch.clamp(gamma, min=0.1, max=10.0).to(bn_module.weight.device)
        
        bn_module.weight.data.copy_(bn_module.weight.data * gamma)
        bn_module.bias.data.copy_(bn_module.bias.data * gamma)

def apply_fnbc(merged_model, experts):
    conv_to_bn = {}
    prev_conv_name = None
    for name, module in merged_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prev_conv_name = name
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if prev_conv_name is not None:
                conv_to_bn[prev_conv_name] = name
                
    merged_modules = dict(merged_model.named_modules())
    expert_modules_lists = [dict(e.named_modules()) for e in experts]
    
    for conv_name, bn_name in conv_to_bn.items():
        merged_conv = merged_modules[conv_name]
        expert_convs = [em[conv_name] for em in expert_modules_lists]
        
        w_merged = merged_conv.weight.data
        w_experts = [ec.weight.data for ec in expert_convs]
        
        norm_merged = torch.sum(w_merged ** 2)
        norm_experts = torch.mean(torch.stack([torch.sum(we.to(w_merged.device) ** 2) for we in w_experts]))
        
        R_l = norm_merged / (norm_experts + 1e-8)
        R_l = torch.clamp(R_l, min=0.1, max=10.0)
        
        bn_module = merged_modules[bn_name]
        bn_module.running_var.copy_(R_l.to(bn_module.running_var.device) * bn_module.running_var)

def apply_spttbc(model, alpha=1.0):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]
            setattr(parent, attr_name, BlendedBatchNorm2d(module, alpha=alpha))

def run_shared_bn_calibration(model, calibration_loaders, num_batches=20, device='cpu'):
    model.to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        for loader in calibration_loaders:
            count = 0
            for x, _ in loader:
                x = x.to(device)
                _ = model(x)
                count += 1
                if count >= num_batches:
                    break
    model.eval()

def run_task_specific_bn_calibration(model, task_loader, num_batches=20, device='cpu'):
    model.to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            
    with torch.no_grad():
        count = 0
        for x, _ in task_loader:
            x = x.to(device)
            _ = model(x)
            count += 1
            if count >= num_batches:
                break
    model.eval()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    transform_mnist = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    transform_fmnist = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
    test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)

    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    
    train_mnist_sub = get_subset(train_mnist)
    train_fmnist_sub = get_subset(train_fmnist)
    train_cifar_sub = get_subset(train_cifar)
    
    loader_mnist = DataLoader(train_mnist_sub, batch_size=128, shuffle=True)
    loader_fmnist = DataLoader(train_fmnist_sub, batch_size=128, shuffle=True)
    loader_cifar = DataLoader(train_cifar_sub, batch_size=128, shuffle=True)
    
    test_loader_mnist = DataLoader(test_mnist, batch_size=256, shuffle=False)
    test_loader_fmnist = DataLoader(test_fmnist, batch_size=256, shuffle=False)
    test_loader_cifar = DataLoader(test_cifar, batch_size=256, shuffle=False)
    
    print("Preparing models...")
    progenitor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    expert_mnist = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert_mnist.fc = nn.Linear(512, 10)
    
    expert_fmnist = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert_fmnist.fc = nn.Linear(512, 10)
    
    expert_cifar = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert_cifar.fc = nn.Linear(512, 10)
    
    if os.path.exists("expert_mnist_severe.pth") and os.path.exists("expert_fmnist_severe.pth") and os.path.exists("expert_cifar_severe.pth"):
        print("Loading pre-trained severe-drift experts from disk...")
        expert_mnist.load_state_dict(torch.load("expert_mnist_severe.pth", map_location=device))
        expert_fmnist.load_state_dict(torch.load("expert_fmnist_severe.pth", map_location=device))
        expert_cifar.load_state_dict(torch.load("expert_cifar_severe.pth", map_location=device))
    else:
        print("\n--- Training MNIST Expert (Severe Drift) ---")
        train_expert(expert_mnist, loader_mnist, epochs=10, lr=1e-3, weight_decay=0.0, device=device)
        torch.save(expert_mnist.state_dict(), "expert_mnist_severe.pth")
        
        print("\n--- Training Fashion-MNIST Expert (Severe Drift) ---")
        train_expert(expert_fmnist, loader_fmnist, epochs=10, lr=1e-3, weight_decay=0.0, device=device)
        torch.save(expert_fmnist.state_dict(), "expert_fmnist_severe.pth")
        
        print("\n--- Training CIFAR-10 Expert (Severe Drift) ---")
        train_expert(expert_cifar, loader_cifar, epochs=10, lr=1e-3, weight_decay=0.0, device=device)
        torch.save(expert_cifar.state_dict(), "expert_cifar_severe.pth")
    
    experts = [expert_mnist, expert_fmnist, expert_cifar]
    dataloaders = [test_loader_mnist, test_loader_fmnist, test_loader_cifar]
    calib_loaders = [
        DataLoader(train_mnist_sub, batch_size=64, shuffle=True),
        DataLoader(train_fmnist_sub, batch_size=64, shuffle=True),
        DataLoader(train_cifar_sub, batch_size=64, shuffle=True)
    ]
    
    acc_mnist = evaluate(expert_mnist, test_loader_mnist, device)
    acc_fmnist = evaluate(expert_fmnist, test_loader_fmnist, device)
    acc_cifar = evaluate(expert_cifar, test_loader_cifar, device)
    print(f"\nExpert Test Accuracies:\nMNIST: {acc_mnist:.4%}\nFashion-MNIST: {acc_fmnist:.4%}\nCIFAR-10: {acc_cifar:.4%}")
    
    results = {
        "experts": {
            "MNIST": acc_mnist,
            "Fashion-MNIST": acc_fmnist,
            "CIFAR-10": acc_cifar,
            "Average": (acc_mnist + acc_fmnist + acc_cifar) / 3
        },
        "merging": []
    }
    
    lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]
    merge_configs = [('WA', None)] + [('TA', l) for l in lambdas]
    
    for method, lam in merge_configs:
        config_name = f"{method} (lambda={lam})" if lam is not None else "WA"
        print(f"\n==================== Evaluating {config_name} ====================")
        
        # A. No Calib
        m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
        m_model.fc = nn.Linear(512, 10)
        accs_no_calib = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_no_calib.append(acc)
        avg_no_calib = sum(accs_no_calib) / 3
        print(f"No Calibration - MNIST: {accs_no_calib[0]:.4%} | FMNIST: {accs_no_calib[1]:.4%} | CIFAR-10: {accs_no_calib[2]:.4%} | Avg: {avg_no_calib:.4%}")
        
        # B. REPAIR
        m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
        apply_repair(m_model, experts, calib_loaders, device)
        accs_repair = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_repair.append(acc)
        avg_repair = sum(accs_repair) / 3
        print(f"REPAIR         - MNIST: {accs_repair[0]:.4%} | FMNIST: {accs_repair[1]:.4%} | CIFAR-10: {accs_repair[2]:.4%} | Avg: {avg_repair:.4%}")
        
        # C. SP-TTBC
        m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
        apply_spttbc(m_model, alpha=1.0)
        accs_spttbc = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_spttbc.append(acc)
        avg_spttbc = sum(accs_spttbc) / 3
        print(f"SP-TTBC (1.0)  - MNIST: {accs_spttbc[0]:.4%} | FMNIST: {accs_spttbc[1]:.4%} | CIFAR-10: {accs_spttbc[2]:.4%} | Avg: {avg_spttbc:.4%}")
        
        # D. FNBC
        m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
        apply_fnbc(m_model, experts)
        accs_fnbc = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_fnbc.append(acc)
        avg_fnbc = sum(accs_fnbc) / 3
        print(f"FNBC (Ours)    - MNIST: {accs_fnbc[0]:.4%} | FMNIST: {accs_fnbc[1]:.4%} | CIFAR-10: {accs_fnbc[2]:.4%} | Avg: {avg_fnbc:.4%}")
        
        # E. Shared BN Calib
        m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
        m_model.fc = nn.Linear(512, 10)
        run_shared_bn_calibration(m_model, calib_loaders, num_batches=20, device=device)
        accs_shared_bc = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_shared_bc.append(acc)
        avg_shared_bc = sum(accs_shared_bc) / 3
        print(f"Shared BN Calib- MNIST: {accs_shared_bc[0]:.4%} | FMNIST: {accs_shared_bc[1]:.4%} | CIFAR-10: {accs_shared_bc[2]:.4%} | Avg: {avg_shared_bc:.4%}")

        # F. Task-Specific BN Calib (Ours)
        accs_ts_bc = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            # Refresh merge model for each task to calibrate specifically
            m_model = merge_models(progenitor, experts, method=method, lam=lam if lam is not None else 0.5)
            m_model.fc = nn.Linear(512, 10)
            run_task_specific_bn_calibration(m_model, calib_loaders[i], num_batches=20, device=device)
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_ts_bc.append(acc)
        avg_ts_bc = sum(accs_ts_bc) / 3
        print(f"TS-BN Calib    - MNIST: {accs_ts_bc[0]:.4%} | FMNIST: {accs_ts_bc[1]:.4%} | CIFAR-10: {accs_ts_bc[2]:.4%} | Avg: {avg_ts_bc:.4%}")

        results["merging"].append({
            "method": method,
            "lambda": lam,
            "No_Calib": {"MNIST": accs_no_calib[0], "Fashion-MNIST": accs_no_calib[1], "CIFAR-10": accs_no_calib[2], "Average": avg_no_calib},
            "REPAIR": {"MNIST": accs_repair[0], "Fashion-MNIST": accs_repair[1], "CIFAR-10": accs_repair[2], "Average": avg_repair},
            "SP-TTBC": {"MNIST": accs_spttbc[0], "Fashion-MNIST": accs_spttbc[1], "CIFAR-10": accs_spttbc[2], "Average": avg_spttbc},
            "FNBC": {"MNIST": accs_fnbc[0], "Fashion-MNIST": accs_fnbc[1], "CIFAR-10": accs_fnbc[2], "Average": avg_fnbc},
            "Shared_BC": {"MNIST": accs_shared_bc[0], "Fashion-MNIST": accs_shared_bc[1], "CIFAR-10": accs_shared_bc[2], "Average": avg_shared_bc},
            "TS_BC": {"MNIST": accs_ts_bc[0], "Fashion-MNIST": accs_ts_bc[1], "CIFAR-10": accs_ts_bc[2], "Average": avg_ts_bc}
        })
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults successfully saved to results.json!")

if __name__ == '__main__':
    main()
